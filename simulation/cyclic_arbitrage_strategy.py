from simulation.trading_simulator import Order, StateManager, Strategy, Trade
from simulation.sim_types import Token, TokenPair
from smart_order_router.graph import Graph
from smart_order_router.sor import single_sor_no_fees, single_sor_with_fees

from collections import defaultdict
from glob import glob
import logging
from scipy.optimize import minimize_scalar
from typing import Any, Dict, List, Set, Tuple


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

class CyclicArbitrageStrategy(Strategy):

    def __init__(self, config):
        super().__init__()
        self.token_trading_universe = config['venue']['stellaswap']['token_trading_universe']

        # Never put in less than 1 cent or more than $10,000 in a single swap
        # Gas fees are generally 1 cent, so it makes no sense to go below that
        # This seems quite high (and it is), but in theory with limit prices, our loss
        # is capped at gas fees of a failed transaction
        strategy_config = config['strategy']['cyclic_arbitrage']
        self.amount_in_lower_bound_usd = strategy_config['amount_in_lower_bound_usd']
        self.amount_in_upper_bound_usd = strategy_config['amount_in_upper_bound_usd']
        self.acceptable_slippage_rate = strategy_config['acceptable_slippage_rate']

        # We need to expect to make at least 3 cents to justify making the swap
        # Because we just net +tokens of what we put in, we do not incur any added cost
        # if the price of the token goes down
        self.min_expected_profit_usd = strategy_config['min_expected_profit_usd']

        self.pending_order_ids: Set[int] = set()

    def on_state_update(self, state_manager: StateManager) -> List[Order]:
        if len(self.pending_order_ids) > 0:
            logging.info('Pending an order fill, so we do not generate more orders')
            return []
        potential_orders, order_id_to_token_pair_path = self._generate_profitable_orders(state_manager)
        portfolio_orders = self._generate_order_portfolio(potential_orders, order_id_to_token_pair_path)
        self.pending_order_ids = self.pending_order_ids.union(set([order.order_id for order in portfolio_orders]))
        return portfolio_orders

    def on_event(self, state_manager: StateManager, event_data: Dict[str, Any]):
        # Later we can perhaps define events (like a 3rd party swap txn) and call this callback with that as an argument
        raise NotImplementedError

    def on_filled_orders(self, state_manager: StateManager, new_trades: List[Trade]):
        filled_order_ids = set([trade.order_id for trade in new_trades])
        self.pending_order_ids -= filled_order_ids
        # We fill all outstanding orders in sim, so we should have no pending ones anymore
        assert(len(self.pending_order_ids) == 0)

        failed_trades = [t for t in new_trades if not t.is_success]
        logging.warning(f'Order(s) was/were rejected, trade(s) = {failed_trades}')

    def _generate_profitable_orders(self, state_manager: StateManager) -> Tuple[List[Order], Dict[str, List[str]]]:
        '''
        Returns list of profitable orders, order ID -> token pair path
        Note that each order is individually profitable, but there is NO guarantee that any two orders together
        are still profitable! In fact a group of orders is likely NOT profitable
        '''
        potential_orders: List[Order] = []
        order_id_to_token_pair_path: Dict[str, List[str]] = {}

        for token_address in self.token_trading_universe:
            token = state_manager.address_to_token[token_address]

            best_amount_in, best_amount_out, best_token_pair_path = self._scipy_find_optimal_amount_in(state_manager.token_graph, token)
            expected_pnl_usd = (best_amount_out - best_amount_in) * token.get_usd_value()

            if expected_pnl_usd > self.min_expected_profit_usd:
                amount_out_min = (best_amount_out - best_amount_in) * (1 - self.acceptable_slippage_rate) + best_amount_in
                new_order = Order(
                    order_id=state_manager.generate_order_id(),
                    amount_in=best_amount_in,
                    amount_out_min=amount_out_min,
                    path=self._get_token_path(best_token_pair_path, token.token_address),
                    block_num=state_manager.cur_block_num,
                    last_txn_index=state_manager.last_txn_index,
                    metadata={
                        'input_usd_notional': best_amount_in * token.get_usd_value(),
                        'expected_out_amount': (best_amount_out - best_amount_in),
                        'expected_pnl': expected_pnl_usd,
                    },
                )
                potential_orders.append(new_order)
                order_id_to_token_pair_path[new_order.order_id] = best_token_pair_path
        return potential_orders, order_id_to_token_pair_path

    @staticmethod
    def _generate_order_portfolio(potential_orders: List[Order], order_id_to_token_pair_path: Dict[str, List[str]]) -> List[Order]:
        '''
        The 'order portfolio' is the set of orders that maximizes profit from the set of potential_orders
        Greedily picks orders from potential_orders and selects those that have no token pair overlaps
        This certainly isn't optimal but frankly, the optimal 'portfolio' is likely just the single most profitable order
        In any case, TODO: come up with a more optimal portfolio selection
        '''
        portfolio_orders = []
        sorted_orders = sorted(potential_orders, key=lambda o: o.metadata['expected_pnl'], reverse=True)
        
        portfolio_token_pairs_crossed: Set[str] = set()
        for o in sorted_orders:
            token_pairs = set(order_id_to_token_pair_path[o.order_id])
            if len(portfolio_token_pairs_crossed.intersection(token_pairs)) == 0:
                portfolio_token_pairs_crossed = portfolio_token_pairs_crossed.union(token_pairs)
                portfolio_orders.append(o)
                
        return portfolio_orders

    def _naive_find_optimal_amount_in(self, graph: Graph, token: Token):
        amount_in = self.amount_in_lower_bound_usd / token.get_usd_values()
        best_amount_in = amount_in
        best_amount_out = amount_in
        best_token_pair_path = []
        
        while True:
            amount_out, token_pair_path = single_sor_with_fees(graph, token.token_address, token.token_address, amount_in=amount_in)
            if amount_out - amount_in > best_amount_out - best_amount_in:
                best_amount_in, best_amount_out, best_token_pair_path = amount_in, amount_out, token_pair_path
            else:
                break
            amount_in = best_amount_in * 2

        return best_amount_in, best_amount_out, best_token_pair_path
    
    def _scipy_find_optimal_amount_in(self, graph: Graph, token: Token):
        def objective_func(amount_in):
            amount_out, path = single_sor_with_fees(graph, token.token_address, token.token_address, amount_in=amount_in)
            # If path == [] and amount_out == amount_in, our amount_in is too high. Penalize this as amount_in increases
            # because scipy cannot optimize if a bunch of values are zero. This is definitely a hack because the optimizer is
            # poorly suited (honestly could likely write a parabola solver that's better)
            return amount_in - amount_out if len(path) > 0 else amount_in
        
        if single_sor_with_fees(graph, token.token_address, token.token_address, amount_in=1)[0] - 1 <= 0:
            return 0, 0, []
        
        lower_bound_tokens = self.amount_in_lower_bound_usd / token.get_usd_value()
        upper_bound_tokens = self.amount_in_upper_bound_usd / token.get_usd_value()
        
        # Find the optimum within 0.1 cents 
        tol_tokens = 0.001 / token.get_usd_value()
        res = minimize_scalar(objective_func, bounds=(lower_bound_tokens, upper_bound_tokens), method='bounded', options={'xatol': tol_tokens, 'disp': 0})
        best_amount_in = res.x
        best_amount_out, best_token_pair_path = single_sor_with_fees(graph, token.token_address, token.token_address, amount_in=best_amount_in)
        return best_amount_in, best_amount_out, best_token_pair_path

    @staticmethod
    def _get_token_path(token_pair_path: List[TokenPair], start_token: str) -> List[str]:
        token_path = [start_token]
        for pair in token_pair_path:
            assert(token_path[-1] in [pair.token0_address, pair.token1_address])
            if pair.token0_address != token_path[-1]:
                token_path.append(pair.token0_address)
            else:
                token_path.append(pair.token1_address)
        return token_path


if __name__ == '__main__':
    from common.helpers import load_config
    from simulation.trading_simulator import StateManager, FillEngine, PnlCalculator, SimDriver

    config = load_config(filename='sim_cfg.yaml')

    token_holdings = defaultdict(int)
    for token_address, holdings in config['venue']['stellaswap']['holdings'].items():
        token_holdings[token_address] = holdings

    state_manager = StateManager(config, token_holdings)
    fill_engine = FillEngine(should_update_state_reserves_after_trades=True)
    pnl_calculator = PnlCalculator()
    cycle_strategy = CyclicArbitrageStrategy(config)
    sim_driver = SimDriver(
        state_manager=state_manager,
        fill_engine=fill_engine,
        pnl_calculator=pnl_calculator,
        strategy=cycle_strategy,
        should_trigger_strategy_on_end_of_block=True,
        should_trigger_strategy_on_txn=False,
    )

    files = [
        'data/stellaswap_txn_history/all/stellaswap_data_1740000_1749999.feather',
        # 'data/stellaswap_txn_history/all/stellaswap_data_1750000_1759999.feather',
        # 'data/stellaswap_txn_history/all/stellaswap_data_1760000_1769999.feather',
    ]
    sim_driver.process_files(files)
