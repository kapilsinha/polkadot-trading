from live_trader.order import Order
from live_trader.state_manager import StateManager
from live_trader.strategy import Strategy
from live_trader.trade import Trade
from stellaswap.stellaswap_token import StellaswapTokenContainer, StellaswapTokenPairContainer
from smart_order_router.graph import Graph
from smart_order_router.sor import single_sor_no_fees, single_sor_with_fees

import logging
from scipy.optimize import minimize_scalar
from typing import Any, Dict, List, Set, Tuple


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

"""
This is a copy of this file in the simulation folder. Longer term we should consolidate them
into one file (which requires defining common interfaces, etc.). It's actually a decent bit
of work, so I've punted that for another day. Copy-paste is bad 
"""
class StellaswapCyclicArbitrageStrategy(Strategy):

    def __init__(self, strategy_config, wallet_address, token_trading_universe: List[StellaswapTokenContainer]):
        self.wallet_address = wallet_address
        # The start and end of our paths across liquidity pools must be in this set of tokens
        self.token_trading_universe = token_trading_universe
        
        # Never put in less than 1 cent or more than $10,000 in a single swap
        # Gas fees are generally 1 cent, so it makes no sense to go below that
        # This seems quite high (and it is), but in theory with limit prices, our loss
        # is capped at gas fees of a failed transaction
        self.amount_in_lower_bound_usd = strategy_config['amount_in_lower_bound_usd']
        self.amount_in_upper_bound_usd = strategy_config['amount_in_upper_bound_usd']

        # We need to expect to make at least 3 cents to justify making the swap
        # Because we just net +tokens of what we put in, we do not incur any added cost
        # if the price of the token goes down
        self.min_expected_profit_usd = strategy_config['min_expected_profit_usd']

        # Do not execute if amount_out < ((expected_amount_out - amount_in) * (1 - slippage_rate)) + amount_in
        self.acceptable_slippage_rate = strategy_config['acceptable_slippage_rate']

        self.order_timeout_seconds = strategy_config['order_timeout_seconds']

        self.pending_order_ids: Set[int] = set()

    def on_state_update(self, state_manager: StateManager) -> List[Order]:
        if len(self.pending_order_ids) > 0:
            logging.warning(f'Not generating orders because we are pending receipt for our orders: {self.pending_order_ids}')
            return []
        potential_orders, order_id_to_token_pair_path = self._generate_profitable_orders(state_manager)
        portfolio_orders = self._generate_order_portfolio(potential_orders, order_id_to_token_pair_path)
        self.pending_order_ids = self.pending_order_ids.union(set([order.order_id for order in portfolio_orders]))
        return portfolio_orders

    def on_event(self, state_manager: StateManager, event_data: Dict[str, Any]) -> List[Order]:
        '''
        This strategy should never receive events
        '''
        raise NotImplementedError()

    def on_our_trades(self, state_manager: StateManager, new_trades: List[Trade]):
        logging.info(f'Received trades: {new_trades}')
        for t in new_trades:
            self.pending_order_ids.remove(t.order_id)

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

        for token in self.token_trading_universe:
            best_amount_in, best_amount_out, best_token_pair_path = self._scipy_find_optimal_amount_in(state_manager.token_graph, token)
            expected_pnl_usd = (best_amount_out - best_amount_in) * token.get_usd_value()

            if expected_pnl_usd > self.min_expected_profit_usd:
                amount_out_min = int((best_amount_out - best_amount_in) * (1 - self.acceptable_slippage_rate) + best_amount_in)
                new_order = Order(
                    order_id=state_manager.generate_order_id(),
                    amount_in=best_amount_in,
                    amount_out_min=amount_out_min,
                    path=self._get_token_path(best_token_pair_path, token.token_address),
                    to=self.wallet_address,
                    deadline=state_manager.last_timestamp + self.order_timeout_seconds,
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

    def _naive_find_optimal_amount_in(self, graph: Graph, token: StellaswapTokenContainer):
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
    
    def _scipy_find_optimal_amount_in(self, graph: Graph, token: StellaswapTokenContainer):
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
        wallet_balance = token.get_wallet_balance(self.wallet_address)
        if 0.9 * wallet_balance < upper_bound_tokens:
            # logging.warning(f'90% of Wallet balance ({wallet_balance}) is less than upper bound by USD equivalent ({upper_bound_tokens}). '
            #                 f'Setting the token upper bound to 90% of our wallet balance ({0.9 * wallet_balance})')
            upper_bound_tokens = 0.9 * wallet_balance
        
        # Find the optimum within 0.1 cents 
        tol_tokens = 0.001 / token.get_usd_value()
        res = minimize_scalar(objective_func, bounds=(lower_bound_tokens, upper_bound_tokens), method='bounded', options={'xatol': tol_tokens, 'disp': 0})
        best_amount_in = int(res.x)
        best_amount_out, best_token_pair_path = single_sor_with_fees(graph, token.token_address, token.token_address, amount_in=best_amount_in)
        return best_amount_in, best_amount_out, best_token_pair_path

    @staticmethod
    def _get_token_path(token_pair_path: List[StellaswapTokenPairContainer], start_token: str) -> List[str]:
        token_path = [start_token]
        for pair in token_pair_path:
            assert(token_path[-1] in [pair.token0_address, pair.token1_address])
            if pair.token0_address != token_path[-1]:
                token_path.append(pair.token0_address)
            else:
                token_path.append(pair.token1_address)
        return token_path
