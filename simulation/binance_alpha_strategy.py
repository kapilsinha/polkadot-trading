from common.enums import AlphaDirection, Direction
from common.trigger import ShouldCloseDecision, ShouldOpenDecision, TriggerContainer
from simulation.trading_simulator import Order, StateManager, Strategy, Trade
from simulation.sim_types import Token, TokenPair

from collections import defaultdict
from dataclasses import asdict, dataclass
from glob import glob
import logging
from typing import Any, Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

"""
By convention, we consider closing a position as selling back the amount we bought e.g.
If we swap 10 A for 5 B, then we close this position by later swapping 5 B for 9.9 A
"""
@dataclass
class TokenPosition:
    token_address: str
    allowance: float
    cumulative_position_delta: float = 0
    open_position_amount: float = 0

@dataclass(frozen=True)
class SimpleAlphaContainer:
    alpha_bps: float
    binance_price: float
    stellaswap_price: float

@dataclass(frozen=True)
class AlphaContainer(SimpleAlphaContainer):
    trigger_name: str
    alpha_direction: Optional[AlphaDirection]

    @staticmethod
    def create_from_open_decision(
        simple_alpha: SimpleAlphaContainer, open_decision: ShouldOpenDecision
    ):
        return AlphaContainer(
            alpha_bps=simple_alpha.alpha_bps,
            binance_price=simple_alpha.binance_price,
            stellaswap_price=simple_alpha.stellaswap_price,
            trigger_name=open_decision.trigger_name,
            alpha_direction=open_decision.alpha_direction,
        )

    @staticmethod
    def create_from_close_decision(
        simple_alpha: SimpleAlphaContainer, close_decision: ShouldCloseDecision
    ):
        return AlphaContainer(
            alpha_bps=simple_alpha.alpha_bps,
            binance_price=simple_alpha.binance_price,
            stellaswap_price=simple_alpha.stellaswap_price,
            trigger_name=close_decision.trigger_name,
            alpha_direction=None,
        )

class BinanceAlphaTokenPairStrategy(Strategy):

    def __init__(self, strategy_config, token_trading_universe: List[str], state_manager: StateManager):
        super().__init__()
        self.token_trading_universe = token_trading_universe
        self.pair_address = strategy_config['pair_address']

        # I don't think we ever will allow consecutive same side orders, so we don't support its logic anywhere
        assert(not strategy_config['allow_consecutive_same_side_orders'])

        self.allow_opposite_side_order_to_close = strategy_config['allow_opposite_side_order_to_close']
        self.max_our_trade_impact_rate_bps = strategy_config['max_our_trade_impact_rate_bps']
        self.risk = strategy_config['risk']

        pair = state_manager.address_to_token_pair[self.pair_address]
        self.token0_pos = TokenPosition(
            token_address=pair.token0_address,
            allowance=strategy_config['allowance'][pair.token0_address]
        )
        self.token1_pos = TokenPosition(
            token_address=pair.token1_address,
            allowance=strategy_config['allowance'][pair.token1_address]
        )

        self.binance_quote_cfg = strategy_config['binance_quote_calculation']
        self.binance_symbol_to_smooth_price = {}
        self.binance_symbol_to_price_exp_smooth_factor = {}
        for field in ['numerator', 'denominator']:
            if self.binance_quote_cfg[field]['type'] == 'binance':
                symbol = self.binance_quote_cfg[field]['symbol']
                self.binance_symbol_to_price_exp_smooth_factor[symbol] \
                    = self.binance_quote_cfg[field]['price_exp_smooth_factor']
        
        self.trigger_container = TriggerContainer(strategy_config)
        
        self.open_position_alpha_direction: Optional[AlphaDirection] = None
        # This is only needed to map trades to AlphaDirection, to update self.open_position_alpha_direction.
        # It can be cleared thereafter
        self.order_id_to_alpha_direction: Dict[int, AlphaDirection] = {}
    
    def on_state_update(self, state_manager: StateManager) -> List[Order]:
        simple_alpha = self.compute_binance_alpha(state_manager)
        if simple_alpha is None:
            logging.warning('Binance alpha is null (not yet initialized)...')
            return []
        open_decision = self.trigger_container.decide_should_open_position(alpha_bps=simple_alpha.alpha_bps)

        orders = []
        if self.open_position_alpha_direction is not None:
            if self.allow_opposite_side_order_to_close \
                    and open_decision.should_open_position \
                    and open_decision.alpha_direction != self.open_position_alpha_direction:
                # If we can open a position in the opposite direction to effectively close the current
                # open position, we choose to do so because it is more efficient than closing the current
                # position and then opening a position on the opposite side (one txn instead of two).
                alpha = AlphaContainer.create_from_open_decision(simple_alpha, open_decision)
                orders = self._generate_open_order(state_manager, alpha)
            else:
                close_decision = self.trigger_container.decide_should_close_position(
                    simple_alpha.alpha_bps, self._get_quote(state_manager))
                if close_decision.should_close_position:
                    alpha = AlphaContainer.create_from_close_decision(simple_alpha, close_decision)
                    orders = self._generate_close_order(state_manager, alpha)
        elif open_decision.should_open_position:
            alpha = AlphaContainer.create_from_open_decision(simple_alpha, open_decision)
            orders = self._generate_open_order(state_manager, alpha)
                    
        return orders

    def on_event(self, state_manager: StateManager, event_data: Dict[str, Any]):
        binance_symbol = event_data['binance_symbol']
        binance_price = event_data['binance_price']
        price_exp_smooth_factor = self.binance_symbol_to_price_exp_smooth_factor[binance_symbol]

        if binance_symbol not in self.binance_symbol_to_smooth_price:
            self.binance_symbol_to_smooth_price[binance_symbol] = binance_price
        else:
            self.binance_symbol_to_smooth_price[binance_symbol] = binance_price * price_exp_smooth_factor \
                + self.binance_symbol_to_smooth_price[binance_symbol] * (1 - price_exp_smooth_factor)
        
        return self.on_state_update(state_manager)

    def on_filled_orders(self, state_manager: StateManager, new_trades: List[Trade]):
        self.open_position_alpha_direction = self.order_id_to_alpha_direction[new_trades[0].order_id]
        assert(len(self.order_id_to_alpha_direction) == 1)
        del self.order_id_to_alpha_direction[new_trades[0].order_id]

        assert(len(new_trades) == 1) # since we only allow one outstanding order at a time
        new_trade = new_trades[0]
        if self.open_position_alpha_direction is not None:
            # We opened a position; it's possible it was not explicitly closed before if 
            # the new position is on the opposite side of the original and we have 
            # allow_opposite_side_order_to_close enabled
            self.trigger_container.create_close_triggers(
                self.open_position_alpha_direction, self._get_quote(state_manager))
            # Note that we CANNOT just set open_pos_amount to the new_trade_amount_delta
            # if we did an opposite-side order instead of closing the original one
            self.token0_pos.open_position_amount -= new_trade.amount0_delta
            self.token1_pos.open_position_amount -= new_trade.amount1_delta
        else:
            self.trigger_container.clear_close_triggers()
            assert(any((
                self.token0_pos.open_position_amount - new_trade.amount0_delta == 0,
                self.token1_pos.open_position_amount - new_trade.amount1_delta == 0,
            )))

            # The side that we just bought likely does not match amount_delta exactly, which accounts
            # for any profit or loss we made between the position's open and close
            self.token0_pos.open_position_amount = 0
            self.token1_pos.open_position_amount = 0
        
        self.token0_pos.cumulative_position_delta -= new_trade.amount0_delta
        self.token1_pos.cumulative_position_delta -= new_trade.amount1_delta
        assert(self.token0_pos.allowance + self.token0_pos.cumulative_position_delta >= 0)
        assert(self.token1_pos.allowance + self.token1_pos.cumulative_position_delta >= 0)

    def compute_binance_alpha(self, state_manager: StateManager) -> Optional[AlphaContainer]:
        def helper(binance_price, stellaswap_price):
            return 10_000 * (binance_price - stellaswap_price) / binance_price
        
        binance_price = self._compute_binance_token_pair_price(state_manager)
        if binance_price is None:
            return None
        stellaswap_price = self._get_quote(state_manager)
        alpha_bps = helper(binance_price, stellaswap_price)
        return SimpleAlphaContainer(
            alpha_bps=alpha_bps,
            binance_price=binance_price,
            stellaswap_price=stellaswap_price,
        )

    def _compute_binance_token_pair_price(self, state_manager: StateManager):
        def get_component_price(cfg):
            if cfg['type'] == 'binance':
                return self.binance_symbol_to_smooth_price.get(cfg['symbol'])
            if cfg['type'] == 'stellaswap':
                # Note that the USDC.usd_value() fluctuates a decent bit because any large trade changes it drastically
                # So it needs to be smoothed!
                decimals = state_manager.address_to_token_decimals[cfg['token_address']]
                return state_manager.address_to_token[cfg['token_address']].get_usd_value() * 10**decimals
            if cfg['type'] == 'constant':
                return cfg['value']
            raise ValueError('Should not reach here')

        num = get_component_price(self.binance_quote_cfg['numerator'])
        denom = get_component_price(self.binance_quote_cfg['denominator'])

        # We explicitly adjust the rate with decimals because all our quotes are in terms of wei
        token_pair = state_manager.address_to_token_pair[self.pair_address]
        token0_decimals = state_manager.address_to_token_decimals[token_pair.token0_address]
        token1_decimals = state_manager.address_to_token_decimals[token_pair.token1_address]

        return 10**(token1_decimals - token0_decimals) * (num / denom) if num is not None and denom is not None else None

    def _generate_open_order(
            self,
            state_manager: StateManager,
            alpha: AlphaContainer) -> List[Order]:
        token_pair = state_manager.address_to_token_pair[self.pair_address]
        if alpha.alpha_direction == AlphaDirection.BULLISH:
            trade_direction = Direction.REVERSE
            path = [token_pair.token1_address, token_pair.token0_address]
            limit_price = self._get_quote(state_manager) * (1 + self.max_our_trade_impact_rate_bps / 10_000)
            in_token_pos = self.token1_pos
        elif alpha.alpha_direction == AlphaDirection.BEARISH:
            trade_direction = Direction.FORWARD
            path = [token_pair.token0_address, token_pair.token1_address]
            limit_price = self._get_quote(state_manager) * (1 - self.max_our_trade_impact_rate_bps / 10_000)
            in_token_pos = self.token0_pos
        else:
            raise ValueError('Should not reach here')

        _trade_dir, max_amount_in, _ = token_pair.compute_trade_for_target_forward_quote(limit_price)
        assert(_trade_dir == trade_direction)

        # Need to cast to int since wei is all in ints
        amount_in = min(
            int(self.risk * (in_token_pos.allowance + in_token_pos.cumulative_position_delta)),
            int(max_amount_in),
        )
        new_order = self._create_order(amount_in, path, state_manager, alpha)
        return [new_order]

    def _generate_close_order(self, state_manager: StateManager, alpha: AlphaContainer) -> List[Order]:
        token_pair = state_manager.address_to_token_pair[self.pair_address]
        if self.open_position_alpha_direction == AlphaDirection.BULLISH:
            # Forward direction trade to close the position
            amount_in = self.token0_pos.open_position_amount
            path = [token_pair.token0_address, token_pair.token1_address]
        elif self.open_position_alpha_direction == AlphaDirection.BEARISH:
            amount_in = self.token1_pos.open_position_amount
            path = [token_pair.token1_address, token_pair.token0_address]
        else:
            raise ValueError('Attempting to close a position but open position is null')
        
        assert(amount_in > 0)
        new_order = self._create_order(amount_in, path, state_manager, alpha)
        return [new_order]

    def _create_order(self, amount_in: int, path: List[str],
                      state_manager: StateManager,
                      alpha: AlphaContainer) -> Order:
        order = Order(
            order_id=state_manager.generate_order_id(),
            amount_in=amount_in,
            path=path,
            block_num=state_manager.cur_block_num,
            last_txn_index=state_manager.last_txn_index,
            metadata={
                'input_usd_notional': amount_in * state_manager.address_to_token[path[0]].get_usd_value(),
                **asdict(alpha),
            },
        )
        self.order_id_to_alpha_direction[order.order_id] = alpha.alpha_direction
        return order

    def _get_quote(self, state_manager: StateManager):
        token_pair = state_manager.address_to_token_pair[self.pair_address]
        return token_pair.quote_no_fees(Direction.FORWARD)


if __name__ == '__main__':
    from common.helpers import load_config
    from simulation.trading_simulator import StateManager, FillEngine, PnlCalculator, SimDriver
    import sys

    config_name = sys.argv[1]
    logging.info(f'Below results are for {config_name}')

    config = load_config(filename='sim_cfg.yaml')
    strategy_config = config['strategy']['binance_alpha'][config_name]

    token_holdings = defaultdict(int)
    for token_address, holdings in config['venue']['stellaswap']['holdings'].items():
        token_holdings[token_address] = holdings

    state_manager = StateManager(config, token_holdings)
    fill_engine = FillEngine(should_update_state_reserves_after_trades=True)
    pnl_calculator = PnlCalculator()
    cycle_strategy = BinanceAlphaTokenPairStrategy(
        strategy_config, config['venue']['stellaswap']['token_trading_universe'], state_manager
    )
    sim_driver = SimDriver(
        state_manager=state_manager,
        fill_engine=fill_engine,
        pnl_calculator=pnl_calculator,
        strategy=cycle_strategy,
        should_trigger_strategy_on_end_of_block=True,
        should_trigger_strategy_on_txn=False,
    )

    files = [
        f'data/stellaswap_txn_history/all/stellaswap_data_{x}0000_{x}9999.feather' for x in range(175, 185)
    ]
    binance_symbol_to_data_feather_files = {
        'ETHUSDT': [f.replace(
                'data/stellaswap_txn_history/all/stellaswap_data',
                'data/binance_history/eth_usdt/processed/binance_data'
            ) for f in files]
    }
    sim_driver.process_files(files, binance_symbol_to_data_feather_files)
    logging.info(f'Above results are for {config_name}')
