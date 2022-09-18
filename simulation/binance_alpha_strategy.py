from common.enums import AlphaDirection, Direction
from common.trigger import TriggerContainer
from simulation.trading_simulator import Order, StateManager, Strategy, Trade
from simulation.sim_types import Token, TokenPair

from collections import defaultdict
from dataclasses import dataclass
from glob import glob
import logging
from typing import Any, Dict, List, Optional, Set, Tuple


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
        alpha_bps = self.compute_binance_alpha_bps(state_manager)
        if alpha_bps is None:
            logging.warning('Binance alpha is null (not yet initialized)...')
            return []
        should_open_position, new_direction = \
            self.trigger_container.should_open_position_and_direction(alpha_bps=alpha_bps)

        orders = []
        if self.open_position_alpha_direction is not None:
            if self.allow_opposite_side_order_to_close \
                    and should_open_position and new_direction != self.open_position_alpha_direction:
                # If we can open a position in the opposite direction to effectively close the current
                # open position, we choose to do so because it is more efficient than closing the current
                # position and then opening a position on the opposite side (one txn instead of two).
                orders = self._generate_open_order(state_manager, new_direction, alpha_bps)
            elif self.trigger_container.should_close_position(alpha_bps, self._get_quote(state_manager)):
                orders = self._generate_close_order(state_manager, alpha_bps)
        elif should_open_position:
            orders = self._generate_open_order(state_manager, new_direction, alpha_bps)
                    
        return orders

    def on_event(self, state_manager: StateManager, event_data: Dict[str, Any]):
        binance_symbol = event_data['binance_symbol']
        binance_price = event_data['binance_price']
        price_exp_smooth_factor = self.binance_symbol_to_price_exp_smooth_factor[binance_symbol]

        if binance_symbol not in self.binance_symbol_to_smooth_price:
            self.binance_symbol_to_smooth_price[binance_symbol] = binance_price
        else:
            self.binance_symbol_to_smooth_price[binance_symbol] = binance_price * (1 - price_exp_smooth_factor) \
                + self.binance_symbol_to_smooth_price[binance_symbol] * price_exp_smooth_factor
        
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
            self.token0_pos.open_position_amount = -new_trade.amount0_delta
            self.token1_pos.open_position_amount = -new_trade.amount1_delta
        else:
            self.trigger_container.clear_close_triggers()
            if self.token0_pos.open_position_amount > 0:
                assert(self.token0_pos.open_position_amount - new_trade.amount0_delta == 0)
            elif self.token1_pos.open_position_amount > 0:
                assert(self.token1_pos.open_position_amount - new_trade.amount1_delta == 0)
            else:
                raise ValueError('One of token0 or token1 open positions should have been positive')
            # The side that we just bought likely does not match amount_delta exactly, which accounts
            # for any profit or loss we made between the position's open and close
            self.token0_pos.open_position_amount = 0
            self.token1_pos.open_position_amount = 0
        
        self.token0_pos.cumulative_position_delta -= new_trade.amount0_delta
        self.token1_pos.cumulative_position_delta -= new_trade.amount1_delta
        assert(self.token0_pos.allowance + self.token0_pos.cumulative_position_delta >= 0)
        assert(self.token1_pos.allowance + self.token1_pos.cumulative_position_delta >= 0)

    def compute_binance_alpha_bps(self, state_manager: StateManager):
        def helper(binance_price, stellaswap_price):
            return 10_000 * (binance_price - stellaswap_price) / binance_price
        
        binance_price = self._compute_binance_token_pair_price(state_manager)
        if binance_price is None:
            return None
        stellaswap_price = self._get_quote(state_manager)
        return helper(binance_price, stellaswap_price)

    def _compute_binance_token_pair_price(self, state_manager: StateManager):
        def get_component_price(cfg):
            if cfg['type'] == 'binance':
                return self.binance_symbol_to_smooth_price.get(cfg['symbol'])
            if cfg['type'] == 'stellaswap':
                decimals = state_manager.address_to_token_decimals[cfg['token_address']]
                return state_manager.address_to_token[cfg['token_address']].get_usd_value() * 10**decimals
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
            alpha_direction: AlphaDirection,
            alpha_bps: float) -> List[Order]:
        token_pair = state_manager.address_to_token_pair[self.pair_address]
        if alpha_direction == AlphaDirection.BULLISH:
            trade_direction = Direction.REVERSE
            path = [token_pair.token1_address, token_pair.token0_address]
            limit_price = self._get_quote(state_manager) * (1 + self.max_our_trade_impact_rate_bps / 10_000)
            in_token_pos = self.token1_pos
        elif alpha_direction == AlphaDirection.BEARISH:
            trade_direction = Direction.FORWARD
            path = [token_pair.token0_address, token_pair.token1_address]
            limit_price = self._get_quote(state_manager) * (1 - self.max_our_trade_impact_rate_bps / 10_000)
            in_token_pos = self.token0_pos
        else:
            raise ValueError('Should not reach here')

        _trade_dir, max_amount_in, _ = token_pair.compute_trade_for_target_forward_quote(limit_price)
        assert(_trade_dir == trade_direction)

        amount_in = min(
            self.risk * (in_token_pos.allowance + in_token_pos.cumulative_position_delta),
            max_amount_in,
        )
        new_order = self._create_order(amount_in, path, alpha_direction, state_manager, alpha_bps)
        return [new_order]

    def _generate_close_order(self, state_manager: StateManager, alpha_bps: float) -> List[Order]:
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
        new_order = self._create_order(amount_in, path, None, state_manager, alpha_bps)
        return [new_order]

    def _create_order(self, amount_in: float, path: List[str],
                      new_alpha_direction: Optional[AlphaDirection],
                      state_manager: StateManager,
                      alpha_bps: float) -> Order:
        order = Order(
            order_id=state_manager.generate_order_id(),
            amount_in=amount_in,
            path=path,
            block_num=state_manager.cur_block_num,
            last_txn_index=state_manager.last_txn_index,
            metadata={
                'input_usd_notional': amount_in * state_manager.address_to_token[path[0]].get_usd_value(),
                'alpha_bps': alpha_bps,
                'cur_price': self._get_quote(state_manager),
            },
        )
        self.order_id_to_alpha_direction[order.order_id] = new_alpha_direction
        return order

    def _get_quote(self, state_manager: StateManager):
        token_pair = state_manager.address_to_token_pair[self.pair_address]
        return token_pair.quote_no_fees(Direction.FORWARD)


if __name__ == '__main__':
    from common.helpers import load_config
    from simulation.trading_simulator import StateManager, FillEngine, PnlCalculator, SimDriver

    config = load_config(filename='sim_cfg.yaml')
    strategy_config = config['strategy']['binance_alpha']['usdc_eth']

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
        'data/stellaswap_txn_history/all//stellaswap_data_1740000_1749999.feather',
        # 'data/stellaswap_txn_history/all/stellaswap_data_1750000_1759999.feather',
        # 'data/stellaswap_txn_history/all/stellaswap_data_1760000_1769999.feather',
    ]
    binance_symbol_to_data_feather_files = {
        'ETHUSDT': [
            'data/binance_history/eth_usdt/processed/binance_data_1740000_1749999.feather',
            # 'data/binance_history/eth_usdt/processed/binance_data_1750000_1759999.feather',
            # 'data/binance_history/eth_usdt/processed/binance_data_1760000_1769999.feather',
        ]
    }
    sim_driver.process_files(files, binance_symbol_to_data_feather_files)
