from common.enums import DataRowType, Direction
from smart_order_router.graph import Graph
from simulation.sim_types import Order, Token, TokenPair, Trade

from abc import ABC, abstractmethod
import logging
import numpy as np
import pandas as pd
from typing import Any, DefaultDict, Dict, List, Optional


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

class StateManager:

    _order_id = 1

    def __init__(self, config, address_to_token_holdings: DefaultDict[str, int]):
        self.is_bootstrapped = False # set and accessed only by the driver
        self.cur_block_num: Optional[int] = None
        self.last_txn_index: Optional[int] = None
        self.last_swaps: Optional[pd.DataFrame] = None # unused currently but this can lend itself to an onEvent callback
        self.address_to_token: Dict[str, Token] = {}

        # We initialize this with all the token pairs we consider trading. We don't distinguish between those that 'exist'
        # (contract was created) at a given block vs those that do not.
        # The ones that don't exist will have reserve0 = reserve1 = None
        # Hence the graph of const TokenPair references can remain static (just the reserve amounts within TokenPairs change)
        data_paths = config['venue']['stellaswap']['data_path']
        liquid_pairs = pd.read_csv(data_paths['liquid_pairs']).pair_address
        df = pd.read_csv(data_paths['pair_metadata'])
        df = df[df.pair_address.isin(liquid_pairs)]

        self.address_to_token_decimals = {
            **dict(zip(df.token0_address, df.token0_decimals)),
            **dict(zip(df.token1_address, df.token1_decimals)),
        }

        self.address_to_start_token_holdings: DefaultDict[str, int] = address_to_token_holdings.copy()
        self.address_to_token_holdings: DefaultDict[str, int] = address_to_token_holdings
        self.address_to_token_pair: Dict[str, TokenPair] = {
            pair_address: TokenPair(pair_address, token0, token1) for pair_address, token0, token1 \
                in zip(df.pair_address, df.token0_address, df.token1_address)
        }
        self.token_graph = Graph(self.address_to_token_pair.values())

        # Stores (token0, token1) -> pair_address and (token1, token0) -> pair_address
        self.tokens_to_pair = {
            **dict(zip(tuple(zip(df.token0_address, df.token1_address)), df.pair_address)),
            **dict(zip(tuple(zip(df.token1_address, df.token0_address)), df.pair_address)),
        }

    @classmethod
    def generate_order_id(cls):
        order_id = cls._order_id
        cls._order_id += 1
        return order_id

    def on_end_of_block(self, block_number, data: pd.DataFrame):
        assert((data['block_number'] == block_number).all())
        assert(set(data['row_type']).issubset(set([DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT, DataRowType.END_OF_BLOCK_TOKEN_SNAPSHOT])))
        self.cur_block_num = block_number
        
        token_pair_rows = data[data.row_type == DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT]
        for pair_address, r0, r1 in zip(token_pair_rows.pair_address, token_pair_rows.reserve0, token_pair_rows.reserve1):
            self.address_to_token_pair[pair_address].set_reserves(r0, r1)

        token_rows = data[data.row_type == DataRowType.END_OF_BLOCK_TOKEN_SNAPSHOT]
        for token_address, dai_value in zip(token_rows.token_address, token_rows['dai-multi_equiv_no_fees']):
            self.address_to_token[token_address] = Token(token_address, usd_value=dai_value * 1e-18)
        
        self.last_txn_index = None

    def on_txn(self, block_number, txn_index, data: pd.DataFrame):
        assert((data['block_number'] == block_number).all())
        assert((data['txn_index'] == txn_index).all())
        assert(set(data['row_type']).issubset(set([DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT, DataRowType.ON_UPDATE_TOKEN_SNAPSHOT, DataRowType.SWAP_TXN])))
        self.cur_block_num = block_number

        token_pair_rows = data[data.row_type == DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT]
        for pair_address, r0, r1 in zip(token_pair_rows.pair_address, token_pair_rows.reserve0, token_pair_rows.reserve1):
            self.address_to_token_pair[pair_address].set_reserves(r0, r1)

        token_rows = data[data.row_type == DataRowType.ON_UPDATE_TOKEN_SNAPSHOT]
        for token_address, dai_value in zip(token_rows.token_address, token_rows['dai-multi_equiv_no_fees']):
            self.address_to_token[token_address] = Token(token_address, usd_value=dai_value * 1e-18)

        self.last_swaps = data[data.row_type == DataRowType.SWAP_TXN]
        self.last_txn_index = txn_index
    
    def get_readable_start_holdings(self):
        return {
            token_address: f'${self.address_to_token[token_address].get_usd_value() * holdings} ({holdings} wei)' \
                for token_address, holdings in self.address_to_start_token_holdings.items()
        }

    def get_readable_holdings(self):
        return {
            token_address: f'${self.address_to_token[token_address].get_usd_value() * holdings} ({holdings} wei)' \
                for token_address, holdings in self.address_to_token_holdings.items()
        }


class FillEngine:

    _trade_id = 1

    def __init__(self, should_update_state_reserves_after_trades):
        self.should_update_state_reserves_after_trades = should_update_state_reserves_after_trades

    @staticmethod
    def _update_state_reserves_after_trades(state_manager: StateManager, trade: Trade):
        '''
        TODO: If these schemes become more sophisticated, place them in a new MarketImpactModel ?
        This can be considered a crude attempt to model "market impact"
        We update the reserves based on our trades. These will get completely overwritten
        the next time we see an update of course, so this is really just a de-duplication scheme
        Ideally we would also update token USD values, but that's really difficult and
        likely doesn't change drastically as a result of our trading
        '''
        token_pair = state_manager.address_to_token_pair[trade.pair_address]
        token_pair.set_reserves(token_pair.reserve0 + trade.amount0_delta, token_pair.reserve1 + trade.amount1_delta)
    
    @staticmethod
    def _update_our_holdings_after_trades(state_manager: StateManager, trade: Trade):
        token_pair = state_manager.address_to_token_pair[trade.pair_address]
        state_manager.address_to_token_holdings[token_pair.token0_address] -= trade.amount0_delta
        state_manager.address_to_token_holdings[token_pair.token1_address] -= trade.amount1_delta
        
        if state_manager.address_to_token_holdings[token_pair.token0_address] < 0 or \
            state_manager.address_to_token_holdings[token_pair.token1_address] < 0:
            logging.error(
                f'Holdings went below 0. You cannot sell what you do not have. The strategy violates holdings constraints.\n'
                f'Holdings: {state_manager.get_readable_holdings()}\n'
                f'Violating trade: {trade}'
            )
            raise ValueError('Holdings for tokens went below 0')

    @classmethod
    def generate_trade_id(cls):
        trade_id = cls._trade_id
        cls._trade_id += 1
        return trade_id

    '''
    Loops over orders (in order) and just fills them all (assumes no slippage, execution failures, etc)
    This does NOT update the reserve amounts in the token pairs for simplicity i.e. we are unrealistically
    assuming that we have zero market impact.
    '''
    def fill_orders(self, state_manager: StateManager, orders: List[Order]) -> List[Trade]:
        def are_order_endpoints_disjoint(orders):
            # As a sanity check, we disallow the in- and out- tokens for any orders to overlap
            s = set()
            for order in orders:
                # Note that start and end can be the same
                start, end = order.path[0], order.path[-1]
                if start in s or end in s:
                    return False
                s.add(start)
                s.add(end)
            return True

        assert(are_order_endpoints_disjoint(orders))
        out_trades = []
        for o in orders:
            amount_in = o.amount_in
            for i in range(len(o.path) - 1):
                start_token = o.path[i]
                end_token = o.path[i + 1]
                
                pair_address = state_manager.tokens_to_pair[(start_token, end_token)]
                token_pair = state_manager.address_to_token_pair[pair_address]
                if start_token == token_pair.token0_address and end_token == token_pair.token1_address:
                    amount_out = int(token_pair.quote_with_fees(Direction.FORWARD, amount_in))
                    amount0_delta = amount_in
                    amount1_delta = -amount_out
                elif start_token == token_pair.token1_address and end_token == token_pair.token0_address:
                    amount_out = int(token_pair.quote_with_fees(Direction.REVERSE, amount_in))
                    amount0_delta = -amount_out
                    amount1_delta = amount_in
                else:
                    raise ValueError('StateManager.tokens_to_pair was incorrectly initialized')
                                
                new_trade = Trade(
                    order_id=o.order_id,
                    trade_id=self.generate_trade_id(),
                    pair_address=pair_address,
                    amount0_delta=amount0_delta,
                    amount1_delta=amount1_delta,
                )
                
                self._update_our_holdings_after_trades(state_manager, new_trade)
                if self.should_update_state_reserves_after_trades:
                    self._update_state_reserves_after_trades(state_manager, new_trade)

                out_trades.append(new_trade)
                # Propagate the amount_out token as the amount_in for the next liquidity pool in the path
                amount_in = amount_out
        return out_trades


class PnlCalculator:

    def __init__(self):
        self.is_bootstrapped = False # set and accessed only by the driver
        self.called_bootstrap = False # used only as a sanity check
        
        self.start_holdings_value = None
        self.cur_holdings_value = None
        
        self.total_exec_pnl = 0
        self.trade_id_to_exec_pnl = {}
        
        self.prev_total_pnl = 0
        self.total_pnl = 0

    def bootstrap(self, state_manager: StateManager):
        assert(not self.called_bootstrap)
        self.called_bootstrap = True
        self.start_holdings_value = self.calc_holdings_value(state_manager)
        self.cur_holdings_value = self.start_holdings_value

    def calc_pnl(self, state_manager: StateManager, trades: List[Trade]):
        assert(self.is_bootstrapped)
        self.calc_exec_pnl(state_manager, trades)
        self.prev_total_pnl = self.total_pnl
        self.cur_holdings_value = self.calc_holdings_value(state_manager)
        self.total_pnl = self.cur_holdings_value - self.start_holdings_value

    def calc_holdings_value(self, state_manager: StateManager):
        total_value = 0
        for token, holding in state_manager.address_to_token_holdings.items():
            total_value += holding * state_manager.address_to_token[token].get_usd_value()
        return total_value
    
    # We consider this an 'execution pnl' but I think it's a bad metric. Legacy from the cyclic arbitrage strat
    def calc_exec_pnl(self, state_manager: StateManager, trades: List[Trade]):
        for trade in trades:
            token_pair = state_manager.address_to_token_pair[trade.pair_address]
            pnl = -trade.amount0_delta * state_manager.address_to_token[token_pair.token0_address].get_usd_value() \
                 - trade.amount1_delta * state_manager.address_to_token[token_pair.token1_address].get_usd_value()
            self.trade_id_to_exec_pnl[trade.trade_id] = pnl
            self.total_exec_pnl += pnl


class Strategy(ABC):
    '''
    We will pretend we have unlimited amounts of every token. Can make that more realistic later.
    But for actual trading I hope it suffices to buy some of all the major tokens (depends on strategy logic).
    '''
    @abstractmethod
    def on_state_update(self, state_manager: StateManager) -> List[Order]:
        pass

    @abstractmethod
    def on_event(self, state_manager: StateManager, event_data: Dict[str, Any]) -> List[Order]:
        pass

    @abstractmethod
    def on_filled_orders(self, state_manager: StateManager, new_trades: List[Trade]):
        pass


class SimDriver:
    '''
    Plays through the raw data files, processing block by block, transaction by transaction, NOT log by log
    because it makes no sense to stop the flow within a transaction. Owns all the simulation components.
    
    For a given block, the data order is
        For each transaction, zero or more chunks of these (in order):
        1. DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT
        2. DataRowType.ON_UPDATE_TOKEN_SNAPSHOT
        3. DataRowType.SWAP_TXN
        At the end of a block. Zero or more chunks of these (in order):
        1. DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT
        2. DataRowType.END_OF_BLOCK_TOKEN_SNAPSHOT
    '''

    def __init__(
            self,
            state_manager: StateManager,
            fill_engine: FillEngine,
            pnl_calculator: PnlCalculator,
            strategy: Strategy,
            should_trigger_strategy_on_end_of_block: bool,
            should_trigger_strategy_on_txn: bool
        ):
        self.state_manager = state_manager
        self.fill_engine = fill_engine
        self.pnl_calculator = pnl_calculator
        self.strategy = strategy

        self.should_trigger_strategy_on_end_of_block = should_trigger_strategy_on_end_of_block
        self.should_trigger_strategy_on_txn = should_trigger_strategy_on_txn

    """
    stellaswap_data_feather_files: List of files containing the blockchain data
    binance_data_feather_files: List of binance data files with several constraints
    Note that binance_data_feather_files[i] must match (time-wise) with stellaswap_data_feather_files[i]
    """
    def process_files(
            self,
            stellaswap_data_feather_files: List[str],
            binance_symbol_to_data_feather_files: Dict[str, List[str]] = {}
        ):
        for binance_data_feather_files in binance_symbol_to_data_feather_files.values():
            assert(len(stellaswap_data_feather_files) == len(binance_data_feather_files))
        
        # We process a file at a time to avoid unnecessarily loading lots of data into memory
        for i, stella_file in enumerate(stellaswap_data_feather_files):
            stellaswap_df = pd.read_feather(stella_file)
            binance_files = {symbol: files[i] for symbol, files in binance_symbol_to_data_feather_files.items()}
            binance_dfs = {symbol: pd.read_feather(f) for symbol, f in binance_files.items()}
            logging.info(f'Processing {stella_file}, {binance_files}...')
            
            self._process_dfs(stellaswap_df, binance_dfs)
        
        self.pnl_calculator.calc_pnl(self.state_manager, [])
        logging.info(f'Start holdings value: ${self.pnl_calculator.start_holdings_value:0.2f} '
                     f'({self.state_manager.get_readable_start_holdings()})')
        logging.info(f'End holdings value: ${self.pnl_calculator.cur_holdings_value:0.2f}'
                     f'({self.state_manager.get_readable_holdings()})')
        
        token_to_delta_usd_value = self._calc_token_to_delta_usd_value()
        logging.info(f'Actual total PnL: ${self.pnl_calculator.total_pnl:0.2f}')
        logging.info(f'Total PnL (compared to doing nothing): ${sum(token_to_delta_usd_value.values()):0.2f} '
                     f'({self.format_token_to_usd(token_to_delta_usd_value)})')

    def _process_dfs(self, stellaswap_df: pd.DataFrame, binance_dfs: Dict[str, pd.DataFrame]):
        def is_chunk_full_snapshot(block_number, txn_index):
            # Sorta hacky. Corresponds to the DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT,
            # DataRowType.END_OF_BLOCK_TOKEN_SNAPSHOT full snapshot chunk
            # that occurs every 200 blocks.
            return block_number % 200 == 0 and np.isnan(txn_index)

        def validate_df_timestamps_overlap(stellaswap_df, binance_dfs):
            # Sanity check that df and aux_dfs time ranges overlap
            min_block_millis = stellaswap_df['block_timestamp'].min() * 1000
            max_block_millis = stellaswap_df['block_timestamp'].max() * 1000
            for binance_df in binance_dfs.values():
                assert(any((
                    binance_df['timestamp'].min() <= min_block_millis <= binance_df['timestamp'].max(),
                    min_block_millis <= binance_df['timestamp'].min() <= max_block_millis,
                )))

        validate_df_timestamps_overlap(stellaswap_df, binance_dfs)
        binance_symbol_to_next_index = {sym: 0 for sym in binance_dfs.keys()}
        num_processed_rows = 0

        for (block_number, txn_index), indices in stellaswap_df.groupby([
            'block_number', 'txn_index'], dropna=False).indices.items():
            # indices happens to be sorted as we want due to implementation quirks,
            # but we verify that and also the built-in row order
            assert(np.array_equal(indices, np.arange(num_processed_rows, num_processed_rows + len(indices))))
            num_processed_rows += len(indices)

            epoch_millis = stellaswap_df.iloc[indices[0]]['block_timestamp'] * 1000
            self._dispatch_binance_price_events(
                binance_dfs, epoch_millis, binance_symbol_to_next_index, block_number, txn_index)

            if not self.state_manager.is_bootstrapped and not is_chunk_full_snapshot(block_number, txn_index):
                logging.warning(f'Waiting for state manager to bootstrap off a full snapshot; '
                                f'we are throwing away block {block_number}, txn {txn_index}')
                continue
            elif np.isnan(txn_index):
                self.state_manager.on_end_of_block(block_number, stellaswap_df.iloc[indices])
                self.state_manager.is_bootstrapped = True
                if not self.pnl_calculator.is_bootstrapped:
                    self.pnl_calculator.bootstrap(self.state_manager)
                    self.pnl_calculator.is_bootstrapped = True
                should_trigger_strategy = self.should_trigger_strategy_on_end_of_block
            else:
                self.state_manager.on_txn(block_number, txn_index, stellaswap_df.iloc[indices])
                should_trigger_strategy = self.should_trigger_strategy_on_txn
            
            if should_trigger_strategy:
                new_orders = self.strategy.on_state_update(self.state_manager)
                self._handle_new_orders(new_orders, block_number, txn_index)

        self._dispatch_binance_price_events(
            binance_dfs, float('inf'), binance_symbol_to_next_index, block_number, txn_index)

    def _dispatch_binance_price_events(
            self,
            binance_dfs: Dict[str, pd.DataFrame],
            timestamp_millis_hi: float,
            binance_symbol_to_next_index: Dict[str, int],
            block_number: int,
            txn_index: int
        ):
        for event in self._get_binance_price_events(binance_dfs, timestamp_millis_hi, binance_symbol_to_next_index):
            new_orders = self.strategy.on_event(self.state_manager, event)
            self._handle_new_orders(new_orders, block_number, txn_index)
    
    def _get_binance_price_events(
            self,
            binance_dfs: Dict[str, pd.DataFrame],
            timestamp_millis_hi: int,
            binance_symbol_to_next_index: Dict[str, int]
        ):
        # Note that binance_symbol_to_next_index gets updated inside this function!
        events = []
        for binance_symbol, binance_df in binance_dfs.items():
            while binance_symbol_to_next_index[binance_symbol] < len(binance_df) \
                and binance_df.iloc[binance_symbol_to_next_index[binance_symbol]]['timestamp'] < timestamp_millis_hi:
                row = binance_df.iloc[binance_symbol_to_next_index[binance_symbol]]
                events.append({
                    'binance_symbol': binance_symbol,
                    'timestamp': row['timestamp'],
                    'binance_price': row['price'],
                })
                binance_symbol_to_next_index[binance_symbol] += 1
        # Sorting should not matter but we sort across the binance symbols for good measure
        return sorted(events, key=lambda x: x['timestamp'])
    
    def _handle_new_orders(self, new_orders: List[Order], block_number: int, txn_index: int):
        if len(new_orders) > 0:           
            new_trades = self.fill_engine.fill_orders(self.state_manager, new_orders)
            self.strategy.on_filled_orders(self.state_manager, new_trades)

            self.pnl_calculator.calc_pnl(self.state_manager, new_trades)

            # TODO: This belongs in PnlCalculator but I don't know what to call this metric. Find a name and move
            # I hate this name exec2_pnl
            token_to_delta_usd_value = self._calc_token_to_delta_usd_value()
            cumulative_exec2_pnl = sum(token_to_delta_usd_value.values())
            logging.info(f'Block {block_number}, txn {txn_index}: orders = {new_orders}')
            logging.info(f'Block {block_number}, txn {txn_index}: trades =' \
                        f'{ [f"{t} (pnl=${self.pnl_calculator.trade_id_to_exec_pnl[t.trade_id]})" for t in new_trades] }')
            logging.info(f'Block {block_number}, holdings = {self.state_manager.get_readable_holdings()}')
            logging.info(f'Block {block_number}, txn {txn_index}: '
                        # f'total_pnl_delta_since_last_trades=${(self.pnl_calculator.total_pnl - self.pnl_calculator.prev_total_pnl):0.2f}, '
                        # f'cumulative_exec_pnl=${self.pnl_calculator.total_exec_pnl:0.2f}, '
                        # f'cumulative_total_pnl=${self.pnl_calculator.total_pnl:0.2f}, '
                        f'cumulative_exec2_pnl=${cumulative_exec2_pnl:0.2f}, '
                        f'({self.format_token_to_usd(token_to_delta_usd_value)})\n')

    def _calc_token_to_delta_usd_value(self):
        token_to_delta_usd_value = {}
        for token in self.state_manager.address_to_token_holdings.keys():
            delta_token_holdings = self.state_manager.address_to_token_holdings[token] \
                - self.state_manager.address_to_start_token_holdings[token]
            delta_token_usd = self.state_manager.address_to_token[token].get_usd_value() * delta_token_holdings
            if delta_token_usd != 0:
                token_to_delta_usd_value[token] = delta_token_usd
        return token_to_delta_usd_value

    @staticmethod
    def format_token_to_usd(token_to_usd):
        return {k: f'${v:0.2f}' for k, v in token_to_usd.items()}
