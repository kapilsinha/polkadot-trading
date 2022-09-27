from common.helpers import load_config, load_contract
from live_trader.order import Order
from live_trader.state_manager import StateManager
from live_trader.trade import Trade
from live_trader.order_sender import OrderSender
from live_trader.filter_container import BlockFilterContainer, ContractEventFilterContainer, FilterContainer

import datetime
import logging
import sys
from time import sleep
from timer import timer
from typing import Any, Callable, Dict, List, Optional
import web3 as w3


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)
timer.set_level(logging.DEBUG)

class LiveTrader:
    def __init__(self, config, rpc_provider, rpc_type = 'ws'):
        if rpc_type == 'ws':
            self.web3 = w3.Web3(w3.Web3.WebsocketProvider(config['rpc_endpoint']['ws'][rpc_provider]))
        elif rpc_type == 'http':
            self.web3 = w3.Web3(w3.Web3.HTTPProvider(config['rpc_endpoint']['http'][rpc_provider]))
        else:
            raise ValueError('Invalid rpc_type: must be "ws" or "http"')

        self.cfg = config['venue']['stellaswap']
        self.state_manager = StateManager(self.cfg, self.web3)

        router = load_contract(
            self.cfg['contract_address']['router'],
            self.cfg['abi_path']['router'],
            self.web3,
        )

        # Simulates a fill for an order if its path is length 2 (i.e. one token pair).
        # TODO: can extend this to simulate arbitrary orders like the FillEngine in our sim
        # but for sake of speed, this does just simple orders
        self.simple_simulate_fills = config['order_sender']['simple_simulate_fills']
        if self.simple_simulate_fills and config['order_sender']['should_send_orders']:
            raise ValueError('Cannot simulate fills if we are actually sending orders. '
                             'That would be really bad because our strategy would get a mix of '
                             'real and simulated fills...')
        self.order_sender = OrderSender(
            router,
            self.web3,
            self.state_manager.address_to_token['0xAcc15dC74880C9944775448304B263D191c6077F'],
            config['order_sender']
        )

        self.strategy = None
        self.create_filters()
        logging.info('Finished initializing trader bot')

    def set_strategy(self, strategy):
        self.strategy = strategy

    def create_filters(self) -> List[FilterContainer]:
        block_filter = BlockFilterContainer(self.web3, self.state_manager.cur_block_num, self.handle_new_block)
        sync_filters = [
            ContractEventFilterContainer(token_pair.contract.events.Sync, self.state_manager.cur_block_num, self.handle_token_pair_sync_event) \
                for token_pair in self.state_manager.address_to_token_pair.values()
        ]
        swap_filters = [
            ContractEventFilterContainer(token_pair.contract.events.Swap, self.state_manager.cur_block_num, self.handle_token_pair_swap_event) \
                for token_pair in self.state_manager.address_to_token_pair.values()
        ]
        # TODO: Figure out why the tx filter doesn't work even for our LAN node
        # tx_filter = w3.eth.filter('pending')
        return [
            block_filter,
            *sync_filters,
            *swap_filters,
        ]

    def handle_new_block(self, block_hash):
        num_retries = 5
        while num_retries > 0:
            try:
                block_info = self.web3.eth.get_block(block_hash)
                logging.info(f'Processing block number {block_info.number}')
                txns = [txn.hex() for txn in block_info.transactions]
                for txn_hash in txns:
                    if txn_hash in self.state_manager.our_pending_txns:
                        # Status == 0 -> failed txn, status == 1 -> successful txn
                        is_fail = self.web3.eth.get_transaction_receipt(txn_hash).status == 0
                        if is_fail:
                            self.state_manager.remove_pending_txn(txn_hash)
                            order_id = self.order_sender.pop_order_id_for_txn(txn_hash)
                            self.state_manager.add_new_trade(Trade(order_id=order_id, txn_hash=txn_hash, is_success=False, amount0_delta=0, amount1_delta=0))
                
                self.state_manager.set_cur_block_num(block_info.number)
                self.state_manager.set_last_timestamp(block_info.timestamp)
                return
            except TypeError as e:
                if len(e.args) > 0 and 'unrecognized block reference' in e.args[0]:
                    logging.error(f'Block handler received "unknown block" with error {e}. '
                                  f'Sleeping for 1 second and will retry ({num_retries} retries left).')
                else:
                    raise e
            except w3.exceptions.BlockNotFound as e:
                logging.error(f'Block handler received "unknown block" with error {e}. '
                              f'Sleeping for 1 second and will retry ({num_retries} retries left).')
            sleep(1)
            num_retries -= 1
        raise SystemError("Ran out of retries")

    def handle_token_pair_sync_event(self, event):
        """
        Example event: AttributeDict(
            {'args': AttributeDict({'reserve0': 17514971840, 'reserve1': 13221901822289375953}),
             'event': 'Sync', 'logIndex': 21, 'transactionIndex': 6,
             'transactionHash': HexBytes('0x7c2d4ea683f4cba758acbfaeb7a0a7d181165ea80a1b9b8b52fd04b5d95ab67b'),
             'address': '0x0Aa48bF937ee8F41f1a52D225EF5A6F6961e39FA',
             'blockHash': HexBytes('0x3738df17a262e68d740feaaa7c407b9115e6b4c7fc68838dbbcaf288450df345'),
             'blockNumber': 1917600
            })
        """
        token_pair = self.state_manager.update_token_pair(
            pair_address=event.address,
            block_num=event.blockNumber,
            reserve0=event.args.reserve0,
            reserve1=event.args.reserve1
        )

        logging.debug(f'Received token pair sync event for block {event.blockNumber}: updated {token_pair}')

    def handle_token_pair_swap_event(self, event):
        amount0_delta = event.args.amount0In - event.args.amount0Out
        amount1_delta = event.args.amount1In - event.args.amount1Out
        txn_hash = event.transactionHash.hex()
        logging.debug(f'Received swap txn {txn_hash} with amount0_delta={amount0_delta}, amount1_delta={amount1_delta}')
        if txn_hash in self.state_manager.our_pending_txns:
            # The transaction must have succeeded for it to have generated logs
            self.state_manager.remove_pending_txn(txn_hash)
            order_id = self.order_sender.pop_order_id_for_txn(txn_hash)
            trade = Trade(order_id=order_id, txn_hash=txn_hash, is_success=True, amount0_delta=amount0_delta, amount1_delta=amount1_delta)
            self.state_manager.add_new_trade(trade)
            logging.info(f'Received token pair swap event for our order: block {event.blockNumber}, pair {event.address}')

    def create_trades_for_unsent_orders(self, orders: List[Order]) -> List[Trade]:
        '''
        This is needed to inform the strategy that its desired order failed
        i.e. to trigger the strategy's on_our_trades callback
        '''
        if self.simple_simulate_fills and all([len(o.path) == 2 for o in orders]):
            trades = []
            for o in orders:
                pair = self.state_manager.token_addrs_to_pair[(o.path[0], o.path[1])]
                if pair.token0_address == o.path[0] and pair.token1_address == o.path[1]:
                    amount0_delta = o.amount_in
                    amount1_delta = -o.amount_out_min
                elif pair.token0_address == o.path[1] and pair.token1_address == o.path[0]:
                    amount0_delta = -o.amount_out_min
                    amount1_delta = o.amount_in
                else:
                    raise ValueError(f'Order = {o}, Token addrs to pair was initialized incorrectly: pair={pair}')
                trades.append(Trade(
                    order_id=o.order_id,
                    txn_hash='0xunsent_order',
                    is_success=True,
                    amount0_delta=amount0_delta,
                    amount1_delta=amount1_delta,
                ))
            logging.info(f'Simulated (fake!) filled trades: {trades}')
        else:
            trades = [Trade(
                order_id=o.order_id,
                txn_hash='0xunsent_order',
                is_success=False,
                amount0_delta=0,
                amount1_delta=0,
            ) for o in orders]
        return trades


def token_pair_log_loop(trader: LiveTrader, poll_interval: int, event_generators: List[Callable]=[], send_is_blocking=False):
    filters = trader.create_filters()
    
    while True:
        # There is a chance that the sleep ends when some but not all of the events from a new block
        # have arrived. We used to wait an additional loop to pull zero events for this purpose,
        # but have since gotten rid of it because we pull new data sources now.
        for filter in filters:
            filter.handle_new_entries(trader.state_manager.cur_block_num)

        our_new_trades = trader.state_manager.our_new_trades
        if len(our_new_trades) > 0:
            trader.strategy.on_our_trades(trader.state_manager, our_new_trades)
            trader.state_manager.clear_new_trades()

        orders = []

        events = []
        for gen in event_generators:
            events += gen()
        for event in events:
            orders += trader.strategy.on_event(trader.state_manager, event)
        
        orders += trader.strategy.on_state_update(trader.state_manager)
        if len(orders) > 0:
            logging.warning(f'Generated orders: {orders}')
            
            if send_is_blocking:
                did_send_orders, trades = trader.order_sender.send_orders_blocking(orders)
                if not did_send_orders:
                    trades = trader.create_trades_for_unsent_orders(orders)
                trader.strategy.on_our_trades(trader.state_manager, trades)
            else:
                did_send_orders, pending_txns = trader.order_sender.send_orders_nonblocking(orders)
                if did_send_orders:
                    trader.state_manager.add_pending_txns(pending_txns)
                else:
                    trades = trader.create_trades_for_unsent_orders(orders)
                    trader.strategy.on_our_trades(trader.state_manager, trades)
        
        sleep(poll_interval)


def cyclic_arbitrage_main(rpc_provider):
    from live_trader.stellaswap_cyclic_arbitrage_strategy import StellaswapCyclicArbitrageStrategy
    config = load_config()

    trader = LiveTrader(config, rpc_provider=rpc_provider, rpc_type='ws')
    strategy = StellaswapCyclicArbitrageStrategy(
        config['strategy']['cyclic_arbitrage'],
        config['order_sender']['wallet_address'],
        [trader.state_manager.address_to_token[token_address] for token_address in trader.cfg['token_trading_universe']],
    )
    trader.set_strategy(strategy)

    token_pair_log_loop(trader, 1)
    
def binance_alpha_main(rpc_provider: str, token_pair_name: str):
    from binance.spot import Spot
    from live_trader.stellaswap_binance_alpha_strategy import StellaswapBinanceAlphaTokenPairStrategy

    class BinancePriceEventGenerator:
        def __init__(self, client: Spot, binance_symbol: str, pull_rate_secs: float):
            self.client = client
            self.binance_symbol = binance_symbol
            self.pull_rate_secs = pull_rate_secs
            self.last_event_generated_time = 0

        def __call__(self):
            return self.get_binance_price_event()

        def get_binance_price_event(self) -> List[Dict[str, Any]]:
            cur_time = datetime.datetime.now().timestamp()
            if cur_time - self.last_event_generated_time >= self.pull_rate_secs:
                self.last_event_generated_time = cur_time
                trade = client.trades(symbol=self.binance_symbol, limit=1)[0]
                return [{
                    'binance_symbol': self.binance_symbol,
                    'timestamp': trade['time'],
                    'binance_price': float(trade['price']),
                }]
            return []

    config = load_config()
    strategy_config = config['strategy']['binance_alpha'][token_pair_name]
    logging.info(f'Strategy config: {strategy_config}')
    
    binance_price_event_generators = []
    client = Spot()
    for x in ['numerator', 'denominator']:
        if strategy_config['binance_quote_calculation'][x]['type'] == 'binance':
            symbol = strategy_config['binance_quote_calculation'][x]['symbol']
            pull_rate_secs = strategy_config['binance_quote_calculation'][x]['price_data_pull_rate_seconds']
            binance_price_event_generators.append(BinancePriceEventGenerator(client, symbol, pull_rate_secs))
    
    trader = LiveTrader(config, rpc_provider=rpc_provider, rpc_type='ws')
    strategy = StellaswapBinanceAlphaTokenPairStrategy(
        strategy_config,
        trader.state_manager,
        config['order_sender']['wallet_address'],
    )
    trader.set_strategy(strategy)

    token_pair_log_loop(trader, poll_interval=1, event_generators=binance_price_event_generators)


if __name__ == '__main__':
    rpc_provider = sys.argv[1]
    token_pair_name = sys.argv[2]
    binance_alpha_main(rpc_provider, token_pair_name)

    # cyclic_arbitrage_main(rpc_provider)
