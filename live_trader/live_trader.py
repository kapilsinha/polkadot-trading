from common.helpers import load_config, load_contract
from live_trader.order import Order
from live_trader.trade import Trade
from live_trader.order_sender import OrderSender
from live_trader.filter_container import BlockFilterContainer, ContractEventFilterContainer, FilterContainer
from stellaswap.stellaswap_token import StellaswapTokenContainer, StellaswapTokenPairContainer
from smart_order_router.graph import Graph

import logging
import pandas as pd
import pdb
import sys
from time import sleep
from timer import timer
from typing import Dict, List, Set
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
        block_info = self.web3.eth.get_block('latest')
        self.cur_block_num = block_info.number
        self.last_timestamp = block_info.timestamp
        logging.info(f'Initializing trader bot (currently at block {self.cur_block_num})...')
                
        pair_addresses = pd.read_csv(self.cfg['data_path']['liquid_pairs']).pair_address.tolist()
        self.address_to_token_pair: Dict[str, StellaswapTokenPairContainer] = {
            pair_address: StellaswapTokenPairContainer(
                pair_address, self.cfg['abi_path']['pair'], self.web3, self.cur_block_num
                ) for pair_address in pair_addresses
        }
        # Think of this as a graph of TokenPairContainer const references
        self.token_graph = Graph(self.address_to_token_pair.values())

        self.address_to_token: Dict[str, StellaswapTokenContainer] = {
            token_address: StellaswapTokenContainer(
                token_address, self.cfg['abi_path']['token'], self.web3, self.token_graph
                ) for token_address in self.token_graph.get_tokens()
        }
        
        router = load_contract(
            self.cfg['contract_address']['router'],
            self.cfg['abi_path']['router'],
            self.web3,
        )
        self.order_sender = OrderSender(
            router,
            self.web3,
            self.address_to_token['0xAcc15dC74880C9944775448304B263D191c6077F'],
            config['order_sender']
        )

        self.strategy = None
        self.our_pending_txns: Set[str] = set() # txn hashes that the strategy has sent but have not been mined
        self.our_new_trades: List[Trade] = []
        self.create_filters()
        logging.info('Finished initializing trader bot')

    def set_strategy(self, strategy):
        self.strategy = strategy

    def add_pending_txns(self, txns: List[str]):
        self.our_pending_txns = self.our_pending_txns.union(txns)

    def create_filters(self) -> List[FilterContainer]:
        block_filter = BlockFilterContainer(self.web3, self.cur_block_num, self.handle_new_block)
        sync_filters = [
            ContractEventFilterContainer(token_pair.contract.events.Sync, self.cur_block_num, self.handle_token_pair_sync_event) \
                for token_pair in self.address_to_token_pair.values()
        ]
        # swap_filters = [token_pair.contract.events.Swap.createFilter(fromBlock=trader.cur_block_num, toBlock='latest') \
        #                  for token_pair in trader.address_to_token_pair.values()]
        # TODO: Figure out why the tx filter doesn't work even for our LAN node
        # tx_filter = w3.eth.filter('pending')
        return [
            block_filter,
            *sync_filters,
        ]

    def handle_new_block(self, block_hash):
        num_retries = 5
        while num_retries > 0:
            try:
                block_info = self.web3.eth.get_block(block_hash)
                logging.info(f'Processing block number {block_info.number}')
                txns = [txn.hex() for txn in block_info.transactions]
                for txn_hash in txns:
                    if txn_hash in self.our_pending_txns:
                        # Status == 0 -> failed txn, status == 1 -> successful txn
                        is_success = self.web3.eth.get_transaction_receipt(txn_hash).status == 1
                        self.our_pending_txns.remove(txn_hash)
                        order_id = self.order_sender.pop_order_id_for_txn(txn_hash)
                        self.our_new_trades.append(Trade(order_id=order_id, txn_hash=txn_hash, is_success=is_success))
                
                self.cur_block_num = block_info.number
                self.last_timestamp = block_info.timestamp
                return
            except TypeError as e:
                if len(e.args) > 0 and 'unrecognized block reference' in e.args[0]:
                    logging.error(f'Block handler received "unknown block" with error {e}. '
                                  f'Sleeping for 1 second and will retry ({num_retries} retries left).')
                else:
                    raise e
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
        token_pair_address = event.address

        self.address_to_token_pair[token_pair_address].update(
            block_num=event.blockNumber,
            reserve0=event.args.reserve0,
            reserve1=event.args.reserve1
        )
        logging.info(f'Received token pair sync event for block {event.blockNumber}: updated {self.address_to_token_pair[token_pair_address]}')

    def handle_token_pair_swap_event(self, event):
        logging.info(f'Received token pair swap event: {event}')
        raise NotImplementedError('We don\'t handle swap events because our strategies don\'t make use of them')


def token_pair_log_loop(trader: LiveTrader, poll_interval: int, send_is_blocking=False):
    filters = trader.create_filters()
    
    while True:
        # There is a chance that the sleep ends when some but not all of the events from a new block
        # have arrived. We used to wait an additional loop to pull zero events for this purpose,
        # but have since gotten rid of it because we pull new data sources now.
        for filter in filters:
            filter.handle_new_entries(trader.cur_block_num)

        if len(trader.our_new_trades) > 0:
            trader.strategy.on_our_trades(trader.our_new_trades)
            trader.our_new_trades.clear()
        
        orders = trader.strategy.on_state_update(trader.token_graph, trader.last_timestamp)
        if len(orders) > 0:
            logging.warning(f'Generated orders: {orders}')
            
            if send_is_blocking:
                trades = trader.order_sender.send_orders_blocking(orders)
                trader.strategy.on_our_trades(trades)
            else:
                pending_txns = trader.order_sender.send_orders_nonblocking(orders)
                trader.add_pending_txns(pending_txns)
        
        sleep(poll_interval)


"""
I had started this as an asyncio coroutine but realized that's not needed because
we want to always get the full state and then trigger orders instead of triggering
off of incremental updates
"""
def main(rpc_provider):
    from live_trader.stellaswap_cyclic_arbitrage_strategy import StellaswapCyclicArbitrageStrategy
    config = load_config()

    # Unclean initialization, can fix later
    trader = LiveTrader(config, rpc_provider=rpc_provider, rpc_type='ws')
    strategy = StellaswapCyclicArbitrageStrategy(
        config['strategy']['cyclic_arbitrage'],
        config['order_sender']['wallet_address'],
        [trader.address_to_token[token_address] for token_address in trader.cfg['token_trading_universe']],
    )
    trader.set_strategy(strategy)

    token_pair_log_loop(trader, 1)
    

if __name__ == '__main__':
    rpc_provider = sys.argv[1]
    #asyncio.run(main(rpc_provider))
    main(rpc_provider)
