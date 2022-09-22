from live_trader.trade import Trade
from stellaswap.stellaswap_token import StellaswapTokenContainer, StellaswapTokenPairContainer
from smart_order_router.graph import Graph

import logging
import pandas as pd
from typing import Dict, List, Optional, Set


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

'''
We don't bother creating getters, but any state should be modified through
a function instead of modifying a variable directly
'''
class StateManager:
    
    _order_id = 1

    def __init__(self, stellaswap_cfg, web3):
        block_info = web3.eth.get_block('latest')
        self.cur_block_num = block_info.number
        self.last_timestamp = block_info.timestamp
        logging.info(f'Initializing state (currently at block {self.cur_block_num})...')
        
        pair_addresses = pd.read_csv(stellaswap_cfg['data_path']['liquid_pairs']).pair_address.tolist()
        self.address_to_token_pair: Dict[str, StellaswapTokenPairContainer] = {
            pair_address: StellaswapTokenPairContainer(
                pair_address, stellaswap_cfg['abi_path']['pair'], web3, self.cur_block_num
                ) for pair_address in pair_addresses
        }
        self.token_addrs_to_pair = {
            **{(pair.token0_address, pair.token1_address): pair for pair in self.address_to_token_pair.values()},
            **{(pair.token1_address, pair.token0_address): pair for pair in self.address_to_token_pair.values()},
        }
        # Think of this as a graph of TokenPairContainer const references
        self.token_graph = Graph(self.address_to_token_pair.values())

        self.address_to_token: Dict[str, StellaswapTokenContainer] = {
            token_address: StellaswapTokenContainer(
                token_address, stellaswap_cfg['abi_path']['token'], web3, self.token_graph
                ) for token_address in self.token_graph.get_tokens()
        }
        
        self.our_pending_txns: Set[str] = set() # txn hashes that the strategy has sent but have not been mined
        self.our_new_trades: List[Trade] = []

    @classmethod
    def generate_order_id(cls):
        order_id = cls._order_id
        cls._order_id += 1
        return order_id

    def set_cur_block_num(self, block_num):
        self.cur_block_num = block_num

    def set_last_timestamp(self, timestamp):
        self.last_timestamp = timestamp

    def update_token_pair(self, pair_address: str, block_num: int, reserve0: int, reserve1: int) -> StellaswapTokenPairContainer:
        self.address_to_token_pair[pair_address].update(
            block_num=block_num,
            reserve0=reserve0,
            reserve1=reserve1
        )
        return self.address_to_token_pair[pair_address]
    
    def add_pending_txns(self, txns: List[str]):
        self.our_pending_txns = self.our_pending_txns.union(txns)

    def remove_pending_txn(self, txn):
        self.our_pending_txns.remove(txn)
        
    def add_new_trade(self, new_trade: Trade):
        self.our_new_trades.append(new_trade)
    
    def clear_new_trades(self):
        self.our_new_trades.clear()
