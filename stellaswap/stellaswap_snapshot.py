from common.helpers import load_config, load_contract, retry_call_wrapper

from web3 import Web3

from functools import lru_cache
import logging
import pandas as pd
from timer import timer


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)
timer.set_level(logging.INFO)

class StellaswapSnapshotContainer:
    @timer
    def __init__(self, config, rpc_type='ws'):
        self.cfg = config['venue']['stellaswap']
        if rpc_type == 'ws':
            self.web3 = Web3(Web3.WebsocketProvider(config['rpc_endpoint']['ws']['blast']))
        elif rpc_type == 'http':
            self.web3 = Web3(Web3.HTTPProvider(config['rpc_endpoint']['http']['blast']))
        else:
            raise ValueError('Invalid rpc_type: must be "ws" or "http"')
        self.router = load_contract(
            self.cfg['contract_address']['router'],
            self.cfg['abi_path']['router'],
            self.web3,
        )
        self.factory = load_contract(
            self.cfg['contract_address']['factory'],
            self.cfg['abi_path']['factory'],
            self.web3,
        )
        self.block_identifier = self.web3.eth.block_number
        num_pairs = self.call(self.factory.functions.allPairsLength())
        # We can parallelize the below with asyncio, threading, or multiprocessing - but we likely will only
        # trade the most liquid pairs so I won't bother for now. It's also tricky with public RPC endpoints because
        # they get throttled
        self.pairs = [self._get_pair_metadata_by_index(i) for i in range(num_pairs)]
        logging.info(f"Snapshot is taken at block {self.block_identifier}")

    def call(self, f):
        return retry_call_wrapper(f, self.block_identifier, num_retries=5)
        
    def dump_metadata_df(self):
        df = pd.DataFrame(self.pairs)
        for t in ['token0', 'token1']:
            df[f'{t}_address'] = df[t].apply(lambda col: col['address'])
            df[f'{t}_symbol'] = df[t].apply(lambda col: col['symbol'])
            df[f'{t}_decimals'] = df[t].apply(lambda col: col['decimals'])
        sorted_cols = ['symbol', 'rate', 'pair_address', 'last_block_timestamp', 'reserve0', 'reserve1',
                       'token0_address', 'token0_symbol', 'token0_decimals',
                       'token1_address', 'token1_symbol', 'token1_decimals']
        return df[sorted_cols].reset_index()

    def _get_pair_metadata_by_index(self, pair_index):
        pair_address = self.call(self.factory.functions.allPairs(pair_index))
        return self._get_pair_metadata_by_address(pair_address)
    
    @timer
    def _get_pair_metadata_by_address(self, pair_address):
        """
        This metadata is NOT expected to remain static for all time.
        Namely, the reserve quantities will evolve.
        """
        pair = load_contract(
            pair_address,
            self.cfg['abi_path']['pair'],
            self.web3,
        )
        token0_address = self.call(pair.functions.token0())
        token0 = self._get_token_metadata(token0_address)
        token1_address = self.call(pair.functions.token1())
        token1 = self._get_token_metadata(token1_address)
        reserve0, reserve1, last_block_timestamp = self.call(pair.functions.getReserves())
        rate = self.quote_no_fees(reserve0, reserve1)
        return {
            'symbol': f'{token0["symbol"]}/{token1["symbol"]}', # we construct this symbol for our convenience
            'pair_address': pair_address,
            'token0': token0,
            'token1': token1,
            'reserve0': reserve0,
            'reserve1': reserve1,
            'rate': rate,
            'last_block_timestamp': last_block_timestamp,
        }

    @lru_cache(maxsize=1000)
    @timer
    def _get_token_metadata(self, token_address):
        """
        This metadata is expected to remain static for all time
        """
        token = load_contract(
            token_address,
            self.cfg['abi_path']['token'],
            self.web3,
        )
        return {
            'address':  token_address,
            'symbol':   self.call(token.functions.symbol()),
            'name':     self.call(token.functions.name()),
            'decimals': self.call(token.functions.decimals()),
        }
    
    @staticmethod
    def quote_no_fees(reserve0, reserve1, amount_in=1):
        """
        self.router.functions.quote(amount0, reserve0, reserve1).call() is a higher-level call.
        It performs integer division and so would require mucking with decimals, which I'd rather avoid for now.
        rate = amount of token1 that has the same value as amount_in token0s. Does not include fees!
        Source: https://github.com/stellaswap/core/blob/70e9b4be6fa861855b20ed1bb6b2a993d179bf44/amm/libraries/StellaSwapV2Library.sol#L38
        """
        rate = amount_in * reserve1 / reserve0 if reserve0 > 0 and reserve1 > 0 else None
        return rate
    
    @staticmethod
    def quote_with_fees(reserve0, reserve1, amount_in=1):
        """
        self.router.functions.getAmountOut(amountIn, reserveIn, reserveOut) is a higher-level call.
        It performs integer division and so would require mucking with decimals, which I'd rather avoid for now.
        amount_out = amount of token1 that you can actually receive (after fees) for amount_in token0s.
        Source: https://github.com/stellaswap/core/blob/70e9b4be6fa861855b20ed1bb6b2a993d179bf44/amm/libraries/StellaSwapV2Library.sol#L45
        """
        amount_out = (amount_in * .9975 * reserve1) / (reserve0 + amount_in * .9975)
        return amount_out


if __name__ == '__main__':
    config = load_config()
    stellaswap = StellaswapSnapshotContainer(config, rpc_type='http')
    df = stellaswap.dump_metadata_df()
    df.to_csv('out/stellaswap_metadata_snapshot.csv', index=False)
