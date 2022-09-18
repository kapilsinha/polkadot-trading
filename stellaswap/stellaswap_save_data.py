from common.enums import DataRowType
from common.helpers import load_config, save_stellaswap_data_to_feather
from stellaswap.stellaswap_token import StellaswapTokenContainer, StellaswapTokenPairContainer
from smart_order_router.graph import Graph

import web3 as w3

from collections import defaultdict
import logging
from multiprocessing import Pool
import pandas as pd
import sys
from timer import timer
from typing import Any, Dict, List


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)
timer.set_level(logging.DEBUG)


class StellaswapSaveDataDriver:
    
    cur_block_num: int

    # every (200 * b)th block, we take a full snapshot of all token pairs and tokens
    full_snapshot_block_interval = 200
    address_to_token_pair: Dict[str, StellaswapTokenPairContainer] = {}

    # Think of this as a graph of TokenPairContainer const references
    token_graph: Graph

    address_to_token: Dict[str, StellaswapTokenContainer] = {}
    
    # pending_pairs = {start block num : [token pair addresses]} for all token pairs that don't exist yet at the cur block
    # In process_next_block, we will add token pairs as they are created to our state variables. This avoids nasty
    # exceptions with calling contract functions at a block when the contract doesn't yet exist.
    pending_pairs = defaultdict(list)

    def __init__(self, config, block_num_start, rpc_provider, rpc_type = 'ws'):
        if rpc_type == 'ws':
            self.web3 = w3.Web3(w3.Web3.WebsocketProvider(config['rpc_endpoint']['ws'][rpc_provider]))
        elif rpc_type == 'http':
            self.web3 = w3.Web3(w3.Web3.HTTPProvider(config['rpc_endpoint']['http'][rpc_provider]))
        else:
            raise ValueError('Invalid rpc_type: must be "ws" or "http"')

        self.cur_block_num = block_num_start
        self.cfg = config['venue']['stellaswap']
                
        liquid_pairs_df = pd.read_csv(self.cfg['data_path']['liquid_pairs'])
        for pair_address, block_num_created in zip(liquid_pairs_df.pair_address, liquid_pairs_df.block_num_created):
            start_block_num = max(block_num_created, self.new_token_pair_block_num_threshold())
            self.pending_pairs[start_block_num].append(pair_address)
    
    @timer
    def process_next_block(self) -> List[Dict[str, Any]]:
        block_num = self.cur_block_num
        self.add_token_pairs_if_newly_created()

        block_info = self.web3.eth.get_block(block_num)
        block_timestamp = block_info.timestamp
        txn_hex_hashes = block_info.transactions
        res = []
        for txn_hex_hash in txn_hex_hashes:
            res += self.process_txn(block_num, block_timestamp, txn_hex_hash.hex())
        
        shared_fields = {
            'block_number': block_num,
            'block_timestamp': block_timestamp,
        }

        # Snapshot all token pairs every full_snapshot_block_interval blocks; otherwise just snapshot the updated ones
        # Snapshot all tokens if ANY token pairs are getting snapshotted
        # i.e. for a full snapshot or if any token pairs were updated
        snapshot_token_pair_addresses = sorted(
            self.address_to_token_pair.keys()) if block_num % self.full_snapshot_block_interval == 0 else sorted(
                set([x['pair_address'] for x in res if x['row_type'] == DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT]))

        for pair_address in snapshot_token_pair_addresses:
            token_pair = self.address_to_token_pair[pair_address]
            r0, r1 = token_pair.reserve0, token_pair.reserve1
            token_pair.take_snapshot(block_num)
            # This is an important sanity check! take_snapshot (querying for the reserves) should not update our state
            # because in theory, we have all the updates by parsing the Sync event logs. If these do mismatch, that
            # means we missed a Sync event (perhaps our filtering logic is wrong)
            assert(r0 == token_pair.reserve0)
            assert(r1 == token_pair.reserve1)
            res.append({
                'row_type': DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT,
                **token_pair.dump(),
                **shared_fields,
            })
        if len(snapshot_token_pair_addresses) > 0:
            res += self.update_and_dump_all_tokens(DataRowType.END_OF_BLOCK_TOKEN_SNAPSHOT, shared_fields)
        
        self.cur_block_num += 1
        return res

    # @timer
    def process_txn(self, block_num, block_timestamp, txn_hash) -> List[Dict[str, Any]]:
        receipt = self.web3.eth.get_transaction_receipt(txn_hash)
        res = []
        if receipt.status == 1:
            # Only process the transaction if the transaction succeeded (status == 0 is a failed transaction)
            # Do NOT filter by contract address here because 
            # 1. there are more routers than what they list on their documentation.
            # 2. 3rd party smart contracts can interface with the  underlying token pair contracts
            #    (e.g. 0x035792e8171197df27acedd499dd539a9bc3effa)
            # so we need to parse through every transaction's logs
            txn_index = receipt.transactionIndex
            for log in receipt.logs:
                res += self.process_log(block_num, block_timestamp, txn_hash, txn_index, log)
        return res

    # @timer
    def process_log(self, block_num, block_timestamp, txn_hash, txn_index, log) -> List[Dict[str, Any]]:
        token_pair = self.address_to_token_pair.get(log.address)
        res = []
        if token_pair is not None:
            log_index = log.logIndex
            event_signature = log.topics[0].hex()
            decoded_log = token_pair.event_signature_to_event[event_signature].processLog(log)
            shared_fields = {
                'block_number': block_num,
                'block_timestamp': block_timestamp,
                'txn_hash': txn_hash,
                'txn_index': txn_index,
                'log_index': log_index,
            }

            if decoded_log.event == 'Sync':
                args = decoded_log.args

                # Update and record token pair state
                token_pair.update(block_num=block_num, reserve0=args.reserve0, reserve1=args.reserve1)
                res.append({
                    'row_type': DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT,
                    **token_pair.dump(),
                    **shared_fields,
                })
                res += self.update_and_dump_all_tokens(DataRowType.ON_UPDATE_TOKEN_SNAPSHOT, shared_fields)
            
            elif decoded_log.event == 'Swap':
                args = decoded_log.args
                # The below assertion is true 99.99% of the time but I do see exceptions in custom 3rd party contracts
                # assert(
                #     (args.amount0In >  0 and args.amount1In == 0 and args.amount0Out == 0 and args.amount1Out >  0) or
                #     (args.amount0In == 0 and args.amount1In >  0 and args.amount0Out >  0 and args.amount1Out == 0)
                # )
                amount0_delta = args.amount0In - args.amount0Out
                amount1_delta = args.amount1In - args.amount1Out
                res.append({
                    'row_type': DataRowType.SWAP_TXN,
                    'pair_address': token_pair.pair_address,
                    'amount0_delta': amount0_delta,
                    'amount1_delta': amount1_delta,
                    **shared_fields,
                })

        return res

    def add_token_pairs_if_newly_created(self):
        newly_created_pairs = self.pending_pairs[self.new_token_pair_block_num_threshold()]
        if len(newly_created_pairs) > 0:
            # Add just the new token pairs to our state because old ones are unaffected
            self.address_to_token_pair.update({
                pair_address: StellaswapTokenPairContainer(
                    pair_address, self.cfg['abi_path']['pair'], self.web3, self.cur_block_num
                    ) for pair_address in newly_created_pairs
            })

            # Update the entire graph and all the tokens because all tokens can in theory
            # have new values with the addition of a single token pair
            self.token_graph = Graph(self.address_to_token_pair.values())

            self.address_to_token = {
                token_address: StellaswapTokenContainer(
                    token_address, self.cfg['abi_path']['token'], self.web3, self.token_graph
                    ) for token_address in self.token_graph.get_tokens()
            }
            del self.pending_pairs[self.new_token_pair_block_num_threshold()]
            logging.warning(f'Added token pairs {newly_created_pairs} to our state at block {self.cur_block_num}! '
                            f'This is expected behavior when the driver starts and when the token pair was created in this block.')
            logging.warning(f'Remaining token pairs not yet created (creation block number -> token pair address): '
                            f'{ {k: v for k, v in self.pending_pairs.items() if len(v) > 0} }')
    
    def new_token_pair_block_num_threshold(self):
        # This is a confusing name, but we consider a token pair to be 'newly created' if it was created on this block number
        # We do a -1 offset in order to avoid potential issues with the contract not existing at the start of the block
        return self.cur_block_num - 1
    
    # @timer
    def update_and_dump_all_tokens(self, row_type, extra_fields) -> List[Dict[str, Any]]:
        res = []
        for _, token in sorted(self.address_to_token.items()):
            token.update(self.token_graph)
            res.append({
                'row_type': row_type,
                **token.dump(),
                **extra_fields,
            })
        return res


def process_block_range(args):
    rpc_provider = args['rpc_provider']
    block_num_start = args['block_num_start']
    block_num_end = args['block_num_end']
    config = args['config']
    logging.info(f'Started processing blocks {block_num_start} to {block_num_end}...')
    driver = StellaswapSaveDataDriver(config, block_num_start=block_num_start, rpc_provider=rpc_provider, rpc_type='ws')
    
    res = []
    block_range = range(block_num_start, block_num_end + 1)
    for block in block_range:
        if block % 1000 == 0:
            logging.info(f'Processing block {block}')
        res += driver.process_next_block()
    
    # These equiv cols must match the fields in StellaswapTokenContainer.dump(...)
    equiv_cols = sorted([f'{col}_equiv_no_fees' for col in StellaswapTokenContainer.equivalent_names] \
                      + [f'{col}_equiv_with_fees' for col in StellaswapTokenContainer.equivalent_names])
    # Force the below columns to appear (even if there are no swaps, for example)
    df = pd.DataFrame.from_records(res, columns = [
        'row_type',
        'block_number',
        'block_timestamp',
        'reserve0',
        'reserve1',
        'txn_index',
        'log_index',
        'pair_address',
        'txn_hash',
        'token_address',
        'amount0_delta',
        'amount1_delta',
    ] + equiv_cols)
    df.to_csv(f'out/stellaswap_data_{block_num_start}_{block_num_end}.csv', index=False)
    save_stellaswap_data_to_feather(df, f'out/stellaswap_data_{block_num_start}_{block_num_end}.feather')
    
    logging.info(f'Finished processing blocks {block_num_start} to {block_num_end}')
    return len(df)


def main(rpc_provider, block_num_start, block_num_end, block_chunk_size=10_000, num_processes=4, should_parallelize=False):
    def chunk_range(start, end, chunk_size):
        while start <= end:
            next_start = start + chunk_size
            yield start, min(next_start - 1, end)
            start = next_start

    config = load_config()
    args = [{
        'config': config, 'rpc_provider': rpc_provider, 'block_num_start': start, 'block_num_end': end
        } for start, end in chunk_range(block_num_start, block_num_end, block_chunk_size)]

    if should_parallelize:
        raise ValueError('Unsupported! The Web3Py library is not thread-safe apparently. '
                         'The multiprocessing library does some weird stuff with spawning threads. '
                         'We see HTTP and Websockets exceptions when using a pool. '
                         'Just spawn multiple programs if you need to do parallel processing')
        with Pool(processes=num_processes) as pool:
            pool.map(process_block_range, args)
    else:
        for arg in args:
            process_block_range(arg)
    

if __name__ == '__main__':
    rpc_provider = sys.argv[1]
    block_start = int(sys.argv[2])
    block_end = int(sys.argv[3])
    main(rpc_provider, block_start, block_end)

