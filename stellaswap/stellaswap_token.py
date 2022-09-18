from common.helpers import load_contract, retry_call_wrapper
from common.enums import Direction
from smart_order_router.graph import Graph
from smart_order_router.sor import single_sor_no_fees, single_sor_with_fees

from dataclasses import dataclass
import json
import logging
from timer import timer
from typing import Dict, Optional
import web3 as w3


class StellaswapTokenPairContainer:
    @timer
    def __init__(self, pair_address, token_pair_abi_path, web3, block_num):
        """
        Note that the call function calls can raise exceptions. We will just pass
        that exception up the stack
        """
        self.pair_address = pair_address
        self.contract = load_contract(
            pair_address,
            token_pair_abi_path,
            web3,
        )
        self.token0_address = self.call(self.contract.functions.token0(), block_num)
        self.token1_address = self.call(self.contract.functions.token1(), block_num)
        # An event log's first topic is the event signature,
        # a keccak256 hash of the name of the event plus types of its parameters
        self.event_signature_to_event : Dict[str, w3.contract.ContractEvent] \
            = self._get_topic_to_event(token_pair_abi_path, web3)
        
        # Initialize state with a snapshot
        self.take_snapshot(block_num)
        
    def _get_topic_to_event(self, token_pair_abi_path, web3):
        with open(token_pair_abi_path, 'r') as a:
            abi_json = json.load(a)
        abi_events = [x for x in abi_json if x['type'] == 'event']
        topic_to_event = {}
        for event in abi_events:
            name = event['name']
            inputs = ','.join([param['type'] for param in event['inputs']])
            topic_hex = web3.toHex(web3.keccak(text=f'{name}({inputs})'))
            topic_to_event[topic_hex] = self.contract.events[name]()
        return topic_to_event

    # @timer  
    def take_snapshot(self, block_num):
        self.last_processed_block_num = block_num
        self.reserve0, self.reserve1, _ = self.call(self.contract.functions.getReserves(), block_num)
        
    def call(self, f, block_identifier):
        return retry_call_wrapper(f, block_identifier, num_retries=5)

    def update(self, block_num, reserve0, reserve1):
        self.last_processed_block_num = block_num
        self.reserve0 = reserve0
        self.reserve1 = reserve1

    def quote_no_fees(self, direction, amount_in=1):
        r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
        rate = amount_in * r1 / r0 if r0 > 0 and r1 > 0 else None
        return rate
    
    def quote_with_fees(self, direction, amount_in=1):
        r0, r1 = (self.reserve0, self.reserve1) if direction == Direction.FORWARD else (self.reserve1, self.reserve0)
        amount_out = (amount_in * .9975 * r1) / (r0 + amount_in * .9975)
        return amount_out
    
    def dump(self):
        return {
            'pair_address': self.pair_address,
            'reserve0': self.reserve0,
            'reserve1': self.reserve1,
        }

    def __repr__(self):
        return '[ ' + ', '.join([f'{k}={v}' for k, v in self.dump().items()]) + ' ]'


@dataclass
class StellaswapTokenEquivalent:
    address: str
    value_no_fees: Optional[float] = None
    value_with_fees: Optional[float] = None


class StellaswapTokenContainer:
    
    equivalent_names = ['xcdot', 'glmr', 'frax', 'dai-multi', 'usdc-multi']

    def __init__(self, token_address, token_abi_path, web3, token_graph: Optional[Graph] = None):
        self.token_address = token_address

        self.equivalents: Dict[str, StellaswapTokenEquivalent] = {
            'xcdot': StellaswapTokenEquivalent('0xFfFFfFff1FcaCBd218EDc0EbA20Fc2308C778080'),
            'glmr': StellaswapTokenEquivalent('0xAcc15dC74880C9944775448304B263D191c6077F'),
            'frax': StellaswapTokenEquivalent('0x322E86852e492a7Ee17f28a78c663da38FB33bfb'),
            'dai-multi': StellaswapTokenEquivalent('0x765277EebeCA2e31912C9946eAe1021199B39C61'),
            'usdc-multi': StellaswapTokenEquivalent('0x818ec0A7Fe18Ff94269904fCED6AE3DaE6d6dC0b'),
        }

        self.contract = load_contract(
            token_address,
            token_abi_path,
            web3,
        )
        self.decimals = self.contract.functions.decimals().call()
        if token_graph is not None:
            self.update(token_graph)
    
    # @timer
    def update(self, token_graph: Graph):
        for equiv in self.equivalents.values():
            # Note we rely on equiv being passed by reference because it is not a primitive
            equiv.value_no_fees, _ = single_sor_no_fees(token_graph, self.token_address, equiv.address)
            equiv.value_with_fees, _ = single_sor_with_fees(token_graph, self.token_address, equiv.address)
    
    def get_usd_value(self):
        return self.equivalents['dai-multi'].value_no_fees * 1e-18

    def get_wallet_balance(self, wallet_address):
        return self.contract.functions.balanceOf(wallet_address).call()

    def decimals(self):
        return self.decimals
    
    def dump(self):
        return {
            'token_address': self.token_address,
            **{f'{name}_equiv_no_fees': equiv.value_no_fees for name, equiv in self.equivalents.items()},
            **{f'{name}_equiv_with_fees': equiv.value_with_fees for name, equiv in self.equivalents.items()},
        }

    def __repr__(self):
        return '[ ' + ', '.join([f'{k}={v}' for k, v in self.dump().items()]) + ' ]'
