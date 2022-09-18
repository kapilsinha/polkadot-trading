from abc import ABC, abstractmethod
import asyncio
import logging
import re
import web3 as w3
import traceback
from typing import Any, Callable, List


logging.basicConfig(
    level=logging.INFO,
    format= '[%(asctime)s.%(msecs)03d] %(levelname)s:%(name)s %(message)s | %(pathname)s:%(lineno)d',
    datefmt='%Y%m%d,%H:%M:%S'
)

'''
This file exists primarily because filters seem to expire randomly even though they're supposed to
remain active if you keep polling. It also encapuslates the handler with the filter, which is nice
'''

NEXT_FILTER_ID = 1

'''
Abstract class to be extended (see bottom of file), not instantiated directly
'''
class FilterContainer(ABC):
    def __init__(self, block_num_start: int, handler: Callable):
        global NEXT_FILTER_ID # needed to increment it

        self.filter = self.create_filter(block_num_start)
        self.handler = handler
        self.expired_filter_regex = re.compile('Filter id \d+ does not exist.')
        self.filter_id = NEXT_FILTER_ID
        NEXT_FILTER_ID += 1

    @abstractmethod
    def create_filter(self, last_block_num) -> w3._utils.filters.Filter:
        raise NotImplementedError('Abstract method: subtypes should return a subclass of type w3._utils.filters.Filter')
    
    def handle_new_entries(self, last_block_num) -> bool:
        '''
        Returns True if we received (and handled) new entries, else False.
        '''
        new_entries = self._get_new_entries(last_block_num)
        for entry in new_entries:
            self.handler(entry)
        return len(new_entries) > 0

    def _get_new_entries(self, last_block_num) -> List[Any]:
        try:
            new_entries = self.filter.get_new_entries()
        except ValueError as e:
            # catches errors of the type ValueError({'code': -32603, 'message': 'Filter id 25 does not exist.'})
            if len(e.args) > 0 and self.expired_filter_regex.match(e.args[0].get('message', '')) is not None:
                logging.warning(f'Filter (id {self.filter_id}) expired, so we are creating a new one')
                self.filter = self.create_filter(last_block_num)
                new_entries = self.filter.get_new_entries()
            else:
                # Otherwise it's an unknown exception, and raise it again (crash the program)
                raise e
        except TypeError as e:
            # I have seen this error get thrown - I think due to a bug in the Web3Py library's decoding logic
            # Not sure how to handle it so we just try again
            if len(e.args) > 0 and e.args[0] == 'byte indices must be integers or slices, not str':
                logging.error(f'Filter (id {self.filter_id}) received error {e}: {traceback.print_exc()}. We will try creating a new filter anyway')
                self.filter = self.create_filter(last_block_num)
                new_entries = self.filter.get_new_entries()
            else:
                raise e
        except asyncio.exceptions.TimeoutError as e:
            # I have seen this error periodically. Same handling as above
            self.filter = self.create_filter(last_block_num)
            new_entries = self.filter.get_new_entries()
        return new_entries


'''
Manages a 'block filter', which looks for new block hashes
'''
class BlockFilterContainer(FilterContainer):
    def __init__(self, web3: w3.Web3, block_num_start, handler: Callable):
        # self.web3 must be created before calling super().__init__ because it is needed in create_filter
        self.web3 = web3
        super().__init__(block_num_start, handler)
    
    def create_filter(self, last_block_num) -> w3._utils.filters.BlockFilter:
        '''
        NOTE: There is apaprently no support to get the block infos from a last_block_num, so we
        don't use that argument. This can be dangerous if we somehow wait for an entire block to
        pass since our last_block_num - but should be very unlikely
        '''
        return self.web3.eth.filter('latest')


'''
Manages a contract event / log filter
'''
class ContractEventFilterContainer(FilterContainer):
    def __init__(self, event: w3.contract.ContractEvent, block_num_start: int, handler: Callable):
        # self.event must be created before calling super().__init__ because it is needed in create_filter
        self.event = event
        super().__init__(block_num_start, handler)
    
    def create_filter(self, last_block_num) -> w3._utils.filters.LogFilter:
        return self.event.createFilter(fromBlock=last_block_num, toBlock='latest')
