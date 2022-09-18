import asyncio
import logging
import requests
from time import sleep
import web3
import yaml

def load_config(filename='cfg.yaml'):
    with open(filename, 'r') as f:
        config = yaml.safe_load(f)
    return config

def load_contract(contract_address, abi_path, web3):
    with open(abi_path, 'r') as a:
        abi = a.read()
    return web3.eth.contract(address=contract_address, abi=abi)

"""
Note that any of the Web3.py .call() function calls may fail if the network
connection is lost or due to timeouts. For now we use crude retry logic.
Will need to think of more appropriate exception handling behavior later
"""
def retry_call_wrapper(f, block_identifier, num_retries):
    while num_retries >= 0:
        try:
            return f.call(block_identifier=block_identifier)
        except web3.exceptions.BadFunctionCallOutput as e:
            logging.warn(f'Web3 bad function call output ({e})! This likely means the contract did not exist at block {block_identifier}. Re-raising this as a ValueError')
            raise ValueError('Bad function call {e} at block {block_identifier}. Contract likely did not exist')
            
        except web3.exceptions.TimeExhausted as e:
            logging.error(f'Web3 time exhausted({e})! {num_retries} retries remaining...')
        except web3.exceptions.CannotHandleRequest as e:
            logging.error(f'Web3 cannot handle request ({e})! {num_retries} retries remaining...')
        except web3.exceptions.TooManyRequests as e:
            logging.error(f'Web3 too many requests ({e})! {num_retries} retries remaining...')

        except requests.exceptions.Timeout as e:
            logging.error(f'Request timed out ({e})! {num_retries} retries remaining...')
        except requests.exceptions.RequestException as e:
            logging.error(f'Generic request error ({e})! {num_retries} retries remaining...')

        except asyncio.exceptions.TimeoutError as e:
            logging.error(f'Asyncio timeout error ({e})! {num_retries} retries remaining...')
        except asyncio.exceptions.CancelledError as e:
            logging.error(f'Asyncio cancelled error ({e})! {num_retries} retries remaining...')
        except Exception as e:
            logging.error(f'Unknown error (catch-all handler) ({e})! {num_retries} retries remaining...')
        sleep(1)
        num_retries -= 1
    raise SystemError("Ran out of retries")

"""
We cannot store uint256 as an int in feather format (C has no equivalent) so we sacrifice negligible
precision and store it as a float
"""
def save_stellaswap_data_to_feather(df, filename):
    # We need to reset index because feather requires a default index (which an empty DataFrame does not have)
    # ValueError: feather does not support serializing <class 'pandas.core.indexes.base.Index'> for the index;
    # you can .reset_index() to make the index into column(s)
    df.reset_index(drop=True).astype({
        'row_type': 'uint8',
        'block_number': 'uint64', # int because block_number and block_timestamp are never NaN
        'block_timestamp': 'uint64',
        'reserve0': 'float64',
        'reserve1': 'float64',
        'txn_index': 'float64', # cast as floats because NaNs cannot be represented as ints
        'log_index': 'float64',
        'pair_address': 'object',
        'txn_hash': 'object',
        'token_address': 'object',
        'amount0_delta': 'float64',
        'amount1_delta': 'float64',
        **{equiv_col: 'float64' for equiv_col in df.columns if 'equiv' in equiv_col}
    }).to_feather(filename)
