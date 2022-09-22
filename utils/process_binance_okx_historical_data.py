import argparse
import datetime
from glob import glob
from lib2to3.pgen2 import token
import os
import pandas as pd


"""
There will be some missing (potentially duplicate if step_length_secs > 0.5 * time between blocks) data
at the start and end of consecutive processed binance files. This should be insignificant though.
"""

def generate_binance_file(token_pair_directory, stellaswap_filepath):
    def create_merged_df(*, raw_input_dir, token_pair, dates):
        binance_files = [os.path.join(raw_input_dir, 
            f'{token_pair.replace("_", "").upper()}-aggTrades-{date.strftime("%Y-%m-%d")}.zip'
        ) for date in dates]
        header = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_trade_best_price_match']
        return pd.concat([pd.read_csv(f, names=header, compression='zip') for f in binance_files])
    
    return _generate_cex_file(token_pair_directory, stellaswap_filepath, 'binance', create_merged_df)

def generate_okx_file(token_pair_directory, stellaswap_filepath):
    def create_merged_df(*, raw_input_dir, token_pair, dates):
        okx_files = [os.path.join(raw_input_dir, 
            f'{token_pair.replace("_", "-").upper()}-aggTrades-{date.strftime("%Y-%m-%d")}.zip'
        ) for date in dates]

        dfs = []
        for f in okx_files:
            df = pd.read_csv(f, compression='zip', skiprows=1, encoding_errors='backslashreplace', 
                names=['agg_trade_id', 'is_buyer_maker', 'quantity', 'price', 'timestamp'])
            df['is_buyer_maker'] = df['is_buyer_maker'].map({'buy': True, 'sell': False})
            dfs.append(df)
        return pd.concat(dfs)
    
    return _generate_cex_file(token_pair_directory, stellaswap_filepath, 'okx', create_merged_df)

def _generate_cex_file(token_pair_directory, stellaswap_filepath, venue, create_merged_df_func, step_length_secs=3):
    _, token_pair = os.path.split(token_pair_directory) # e.g. eth_usdt
    raw_input_dir = os.path.join(token_pair_directory, 'raw')
    _, stellaswap_filename = os.path.split(stellaswap_filepath)
    out_filepath = os.path.join(
        token_pair_directory,
        'processed',
        stellaswap_filename.replace('stellaswap', venue)
    )
    if os.path.isfile(out_filepath):
        print(f'{out_filepath} already exists. Skipping it.')
        return
    
    stellaswap_df = pd.read_feather(stellaswap_filepath)
    min_timestamp, max_timestamp = stellaswap_df['block_timestamp'].min(), stellaswap_df['block_timestamp'].max()
    start = datetime.datetime.utcfromtimestamp(min_timestamp - step_length_secs).date()
    end = datetime.datetime.utcfromtimestamp(max_timestamp + step_length_secs).date()
    dates = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]

    cex_df = create_merged_df_func(raw_input_dir=raw_input_dir, token_pair=token_pair, dates=dates)
        
    timestamp_filter = cex_df['timestamp'].between(min_timestamp * 1e3, max_timestamp * 1e3)
    cex_df = cex_df[timestamp_filter]
    cex_df['bucket'] = (cex_df.timestamp / (step_length_secs * 1e3)).astype(int)
    cex_df = cex_df.groupby('bucket').last()
    
    cex_df[['agg_trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker']].reset_index(drop=True).astype({
        'agg_trade_id': 'uint64',
        'timestamp': 'uint64',
        'price': 'float64',
        'quantity': 'float64',
        'is_buyer_maker': 'bool',
    }).to_feather(out_filepath)
    print(f'Outputted file to {out_filepath}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process raw CEX files into feather files that map to stellaswap data files.')
    parser.add_argument('--venue', choices=['binance', 'okx'], required=True)
    parser.add_argument('--symbol', type=str, required=True)
    args = parser.parse_args()

    if args.venue == 'binance':
        subdir = 'binance_history'
        file_generator = generate_binance_file
    elif args.venue == 'okx':
        subdir = 'okx_history'
        file_generator = generate_okx_file
    else:
        raise ValueError('Invalid venue')

    token_pair_directory = f'data/{args.venue}_history/{args.symbol}'
    if not os.path.isdir(token_pair_directory):
        raise ValueError(f'{token_pair_directory} is not a directory')
    os.makedirs(os.path.join(token_pair_directory, 'processed'), exist_ok=True)
    stellaswap_filepaths = sorted(glob(f'data/stellaswap_txn_history/all/stellaswap_data_1[6-8]*.feather'))
    for stellaswap_filepath in stellaswap_filepaths:
        file_generator(token_pair_directory, stellaswap_filepath)
