import datetime
from glob import glob
import os
import pandas as pd


# There will be some missing (potentially duplicate if step_length_secs > 0.5 * time between blocks) data
# at the start and end of consecutive processed binance files. This should be insignificant though.
def generate_binance_file(binance_token_pair_directory, stellaswap_filepath, step_length_secs=3):
    _, token_pair = os.path.split(binance_token_pair_directory) # e.g. eth_usdt
    binance_raw_input_dir = os.path.join(binance_token_pair_directory, 'raw')
    _, stellaswap_filename = os.path.split(stellaswap_filepath)
    out_filepath = os.path.join(
        binance_token_pair_directory,
        'processed',
        stellaswap_filename.replace('stellaswap', 'binance')
    )
    
    stellaswap_df = pd.read_feather(stellaswap_filepath)
    min_timestamp, max_timestamp = stellaswap_df['block_timestamp'].min(), stellaswap_df['block_timestamp'].max()
    start = datetime.datetime.utcfromtimestamp(min_timestamp - step_length_secs).date()
    end = datetime.datetime.utcfromtimestamp(max_timestamp + step_length_secs).date()
    dates = [start + datetime.timedelta(days=x) for x in range((end - start).days + 1)]
    binance_files = [os.path.join(binance_raw_input_dir, 
        f'{token_pair.replace("_", "").upper()}-aggTrades-{date.strftime("%Y-%m-%d")}.zip'
    ) for date in dates]
    
    header = ['agg_trade_id', 'price', 'quantity', 'first_trade_id', 'last_trade_id', 'timestamp', 'is_buyer_maker', 'is_trade_best_price_match']
    binance_df = pd.concat([pd.read_csv(f, names=header, compression='zip') for f in binance_files])
    timestamp_filter = binance_df['timestamp'].between(min_timestamp * 1e3, max_timestamp * 1e3)
    binance_df = binance_df[timestamp_filter]
    binance_df['bucket'] = (binance_df.timestamp / (step_length_secs * 1e3)).astype(int)
    binance_df = binance_df.groupby('bucket').last()
    
    binance_df[['agg_trade_id', 'timestamp', 'price', 'quantity', 'is_buyer_maker']].reset_index(drop=True).astype({
        'agg_trade_id': 'uint64',
        'timestamp': 'uint64',
        'price': 'float64',
        'quantity': 'float64',
        'is_buyer_maker': 'bool',
    }).to_feather(out_filepath)
    print(f'Outputted file to {out_filepath}')


if __name__ == '__main__':
    binance_token_pair_directory = 'data/binance_history/avax_usdt'
    stellaswap_filepaths = sorted(glob(f'data/stellaswap_txn_history/all/stellaswap_data_1[6-8]*.feather'))
    for stellaswap_filepath in stellaswap_filepaths:
        generate_binance_file(binance_token_pair_directory, stellaswap_filepath)
