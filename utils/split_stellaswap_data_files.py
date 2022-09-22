import os
import glob
import pandas as pd


# This script splits the StellaSwap data files by row type
def split_file(filename):
    d, f = os.path.split(filename)
    df = pd.read_feather(filename)
    x = df[df.row_type == 1]
    x.reset_index(drop=True).astype({
        'row_type': 'uint8',
        'block_number': 'uint64', # int because block_number and block_timestamp are never NaN
        'block_timestamp': 'uint64',
        'reserve0': 'float64',
        'reserve1': 'float64',
        'pair_address': 'object',
    }).to_feather(os.path.join(d, '../end_of_block_token_pair', f))

    y = df[df.row_type == 3]
    y.reset_index(drop=True).astype({
        'row_type': 'uint8',
        'block_number': 'uint64', # int because block_number and block_timestamp are never NaN
        'block_timestamp': 'uint64',
        'token_address': 'object',
        'amount0_delta': 'float64',
        'amount1_delta': 'float64',
        **{equiv_col: 'float64' for equiv_col in df.columns if 'equiv' in equiv_col}
    }).to_feather(os.path.join(d, '../end_of_block_token', f))


if __name__ == '__main__':
    filenames = glob.glob('data/stellaswap_txn_history/all/stellaswap_data_18[5-9]*.feather')
    for f in filenames:
        split_file(f)
