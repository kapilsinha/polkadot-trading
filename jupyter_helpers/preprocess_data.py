from common.enums import DataRowType

import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import statsmodels.formula.api as smf

def compute_deltas(df):
    token_pair_group = df.groupby(['row_type', 'pair_address'])
    df['rate_bps_delta'] = token_pair_group['rate'].apply(pd.Series.pct_change) * 10_000
    df['revrate_bps_delta'] = token_pair_group['revrate'].apply(pd.Series.pct_change) * 10_000
    token_group = df.groupby(['row_type', 'token_address'])
    for equiv_col in [x for x in df.columns if 'equiv' in x]:
        df[f'{equiv_col}_bps_delta'] = token_group[equiv_col].apply(pd.Series.pct_change) * 10_000
    return df

def compute_deltas_token_pair(df):
    token_pair_group = df.groupby(['row_type', 'pair_address'])
    df['rate_bps_delta'] = token_pair_group['rate'].apply(pd.Series.pct_change) * 10_000
    df['revrate_bps_delta'] = token_pair_group['revrate'].apply(pd.Series.pct_change) * 10_000
    token_group = df.groupby(['row_type', 'token_address'])
    return df

def compute_deltas_token(df):
    token_group = df.groupby(['row_type', 'token_address'])
    for equiv_col in [x for x in df.columns if 'equiv' in x]:
        df[f'{equiv_col}_bps_delta'] = token_group[equiv_col].apply(pd.Series.pct_change) * 10_000
    return df

def augment_swap_rows(df, pair_to_tokens):
    # Can't figure out how to do this with vectorized operations so we'll just loop. Hence this is ridiculously slow
    df['amountIn_usd_no_fee'] = np.nan
    df['amountOut_usd_no_fee'] = np.nan
    df['amountOut_usd_with_fee'] = np.nan # this is approximate (an overestimate) because the equiv_dai_with_fee assumes amountIn = 1 and the fees increase with amountIn

    token_to_usd_no_fees = {}
    token_to_usd_with_fees = {}
    for index, row in df.iterrows():
        if row['row_type'] == DataRowType.ON_UPDATE_TOKEN_SNAPSHOT:
            token_to_usd_no_fees[row['token_address']] = row['dai-multi_equiv_no_fees'] * 1e-18
            token_to_usd_with_fees[row['token_address']] = row['dai-multi_equiv_with_fees'] * 1e-18
        elif row['row_type'] == DataRowType.SWAP_TXN:
            token0, token1 = pair_to_tokens[row['pair_address']]
            in_token, in_token_abs_amount, out_token, out_token_amount = (token0, -row['amount0_delta'], token1, row['amount1_delta']) \
                                        if row['amount0_delta'] < 0 else (token1, -row['amount1_delta'], token0, row['amount0_delta'])
            df.at[index, 'amountIn_usd_no_fee'] = in_token_abs_amount * token_to_usd_no_fees[in_token]
            df.at[index, 'amountOut_usd_no_fee'] = out_token_amount    * token_to_usd_no_fees[out_token]
            df.at[index, 'amountOut_usd_with_fee'] = out_token_amount    * token_to_usd_with_fees[out_token]

    # Actually it does not make sense to compute profit this way because the current value of the in-token may be amounIn_usd_no_fee
    # but that is NOT how much we bought it for. Moreover, because we compute the USD equiv by traversing token pairs, the value of
    # the in-amount should be greater than the value of the out-amount (due to fees). I see 1 in 10,000 rows that have sizable paper
    # profit measured this way. Also it makes more sense to aggregate the swaps in a given transaction when measuring 'actual profit'
    # otherwise we unfairly discount each step along the path
    df['paper_profit'] = df['amountOut_usd_no_fee'] - df['amountIn_usd_no_fee']
    df['theoretical_actual_profit'] = df['amountOut_usd_with_fee'] - df['amountIn_usd_no_fee']
    return df

def add_exp_smooth_token_pair(df, smoothing_level=0.25):
    row_types = list(df.row_type.unique())
    assert(len(row_types) == 1)
    if row_types[0] not in [DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT, DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT]:
        s = pd.Series(np.nan, index=df.index)
        t = pd.Series(np.nan, index=df.index)
    else:
        rate_fit = SimpleExpSmoothing(df['rate'].reset_index(drop=True), initialization_method='estimated').fit(smoothing_level=smoothing_level)
        s = rate_fit.fittedvalues.set_axis(df.index)
        revrate_fit = SimpleExpSmoothing(df['revrate'].reset_index(drop=True), initialization_method='estimated').fit(smoothing_level=smoothing_level)
        t = revrate_fit.fittedvalues.set_axis(df.index)
    df['smoothed_rate'] = s
    df['smoothed_revrate'] = t
    return df

def add_detrended_rate(df):
    row_types = list(df.row_type.unique())
    assert(len(row_types) == 1)
    pair_addresses = list(df.pair_address.unique())
    assert(len(pair_addresses) == 1)
    
    if row_types[0] not in [DataRowType.END_OF_BLOCK_TOKEN_PAIR_SNAPSHOT, DataRowType.ON_UPDATE_TOKEN_PAIR_SNAPSHOT]:
        s = pd.Series(np.nan, index=df.index)
        t = pd.Series(np.nan, index=df.index)
    else:
        data = pd.DataFrame({'x': df.block_timestamp - df.block_timestamp.min(), 'rate': df.rate})
        mod = smf.ols(formula='rate ~ x', data=data)
        res = mod.fit()
        s = res.fittedvalues
        t = data.rate - res.fittedvalues
    df['predicted_rate'] = s
    df['detrended_rate'] = t
    return df

def add_exp_smooth_token(df, smoothing_level=5e-3):
    row_types = list(df.row_type.unique())
    assert(len(row_types) == 1)
    if row_types[0] not in [DataRowType.END_OF_BLOCK_TOKEN_SNAPSHOT, DataRowType.ON_UPDATE_TOKEN_SNAPSHOT]:
        t = pd.Series(np.nan, index=df.index)
    else:
        fit = SimpleExpSmoothing(df['dai-multi_equiv_no_fees'].reset_index(drop=True), initialization_method='estimated').fit(smoothing_level=smoothing_level)
        t = fit.fittedvalues.set_axis(df.index) # dumb fucking name to change the index
    df['smoothed_dai-multi_equiv_no_fees'] = t
    return df
