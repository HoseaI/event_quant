
import pandas as pd


def cal_rank_factor(df: pd.DataFrame, extra_cols: list):
    df['rank_factor1'] = df['总市值']
    extra_cols.append('rank_factor1')
    return df, extra_cols
