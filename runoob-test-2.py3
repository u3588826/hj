#!/usr/bin/python
import pandas as pd
import datetime
import numpy as np

data = pd.read_parquet('hw2_signal.parquet', engine='pyarrow')
msf = pd.read_parquet('crsp.msf.parquet', engine='pyarrow')[['permno', 'hexcd', 'prc', 'shrout', 'date']]
msf['prc'] = msf['prc'].abs()
msf['shrout'] *= msf['prc']
msf = msf.rename(columns = {'permno': 'focal_permno', 'shrout': 'mkt_cap', 'date': 'date_of_return'})
data = data.merge(msf, on = ['focal_permno', 'date_of_return'])
dse = pd.read_parquet('crsp.dsenames.parquet', engine='pyarrow')[['permno', 'shrcd']]
dse = dse.rename(columns = {'permno': 'focal_permno'})
data = data.merge(dse.drop_duplicates(), on = 'focal_permno')
data["date_of_return"] = data["date_of_return"].apply(lambda g: datetime.datetime.strptime(g, "%Y-%m-%d"))
data.index = data["date_of_return"]

def apply_quantiles(x, include_in_quantiles=None, bins=10):
    if include_in_quantiles is None:
        include_in_quantiles = [True] * len(x)
    x = pd.Series(x)
    quantiles = np.quantile(
        x[x.notnull() & include_in_quantiles],
        np.linspace(0, 1, bins+1)
    )
    quantiles[0] = x.dropna().min() - 1
    quantiles[-1] = x.dropna().max() + 1
    return pd.cut(x, quantiles, labels=False, duplicates='drop') + 1

lags = 1
data = data.groupby('focal_permno', group_keys=0).apply(lambda g: g.sort_index().drop(['prc', 'mkt_cap'], axis=1).join(g.sort_index()[['prc', 'mkt_cap']].shift(lags).rename(columns={'prc':'prc_lag', 'mkt_cap':'mkt_cap_lag'}))).reset_index(drop=1).dropna()
tradeable_data = data[data['hexcd'].isin([1, 2, 3])]
tradeable_data = tradeable_data[tradeable_data['shrcd'].isin([10, 11])]
tradeable_data = tradeable_data[tradeable_data["prc_lag"] > 5]
tradeable_data = tradeable_data.groupby('date_of_return').apply(lambda g: g[g.mkt_cap_lag > (g['mkt_cap_lag'].max() - g['mkt_cap_lag'].min()) * 0.2 + g['mkt_cap_lag'].min()]).reset_index(drop=1)
tradeable_data = tradeable_data.sort_values(["focal_permno","date_of_return"])
tradeable_data.loc[:, 'TECHRET'] = tradeable_data.groupby("focal_permno", group_keys=0).apply(lambda g: g["ret_tech_peers"].shift(lags)).reset_index(drop=1).values
tradeable_data = tradeable_data.dropna()
tradeable_data['num_per_day'] = tradeable_data.groupby(["date_of_return"])['ret_tech_peers'].transform('count')
tradeable_data = tradeable_data[tradeable_data['num_per_day'] > 1]
tradeable_data = tradeable_data.drop('num_per_day', axis=1)
tradeable_data['bin'] = (
    tradeable_data
    .groupby('date_of_return')
    .apply(lambda g: apply_quantiles(g['TECHRET'], bins=10))
    .reset_index([0], drop=True)
    .sort_index()
)
portfolios = (
    tradeable_data[
        tradeable_data['bin'].notnull() &
        tradeable_data['ret_tech_peers'].notnull() &
        tradeable_data['mkt_cap_lag'].notnull()
    ]
    .groupby(['bin', 'date_of_return'])[['ret_tech_peers', 'mkt_cap_lag']]
    .apply(
        lambda g: pd.Series({
            'ew': g['ret_tech_peers'].mean(),
            'vw': (g['ret_tech_peers'] * g['mkt_cap_lag']).sum() / g['mkt_cap_lag'].sum()
        })
    )
    .reset_index()
)
portfolios.groupby('bin').agg(weighted_return=('vw', 'mean')).plot(kind='bar')
portfolios.groupby('bin').agg(equal_weight_return=('ew', 'mean')).plot(kind='bar')
portfolio_LS = pd.merge(
    portfolios.query('bin==10'),
    portfolios.query('bin==1'),
    suffixes=['_long', '_short'],
    on='date_of_return'
)
portfolio_LS['strategy_vw'] = (portfolio_LS['vw_long'] - portfolio_LS['vw_short'])
portfolio_LS['strategy_vw'] = (portfolio_LS['vw_long'] - portfolio_LS['vw_short'])

portfolio_LS['cum_vw'] = (portfolio_LS['strategy_vw'] + 1).cumprod() - 1 
portfolio_LS['strategy_ew'] = (portfolio_LS['ew_long'] - portfolio_LS['ew_short'])
portfolio_LS['cum_ew'] = (portfolio_LS['strategy_ew'] + 1).cumprod() - 1 
(
     portfolio_LS
    .assign(date=pd.to_datetime(portfolio_LS['date_of_return']))
    .assign(cum_vw=portfolio_LS['cum_vw']+1)
    .assign(cum_ew=portfolio_LS['cum_ew']+1)
    .plot(x='date', y=['cum_ew', 'cum_vw'], logy=True).grid(axis='y')
)

import statsmodels.formula.api as smf
import statsmodels.api as sm

benchmark = pd.read_parquet('four_factor_monthly.parquet')
benchmark_merged = pd.merge(
    (
        benchmark
        .assign(date=pd.to_datetime(benchmark['dt']))
        .assign(yearmonth=lambda df: df['date'].dt.year * 12 + df['date'].dt.month)
    ),
    (
        portfolio_LS
        .assign(date=pd.to_datetime(portfolio_LS['date_of_return']))
        .assign(yearmonth=lambda df: df['date'].dt.year * 12 + df['date'].dt.month)
    ),
    on='yearmonth'
)
l1 = smf.ols('cum_vw ~ 1 + smb + hml + mkt_rf + mom', data=benchmark_merged).fit()
l2 = smf.ols('cum_ew ~ 1 + smb + hml + mkt_rf + mom', data=benchmark_merged).fit()
sm.iolib.summary2.summary_col([l1, l2], stars=True)

