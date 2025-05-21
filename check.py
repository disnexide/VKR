import pandas as pd

anom = pd.read_csv('anomalies.csv')

metrics = {
    'mean_amount':               anom['amount'].mean(),
    'median_amount':             anom['amount'].median(),
    'std_amount':                anom['amount'].std(),
    'min_amount':                anom['amount'].min(),
    'max_amount':                anom['amount'].max(),
    'mean_time_diff':            anom['time_diff'].mean(),
    'mean_days_since_last_txn':  anom['days_since_last_transaction'].mean(),
    'fraction_weekend':          anom['is_weekend'].mean(),
    'fraction_night':            anom['is_night'].mean(),
    'mean_transaction_count_by_client': anom['transaction_count_by_client'].mean()
}

metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['value'])
metrics_df.index.name = 'metric'
print(metrics_df)
