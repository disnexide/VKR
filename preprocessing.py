import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(data):
    # Make a copy to avoid modifying original DataFrame
    df = data.copy()

    # Fill missing categorical values explicitly
    if 'zip' in df.columns:
        mode_zip = df['zip'].mode()[0]
        df['zip'] = df['zip'].fillna(mode_zip)
    if 'merchant_city' in df.columns:
        df['merchant_city'] = df['merchant_city'].fillna('unknown')
    if 'merchant_state' in df.columns:
        df['merchant_state'] = df['merchant_state'].fillna('unknown')

    # Fill remaining missing values
    df = df.fillna(0)

    # Encode binary and small categorical features
    le = LabelEncoder()
    if 'use_chip' in df.columns:
        df['use_chip'] = le.fit_transform(df['use_chip'].astype(str))
    if 'merchant_state' in df.columns:
        df['merchant_state'] = le.fit_transform(df['merchant_state'].astype(str))

    # Select numeric columns for scaling
    columns_to_scale = [
        'amount',
        'transaction_count_by_client',
        'avg_transaction_by_card',
        'unique_merchants_by_client',
        'same_city_transaction_ratio'
    ]
    # Keep only existing columns
    columns_to_scale = [c for c in columns_to_scale if c in df.columns]

    # Scale selected features
    scaler = StandardScaler()
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df, columns_to_scale