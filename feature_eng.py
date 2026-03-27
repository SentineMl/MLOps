import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def process_features(df):
    """
    Processes the raw transactions dataset and returns a feature-engineered dataframe.
    """
    df = df.copy()

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp']).reset_index(drop=True)

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['weekend_transaction'] = df['timestamp'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)
    df['night_transaction'] = df['hour'].apply(lambda x: 1 if (0 <= x <= 5) else 0)

    # Standardize 'amount' with z-score then log transform
    scaler = StandardScaler()
    amount_scaled = scaler.fit_transform(df[['amount']])
    # Handle negative scaled values before log by taking the absolute value (or shift)
    # Using np.abs to safely satisfy "log(amount +1)" after standardization 
    df['amount_scaled'] = np.log1p(np.abs(amount_scaled))


    # Distance from home
    # If missing, fill with median or 0
    if 'distance_from_home' in df.columns:
         df['distance_from_home'] = df['distance_from_home'].fillna(df['distance_from_home'].median())
    
    # If city_size is present
    if 'city_size' in df.columns:
        df['city_size'] = df['city_size'].astype('category').cat.codes
    
    # Drop device_fingerprint to avoid high-cardinality memory crash
    if 'device_fingerprint' in df.columns:
        df = df.drop(columns=['device_fingerprint'])

    # One-hot encoding on lower cardinality alternatives
    dummy_cols = ['currency', 'country', 'device']
    cols_to_encode = [c for c in dummy_cols if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, dummy_na=False)

    # Encode remaining categorical specifically for city_size if it exists
    if 'city_size' in df.columns and (df['city_size'].dtype == 'object' or df['city_size'].dtype.name == 'category'):
        df['city_size'] = df['city_size'].astype('category').cat.codes

    # Keep exactly the explicitly requested features + the target label 'is_fraud'
    expected_base_cols = [
        'is_fraud',
        'amount_scaled', 
        'hour', 
        'weekend_transaction', 
        'night_transaction', 
        'distance_from_home', 
        'city_size'
    ]
    
    # Identify all the dummy columns that were dynamically generated
    dummy_generated_cols = [c for c in df.columns if c.startswith(('currency_', 'country_', 'device_'))]
    
    # Combine the lists and ensure we only keep columns that actually exist in the dataframe
    final_cols_to_keep = expected_base_cols + dummy_generated_cols
    final_cols_to_keep = [c for c in final_cols_to_keep if c in df.columns]
    
    features_df = df[final_cols_to_keep].copy()
    
    return features_df

if __name__ == "__main__":
    # Test execution
    print("Feature engineering functions loaded correctly.")
