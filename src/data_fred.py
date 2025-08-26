import pandas as pd
from fredapi import Fred
import os
import sys

# Get API key from environment variable
API_KEY = os.getenv('FRED_API_KEY')

# Define the 8 currencies we will be using
CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'SEK']

# FRED Series IDs identified in the previous step
# Using 10-Year yields as a proxy for IR, as discussed
IR_SERIES_IDS = {
    'USD': 'DGS10',
    'EUR': 'IRLTLT01EZM156N',
    'JPY': 'IRLTLT01JPM156N',
    'GBP': 'IRLTLT01GBM156N',
    'AUD': 'IRLTLT01AUM156N',
    'CAD': 'IRLTLT01CAM156N',
    'CHF': 'IRLTLT01CHM156N',
    'SEK': 'IRLTLT01SEM156N',
}

# FX rates are vs USD. Some are USD per FX, others are FX per USD.
# We will need to standardize them later.
FX_SERIES_IDS = {
    'EUR': 'DEXUSEU', # USD per EUR
    'JPY': 'DEXJPUS', # JPY per USD
    'GBP': 'DEXUSUK', # USD per GBP
    'AUD': 'DEXUSAL', # USD per AUD
    'CAD': 'DEXCAUS', # CAD per USD
    'CHF': 'DEXSZUS', # CHF per USD
    'SEK': 'DEXSDUS', # SEK per USD
}

def download_fred_data(start_date='2010-01-01', end_date='2024-12-31', output_dir='data'):
    """
    Downloads specified FX and IR series from FRED, combines them,
    and saves to a CSV file.
    """
    if not API_KEY:
        print("ERROR: FRED_API_KEY environment variable not set.")
        print("Please set the environment variable with your FRED API key.")
        sys.exit(1)

    print("Initializing FRED API...")
    fred = Fred(api_key=API_KEY)

    all_series_data = {}

    # 1. Download Interest Rate data
    print("Downloading interest rate series...")
    for currency, series_id in IR_SERIES_IDS.items():
        print(f"  Fetching {currency} IR: {series_id}")
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        all_series_data[f'IR_{currency}'] = series

    # 2. Download FX data
    print("\nDownloading FX rate series...")
    for currency, series_id in FX_SERIES_IDS.items():
        print(f"  Fetching {currency} FX: {series_id}")
        series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        all_series_data[f'FX_VS_USD_{currency}'] = series

    # 3. Combine into a single DataFrame
    print("\nCombining and processing data...")
    df = pd.DataFrame(all_series_data)

    # Reindex to a full daily index to ensure we can ffill correctly
    full_date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
    df = df.reindex(full_date_range)

    # 4. Handle missing values
    # The paper suggests forward-filling. This is especially important for the monthly IR data.
    df.ffill(inplace=True)

    # Drop any remaining NaNs (e.g., at the very beginning of the series)
    df.dropna(inplace=True)

    # Filter to only business days, similar to the synthetic data generator
    df = df[df.index.dayofweek < 5]

    print(f"Processed data has {len(df)} rows, from {df.index.min().date()} to {df.index.max().date()}.")

    # 5. Save to CSV
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, 'raw_fred_data.csv')
    df.to_csv(output_path)

    print(f"\nRaw FRED data downloaded and saved to {output_path}")
    return df

def process_raw_data(raw_df, output_dir='data'):
    """
    Processes the raw downloaded FRED data into a model-ready format.
    - Standardizes all FX rates into a full C x C matrix.
    - Renames columns to match model's expected format.
    """
    print("\nProcessing raw data into model-ready format...")

    # Standardize FX rates
    # The raw data is all vs USD. We need to create a full matrix.
    # FX_VS_USD_EUR = USD per EUR -> invert to get EUR per USD
    # FX_VS_USD_JPY = JPY per USD

    fx_data = pd.DataFrame(index=raw_df.index)

    # First, get all rates in terms of how many units of that currency buy 1 USD
    per_usd = {}

    # Handle USD per FX (e.g., DEXUSEU) -> invert it
    for curr in ['EUR', 'GBP', 'AUD']:
        per_usd[curr] = 1.0 / raw_df[f'FX_VS_USD_{curr}']

    # Handle FX per USD (e.g., DEXJPUS) -> use as is
    for curr in ['JPY', 'CAD', 'CHF', 'SEK']:
        per_usd[curr] = raw_df[f'FX_VS_USD_{curr}']

    # USD is the base
    per_usd['USD'] = 1.0

    # Now, calculate all C x C cross rates
    for curr1 in CURRENCIES:
        for curr2 in CURRENCIES:
            if curr1 == curr2:
                continue
            # Rate is how much of curr2 for one unit of curr1
            # (units of curr2 / USD) / (units of curr1 / USD)
            fx_data[f'FX_{curr1}_{curr2}'] = per_usd[curr2] / per_usd[curr1]

    # Combine with IR data
    ir_data = raw_df[[f'IR_{c}' for c in CURRENCIES]]

    model_ready_df = pd.concat([fx_data, ir_data], axis=1)

    # Save to new CSV
    output_path = os.path.join(output_dir, 'model_ready_data.csv')
    model_ready_df.to_csv(output_path)

    print(f"Model-ready data created and saved to {output_path}")
    return model_ready_df


if __name__ == '__main__':
    raw_df = download_fred_data()
    if raw_df is not None and not raw_df.empty:
        process_raw_data(raw_df)
