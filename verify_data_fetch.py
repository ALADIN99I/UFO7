import pandas as pd
import requests
import sys
import yfinance as yf
import numpy as np
from datetime import datetime

# --- Constants and Config ---
FRED_API_KEY = '8f503a1e7348fa967987e5ad187992b9'
BOND_SERIES = {
    'USD': {'1Y': 'DGS1', '2Y': 'DGS2', '5Y': 'DGS5', '10Y': 'DGS10'},
    'EUR': {'10Y': 'IRLTLT01EZM156N'}, 'JPY': {'10Y': 'IRLTLT01JPM156N'},
    'GBP': {'10Y': 'IRLTLT01GBM156N'}, 'AUD': {'10Y': 'IRLTLT01AUM156N'},
    'CAD': {'10Y': 'IRLTLT01CAM156N'}, 'CHF': {'10Y': 'IRLTLT01CHM156N'},
    'HKD': {'10Y': 'IRLTLT01HKM156N'}, 'SGD': {'10Y': 'IRLTLT01SGM156N'},
    'SEK': {'10Y': 'IRLTLT01SEM156N'}
}
CURRENCIES = ['EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'HKD', 'SGD', 'SEK']
FX_TICKERS = [f'{currency}USD=X' for currency in CURRENCIES] + ['USDJPY=X']


# --- Function Definitions ---
def fetch_fred_data(series_id, api_key, start_date='1995-01-01', end_date='2024-12-31'):
    """Fetches data from FRED API with error handling."""
    url = f'https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json&observation_start={start_date}&observation_end={end_date}'
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if 'observations' not in data:
            print(f"API Error for {series_id}: 'observations' key not in response.", file=sys.stderr)
            print(f"Response: {data}", file=sys.stderr)
            return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"HTTP Request failed for {series_id}: {e}", file=sys.stderr)
        return pd.DataFrame()
    except ValueError:
        print(f"Failed to decode JSON for {series_id}.", file=sys.stderr)
        print(f"Response text: {response.text}", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(data['observations'])
    if df.empty:
        return df
    df = df[['date', 'value']]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df['value'] = pd.to_numeric(df['value'], errors='coerce')
    return df

def fetch_fx_data(tickers, start_date='1995-01-01', end_date='2024-12-31'):
    """Fetches FX data from Yahoo Finance."""
    with pd.option_context('future.no_silent_downcasting', True):
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    return data

# --- Main Execution Block ---
def main():
    """Main function to run the data fetching process."""
    print("--- Starting Data Fetching Verification ---")

    bond_data = {}
    print("\n--- Fetching Bond Yields ---")
    for currency, series in BOND_SERIES.items():
        bond_data[currency] = {}
        for term, series_id in series.items():
            print(f'Fetching {currency} {term} ({series_id})...')
            df = fetch_fred_data(series_id, FRED_API_KEY)
            bond_data[currency][term] = df
            if df.empty:
                print(f'--> FAILED to fetch {series_id}')
            else:
                print(f'--> Success ({len(df)} rows)')


    print('\n--- Fetching FX Rates ---')
    fx_data = fetch_fx_data(FX_TICKERS)
    if fx_data.empty:
        print("--> FAILED to fetch FX data.")
    else:
        print(f"--> Success ({fx_data.shape})")
        print("FX Data Head:")
        print(fx_data.head())

    print("\n--- Data Fetching Verification Complete ---")

if __name__ == "__main__":
    main()
