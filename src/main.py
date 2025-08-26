"""
Main script to run the full FX prediction and trading strategy pipeline.
"""
import os
from src.preprocess import load_data
from src.fxrp_model import FXRP_Model
from src.fxsa_model import FXSA_Model
from src.data_fred import download_fred_data, process_raw_data

# --- Configuration ---
CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'SEK']
HOME_CURRENCY = 'USD'
LOOKBACKS = [1, 5, 10, 20] # Lookback windows for feature engineering

def run_pipeline():
    """
    Executes the end-to-end pipeline.
    """
    print("--- Starting FX Arbitrage GNN Pipeline ---")

    # 1. Load Data (or download if it doesn't exist)
    print("\n[Step 1/4] Loading data...")
    model_data_path = 'data/model_ready_data.csv'
    if not os.path.exists(model_data_path):
        print("Model-ready data not found. Running FRED data pipeline...")
        raw_data = download_fred_data()
        if raw_data is None or raw_data.empty:
            print("Failed to download data. Exiting.")
            return
        process_raw_data(raw_data)

    data = load_data(model_data_path)
    if data is None:
        print("Failed to load data even after download attempt. Exiting.")
        return
    # Use all but the last day as historical input for the prediction
    historical_data = data.iloc[:-1]
    print(f"Data loaded. Using {len(historical_data)} days of historical data.")

    # 2. Run FXRP Model to get predictions
    print("\n[Step 2/4] Running FX Rate Prediction Model (FXRP)...")
    fxrp_model = FXRP_Model(CURRENCIES, LOOKBACKS)
    predicted_fx, predicted_log_v = fxrp_model.predict(historical_data)
    print("FXRP model generated predictions for the next day.")

    # 3. Run FXSA Model to get trading strategy
    print("\n[Step 3/4] Running FX Statistical Arbitrage Model (FXSA)...")
    fxsa_model = FXSA_Model(CURRENCIES, HOME_CURRENCY)
    trading_weights = fxsa_model.generate_strategy(predicted_fx, predicted_log_v)
    print("FXSA model generated a trading strategy.")

    # 4. Display Results
    print("\n[Step 4/4] Displaying Results...")
    print("-----------------------------------------")
    print("          FXSA Trading Strategy          ")
    print("-----------------------------------------")

    non_zero_weights = trading_weights[trading_weights > 1e-9]

    if non_zero_weights.empty:
        print("No arbitrage opportunity found. No trades suggested.")
    else:
        print(f"Strategy suggests allocating capital to {len(non_zero_weights)} trades:")
        for trade, weight in non_zero_weights.items():
            print(f"  - Trade {trade.replace('_', '/')}: {weight:.2%}")

    print("-----------------------------------------")
    print("--- Pipeline Finished ---")

if __name__ == '__main__':
    run_pipeline()
