import pandas as pd

def load_data(filepath='data/model_ready_data.csv'):
    """
    Loads the pre-processed data from the specified CSV file.
    """
    try:
        data = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return data
    except FileNotFoundError:
        print(f"Error: Data file not found at {filepath}")
        print("Please run `python -m src.data_fred` first to download the data from the FRED API.")
        return None

def clean_data(df):
    """
    Placeholder for data cleaning operations.
    Our synthetic data is clean, but this function would handle
    missing values, outliers, etc. in a real-world scenario.
    """
    # Example: forward-fill missing values
    df_cleaned = df.ffill()
    return df_cleaned

if __name__ == '__main__':
    raw_data = load_data()
    if raw_data is not None:
        cleaned_data = clean_data(raw_data)
        print("Data loaded and cleaned successfully.")
        print("Data head:")
        print(cleaned_data.head())
        print("\nData info:")
        cleaned_data.info()
