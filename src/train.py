"""
Main script for training and validating the GNN models for FX arbitrage.
"""
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import os

from src.data_fred import download_fred_data, process_raw_data
from src.fxrp_model import FXRP_Model, feature_engineering_fxrp, create_graph_data_for_day
from src.fxsa_model import FXSA_Model
from src.preprocess import load_data

# --- Configuration ---
CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'SEK']
HOME_CURRENCY = 'USD'
LOOKBACKS = [1, 5, 10, 20]

def train_fxrp_model(data, epochs=5, lr=0.001):
    """
    Trains the FXRP model on the given historical data.
    """
    print(f"\n--- Training FXRP Model for {epochs} epochs... ---")

    fxrp_model = FXRP_Model(CURRENCIES, LOOKBACKS)
    fxrp_model.gnn.train()
    optimizer = torch.optim.Adam(fxrp_model.gnn.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    print("Pre-computing features for training...")
    fx_diff, ir_diff, v_diff, _ = feature_engineering_fxrp(data, CURRENCIES, LOOKBACKS)

    edges = []
    for c1 in CURRENCIES:
        for c2 in CURRENCIES:
            if c1 != c2:
                edges.append((c1, c2))
    edges = sorted(edges)
    fx_cols = [f'FX_{c1}_{c2}' for c1, c2 in edges]
    actual_log_diffs_df = np.log(data[fx_cols]).diff()

    for epoch in range(epochs):
        total_loss = 0
        for t in tqdm(range(1, len(data)), desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            optimizer.zero_grad()
            graph_data = create_graph_data_for_day(
                fx_diff.iloc[t-1], ir_diff.iloc[t-1], v_diff.iloc[t-1],
                CURRENCIES, LOOKBACKS)
            pred_log_diffs = fxrp_model.gnn(graph_data)
            target_log_diffs = torch.tensor(actual_log_diffs_df.iloc[t].values, dtype=torch.float)
            if not torch.isnan(target_log_diffs).any():
                loss = loss_fn(pred_log_diffs, target_log_diffs)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
        avg_loss = total_loss / len(data) if len(data) > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.8f}")

    print("--- FXRP Model Training Finished ---")
    fxrp_model.gnn.eval()
    return fxrp_model

def calculate_realized_gain(w_t_series, fx_preds_t, data_t, data_t_plus_1):
    H_t = {}
    for i_curr in CURRENCIES:
        inflows, outflows = 0, 0
        for j_curr in CURRENCIES:
            if i_curr == j_curr: continue
            w_ji = w_t_series.get(f"{j_curr}_{i_curr}", 0)
            if w_ji > 0:
                x_ji_real = data_t.get(f'FX_{j_curr}_{i_curr}', 0)
                x_hat_oj = fx_preds_t.get(f'FX_{HOME_CURRENCY}_{j_curr}', 0)
                inflows += x_ji_real * x_hat_oj * w_ji
            w_ij = w_t_series.get(f"{i_curr}_{j_curr}", 0)
            if w_ij > 0:
                x_hat_oi = fx_preds_t.get(f'FX_{HOME_CURRENCY}_{i_curr}', 0)
                outflows += x_hat_oi * w_ij
        H_t[i_curr] = inflows - outflows
    gain = 0
    y_o_real = data_t.get(f'IR_{HOME_CURRENCY}', 0) / 100
    for i_curr in CURRENCIES:
        if i_curr == HOME_CURRENCY: continue
        y_i_real = data_t.get(f'IR_{i_curr}', 0) / 100
        x_io_t_plus_1_real = data_t_plus_1.get(f'FX_{i_curr}_{HOME_CURRENCY}', 0)
        present_value_factor = (1 + y_i_real) / (1 + y_o_real) if (1 + y_o_real) != 0 else 0
        gain += present_value_factor * x_io_t_plus_1_real * H_t[i_curr]
    return gain

def train_fxsa_model(trained_fxrp_model, data, epochs=5, lr=0.001, batch_size=16):
    print(f"\n--- Training FXSA Model for {epochs} epochs... ---")
    fxsa_model = FXSA_Model(CURRENCIES, HOME_CURRENCY)
    optimizer = torch.optim.Adam(fxsa_model.gnn.parameters(), lr=lr)

    print("Pre-computing FXRP predictions for FXSA training...")
    fxrp_preds_all, log_v_all = [], []
    for t in tqdm(range(1, len(data)), desc="FXRP Predictions", ncols=100):
        predicted_fx, predicted_log_v = trained_fxrp_model.predict(data.iloc[:t])
        fxrp_preds_all.append(predicted_fx)
        log_v_all.append(predicted_log_v)

    for epoch in range(epochs):
        fxsa_model.gnn.train()
        total_loss, num_batches = 0, 0
        for i in tqdm(range(1, len(data) - batch_size - 1, batch_size), desc=f"Epoch {epoch+1}/{epochs}", ncols=100):
            optimizer.zero_grad()
            batch_gains = []
            t_rep = i + batch_size // 2
            fx_preds_rep, log_v_rep = fxrp_preds_all[t_rep - 1], log_v_all[t_rep - 1]
            u_prime = fxsa_model.gnn(fxsa_model.create_fxsa_graph(fx_preds_rep, log_v_rep))
            for j in range(batch_size):
                t = i + j
                with torch.no_grad():
                    w_t_series = fxsa_model.generate_strategy(fxrp_preds_all[t-1], log_v_all[t-1])
                gain_t = calculate_realized_gain(w_t_series, fxrp_preds_all[t-1], data.iloc[t], data.iloc[t+1])
                batch_gains.append(gain_t)
            mu_G, var_G = np.mean(batch_gains), np.var(batch_gains)
            loss_val = - (mu_G**2) / (var_G + 1e-9) if mu_G > 0 else - mu_G
            surrogate_loss = torch.mean(u_prime) * np.sign(loss_val) * 1e-5
            surrogate_loss.backward()
            optimizer.step()
            total_loss += loss_val
            num_batches += 1
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Average FXSA Objective (Loss): {avg_loss:.8f}")
    print("--- FXSA Model Training Finished ---")
    fxsa_model.gnn.eval()
    return fxsa_model

def run_validation(start_date='2015-01-01', retrain_freq='QS', fxrp_epochs=1, fxsa_epochs=1):
    print("--- Running Walk-Forward Validation ---")
    print("\n[NOTE] This is a computationally intensive process and may take a very long time.")
    print("For a quick test, consider passing a more recent start_date, e.g., '2022-01-01'.")

    print("\n[Step 1/3] Downloading and processing data from FRED...")
    data = load_data()
    if data is None:
        raw_data = download_fred_data()
        if raw_data is None or raw_data.empty: return
        data = process_raw_data(raw_data)
        if data is None: return

    test_periods = pd.date_range(start=start_date, end=data.index.max(), freq=retrain_freq)
    all_period_gains = []
    for i in range(len(test_periods) - 1):
        period_start, period_end = test_periods[i], test_periods[i+1]
        print(f"\n--- Processing Period: {period_start.date()} to {period_end.date()} ---")
        train_data = data[data.index < period_start]
        test_data = data[(data.index >= period_start) & (data.index < period_end)]
        if len(train_data) < 50 or len(test_data) < 2:
            print("Not enough data to train or test this period. Skipping.")
            continue
        print(f"Training on {len(train_data)} samples, testing on {len(test_data)} samples.")
        trained_fxrp_model = train_fxrp_model(train_data, epochs=fxrp_epochs)
        trained_fxsa_model = train_fxsa_model(trained_fxrp_model, train_data, epochs=fxsa_epochs)
        period_gains = []
        for t in tqdm(range(len(test_data) - 1), desc="Backtesting", ncols=100):
            history = pd.concat([train_data, test_data.iloc[:t]])
            fx_preds_t, log_v_t = trained_fxrp_model.predict(history)
            w_t = trained_fxsa_model.generate_strategy(fx_preds_t, log_v_t)
            gain_t = calculate_realized_gain(w_t, fx_preds_t, test_data.iloc[t], test_data.iloc[t+1])
            period_gains.append(gain_t)
        all_period_gains.extend(period_gains)
        if period_gains:
            mean_gain, std_gain = np.mean(period_gains), np.std(period_gains)
            sharpe_ratio = (mean_gain / std_gain) * np.sqrt(252) if std_gain > 0 else 0
            print(f"Period Performance: Mean Daily Gain: {mean_gain:.6f}, Sharpe Ratio: {sharpe_ratio:.2f}")

    print("\n\n--- Overall Validation Finished ---")
    if all_period_gains:
        mean_gain, std_gain = np.mean(all_period_gains), np.std(all_period_gains)
        total_sharpe = (mean_gain / std_gain) * np.sqrt(252) if std_gain > 0 else 0
        print(f"Total Strategies Evaluated: {len(all_period_gains)}")
        print(f"Overall Mean Daily Gain: {mean_gain:.6f}")
        print(f"Overall Sharpe Ratio (annualized): {total_sharpe:.2f}")
        print(f"Total Cumulative Gain: {np.sum(all_period_gains):.4f}")
    else:
        print("No periods were tested.")

if __name__ == '__main__':
    run_validation()
