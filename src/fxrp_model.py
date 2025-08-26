import numpy as np
import pandas as pd
from scipy.linalg import lstsq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from src.preprocess import load_data

# --- Part 1: Feature Engineering ---

def calculate_currency_values_for_day(fx_rates_today, currencies):
    """
    Solves the least-squares problem to find log currency values log(V_ti) for a single timestamp.
    Implements equations 8 and 9 from the paper.

    Args:
        fx_rates_today (pd.Series): Series with FX rates for a single day.
                                    Index format: 'FX_CURR1_CURR2'.
        currencies (list): List of currency strings.

    Returns:
        np.array: Array of log currency values, log(V_ti), for the given day.
    """
    num_currencies = len(currencies)
    currency_map = {name: i for i, name in enumerate(currencies)}

    equations = []
    constants = []

    # Equation 8: log(V_i) - log(V_j) = log(X_ij)
    for i_idx, i_curr in enumerate(currencies):
        for j_idx, j_curr in enumerate(currencies):
            if i_idx >= j_idx:
                continue

            fx_col = f'FX_{i_curr}_{j_curr}'
            if fx_col in fx_rates_today and pd.notna(fx_rates_today[fx_col]) and fx_rates_today[fx_col] > 0:
                row = np.zeros(num_currencies)
                row[i_idx] = 1
                row[j_idx] = -1
                equations.append(row)
                constants.append(np.log(fx_rates_today[fx_col]))

    if not equations: # Handle case with no valid FX rates
        return np.full(num_currencies, np.nan)

    # Equation 9: sum(log(V_i)) = 0 (to ensure a unique solution)
    row = np.ones(num_currencies)
    equations.append(row)
    constants.append(0)

    A = np.array(equations)
    b = np.array(constants)

    # Solve the system Ax = b for x = [log(V_i)]
    solution, _, _, _ = lstsq(A, b)

    return solution

def get_temporal_log_diffs(df, lookback_windows):
    """
    Calculates the average temporal log differences for a DataFrame over given lookback windows.
    This is a helper function for creating x_tij, y_ti, and v_ti features.
    """
    log_df = np.log(df)
    # The paper computes log(P_t / P_{t-1}), which is log(P_t) - log(P_{t-1})
    log_returns = log_df.diff()

    feature_dfs = {}
    for window in lookback_windows:
        # The paper uses a simple moving average of these log returns.
        avg_log_returns = log_returns.rolling(window=window, min_periods=1).mean()
        feature_dfs[window] = avg_log_returns

    # Restructure dataframe to have one column per (original_col, window)
    feature_df = pd.concat(feature_dfs, axis=1)
    feature_df.columns = [f'{col}_log_diff_{window}d' for window, col in feature_df.columns]
    return feature_df

def feature_engineering_fxrp(raw_data, currencies, lookback_windows):
    """
    Main function to create all features for the FXRP model (h_PI in the paper).
    """
    print("Starting feature engineering...")

    # 1. Calculate currency values (V_t) for all timestamps
    fx_cols = [f'FX_{c1}_{c2}' for c1 in currencies for c2 in currencies if c1 != c2]
    fx_data = raw_data[fx_cols]

    print("Calculating currency values (V_t)...")
    log_v_data_list = fx_data.apply(
        lambda row: calculate_currency_values_for_day(row, currencies),
        axis=1
    )
    log_v_data = pd.DataFrame(log_v_data_list.tolist(), index=raw_data.index, columns=[f'LOG_V_{c}' for c in currencies])

    # 2. Get temporal log differences for FX, IR, and V
    print("Calculating temporal log differences...")
    fx_log_diffs = get_temporal_log_diffs(fx_data, lookback_windows)

    ir_cols = [f'IR_{c}' for c in currencies]
    ir_data = raw_data[ir_cols]
    # Add 1 before taking log as per paper: log((1+Y_t)/(1+Y_{t-1}))
    ir_log_diffs = get_temporal_log_diffs(1 + ir_data, lookback_windows)

    v_data = np.exp(log_v_data) # Convert log(V) to V
    v_log_diffs = get_temporal_log_diffs(v_data, lookback_windows)

    print("Feature engineering complete.")
    return fx_log_diffs, ir_log_diffs, v_log_diffs, log_v_data


# --- Part 2: GNN Model ---

class FXRPGNNLayer(MessagePassing):
    """
    A single layer of the FXRP GNN, implementing equations 10 and 11.
    Updates node features first, then edge features.
    """
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super().__init__(aggr='mean') # Equation 10 uses mean aggregation

        # SLP_N,l in Eq. 10: MLP for node updates
        self.mlp_node_update = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, node_dim)
        )

        # SLP_E,l in Eq. 11: MLP for edge updates
        self.mlp_edge_update = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, edge_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        # Store original features for message passing
        x_orig = x
        edge_attr_orig = edge_attr

        # --- Node Update (Equation 10) ---
        # The `propagate` call triggers the message passing process.
        # We pass the original node features `x_orig` to be used in `message`.
        x_new = self.propagate(edge_index, x=x_orig, edge_attr=edge_attr_orig)

        # --- Edge Update (Equation 11) ---
        row, col = edge_index
        # Concatenate new node features and old edge features
        edge_update_input = torch.cat([x_new[row], x_new[col], edge_attr_orig], dim=-1)
        edge_attr_new = self.mlp_edge_update(edge_update_input)

        return x_new, edge_attr_new

    def message(self, x_i, x_j, edge_attr):
        # This function computes the message from j to i.
        # The message is SLP_N,l([n_i; e_ji; n_j])
        # Note: torch_geometric gives us e_ij. For an undirected graph, e_ji = e_ij.
        # We assume the graph is undirected as per the paper's use of L_t.
        msg_input = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp_node_update(msg_input)

    def update(self, aggr_out):
        # The aggregation is 'mean', so aggr_out is already the new node feature.
        return aggr_out


class FXRPGNN(nn.Module):
    """
    The full FXRP GNN model (g_P in the paper).
    """
    def __init__(self, num_node_features, num_edge_features, hidden_dim, num_layers=2):
        super().__init__()

        # Initial projection layers (g_CS in the paper)
        self.node_in_proj = nn.Linear(num_node_features, hidden_dim)
        self.edge_in_proj = nn.Linear(num_edge_features, hidden_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(FXRPGNNLayer(hidden_dim, hidden_dim, hidden_dim))

        # Final output layer (SLP_H in the paper)
        # Predicts the log difference for each edge
        self.output_proj = nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Initial projection
        x = F.leaky_relu(self.node_in_proj(x))
        edge_attr = F.leaky_relu(self.edge_in_proj(edge_attr))

        # GNN layers
        for layer in self.layers:
            x, edge_attr = layer(x, edge_index, edge_attr)

        # Final prediction for each edge
        # The paper is an edge-level regression, so we use the final edge_attr
        out = self.output_proj(edge_attr)
        return out.squeeze(-1)

# --- Part 3: Data Preparation for GNN ---

def create_graph_data_for_day(
    fx_log_diffs_day,
    ir_log_diffs_day,
    v_log_diffs_day,
    currencies,
    lookback_windows
):
    """
    Creates a torch_geometric.data.Data object for a single day.
    """
    num_currencies = len(currencies)
    currency_map = {name: i for i, name in enumerate(currencies)}

    # Node features (c_ti): concatenate IR and V log diffs
    node_features = []
    for curr in currencies:
        ir_feats = [ir_log_diffs_day.get(f'IR_{curr}_log_diff_{w}d', 0) for w in lookback_windows]
        v_feats = [v_log_diffs_day.get(f'LOG_V_{curr}_log_diff_{w}d', 0) for w in lookback_windows]
        node_features.append(ir_feats + v_feats)
    x = torch.tensor(node_features, dtype=torch.float).nan_to_num()

    # Edges and edge features (x_tij)
    edge_list = []
    edge_attr_list = []
    for i_curr in currencies:
        for j_curr in currencies:
            if i_curr == j_curr:
                continue

            i_idx = currency_map[i_curr]
            j_idx = currency_map[j_curr]
            edge_list.append([i_idx, j_idx])

            fx_feats = [fx_log_diffs_day.get(f'FX_{i_curr}_{j_curr}_log_diff_{w}d', 0) for w in lookback_windows]
            edge_attr_list.append(fx_feats)

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float).nan_to_num()

    # The paper assumes reciprocal edges. In torch_geometric, it's good practice
    # to ensure the graph is undirected if the model assumes it.
    # edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


# --- Part 4: Full Model Wrapper ---

class FXRP_Model:
    """
    A wrapper class that combines feature engineering and the GNN model for FXRP.
    """
    def __init__(self, currencies, lookback_windows, hidden_dim=32, num_layers=2):
        self.currencies = currencies
        self.lookback_windows = lookback_windows

        # Determine feature dimensions to init the GNN
        num_node_features = len(lookback_windows) * 2 # ir_feats + v_feats
        num_edge_features = len(lookback_windows) # fx_feats

        self.gnn = FXRPGNN(
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers
        )
        # NOTE: In a full implementation, this model would be trained.
        # For this PoC, we use it with its initial random weights.
        self.gnn.eval()

    def predict(self, historical_data):
        """
        Takes historical data up to day t-1 and predicts FX rates for day t.
        """
        # 1. Feature Engineering
        fx_diff, ir_diff, v_diff, log_v = feature_engineering_fxrp(
            historical_data, self.currencies, self.lookback_windows
        )

        # 2. Create graph for the latest day (t-1)
        latest_day_index = -1
        graph_data = create_graph_data_for_day(
            fx_diff.iloc[latest_day_index],
            ir_diff.iloc[latest_day_index],
            v_diff.iloc[latest_day_index],
            self.currencies,
            self.lookback_windows
        )

        # 3. Run GNN to get predicted log differences
        with torch.no_grad():
            pred_log_diffs = self.gnn(graph_data).cpu().numpy()

        # 4. Apply output scaling (h_PO from paper)
        # X_hat_t = X_{t-1} * exp(pred_log_diff)
        edges = []
        for i_curr in self.currencies:
            for j_curr in self.currencies:
                if i_curr == j_curr: continue
                edges.append((i_curr, j_curr))

        fx_rates_t_minus_1 = historical_data[[f'FX_{c1}_{c2}' for c1,c2 in edges]].iloc[-1]

        predicted_fx_rates = fx_rates_t_minus_1 * np.exp(pred_log_diffs)

        # The paper also re-enforces the reciprocal constraint on predictions
        # X_hat_ij = 1 / X_hat_ji. We can do this with a geometric average.
        edge_map = {edge: i for i, edge in enumerate(edges)}
        final_preds = predicted_fx_rates.copy()
        for i_curr in self.currencies:
            for j_curr in self.currencies:
                if i_curr >= j_curr: continue

                idx1 = edge_map[(i_curr, j_curr)]
                idx2 = edge_map[(j_curr, i_curr)]

                pred1 = predicted_fx_rates.iloc[idx1]
                pred2 = 1 / predicted_fx_rates.iloc[idx2]

                geom_mean = np.sqrt(pred1 * pred2)

                final_preds.iloc[idx1] = geom_mean
                final_preds.iloc[idx2] = 1 / geom_mean

        latest_log_v = log_v.iloc[-1]

        return final_preds, latest_log_v

if __name__ == '__main__':
    from src.preprocess import load_data

    print("Running FXRP model end-to-end test...")
    CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'SEK']
    LOOKBACKS = [1, 5, 10, 20]

    data = load_data()
    if data is not None:
        # Use a slice of data as "historical"
        historical_data = data.iloc[:-1]

        fxrp_model = FXRP_Model(CURRENCIES, LOOKBACKS)

        predicted_fx, predicted_log_v = fxrp_model.predict(historical_data)

        print("\nFXRP Model Prediction successful.")
        print("Predicted FX Rates for next day (sample):")
        print(predicted_fx.head())
        print("\nPredicted Log V for next day (sample):")
        print(predicted_log_v.head())
