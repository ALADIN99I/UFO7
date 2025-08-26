import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import null_space
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# --- Part 1: Constraint Enforcement Logic ---

def get_constraint_matrix_M(fx_preds_day, currencies, edge_map, home_currency='USD'):
    """
    Constructs the matrix M for the linear constraints on u (Eqs 25, 26),
    such that M @ u = 0.
    """
    num_currencies = len(currencies)
    num_edges = len(edge_map)

    M_constraints = []

    # Equation 25: sum_j(u_ij) = 0 for each i in C \ {o}
    for i_curr in currencies:
        if i_curr == home_currency:
            continue
        row = np.zeros(num_edges)
        for j_curr in currencies:
            if i_curr == j_curr:
                continue
            edge = (i_curr, j_curr)
            if edge in edge_map:
                edge_idx = edge_map[edge]
                row[edge_idx] = 1
        M_constraints.append(row)

    # Equation 26: X_oi*u_ij + X_oj*X_ji*u_ji = 0
    for i_idx, i_curr in enumerate(currencies):
        for j_idx, j_curr in enumerate(currencies):
            if i_idx >= j_idx:
                continue

            edge1 = (i_curr, j_curr)
            edge2 = (j_curr, i_curr)

            if edge1 in edge_map and edge2 in edge_map:
                row = np.zeros(num_edges)
                x_oi = fx_preds_day.get(f'FX_{home_currency}_{i_curr}', 1.0) if i_curr != home_currency else 1.0
                x_oj = fx_preds_day.get(f'FX_{home_currency}_{j_curr}', 1.0) if j_curr != home_currency else 1.0
                x_ji = fx_preds_day.get(f'FX_{j_curr}_{i_curr}', 1.0)

                row[edge_map[edge1]] = x_oi
                row[edge_map[edge2]] = x_oj * x_ji
                M_constraints.append(row)

    if not M_constraints:
        return np.zeros((0, num_edges))

    M = np.array(M_constraints)
    return M

def get_projection_matrix(M):
    """
    Calculates the projection matrix P_t for the null space of M.
    """
    if M.shape[0] == 0: # No constraints
        return np.eye(M.shape[1])

    # Basis for the null space of M (B_t in the paper)
    # This basis is orthonormal, so B.T @ B = I
    basis = null_space(M)

    # P_t = B @ (B.T @ B)^-1 @ B.T simplifies to B @ B.T
    P_t = basis @ basis.T
    return P_t

def apply_h_so(u_prime, P_t):
    """
    Applies the constraint enforcement layer h_SO from the paper.
    Projects u' onto the constraint subspace, then applies ReLU and normalization.
    """
    # Project u' onto the constraint subspace D_t
    u = P_t @ u_prime

    # Apply ReLU to get non-negative values
    w_unscaled = np.maximum(0, u)

    # Normalize to satisfy sum(w_ij) = 1 (Equation 28)
    total_w = w_unscaled.sum()
    if total_w > 1e-9: # Avoid division by zero
        w = w_unscaled / total_w
    else:
        w = np.zeros_like(u) # If all are zero, the sum is zero

    return w

# --- Part 2: FXSA Feature & Graph Engineering ---

def calculate_arbitrage_features(fx_preds_day, log_v_day, edges, edge_map):
    """Calculates alpha_hat features for each exchange (node in the FXSA graph)."""
    alpha_hats = np.zeros(len(edges))
    log_v_map = {f'LOG_V_{k}':v for k,v in log_v_day.items()}

    for edge, idx in edge_map.items():
        curr1, curr2 = edge
        log_x_hat = np.log(fx_preds_day.get(f'FX_{curr1}_{curr2}', 1.0))
        log_v_hat1 = log_v_map.get(f'LOG_V_{curr1}', 0.0)
        log_v_hat2 = log_v_map.get(f'LOG_V_{curr2}', 0.0)
        alpha_hats[idx] = log_x_hat - log_v_hat1 + log_v_hat2
    return alpha_hats

def create_fxsa_graph_data(node_features, projection_matrix, epsilon_S=1e-6):
    """
    Creates the torch_geometric.data.Data object for the FXSA GNN.
    Nodes are exchanges. Edges represent influence from projection matrix.
    """
    x = torch.tensor(node_features, dtype=torch.float)

    # Create edges based on projection matrix
    adj = np.abs(projection_matrix) > epsilon_S
    np.fill_diagonal(adj, 0)
    edge_index = torch.tensor(np.array(np.where(adj)), dtype=torch.long)

    # Edge features are the values from the projection matrix
    edge_attr = torch.tensor(projection_matrix[adj], dtype=torch.float).unsqueeze(1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# --- Part 3: FXSA GNN Model ---
class FXSAGNN(nn.Module):
    """
    GNN for the FXSA model. Nodes are exchanges.
    The paper is light on details for g_S, so we use a standard GCN.
    """
    def __init__(self, num_node_features, hidden_dim, out_dim=1):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.output_layer(x)
        return x.squeeze(-1)

# --- Part 4: Full FXSA Model Wrapper ---
class FXSA_Model:
    def __init__(self, currencies, home_currency, hidden_dim=16):
        self.currencies = currencies
        self.home_currency = home_currency
        self.edges = sorted([(c1, c2) for c1 in self.currencies for c2 in self.currencies if c1 != c2])
        self.edge_map = {edge: i for i, edge in enumerate(self.edges)}

        # The paper mentions temporal averages of alpha_hat as features.
        # For simplicity, we'll assume num_node_features=1 (the current alpha_hat)
        self.gnn = FXSAGNN(num_node_features=1, hidden_dim=hidden_dim)
        # NOTE: In a full implementation, this model would be trained.
        # For this PoC, we use it with its initial random weights.
        self.gnn.eval()

    def create_fxsa_graph(self, fx_preds_day, log_v_day):
        """Helper method to create the graph for the GNN forward pass."""
        M = get_constraint_matrix_M(fx_preds_day, self.currencies, self.edge_map, self.home_currency)
        P_t = get_projection_matrix(M)
        alpha_hat_features = calculate_arbitrage_features(fx_preds_day, log_v_day, self.edges, self.edge_map)
        return create_fxsa_graph_data(alpha_hat_features[:, None], P_t)

    def generate_strategy(self, fx_preds_day, log_v_day):
        fxsa_graph = self.create_fxsa_graph(fx_preds_day, log_v_day)

        with torch.no_grad():
            u_prime = self.gnn(fxsa_graph).cpu().numpy()

        # We need P_t again for the h_SO step
        M = get_constraint_matrix_M(fx_preds_day, self.currencies, self.edge_map, self.home_currency)
        P_t = get_projection_matrix(M)
        w = apply_h_so(u_prime, P_t)

        return pd.Series(w, index=[f"{e[0]}_{e[1]}" for e in self.edges])

# --- Part 5: Updated Test Block ---
if __name__ == "__main__":
    from src.fxrp_model import FXRP_Model
    from src.preprocess import load_data

    print("Running FXSA model end-to-end test...")
    CURRENCIES = ['USD', 'EUR', 'JPY', 'GBP', 'AUD', 'CAD', 'CHF', 'SEK']
    HOME_CURRENCY = 'USD'
    LOOKBACKS = [1, 5, 10, 20]

    # 1. Load data
    data = load_data()
    if data is not None:
        historical_data = data.iloc[:-1]

        # 2. Get predictions from FXRP model
        fxrp_model = FXRP_Model(CURRENCIES, LOOKBACKS)
        predicted_fx, predicted_log_v = fxrp_model.predict(historical_data)
        print("\nGot FXRP predictions.")

        # 3. Instantiate FXSA model and generate strategy
        fxsa_model = FXSA_Model(CURRENCIES, HOME_CURRENCY)
        trading_weights = fxsa_model.generate_strategy(predicted_fx, predicted_log_v)

        print("\nFXSA Model generated a trading strategy successfully.")
        print(f"Sum of weights: {trading_weights.sum():.6f}")
        print("Top 5 trading weights:")
        print(trading_weights.nlargest(5))
