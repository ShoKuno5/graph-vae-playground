# graphvae_core.py
import torch
from torch import nn
from torch_geometric.nn import GCNConv

# ---------- GNN Encoder (そのまま) -----------------
class GEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim):
        super().__init__()
        self.conv1  = GCNConv(in_dim, hid_dim)
        self.mu     = GCNConv(hid_dim, z_dim)
        self.logvar = GCNConv(hid_dim, z_dim)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        return self.mu(h, edge_index), self.logvar(h, edge_index)

# ---------- 新: EdgeMLP Decoder --------------------
class EdgeMLP(nn.Module):
    """
    入力:  z  (N, z_dim)
    出力:  A_hat  (N, N) 0-1 確率隣接行列
    """
    def __init__(self, z_dim, hid_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, hid_dim), nn.ReLU(),
            nn.Linear(hid_dim, 1)
        )

    def forward(self, z):
        N = z.size(0)
        zi = z.unsqueeze(1).expand(N, N, -1)   # (N,N,z)
        zj = z.unsqueeze(0).expand(N, N, -1)   # (N,N,z)
        logits = self.net(torch.cat([zi, zj], dim=-1)).squeeze(-1)  # (N,N)
        return torch.sigmoid(logits)
