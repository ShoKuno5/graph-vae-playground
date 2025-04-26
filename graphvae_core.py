# graphvae_core.py
import torch
from torch import nn
from torch_geometric.nn import GCNConv

class GEncoder(nn.Module):
    def __init__(self, in_dim, hid_dim, z_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim)
        self.mu    = GCNConv(hid_dim, z_dim)
        self.logvar= GCNConv(hid_dim, z_dim)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        return self.mu(h, edge_index), self.logvar(h, edge_index)

class GDecoder(nn.Module):
    def forward(self, z):
        return torch.sigmoid(z @ z.t())  # inner-product decoder
