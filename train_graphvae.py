import torch, torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from graphvae_core import GEncoder, GDecoder

class GraphVAE(torch.nn.Module):
    def __init__(self, in_dim, hid=64, z=32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z)
        self.dec = GDecoder()

    def forward(self, data):
        mu, logvar = self.enc(data.x, data.edge_index)
        z = mu                               # ★後で reparam
        adj_hat = self.dec(z)                # NxN
        return adj_hat, mu, logvar, z

def main():
    ds = QM9(root="data/QM9")[:100]          # 小規模でテスト
    loader = DataLoader(ds, batch_size=1)
    device = "cpu"
    model = GraphVAE(ds.num_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(2):
        for data in loader:
            data = data.to(device)
            adj_hat, mu, logvar, _ = model(data)

            # 真の隣接行列
            adj_true = torch.zeros_like(adj_hat)
            adj_true[data.edge_index[0], data.edge_index[1]] = 1

            recon = F.binary_cross_entropy(adj_hat, adj_true)
            kl = -0.5 * (1 + logvar - mu**2 - logvar.exp()).mean()
            loss = recon + kl

            opt.zero_grad(); loss.backward(); opt.step()
        print(f"Epoch {epoch}  Loss {loss.item():.4f}")

if __name__ == "__main__":
    main()
