# train_graphvae.py
# -------------------------------------------------
import os, argparse, torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from graphvae_core import GEncoder          # 既存エンコーダ
from scipy.optimize import linear_sum_assignment

# ---------- Hungarian --------------------------------------------------
def permute_adj(A_true, A_pred):
    eps = 1e-9
    cost = -(A_true * torch.log(A_pred + eps) +
             (1 - A_true) * torch.log(1 - A_pred + eps))
    r, c = linear_sum_assignment(cost.detach().cpu().numpy())
    P = torch.zeros_like(A_true)
    P[r, c] = 1.0
    return P.T @ A_true @ P
# -----------------------------------------------------------------------

# ---------- 新: MLP デコーダ ------------------------------------------
class EdgeMLP(nn.Module):
    def __init__(self, z_dim, hid=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim * 2, hid), nn.ReLU(),
            nn.Linear(hid, 1)
        )

    def forward(self, z):                 # z: (N, z_dim)
        N = z.size(0)
        zi = z.unsqueeze(1).expand(N, N, -1)
        zj = z.unsqueeze(0).expand(N, N, -1)
        logits = self.net(torch.cat([zi, zj], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits)      # (N, N)
# -----------------------------------------------------------------------

class GraphVAE(nn.Module):
    def __init__(self, in_dim, hid=64, z_dim=32):
        super().__init__()
        self.enc = GEncoder(in_dim, hid, z_dim)
        self.dec = EdgeMLP(z_dim)         # ← 置き換え

    def forward(self, data):
        mu, logvar = self.enc(data.x, data.edge_index)
        eps = torch.randn_like(mu)
        z = mu + eps * torch.exp(0.5 * logvar)
        A_hat = self.dec(z)               # (N,N) 0–1
        return A_hat, mu, logvar

# -----------------------------------------------------------------------
def train(args):
    device = torch.device("cpu")
    ds = QM9(root="data/QM9")[:100]               # debug subset
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    model = GraphVAE(ds.num_features).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs("runs", exist_ok=True)

    for ep in range(args.epochs):
        L = R = K = 0
        # --- 動的パラメータスケジュール ---
        neg_w = args.neg_w_start - (args.neg_w_start - args.neg_w_end) * ep / args.epochs
        l1    = args.l1_start  - (args.l1_start  - args.l1_end ) * ep / args.epochs
        kl_w  = min(1.0, ep / args.kl_w_warmup) if args.anneal else 1.0

        for data in loader:
            data = data.to(device)
            A_hat, mu, logvar = model(data)

            # 真値隣接 → Hungarian で並び替え
            A_true = torch.zeros_like(A_hat)
            A_true[data.edge_index[0], data.edge_index[1]] = 1
            A_perm = permute_adj(A_true, A_hat)

            # 重み付き BCE
            bce = F.binary_cross_entropy(A_hat, A_perm, reduction='none')
            recon = ((1 - A_perm) * neg_w * bce + A_perm * bce).mean()

            # KL
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

            # L1 sparsity
            loss = recon + kl_w * kl + l1 * A_hat.sum()

            opt.zero_grad(); loss.backward(); opt.step()
            L += loss.item(); R += recon.item(); K += kl.item()

        print(f"Ep {ep:02d} | Loss {L/len(loader):.4f}  "
              f"Re {R/len(loader):.3f} KL {K/len(loader):.3f} "
              f"(neg_w={neg_w:.1f}, l1={l1:.4g}, klw={kl_w:.2f})")

    ckpt = f"runs/graphvae_epoch{args.epochs}.pt"
    torch.save(model.state_dict(), ckpt)
    print(f"✔ saved {ckpt}")

# -----------------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--epochs", type=int, default=50)

    # 動的 neg_w : start (強) → end (弱)
    pa.add_argument("--neg_w_start", type=float, default=50.0)
    pa.add_argument("--neg_w_end",   type=float, default=10.0)

    # 動的 L1 : start (強) → end (弱)
    pa.add_argument("--l1_start", type=float, default=1e-3)
    pa.add_argument("--l1_end",   type=float, default=5e-5)

    # KL weight warm-up epochs
    pa.add_argument("--kl_w_warmup", type=int, default=30)
    pa.add_argument("--anneal", action="store_true", help="use KL annealing")
    args = pa.parse_args()
    train(args)
