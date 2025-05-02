#!/usr/bin/env python
# eval_stable.py – 固定 SEED / 2 000 sample ×5 回平均 / 対称化・α=2.2
import random, torch, numpy as np
from statistics import mean, pstdev
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
import metrics as M
from graphvae_core import GEncoder, EdgeMLP

# -------- reproducible -----------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
# ---------------------------------

class GVAE(torch.nn.Module):
    def __init__(self, dim, hid=64, z=32):
        super().__init__()
        self.enc = GEncoder(dim, hid, z)
        self.dec = EdgeMLP(z)
    def encode(self, x, e):
        mu, _ = self.enc(x, e)
        return mu

@torch.no_grad()
def sample(model, loader, k, alpha=2.5):
    """k 個のグラフを生成して (ref, gen) を返す"""
    ref, gen = [], []
    for d in loader:
        if len(gen) >= k:
            break
        z = model.encode(d.x, d.edge_index)
        p = model.dec(z).clamp_min(0.06)          # 低確率を底上げ
        mask = torch.rand_like(p) < p * alpha
        mask = torch.triu(mask, diagonal=1)       # 上三角だけ使う
        edges = torch.cat([mask.nonzero(),
                           mask.nonzero()[:, [1, 0]]]).t()  # 対称化
        gen.append(M.to_nx(edges, d.num_nodes))
        ref.append(M.to_nx(d.edge_index, d.num_nodes))
    return ref, gen

def main():
    # --------- データセット & モデル ----------
    ds = QM9(root="data/QM9")[:3000]
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = GVAE(ds.num_features)
    model.load_state_dict(torch.load("runs/graphvae_stable.pt",
                                     map_location="cpu"))
    model.eval()
    # -----------------------------------------

    valid, uniq_iso, mmd = [], [], []
    for _ in range(5):                             # 5 回平均
        ref, gen = sample(model, loader, 10000)
        valid.append(M.validity(gen))
        uniq_iso.append(M.uniqueness_iso(gen))     # ← 同型を除いた多様性
        mmd.append(M.degree_mmd(ref, gen))

    print(f"Validity        : {mean(valid):.3f} ± {pstdev(valid):.3f}")
    print(f"Uniqueness (iso): {mean(uniq_iso):.3f} ± {pstdev(uniq_iso):.3f}")
    print(f"Degree-MMD      : {mean(mmd):.3f} ± {pstdev(mmd):.3f}")

if __name__ == "__main__":
    main()
