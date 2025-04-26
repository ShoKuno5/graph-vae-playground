# eval.py
import argparse, torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import VGAE
from baseline_vgae import Encoder           # 既存クラスを再利用
import metrics as M

@torch.no_grad()
def sample_graphs(model, data_loader, device, num_graphs=100):
    model.eval()
    gen_graphs = []
    ref_graphs = []
    for i, data in enumerate(data_loader):
        if len(gen_graphs) >= num_graphs: break
        data = data.to(device)

        # --- encode  & sample ---
        z = model.encode(data.x, data.edge_index)
        # decode: inner-product decoder（sigmoid>0.5）
        adj_logits = (z @ z.t()).sigmoid()
        edge_index = (adj_logits > 0.5).nonzero(as_tuple=False).t()

        g_gen = M.to_nx(edge_index, data.num_nodes)   # generated
        g_ref = M.to_nx(data.edge_index, data.num_nodes)  # ground truth

        gen_graphs.append(g_gen)
        ref_graphs.append(g_ref)

    return ref_graphs, gen_graphs

def main(args):
    device = torch.device("cpu")   # Mac CPU 前提
    ds = QM9(root="data/QM9")[:args.sample]   # 小さく切る
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    model = VGAE(Encoder(ds.num_features, 64, 32))
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model = model.to(device)

    ref, gen = sample_graphs(model, loader, device, num_graphs=args.sample)

    print(f"Validity   : {M.validity(gen):.3f}")
    print(f"Uniqueness : {M.uniqueness(gen):.3f}")
    print(f"Degree-MMD : {M.degree_mmd(ref, gen):.3f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path to model.pt")
    p.add_argument("--sample", type=int, default=100)
    args = p.parse_args()
    main(args)
