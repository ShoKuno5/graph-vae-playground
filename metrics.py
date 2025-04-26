# metrics.py
import networkx as nx
import numpy as np
from scipy.stats import wasserstein_distance

# ---------- util ----------
def to_nx(edge_index, num_nodes):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    for src, dst in edge_index.t().tolist():
        if src != dst:          # ignore self-loops
            G.add_edge(src, dst)
    return G

# ---------- metrics ----------
def validity(graphs):
    """connected かつ self-loop 0 のグラフを valid とみなす"""
    valids = [
        nx.is_connected(g) and nx.number_of_selfloops(g) == 0
        for g in graphs
    ]
    return sum(valids) / len(graphs)


def uniqueness(graphs):
    """edge list を tuple 化してユニーク判定"""
    canon = [tuple(sorted(g.edges())) for g in graphs]
    return len(set(canon)) / len(graphs)

def degree_mmd(graphs_ref, graphs_gen, max_deg=10):
    def avg_hist(graphs):
        h_sum = np.zeros(max_deg + 1)
        cnt   = 0
        for g in graphs:
            degs = np.array([d for _, d in g.degree()])
            if degs.size == 0:   # 空グラフ→スキップ
                continue
            h, _ = np.histogram(degs, bins=range(max_deg + 2))
            if h.sum() == 0:
                continue         # すべて次数 0 → 情報なし
            h_sum += h / h.sum()
            cnt   += 1
        return h_sum / max(cnt, 1)   # cnt==0 ならゼロベクトル
    X = avg_hist(graphs_ref)
    Y = avg_hist(graphs_gen)
    return float(np.abs(X - Y).sum())

