## HOT‑VAE: Hierarchical **O**T‑regularised **T**emporal Variational Autoencoder

> **目的** — 静的・単層グラフ中心だった既存 VAE 系列を拡張し、**動的 × 階層グラフ**を高速・高品質に生成する。軽量 GNN×VAE フレームワークに **OT 正則化**を組み込み、拡散モデル系より桁違いに高速な推論を実現する。

---

### 1. 背景と課題意識
- **動的グラフ生成の空白**  
    Dyn‑VGAE が時間依存を扱うものの階層性を無視。
- **階層表現の効果**  
    HG‑VAE は多段潜在で長期動作を安定生成。
- **OT 正則化の有望性**  
    Coupled/OT‑VAE 系は潜在崩壊を抑制し生成品質を改善。
- **評価基盤の不足**  
    Dynamic Graph Benchmark (DGB) などが多様データを整備し始めたが網羅性が不十分。最新サーベイも同問題を指摘。
- **重い拡散モデルへの対抗**  
    Scene‑Graph‑Guided Diffusion（DiffuseSG 等）は高画質だが推論が遅い。軽量 VAE 系による代替を探る。

---

### 2. 研究課題
| ID | 仮説 | 検証指標 |
|----|------|----------|
| **H1** | 階層潜在導入で MMD / Rec@k ≥ **+5 %** | 静的 MMD, 動的リンク予測 AUC |
| **H2** | OT 正則化で学習安定 & 長期予測向上 | KL 収束曲線, 長期再構成誤差 |
| **H3** | 軽量 VAE で Diffusion 系並の構造一貫性 | 一貫性スコア, 生成速度 |

---

### 3. 手法概要

#### 3.1 エンコーダ
- **階層 GNN**：下位層 — GraphSAGE / GCN、上位層 — Transformer で抽象ノード集約  
- **時間処理**：GRU または TGN‑style メモリ

#### 3.2 潜在構造
\[
z^{(h,t)},\; z^{(H)} \qquad (h: \text{階層},\; t: \text{時刻})
\]
- 低層 \(z^{(h,t)}\)：局所トポロジ
- 高層 \(z^{(H)}\)：系列全体の文脈（Graphite の低ランク近似思想を応用）

#### 3.3 デコーダ
- Graphite 型 *iterative refinement* で隣接行列生成  
- ロス関数  
    \[
    \mathcal{L} = \text{Recon} - \beta\,\text{KL} - \lambda_\text{temp}\Vert z_t - z_{t-1}\Vert^2
                             - \gamma\,W_\varepsilon(z_t, z_{t-1})
    \]

#### 3.4 OT 実装
GPU Sinkhorn (PyTorchOT / POT) を利用し計算コストを抑制。

---

### 4. 実験計画

| ステージ | 内容 | データ/リソース |
|----------|------|-----------------|
| **Data** | DGB 6種 + 合成階層ツリー | 〜5 M edges |
| **実装** | PyTorch + PyG | Wisteria (V100×4) |
| **Baseline** | Dyn‑VGAE / DyVGRNN / Graphite |
| **評価** | MMD, AUC, 一貫性, 推論時間 |  |

---

### 5. 4 週間ロードマップ

| 週 | ゴール | 主なアウトプット |
|----|--------|-----------------|
| **W‑4** | コード骨格 & データ取得 | `repo/` と dataset README |
| **W‑3** | Encoder/Decoder 試作 | 再構成 & KL 曲線 |
| **W‑2** | Temporal & OT 実装 | 動的リンク予測リザルト |
| **W‑1** | HP 探索・アブレーション | 5 分デモ + 発表スライド |

---

### 6. 期待される貢献
1. **動的×階層グラフ生成の初ベンチマーク** — DGB 拡張と共に公開。  
2. **OT 正則化の新応用** — 離散潜在崩壊を緩和し安定学習。
3. **高速推論** — Diffusion 系より ≥10× 速い推論で実運用可能域へ。  

### 7. リスク & 対処
| リスク | 対策 |
|--------|------|
| 階層アノテ不足 | 合成ツールで擬似データ生成 |
| OT 計算負荷 | GPU Sinkhorn ＋ 早期収束条件 |
| ハイパラ爆発 | Optuna で自動探索／Early‑stop |

---

### 8. 参考文献
1. Dyn‑VGAE (2019)
2. Graphite (2019)
3. HG‑VAE (2022)
4. Coupled/OT‑VAE (2023)
5. Dynamic Graph Benchmark (2022)
6. Dynamic GNN Survey (2024)
7. Scene‑Graph Diffusion (2023)
8. Temporal Graph Networks (2020)
9. GraphSAGE (2017)
10. PyTorchOT (2019)
11. DyVGRNN (2023)

---

> **ライセンス / 再利用**  
> この README は研究発表目的で公開される予定の HOT‑VAE プロジェクトに付随する文書です。引用・再利用時は各論文・コードのライセンス条件をご確認ください。