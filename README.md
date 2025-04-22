生成予定の HOT‑VAE （**H**ierarchical, **O**ptimal‑transport‑regularised, **T**emporal VAE）の全体像を、4 週間後の研究室内発表までに仕上げることを目的に、研究動機・方法・評価・短期ロードマップを再整理しました。要点は「静的・単層グラフ中心の既存 VAE では表現しきれない“動的×階層”構造を、軽量な GNN × VAE フレームワークでモデル化し、Temporal ELBO ＋ (可選) OT 正則化で学習する」ことです。以下に詳細をまとめます。

## 1. 背景と狙い  

- **動的グラフの生成は未だ手薄**  
  Dyn‑VGAE などの先行研究は時間的依存を捉えつつも階層構造を扱わない citeturn1view0。  
- **階層表現は生成品質を高める鍵**  
  HG‑VAE は多層潜在変数で高次コンテキストを捕捉し動作生成に成功 citeturn2view0。  
- **OT 正則化は離散潜在の崩壊防止に有効**  
  OT‑VAE が示す通り、エントロピー正則化 OT はコードブック利用を均衡化し再構成性能を向上させる citeturn4view0。  
- **ベンチマークの整備が急務**  
  DGB や最近の統一ベンチマーク論文が多様ドメイン・評価設定の不足を指摘 citeturn5view0turn9view0。  

## 2. 研究課題  

1. **H1 — 階層潜在は動的グラフ生成性能（MMD, Rec@k）を≥5 %改善できるか**。  
2. **H2 — OT 正則化は学習の安定性と長期予測精度を高めるか**。  
3. **H3 — 軽量 VAE で Diffusion 系に匹敵する構造一貫性を実現できるか**（DiffuseSG 等の重モデル対比） citeturn10search2。  

## 3. HOT‑VAE の設計  

### 3.1 エンコーダ  
- **階層 GNN**：下位層＝時間 t のスナップショットを GraphSAGE/GCN で埋め込み、上位層＝各階層間の抽象ノードを Transformer リレーションで集約。  
- **時間モジュール**：GRU または TGAT‑style 時間エンコーディングで潜在を更新。  

### 3.2 潜在構造  
- **\(z^{(h,t)}\)**：階層 h, 時刻 t の局所潜在。  
- **\(z^{(H)}\)**：全時系列を統括するグローバル潜在（Graphite の低ランク近似思想を流用 citeturn11view0）。  

### 3.3 デコーダ  
- Iterative edge‑refinement（Graphite 系）で隣接行列を生成し、階層間拘束を付与。  
- ダイナミックロス：再構成誤差 ＋ KL ＋ 時間滑らかさ　＋ (任意) EOT 距離 \(W_\epsilon\)。  

### 3.4 学習目標  
\[
\mathcal{L}=\sum_{t}\Bigl[\underbrace{\mathbb{E}_{q_\phi}\!\left[\log p_\theta(G_t|z^{(\cdot,t)})\right]}_{\text{再構成}}-\beta D_\mathrm{KL}(q_\phi||p(z))\Bigr]
-\lambda_\mathrm{temp}\!\sum_t\|z^{(\cdot,t)}-z^{(\cdot,t-1)}\|_2^2
-\gamma\,W_\epsilon(q_\phi(z^{(\cdot,t)}),q_\phi(z^{(\cdot,t-1)}))
\]

## 4. 実験計画  

| フェーズ | 内容 | 主要リソース・指標 |
|---------|------|-------------------|
| **Data** | DGB 6データセット ＋ DyGraph 合成 ＋ 階層構造を持つ合成ツリー | サイズ、階層深さ |
| **実装** | PyTorch + PyG、ELBO & OT 実装、Wisteria HPC | GPU V100×4 |
| **Baselines** | Dyn‑VGAE citeturn1view0、DyVGRNN citeturn10search1、Graphite citeturn11view0 |
| **評価** | 静的 MMD、動的リンク予測 AUC、階層一致率、推論時間 |  |

## 5. 4 週間ロードマップ  

| 週 | 目標 | 主要 Deliverable |
|----|------|-----------------|
| **W‑4** | 既存コードベース整理・データ取得 | Git repo & dataset README |
| **W‑3** | Encoder/Decoder 最小実装・静的再構成実験 | 再構成 & KL 曲線 |
| **W‑2** | Temporal モジュール & OT 正則化実装 | 動的リンク予測結果 |
| **W‑1** | ハイパラ探索・アブレーション・スライド作成 | 5 分デモ＋発表スライド |

## 6. 期待される貢献とリスク  

- **貢献**  
  - 動的×階層グラフ生成における初の軽量 VAE ベンチマーク (動的 GNN 分野で求められている統一評価系への回答) citeturn9view0  
  - OT 正則化の動的グラフ生成への適用例を提供し、離散潜在崩壊問題を緩和 citeturn4view0  
  - 高速推論 (≤1 s/グラフ) により Diffusion 系の実運用課題を補完 citeturn10search2  

- **リスク & 対応**  
  - **データ階層アノテーション不足** → 合成階層グラフ生成スクリプトを併用。  
  - **OT 計算コスト** → Sinkhorn GPU 実装 + 近似カットオフ。  

---

### 参考にした主な文献  
(本文中に都度引用)

- Dyn‑VGAE — Mahdavi et al. 2019 citeturn1view0  
- HG‑VAE — Bourached et al. 2022 citeturn2view0  
- OT‑VAE — Bie et al. 2023 citeturn4view0  
- Dynamic Graph Benchmark (DGB) citeturn5view0  
- Unified Dynamic GNN Benchmark — Zhang 2024 citeturn9view0  
- Graphite — Grover et al. 2019 citeturn11view0  
- Dynamic Graph Representation Survey — Kazemi et al. 2020 citeturn12view0  
- Recent Dynamic GNN Survey — 2025 Springer citeturn10search1  
- DiffuseSG (scene‑graph diffusion) — 2024 citeturn10search2  
- HVAE 概説 — Activeloop Glossary citeturn13view0