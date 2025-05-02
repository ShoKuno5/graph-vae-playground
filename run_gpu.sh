#!/bin/bash
#PJM -L rscgrp=share              
#PJM -L gpu=1               
#PJM -L elapse=02:00:00
#PJM -L jobenv=singularity
#PJM -g jh210022a
#PJM -j                           

module load singularity/3.9.5 cuda/12.0

# -------- パス設定 --------
ROOT=/work/jh210022o/q25030
IMG=$ROOT/graph-vae-playground/images/hot_gvae_cuda.sif
DATA=$ROOT/data                     # QM9 を置く場所
CODE=$ROOT/graph-vae-playground     # ソースコード
mkdir -p "$DATA"

# -------- 実行 --------
singularity exec --nv --userns \
    -B "$DATA":/dataset \
    -B "$CODE":/workspace \
    "$IMG" \
    python /workspace/src/train_graphvae_stable.py \
           --epochs 40 \
           --data_root /dataset/QM9