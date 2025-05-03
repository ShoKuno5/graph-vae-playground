#!/bin/bash
#PJM -L rscgrp=short-a
#PJM -L node=1
#PJM -L elapse=02:00:00
#PJM -L jobenv=singularity
#PJM -g jh210022a
#PJM -j

module load singularity/3.9.5 cuda/12.0

ROOT=/work/jh210022o/q25030
IMG=$ROOT/graph-vae-playground/images/hot_gvae_cuda.sif
DATA=$ROOT/data
CODE=$ROOT/graph-vae-playground
mkdir -p "$DATA" "$CODE/runs"
cd "$CODE"

# ---------- 通信設定 ----------
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=ib0,eth0
export GLOO_SOCKET_IFNAME=ib0,eth0
export OMP_NUM_THREADS=8
# ------------------------------

singularity exec --nv --pwd /workspace \
    -B "$DATA":/dataset \
    -B "$CODE":/workspace \
    "$IMG" \
    python src/train_graphvae_ddp.py \
           --epochs 80 \
           --data_root /dataset/QM9
