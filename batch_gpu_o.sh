#!/bin/bash
#PJM -L rscgrp=share
#PJM -L gpu=1
#PJM -L elapse=02:00:00
#PJM -g jh210022a
#PJM -j

source /work/jh210022o/q25030/graph-vae-playground/gvae_env.sh

python src/train_graphvae_stable.py --epochs 40
