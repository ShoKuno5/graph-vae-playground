export PATH="/work/01/jh210022o/q25030/bin:$PATH"
export MAMBA_ROOT_PREFIX=/work/01/jh210022o/q25030/mamba
export MAMBA_PKGS_DIRS=$MAMBA_ROOT_PREFIX/pkgs
module load cuda/12.0
eval "$(micromamba shell hook --shell bash)"
micromamba activate gvae-cuda