#!/bin/bash
# Common setup script for MHC experiments (uses PGLD_MHC_CONDA_ENV, default pgld_basic)

source "${HOME}/pgld/.env.local"

# Load cluster modules - update PGLD_*_MODULE in .env.local for your cluster
module load ${PGLD_PYTHON_MODULE}
module load ${PGLD_STACK_MODULE} ${PGLD_GCC_MODULE}
module load ${PGLD_CUDA_MODULE}

# Activate pgld_basic conda environment directly
MHC_ENV=${PGLD_MHC_CONDA_ENV:-pgld_basic}
source ${PGLD_CONDA_BASE}/bin/activate ${MHC_ENV}

# Load CUDA module and set XLA flags to fix TensorFlow libdevice issue
module load cuda 2>/dev/null || true
if [ -n "$CUDA_HOME" ]; then
    export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CUDA_HOME"
fi
export TF_XLA_FLAGS="--tf_xla_auto_jit=0"
