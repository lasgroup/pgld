#!/bin/bash
#SBATCH --job-name=setup_pgld_plm
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1

# Setup script for pgld_plm conda environment
# FIRST create the env: conda create -n pgld_plm python=3.10 -y
# THEN run: sbatch setup_pgld_plm_env.sh
# NOTE: Requires internet access for pip downloads

# Clear any inherited conda state
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PATH="/usr/bin:/bin:/usr/sbin:/sbin"

# Initialize modules (cluster-specific, may need to modify)
if [ -f /cluster/apps/local/env2lmod.sh ]; then
    source /cluster/apps/local/env2lmod.sh
fi

# Load environment variables
if [ -f "${HOME}/pgld/.env.local" ]; then
    source "${HOME}/pgld/.env.local"
else
    echo "ERROR: ${HOME}/pgld/.env.local not found."
    exit 1
fi

# Load CUDA and GCC modules (required for flash-attn compilation)
# Update PGLD_*_MODULE in .env.local to match your cluster
module load ${PGLD_STACK_MODULE} ${PGLD_GCC_MODULE}
module load ${PGLD_CUDA_MODULE}

# Initialize conda fresh
source ${PGLD_CONDA_BASE}/etc/profile.d/conda.sh

# Activate the target environment
conda activate ${PGLD_PLM_CONDA_ENV}

# Verify we're in the right environment
echo "Active conda env: $CONDA_DEFAULT_ENV"
if [ "$CONDA_DEFAULT_ENV" != "$PGLD_PLM_CONDA_ENV" ]; then
    echo "ERROR: Failed to activate $PGLD_PLM_CONDA_ENV"
    echo "Did you create it first? Run: conda create -n $PGLD_PLM_CONDA_ENV python=3.10 -y"
    exit 1
fi

echo "Setting up pgld_plm environment..."
echo "Python: $(which python)"
nvidia-smi

# Install ninja first (speeds up flash-attn compilation significantly)
pip install ninja

# Install PyTorch with CUDA
pip3 install torch torchvision

# Install flash-attn (requires GPU, takes ~30min)
MAX_JOBS=4 pip install flash-attn --no-build-isolation

# Install ESM-efficient
pip install esm-efficient safetensors

# Install core ML packages
pip install pytorch-lightning hydra-core omegaconf wandb transformers accelerate

# Install IgLM
pip install iglm

# Install scientific packages
pip install numpy pandas scipy scikit-learn matplotlib seaborn

# Install utilities
pip install openpyxl tqdm editdistance biopython pyyaml

# Visualization
pip install logomaker numba

# Fix pyarrow if needed
pip install --force-reinstall --no-deps pyarrow

echo "Done! pgld_plm environment setup complete."
