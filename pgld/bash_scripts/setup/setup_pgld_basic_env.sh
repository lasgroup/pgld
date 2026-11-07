#!/bin/bash
#SBATCH --job-name=setup_pgld_basic
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1

# Setup pgld_basic conda environment
# FIRST: conda create -n pgld_basic python=3.9 -y
# THEN: sbatch setup_pgld_basic_env.sh
# NOTE: Requires internet access for pip downloads

set -e

# Initialize modules (cluster-specific, may need to modify)
if [ -f /cluster/apps/local/env2lmod.sh ]; then
    source /cluster/apps/local/env2lmod.sh
fi
source "${HOME}/pgld/.env.local"

# Load CUDA and GCC modules - update PGLD_*_MODULE in .env.local for your cluster
module load ${PGLD_STACK_MODULE} ${PGLD_GCC_MODULE} ${PGLD_CUDA_MODULE}

source ${PGLD_CONDA_BASE}/etc/profile.d/conda.sh
conda activate pgld_basic

echo "Python: $(which python)"
nvidia-smi

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install pytorch-lightning hydra-core hydra-submitit-launcher omegaconf wandb
pip install numpy pandas scipy scikit-learn matplotlib seaborn
pip install mhcflurry tensorflow
pip install gReLU==1.0.2
pip install openpyxl tqdm editdistance biopython pyyaml
pip install logomaker numba

mhcflurry-downloads fetch

echo "Done!"
