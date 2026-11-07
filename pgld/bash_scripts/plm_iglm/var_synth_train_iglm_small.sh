#!/bin/bash
#SBATCH --job-name=var_synth_iglm
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16g

# Usage: sbatch var_synth_train_iglm_small.sh [nseeds] [nsamples]
#   $1: number of seeds (default: 3)
#   $2: number of samples (default: 10000)
#
# Examples:
#   sbatch var_synth_train_iglm_small.sh 1              # 1 seed, 10000 samples
#   sbatch var_synth_train_iglm_small.sh 3 50000        # 3 seeds, 50000 samples

NSEEDS=${1:-3}
NSAMPLES=${2:-10000}

# Load environment variables
source "${HOME}/pgld/.env.local"

# Update target_samples path in config before running
CONFIG_PATH="${PGLD_PROJECT_ROOT}/var_synth/configs/iglm_small.cfg"

# Activate var-synth conda environment
source ${PGLD_CONDA_BASE}/bin/activate var-synth

# Disable Python output buffering so prints show up in SLURM logs immediately
export PYTHONUNBUFFERED=1

echo "Running variational synthesis training (IgLM Small data)"
echo "  Config: ${CONFIG_PATH}"
echo "  Seeds: ${NSEEDS}"
echo "  Samples: ${NSAMPLES}"

cd ${HOME}/pgld/variational-synthesis

python ${PGLD_PROJECT_ROOT}/var_synth/run_var_synth.py \
    ${CONFIG_PATH} \
    --nsamples ${NSAMPLES} \
    --nseeds ${NSEEDS}
