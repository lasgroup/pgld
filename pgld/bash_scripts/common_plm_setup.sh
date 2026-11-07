#!/bin/bash
# Common setup for PLM experiments (ESM2, IgLM): loads cluster modules and
# activates the PLM conda environment (PGLD_PLM_CONDA_ENV, default pgld_plm).

source "${HOME}/pgld/.env.local"

# Load cluster modules (skip any left unset in .env.local)
[ -n "${PGLD_PYTHON_MODULE}" ] && module load ${PGLD_PYTHON_MODULE}
[ -n "${PGLD_STACK_MODULE}" ] && module load ${PGLD_STACK_MODULE}
[ -n "${PGLD_GCC_MODULE}" ] && module load ${PGLD_GCC_MODULE}
[ -n "${PGLD_CUDA_MODULE}" ] && module load ${PGLD_CUDA_MODULE}

# Activate conda environment
source ${PGLD_CONDA_BASE}/bin/activate ${PGLD_PLM_CONDA_ENV}

echo "PLM environment (${PGLD_PLM_CONDA_ENV}) loaded for user: ${PGLD_USER}"

# Disable Python output buffering so prints show up in SLURM logs immediately
export PYTHONUNBUFFERED=1
