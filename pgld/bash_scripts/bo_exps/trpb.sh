#!/bin/bash

#SBATCH -n 1
#SBATCH --time=8:00:00
#SBATCH --gpus=rtx_2080:1
#SBATCH --mem-per-cpu=16G
#SBATCH --tmp=4000
#SBATCH --job-name=bo_trpb
#SBATCH --array=0-4

source "${HOME}/pgld/pgld/bash_scripts/common_basic_setup.sh"

TEMPLATE=50
SEEDS=(52 53 54 55 56)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

python pgld/train_bo.py \
    landscape=trpb \
    \
    hydra.run.dir=${PGLD_OUTPUT_DIR}/bo_paper_exp/trpb/seed_${SEED} \
    \
    seed=${SEED} \
    policy.num_templates=${TEMPLATE} \
    policy=iupac \
    training.optimization.iters=8000 \
    training.sampling.mc_samples=5 \
    training.sampling.batch_size=5000 \
    training.reward_type=hits \
    training.reward_aggregation_unit=template \
    training.per_sequence_rewards=false \
    training.entropy.lambda_=0.0 \
    training.entropy.sequence_type=peptide \
    training.entropy.type=marginal_gain \
    training.optimization.lr=0.02 \
    training.evaluation.extra_eval=false \
    training.evaluation.checkpointing=false \
    training.pooling.uniqueness=peptide \
    training.pooling.enabled=true \
    training.pooling.gpu_accelerated=true \
    training.pooling.queries_per_batch=100 \
    \
    bo.bo_iters=5 \
    bo.generate_per_round=5000 \
    bo.oracle_training.n_epochs=200 \
    bo.oracle_training.lr=1e-3 \
    bo.oracle_training.batch_size=64 \
    bo.oracle_training.patience=50 \
    bo.oracle_training.n_ensembles=1 \
    bo.with_replacement=False \
    bo.baseline=all \
    bo.save_samples=True
