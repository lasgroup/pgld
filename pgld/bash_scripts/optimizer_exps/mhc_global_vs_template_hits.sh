#!/bin/bash
#SBATCH --job-name=mhc_hits_lr_sweep
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --cpus-per-task=8

# MHC flurry LR and reward_aggregation_unit sweep (HITS reward)
# - 5 learning rates
# - 32 templates
# - Sweeps reward_aggregation_unit: template vs batch
# - No per-template baseline (baseline_per_template=false)
# Total: 5 LRs x 2 reward units = 10 experiments

source "${HOME}/pgld/pgld/bash_scripts/common_mhc_setup.sh"
source "${HOME}/pgld/pgld/bash_scripts/optimizer_exps/experiment_utils.sh"

MODE_FLAG=${1:-}

case $MODE_FLAG in
    --skip-existing) CHECK_MODE="--skip" ;;
    --overwrite)     CHECK_MODE="--overwrite" ;;
    "")              CHECK_MODE="--fail" ;;
    *)
        echo "Unknown flag: $MODE_FLAG"
        exit 1
        ;;
esac

OUTPUT_BASE="${PGLD_OUTPUT_DIR}/optimizer_exps/mhc_flurry_hits_lr_sweep"

# Configuration
LR_VALUES=(0.005 0.01 0.02 0.05 0.15)
REWARD_UNITS=(template batch)
NUM_TEMPLATES=32
LAMBDA=0.0

# Total experiments = 5 LRs x 2 reward units = 10
TOTAL_EXPERIMENTS=$((${#LR_VALUES[@]} * ${#REWARD_UNITS[@]}))

# Self-submission
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "Submitting $TOTAL_EXPERIMENTS jobs for LR x reward_unit sweep (hits)"
    sbatch --export=NONE --array=0-$((TOTAL_EXPERIMENTS-1)) "$0" "$MODE_FLAG"
    exit 0
fi

# Decode array index into (lr_idx, reward_unit_idx)
LR_IDX=$((SLURM_ARRAY_TASK_ID / ${#REWARD_UNITS[@]}))
REWARD_UNIT_IDX=$((SLURM_ARRAY_TASK_ID % ${#REWARD_UNITS[@]}))

LR=${LR_VALUES[$LR_IDX]}
REWARD_UNIT=${REWARD_UNITS[$REWARD_UNIT_IDX]}

# MHC-specific settings (same as lambda sweep)
BATCH_SIZE=2048
EVAL_BATCH_SIZE=2048
ITERS=5000
MC_SAMPLES=5

EXP_NAME="t${NUM_TEMPLATES}_lr${LR}_${REWARD_UNIT}"

# Set entropy type and baseline based on reward aggregation unit
# - template: marginal_gain with per-template baseline
# - batch: full_mixture with global baseline
if [ "$REWARD_UNIT" == "template" ]; then
    ENTROPY_TYPE="marginal_gain"
    BASELINE_PER_TEMPLATE=true
else
    ENTROPY_TYPE="full_mixture"
    BASELINE_PER_TEMPLATE=false
fi

check_experiment_dir "${OUTPUT_BASE}/${EXP_NAME}" $CHECK_MODE || exit 0

echo "Running: templates=${NUM_TEMPLATES}, lr=${LR}, reward_unit=${REWARD_UNIT}, entropy=${ENTROPY_TYPE}, lambda=${LAMBDA}, reward_type=hits"

python ${PGLD_PKG_ROOT}/train.py \
    landscape=mhc_flurry \
    policy=iupac \
    policy.num_templates=${NUM_TEMPLATES} \
    training.optimization.iters=${ITERS} \
    training.optimization.lr=${LR} \
    training.sampling.batch_size=${BATCH_SIZE} \
    training.sampling.mc_samples=${MC_SAMPLES} \
    training.entropy.lambda_=${LAMBDA} \
    training.entropy.type=${ENTROPY_TYPE} \
    training.entropy.sequence_type=peptide \
    training.entropy.baseline_per_template=${BASELINE_PER_TEMPLATE} \
    training.pooling.enabled=true \
    training.pooling.uniqueness=peptide \
    training.pooling.queries_per_batch=null \
    training.evaluation.extra_eval=true \
    training.evaluation.eval_batch_size=${EVAL_BATCH_SIZE} \
    training.evaluation.checkpointing=false \
    training.reward_type=hits \
    training.reward_aggregation_unit=${REWARD_UNIT} \
    hydra.run.dir="${OUTPUT_BASE}/${EXP_NAME}"
