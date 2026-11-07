#!/bin/bash
#SBATCH --job-name=lambda_sweep
#SBATCH --time=4:00:00
#SBATCH --mem-per-cpu=16G

# Unified lambda sweep for paper experiments (IUPAC, mean fitness objective)
# Uses same lambda values as TrpB sweeps for comparison across landscapes
# Usage: ./lambda_linear_sweep.sh <landscape> <num_templates> [--skip-existing|--overwrite]
#   landscape: gb1 | phoq | dhfr | mhc_flurry | riboswitch
#   num_templates: 1 | 16

LANDSCAPE=${1:-gb1}

# Load correct environment based on landscape
if [ "$LANDSCAPE" == "mhc_flurry" ]; then
    source "${HOME}/pgld/pgld/bash_scripts/common_mhc_setup.sh"
else
    source "${HOME}/pgld/pgld/bash_scripts/common_basic_setup.sh"
fi
source "${HOME}/pgld/pgld/bash_scripts/optimizer_exps/experiment_utils.sh"

NUM_TEMPLATES=${2:-1}
MODE_FLAG=${3:-}

case $MODE_FLAG in
    --skip-existing) CHECK_MODE="--skip" ;;
    --overwrite)     CHECK_MODE="--overwrite" ;;
    "")              CHECK_MODE="--fail" ;;
    *)
        echo "Unknown flag: $MODE_FLAG"
        exit 1
        ;;
esac

# Validate landscape
case $LANDSCAPE in
    gb1|phoq|dhfr|mhc_flurry|riboswitch) ;;
    *)
        echo "Unknown landscape: $LANDSCAPE"
        echo "Supported: gb1, phoq, dhfr, mhc_flurry, riboswitch"
        exit 1
        ;;
esac

OUTPUT_BASE="${PGLD_OUTPUT_DIR}/optimizer_exps/${LANDSCAPE}"

# Same lambda values as TrpB sweeps for comparison
LOG_RANGE=$(python3 -c "import numpy as np; print(' '.join(f'{x:.4g}' for x in np.logspace(np.log10(0.01), np.log10(0.1), 20)))")
LINEAR_01_02=$(python3 -c "import numpy as np; print(' '.join(f'{x:.4g}' for x in np.linspace(0.1, 0.2, 7)[1:-1]))")
LINEAR_02_05=$(python3 -c "import numpy as np; print(' '.join(f'{x:.4g}' for x in np.linspace(0.2, 0.5, 7)[1:-1]))")
LINEAR_05_10=$(python3 -c "import numpy as np; print(' '.join(f'{x:.4g}' for x in np.linspace(0.5, 1.0, 12)[1:-1]))")

# Riboswitch gets extra lambda values in 0.5-1.0 range
if [ "$LANDSCAPE" == "riboswitch" ]; then
    LAMBDA_VALUES=(0.0 ${LOG_RANGE} ${LINEAR_01_02} 0.2 ${LINEAR_02_05} 0.5 ${LINEAR_05_10} 1.0)
else
    LAMBDA_VALUES=(0.0 ${LOG_RANGE} ${LINEAR_01_02} 0.2 ${LINEAR_02_05} 0.5 1.0)
fi

# Self-submission: if not running inside SLURM, compute array size and submit
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    ARRAY_SIZE=${#LAMBDA_VALUES[@]}
    echo "Submitting $ARRAY_SIZE jobs for ${LANDSCAPE} t=${NUM_TEMPLATES} lambda sweep"
    echo "Lambda values: ${LAMBDA_VALUES[*]}"
    # MHC Flurry needs more time and memory
    if [ "$LANDSCAPE" == "mhc_flurry" ]; then
        sbatch --export=NONE --time=16:00:00 --mem-per-cpu=32G --array=0-$((ARRAY_SIZE-1)) "$0" "$LANDSCAPE" "$NUM_TEMPLATES" "$MODE_FLAG"
    else
        sbatch --export=NONE --array=0-$((ARRAY_SIZE-1)) "$0" "$LANDSCAPE" "$NUM_TEMPLATES" "$MODE_FLAG"
    fi
    exit 0
fi

# Landscape-specific settings (iters match existing sweep scripts)
case $LANDSCAPE in
    gb1|phoq)
        BATCH_SIZE=1024
        EVAL_BATCH_SIZE=1024
        BASE_ITERS=10000
        LR=0.02
        SEQUENCE_TYPE="peptide"
        POOLING_UNIQUENESS="peptide"
        EXTRA_ARGS=""
        ;;
    dhfr)
        BATCH_SIZE=1024
        EVAL_BATCH_SIZE=1024
        BASE_ITERS=10000
        LR=0.02
        SEQUENCE_TYPE=""  # DNA, no sequence_type needed
        POOLING_UNIQUENESS=""
        EXTRA_ARGS=""
        ;;
    mhc_flurry)
        BATCH_SIZE=2048
        EVAL_BATCH_SIZE=2048
        BASE_ITERS=5000
        LR=0.02
        SEQUENCE_TYPE="peptide"
        POOLING_UNIQUENESS="peptide"
        EXTRA_ARGS=""
        ;;
    riboswitch)
        BATCH_SIZE=1024
        EVAL_BATCH_SIZE=1024
        BASE_ITERS=10000
        LR=0.02
        SEQUENCE_TYPE=""  # DNA
        POOLING_UNIQUENESS=""
        EXTRA_ARGS="landscape.objectives=null ++landscape.objectives.ATP.threshold=6 ++landscape.objectives.ATP.invert=false"
        ;;
esac

MC_SAMPLES=5

# Scale iterations for higher template counts (except mhc_flurry which is already slow)
if [ "$NUM_TEMPLATES" -ge 16 ] && [ "$LANDSCAPE" != "mhc_flurry" ]; then
    ITERS=$((BASE_ITERS * 2))
else
    ITERS=$BASE_ITERS
fi

# Entropy type based on template count
if [ "$NUM_TEMPLATES" -eq 1 ]; then
    ENTROPY_TYPE="exact"
else
    ENTROPY_TYPE="marginal_gain"
fi

LAMBDA_VAL=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}
SEED=$((42 + SLURM_ARRAY_TASK_ID))
EXP_NAME="iupac_lambda_sweep_t${NUM_TEMPLATES}/l${LAMBDA_VAL}"

check_experiment_dir "${OUTPUT_BASE}/${EXP_NAME}" $CHECK_MODE || exit 0
echo "Running: landscape=${LANDSCAPE}, lambda=${LAMBDA_VAL}, templates=${NUM_TEMPLATES}, iters=${ITERS}"

# Build command with optional args
CMD="python ${PGLD_PKG_ROOT}/train.py \
    landscape=${LANDSCAPE} \
    ${EXTRA_ARGS} \
    policy=iupac \
    policy.num_templates=${NUM_TEMPLATES} \
    training.optimization.iters=${ITERS} \
    training.optimization.lr=${LR} \
    training.sampling.batch_size=${BATCH_SIZE} \
    training.sampling.mc_samples=${MC_SAMPLES} \
    training.entropy.lambda_=${LAMBDA_VAL} \
    training.entropy.type=${ENTROPY_TYPE} \
    training.entropy.baseline_per_template=true \
    training.pooling.enabled=true \
    training.pooling.queries_per_batch=256 \
    training.evaluation.extra_eval=true \
    training.evaluation.eval_batch_size=${EVAL_BATCH_SIZE} \
    training.evaluation.checkpointing=false \
    training.reward_type=average \
    training.reward_aggregation_unit=template \
    seed=${SEED} \
    hydra.run.dir=${OUTPUT_BASE}/${EXP_NAME}"

# Add sequence_type if needed (peptide landscapes)
if [ -n "$SEQUENCE_TYPE" ]; then
    CMD="$CMD training.entropy.sequence_type=${SEQUENCE_TYPE}"
fi

# Add pooling uniqueness if needed
if [ -n "$POOLING_UNIQUENESS" ]; then
    CMD="$CMD training.pooling.uniqueness=${POOLING_UNIQUENESS}"
fi

eval $CMD
