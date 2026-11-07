#!/bin/bash
#SBATCH --job-name=mhc_sweep
#SBATCH --time=8:00:00
#SBATCH --mem-per-cpu=32G

# MHC class I binding sweep for optimizer experiments
# Sweeps over: template counts for HLA-A*02:01
# Usage: ./mhc_sweep.sh <experiment_type> [--skip-existing|--overwrite]
#   experiment_type: "template" | "arbitrary"
#   --skip-existing: skip experiments that already have results
#   --overwrite: wipe existing results and rerun
# The script self-submits with the correct array size.

source "${HOME}/pgld/pgld/bash_scripts/common_mhc_setup.sh"
source "${HOME}/pgld/pgld/bash_scripts/optimizer_exps/experiment_utils.sh"

EXPERIMENT_TYPE=${1:-template}
MODE_FLAG=${2:-}

case $MODE_FLAG in
    --skip-existing) CHECK_MODE="--skip" ;;
    --overwrite)     CHECK_MODE="--overwrite" ;;
    "")              CHECK_MODE="--fail" ;;
    *)
        echo "Unknown flag: $MODE_FLAG"
        exit 1
        ;;
esac

OUTPUT_BASE="${PGLD_OUTPUT_DIR}/optimizer_exps/mhc_flurry"

# Configuration
TEMPLATE_COUNTS=(1 2 4 8 16 32)
ALLELE="HLA-A*02:01"

# Self-submission: if not running inside SLURM, compute array size and submit
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    case "$EXPERIMENT_TYPE" in
        template)  ARRAY_SIZE=${#TEMPLATE_COUNTS[@]} ;;
        arbitrary) ARRAY_SIZE=1 ;;
        *)
            echo "Unknown experiment type: $EXPERIMENT_TYPE"
            echo "Usage: ./mhc_sweep.sh <template|arbitrary>"
            exit 1
            ;;
    esac
    echo "Submitting $ARRAY_SIZE jobs for $EXPERIMENT_TYPE"
    sbatch --export=NONE --array=0-$((ARRAY_SIZE-1)) "$0" "$EXPERIMENT_TYPE" "$MODE_FLAG"
    exit 0
fi

# MHC-specific settings (larger batches for peptide landscapes)
# Threshold is fixed to 0.995 in conf/landscape/mhc_flurry.yaml
BATCH_SIZE=2048
EVAL_BATCH_SIZE=2048
ITERS=5000
MC_SAMPLES=5

run_experiment() {
    local num_templates=$1
    local lambda_val=$2
    local exp_name=$3
    local policy_type=${4:-iupac}
    local reward_type=${5:-hits}

    # Use exact entropy for single template, marginal_gain for multiple
    if [ "$num_templates" -eq 1 ]; then
        local entropy_type="exact"
    else
        local entropy_type="marginal_gain"
    fi

    python ${PGLD_PKG_ROOT}/train.py \
        landscape=mhc_flurry \
        policy=${policy_type} \
        policy.num_templates=${num_templates} \
        training.optimization.iters=${ITERS} \
        training.optimization.lr=0.02 \
        training.sampling.batch_size=${BATCH_SIZE} \
        training.sampling.mc_samples=${MC_SAMPLES} \
        training.entropy.lambda_=${lambda_val} \
        training.entropy.type="${entropy_type}" \
        training.entropy.sequence_type="peptide" \
        training.entropy.baseline_per_template=true \
        training.pooling.enabled=true \
        training.pooling.uniqueness="peptide" \
        training.pooling.queries_per_batch=256 \
        training.evaluation.extra_eval=true \
        training.evaluation.eval_batch_size=${EVAL_BATCH_SIZE} \
        training.evaluation.checkpointing=false \
        training.reward_type="${reward_type}" \
        training.reward_aggregation_unit=template \
        hydra.run.dir="${OUTPUT_BASE}/${exp_name}"
}

if [ "$EXPERIMENT_TYPE" == "template" ]; then
    # Template sweep: vary num_templates (lambda=0)
    NUM_TEMPLATE=${TEMPLATE_COUNTS[$SLURM_ARRAY_TASK_ID]}
    EXP_NAME="template_sweep/t${NUM_TEMPLATE}"
    check_experiment_dir "${OUTPUT_BASE}/${EXP_NAME}" $CHECK_MODE || exit 0
    echo "Running: templates=${NUM_TEMPLATE}"
    run_experiment $NUM_TEMPLATE 0.0 $EXP_NAME

elif [ "$EXPERIMENT_TYPE" == "arbitrary" ]; then
    # Arbitrary policy: 1 template (lambda=0)
    EXP_NAME="arbitrary/t1"
    check_experiment_dir "${OUTPUT_BASE}/${EXP_NAME}" $CHECK_MODE || exit 0
    echo "Running: policy=arbitrary, templates=1"
    run_experiment 1 0.0 $EXP_NAME arbitrary

else
    echo "Unknown experiment type: $EXPERIMENT_TYPE"
    echo "Usage: ./mhc_sweep.sh <template|arbitrary>"
    exit 1
fi
