#!/bin/bash
#SBATCH --job-name=tf_sweep
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=20G

# TF-binding sweep for optimizer experiments
# Sweeps over: TF objectives and template counts
# Usage: ./tf_binding_sweep.sh <experiment_type> [--skip-existing|--overwrite]
#   experiment_type: "template" | "arbitrary"
#   --skip-existing: skip experiments that already have results
#   --overwrite: wipe existing results and rerun
# The script self-submits with the correct array size.

source "${HOME}/pgld/pgld/bash_scripts/common_basic_setup.sh"
source "${HOME}/pgld/pgld/bash_scripts/optimizer_exps/experiment_utils.sh"

EXPERIMENT_TYPE=${1:-template}
MODE_FLAG=${2:-}

# Set check mode based on flag
case $MODE_FLAG in
    --skip-existing) CHECK_MODE="--skip" ;;
    --overwrite)     CHECK_MODE="--overwrite" ;;
    "")              CHECK_MODE="--fail" ;;
    *)
        echo "Unknown flag: $MODE_FLAG"
        echo "Use --skip-existing or --overwrite"
        exit 1
        ;;
esac

OUTPUT_BASE="${PGLD_OUTPUT_DIR}/optimizer_exps/tf_binding"

# Get all TF names with _REF_R1 pattern
TF_DATA_DIR="${PGLD_PKG_ROOT}/flexs_wrapper/landscapes/data/tf_binding"
TF_NAMES=($(ls ${TF_DATA_DIR}/*_REF_R1_8mers.txt 2>/dev/null | xargs -I{} basename {} _8mers.txt))

if [ ${#TF_NAMES[@]} -eq 0 ]; then
    echo "ERROR: No TF files with _REF_R1 pattern found in ${TF_DATA_DIR}"
    exit 1
fi

echo "Found ${#TF_NAMES[@]} TF objectives with _REF_R1 pattern"

# Configuration
TEMPLATE_COUNTS=(1 2 4 8 16 32)

# Self-submission logic: when run directly (not inside SLURM), this script computes
# the required array size based on experiment type and number of TF landscapes,
# then submits itself as an array job. When SLURM runs each array task, this block
# is skipped and the actual experiment code below executes.
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    NUM_TFS=${#TF_NAMES[@]}
    case "$EXPERIMENT_TYPE" in
        template)  ARRAY_SIZE=$((NUM_TFS * ${#TEMPLATE_COUNTS[@]})) ;;
        arbitrary) ARRAY_SIZE=$NUM_TFS ;;
        *)
            echo "Unknown experiment type: $EXPERIMENT_TYPE"
            echo "Usage: ./tf_binding_sweep.sh <template|arbitrary>"
            exit 1
            ;;
    esac
    echo "Submitting $ARRAY_SIZE jobs for $EXPERIMENT_TYPE ($NUM_TFS TFs)"
    sbatch --export=NONE --array=0-$((ARRAY_SIZE-1)) "$0" "$EXPERIMENT_TYPE" "$MODE_FLAG"
    exit 0
fi

# Common settings
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
ITERS=25000
MC_SAMPLES=5
TOP_PROP=0.05

run_experiment() {
    local tf_name=$1
    local num_templates=$2
    local lambda_val=$3
    local exp_name=$4
    local policy_type=${5:-iupac}
    local reward_type=${6:-hits}

    # Use exact entropy for single template, marginal_gain for multiple
    if [ "$num_templates" -eq 1 ]; then
        local entropy_type="exact"
    else
        local entropy_type="marginal_gain"
    fi

    python ${PGLD_PKG_ROOT}/train.py \
        landscape=tf_binding \
        landscape.objectives=null \
        ++landscape.objectives.${tf_name}.top_prop=${TOP_PROP} \
        ++landscape.objectives.${tf_name}.invert=false \
        policy=${policy_type} \
        policy.num_templates=${num_templates} \
        training.optimization.iters=${ITERS} \
        training.optimization.lr=0.02 \
        training.sampling.batch_size=${BATCH_SIZE} \
        training.sampling.mc_samples=${MC_SAMPLES} \
        training.entropy.lambda_=${lambda_val} \
        training.entropy.type="${entropy_type}" \
        training.entropy.baseline_per_template=true \
        training.pooling.enabled=true \
        training.pooling.queries_per_batch=null \
        training.evaluation.extra_eval=true \
        training.evaluation.eval_batch_size=${EVAL_BATCH_SIZE} \
        training.evaluation.checkpointing=false \
        training.reward_type="${reward_type}" \
        training.reward_aggregation_unit=template \
        hydra.run.dir="${OUTPUT_BASE}/${exp_name}"
}

if [ "$EXPERIMENT_TYPE" == "template" ]; then
    # Template sweep: vary num_templates for each TF (lambda=0)
    # Array job index maps to: TF_index * len(templates) + template_index
    NUM_TFS=${#TF_NAMES[@]}
    NUM_TEMPLATES=${#TEMPLATE_COUNTS[@]}

    TF_IDX=$((SLURM_ARRAY_TASK_ID / NUM_TEMPLATES))
    TEMPLATE_IDX=$((SLURM_ARRAY_TASK_ID % NUM_TEMPLATES))

    TF_NAME=${TF_NAMES[$TF_IDX]}
    NUM_TEMPLATE=${TEMPLATE_COUNTS[$TEMPLATE_IDX]}

    EXP_NAME="template_sweep/${TF_NAME}_t${NUM_TEMPLATE}"
    check_experiment_dir "${OUTPUT_BASE}/${EXP_NAME}" $CHECK_MODE || exit 0
    echo "Running: TF=${TF_NAME}, templates=${NUM_TEMPLATE}"
    run_experiment $TF_NAME $NUM_TEMPLATE 0.0 $EXP_NAME

elif [ "$EXPERIMENT_TYPE" == "arbitrary" ]; then
    # Arbitrary policy sweep: 1 template per TF (lambda=0)
    TF_NAME=${TF_NAMES[$SLURM_ARRAY_TASK_ID]}
    EXP_NAME="arbitrary/${TF_NAME}_t1"
    check_experiment_dir "${OUTPUT_BASE}/${EXP_NAME}" $CHECK_MODE || exit 0
    echo "Running: TF=${TF_NAME}, policy=arbitrary, templates=1"
    run_experiment $TF_NAME 1 0.0 $EXP_NAME arbitrary

else
    echo "Unknown experiment type: $EXPERIMENT_TYPE"
    echo "Usage: ./tf_binding_sweep.sh <template|arbitrary>"
    exit 1
fi
