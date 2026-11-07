#!/bin/bash
#SBATCH --job-name=esm2_gibbs_1M
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:8

# Usage: ./esm2_gibbs_sample_1M.sh [MODEL_SIZE] [SEED] [MAX_EDIT_DISTANCE]
#   MODEL_SIZE: 8m (default), 35m, or 150m
#   SEED: master seed (default: 42). Each GPU gets seed + GPU_ID*1000
#   MAX_EDIT_DISTANCE: max edit distance from wild-type (default: 6)

MODEL_SIZE="${1:-8m}"
BASE_SEED="${2:-42}"
MAX_EDIT_DISTANCE="${3:-6}"
if [[ ! "$MODEL_SIZE" =~ ^(8m|35m|150m)$ ]]; then
    echo "Error: MODEL_SIZE must be 8m, 35m, or 150m (got: $MODEL_SIZE)"
    exit 1
fi

source "${HOME}/pgld/pgld/bash_scripts/common_plm_setup.sh"

export PYTHONUNBUFFERED=1

# ESM2 Gibbs sampling - 1M sequences across 8 GPUs
# Each GPU: 125k samples with batch_size=1024, save_every=100
# 125k / 1024 = 122 saves -> 12200 steps
#
# TEST RUN (1k sequences) - uncomment to use:
# NUM_GPUS=4; SAMPLES_PER_GPU=256; TOTAL_STEPS=100; BURN_IN=10; SAVE_EVERY=100; EVAL_EVERY=50; BATCH_SIZE=256

NUM_GPUS=8
SAMPLES_PER_GPU=125000
TOTAL_STEPS=12200
BURN_IN=1000
SAVE_EVERY=100
EVAL_EVERY=5000
BATCH_SIZE=1024
TEMPERATURE=1.0
SCORING_MODEL="ESM2"
ANTIBODY="c05"

TOTAL_SAMPLES=$((SAMPLES_PER_GPU * NUM_GPUS))
TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="${PGLD_SCRATCH}/esm2_gibbs_runs"
mkdir -p ${OUTPUT_DIR}
FINAL_OUTPUT="${OUTPUT_DIR}/${TIMESTAMP}_${ANTIBODY}_esm2_${MODEL_SIZE}_ed${MAX_EDIT_DISTANCE}_gibbs_${TOTAL_SAMPLES}.fasta"

echo "Running ESM2-${MODEL_SIZE^^} Gibbs sampling (parallel across ${NUM_GPUS} GPUs)"
echo "  Model: ESM2-${MODEL_SIZE^^}"
echo "  Base seed: ${BASE_SEED}"
echo "  Antibody: ${ANTIBODY}"
echo "  Total samples: ${TOTAL_SAMPLES}"
echo "  Samples per GPU: ${SAMPLES_PER_GPU}"
echo "  Total steps: ${TOTAL_STEPS}, Burn-in: ${BURN_IN}"
echo "  Save every: ${SAVE_EVERY}, Eval every: ${EVAL_EVERY}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max edit distance: ${MAX_EDIT_DISTANCE}"
echo "  Output: ${FINAL_OUTPUT}"

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PARTIAL_DIR="${OUTPUT_DIR}/${TIMESTAMP}_${MODEL_SIZE}_partial_gpu${GPU_ID}"
    SEED=$((BASE_SEED + GPU_ID * 1000))

    # Only GPU 0 gets verbose output
    if [ ${GPU_ID} -eq 0 ]; then
        VERBOSE_FLAG="--verbose"
        echo "Launching GPU ${GPU_ID} (seed=${SEED}, verbose) -> ${PARTIAL_DIR}"
    else
        VERBOSE_FLAG=""
        echo "Launching GPU ${GPU_ID} (seed=${SEED}, silent) -> ${PARTIAL_DIR}"
    fi

    CUDA_VISIBLE_DEVICES=${GPU_ID} python pgld/scripts/sample_esm2_gibbs.py \
        --antibody ${ANTIBODY} \
        --batch-size ${BATCH_SIZE} \
        --total-steps ${TOTAL_STEPS} \
        --burn-in ${BURN_IN} \
        --save-every ${SAVE_EVERY} \
        --eval-every ${EVAL_EVERY} \
        --temperature ${TEMPERATURE} \
        --scoring-model ${SCORING_MODEL} \
        --max-edit-distance ${MAX_EDIT_DISTANCE} \
        --model-size ${MODEL_SIZE} \
        --output-dir ${PARTIAL_DIR} \
        --seed ${SEED} \
        ${VERBOSE_FLAG} &

    PIDS+=($!)
done

echo "Waiting for ${NUM_GPUS} processes to complete..."
for PID in ${PIDS[@]}; do
    wait ${PID}
done

echo "Concatenating partial outputs..."
cat ${OUTPUT_DIR}/${TIMESTAMP}_${MODEL_SIZE}_partial_gpu*/sequences.fasta > ${FINAL_OUTPUT}

# Clean up partial directories (histograms etc kept in first GPU's dir for reference)
for GPU_ID in $(seq 1 $((NUM_GPUS - 1))); do
    rm -rf ${OUTPUT_DIR}/${TIMESTAMP}_${MODEL_SIZE}_partial_gpu${GPU_ID}
done
mv ${OUTPUT_DIR}/${TIMESTAMP}_${MODEL_SIZE}_partial_gpu0 ${OUTPUT_DIR}/${TIMESTAMP}_${ANTIBODY}_${MODEL_SIZE}_diagnostics

echo "Done! Output: ${FINAL_OUTPUT}"
echo "Total sequences: $(grep -c '^>' ${FINAL_OUTPUT})"
