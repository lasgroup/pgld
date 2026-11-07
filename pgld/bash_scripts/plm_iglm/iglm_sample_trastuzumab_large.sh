#!/bin/bash
#SBATCH --job-name=iglm_sample
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:16g

# Usage: sbatch iglm_sample_trastuzumab.sh [seed]
#   $1: master seed (default: 42). Each GPU gets seed + GPU_ID*1000

source "${HOME}/pgld/pgld/bash_scripts/common_plm_setup.sh"

# IgLM-S sampling for trastuzumab CDRH3
# Generates 1M sequences using fixed-length infilling (duplicates allowed)
# Parallelized across 4 GPUs with different seeds

BASE_SEED=${1:-42}
TOTAL_SAMPLES=1000000
NUM_GPUS=4
SAMPLES_PER_GPU=$((TOTAL_SAMPLES / NUM_GPUS))
MODEL="IgLM"
ANTIBODY="trastuzumab"
TEMPERATURE=1.0
TOP_P=1.0

TIMESTAMP=$(date +%Y-%m-%d_%H-%M-%S)
OUTPUT_DIR="${PGLD_SCRATCH}/iglm_large_samples"
mkdir -p ${OUTPUT_DIR}
FINAL_OUTPUT="${OUTPUT_DIR}/${TIMESTAMP}_${ANTIBODY}_${MODEL}_${TOTAL_SAMPLES}.fasta"

echo "Running IgLM sampling (parallel across ${NUM_GPUS} GPUs)"
echo "  Model: ${MODEL}"
echo "  Antibody: ${ANTIBODY}"
echo "  Total samples: ${TOTAL_SAMPLES}"
echo "  Samples per GPU: ${SAMPLES_PER_GPU}"
echo "  Temperature: ${TEMPERATURE}"
echo "  Top-p: ${TOP_P}"
echo "  Base seed: ${BASE_SEED}"
echo "  Output: ${FINAL_OUTPUT}"

# Launch parallel processes on each GPU with different seeds
PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PARTIAL_OUTPUT="${OUTPUT_DIR}/${TIMESTAMP}_partial_gpu${GPU_ID}.fasta"
    SEED=$((BASE_SEED + GPU_ID * 1000))
    echo "Launching GPU ${GPU_ID} (seed=${SEED}) -> ${PARTIAL_OUTPUT}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python pgld/scripts/sample_iglm_infill.py \
        --antibody ${ANTIBODY} \
        --model ${MODEL} \
        --num-samples ${SAMPLES_PER_GPU} \
        --temperature ${TEMPERATURE} \
        --top-p ${TOP_P} \
        --seed ${SEED} \
        --output ${PARTIAL_OUTPUT} &

    PIDS+=($!)
done

# Wait for all processes to complete
echo "Waiting for ${NUM_GPUS} processes to complete..."
for PID in ${PIDS[@]}; do
    wait ${PID}
done

# Concatenate partial outputs
echo "Concatenating partial outputs..."
cat ${OUTPUT_DIR}/${TIMESTAMP}_partial_gpu*.fasta > ${FINAL_OUTPUT}

# Keep histogram files, rename to final naming scheme
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    mv ${OUTPUT_DIR}/${TIMESTAMP}_partial_gpu${GPU_ID}_duplicate_hist.png \
       ${OUTPUT_DIR}/${TIMESTAMP}_${ANTIBODY}_${MODEL}_gpu${GPU_ID}_duplicate_hist.png 2>/dev/null || true
done

# Clean up partial fastas
rm ${OUTPUT_DIR}/${TIMESTAMP}_partial_gpu*.fasta

echo "Done! Output: ${FINAL_OUTPUT}"
echo "Sequences: $(grep -c '^>' ${FINAL_OUTPUT})"
