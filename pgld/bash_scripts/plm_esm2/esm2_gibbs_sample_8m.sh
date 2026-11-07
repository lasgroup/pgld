#!/bin/bash
#SBATCH --job-name=esm2_gibbs
#SBATCH --time=8:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_3090:1

source "${HOME}/pgld/pgld/bash_scripts/common_plm_setup.sh"

export PYTHONUNBUFFERED=1

# ESM2 Gibbs sampling - small test run
# Sequences generated = (TOTAL_STEPS / SAVE_EVERY) * BATCH_SIZE = 5 * 1024 = 5120

TOTAL_STEPS=5000
BURN_IN=1000
SAVE_EVERY=1000
EVAL_EVERY=1000
BATCH_SIZE=1024
TEMPERATURE=1.0
SCORING_MODEL="ESM2"
MAX_EDIT_DISTANCE=6
ANTIBODY="c05"

echo "Running ESM2 Gibbs sampling"
echo "  Antibody: ${ANTIBODY}"
echo "  Total steps: ${TOTAL_STEPS}, Burn-in: ${BURN_IN}"
echo "  Save every: ${SAVE_EVERY}, Eval every: ${EVAL_EVERY}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Max edit distance: ${MAX_EDIT_DISTANCE}"

python pgld/scripts/sample_esm2_gibbs.py \
    --antibody ${ANTIBODY} \
    --batch-size ${BATCH_SIZE} \
    --total-steps ${TOTAL_STEPS} \
    --burn-in ${BURN_IN} \
    --save-every ${SAVE_EVERY} \
    --eval-every ${EVAL_EVERY} \
    --temperature ${TEMPERATURE} \
    --scoring-model ${SCORING_MODEL} \
    --max-edit-distance ${MAX_EDIT_DISTANCE} \
    --verbose
