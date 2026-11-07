#!/bin/bash
#SBATCH --job-name=gen_plots
#SBATCH --time=1:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g

# Generate paper plots from experiment results

source "${HOME}/pgld/.env.local"
source ${PGLD_CONDA_BASE}/bin/activate pgld_basic

RESULTS_DIR="${PGLD_OUTPUT_DIR}/optimizer_exps"
PLOTS_DIR="${RESULTS_DIR}/plots"
HARD_DIR="${PLOTS_DIR}/hard"

# 1. Hard greedy re-eval of every saved policy, then the method-comparison + template-sweep plots.
python pgld/scripts/eval_optimizer_hits.py --results-dir "${RESULTS_DIR}" --landscapes all --out "${HARD_DIR}/hard_hits_all.pkl"
python pgld/scripts/final_paper_plots.py --results-dir "${RESULTS_DIR}" --hard-cache "${HARD_DIR}/hard_hits_all.pkl" --out-dir "${HARD_DIR}"

# 2. Diversity/fitness Pareto and credit-assignment curves.
python pgld/scripts/aggregate_paper_results.py diversity_fitness --results-dir "${RESULTS_DIR}" -o "${PLOTS_DIR}/diversity_fitness/"
python pgld/scripts/aggregate_paper_results.py credit_assignment --results-dir "${RESULTS_DIR}" -o "${PLOTS_DIR}/credit_assignment/"

echo "Plots saved to ${PLOTS_DIR}/"
