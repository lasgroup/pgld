#!/bin/bash
#SBATCH --job-name=esm2_150m_lambda
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=nvidia_geforce_rtx_4090:8

# Lambda sweep for ESM2-150M C05 experiments
# Tests entropy lambda values to see effect on diversity vs PLM score
# Self-submits as array job

# Lambda values to sweep
LAMBDA_VALUES=(3000.0)

# Check for init mode flag before self-submission
INIT_FLAG=""
if [ "$1" == "--from-fasta" ]; then
    INIT_FLAG="--from-fasta"
elif [ "$1" == "--no-init" ]; then
    INIT_FLAG="--no-init"
elif [ "$1" == "--nnk-init" ]; then
    INIT_FLAG="--nnk-init"
fi

# Self-submission: if not running inside SLURM, submit as array job
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    ARRAY_SIZE=${#LAMBDA_VALUES[@]}
    echo "Submitting $ARRAY_SIZE jobs for lambda sweep: ${LAMBDA_VALUES[*]}"
    sbatch --array=0-$((ARRAY_SIZE-1)) "$0" ${INIT_FLAG}
    exit 0
fi

# Only source environment after self-submission (inside SLURM job)
source "${HOME}/pgld/pgld/bash_scripts/common_plm_setup.sh"

# Get lambda for this array task
LAMBDA=${LAMBDA_VALUES[$SLURM_ARRAY_TASK_ID]}
echo "Running with lambda=${LAMBDA}"

SAMPLE_BATCH_SIZE=2048
# ESM2 creates 24 masked versions per sequence, so oracle batch must be smaller
ORACLE_BATCH_SIZE_PER_GPU=64
ITERS=30000

# Set init args based on mode
if [ "$1" == "--from-fasta" ]; then
    # Update path to your sampled fasta file from esm2_gibbs_sample_1M.sh
    FASTA_PATH="${PGLD_SCRATCH}/esm2_gibbs_runs/<your_esm2_150m_samples>.fasta"
    echo "Initializing from FASTA: ${FASTA_PATH}"
    INIT_ARGS=(
        policy.init_fasta_path=${FASTA_PATH}
        policy.init_epsilon=0.005
        policy.init_fasta_seed=42
    )
    OUTPUT_SUFFIX="_fasta"
elif [ "$1" == "--no-init" ]; then
    echo "No initialization (random)"
    INIT_ARGS=()
    OUTPUT_SUFFIX="_random"
elif [ "$1" == "--nnk-init" ]; then
    # NNK codon for each of the 24 amino acid positions (C05 CDR-H3)
    INIT_SEQ=NNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNKNNK
    echo "Initializing from NNK: ${INIT_SEQ}"
    INIT_ARGS=(
        policy.init_iupac_sequence=${INIT_SEQ}
        policy.init_epsilon=0.05
    )
    OUTPUT_SUFFIX="_nnk"
else
    # Init sequence encodes HMSMQQVVSAGWERADLVGDAFDV (C05 wild-type CDR-H3)
    INIT_SEQ=CATATGAGCATGCAGCAGGTGGTGAGCGCGGGCTGGGAACGCGCGGATCTGGTGGGCGATGCGTTTGATGTG
    echo "Initializing from IUPAC: ${INIT_SEQ}"
    INIT_ARGS=(
        policy.init_iupac_sequence=${INIT_SEQ}
        policy.init_epsilon=0.005
    )
    OUTPUT_SUFFIX=""
fi

# Output directory with clear lambda naming
OUTPUT_DIR="${PGLD_OUTPUT_DIR}/plm_run/esm2_150m_c05_lambda_sweep${OUTPUT_SUFFIX}/lambda_${LAMBDA}"

python pgld/train.py \
    landscape=plm_esm2 \
    landscape.antibody=c05 \
    landscape.models.ESM2.args.model_name=esm2_150m \
    landscape.oracle_batch_size_per_gpu=${ORACLE_BATCH_SIZE_PER_GPU} \
    landscape.max_edit_distance=6 \
    policy=iupac \
    policy.num_templates=1024 \
    "${INIT_ARGS[@]}" \
    training.optimization.iters=${ITERS} \
    training.optimization.lr=0.02 \
    training.sampling.mc_samples=1 \
    training.sampling.batch_size=${SAMPLE_BATCH_SIZE} \
    training.reward_type=average \
    training.reward_aggregation_unit=template \
    training.per_sequence_rewards=false \
    training.entropy.lambda_=${LAMBDA} \
    training.entropy.sequence_type=peptide \
    training.entropy.type=marginal_gain \
    training.entropy.baseline_per_template=true \
    training.pooling.uniqueness=peptide \
    training.pooling.enabled=true \
    training.evaluation.extra_eval=false \
    training.evaluation.checkpointing=false \
    ++training.fast_mode=true \
    ++training.evaluation.steps_til_vis=2000 \
    "hydra.run.dir=${OUTPUT_DIR}"
