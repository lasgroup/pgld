# PGLD

Code for "Scalable and Synthesizable DNA Library Design via Reinforcement Learning"

## Quick Start

Extract or rename this folder to `pgld` in your home directory (scripts expect `~/pgld`).

```bash
# 1. Copy environment config
cp .env.template .env.local
# Edit .env.local with your paths (see Configuration section)

# 2. Create conda environment (see Environment Setup)

# 3. Run a smoke test (from the repo root, with the base env active)
conda activate pgld_basic
bash pgld/bash_scripts/smoke_tests/hits_pooled.sh
```

All commands below are run from the repo root.

These instructions are written assuming access to a GPU cluster with Slurm.
It should be possible to adapt them for also running e.g. locally.

## Environment Setup

Two separate conda environments are needed due to dependency conflicts:

| Environment | Python | Use Case |
|-------------|--------|----------|
| `pgld_basic` | 3.9 | TF binding, MHC, riboswitch, protein fitness landscapes (5.1, 5.2)|
| `pgld_plm` | 3.10 | PLM experiments (ESM2, IgLM) (5.3)|

**pgld_basic environment** (non-PLM experiments):

```bash
conda create -n pgld_basic python=3.9 -y
sbatch pgld/bash_scripts/setup/setup_pgld_basic_env.sh
```

**pgld_plm environment** (PLM experiments):

```bash
conda create -n pgld_plm python=3.10 -y
sbatch pgld/bash_scripts/setup/setup_pgld_plm_env.sh
```

## Configuration

### 1. Create .env.local

```bash
cp .env.template .env.local
```

### 2. Edit paths in .env.local

Most paths use `${USER}` and work out of the box. Update `PGLD_PROJECT_ROOT` if you renamed the folder. Key variables:

| Variable | Description |
|----------|-------------|
| `PGLD_CONDA_BASE` | Path to your miniconda/anaconda installation |
| `PGLD_SCRATCH` | Scratch directory for outputs |
| `PGLD_ASSETS_DIR` | Model checkpoints and datasets |

### 3. Configure cluster modules

`flash-attn` compiles from source and needs CUDA and GCC modules. Update `.env.local` to match your cluster.
For example:

```bash
export PGLD_GCC_MODULE=gcc/12.2.0
export PGLD_CUDA_MODULE=cuda/13.0.2
```

### 4. Source before running

```bash
source .env.local
```

Or add to your `.bashrc`.
All bash scripts in `pgld/bash_scripts/` automatically source `.env.local` to load these variables.

## Assets

Model checkpoints and datasets should be placed in `assets/`. The expected structure:

```
assets/
├── esm2/                    # ESM2 safetensors (for PLM experiments)
│   ├── esm2_8m.safetensors
│   └── esm2_150m.safetensors
├── datasets/                # Protein fitness datasets (GB1, PhoQ, etc.)
└── riboswitch/              # Riboswitch oracle
```

The TF binding landscape data ships with the repository. The remaining
assets — the riboswitch oracle checkpoint and the protein-fitness datasets — are provided
as a supplementary-materials zip on the paper's OpenReview submission. Unzip it into
`$PGLD_ASSETS_DIR` (the repository's `assets/` by default):

```bash
unzip pgld_supplementary_assets.zip -d .   # from the repo root, with assets/ at the default location
```

It contains:
- `assets/riboswitch/atp_oracle.ckpt` — riboswitch oracle (for the riboswitch landscape)
- `assets/datasets/{gb1_fitness.csv, PhoQ.xlsx, trpb.csv, dhfr.csv}` — GB1, PhoQ, TrpB, DHFR fitness landscapes

### Downloading ESM2 Weights

ESM2 weights are needed for PLM experiments (5.3). Download from [mhcelik/esm-efficient](https://huggingface.co/mhcelik/esm-efficient/tree/main):

```bash
mkdir -p assets/esm2
cd assets/esm2
wget https://huggingface.co/mhcelik/esm-efficient/resolve/main/esm2_8m.safetensors
wget https://huggingface.co/mhcelik/esm-efficient/resolve/main/esm2_150m.safetensors
```

IgLM weights are downloaded automatically from HuggingFace at runtime.

## Experiments

### Optimizer Experiments

See `pgld/bash_scripts/optimizer_exps/README.md` for full details.

Requires `pgld_basic` environment.

To generate plots from the main paper for all landscapes:

```bash
bash pgld/bash_scripts/optimizer_exps/launch_all.sh
```

**Generate plots**:
```bash
python pgld/scripts/eval_optimizer_hits.py --results-dir ${PGLD_OUTPUT_DIR}/optimizer_exps --landscapes all --out plots/hard/hard_hits_all.pkl
python pgld/scripts/final_paper_plots.py --results-dir ${PGLD_OUTPUT_DIR}/optimizer_exps --hard-cache plots/hard/hard_hits_all.pkl --out-dir plots/hard
```

### Lab-in-the-Loop

Requires `pgld_basic` environment.

See `pgld/bash_scripts/bo_exps/README.md`.

### PLM Experiments (ESM2, IgLM)

Requires `pgld_plm` environment.

The workflow is: (1) sample a training set and a separate validation set from the PLM (different
seeds); (2) fit a variational-synthesis policy to the training set; (3) fit a PGLD policy to the
target PLM then (4) compare both learned libraries against the held-out validation set. The
commands below use the larger models as in the paper; scaled-down scripts for smaller models
(ESM2 8M, IgLM small) sit alongside them.

```bash
# 1. Sample a training set and a validation set from the PLM.
#    Run the sampler twice with different seeds; each run writes one fasta:
#      ESM2 -> ${PGLD_SCRATCH}/esm2_gibbs_runs/<timestamp>_..._gibbs_<N>.fasta
#      IgLM -> ${PGLD_SCRATCH}/iglm_large_samples/<...>.fasta
#    The validation set is held out for the comparison in the metrics step below.
sbatch pgld/bash_scripts/plm_esm2/esm2_gibbs_sample_1M.sh 150m 42     # training set
sbatch pgld/bash_scripts/plm_esm2/esm2_gibbs_sample_1M.sh 150m 123    # validation set
sbatch pgld/bash_scripts/plm_iglm/iglm_sample_trastuzumab_large.sh 42     # training set
sbatch pgld/bash_scripts/plm_iglm/iglm_sample_trastuzumab_large.sh 123    # validation set

# 2. Fit the variational-synthesis baseline to the training set.
#    Clone https://github.com/debbiemarkslab/variational-synthesis into
#    ${PGLD_PROJECT_ROOT}/variational-synthesis and create its conda env per its README.
#    Set target_samples in var_synth/configs/*.cfg to the step-1 training fasta. Each run writes
#    its own designed-library fasta under ${PGLD_SCRATCH}/var_synth/.
sbatch pgld/bash_scripts/plm_esm2/var_synth_train_esm2_150m.sh
sbatch pgld/bash_scripts/plm_iglm/var_synth_train_iglm_large.sh

# 3. Fit the PGLD policy to the target PLM.
#    The policy optimizes against the PLM itself (landscape=plm_esm2/plm_iglm); --from-fasta
#    initializes the policy using the step-1 training fasta (set FASTA_PATH in each script first).
sbatch pgld/bash_scripts/plm_esm2/esm2_c05_lambda_sweep_150m.sh --from-fasta
sbatch pgld/bash_scripts/plm_iglm/iglm_large_trastuzumab_lambda_sweep.sh --from-fasta
```

**Generate visualizations / metrics:**
```bash
python pgld/regenerate_vis.py \
    --run-dir ${PGLD_OUTPUT_DIR}/<plm_run>/lambda_<value>/ \
    --reference-fasta ${PGLD_SCRATCH}/esm2_gibbs_runs/<validation_samples>.fasta \
    --var-synth-fasta ${PGLD_SCRATCH}/var_synth/<model>/log/<samples_file>.fasta \
    --max-vis-samples 10000 \
    --mmd-seq-count 100000 \
    --paper-ready
```

Here `--run-dir` is the PGLD policy run from step 3 (the script samples the learned library
from it), `--reference-fasta` is the held-out validation set from step 1, and `--var-synth-fasta`
is var-synth's designed library from step 2.

## How the Repo Works

When you run `sbatch pgld/bash_scripts/optimizer_exps/tf_binding_sweep.sh`:

```
bash script
    │
    ├── Sources .env.local → loads your paths
    ├── Activates conda environment
    │
    └── python train.py landscape=tf_binding policy=iupac ...
            │
            ├── Hydra merges: config.yaml + landscape/tf_binding.yaml + policy/iupac.yaml
            │
            └── Training loop:
                  1. Sample DNA sequences from policy
                  2. Score sequences with landscape oracle
                  3. Compute REINFORCE gradients
                  4. Update policy → repeat
                  5. Save outputs to hydra.run.dir/vis/
```

**Core components:**
- **Policy** (`policies.py`): Parameterizes a DNA library. `IupacBaseOptimizer` learns degenerate codons (NNK, etc) per template as in PGLD, `ArbitraryBaseOptimizer` learns per-position distributions over nucleotides as in Zhu et al.
- **Landscape** (`landscapes.py`): Oracle that scores sequences. Wraps TF binding models, MHCflurry, PLMs, etc.
- **Config** (`conf/`): Hydra configs. Override anything from CLI: `training.entropy.lambda_=0.1`

For the PLM experiments, the training loop routes through a 'fast mode' that is more optimized but specific to the mean-entropy objective and PLM landscapes.
All other experiments use `pooled_train_policy` in `policy_training.py`.
The `train_policy` function is legacy code kept for reference but not used in practice.

## Code Structure

```
/
├── pgld/
│   ├── train.py              # Main training script
│   ├── policies.py           # Policy optimizers (Arbitrary, IUPAC)
│   ├── landscapes.py         # Fitness oracles (TF binding, MHC, PLM, etc.)
│   ├── conf/                 # Hydra configs
│   │   ├── config.yaml
│   │   ├── policy/
│   │   └── landscape/
│   ├── bash_scripts/
│   │   ├── optimizer_exps/       # Main experiments
│   │   ├── plm_esm2/             # ESM2 experiments
│   │   ├── plm_iglm/             # IgLM experiments
│   │   ├── bo_exps/              # Lab-in-the-loop baseline
│   │   ├── smoke_tests/          # Quick validation
│   │   └── setup/                # Environment setup
│   ├── scripts/              # Analysis & plotting
│   └── tests/                # Unit tests
├── assets/                   # Model checkpoints and datasets
├── .env.template             # Environment config template
└── .env.local                # Your local config (git-ignored)
```

## Testing

Run from the repo root with the `pgld_basic` environment active.

```bash
# Unit tests
pytest pgld/tests/

# Smoke tests (quick integration tests)
bash pgld/bash_scripts/smoke_tests/hits_pooled.sh
```

## Troubleshooting

### MHCflurry setup
MHCflurry requires downloading model weights after installation:
```bash
mhcflurry-downloads fetch
```

### Flash attention
`flash-attn` compiles from source. Load your cluster's CUDA and GCC modules before installing:
```bash
module load cuda/13.0.2 gcc/12.2.0  # adjust for your cluster
pip install flash-attn --no-build-isolation
```
