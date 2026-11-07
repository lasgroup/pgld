# Optimizer Experiments

Scripts for reproducing the main optimization experiments in the paper.

## Plots

| CLI Subcommand | Experiment Scripts | Description |
|----------------|-------------------|-------------|
| `policy_comparison` | `launch_all.sh all template` + `launch_all.sh all arbitrary` | Random vs Arbitrary (1T) vs IUPAC (1T, 8T) |
| `template_sweep` | `launch_all.sh all template` | Template count sweep (1-32 templates) |
| `credit_assignment` | `mhc_global_vs_template_*.sh` | Per-template vs global credit assignment |

## Experiments

### 1. Template Sweep (all landscapes)
Sweeps number of templates: 1, 2, 4, 8, 16, 32 with the hits objective for PGLD.

```bash
sbatch dhfr_sweep.sh template
sbatch gb1_sweep.sh template
sbatch mhc_sweep.sh template
sbatch phoq_sweep.sh template
sbatch riboswitch_sweep.sh template
sbatch tf_binding_sweep.sh template
sbatch trpb_sweep.sh template
```

### 2. Arbitrary Baseline (all landscapes)
Single template with arbitrary policy and expected hits objective.

```bash
sbatch dhfr_sweep.sh arbitrary
sbatch gb1_sweep.sh arbitrary
sbatch mhc_sweep.sh arbitrary
sbatch phoq_sweep.sh arbitrary
sbatch riboswitch_sweep.sh arbitrary
sbatch tf_binding_sweep.sh arbitrary
sbatch trpb_sweep.sh arbitrary
```

### 3. Lambda Sweep (unified)
Entropy weight sweep with mean-entropy objective.

**IUPAC policy:**
```bash
sbatch lambda_sweep.sh mhc_flurry 1 #1 template
sbatch lambda_sweep.sh mhc_flurry 16 #16 templates
```

**Arbitrary policy:**
```bash
sbatch arbitrary_lambda_sweep.sh mhc_flurry
```

### 4. Global vs Template Credit Assignment (MHC only)
Compares per-template rewards/entropy vs global (batch-level) credit.
32 templates, sweeps learning rate.

```bash
sbatch mhc_global_vs_template_mean.sh   # mean-entropy objective
sbatch mhc_global_vs_template_hits.sh   # expected hits objective
```

## Launch All

```bash
bash launch_all.sh                           # all experiments
bash launch_all.sh all template              # template sweep only
bash launch_all.sh all arbitrary             # arbitrary only
bash launch_all.sh mhc_flurry all            # single landscape
bash launch_all.sh all all --skip-existing   # resume interrupted run
```

## Generate Plots

```bash
# Method comparison + template sweep (every policy is scored, then plotted per landscape group).
python pgld/scripts/eval_optimizer_hits.py --results-dir ${PGLD_OUTPUT_DIR}/optimizer_exps --landscapes all --out plots/hard/hard_hits_all.pkl
python pgld/scripts/final_paper_plots.py --results-dir ${PGLD_OUTPUT_DIR}/optimizer_exps --hard-cache plots/hard/hard_hits_all.pkl --out-dir plots/hard

# Diversity vs fitness Pareto curves (per landscape)
python pgld/scripts/aggregate_paper_results.py diversity_fitness --results-dir ${PGLD_OUTPUT_DIR}/optimizer_exps -o plots/diversity_fitness/

# Credit assignment comparison (MHC only)
python pgld/scripts/aggregate_paper_results.py credit_assignment --results-dir ${PGLD_OUTPUT_DIR}/optimizer_exps -o plots/credit_assignment/
```
