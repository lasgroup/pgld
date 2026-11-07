# Lab-in-the-Loop Experiments

Lab-in-the-loop baseline experiments across all landscapes. 
Results used for Figure 3 in the paper.

## Run

```bash
sbatch dhfr.sh
sbatch gb1.sh
sbatch mhc.sh
sbatch phoq.sh
sbatch riboswitch.sh
sbatch tfb.sh
sbatch trpb.sh
```

Each runs 5 seeds (array job 0-4), 50 templates, 5 iterations.

## Config

| Parameter | Value |
|-----------|-------|
| Templates | 50 |
| Seeds | 52-56 |
| Iterations | 5 |
| Samples per round | 5000 (TF binding 2000, MHC 10000) |
| Policy | IUPAC |
| Reward | hits |

## Generate Plots

After experiments complete:

```bash
python pgld/scripts/generate_bo_paper_plots.py
```

This generates Figure 3 comparing lab-in-the-loop results across landscapes.
