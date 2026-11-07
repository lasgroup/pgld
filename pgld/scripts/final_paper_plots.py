"""Final all-hard paper plots: template-scaling line + method-comparison bar per
landscape group, with every policy point replaced by its hard greedy-resample hits
(from eval_optimizer_hits.py). Reuses the existing paper plot functions; only the
dataframe values are overridden.

Usage (pgld_basic env; GPU for the riboswitch Random baseline):
    python pgld/scripts/final_paper_plots.py \
        --hard-cache $PGLD_OUTPUT_DIR/optimizer_exps/plots_paper_hard/hard_hits_all.pkl \
        --out-dir $PGLD_OUTPUT_DIR/optimizer_exps/plots_paper_hard
"""
import argparse
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))

from aggregate_paper_results import (
    load_experiment_results,
    plot1_random_vs_policies,
    plot2_template_sweep,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(os.environ["PGLD_OUTPUT_DIR"], "optimizer_exps"))
    p.add_argument("--hard-cache", required=True, help="pickle from eval_optimizer_hits")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 1. Soft df (metadata + columns) from all runs.
    df = load_experiment_results(args.results_dir)

    # 2. Override policy hits with the hard greedy-resample values, matched by run_dir.
    with open(args.hard_cache, "rb") as f:
        hard = pd.DataFrame(pickle.load(f)).set_index("run_dir")
    n = 0
    for i in df.index:
        rd = df.loc[i, "run_dir"]
        if rd in hard.index:
            df.loc[i, "unique_hits"] = hard.loc[rd, "unique_hits"]
            df.loc[i, "all_hits"] = hard.loc[rd, "all_hits"]
            df.loc[i, "batch_size"] = hard.loc[rd, "batch_size"]
            n += 1
    print(f"overrode {n} runs with hard values; landscapes={sorted(df['landscape'].unique())}", flush=True)

    # 3. Reuse the paper plot functions (per landscape group), Random baseline included.
    plot2_template_sweep(df, os.path.join(args.out_dir, "template_sweep.pdf"))
    plot1_random_vs_policies(df, os.path.join(args.out_dir, "policy_comparison.pdf"), compute_baselines=True)
    print("FINAL HARD PLOTS DONE ->", args.out_dir, flush=True)


if __name__ == "__main__":
    main()
