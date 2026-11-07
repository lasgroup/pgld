#!/usr/bin/env python
"""Hard (greedy) re-eval of saved optimizer-experiment policies across all landscapes.

For every template_sweep / arbitrary run that has a saved parameters.pth, greedy-sample
the hardened library at the run's config batch_size, score with the landscape, and count
hits at the config threshold (top_prop resolved via setup.build_thresholds). Greedy
hardens the IUPAC library (argmax bottle selection); for the arbitrary policy greedy is a
no-op so it samples normally. Writes a pickle + CSV of per-run hits.

Usage (pgld_basic env; GPU needed only for the riboswitch oracle):
    python pgld/scripts/eval_optimizer_hits.py \
        --results-dir $PGLD_OUTPUT_DIR/optimizer_exps --landscapes all \
        --out $PGLD_OUTPUT_DIR/optimizer_exps/plots_paper_hard/hard_hits_all.pkl
"""
import argparse
import glob
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from setup import build_invalid_scores, build_landscape, build_policy, build_thresholds
from landscapes import sequences_to_strings_vectorized
from train_utils_policies import count_hits_from_sequences


def eval_run(run_dir: str, device: torch.device) -> Dict[str, object]:
    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    obj_names = list(cfg.landscape.objectives.keys())
    invalid = build_invalid_scores(cfg, obj_names)
    landscape = build_landscape(cfg, obj_names, device, invalid)
    thresholds = build_thresholds(cfg, obj_names, landscape)
    policy = build_policy(cfg, landscape, len(landscape.alphabet), device)
    policy.load_state_dict(
        torch.load(os.path.join(run_dir, "parameters.pth"), map_location=device, weights_only=True)
    )
    bs = int(cfg.training.sampling.batch_size)
    torch.manual_seed(0)
    np.random.seed(0)
    sr = policy.sample(bs, greedy=True)  # IUPAC: argmax-hardened library; arbitrary: no-op
    seqs = sequences_to_strings_vectorized(sr.sequences, landscape.alphabet)
    if len(obj_names) == 1:
        thr, thrs = thresholds[obj_names[0]], None
    else:
        thr, thrs = None, thresholds
    u, a, _, mf = count_hits_from_sequences(landscape, seqs, thr, thrs)
    return {
        "landscape": cfg.landscape.name,
        "policy": cfg.policy.name,
        "num_templates": int(cfg.policy.num_templates),
        "lambda": float(cfg.training.entropy.lambda_),
        "batch_size": bs,
        "objective": obj_names[0],
        "reward_type": getattr(cfg.training, "reward_type", "hits"),
        "unique_hits": int(u),
        "all_hits": int(a),
        "unique_pct": 100.0 * u / bs,
        "all_pct": 100.0 * a / bs,
        "mean_fitness": float(mf),
        "run_dir": run_dir,
        "is_ablation": False,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=os.path.join(os.environ["PGLD_OUTPUT_DIR"], "optimizer_exps"))
    p.add_argument("--landscapes", default="all", help="comma-separated landscape names, or 'all'")
    p.add_argument("--out", required=True, help="output pickle path")
    args = p.parse_args()

    want = None if args.landscapes == "all" else set(args.landscapes.split(","))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} landscapes={args.landscapes}", flush=True)

    rows = []
    params = sorted(glob.glob(os.path.join(args.results_dir, "**", "parameters.pth"), recursive=True))
    for pth in params:
        rd = os.path.dirname(pth)
        if not ("/template_sweep/" in rd or "/arbitrary/" in rd):
            continue
        cfgp = os.path.join(rd, ".hydra", "config.yaml")
        if not os.path.exists(cfgp):
            continue
        try:
            lname = OmegaConf.load(cfgp).landscape.name
        except Exception as e:
            print(f"  SKIP cfg {rd}: {e}", flush=True)
            continue
        if want is not None and lname not in want:
            continue
        try:
            r = eval_run(rd, device)
            rows.append(r)
            print(f"  {r['landscape']:12s} {os.path.basename(rd):22s} t{r['num_templates']:<2d} {r['policy']:9s} "
                  f"unique={r['unique_pct']:5.1f}% all={r['all_pct']:5.1f}%", flush=True)
        except Exception as e:
            print(f"  SKIP {rd}: {type(e).__name__}: {e}", flush=True)

    with open(args.out, "wb") as f:
        pickle.dump(rows, f)
    pd.DataFrame(rows).to_csv(args.out.replace(".pkl", ".csv"), index=False)
    print(f"saved {len(rows)} rows -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
