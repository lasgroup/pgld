#!/usr/bin/env python
"""
Compute peptide-level entropy ("effective sequence space") of a library design.

Convention: we measure the entropy of the distribution over FULL-LENGTH,
stop-free peptides, i.e. we condition on "no internal stop at any codon
position". Reported quantities:

  - H            : Shannon entropy of the conditional peptide distribution (nats & bits)
  - eff_seq_space: exp(H), the effective number of distinct peptides ("effective
                   sequence space", Hill number of order 1)
  - p_no_stop    : P(full-length, stop-free) under the library (usable/yield fraction)
  - mc_stderr    : Monte-Carlo standard error of H (95% CI = H +/- 1.96*stderr)

Both PGLD policy and variational-synthesis libraries are mixtures of
position-factorized templates:

    p(x) = sum_t w_t prod_l p_t(x_l | l)

so a single estimator applies to both once a design is expressed as per-template
per-codon amino-acid probabilities (T, Lc, 21) plus template weights (T,), with
the stop symbol as the last (index 20) column.

Env note: the PGLD adapter needs the `pgld_basic` conda env; the var-synth adapter
needs the `var-synth` env (to unpickle SynthesisModel). Env-specific imports are
therefore deferred into each adapter so the shared core runs in either env.

Usage:
    # PGLD (run in pgld_basic env)
    python compute_library_entropy.py --pgld_run_dir <run_dir> --out out.json
    # var-synth (run in var-synth env)
    python compute_library_entropy.py --var_synth_run_dir <run_dir> --out out.json
"""

import argparse
import json
import os
from typing import Optional, Tuple

import numpy as np
import torch

N_AA = 20  # amino acids excluding stop; stop is the last column (index 20)


# Core entropy estimator


def condition_full_length(
    aa_probs: np.ndarray, weights: np.ndarray, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Condition a template mixture on "no stop at any position" (convention 1).

    Within a template positions are independent, so conditioning renormalizes
    each position over the 20 non-stop amino acids and reweights templates by
    their stop-free probability.

    Args:
        aa_probs: (T, Lc, 21) per-template per-codon AA probs (col 20 = stop).
        weights:  (T,) template mixture weights.

    Returns:
        log_ptilde: (T, Lc, 20) log renormalized per-position AA probs.
        log_wtilde: (T,) log reweighted template priors (log-sum-exp to 0).
        p_no_stop:  P(full-length, stop-free) under the library.
    """
    stop = aa_probs[:, :, -1]  # (T, Lc)
    keep = aa_probs[:, :, :N_AA]  # (T, Lc, 20)
    surv = np.clip(1.0 - stop, eps, 1.0)  # (T, Lc)

    ptilde = keep / surv[:, :, None]  # (T, Lc, 20)
    log_ptilde = np.log(np.clip(ptilde, eps, 1.0))

    log_surv_per_template = np.log(surv).sum(axis=1)  # (T,)
    log_w = np.log(np.clip(weights, eps, None))  # (T,); lower guard only, no upper cap
    log_joint = log_w + log_surv_per_template  # (T,)
    log_Z = np.logaddexp.reduce(log_joint)  # scalar = log P(no stop)
    log_wtilde = log_joint - log_Z
    return log_ptilde, log_wtilde, float(np.exp(log_Z))


def mc_entropy(
    log_ptilde: np.ndarray,
    log_wtilde: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: str,
    seed: int,
    wt_idx: Optional[np.ndarray] = None,
    max_ed: Optional[int] = None,
) -> dict:
    """
    Monte-Carlo estimate of H for the conditional peptide mixture.

    Samples peptides from the mixture and scores each under the full mixture
    (exact log-prob via logsumexp over templates), so H_hat = -mean(log p).

    If wt_idx (Lc,) and max_ed are given, also accumulates -log p over the samples
    within Hamming distance max_ed of the WT (for the edit-distance-ball entropy).

    Args:
        log_ptilde: (T, Lc, 20) log per-position AA probs.
        log_wtilde: (T,) log template priors.
        n_samples:  number of MC samples.
        batch_size: samples per batch.
        device:     torch device.
        seed:       RNG seed.
        wt_idx:     (Lc,) WT amino-acid indices (adapter column order), or None.
        max_ed:     max Hamming distance to WT for the ball, or None.

    Returns:
        Dict with H_nats, stderr, and (if wt_idx given) ball statistics.
    """
    T, Lc, A = log_ptilde.shape
    gen = torch.Generator(device=device).manual_seed(seed)
    lp = torch.tensor(log_ptilde, device=device, dtype=torch.float32)  # (T, Lc, 20)
    lw = torch.tensor(log_wtilde, device=device, dtype=torch.float32)  # (T,)
    w = torch.exp(lw)  # (T,)
    p_pos = torch.exp(lp)  # (T, Lc, 20)
    pos_idx = torch.arange(Lc, device=device)
    wt = None if wt_idx is None else torch.tensor(wt_idx, device=device).long()  # (Lc,)
    if wt is not None:
        assert max_ed is not None and max_ed >= 0, "max_ed must be a non-negative int when wt_idx is given"

    neg_logps = []
    ball_neg = []
    remaining = n_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        remaining -= b
        t_idx = torch.multinomial(w, b, replacement=True, generator=gen)  # (b,)
        probs_b = p_pos[t_idx]  # (b, Lc, 20)
        X = torch.multinomial(probs_b.reshape(b * Lc, A), 1, generator=gen).reshape(
            b, Lc
        )  # (b, Lc)

        gathered = lp[:, pos_idx[None, :], X]  # (T, b, Lc)
        log_cond = gathered.sum(dim=-1) + lw[:, None]  # (T, b)
        log_p = torch.logsumexp(log_cond, dim=0)  # (b,)
        neg = -log_p
        neg_logps.append(neg.cpu())
        if wt is not None:
            hamming = (X != wt[None, :]).sum(dim=1)  # (b,)
            ball_neg.append(neg[hamming <= max_ed].cpu())

    neg = torch.cat(neg_logps)  # (N,)
    out = {
        "H_nats": neg.mean().item(),
        "stderr": (neg.std(unbiased=True) / (len(neg) ** 0.5)).item(),
    }
    if wt is not None:
        bn = torch.cat(ball_neg)  # (N_ball,)
        n_ball = len(bn)
        out["n_ball"] = int(n_ball)
        out["accept_frac"] = n_ball / len(neg)
        out["ball_neg_mean"] = bn.mean().item() if n_ball >= 1 else float("nan")
        out["ball_neg_stderr"] = (
            (bn.std(unbiased=True) / (n_ball**0.5)).item()
            if n_ball > 1
            else float("nan")
        )
    return out


def log_p_within_ed_ball(
    log_ptilde: np.ndarray, log_wtilde: np.ndarray, wt_idx: np.ndarray, max_ed: int
) -> float:
    """
    Exact log P(Hamming(x, WT) <= max_ed) under the conditional mixture.

    Per template the number of mismatches is Poisson-binomial over positions
    (mismatch prob 1 - ptilde_t(WT_l | l)); a DP over positions gives the exact
    per-template CDF, combined across templates by their weights.
    """
    T, Lc, _ = log_ptilde.shape
    match = np.exp(
        log_ptilde[np.arange(T)[:, None], np.arange(Lc)[None, :], wt_idx[None, :]]
    )  # (T, Lc) P(match WT at each position)
    mism = 1.0 - match

    dp = np.zeros((T, max_ed + 1))  # dp[:, k] = P(exactly k mismatches so far)
    dp[:, 0] = 1.0
    for l in range(Lc):
        m, mm = match[:, l], mism[:, l]
        new = np.zeros_like(dp)
        new[:, 0] = dp[:, 0] * m
        for k in range(1, max_ed + 1):
            new[:, k] = dp[:, k] * m + dp[:, k - 1] * mm
        dp = new
    log_cdf = np.log(np.clip(dp.sum(axis=1), 1e-300, 1.0))  # (T,)
    return float(np.logaddexp.reduce(log_wtilde + log_cdf))


def entropy_report(
    aa_probs: np.ndarray,
    weights: np.ndarray,
    n_samples: int,
    batch_size: int,
    device: str,
    seed: int,
    wt_idx: Optional[np.ndarray] = None,
    max_ed: Optional[int] = None,
) -> dict:
    """Compute the entropy / effective-sequence-space report for a design.

    If wt_idx and max_ed are given, additionally reports the entropy restricted to
    the Hamming ball of radius max_ed around the WT (the design regime for the
    edit-distance-constrained ESM2 comparison).
    """
    assert aa_probs.shape[0] == weights.shape[0], "template count mismatch"
    assert aa_probs.shape[-1] == N_AA + 1, "expected 21 AA columns (stop last)"
    log_ptilde, log_wtilde, p_no_stop = condition_full_length(aa_probs, weights)
    mc = mc_entropy(
        log_ptilde, log_wtilde, n_samples, batch_size, device, seed, wt_idx, max_ed
    )
    h_nats, stderr = mc["H_nats"], mc["stderr"]
    ln2 = np.log(2.0)
    report = {
        "H_nats": h_nats,
        "H_bits": h_nats / ln2,
        "mc_stderr_nats": stderr,
        "eff_seq_space": float(np.exp(h_nats)),
        "eff_seq_space_ci95": [
            float(np.exp(h_nats - 1.96 * stderr)),
            float(np.exp(h_nats + 1.96 * stderr)),
        ],
        "p_no_stop": p_no_stop,
        "num_templates": int(len(weights)),
        "n_samples": n_samples,
    }
    if wt_idx is not None:
        log_p_ball = log_p_within_ed_ball(log_ptilde, log_wtilde, wt_idx, max_ed)
        h_ball = (
            mc["ball_neg_mean"] + log_p_ball
        )  # nats: -E[log p | ball] + log P(ball)
        se_ball = mc["ball_neg_stderr"]
        report.update(
            {
                "max_edit_distance": int(max_ed),
                "p_within_ed": float(np.exp(log_p_ball)),  # exact
                "accept_frac_mc": mc["accept_frac"],  # MC cross-check of p_within_ed
                "n_ball_samples": mc["n_ball"],
                "H_ball_nats": h_ball,
                "H_ball_bits": h_ball / ln2,
                "H_ball_stderr_nats": se_ball,
                "eff_seq_space_ed": float(np.exp(h_ball)),
                "eff_seq_space_ed_ci95": [
                    float(np.exp(h_ball - 1.96 * se_ball)),
                    float(np.exp(h_ball + 1.96 * se_ball)),
                ],
            }
        )
    return report


# PGLD adapter


def pgld_design(
    run_dir: str, param_file: str, device: str
) -> Tuple[np.ndarray, np.ndarray, str, list]:
    """Load a PGLD policy and return (AA probs, weights, name, aa_order)."""
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from omegaconf import OmegaConf

    from antibody_config import load_antibody_config
    from constants import AMINO_ACIDS
    from setup import build_policy
    from vis import dna_to_amino_acid_probabilities

    class LandscapeStub:
        def __init__(self, str_length: int):
            self.str_length = str_length
            self.alphabet = ["A", "C", "G", "T"]

    cfg = OmegaConf.load(os.path.join(run_dir, ".hydra", "config.yaml"))
    if cfg.landscape.name.startswith("plm_"):
        ab_cfg = load_antibody_config(cfg.landscape.antibody)
        str_length = ab_cfg.cdrh3_length * 3
    else:
        str_length = cfg.landscape.str_length

    policy = build_policy(cfg, LandscapeStub(str_length), 4, device)
    policy.load_state_dict(
        torch.load(os.path.join(run_dir, param_file), map_location=device)
    )

    template_dna_probs = policy.get_template_dna_probs()  # (T, L, 4)
    aa_probs = np.stack(
        [dna_to_amino_acid_probabilities(t) for t in template_dna_probs], axis=0
    )  # (T, Lc, 21)
    weights = policy.get_probs()[-1].detach().cpu().numpy()  # (T,)
    aa_order = [a for a in AMINO_ACIDS if a != "*"]  # 20 non-stop, column order
    return aa_probs, weights, str(cfg.policy.name), aa_order


# Variational-synthesis adapter


def var_synth_design(run_dir: str) -> Tuple[np.ndarray, np.ndarray, str, list]:
    """Load a trained SynthesisModel and return (AA probs, weights, name, aa_order)."""
    # Deferred: unpickling SynthesisModel requires the var-synth env.
    import configparser
    import io
    import pickle

    from VariationalSynthesis import bio_utils as bu

    torch.set_default_tensor_type(torch.DoubleTensor)

    class CPUUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch.storage" and name == "_load_from_bytes":
                return lambda b: torch.load(
                    io.BytesIO(b), map_location="cpu", weights_only=False
                )
            return super().find_class(module, name)

    config = configparser.ConfigParser()
    config.read(os.path.join(run_dir, "config.cfg"))
    params_file = config["results"]["params_file"]
    with open(params_file, "rb") as f:
        model = CPUUnpickler(f).load()

    assert model.assembly == "fixed", "only fixed assembly supported"
    # per_well_marg: (M, L, 21) per-template per-position AA probs (stop last)
    per_well_marg = torch.exp(model._make_c_marg_mat()).detach().cpu().numpy()
    weights = model.params["w"].detach().cpu().numpy()  # (M,)
    torch.set_default_tensor_type(torch.FloatTensor)
    aa_order = [a for a in bu.alphabets["aa"] if a != "*"]  # 20 non-stop, column order
    return (
        per_well_marg.astype(np.float64),
        weights.astype(np.float64),
        "var_synth",
        aa_order,
    )


# CLI


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute peptide-level entropy of a library design."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pgld_run_dir", help="PGLD policy run dir (pgld_basic env)")
    group.add_argument("--var_synth_run_dir", help="var-synth run dir (var-synth env)")
    parser.add_argument("--param_file", default="parameters.pth")
    parser.add_argument("--n_samples", type=int, default=2_000_000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument(
        "--wt",
        default=None,
        help="WT peptide (AA string); enables edit-distance-ball entropy",
    )
    parser.add_argument("--max_ed", type=int, default=6, help="Hamming radius for --wt")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"

    if args.pgld_run_dir:
        aa_probs, weights, name, aa_order = pgld_design(
            args.pgld_run_dir, args.param_file, device
        )
        run_dir = args.pgld_run_dir
    else:
        aa_probs, weights, name, aa_order = var_synth_design(args.var_synth_run_dir)
        run_dir = args.var_synth_run_dir

    wt_idx, max_ed = None, None
    if args.wt:
        assert (
            len(args.wt) == aa_probs.shape[1]
        ), f"WT length {len(args.wt)} != peptide length {aa_probs.shape[1]}"
        wt_idx = np.array([aa_order.index(a) for a in args.wt], dtype=int)
        max_ed = args.max_ed

    report = entropy_report(
        aa_probs,
        weights,
        args.n_samples,
        args.batch_size,
        device,
        args.seed,
        wt_idx,
        max_ed,
    )
    report["label"] = args.label or os.path.basename(run_dir.rstrip("/"))
    report["run_dir"] = run_dir
    report["design"] = name

    print(json.dumps(report, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved report to: {args.out}")


if __name__ == "__main__":
    main()
