"""
aggregate_paper_results.py

Aggregates experiment results and generates paper plots.

Usage:
    python aggregate_paper_results.py diversity_fitness --results-dir <path>
    python aggregate_paper_results.py credit_assignment --results-dir <path>

Method-comparison and template-sweep plots: see final_paper_plots.py.
"""

import argparse
import glob
import os
import pickle
import sys
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from landscapes import (
    TFBindingLandscape,
    MHCflurryLandscape,
    RiboswitchLandscape,
    GB1FitnessLandscape,
    PhoQFitnessLandscape,
    DHFRFitnessLandscape,
    TrpBFitnessLandscape,
    generate_uniform_random_sequences,
    generate_nnk_sequences,
    enumerate_sequences_and_score,
    get_top_proportion_threshold,
)
from plot_config import PAPER_FIGSIZE, apply_paper_style
from constants import PAPER_MARGINS, PAPER_COLORS, DEFAULT_DIVERSITY_THRESHOLD
from vis import _load_cache, _save_cache, dna_to_amino_acid_probabilities
from policies import IupacBaseOptimizer, ArbitraryBaseOptimizer
from diversity_utils import compute_diversity_with_threshold, greedy_mean_fitness
from compute_library_entropy import entropy_report, pgld_design

apply_paper_style()

# MC sample budget for the effective-diversity exp(H) estimate (per run).
ENTROPY_MC_SAMPLES = 2_000_000

LANDSCAPE_DISPLAY_NAMES = {
    "tf_binding": "TF Binding",
    "mhc_flurry": "MHCI",
    "riboswitch": "Switch",
    "gb1": "GB1",
    "phoq": "PhoQ",
    "trpb": "TrpB",
    "dhfr": "DHFR",
}

# Landscape groupings for plotting
LANDSCAPE_GROUPS = {
    "dna": ["tf_binding", "dhfr"],  # DNA, ground truth lookup
    "protein": ["gb1", "trpb", "phoq"],  # Protein, ground truth lookup
    "ml": ["riboswitch", "mhc_flurry"],  # ML model oracles
}

GROUP_TITLES = {
    "dna": "DNA Landscapes",
    "protein": "Protein Landscapes",
    "ml": "ML Oracle Landscapes",
}

METHOD_COLORS = {
    "random": PAPER_COLORS["NNK"],
    "arbitrary": PAPER_COLORS["Arbitrary"],
    "iupac": PAPER_COLORS["IUPAC"],
    "iupac_8": PAPER_COLORS["IUPAC_8"],
}


def compute_theoretical_diversity(
    run_dir: str, num_codons: int = 4, num_aas: int = 20, threshold: float = 0.01
) -> Optional[int]:
    """
    Compute theoretical diversity (union diversity) for multi-template policies.

    Counts the number of unique peptide sequences that have probability above threshold
    in at least one template (union across all templates).

    Args:
        run_dir: Directory containing parameters.pth and .hydra/config.yaml
        num_codons: Number of codon positions (peptide length)
        num_aas: Number of amino acids (excluding stop codon)
        threshold: Minimum probability for a sequence to be counted

    Returns:
        Number of unique sequences with p > threshold, or None if computation fails
    """
    params_path = os.path.join(run_dir, "parameters.pth")
    config_path = os.path.join(run_dir, ".hydra", "config.yaml")

    if not os.path.exists(params_path) or not os.path.exists(config_path):
        return None

    try:
        cfg = OmegaConf.load(config_path)
        state_dict = torch.load(params_path, map_location="cpu")

        str_length = num_codons * 3  # DNA length
        policy_name = cfg.policy.name

        if policy_name == "arbitrary":
            policy = ArbitraryBaseOptimizer(
                str_length=str_length,
                alphabet_size=4,
                batch_size=1,
                num_templates=cfg.policy.num_templates,
                device="cpu",
                uniform_template_mix=cfg.policy.get("uniform_template_mix", False),
            )
            policy.load_state_dict(state_dict)
            template_dna_probs = policy.get_template_probs().detach().cpu().numpy()
        else:
            policy = IupacBaseOptimizer(
                str_length=str_length,
                alphabet_size=4,
                batch_size=1,
                num_templates=cfg.policy.num_templates,
                device="cpu",
                uniform_template_mix=cfg.policy.get("uniform_template_mix", False),
            )
            policy.load_state_dict(state_dict)
            template_dna_probs = policy.get_template_dna_probs()  # (T, L, 4)

        # Convert each template to AA probs: (T, num_codons, 21)
        template_aa_probs = np.stack(
            [
                dna_to_amino_acid_probabilities(template_dna_probs[t])
                for t in range(policy.num_templates)
            ]
        )
        # Only keep first 20 AAs (exclude stop codon)
        template_aa_probs = template_aa_probs[:, :, :num_aas]
        template_aa_probs = template_aa_probs / template_aa_probs.sum(
            axis=-1, keepdims=True
        )

        # For each template, compute which peptides have p > threshold
        # A peptide is "possible" in template t if all positions have p >= threshold
        # Union diversity = peptides possible in ANY template
        all_peptides = np.array(list(product(range(num_aas), repeat=num_codons)))

        # For each template, check if peptide is possible (all positions above threshold)
        possible_in_any = np.zeros(len(all_peptides), dtype=bool)

        for t in range(policy.num_templates):
            # Check each position: is the AA probability >= threshold?
            position_ok = np.ones(len(all_peptides), dtype=bool)
            for pos in range(num_codons):
                aa_probs_at_pos = template_aa_probs[t, pos, all_peptides[:, pos]]
                position_ok &= aa_probs_at_pos >= threshold
            possible_in_any |= position_ok

        diversity = int(np.sum(possible_in_any))
        return diversity

    except Exception as e:
        print(f"  Error computing diversity for {run_dir}: {e}")
        return None


def get_landscape_display_name(landscape: str) -> str:
    """Get the display name for a landscape, with proper capitalization."""
    return LANDSCAPE_DISPLAY_NAMES.get(landscape, landscape.replace("_", " ").title())


def parse_theoretical_diversity(run_dir: str) -> Optional[int]:
    """Parse union diversity from theoretical_diversity.txt."""
    div_file = os.path.join(run_dir, "vis", "theoretical_diversity.txt")
    if not os.path.exists(div_file):
        return None
    try:
        sum_diversity = None
        with open(div_file) as f:
            for line in f:
                # For t>1: "Union diversity (exact):       158,624"
                if (
                    "Union diversity (exact)" in line
                    or "Union diversity (approx)" in line
                ):
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val_str = parts[1].strip().split()[0].replace(",", "")
                        return int(float(val_str))
                # For t=1: union is skipped, use sum instead
                if "Sum of individual diversities:" in line:
                    parts = line.split(":")
                    if len(parts) >= 2:
                        val_str = parts[1].strip().replace(",", "")
                        sum_diversity = int(float(val_str))
        # Return sum if no union found (t=1 case)
        return sum_diversity
    except Exception:
        pass
    return None


def load_experiment_results(base_dir: str) -> pd.DataFrame:
    """Load all metrics.csv files from experiment directories."""
    results = []

    for metrics_file in glob.glob(f"{base_dir}/**/metrics.csv", recursive=True):
        run_dir = os.path.dirname(metrics_file)
        config_path = os.path.join(run_dir, ".hydra", "config.yaml")

        if not os.path.exists(config_path):
            continue

        try:
            cfg = OmegaConf.load(config_path)
            df = pd.read_csv(metrics_file)

            # Filter to exact eval rows only (ignore SGT estimated rows)
            if "eval_type" in df.columns:
                df = df[df["eval_type"] == "exact"]

            if len(df) == 0:
                continue

            final_row = df.iloc[-1].to_dict()

            # Detect ablation runs by path
            is_ablation = "ablation" in run_dir.lower()

            result = {
                "run_dir": run_dir,
                "landscape": cfg.landscape.name,
                "policy": cfg.policy.name,
                "num_templates": cfg.policy.num_templates,
                "lambda": cfg.training.entropy.lambda_,
                "batch_size": cfg.training.sampling.batch_size,
                "reward_type": getattr(cfg.training, "reward_type", "hits"),
                "is_ablation": is_ablation,
                **final_row,
            }

            # Load theoretical diversity from file
            theo_div = parse_theoretical_diversity(run_dir)
            if theo_div is not None:
                result["theoretical_diversity"] = theo_div

            if hasattr(cfg.landscape, "objectives"):
                obj_names = list(cfg.landscape.objectives.keys())
                if len(obj_names) == 1:
                    result["objective"] = obj_names[0]

            results.append(result)
        except Exception as e:
            print(f"Error loading {metrics_file}: {e}")
            continue

    return pd.DataFrame(results)


def compute_baseline_stats(
    landscape_name: str,
    objective: Optional[str],
    n_samples: int,
    threshold: float,
    n_repeats: int = 10,
    invalid_score: float = -20.0,
) -> Dict[str, float]:
    """Compute baseline stats (hits, mean fitness) for random/NNK sampling."""
    invalid_scores = {objective: invalid_score} if objective else {}

    if landscape_name == "tf_binding":
        landscape = TFBindingLandscape(str_length=8, objectives=[objective])
    elif landscape_name == "mhc_flurry":
        landscape = MHCflurryLandscape(
            str_length=27, objectives=[objective], invalid_scores=invalid_scores
        )
    elif landscape_name == "riboswitch":
        landscape = RiboswitchLandscape(
            str_length=10,
            objectives=[objective],
            device="cpu",
            oracle_batch_size=1000,
            oracle_dir=os.environ.get("PGLD_RIBOSWITCH_ORACLE_DIR"),
        )
    elif landscape_name == "gb1":
        data_path = os.path.join(
            os.environ.get("PGLD_DATASETS_DIR", ""), "gb1_fitness.csv"
        )
        landscape = GB1FitnessLandscape(
            str_length=12,
            objectives=[objective],
            data_path=data_path,
            invalid_scores=invalid_scores,
        )
    elif landscape_name == "phoq":
        data_path = os.path.join(os.environ.get("PGLD_DATASETS_DIR", ""), "PhoQ.xlsx")
        landscape = PhoQFitnessLandscape(
            str_length=12,
            objectives=[objective],
            data_path=data_path,
            invalid_scores=invalid_scores,
        )
    elif landscape_name == "trpb":
        data_path = os.path.join(os.environ.get("PGLD_DATASETS_DIR", ""), "trpb.csv")
        landscape = TrpBFitnessLandscape(
            str_length=12,
            objectives=[objective],
            data_path=data_path,
            invalid_scores=invalid_scores,
        )
    elif landscape_name == "dhfr":
        data_path = os.path.join(os.environ.get("PGLD_DATASETS_DIR", ""), "dhfr.csv")
        landscape = DHFRFitnessLandscape(
            str_length=9,
            objectives=[objective],
            data_path=data_path,
            invalid_scores=invalid_scores,
        )
    else:
        raise ValueError(f"Unknown landscape: {landscape_name}")

    hits_list = []
    fitness_list = []
    for _ in range(n_repeats):
        # Peptide landscapes use NNK, DNA landscapes use uniform random
        if landscape.is_peptide:
            sequences = generate_nnk_sequences(landscape.str_length, n_samples)
        else:
            sequences = generate_uniform_random_sequences(
                landscape.alphabet, landscape.str_length, n_samples
            )
        # Score sequences and compute both hits and mean fitness
        scores = np.array(landscape.get_fitness_values(sequences, objective=objective))
        hits = int(np.sum(scores >= threshold))
        mean_fitness = float(np.mean(scores))
        hits_list.append(hits)
        fitness_list.append(mean_fitness)

    return {
        "hits_mean": np.mean(hits_list),
        "hits_std": np.std(hits_list),
        "fitness_mean": np.mean(fitness_list),
        "fitness_std": np.std(fitness_list),
    }


def compute_baseline_hits(
    landscape_name: str,
    objective: Optional[str],
    n_samples: int,
    threshold: float,
    n_repeats: int = 10,
) -> tuple[float, float]:
    """Compute baseline hit rate for random/NNK sampling (legacy wrapper)."""
    stats = compute_baseline_stats(
        landscape_name, objective, n_samples, threshold, n_repeats
    )
    return stats["hits_mean"], stats["hits_std"]


def get_random_theoretical_diversity(landscape_name: str) -> int:
    """Get theoretical diversity for uniform random sampling (full sequence space)."""
    # DNA landscapes: 4^L
    if landscape_name == "tf_binding":
        return 4**8
    elif landscape_name == "dhfr":
        return 4**9
    elif landscape_name == "riboswitch":
        return 4**10
    # Protein landscapes: 20^L (using peptide length, not full codon length)
    elif landscape_name == "gb1":
        return 20**4
    elif landscape_name == "trpb":
        return 20**4
    elif landscape_name == "phoq":
        return 20**4
    elif landscape_name == "mhc_flurry":
        # Sum of 20^L for L in [5, 9] (valid peptide lengths with early stops)
        return sum(20**L for L in range(5, 10))
    else:
        raise ValueError(f"Unknown landscape: {landscape_name}")


def get_threshold_for_landscape(
    landscape_name: str, objective: str, top_prop: float = 0.05
) -> float:
    """Get hit threshold for a landscape objective."""
    if landscape_name == "tf_binding":
        landscape = TFBindingLandscape(str_length=8, objectives=[objective])
        sequence_scores = enumerate_sequences_and_score(landscape, objective=objective)
        return get_top_proportion_threshold(sequence_scores, top_prop)
    elif landscape_name == "mhc_flurry":
        return 0.995
    elif landscape_name == "riboswitch":
        return 6.0
    elif landscape_name == "gb1":
        return 1.0
    elif landscape_name == "phoq":
        return 3.28744733333
    elif landscape_name == "trpb":
        return 0.09171322827781170
    elif landscape_name == "dhfr":
        return 0.85689032917647
    else:
        raise ValueError(f"Unknown landscape: {landscape_name}")


# =============================================================================
# Plot 1: Bar chart - Random vs Arbitrary vs IUPAC
# =============================================================================


def _plot1_for_landscapes(
    results_df: pd.DataFrame,
    landscapes: List[str],
    output_path: str,
    title: str,
    eval_batch_size: Optional[int] = None,
    compute_baselines: bool = True,
) -> None:
    """
    Bar chart comparing random, 1-template arbitrary, and 1-template IUPAC.
    For a specific set of landscapes.
    """
    # Filter out ablation runs
    df = results_df
    if "is_ablation" in df.columns:
        df = df[~df["is_ablation"]]

    # Filter to only landscapes that exist in results
    available = [l for l in landscapes if l in df["landscape"].unique()]
    if not available:
        print(f"No data for landscapes {landscapes}, skipping")
        return

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    plt.subplots_adjust(**PAPER_MARGINS)

    x = np.arange(len(available))
    width = 0.2

    random_pct, random_stds = [], []
    arbitrary_pct, arbitrary_stds = [], []
    iupac_pct, iupac_stds = [], []
    iupac_8_pct, iupac_8_stds = [], []

    for landscape in available:
        landscape_df = df[df["landscape"] == landscape]
        batch_size = eval_batch_size or int(landscape_df["batch_size"].iloc[0])

        # 1-template Arbitrary results (lambda=0, from arbitrary/ not lambda sweeps)
        arb_df = landscape_df[
            (landscape_df["policy"] == "arbitrary")
            & (landscape_df["num_templates"] == 1)
            & (landscape_df["lambda"] == 0.0)
            & (landscape_df["run_dir"].str.contains("arbitrary/"))
        ]
        if len(arb_df) > 0:
            arb_pct_values = arb_df["unique_hits"] / batch_size * 100
            arb_mean = arb_pct_values.mean()
            arb_std = arb_pct_values.std()
        else:
            arb_mean, arb_std = 0, 0

        # 1-template IUPAC results (lambda=0, from template_sweep/ not lambda sweeps)
        iupac_df = landscape_df[
            (landscape_df["policy"] == "iupac")
            & (landscape_df["num_templates"] == 1)
            & (landscape_df["lambda"] == 0.0)
            & (landscape_df["run_dir"].str.contains("template_sweep/"))
        ]
        if len(iupac_df) > 0:
            iupac_pct_values = iupac_df["unique_hits"] / batch_size * 100
            iupac_mean = iupac_pct_values.mean()
            iupac_std = iupac_pct_values.std()
        else:
            iupac_mean, iupac_std = 0, 0

        # 8-template IUPAC results (lambda=0, from template_sweep/)
        iupac_8_df = landscape_df[
            (landscape_df["policy"] == "iupac")
            & (landscape_df["num_templates"] == 8)
            & (landscape_df["lambda"] == 0.0)
            & (landscape_df["run_dir"].str.contains("template_sweep/"))
        ]
        if len(iupac_8_df) > 0:
            iupac_8_pct_values = iupac_8_df["unique_hits"] / batch_size * 100
            iupac_8_mean = iupac_8_pct_values.mean()
            iupac_8_std = iupac_8_pct_values.std()
        else:
            iupac_8_mean, iupac_8_std = 0, 0

        # Random baseline - aggregate across all objectives like the other methods
        if compute_baselines:
            objectives = (
                landscape_df["objective"].unique().tolist()
                if "objective" in landscape_df.columns
                else [None]
            )
            try:
                all_rand_pcts = []
                for objective in objectives:
                    threshold = get_threshold_for_landscape(landscape, objective)
                    obj_mean, _ = compute_baseline_hits(
                        landscape, objective, batch_size, threshold
                    )
                    all_rand_pcts.append(obj_mean / batch_size * 100)
                rand_mean = np.mean(all_rand_pcts)
                rand_std = np.std(all_rand_pcts)
            except Exception as e:
                print(f"Error computing baseline for {landscape}: {e}")
                rand_mean, rand_std = 0, 0
        else:
            rand_mean, rand_std = 0, 0

        random_pct.append(rand_mean)
        random_stds.append(rand_std)
        arbitrary_pct.append(arb_mean)
        arbitrary_stds.append(arb_std if not np.isnan(arb_std) else 0)
        iupac_pct.append(iupac_mean)
        iupac_stds.append(iupac_std if not np.isnan(iupac_std) else 0)
        iupac_8_pct.append(iupac_8_mean)
        iupac_8_stds.append(iupac_8_std if not np.isnan(iupac_8_std) else 0)

    ax.bar(
        x - 1.5 * width,
        random_pct,
        width,
        yerr=random_stds,
        label="Random/NNK",
        color=METHOD_COLORS["random"],
        capsize=3,
    )
    ax.bar(
        x - 0.5 * width,
        arbitrary_pct,
        width,
        yerr=arbitrary_stds,
        label="Arbitrary (1 template)",
        color=METHOD_COLORS["arbitrary"],
        capsize=3,
    )
    ax.bar(
        x + 0.5 * width,
        iupac_pct,
        width,
        yerr=iupac_stds,
        label="IUPAC (1 template)",
        color=METHOD_COLORS["iupac"],
        capsize=3,
    )
    ax.bar(
        x + 1.5 * width,
        iupac_8_pct,
        width,
        yerr=iupac_8_stds,
        label="IUPAC (8 templates)",
        color=METHOD_COLORS["iupac_8"],
        capsize=3,
    )

    ax.set_ylabel("Hits / Batch Size (%)")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([get_landscape_display_name(l) for l in available])
    ax.grid(axis="y", alpha=0.3)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_base = Path(output_path).with_suffix("")

    # Save main plot without legend
    plt.savefig(f"{output_base}.pdf", format="pdf")
    plt.savefig(f"{output_base}.png", format="png", dpi=150)
    plt.close()
    print(f"Saved: {output_base}.pdf and {output_base}.png")

    # Save legend as separate file
    fig_legend, ax_legend = plt.subplots(figsize=(5, 1.5))
    ax_legend.axis("off")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS["random"]),
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS["arbitrary"]),
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS["iupac"]),
        plt.Rectangle((0, 0), 1, 1, color=METHOD_COLORS["iupac_8"]),
    ]
    labels = [
        "Random/NNK",
        "Arbitrary (1 template)",
        "IUPAC (1 template)",
        "IUPAC (8 templates)",
    ]
    ax_legend.legend(handles, labels, loc="center", ncol=4, frameon=True)
    plt.savefig(f"{output_base}_legend.pdf", format="pdf", bbox_inches="tight")
    plt.savefig(f"{output_base}_legend.png", format="png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_base}_legend.pdf and {output_base}_legend.png")


def plot1_random_vs_policies(
    results_df: pd.DataFrame,
    output_path: str,
    eval_batch_size: Optional[int] = None,
    compute_baselines: bool = True,
) -> None:
    """
    Bar chart comparing random, 1-template arbitrary, 1-template IUPAC, and 8-template IUPAC.
    Creates 3 separate plots: DNA, Protein, and ML landscapes.
    """
    output_base = Path(output_path).with_suffix("")
    output_dir = output_base.parent
    base_name = output_base.name

    for group_name, group_landscapes in LANDSCAPE_GROUPS.items():
        group_output = output_dir / f"{base_name}_{group_name}"
        _plot1_for_landscapes(
            results_df,
            group_landscapes,
            str(group_output),
            GROUP_TITLES[group_name],
            eval_batch_size,
            compute_baselines,
        )


# =============================================================================
# Plot 2: Line graph - Template count sweep
# =============================================================================


def _plot2_for_landscapes(
    results_df: pd.DataFrame,
    landscapes: List[str],
    output_path: str,
    title: str,
) -> None:
    """Line graph of template count vs % unique hits for a specific set of landscapes."""
    # Filter out ablation runs
    df = results_df
    if "is_ablation" in df.columns:
        df = df[~df["is_ablation"]]

    # Only include template_sweep runs (not lambda sweeps)
    template_df = df[
        (df["lambda"] == 0.0)
        & (df["policy"] == "iupac")
        & (df["run_dir"].str.contains("template_sweep/"))
    ].copy()

    # Filter to only landscapes that exist in results
    available = [l for l in landscapes if l in template_df["landscape"].unique()]
    if not available:
        print(f"No data for landscapes {landscapes}, skipping")
        return

    template_df = template_df[template_df["landscape"].isin(available)]

    # Calculate % unique hits if batch_size is available
    if "batch_size" in template_df.columns:
        template_df["pct_unique_hits"] = (
            template_df["unique_hits"] / template_df["batch_size"] * 100
        )
        y_col = "pct_unique_hits"
        y_label = "Hits / Batch Size (%)"
    else:
        y_col = "unique_hits"
        y_label = "Unique Hits"

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    plt.subplots_adjust(**PAPER_MARGINS)

    colors = plt.cm.tab10(np.linspace(0, 1, len(available)))
    all_templates = sorted(template_df["num_templates"].unique())

    for i, landscape in enumerate(available):
        landscape_df = template_df[template_df["landscape"] == landscape]
        grouped = landscape_df.groupby("num_templates")[y_col].agg(["mean", "std"])
        grouped = grouped.reset_index()

        ax.errorbar(
            grouped["num_templates"],
            grouped["mean"],
            yerr=grouped["std"],
            marker="o",
            label=get_landscape_display_name(landscape),
            color=colors[i],
            capsize=3,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("# Templates")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(all_templates)
    ax.set_xticklabels([str(t) for t in all_templates])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_base = Path(output_path).with_suffix("")
    plt.savefig(f"{output_base}.pdf", format="pdf")
    plt.savefig(f"{output_base}.png", format="png", dpi=150)
    plt.close()
    print(f"Saved: {output_base}.pdf and {output_base}.png")


def plot2_template_sweep(results_df: pd.DataFrame, output_path: str) -> None:
    """Line graph of template count vs % unique hits, split by landscape group."""
    output_base = Path(output_path).with_suffix("")
    output_dir = output_base.parent
    base_name = output_base.name

    for group_name, group_landscapes in LANDSCAPE_GROUPS.items():
        group_output = output_dir / f"{base_name}_{group_name}"
        _plot2_for_landscapes(
            results_df,
            group_landscapes,
            str(group_output),
            GROUP_TITLES[group_name],
        )


# =============================================================================
# Plot: Diversity vs Fitness
# =============================================================================


def mc_effective_seq_space(
    run_dir: str,
    device: torch.device,
    n_samples: int = ENTROPY_MC_SAMPLES,
    seed: int = 0,
) -> Optional[float]:
    """
    Effective sequence space exp(H) of the stop-free peptide distribution
    (threshold-free), estimated from `n_samples` Monte-Carlo draws.

    Always computed over the GREEDY HARDENED library: pgld_design builds the AA
    tensor from policy.get_template_dna_probs(), which for bottle policies is the
    argmax-bottle design (identical to what sample(greedy=True) draws from) and
    for the arbitrary policy is its single per-position design. Returns None on
    failure.

    NOTE: this measures the effective size of the MAX-LENGTH, full-length,
    stop-free peptide distribution ONLY (e.g. 9-mers when min/max peptide len
    is 8/9). entropy_report conditions on 'no stop at any codon position', so
    peptides truncated by an early stop (e.g. 8-mers) are excluded. This is
    deliberately NOT the combined multi-length space used by the
    theoretical-diversity axis (_multilength_peptide_diversity); when
    min_peptide_len < max_peptide_len the two x-axes are not directly comparable.
    """
    try:
        aa_probs, weights, _, _ = pgld_design(run_dir, "parameters.pth", str(device))
        report = entropy_report(
            aa_probs, weights, n_samples,
            batch_size=4096, device=str(device), seed=seed,
        )
        return float(report["eff_seq_space"])
    except Exception as e:
        print(f"  Error computing entropy for {run_dir}: {e}")
        return None


def plot6_diversity_fitness_pareto(
    results_df: pd.DataFrame,
    output_dir: str,
    effective_diversity: bool = False,
    from_cache: bool = False,
) -> None:
    """
    Pareto curves: library diversity (x) vs greedy mean fitness (y) for varying
    lambda. Different markers per (policy, num_templates), method-based coloring.

    Both axes are computed on the GREEDY HARDENED library so they are consistent:
    the y-axis greedy-samples the hardened policy for its mean fitness, and the
    x-axis measures the same hardened design. The x-axis is either the effective
    diversity exp(H) (threshold-free entropy) or the theoretical multi-length
    diversity at DEFAULT_DIVERSITY_THRESHOLD, selected by `effective_diversity`.

    Args:
        results_df: DataFrame with experiment results.
        output_dir: Directory to save plots.
        effective_diversity: If True, x-axis is effective diversity exp(H) of the
            greedy hardened library; otherwise theoretical (multi-length) diversity.
        from_cache: If True, load the recomputed df from the vis cache instead of
            recomputing (fast re-plot for figure tweaks).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter out ablation runs
    filtered_df = results_df
    if "is_ablation" in filtered_df.columns:
        filtered_df = filtered_df[~filtered_df["is_ablation"]]

    # This is the average-score Pareto: keep only reward_type=average (lambda) sweeps,
    # excluding co-located hits runs (e.g. template_sweep/) that share (policy, t) keys.
    if "reward_type" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["reward_type"] == "average"]

    if "theoretical_diversity" not in filtered_df.columns:
        print("Warning: No theoretical_diversity data found")
        return

    df = filtered_df[filtered_df["theoretical_diversity"].notna()].copy()
    if len(df) == 0:
        print("Warning: No rows with theoretical_diversity data")
        return

    # Cache the recomputed df so figure tweaks re-plot instantly (vis cache convention).
    # Tag by the x-axis metric, the only thing that changes the recomputed numbers.
    metric = "effective" if effective_diversity else "theoretical"
    cache_key = os.path.join(output_dir, f"plot6_recomputed_{metric}.png")
    loaded_from_cache = False
    if from_cache:
        cached = _load_cache(cache_key)
        if cached is not None:
            df = cached["df"]
            loaded_from_cache = True
            print("Loaded plot data from cache")

    # Recompute both axes on the greedy hardened library so they are consistent.
    if not loaded_from_cache:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_desc = "effective diversity exp(H)" if effective_diversity else "theoretical diversity"
        print(f"Using device: {device}")
        print(f"Recomputing greedy mean fitness and {x_desc} on the greedy library...")
        for idx in df.index:
            run_dir = df.loc[idx, "run_dir"]
            config_path = os.path.join(run_dir, ".hydra", "config.yaml")
            if not os.path.exists(config_path):
                continue
            cfg = OmegaConf.load(config_path)

            score = greedy_mean_fitness(run_dir, cfg, device)
            if score is not None:
                df.loc[idx, "mean_fitness"] = score

            if effective_diversity:
                x_val = mc_effective_seq_space(run_dir, device)
            else:
                x_val = compute_diversity_with_threshold(
                    run_dir, cfg, device, DEFAULT_DIVERSITY_THRESHOLD
                )
            if x_val is not None:
                df.loc[idx, "theoretical_diversity"] = x_val
        print("Recomputation complete")
        _save_cache({"df": df}, cache_key)

    all_markers = ["o", "s", "^", "D", "v", "p", "h", "*"]
    all_landscapes = sorted(df["landscape"].unique())

    for landscape in all_landscapes:
        landscape_df = df[df["landscape"] == landscape]
        if "mean_fitness" not in landscape_df.columns:
            print(f"Warning: Missing mean_fitness for {landscape}, skipping")
            continue

        apply_paper_style()
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

        # Collect data per (policy, num_templates) combination
        series_data = {}
        plotted_any = False

        # Only show t=1, t=16, and arbitrary
        allowed_series = {("arbitrary", 1), ("iupac", 1), ("iupac", 16)}

        for policy in landscape_df["policy"].unique():
            policy_df = landscape_df[landscape_df["policy"] == policy]

            for t in sorted(policy_df["num_templates"].unique()):
                if (policy, t) not in allowed_series:
                    continue
                t_df = policy_df[policy_df["num_templates"] == t]
                n_lambdas = t_df["lambda"].nunique()
                if n_lambdas <= 1:
                    continue

                grouped = (
                    t_df.groupby("lambda")[["theoretical_diversity", "mean_fitness"]]
                    .mean()
                    .reset_index()
                )

                series_key = (policy, t)
                series_data[series_key] = {
                    "x": np.array(grouped["theoretical_diversity"].tolist()),
                    "y": np.array(grouped["mean_fitness"].tolist()),
                }
                plotted_any = True

        if not plotted_any:
            print(f"Warning: No data with >1 lambda for {landscape}, skipping")
            plt.close()
            continue

        # Plot each (policy, num_templates) as separate series with method-based colors
        for idx, ((policy, num_t), data) in enumerate(series_data.items()):
            marker = all_markers[idx % len(all_markers)]
            policy_name = "IUPAC" if policy == "iupac" else policy.capitalize()
            label = f"{policy_name} t={num_t}"

            # Color by method
            if policy == "arbitrary":
                color = PAPER_COLORS["Arbitrary"]
            elif policy == "iupac" and num_t == 1:
                color = PAPER_COLORS["IUPAC"]
            else:
                # IUPAC t=8, t=16, etc. use green
                color = PAPER_COLORS["IUPAC_8"]

            ax.scatter(
                data["x"],
                data["y"],
                c=color,
                s=150,
                marker=marker,
                edgecolors="black",
                linewidths=0.5,
                alpha=0.8,
                label=label,
            )

        x_label = "Effective Diversity" if effective_diversity else "Theoretical Diversity"
        ax.set_xscale("log")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Mean Fitness")
        ax.set_title(f"{get_landscape_display_name(landscape)}: Fitness-Diversity")
        ax.grid(alpha=0.3)

        # Save log scale version
        plt.subplots_adjust(**PAPER_MARGINS)
        output_base = os.path.join(output_dir, f"plot6_{landscape}")
        plt.savefig(f"{output_base}.pdf", format="pdf")
        plt.savefig(f"{output_base}.png", format="png", dpi=150)
        print(f"Saved: {output_base}.pdf and {output_base}.png")

        # Save linear scale version. Fold the magnitude into the label so the
        # 1eN offset text does not overlap the (large) axis title.
        ax.set_xscale("linear")
        xmax = ax.get_xlim()[1]
        if xmax > 0:
            exp = int(np.floor(np.log10(xmax)))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v / 10**exp:g}"))
            ax.set_xlabel(f"{x_label} (×10$^{{{exp}}}$)")
        plt.savefig(f"{output_base}_linear.pdf", format="pdf")
        plt.savefig(f"{output_base}_linear.png", format="png", dpi=150)
        plt.close()
        print(f"Saved: {output_base}_linear.pdf and {output_base}_linear.png")

        # Save legend as separate file
        if len(series_data) > 1:
            fig_legend, ax_legend = plt.subplots(figsize=(4, 2))
            ax_legend.axis("off")
            handles = []
            labels = []
            for idx, ((policy, num_t), data) in enumerate(series_data.items()):
                marker = all_markers[idx % len(all_markers)]
                policy_name = "IUPAC" if policy == "iupac" else policy.capitalize()
                label = f"{policy_name} t={num_t}"
                # Same color logic as main plot
                if policy == "arbitrary":
                    color = PAPER_COLORS["Arbitrary"]
                elif policy == "iupac" and num_t == 1:
                    color = PAPER_COLORS["IUPAC"]
                else:
                    color = PAPER_COLORS["IUPAC_8"]
                handle = ax_legend.scatter(
                    [],
                    [],
                    marker=marker,
                    s=150,
                    c=color,
                    edgecolors="black",
                    linewidths=0.5,
                    label=label,
                )
                handles.append(handle)
                labels.append(label)
            ax_legend.legend(handles, labels, loc="center", ncol=2, frameon=True)
            plt.savefig(f"{output_base}_legend.pdf", format="pdf", bbox_inches="tight")
            plt.savefig(
                f"{output_base}_legend.png", format="png", dpi=150, bbox_inches="tight"
            )
            plt.close()
            print(f"Saved: {output_base}_legend.pdf and {output_base}_legend.png")


# =============================================================================
# Plot: Credit Assignment (MHC LR Sweep Reward Curves)
# =============================================================================


# Credit-assignment visual encoding: solid blue = per-template, dashed red = global.
_CREDIT_LINESTYLES = {"template": "-", "batch": "--"}
_CREDIT_COLORMAPS = {"template": plt.cm.Blues, "batch": plt.cm.Reds}
_CREDIT_LABELS = {"template": "Per-template", "batch": "Global"}


def _plot_best_credit_curve(
    runs: List[Dict],
    series_fn: Callable[[Dict], Optional[pd.Series]],
    ylabel: str,
    title: str,
    out_path_base: str,
    smoothing_window: int = 50,
    robust_ylim: bool = False,
) -> None:
    """Plot the best per-iteration curve for each credit unit (template vs batch).

    For every run, ``series_fn`` returns the per-iteration quantity to plot (or
    None to skip). The run with the highest final value is selected per unit,
    smoothed with a rolling mean, and drawn. Saves ``<out_path_base>.{pdf,png}``.

    When ``robust_ylim`` is set, the y-axis is framed on the post-burn-in
    (1st-99th percentile) range of the plotted curves, so a large early transient
    (e.g. the global entropy objective starting near -30000) does not swamp the
    steady-state behaviour. Use it for the entropy / combined-objective curves.
    """
    best_runs: Dict[str, Dict] = {}  # unit -> {run, series, final}
    for run in runs:
        series = series_fn(run)
        if series is None:
            continue
        unit = run["reward_unit"]
        # Rank by the mean of the final 10% of iterations, not a single noisy
        # endpoint (the global full_mixture objective is high-variance per step).
        tail = max(1, len(series) // 10)
        final = float(series.iloc[-tail:].mean())
        if unit not in best_runs or final > best_runs[unit]["final"]:
            best_runs[unit] = {"run": run, "series": series, "final": final}

    if not best_runs:
        print(f"  No plottable runs for '{title}'")
        return

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    plt.subplots_adjust(**PAPER_MARGINS)
    smoothed_curves = []
    for unit, data in best_runs.items():
        smoothed = data["series"].rolling(window=smoothing_window, min_periods=1).mean()
        smoothed_curves.append(smoothed.to_numpy())
        ax.plot(
            smoothed,
            linestyle=_CREDIT_LINESTYLES[unit],
            color=_CREDIT_COLORMAPS[unit](0.7),
            linewidth=2,
        )
        print(
            f"  {_CREDIT_LABELS[unit]}: best LR={data['run']['lr']}, final={data['final']:.4f}"
        )

    # Frame the y-axis on steady-state behaviour, ignoring the early transient.
    if robust_ylim:
        burn_in = min(2 * smoothing_window, max(len(c) for c in smoothed_curves) // 2)
        vals = np.concatenate([c[burn_in:] for c in smoothed_curves])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = np.percentile(vals, [1, 99])
            pad = 0.08 * (hi - lo) if hi > lo else (abs(hi) * 0.1 + 1.0)
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    from matplotlib.lines import Line2D

    handles, labels = [], []
    for unit in ["template", "batch"]:
        if unit in best_runs:
            handles.append(
                Line2D(
                    [0], [0],
                    color=_CREDIT_COLORMAPS[unit](0.7),
                    linestyle=_CREDIT_LINESTYLES[unit],
                    linewidth=2,
                )
            )
            labels.append(_CREDIT_LABELS[unit])
    ax.legend(handles, labels, loc="lower right", title="Credit:")

    plt.savefig(f"{out_path_base}.pdf", format="pdf")
    plt.savefig(f"{out_path_base}.png", format="png", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path_base}.pdf")


def plot9_lr_sweep_curves(base_dir: str, output_dir: str) -> None:
    """
    Plot per-iteration curves for MHC LR sweep (credit-assignment) experiments.

    For each experiment it compares template vs batch credit assignment and
    always emits the reward curve. For experiments with an entropy term
    (lambda_ > 0, i.e. the mean-fitness sweep) it additionally emits an entropy
    curve and a combined full-objective (mean + lambda * entropy) curve so the
    effect of per-template marginal-gain credit on the full objective is visible.

    Experiments:
    - mean_entropy (mhc_flurry_lr_sweep): reward + entropy + combined
    - hits         (mhc_flurry_hits_lr_sweep): reward only (lambda_ = 0)
    """
    os.makedirs(output_dir, exist_ok=True)

    experiments = {
        "mean_entropy": "mhc_flurry_lr_sweep",
        "hits": "mhc_flurry_hits_lr_sweep",
    }

    for exp_name, exp_subdir in experiments.items():
        exp_dir = os.path.join(base_dir, exp_subdir)
        if not os.path.exists(exp_dir):
            print(f"Skipping {exp_name}: {exp_dir} not found")
            continue

        # 1. Collect runs (reward history always; entropy history + lambda if present)
        runs = []
        for run_dir in glob.glob(os.path.join(exp_dir, "t*")):
            dirname = os.path.basename(run_dir)
            parts = dirname.split("_")  # t32_lr0.01_template
            if len(parts) < 3:
                continue
            try:
                lr = float(parts[1].replace("lr", ""))
            except ValueError:
                continue
            reward_unit = parts[2]  # "template" or "batch"

            reward_path = os.path.join(run_dir, "vis", "cache", "reward_history.pkl")
            if not os.path.exists(reward_path):
                print(f"  Warning: No reward cache at {reward_path}")
                continue
            with open(reward_path, "rb") as f:
                reward_history = pickle.load(f)["reward_history"]

            entropy_history = None
            entropy_path = os.path.join(
                run_dir, "vis", "cache", "entropy_objective_history.pkl"
            )
            if os.path.exists(entropy_path):
                with open(entropy_path, "rb") as f:
                    entropy_history = pickle.load(f)["entropy_objective_history"]

            lambda_ = 0.0
            config_path = os.path.join(run_dir, ".hydra", "config.yaml")
            if os.path.exists(config_path):
                lambda_ = float(OmegaConf.load(config_path).training.entropy.lambda_)

            runs.append(
                {
                    "lr": lr,
                    "reward_unit": reward_unit,
                    "reward_history": reward_history,
                    "entropy_history": entropy_history,
                    "lambda_": lambda_,
                }
            )

        if not runs:
            print(f"No runs found for {exp_name}")
            continue

        is_hits_plot = "hits" in exp_subdir.lower()
        filename_base = exp_subdir.replace("mhc_flurry_", "").replace("_", "-")

        # 2. Reward curve (always). Hits reward is a fraction -> show as percentage.
        def reward_series(run: Dict) -> Optional[pd.Series]:
            s = pd.Series(run["reward_history"])
            return s * 100 if is_hits_plot else s

        print(f"[{exp_name}] reward curve")
        _plot_best_credit_curve(
            runs,
            reward_series,
            ylabel="Hits / Batch Size (%)" if is_hits_plot else "Reward",
            title="Reward Curve (MHCI)",
            out_path_base=os.path.join(output_dir, f"{filename_base}-curves"),
        )

        # 3. Entropy + combined full-objective curves (only when an entropy term exists)
        has_entropy = any(
            r["lambda_"] > 0 and r["entropy_history"] is not None for r in runs
        )
        if not has_entropy:
            continue

        def entropy_series(run: Dict) -> Optional[pd.Series]:
            if run["entropy_history"] is None:
                return None
            return pd.Series(run["entropy_history"])

        def combined_series(run: Dict) -> Optional[pd.Series]:
            if run["entropy_history"] is None:
                return None
            reward = pd.Series(run["reward_history"])
            entropy = pd.Series(run["entropy_history"])
            n = min(len(reward), len(entropy))  # guard against length drift
            return reward.iloc[:n] + run["lambda_"] * entropy.iloc[:n]

        print(f"[{exp_name}] entropy curve")
        _plot_best_credit_curve(
            runs,
            entropy_series,
            ylabel="Entropy (nats)",
            title="Entropy Estimate (MHCI)",
            out_path_base=os.path.join(output_dir, f"{filename_base}-entropy-curves"),
            smoothing_window=200,
            robust_ylim=True,
        )

        print(f"[{exp_name}] combined full-objective curve")
        _plot_best_credit_curve(
            runs,
            combined_series,
            ylabel="Mean + $\\lambda\\cdot$Entropy",
            title="Full Objective (MHCI)",
            out_path_base=os.path.join(output_dir, f"{filename_base}-combined-curves"),
            smoothing_window=200,
            robust_ylim=True,
        )


# =============================================================================
# CLI
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate paper experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python aggregate_paper_results.py diversity_fitness --results-dir /path/to/results
  python aggregate_paper_results.py credit_assignment --results-dir /path/to/results
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # diversity_fitness: Pareto curve of theoretical diversity vs mean fitness
    p6 = subparsers.add_parser(
        "diversity_fitness", help="Pareto: Theoretical Diversity vs Mean Fitness"
    )
    p6.add_argument("--results-dir", required=True, help="Results directory")
    p6.add_argument("-o", "--output", default=None, help="Output directory")
    p6.add_argument(
        "--effective-diversity",
        action="store_true",
        help="Use effective diversity exp(H) of the greedy hardened library as the "
        "x-axis instead of the theoretical (multi-length) diversity",
    )
    p6.add_argument(
        "--from-cache",
        action="store_true",
        help="Re-plot from the cached plot data (skip recompute) for fast figure tweaks",
    )

    # credit_assignment: MHC reward curves comparing credit assignment methods
    p9 = subparsers.add_parser(
        "credit_assignment", help="MHC reward curves: credit assignment comparison"
    )
    p9.add_argument("--results-dir", required=True, help="Results directory")
    p9.add_argument("-o", "--output", default=None, help="Output directory")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    print(f"Loading results from {args.results_dir}...")
    results_df = load_experiment_results(args.results_dir)

    if len(results_df) == 0:
        print("No results found. Run experiments first.")
        return

    print(f"Loaded {len(results_df)} experiment results")
    print(f"  Landscapes: {list(results_df['landscape'].unique())}")
    print(f"  Templates: {sorted(results_df['num_templates'].unique())}")
    print(f"  Lambda values: {sorted(results_df['lambda'].unique())}")

    # Execute command
    if args.command == "diversity_fitness":
        output_dir = args.output or os.path.join(
            args.results_dir, "plots", "diversity_fitness"
        )
        plot6_diversity_fitness_pareto(
            results_df,
            output_dir,
            effective_diversity=args.effective_diversity,
            from_cache=args.from_cache,
        )

    elif args.command == "credit_assignment":
        output_dir = args.output or os.path.join(
            args.results_dir, "plots", "credit_assignment"
        )
        plot9_lr_sweep_curves(args.results_dir, output_dir)


if __name__ == "__main__":
    main()
