"""
Generate publication-ready plots for BO paper.

Paper-ready style guidelines:
- Use apply_paper_style() from plot_config for font sizes
- Save plots as PDF
- Use plt.subplots_adjust(**PAPER_MARGINS) for tight margins
- Use PAPER_COLORS for method colors (BO=blue, rational=orange, random=grey)
- Remove all legends (legends added in LaTeX)
- Plots should only have x-axis, y-axis labels and landscape name in title

Usage:
    python scripts/generate_bo_paper_plots.py --input outputs_bo/path/to/run --output my_plot
    python scripts/generate_bo_paper_plots.py --all

Outputs (saved to paper_plots/):
    - {name}_hits.pdf/.png: Cumulative unique hits over iterations
    - {name}_dist.pdf/.png: Fitness distribution histograms
    - {name}_quantile.pdf/.png: Quantile distribution (enumerable landscapes only)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import AMINO_ACIDS, PAPER_COLORS, PAPER_MARGINS
from landscapes import (
    enumerate_sequences,
    enumerate_sequences_and_score,
    sequences_to_valid_peptides,
)
from plot_config import PAPER_FIGSIZE, apply_paper_style
from setup import build_invalid_scores, build_landscape

ENUMERABLE_LANDSCAPES = ["dhfr", "gb1", "tf_binding", "phoq", "trpb"]

PAPER_EXP_BASE = Path(os.environ.get("PGLD_OUTPUT_DIR", "/tmp")) / "bo_paper_exp"
SEEDS = [42, 43, 44, 45, 46, 52, 53, 54, 55, 56]

LANDSCAPE_CONFIG = {
    "dhfr": {
        "title": "DHFR",
        "folder": "dhfr",
    },
    "gb1": {
        "title": "GB1",
        "folder": "gb1",
    },
    "tf_binding": {
        "title": "TF-binding",
        "folder": "tfb",
    },
    "mhc_flurry": {
        "title": "MHC",
        "folder": "mhc",
    },
    "phoq": {
        "title": "PhoQ",
        "folder": "phoq",
    },
    "riboswitch": {
        "title": "Switch",
        "folder": "riboswitch",
    },
    "trpb": {
        "title": "TrpB",
        "folder": "trpb",
    },
}


def get_seed_paths(landscape_key: str) -> List[Path]:
    """Get all seed paths for a landscape."""
    folder = LANDSCAPE_CONFIG[landscape_key]["folder"]
    paths = []
    for seed in SEEDS:
        seed_path = PAPER_EXP_BASE / folder / f"seed_{seed}"
        if seed_path.exists():
            paths.append(seed_path)
    return paths


def load_config(exp_dir: Path) -> Optional[dict]:
    """Load hydra config from experiment directory."""
    config_path = exp_dir / ".hydra" / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_hits_data_single(exp_dir: Path) -> Optional[pd.DataFrame]:
    """Load hits data from a single experiment directory."""
    csv_path = exp_dir / "hits_data.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)

    pkl_path = exp_dir / "cache" / "hits_plot.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        return _convert_pkl_to_df(data)

    return None


def load_hits_data_multi(exp_dirs: List[Path]) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Load hits data from multiple experiment directories (seeds).

    Returns dict with 'mean' and 'std' DataFrames containing cumulative hits stats.
    """
    all_dfs = []
    for exp_dir in exp_dirs:
        df = load_hits_data_single(exp_dir)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        return None

    methods = all_dfs[0]["method"].unique()
    max_iter = max(df["iteration"].max() for df in all_dfs)

    mean_rows = []
    std_rows = []

    for method in methods:
        for iteration in range(int(max_iter) + 1):
            cumulative_hits = []
            for df in all_dfs:
                method_df = df[df["method"] == method].sort_values("iteration")
                if len(method_df) > iteration:
                    cumsum = method_df["new_hits"].cumsum().values
                    cumulative_hits.append(cumsum[iteration])

            if cumulative_hits:
                mean_rows.append(
                    {
                        "method": method,
                        "iteration": iteration,
                        "cumulative_hits": np.mean(cumulative_hits),
                    }
                )
                n_samples = len(cumulative_hits)
                std_rows.append(
                    {
                        "method": method,
                        "iteration": iteration,
                        "cumulative_hits": np.std(cumulative_hits) / np.sqrt(n_samples),
                    }
                )

    return {
        "mean": pd.DataFrame(mean_rows),
        "std": pd.DataFrame(std_rows),
    }


def _convert_pkl_to_df(data: dict) -> pd.DataFrame:
    """Convert the pkl result format to a DataFrame."""
    rows = []
    if data.get("new_hits_history"):
        for i, (new_hits, hits) in enumerate(
            zip(data["new_hits_history"], data["hits_history"])
        ):
            rows.append(
                {"method": "BO", "iteration": i, "new_hits": new_hits, "hits": hits}
            )

    for key in data:
        if key.endswith("_new_hits") and key != "new_hits_history":
            method = key.replace("_new_hits", "")
            hits_key = f"{method}_hits"
            if hits_key in data:
                for i, (new_hits, hits) in enumerate(zip(data[key], data[hits_key])):
                    rows.append(
                        {
                            "method": method,
                            "iteration": i,
                            "new_hits": new_hits,
                            "hits": hits,
                        }
                    )

    return pd.DataFrame(rows)


def load_sampled_sequences(exp_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Load all sampled sequences from the sampled/ directory."""
    sampled_dir = exp_dir / "sampled"
    if not sampled_dir.exists():
        return {}

    method_data: Dict[str, Dict[str, Dict[str, float]]] = {}

    for pkl_file in sampled_dir.glob("*.pkl"):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)

        name = pkl_file.stem
        if name == "init":
            for method in ["BO", "Random", "Rational"]:
                if method not in method_data:
                    method_data[method] = {}
                method_data[method].update(data)
        elif name.startswith("bo_"):
            if "BO" not in method_data:
                method_data["BO"] = {}
            method_data["BO"].update(data)
        elif name.startswith("random_"):
            if "Random" not in method_data:
                method_data["Random"] = {}
            method_data["Random"].update(data)
        elif name.startswith("rational_"):
            if "Rational" not in method_data:
                method_data["Rational"] = {}
            method_data["Rational"].update(data)

    return method_data


def get_fitness_values(
    method_data: Dict[str, Dict[str, Dict[str, float]]], objective: Optional[str] = None
) -> Dict[str, List[float]]:
    """Extract fitness values for each method. Values are log-transformed (from annotate_sequences)."""
    result: Dict[str, List[float]] = {}
    for method, seq_data in method_data.items():
        values = []
        obj_name = objective
        for seq, obj_dict in seq_data.items():
            if obj_name is None:
                obj_name = list(obj_dict.keys())[0]
            if obj_name in obj_dict:
                values.append(obj_dict[obj_name])
        result[method] = values
    return result


def build_landscape_from_config(config: dict) -> object:
    """Build a landscape instance from hydra config dict."""
    cfg = OmegaConf.create(config)
    device = torch.device("cpu")
    objective_names = list(cfg.landscape.objectives.keys())
    invalid_scores = build_invalid_scores(cfg, objective_names)
    return build_landscape(cfg, objective_names, device, invalid_scores)


def _build_seq_to_quantile_map(
    config: dict, landscape_name: str
) -> Optional[Dict[str, float]]:
    """Build sequence -> quantile mapping by enumerating the landscape."""
    landscape = build_landscape_from_config(config)
    objective = list(config["landscape"]["objectives"].keys())[0]

    print(
        f"  Enumerating all DNA sequences for {landscape_name} (objective: {objective})..."
    )
    all_scores_df = enumerate_sequences_and_score(landscape, objective=objective)

    if landscape.is_peptide:
        dna_sequences = all_scores_df["sequence"].tolist()
        valid_idx, valid_peptides = sequences_to_valid_peptides(
            dna_sequences, landscape.min_peptide_len, landscape.max_peptide_len
        )
        scores = all_scores_df["score"].values[valid_idx]
        peptide_scores_df = pd.DataFrame({"peptide": valid_peptides, "score": scores})
        peptide_scores_df = peptide_scores_df.drop_duplicates(subset="peptide")
        sorted_df = peptide_scores_df.sort_values("score").reset_index(drop=True)
        seq_column = "peptide"
        print(f"  Converted to {len(sorted_df)} unique peptides")
    else:
        sorted_df = all_scores_df.sort_values("score").reset_index(drop=True)
        seq_column = "sequence"

    n = len(sorted_df)
    return {seq: (i + 1) / n * 100 for i, seq in enumerate(sorted_df[seq_column])}


def compute_quantiles_for_seed(
    exp_dir: Path,
    seq_to_quantile: Dict[str, float],
) -> Optional[Dict[str, List[float]]]:
    """Compute quantiles for a single seed using precomputed seq_to_quantile map."""
    cache_path = exp_dir / "quantiles.pkl"
    if cache_path.exists():
        print(f"  Loading cached quantiles from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    method_data = load_sampled_sequences(exp_dir)
    if not method_data:
        return None

    quantiles: Dict[str, List[float]] = {}
    for method, seq_data in method_data.items():
        sequences = list(seq_data.keys())
        method_quantiles = [seq_to_quantile.get(seq, 0) for seq in sequences]
        quantiles[method] = method_quantiles

    with open(cache_path, "wb") as f:
        pickle.dump(quantiles, f)
    print(f"  Saved quantiles to {cache_path}")

    return quantiles


def compute_quantiles_for_landscape(
    exp_dirs: List[Path],
) -> Optional[Dict[str, List[float]]]:
    """
    Compute quantiles for sampled sequences across multiple seeds.

    Enumerates landscape once, then computes and caches quantiles per seed.
    Returns combined quantiles from all seeds (for averaging in histogram).
    """
    if not exp_dirs:
        return None

    config = load_config(exp_dirs[0])
    if config is None:
        return None

    landscape_name = config.get("landscape", {}).get("name", "")
    if landscape_name not in ENUMERABLE_LANDSCAPES:
        return None

    # Check if all seeds have cached quantiles
    all_cached = all((exp_dir / "quantiles.pkl").exists() for exp_dir in exp_dirs)

    if all_cached:
        print(f"  Loading cached quantiles for all {len(exp_dirs)} seeds...")
        seq_to_quantile = None
    else:
        seq_to_quantile = _build_seq_to_quantile_map(config, landscape_name)
        if seq_to_quantile is None:
            return None

    # Collect quantiles from all seeds
    combined_quantiles: Dict[str, List[float]] = {}
    for exp_dir in exp_dirs:
        if seq_to_quantile is None:
            # Load from cache
            cache_path = exp_dir / "quantiles.pkl"
            with open(cache_path, "rb") as f:
                seed_quantiles = pickle.load(f)
        else:
            seed_quantiles = compute_quantiles_for_seed(exp_dir, seq_to_quantile)

        if seed_quantiles is None:
            continue

        for method, quantile_list in seed_quantiles.items():
            if method not in combined_quantiles:
                combined_quantiles[method] = []
            combined_quantiles[method].extend(quantile_list)

    return combined_quantiles if combined_quantiles else None


def plot_quantile_distribution(
    quantiles: Dict[str, List[float]],
    landscape_name: str,
    output_path: Path,
    n_seeds: int = 1,
) -> None:
    """Plot quantile distribution histograms averaged over seeds."""
    apply_paper_style()

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    plot_order = ["Random", "Rational", "BO"]
    bins = np.linspace(0, 100, 51)

    # Draw filled histograms with transparency (normalized by n_seeds)
    for method in plot_order:
        if method not in quantiles:
            continue
        values = quantiles[method]
        color = PAPER_COLORS.get(method, "#000000")
        counts, _ = np.histogram(values, bins=bins)
        counts = counts / n_seeds
        ax.fill_between(bins[:-1] + 1, 0, counts, step="post", alpha=0.4, color=color)

    # Draw outlines on top with full opacity
    for method in plot_order:
        if method not in quantiles:
            continue
        values = quantiles[method]
        color = PAPER_COLORS.get(method, "#000000")
        counts, _ = np.histogram(values, bins=bins)
        counts = counts / n_seeds
        ax.step(bins[:-1] + 1, counts, where="post", color=color, linewidth=1.2)

    ax.set_xlabel("Quantile (%)")
    ax.set_ylabel("Count")
    ax.set_title(landscape_name)
    ax.set_xlim(0, 100)

    if ax.get_legend():
        ax.get_legend().remove()

    plt.subplots_adjust(**PAPER_MARGINS)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=300)
    plt.close()


def compute_topk_quantile(
    quantiles: Dict[str, List[float]], k: int = 5000, landscape_name: str = ""
) -> Dict[str, float]:
    """Compute the quantile of the k-th best sequence for each method."""
    result = {}
    for method, values in quantiles.items():
        sorted_values = sorted(values, reverse=True)
        if len(sorted_values) >= k:
            result[method] = sorted_values[k - 1]
        elif sorted_values:
            result[method] = sorted_values[-1]
        else:
            result[method] = 0.0

    if landscape_name:
        print(f"  Quantile of {k}-th best sequence for {landscape_name}:")
        for method, q in result.items():
            n_seqs = len(quantiles[method])
            print(f"    {method}: {q:.2f}% (n={n_seqs})")

    return result


def plot_topk_quantile_bar(
    all_landscape_data: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Plot grouped bar chart of top-k average quantiles across all enumerable landscapes."""
    apply_paper_style()

    fig, ax = plt.subplots(figsize=(PAPER_FIGSIZE[0] * 2, PAPER_FIGSIZE[1]))

    methods = ["Random", "Rational", "BO"]
    landscapes = list(all_landscape_data.keys())
    n_landscapes = len(landscapes)
    n_methods = len(methods)

    bar_width = 0.25
    group_spacing = 0.3
    group_width = n_methods * bar_width + group_spacing

    for i, method in enumerate(methods):
        positions = [j * group_width + i * bar_width for j in range(n_landscapes)]
        values = [all_landscape_data[ls].get(method, 0) for ls in landscapes]
        color = PAPER_COLORS.get(method, "#000000")
        ax.bar(positions, values, bar_width, color=color)

    # Set x-axis ticks at center of each group
    group_centers = [
        j * group_width + (n_methods - 1) * bar_width / 2 for j in range(n_landscapes)
    ]
    landscape_titles = [LANDSCAPE_CONFIG[ls]["title"] for ls in landscapes]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(landscape_titles)

    ax.set_ylabel("Quantile of 5000th Best")
    ax.set_ylim(0, 100)

    if ax.get_legend():
        ax.get_legend().remove()

    plt.subplots_adjust(**PAPER_MARGINS)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=300)
    plt.close()


def plot_fitness_distribution(
    exp_dir: Path,
    landscape_name: str,
    output_path: Path,
    objective: Optional[str] = None,
) -> None:
    """Plot fitness distribution histograms for BO vs Random vs Rational."""
    apply_paper_style()

    method_data = load_sampled_sequences(exp_dir)
    if not method_data:
        raise ValueError(f"No sampled data found in {exp_dir / 'sampled'}")

    fitness_values = get_fitness_values(method_data, objective)

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    plot_order = ["Random", "Rational", "BO"]
    for method in plot_order:
        if method not in fitness_values:
            continue
        values = fitness_values[method]
        color = PAPER_COLORS.get(method, "#000000")
        ax.hist(values, bins=50, alpha=0.6, color=color, edgecolor="none")

    ax.set_xlabel("Fitness")
    ax.set_ylabel("Count")
    ax.set_title(landscape_name)

    if ax.get_legend():
        ax.get_legend().remove()

    plt.subplots_adjust(**PAPER_MARGINS)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=300)
    plt.close()


def plot_cumulative_unique_hits(
    hits_data: Dict[str, pd.DataFrame],
    landscape_name: str,
    output_path: Path,
) -> None:
    """Plot cumulative unique hits over iterations with mean and shaded std."""
    apply_paper_style()

    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    mean_df = hits_data["mean"]
    std_df = hits_data["std"]

    plot_order = ["Random", "Rational", "BO"]
    for method in plot_order:
        if method not in mean_df["method"].unique():
            continue
        method_mean = mean_df[mean_df["method"] == method].sort_values("iteration")
        method_std = std_df[std_df["method"] == method].sort_values("iteration")
        color = PAPER_COLORS.get(method, "#000000")

        iterations = method_mean["iteration"].values
        mean_vals = method_mean["cumulative_hits"].values
        std_vals = method_std["cumulative_hits"].values

        ax.plot(iterations, mean_vals, color=color, linewidth=2.5)
        ax.fill_between(
            iterations,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.2,
        )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Unique Hits")
    ax.set_title(landscape_name)
    ax.ticklabel_format(axis="y", style="scientific", scilimits=(0, 4))

    max_iter = mean_df["iteration"].max()
    ax.set_xticks(range(0, int(max_iter) + 1))

    if ax.get_legend():
        ax.get_legend().remove()

    plt.subplots_adjust(**PAPER_MARGINS)
    plt.savefig(output_path.with_suffix(".pdf"), format="pdf")
    plt.savefig(output_path.with_suffix(".png"), format="png", dpi=300)
    plt.close()


def create_plots(
    exp_dirs: List[Path], output_name: str, landscape_title: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create and save paper-ready plots from multiple seed directories.

    Returns dict with 'quantiles' and 'hits_data' keys.
    """
    if not exp_dirs:
        raise ValueError("No experiment directories provided")

    config = load_config(exp_dirs[0])
    landscape_key = (
        config.get("landscape", {}).get("name", "Unknown") if config else "Unknown"
    )
    if landscape_title is None:
        landscape_title = LANDSCAPE_CONFIG.get(landscape_key, {}).get(
            "title", landscape_key
        )

    output_dir = Path("paper_plots")
    output_dir.mkdir(exist_ok=True)
    n_seeds = len(exp_dirs)

    # Hits plot with mean/std across seeds
    hits_path = output_dir / f"{output_name}_hits"
    hits_data = load_hits_data_multi(exp_dirs)
    if hits_data is None:
        raise ValueError(f"No hits data found in {exp_dirs}")
    plot_cumulative_unique_hits(hits_data, landscape_title, hits_path)
    print(f"Saved {hits_path.with_suffix('.pdf')} and {hits_path.with_suffix('.png')}")

    # Fitness distribution (uses first seed)
    dist_path = output_dir / f"{output_name}_dist"
    plot_fitness_distribution(exp_dirs[0], landscape_title, dist_path)
    print(f"Saved {dist_path.with_suffix('.pdf')} and {dist_path.with_suffix('.png')}")

    # Quantile plot averaged over seeds
    quantiles = None
    if landscape_key in ENUMERABLE_LANDSCAPES:
        quantiles = compute_quantiles_for_landscape(exp_dirs)
        if quantiles:
            quantile_path = output_dir / f"{output_name}_quantile"
            plot_quantile_distribution(
                quantiles, landscape_title, quantile_path, n_seeds=n_seeds
            )
            print(
                f"Saved {quantile_path.with_suffix('.pdf')} and {quantile_path.with_suffix('.png')}"
            )

    return {"quantiles": quantiles, "hits_data": hits_data}


def count_sequences_above_quantile(
    quantiles: Dict[str, List[float]], threshold: float
) -> Dict[str, int]:
    """Count sequences above a given quantile threshold for each method."""
    return {
        method: sum(1 for q in vals if q >= threshold)
        for method, vals in quantiles.items()
    }


def print_hits_table(
    all_hits_data: Dict[str, Dict[str, pd.DataFrame]],
) -> None:
    """Print table of final cumulative hits, BO/Rational ratio, and percentage increase."""
    print("\n" + "=" * 80)
    print("TABLE 1: Cumulative Unique Hits at Final Iteration")
    print("=" * 80)

    header = f"{'Landscape':<12} {'BO':>10} {'Rational':>10} {'Random':>10} {'BO/Rat':>10} {'% Incr':>10}"
    print(header)
    print("-" * 80)

    for landscape_key, hits_data in all_hits_data.items():
        title = LANDSCAPE_CONFIG[landscape_key]["title"]
        mean_df = hits_data["mean"]
        max_iter = mean_df["iteration"].max()

        final_hits = {}
        for method in ["BO", "Rational", "Random"]:
            method_df = mean_df[mean_df["method"] == method]
            if len(method_df) > 0:
                final_val = method_df[method_df["iteration"] == max_iter][
                    "cumulative_hits"
                ].values
                final_hits[method] = final_val[0] if len(final_val) > 0 else 0
            else:
                final_hits[method] = 0

        ratio = (
            final_hits["BO"] / final_hits["Rational"]
            if final_hits["Rational"] > 0
            else float("inf")
        )
        pct_increase = (
            ((final_hits["BO"] - final_hits["Rational"]) / final_hits["Rational"] * 100)
            if final_hits["Rational"] > 0
            else float("inf")
        )
        print(
            f"{title:<12} {final_hits['BO']:>10.1f} {final_hits['Rational']:>10.1f} {final_hits['Random']:>10.1f} {ratio:>10.2f} {pct_increase:>9.1f}%"
        )

    print("=" * 80)


QUANTILE_THRESHOLDS = [99.9, 99, 95, 90, 85]


def compute_quantile_counts(
    all_quantiles: Dict[str, Dict[str, List[float]]],
    n_seeds: int,
) -> Dict[str, Dict[str, Dict[float, float]]]:
    """Compute counts above each quantile threshold for all landscapes and methods."""
    methods = ["BO", "Rational", "Random"]
    result: Dict[str, Dict[str, Dict[float, float]]] = {}

    for landscape_key, quantiles in all_quantiles.items():
        result[landscape_key] = {}
        for method in methods:
            if method not in quantiles:
                continue
            result[landscape_key][method] = {}
            for t in QUANTILE_THRESHOLDS:
                count = count_sequences_above_quantile(quantiles, t).get(method, 0)
                result[landscape_key][method][t] = count / n_seeds

    return result


def print_quantile_table(
    all_quantiles: Dict[str, Dict[str, List[float]]],
    n_seeds: int,
) -> None:
    """Print table of sequences above quantile thresholds for enumerable landscapes."""
    methods = ["BO", "Rational", "Random"]
    counts = compute_quantile_counts(all_quantiles, n_seeds)

    print("\n" + "=" * 100)
    print("TABLE 2: Sequences Above Quantile Thresholds (averaged over seeds)")
    print("=" * 100)

    header = f"{'Landscape':<12} {'Method':<10}"
    for t in QUANTILE_THRESHOLDS:
        header += f" {'>=' + str(t) + '%':>10}"
    print(header)
    print("-" * 100)

    for landscape_key, method_counts in counts.items():
        title = LANDSCAPE_CONFIG[landscape_key]["title"]

        for i, method in enumerate(methods):
            if method not in method_counts:
                continue
            prefix = title if i == 0 else ""
            row = f"{prefix:<12} {method:<10}"
            for t in QUANTILE_THRESHOLDS:
                row += f" {method_counts[method][t]:>10.1f}"
            print(row)

        # Print percent increase row (BO vs Rational)
        if "BO" in method_counts and "Rational" in method_counts:
            row = f"{'':<12} {'% Incr':<10}"
            for t in QUANTILE_THRESHOLDS:
                bo_count = method_counts["BO"][t]
                rat_count = method_counts["Rational"][t]
                pct = (
                    ((bo_count - rat_count) / rat_count * 100)
                    if rat_count > 0
                    else float("inf")
                )
                row += f" {pct:>9.1f}%"
            print(row)
        print("-" * 100)

    print("=" * 100)


def save_quantile_table_latex(
    all_quantiles: Dict[str, Dict[str, List[float]]],
    n_seeds: int,
    output_path: Path,
) -> None:
    """Save quantile table as LaTeX code."""
    methods = ["BO", "Rational", "Random"]
    counts = compute_quantile_counts(all_quantiles, n_seeds)
    n_cols = len(QUANTILE_THRESHOLDS)

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Sequences above quantile thresholds (averaged over seeds)}",
        r"\label{tab:quantiles}",
        r"\begin{tabular}{ll" + "r" * n_cols + "}",
        r"\toprule",
    ]

    # Header row
    header = r"Landscape & Method"
    for t in QUANTILE_THRESHOLDS:
        header += f" & $\\geq {t}\\%$"
    header += r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for landscape_key, method_counts in counts.items():
        title = LANDSCAPE_CONFIG[landscape_key]["title"]

        for i, method in enumerate(methods):
            if method not in method_counts:
                continue
            prefix = title if i == 0 else ""
            row = f"{prefix} & {method}"
            for t in QUANTILE_THRESHOLDS:
                row += f" & {method_counts[method][t]:.1f}"
            row += r" \\"
            lines.append(row)

        # Percent increase row
        if "BO" in method_counts and "Rational" in method_counts:
            row = r" & \% Incr"
            for t in QUANTILE_THRESHOLDS:
                bo_count = method_counts["BO"][t]
                rat_count = method_counts["Rational"][t]
                pct = (
                    ((bo_count - rat_count) / rat_count * 100)
                    if rat_count > 0
                    else float("inf")
                )
                row += f" & {pct:.1f}\\%"
            row += r" \\"
            lines.append(row)

        lines.append(r"\midrule")

    # Remove last midrule and add bottomrule
    lines[-1] = r"\bottomrule"

    lines.extend(
        [
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved LaTeX table to {output_path}")


def run_all_landscapes() -> None:
    """Generate plots for all landscapes defined in LANDSCAPE_CONFIG."""
    all_topk_data: Dict[str, Dict[str, float]] = {}
    all_hits_data: Dict[str, Dict[str, pd.DataFrame]] = {}
    all_quantiles: Dict[str, Dict[str, List[float]]] = {}
    n_seeds = 0

    for landscape_key, config in LANDSCAPE_CONFIG.items():
        print(f"\n{'='*50}")
        print(f"Processing {landscape_key}...")
        print(f"{'='*50}")
        exp_dirs = get_seed_paths(landscape_key)
        if not exp_dirs:
            print(f"WARNING: No seed directories found for {landscape_key}")
            continue
        print(f"Found {len(exp_dirs)} seeds: {[p.name for p in exp_dirs]}")
        n_seeds = len(exp_dirs)
        result = create_plots(exp_dirs, landscape_key, config["title"])

        all_hits_data[landscape_key] = result["hits_data"]
        quantiles = result["quantiles"]

        # Collect top-k quantile data for enumerable landscapes
        if quantiles:
            all_topk_data[landscape_key] = compute_topk_quantile(
                quantiles, landscape_name=config["title"]
            )
            all_quantiles[landscape_key] = quantiles

    # Generate combined bar plot for enumerable landscapes
    if all_topk_data:
        output_dir = Path("paper_plots")
        output_dir.mkdir(exist_ok=True)
        bar_path = output_dir / "all_top5000_quantile"
        plot_topk_quantile_bar(all_topk_data, bar_path)
        print(
            f"\nSaved {bar_path.with_suffix('.pdf')} and {bar_path.with_suffix('.png')}"
        )

    # Print summary tables
    if all_hits_data:
        print_hits_table(all_hits_data)
    if all_quantiles:
        print_quantile_table(all_quantiles, n_seeds)
        output_dir = Path("paper_plots")
        output_dir.mkdir(exist_ok=True)
        save_quantile_table_latex(
            all_quantiles, n_seeds, output_dir / "quantile_table.tex"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready plots for BO paper"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate plots for all landscapes using paths defined in LANDSCAPE_CONFIG",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to experiment output folder or parent folder with seed_* subdirs",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename prefix (required unless --all is used)",
    )

    args = parser.parse_args()

    if args.all:
        run_all_landscapes()
    else:
        if not args.input or not args.output:
            parser.error("--input and --output are required unless --all is used")
        input_path = Path(args.input)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_path}")

        # Check if input has seed subdirectories
        seed_dirs = sorted(input_path.glob("seed_*"))
        if seed_dirs:
            exp_dirs = seed_dirs
            print(f"Found {len(exp_dirs)} seeds: {[p.name for p in exp_dirs]}")
        else:
            exp_dirs = [input_path]

        create_plots(exp_dirs, args.output)


if __name__ == "__main__":
    main()
