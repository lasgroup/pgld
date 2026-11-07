"""
Generate paper-ready edit distance histogram from a trained policy.

Loads policy parameters directly without building the landscape (avoids import conflicts).

Usage:
    python plot_edit_distance_paper.py --run-dir /path/to/run
    python plot_edit_distance_paper.py --run-dir /path/to/run --wild-type HMSMQQVVSAGWERADLVGDAFDV
"""

import argparse
import os
import sys

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import PAPER_COLORS, PAPER_MARGINS, CODON_TABLE
from plot_config import PAPER_FIGSIZE, apply_paper_style
from policies import ArbitraryBaseOptimizer, IupacBaseOptimizer


def dna_to_protein(dna_seq: str) -> str:
    """Translate DNA sequence to protein."""
    protein = []
    for i in range(0, len(dna_seq) - 2, 3):
        codon = dna_seq[i : i + 3]
        aa = CODON_TABLE.get(codon, "X")
        if aa == "*":
            break
        protein.append(aa)
    return "".join(protein)


def load_policy(run_dir: str, device: torch.device) -> torch.nn.Module:
    """Load policy from run directory."""
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    param_path = os.path.join(run_dir, "parameters.pth")

    cfg = OmegaConf.load(cfg_path)
    str_length = cfg.landscape.str_length
    num_templates = cfg.policy.num_templates

    if cfg.policy.name == "arbitrary":
        policy = ArbitraryBaseOptimizer(
            str_length=str_length,
            alphabet_size=4,
            batch_size=100000,
            num_templates=num_templates,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "iupac":
        policy = IupacBaseOptimizer(
            str_length=str_length,
            alphabet_size=4,
            batch_size=100000,
            num_templates=num_templates,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    else:
        raise ValueError(f"Unknown policy: {cfg.policy.name}")

    policy.load_state_dict(
        torch.load(param_path, map_location=device, weights_only=True)
    )
    return policy, cfg


def compute_edit_distances(sequences: list[str], wild_type: str) -> list[int]:
    """Compute edit distances from wild type for each sequence."""
    distances = []
    for seq in sequences:
        pep = dna_to_protein(seq)
        if len(pep) < len(wild_type):
            dist = editdistance.eval(pep, wild_type[: len(pep)]) + (
                len(wild_type) - len(pep)
            )
        else:
            dist = editdistance.eval(pep, wild_type)
        distances.append(dist)
    return distances


def plot_histogram(
    distances: list[int],
    save_path: str,
    title: str = "Edit Distance from WT",
) -> None:
    """Plot paper-ready edit distance histogram."""
    apply_paper_style()
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)

    # Use integer bins for discrete edit distances
    max_dist = max(distances)
    bins = np.arange(0, max_dist + 2) - 0.5  # Center bars on integers

    weights = np.ones(len(distances)) / len(distances)
    ax.hist(
        distances,
        bins=bins,
        weights=weights,
        color=PAPER_COLORS["PGLD"],
        alpha=0.7,
    )
    ax.set_xlabel("Edit Distance")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)

    # Set x-axis to show integers only
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    plt.subplots_adjust(**PAPER_MARGINS)

    # Save PDF and PNG
    output_base = os.path.splitext(save_path)[0]
    plt.savefig(f"{output_base}.pdf", format="pdf")
    plt.savefig(f"{output_base}.png", format="png", dpi=150)
    plt.close()
    print(f"Saved: {output_base}.pdf and {output_base}.png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready edit distance histogram"
    )
    parser.add_argument(
        "--run-dir", required=True, help="Path to training run directory"
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path (default: run_dir/vis/edit_distance_paper.pdf)",
    )
    parser.add_argument(
        "--wild-type",
        default="HMSMQQVVSAGWERADLVGDAFDV",
        help="Wild type peptide sequence",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100000, help="Number of sequences to sample"
    )
    parser.add_argument("--title", default="Edit Distance from WT", help="Plot title")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load policy
    print(f"Loading policy from {args.run_dir}...")
    policy, cfg = load_policy(args.run_dir, device)

    # Sample sequences
    print(f"Sampling {args.n_samples} sequences...")
    with torch.no_grad():
        sample_result = policy.sample(args.n_samples, greedy=True)

    alphabet = ["A", "C", "G", "T"]
    sequences = ["".join([alphabet[c] for c in s]) for s in sample_result.sequences]

    # Compute edit distances
    print(f"Computing edit distances to wild type: {args.wild_type}")
    distances = compute_edit_distances(sequences, args.wild_type)

    # Output path
    if args.output is None:
        vis_dir = os.path.join(args.run_dir, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        args.output = os.path.join(vis_dir, "edit_distance_paper.pdf")

    # Plot
    plot_histogram(distances, args.output, args.title)

    # Print stats
    print(f"\nEdit distance statistics:")
    print(f"  Mean: {np.mean(distances):.2f}")
    print(f"  Median: {np.median(distances):.0f}")
    print(f"  Min: {np.min(distances)}, Max: {np.max(distances)}")


if __name__ == "__main__":
    main()
