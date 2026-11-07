"""
visualizations.py

Contains functions for plotting the results of the optimization experiments.
"""

import logging
import os
import pickle
import random
from collections import Counter
from typing import Dict, List, Optional

import editdistance
import logomaker
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import torch
import torch.nn.functional as F
from numba import njit

import plot_config

plot_config.apply_default_style()

log = logging.getLogger(__name__)

from constants import (
    DNA_BASES,
    CODON_TABLE,
    AMINO_ACIDS,
    IUPAC_BOTTLES,
    DEFAULT_DIVERSITY_THRESHOLD,
    MAX_AA_INDEX,
    PAPER_COLORS,
    PAPER_MARGINS,
)
from landscapes import (
    TFBindingLandscape,
    PLMLogLikelihoodLandscape,
    dna_to_protein,
    enumerate_sequences_and_score,
    generate_nnk_sequences,
    generate_uniform_random_sequences,
    is_landscape_enumerable,
)

from policies import (
    ArbitraryBaseOptimizer,
    BottleBasedOptimizer,
    FiniteBaseOptimizer,
    SampleResult,
    SlotConstrainedIupacOptimizer,
)
from tensor_translation_gpu import (
    build_codon_lookup_gpu,
    translate_dna_to_aa_indices_gpu,
    dna_probs_to_aa_probs,
)

from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, r2_score
from plm_utils import load_fasta
from mmd import encode_sequences


def plot_reward_history(
    reward_history,
    policy_name,
    save_path="reward_history.png",
    reward_name="Number of Unique Hits",
):
    """Plots and saves the reward history over training iterations."""
    plt.figure(figsize=(10, 6))
    pd.Series(reward_history).plot(label="Reward", color="blue", alpha=0.7)
    pd.Series(reward_history).rolling(window=50).mean().plot(
        label="Smoothed (window=50)", color="red"
    )
    plt.title(f"Reward History for {policy_name} Policy")
    plt.xlabel("Iteration")
    plt.ylabel(reward_name)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved reward history plot to {save_path}")

    # Save cache
    _save_cache(
        {
            "reward_history": reward_history,
            "policy_name": policy_name,
        },
        save_path,
    )


def plot_entropy_objective_history(
    entropy_objective_history: list[float],
    policy_name: str,
    save_path: str = "entropy_objective_history.png",
):
    """Plots and saves the mean entropy objective over training iterations."""
    plt.figure(figsize=(10, 6))
    pd.Series(entropy_objective_history).plot(
        label="Entropy Objective", color="green", alpha=0.7
    )
    pd.Series(entropy_objective_history).rolling(window=50).mean().plot(
        label="Smoothed (window=50)", color="darkgreen"
    )
    plt.title(f"Mean Entropy Objective for {policy_name} Policy")
    plt.xlabel("Iteration")
    plt.ylabel("Mean Entropy Objective")
    min_val = min(entropy_objective_history) if entropy_objective_history else 1
    if min_val <= 0:
        plt.yscale("symlog", linthresh=1e-8)
    else:
        plt.yscale("log")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved entropy objective history plot to {save_path}")

    _save_cache(
        {
            "entropy_objective_history": entropy_objective_history,
            "policy_name": policy_name,
        },
        save_path,
    )


def plot_score_histogram(
    scores,
    save_path,
    title,
    xlabel="Fitness Score",
    ylabel="Frequency",
    threshold=None,
    label=None,
):
    """
    General histogram plotting function for sequence scores.
    Optionally marks a threshold with a vertical line.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color="skyblue", alpha=0.8, label=label)
    if threshold is not None:
        plt.axvline(
            threshold,
            color="red",
            linestyle="dashed",
            linewidth=2,
            label=f"Hit Threshold: {threshold:.2f}",
        )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle="--", alpha=0.6)
    if label or threshold is not None:
        plt.legend()
    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved score histogram to {save_path}")

    # Save cache
    _save_cache(
        {
            "scores": list(scores),
            "threshold": threshold,
        },
        save_path,
    )


def plot_likelihood_ecdf_comparison(
    policy_scores: np.ndarray,
    var_synth_scores: np.ndarray,
    save_path: str,
    title: str = "Log-Likelihood CDF",
    wild_type_score: Optional[float] = None,
    plm_sample_scores: Optional[np.ndarray] = None,
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
    xlim: tuple[float, float] = None,
) -> None:
    """Plot ECDF comparison of PGLD, var_synth, and optionally PLM sample distributions.

    Args:
        xlim: Optional (xmin, xmax) to set x-axis limits. Useful when including
              invalid sequences with very low scores to keep plot readable.
    """
    from constants import PAPER_COLORS, PAPER_MARGINS

    fig, ax = plt.subplots(figsize=figsize or (10, 6))

    # Sort scores for ECDF
    policy_sorted = np.sort(policy_scores)
    var_synth_sorted = np.sort(var_synth_scores)

    # Compute ECDF y-values
    policy_ecdf = np.arange(1, len(policy_sorted) + 1) / len(policy_sorted)
    var_synth_ecdf = np.arange(1, len(var_synth_sorted) + 1) / len(var_synth_sorted)

    # Use consistent colors, hide labels in paper mode
    pgld_label = None if paper_ready else "PGLD"
    var_label = None if paper_ready else "Var-Synth"
    plm_label = None if paper_ready else "PLM Samples"

    ax.step(
        policy_sorted,
        policy_ecdf,
        where="post",
        label=pgld_label,
        linewidth=2,
        color=PAPER_COLORS["PGLD"],
    )
    ax.step(
        var_synth_sorted,
        var_synth_ecdf,
        where="post",
        label=var_label,
        linewidth=2,
        color=PAPER_COLORS["Var-Synth"],
    )

    if plm_sample_scores is not None:
        plm_sorted = np.sort(plm_sample_scores)
        plm_ecdf = np.arange(1, len(plm_sorted) + 1) / len(plm_sorted)
        ax.step(
            plm_sorted,
            plm_ecdf,
            where="post",
            label=plm_label,
            linewidth=2,
            color=PAPER_COLORS["PLM Samples"],
        )

    if wild_type_score is not None:
        ax.axvline(
            wild_type_score,
            color="red",
            linestyle="--",
            label=f"Wild Type: {wild_type_score:.2f}",
        )

    ax.set_xlabel("Log-Likelihood under PLM")
    ax.set_ylabel("Cumulative Prob.")
    ax.set_title(title)
    # Only show legend if there are labeled items (wild type in paper mode)
    if not paper_ready or wild_type_score is not None:
        ax.legend()
    ax.grid(True, alpha=0.3)

    if xlim is not None:
        ax.set_xlim(xlim)

    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_landscape_histogram(
    landscape: TFBindingLandscape,
    threshold: float,
    save_path="landscape_histogram.png",
    objective=None,
):
    """
    Enumerates all sequences in a landscape, plots their fitness distribution,
    and marks the hit threshold.

    Args:
        landscape: Landscape to enumerate
        threshold: Fitness threshold for hits
        save_path: Path to save histogram
        objective: Which objective to score (for multi-objective landscapes)
    """
    log.info(
        "Enumerating and scoring all sequences in landscape (this may take a moment)..."
    )
    sequence_scores = enumerate_sequences_and_score(landscape, objective=objective)

    if objective is None:
        obj_label = (
            landscape.active_objs[0] if len(landscape.active_objs) == 1 else "Combined"
        )
    else:
        obj_label = objective

    plot_score_histogram(
        sequence_scores["score"],
        save_path=save_path,
        title=f"Full Fitness Score Distribution for {obj_label}",
        threshold=threshold,
        label="All Sequence Scores",
    )


def compute_marginal_peptide_log_probs(
    policy: torch.nn.Module,
    sample_result: SampleResult,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute marginal peptide log-likelihoods: log p(seq) = log Σ_t w_t * p(seq|t)

    Marginalizes over ALL templates using policy's mixture weights.
    Only includes valid sequences (no stop codons) for fixed-length comparison.

    Args:
        policy: Trained policy (ArbitraryBaseOptimizer or BottleBasedOptimizer).
        sample_result: Output of policy.sample().

    Returns:
        log_probs: (num_valid,) tensor of marginal peptide log-likelihoods.
        valid_msk: (batch_size,) boolean tensor indicating valid sequences.
    """
    device = sample_result.sequences.device

    # 1. Translate DNA to peptide indices
    codon_lookup = build_codon_lookup_gpu(device)
    aa_seqs, stop_idx = translate_dna_to_aa_indices_gpu(
        sample_result.sequences, codon_lookup
    )

    # Filter to valid sequences (no stop codons)
    valid_msk = (stop_idx == -1).cpu()
    valid_seqs = aa_seqs[valid_msk].clamp(max=MAX_AA_INDEX).long()
    n_valid = valid_seqs.shape[0]

    if n_valid == 0:
        return torch.tensor([]), valid_msk

    # 2. Get DNA probs for ALL templates (greedy bottles for bottle-based)
    template_dna_probs = torch.tensor(
        policy.get_template_dna_probs(), device=device, dtype=torch.float32
    )  # (num_templates, str_length, 4)

    # 3. Convert to AA log probs
    aa_log_probs = torch.log(
        torch.stack([dna_probs_to_aa_probs(tp) for tp in template_dna_probs])
    ).clamp(
        min=-1e8
    )  # (num_templates, num_codons, 20)

    num_templates = aa_log_probs.shape[0]

    # 4. Gather log probs at sequence positions for all templates
    # aa_log_probs: (T, L, 20), valid_seqs: (N, L)
    aa_log_probs_exp = aa_log_probs.unsqueeze(1).expand(-1, n_valid, -1, -1)
    seqs_exp = valid_seqs.unsqueeze(0).expand(num_templates, -1, -1).unsqueeze(-1)
    log_probs_per_pos = aa_log_probs_exp.gather(3, seqs_exp).squeeze(-1)
    # (num_templates, n_valid, num_codons)

    log_probs_per_template = log_probs_per_pos.sum(dim=-1)  # (num_templates, n_valid)

    # 5. Get mixture weights from policy and marginalize
    w_mix = torch.softmax(policy.latent_w_mix, dim=0).to(device)  # (num_templates,)
    log_w_mix = torch.log(w_mix).unsqueeze(1)  # (num_templates, 1)
    log_joint = log_probs_per_template + log_w_mix
    log_marginal = torch.logsumexp(log_joint, dim=0)  # (n_valid,)

    return log_marginal.cpu(), valid_msk


def compute_marginal_aa_log_probs_for_sequences(
    policy, aa_sequences: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute marginal AA log probability of arbitrary peptide sequences under the policy.

    Adapts pattern from compute_marginal_peptide_log_probs() but takes AA indices
    directly instead of requiring a SampleResult from policy sampling.
    It can be used to score 'off-policy' by computing likelihoods of externally
    sampled sequences.

    Args:
        policy: Trained policy with get_template_dna_probs() and latent_w_mix.
        aa_sequences: (N, L) int tensor, values 0-19 for amino acids.
        device: Device for computation.

    Returns:
        (N,) tensor of log probabilities. -inf for sequences impossible under all templates.
    """
    aa_sequences = aa_sequences.to(device)
    N, L = aa_sequences.shape

    # 1. Get DNA probs and convert to AA probs for all templates
    template_dna_probs = torch.tensor(
        policy.get_template_dna_probs(), device=device, dtype=torch.float32
    )  # (T, L_dna, 4)

    aa_probs = torch.stack(
        [dna_probs_to_aa_probs(tp) for tp in template_dna_probs]
    )  # (T, L, 20)

    num_templates = aa_probs.shape[0]

    # 2. Gather probs at sequence positions for each template
    aa_probs_exp = aa_probs.unsqueeze(1).expand(-1, N, -1, -1)  # (T, N, L, 20)
    seqs_exp = (
        aa_sequences.unsqueeze(0).expand(num_templates, -1, -1).unsqueeze(-1)
    )  # (T, N, L, 1)
    probs_per_pos = aa_probs_exp.gather(3, seqs_exp).squeeze(-1)  # (T, N, L)

    # 3. Compute log probs per template (use -inf for 0 prob, no clamping needed)
    # For each template, check if ALL positions have non-zero prob
    template_has_zero = (probs_per_pos < 1e-30).any(
        dim=2
    )  # (T, N) - template has any 0-prob position
    log_probs_per_pos = torch.log(
        probs_per_pos.clamp(min=1e-30)
    )  # (T, N, L) - clamp to avoid -inf
    log_probs_per_template = log_probs_per_pos.sum(dim=-1)  # (T, N)
    log_probs_per_template[template_has_zero] = float(
        "-inf"
    )  # Templates with any 0-prob position get -inf

    # 4. Get mixture weights and marginalize via logsumexp
    w_mix = torch.softmax(policy.latent_w_mix, dim=0).to(device)  # (T,)
    log_w_mix = torch.log(w_mix).unsqueeze(1)  # (T, 1)
    log_joint = log_probs_per_template + log_w_mix  # (T, N)
    log_marginal = torch.logsumexp(log_joint, dim=0)  # (N,)

    return log_marginal.cpu()


def plot_policy_likelihood_histogram(
    log_probs: np.ndarray,
    save_path: str,
    title: str = "Policy Log-Likelihood Distribution",
):
    """
    Plot histogram of policy log-likelihoods, handling -inf values.

    Args:
        log_probs: (N,) array of log probabilities, may contain -inf.
        save_path: Path to save the plot.
        title: Plot title.
    """
    finite_mask = np.isfinite(log_probs)
    finite_probs = log_probs[finite_mask]
    n_impossible = (~finite_mask).sum()

    fig, ax = plt.subplots(figsize=(10, 6))

    if len(finite_probs) > 0:
        ax.hist(finite_probs, bins=50, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Log-Likelihood under Policy")
        ax.set_ylabel("Count")
        ax.set_title(title)

        # Add stats
        stats_text = (
            f"Mean: {finite_probs.mean():.2f}\nMedian: {np.median(finite_probs):.2f}"
        )
        if n_impossible > 0:
            stats_text += f"\nImpossible: {n_impossible}"
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
    else:
        ax.text(
            0.5,
            0.5,
            f"All {n_impossible} sequences impossible under policy",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title(title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved policy likelihood histogram to {save_path}")


def compute_nnk_log_probs_for_sequences(
    aa_sequences: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute log probability of AA sequences under a single-template NNK policy.

    NNK codon: N (any base) at pos 1&2, K (G or T) at pos 3.
    Encodes all 20 AAs with specific probabilities based on codon redundancy.

    Args:
        aa_sequences: (N, L) int tensor, values 0-19 for amino acids.
        device: Device for computation.

    Returns:
        (N,) tensor of log probabilities. -inf if sequence contains stop codon.
    """
    aa_sequences = aa_sequences.to(device)
    N, L = aa_sequences.shape

    # Build NNK DNA probs for one codon: (3, 4) where alphabet is [A, C, G, T]
    # Position 1&2: N = uniform [0.25, 0.25, 0.25, 0.25]
    # Position 3: K = G or T = [0, 0, 0.5, 0.5]
    nnk_codon_probs = torch.tensor(
        [
            [0.25, 0.25, 0.25, 0.25],  # N
            [0.25, 0.25, 0.25, 0.25],  # N
            [0.0, 0.0, 0.5, 0.5],  # K (G=0.5, T=0.5)
        ],
        device=device,
        dtype=torch.float32,
    )

    # Tile for L codons (peptide length L)
    dna_probs = (
        nnk_codon_probs.unsqueeze(0).expand(L, -1, -1).reshape(L * 3, 4)
    )  # (L*3, 4)

    # Convert to AA probs using dna_probs_to_aa_probs
    aa_probs = dna_probs_to_aa_probs(dna_probs)  # (L, 20)

    # Compute log probs - handle 0 probabilities (stop codon under NNK)
    aa_log_probs = torch.log(aa_probs + 1e-10)  # (L, 20)

    # Gather log probs at sequence positions using proper indexing
    # aa_log_probs: (L, 20), aa_sequences: (N, L)
    aa_log_probs_exp = aa_log_probs.unsqueeze(0).expand(N, L, 20)  # (N, L, 20)
    seqs_exp = aa_sequences.unsqueeze(-1)  # (N, L, 1)
    log_probs_per_pos = aa_log_probs_exp.gather(2, seqs_exp).squeeze(-1)  # (N, L)
    log_probs = log_probs_per_pos.sum(dim=-1)  # (N,)

    # Check for any position with 0 prob
    aa_probs_exp = aa_probs.unsqueeze(0).expand(N, L, 20)  # (N, L, 20)
    probs_at_positions = aa_probs_exp.gather(2, seqs_exp).squeeze(-1)  # (N, L)
    impossible_mask = (probs_at_positions < 1e-10).any(dim=1)
    log_probs[impossible_mask] = float("-inf")

    return log_probs.cpu()


def plot_sampled_scores_histogram(
    policy: torch.nn.Module,
    landscape: TFBindingLandscape,
    num_sequences: int,
    save_path="sampled_scores_histogram.png",
    **kwargs,
):
    """
    Samples sequences from a trained policy, plots their fitness distribution.
    """
    log.info(f"Sampling {num_sequences} sequences from trained policy...")

    sample_result = policy.sample(num_sequences, greedy=True)
    sequences_str = [
        "".join([landscape.alphabet[c] for c in s]) for s in sample_result.sequences
    ]

    scores = landscape.get_fitness_values(sequences_str, **kwargs)
    plot_score_histogram(
        scores,
        save_path=save_path,
        title=f"Fitness Score Distribution of {num_sequences} Sequences Sampled from {policy.name} Policy",
    )

    if isinstance(landscape, PLMLogLikelihoodLandscape):
        log_probs_peptide, valid_msk = compute_marginal_peptide_log_probs(
            policy, sample_result
        )
        valid_scores = scores[valid_msk]

        if valid_msk.sum() == 0:
            log.warning(
                "No valid sequences (all contain stop codons). Skipping correlation plots."
            )
            return

        save_dir = os.path.dirname(save_path)

        save_path = os.path.join(save_dir, "correlation_peptide.png")
        plot_correlation(
            log_probs_peptide,
            valid_scores,
            title="Policy Marginal Peptide logLH vs. Target PLM logLH",
            save_path=save_path,
        )

        log_probs_dna_per_seq = policy.get_log_probs(sample_result).detach().cpu()

        save_path = os.path.join(save_dir, "correlation_dna.png")

        plot_correlation(
            log_probs_dna_per_seq[valid_msk],
            valid_scores,
            title="Policy DNA vs. Target PLM logLH",
            save_path=save_path,
        )


def plot_finite_optimizer_results(
    policy: FiniteBaseOptimizer, alphabet: list, save_path="finite_policy_results.png"
):
    """Creates and saves a summary plot for the FiniteBaseOptimizer using heatmaps."""
    v_probs, phi_probs, w_mix_probs = policy.get_probs()
    v_probs_np = v_probs.detach().cpu().numpy()
    phi_probs_np = phi_probs.detach().cpu().numpy()
    w_mix_probs_np = w_mix_probs.detach().cpu().numpy()

    num_templates = policy.num_templates
    if num_templates > 5:  # magic number here just so we don't make too many plots
        template_indices = np.random.choice(num_templates, size=5, replace=False)
    else:
        template_indices = range(num_templates)

    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    extension = os.path.splitext(save_path)[1]

    for template_idx in template_indices:
        # Scale figure height based on sequence length and number of bottles
        height = max(6, min(20, policy.str_length * 0.15 + 4))
        fig, axes = plt.subplots(
            1, 3, figsize=(22, height), gridspec_kw={"width_ratios": [1.2, 1, 2.2]}
        )
        fig.suptitle("FiniteBaseOptimizer Final Parameters")

        sns.heatmap(
            v_probs_np,
            annot=policy.num_bottles <= 10,  # Only show numbers if not too crowded
            fmt=".2f",
            cmap="viridis",
            ax=axes[0],
            xticklabels=alphabet,
            yticklabels=[f"Bottle {i+1}" for i in range(policy.num_bottles)],
        )
        axes[0].set_title("Bottle Compositions (v)")
        axes[0].set_xlabel("Base")
        axes[0].set_ylabel("Bottle")

        axes[1].pie(
            w_mix_probs_np,
            labels=[f"Template {i}" for i in range(policy.num_templates)],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.4),
        )
        axes[1].set_title("Template Mixture Weights (w)")
        axes[1].axis("equal")

        axes[2].set_title(f"Bottle Selection Probs (phi) for Template {template_idx}")

        # Improve readability for long sequences
        show_annotations = policy.str_length <= 30 and policy.num_bottles <= 10
        y_tick_step = max(1, policy.str_length // 20)  # Show at most 20 y-ticks
        y_positions = range(0, policy.str_length, y_tick_step)
        y_labels = [str(i + 1) for i in y_positions]

        sns.heatmap(
            phi_probs_np[template_idx],
            annot=show_annotations,
            fmt=".2f",
            cmap="plasma",
            ax=axes[2],
            xticklabels=[f"B{i+1}" for i in range(policy.num_bottles)],
            yticklabels=False,  # We'll set custom y-ticks below
        )

        # Set custom y-ticks to prevent overcrowding
        axes[2].set_yticks(y_positions)
        axes[2].set_yticklabels(y_labels)

        # Rotate x-labels if there are many bottles
        if policy.num_bottles > 8:
            axes[2].tick_params(axis="x", rotation=45)
        axes[2].set_xlabel("Bottle")
        axes[2].set_ylabel("Position in Sequence")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        template_save_path = os.path.join(
            base_dir, f"{base_name}_{template_idx}{extension}"
        )
        plt.savefig(template_save_path)
        plt.close()
        log.info(f"Saved finite policy results plot to {template_save_path}")

        _save_cache(
            {
                "policy_name": policy.name,
                "alphabet": alphabet,
                "v_probs": v_probs_np.tolist(),
                "phi_probs": phi_probs_np.tolist(),
                "w_mix_probs": w_mix_probs_np.tolist(),
                "num_bottles": policy.num_bottles,
                "num_templates": policy.num_templates,
                "str_length": policy.str_length,
            },
            template_save_path,
        )


def plot_arbitrary_optimizer_results(
    policy: ArbitraryBaseOptimizer,
    alphabet: list,
    save_path="arbitrary_policy_results.png",
):
    """Creates and saves a summary plot for the ArbitraryBaseOptimizer using heatmaps."""
    template_probs, mixture_probs = policy.get_probs()
    template_probs_np = template_probs.detach().cpu().numpy()
    mixture_probs_np = mixture_probs.detach().cpu().numpy()

    num_templates = policy.num_templates
    if num_templates > 5:
        template_indices = np.random.choice(num_templates, size=5, replace=False)
    else:
        template_indices = range(num_templates)

    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    extension = os.path.splitext(save_path)[1]

    for template_idx in template_indices:
        # Scale figure height based on sequence length
        height = max(7, min(20, policy.str_length * 0.15 + 5))
        fig, axes = plt.subplots(
            1, 2, figsize=(16, height), gridspec_kw={"width_ratios": [1, 1.5]}
        )
        fig.suptitle("ArbitraryBaseOptimizer Final Parameters")

        axes[0].pie(
            mixture_probs_np,
            labels=[f"Template {i}" for i in range(policy.num_templates)],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.4),
        )
        axes[0].set_title("Template Mixture Weights")
        axes[0].axis("equal")

        axes[1].set_title(f"Base Probabilities for Template {template_idx}")

        # Improve readability for long sequences
        show_annotations = policy.str_length <= 30 and len(alphabet) <= 8
        y_tick_step = max(1, policy.str_length // 20)  # Show at most 20 y-ticks
        y_positions = range(0, policy.str_length, y_tick_step)
        y_labels = [str(i + 1) for i in y_positions]

        sns.heatmap(
            template_probs_np[template_idx],
            annot=show_annotations,
            fmt=".2f",
            cmap="viridis",
            ax=axes[1],
            xticklabels=alphabet,
            yticklabels=False,  # We'll set custom y-ticks below
        )

        # Set custom y-ticks to prevent overcrowding
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(y_labels)

        # Rotate x-labels if there are many bases
        if len(alphabet) > 8:
            axes[1].tick_params(axis="x", rotation=45)
        axes[1].set_xlabel("Base")
        axes[1].set_ylabel("Position in Sequence")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        template_save_path = os.path.join(
            base_dir, f"{base_name}_{template_idx}{extension}"
        )
        plt.savefig(template_save_path)
        plt.close()
        log.info(f"Saved arbitrary policy results plot to {template_save_path}")

        _save_cache(
            {
                "policy_name": policy.name,
                "alphabet": alphabet,
                "template_probs": template_probs_np.tolist(),
                "mixture_probs": mixture_probs_np.tolist(),
                "num_templates": policy.num_templates,
                "str_length": policy.str_length,
            },
            template_save_path,
        )


def plot_objective_weights(
    weights, obj_labels, save_path="reward_objective_weights.png", use_entropy=False
):

    if use_entropy:
        obj_labels = obj_labels + ["Entropy"]

    weights = np.abs(weights)

    fig, ax = plt.subplots()
    bottom = np.zeros(len(weights))
    for i in range(weights.shape[1]):
        ax.bar(
            range(len(weights)),
            weights[:, i].flatten(),
            bottom=bottom,
            width=0.5,
            label=obj_labels[i],
        )
        bottom = bottom + weights[:, i].flatten()

    ax.set_ylabel("Objective weights")
    ax.set_xlabel("Policy")
    ax.set_title("Reward objective weights per policy")
    ax.legend()

    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved reward weights plot to {save_path}")

    # Save cache
    _save_cache(
        {
            "weights": weights.tolist(),
            "objectives": obj_labels,
        },
        save_path,
    )


def plot_pareto_front(
    policies: list[torch.nn.Module],
    landscape,
    num_sequences=100,
    save_path="pareto_front.png",
):
    if len(landscape.active_objs) != 2:
        raise ValueError(f"Can only plot the Pareto front for 2 objectives")
    fig, ax = plt.subplots()
    all_seqs = []
    for i, policy in enumerate(policies):
        log.info(f"Sampling {num_sequences} sequences from policy {i}...")
        result = policy.sample(num_sequences, greedy=True)
        sequences_str = [
            "".join([landscape.alphabet[c] for c in s]) for s in result.sequences
        ]
        all_seqs.append(sequences_str)

        scores = landscape.get_all_objectives(sequences_str)
        ax.scatter(scores[:, 0], scores[:, 1], label=i, s=5)

    ax.legend(title="Policy")
    ax.set_xlabel(landscape.active_objs[0])
    ax.set_ylabel(landscape.active_objs[1])
    ax.set_title("Fitness tradeoff for policy sampled sequences")

    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved reward weights plot to {save_path}")

    _save_cache({"sequences": all_seqs, "objectives": landscape.active_objs}, save_path)


def _save_cache(data: dict, plot_path: str):
    """Save plotting data to a cache file in a 'cache' subdirectory next to the plot."""
    cache_dir = os.path.join(os.path.dirname(plot_path), "cache")
    os.makedirs(cache_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(plot_path))[0]
    cache_path = os.path.join(cache_dir, f"{base}.pkl")
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    log.info(f"Saved plot cache to {cache_path}")


def _load_cache(plot_path: str) -> Optional[dict]:
    """Load plotting data saved by _save_cache, or None if the cache is absent."""
    cache_dir = os.path.join(os.path.dirname(plot_path), "cache")
    base = os.path.splitext(os.path.basename(plot_path))[0]
    cache_path = os.path.join(cache_dir, f"{base}.pkl")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def plot_training_history_oracle(history, save_path="training_history.png"):
    """
    Plot training and validation loss over time.

    Args:
        history: Dictionary containing 'epochs', 'train_loss', and 'val_loss'
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history["epochs"], history["train_loss"], label="Train Loss", linewidth=2)
    plt.plot(history["epochs"], history["val_loss"], label="Val Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE on log10 scale)")
    plt.title("Training and Validation Loss Over Time")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_peptide_length_histogram(
    sequences, save_path, title="Peptide Length Distribution"
):
    """
    Plot histogram of peptide lengths for DNA sequences.

    Args:
        sequences: List of DNA sequence strings
        save_path: Path to save the plot
        title: Title for the plot
    """
    peptides = [dna_to_protein(seq) for seq in sequences]
    lengths = [len(p) for p in peptides]

    plt.figure(figsize=(10, 6))
    plt.hist(lengths, bins=50, color="blue", alpha=0.7, edgecolor="black")
    plt.xlabel("Peptide Length (AA)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    log.info(f"Saved peptide length histogram to {save_path}")
    plt.close()


def plot_edit_distance_histogram(
    sequences: list[str],
    wild_type: str,
    save_path: str,
    title: str = "Edit Distance",
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """
    Plot histogram of edit distances from wild-type for DNA sequences.

    Args:
        sequences: List of DNA sequence strings
        wild_type: Wild-type peptide sequence string
        save_path: Path to save the plot
        title: Title for the plot
        figsize: Optional figure size (width, height) in inches
        paper_ready: If True, use paper-ready styling
    """
    from constants import PAPER_COLORS, PAPER_MARGINS

    peptides = [dna_to_protein(seq) for seq in sequences]

    distances = []
    for pep in peptides:
        if len(pep) < len(wild_type):
            # Count missing positions as edits
            dist = editdistance.eval(pep, wild_type[: len(pep)]) + (
                len(wild_type) - len(pep)
            )
        else:
            dist = editdistance.eval(pep, wild_type)
        distances.append(dist)

    plt.figure(figsize=figsize or (10, 6))
    weights = np.ones(len(distances)) / len(distances)
    plt.hist(
        distances,
        bins=50,
        weights=weights,
        color=PAPER_COLORS["PGLD"],
        alpha=0.7,
        edgecolor="black" if not paper_ready else None,
    )
    plt.xlabel("Edit Distance")
    plt.ylabel("Proportion")
    plt.title(title)
    plt.grid(True, alpha=0.3, axis="y")
    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    log.info(f"Saved edit distance histogram to {save_path}")
    plt.close()


def plot_edit_distance_overlay(
    policy_sequences: list[str],
    var_synth_sequences: list[str],
    wild_type: str,
    save_path: str,
    reference_sequences: list[str] = None,
    title: str = "Edit Distance from WT",
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """
    Plot side-by-side bar chart of edit distances for policy, var-synth, and reference sequences.

    Args:
        policy_sequences: List of DNA sequence strings (will be converted to peptides)
        var_synth_sequences: List of peptide sequence strings (already peptides)
        wild_type: Wild-type peptide sequence string
        save_path: Path to save the plot
        reference_sequences: Optional list of reference peptide sequences (already peptides)
        title: Title for the plot
        figsize: Optional figure size (width, height) in inches
        paper_ready: If True, use paper-ready styling (no legend)
    """
    MAX_DIST = 10  # Bin everything above this into "10+"

    def calc_edit_distances(peptides: list[str], wt: str) -> list[int]:
        distances = []
        for pep in peptides:
            if len(pep) < len(wt):
                dist = editdistance.eval(pep, wt[: len(pep)]) + (len(wt) - len(pep))
            else:
                dist = editdistance.eval(pep, wt)
            distances.append(min(dist, MAX_DIST))  # Cap at MAX_DIST
        return distances

    # 1. Convert policy DNA sequences to peptides and calculate distances
    policy_peptides = [dna_to_protein(seq) for seq in policy_sequences]
    policy_distances = calc_edit_distances(policy_peptides, wild_type)

    # 2. Calculate edit distances for var-synth sequences (already peptides)
    var_synth_distances = calc_edit_distances(var_synth_sequences, wild_type)

    # 3. Calculate edit distances for reference sequences if provided
    ref_distances = None
    if reference_sequences:
        ref_distances = calc_edit_distances(reference_sequences, wild_type)

    # 4. Compute histogram counts for side-by-side bars (0 to MAX_DIST)
    min_dist = 0
    bin_edges = np.arange(min_dist, MAX_DIST + 2)
    positions = np.arange(min_dist, MAX_DIST + 1)
    tick_labels = [str(i) for i in range(MAX_DIST)] + [f"{MAX_DIST}+"]

    policy_counts, _ = np.histogram(policy_distances, bins=bin_edges)
    var_synth_counts, _ = np.histogram(var_synth_distances, bins=bin_edges)
    policy_props = policy_counts / len(policy_distances)
    var_synth_props = var_synth_counts / len(var_synth_distances)

    if ref_distances:
        ref_counts, _ = np.histogram(ref_distances, bins=bin_edges)
        ref_props = ref_counts / len(ref_distances)

    # 5. Create side-by-side bar chart
    fig, ax = plt.subplots(figsize=figsize or (10, 6))
    n_groups = 3 if ref_distances else 2
    bar_width = 0.8 / n_groups

    if ref_distances:
        ax.bar(
            positions - bar_width,
            ref_props,
            bar_width,
            color=PAPER_COLORS["Reference"],
            label="Reference",
            edgecolor="black" if not paper_ready else None,
        )
        ax.bar(
            positions,
            policy_props,
            bar_width,
            color=PAPER_COLORS["PGLD"],
            label="PGLD",
            edgecolor="black" if not paper_ready else None,
        )
        ax.bar(
            positions + bar_width,
            var_synth_props,
            bar_width,
            color=PAPER_COLORS["Var-Synth"],
            label="Var-Synth",
            edgecolor="black" if not paper_ready else None,
        )
    else:
        ax.bar(
            positions - bar_width / 2,
            policy_props,
            bar_width,
            color=PAPER_COLORS["PGLD"],
            label="PGLD",
            edgecolor="black" if not paper_ready else None,
        )
        ax.bar(
            positions + bar_width / 2,
            var_synth_props,
            bar_width,
            color=PAPER_COLORS["Var-Synth"],
            label="Var-Synth",
            edgecolor="black" if not paper_ready else None,
        )

    ax.set_xlabel("Edit Distance")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.3, axis="y")

    if not paper_ready:
        ax.legend()

    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()

    plt.savefig(save_path, dpi=300)
    log.info(f"Saved edit distance overlay to {save_path}")
    plt.close()


def dna_to_amino_acid_probabilities(dna_probs_per_position):
    """
    Convert per-position DNA base probabilities to amino acid probabilities.

    Numpy-based implementation that includes the stop codon.
    For a torch-based version one can backprop through, see policies.dna_to_amino_acid_probabilities.

    Args:
        dna_probs_per_position: Array of shape (seq_length, 4) with DNA base probabilities (A, C, G, T)

    Returns:
        amino_acid_probs: Array of shape (seq_length // 3, 21) with amino acid probabilities
    """
    seq_length = dna_probs_per_position.shape[0]
    if seq_length % 3 != 0:
        raise ValueError(f"DNA sequence length {seq_length} is not divisible by 3")

    num_codons = seq_length // 3
    amino_acid_probs = np.zeros((num_codons, len(AMINO_ACIDS)))

    # Map amino acids to indices
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

    for codon_pos in range(num_codons):
        # Get probabilities for the three positions in this codon
        pos1_probs = dna_probs_per_position[codon_pos * 3]  # A, C, G, T
        pos2_probs = dna_probs_per_position[codon_pos * 3 + 1]  # A, C, G, T
        pos3_probs = dna_probs_per_position[codon_pos * 3 + 2]  # A, C, G, T

        # Calculate probability for each possible codon
        for i, base1 in enumerate(DNA_BASES):
            for j, base2 in enumerate(DNA_BASES):
                for k, base3 in enumerate(DNA_BASES):
                    codon = base1 + base2 + base3
                    codon_prob = pos1_probs[i] * pos2_probs[j] * pos3_probs[k]

                    if codon in CODON_TABLE:
                        amino_acid = CODON_TABLE[codon]
                        aa_idx = aa_to_idx[amino_acid]
                        amino_acid_probs[codon_pos, aa_idx] += codon_prob

    return amino_acid_probs


def compute_theoretical_diversity_from_aa_probs(
    aa_probs: np.ndarray, threshold: float = DEFAULT_DIVERSITY_THRESHOLD
) -> float:
    """
    Compute theoretical diversity from amino acid probabilities.

    Args:
        aa_probs: Array of shape (num_codons, num_amino_acids) with AA probabilities per codon position
        threshold: Probability threshold for considering an AA as possible

    Returns:
        Theoretical diversity (number of unique protein sequences possible) as float
    """
    diversity = 1.0
    for codon_pos in range(aa_probs.shape[0]):
        num_possible_aas = int(np.sum(aa_probs[codon_pos] > threshold))
        diversity *= num_possible_aas
    return diversity


def plot_amino_acid_policy_results(
    policy, alphabet, save_path="amino_acid_policy_results.png"
):
    """
    Creates amino acid probability visualizations for DNA-based policies.
    Shows per-template amino acid probabilities like the existing policy visualizations.

    Args:
        policy: Trained DNA policy (ArbitraryBaseOptimizer or FiniteBaseOptimizer)
        alphabet: DNA alphabet ['A', 'C', 'G', 'T']
        save_path: Path to save the plot
    """
    if len(alphabet) != 4 or set(alphabet) != {"A", "C", "G", "T"}:
        raise ValueError("This function only works with DNA alphabets")

    if policy.str_length % 3 != 0:
        raise ValueError(
            f"DNA sequence length {policy.str_length} is not divisible by 3"
        )

    # Get DNA probabilities from policy
    if not hasattr(policy, "get_probs"):
        raise ValueError("Policy must have get_probs() method")

    num_codons = policy.str_length // 3

    if isinstance(policy, BottleBasedOptimizer):
        # BottleBasedOptimizer (FiniteBaseOptimizer, IupacBaseOptimizer, etc.)
        _, _, w_mix_probs = policy.get_probs()
        mixture_probs_np = w_mix_probs.detach().cpu().numpy()

        template_dna_probs_all = policy.get_template_dna_probs()
        template_aa_probs = np.zeros(
            (policy.num_templates, num_codons, len(AMINO_ACIDS))
        )

        for t in range(policy.num_templates):
            template_aa_probs[t] = dna_to_amino_acid_probabilities(
                template_dna_probs_all[t]
            )

    else:
        # ArbitraryBaseOptimizer and FixedIupacPolicy return (template_probs, mixture_probs)
        template_probs, mixture_probs = policy.get_probs()
        template_probs_np = template_probs.detach().cpu().numpy()
        mixture_probs_np = mixture_probs.detach().cpu().numpy()

        template_aa_probs = np.zeros(
            (policy.num_templates, num_codons, len(AMINO_ACIDS))
        )

        for t in range(policy.num_templates):
            template_aa_probs[t] = dna_to_amino_acid_probabilities(template_probs_np[t])

    # Create per-template visualizations (like existing policy plots)
    num_templates = policy.num_templates
    if num_templates > 5:  # Same logic as existing policy plots
        template_indices = np.random.choice(num_templates, size=5, replace=False)
    else:
        template_indices = range(num_templates)

    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    extension = os.path.splitext(save_path)[1]

    for template_idx in template_indices:
        # Scale figure height based on sequence length (like existing plots)
        height = max(7, min(20, num_codons * 0.4 + 5))
        fig, axes = plt.subplots(
            1, 2, figsize=(20, height), gridspec_kw={"width_ratios": [1, 1.5]}
        )
        fig.suptitle(f"{policy.name} Policy: Amino Acid Probabilities")

        # Left plot: Template mixture weights (same as existing plots)
        axes[0].pie(
            mixture_probs_np,
            labels=[f"Template {i}" for i in range(policy.num_templates)],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops=dict(width=0.4),
        )
        axes[0].set_title("Template Mixture Weights")
        axes[0].axis("equal")

        # Right plot: Amino acid probabilities for this template
        axes[1].set_title(f"Amino Acid Probabilities for Template {template_idx}")

        # Improve readability for long sequences
        show_aa_annotations = num_codons <= 20 and len(AMINO_ACIDS) <= 15
        y_tick_step = max(1, num_codons // 15)  # Show at most 15 y-ticks
        y_positions = range(0, num_codons, y_tick_step)
        y_labels = [str(i + 1) for i in y_positions]

        sns.heatmap(
            template_aa_probs[template_idx],
            annot=show_aa_annotations,
            fmt=".2f",
            cmap="plasma",
            ax=axes[1],
            xticklabels=AMINO_ACIDS,
            yticklabels=False,  # We'll set custom y-ticks below
        )

        # Set custom y-ticks to prevent overcrowding
        axes[1].set_yticks(y_positions)
        axes[1].set_yticklabels(y_labels)

        # Rotate x-labels for amino acids
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set_xlabel("Amino Acid")
        axes[1].set_ylabel("Codon Position")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        template_save_path = os.path.join(
            base_dir, f"{base_name}_{template_idx}{extension}"
        )
        plt.savefig(template_save_path, dpi=300, bbox_inches="tight")
        plt.close()
        log.info(f"Saved amino acid policy results plot to {template_save_path}")

        # Save cache
        _save_cache(
            {
                "policy_name": policy.name,
                "alphabet": alphabet,
                "template_aa_probs": template_aa_probs.tolist(),
                "mixture_probs": mixture_probs_np.tolist(),
                "num_templates": policy.num_templates,
                "str_length": policy.str_length,
                "num_codons": num_codons,
            },
            template_save_path,
        )


def plot_linear_plm_target(model, save_path: str = "linear_plm_target.png"):
    """
    Visualize the Linear PLM's per-position amino acid probability distribution.

    Args:
        model: LinearLogLH model instance
        save_path: Path to save the heatmap
    """
    with torch.no_grad():
        probs = F.softmax(model.logits, dim=-1).cpu().numpy()

    num_positions = probs.shape[0]
    alphabet = model.alphabet

    height = max(7, min(20, num_positions * 0.4 + 5))
    fig, ax = plt.subplots(figsize=(12, height))

    show_annotations = num_positions <= 20 and len(alphabet) <= 15
    y_tick_step = max(1, num_positions // 15)
    y_positions = list(range(0, num_positions, y_tick_step))
    y_labels = [str(i + 1) for i in y_positions]

    sns.heatmap(
        probs,
        annot=show_annotations,
        fmt=".2f",
        cmap="plasma",
        ax=ax,
        xticklabels=alphabet,
        yticklabels=False,
    )

    ax.set_yticks([p + 0.5 for p in y_positions])
    ax.set_yticklabels(y_labels)
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Amino Acid")
    ax.set_ylabel("Position")
    ax.set_title("Linear PLM Target: Per-Position Amino Acid Probabilities")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved Linear PLM target visualization to {save_path}")

    _save_cache(
        {
            "model_name": model.name,
            "alphabet": alphabet,
            "probabilities": probs.tolist(),
            "num_positions": num_positions,
        },
        save_path,
    )


def plot_sequence_diversity(
    sequences, save_path, title="Sequence Diversity", sequence_type="DNA", verbose=True
):
    """
    Plot multiple visualizations of sequence diversity and uniqueness patterns.

    Args:
        sequences: List of sequence strings (DNA or peptide)
        save_path: Path to save the plot
        title: Title for the plot
        sequence_type: "DNA" or "peptide" for labeling
        verbose: If True, print diversity stats to console
    """
    # Count sequence frequencies
    seq_counts = Counter(sequences)
    unique_sequences = list(seq_counts.keys())
    counts = list(seq_counts.values())

    # Sort by count (descending)
    sorted_counts = sorted(counts, reverse=True)

    # Calculate diversity metrics
    total_sequences = len(sequences)
    num_unique = len(unique_sequences)
    diversity_ratio = num_unique / total_sequences
    max_count = max(counts)
    singletons = sum(1 for c in counts if c == 1)

    # Create 2x2 subplot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"{title} - {total_sequences:,} {sequence_type} sequences")

    # Plot 1: Histogram of sequence counts
    ax1.hist(
        counts, bins=min(50, max(counts)), color="skyblue", alpha=0.7, edgecolor="black"
    )
    ax1.set_xlabel(f"Times {sequence_type} sequence appears")
    ax1.set_ylabel("Number of unique sequences")
    ax1.set_title(f"Distribution of {sequence_type} Sequence Frequencies")
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Plot 2: Rank-frequency plot (Zipf-like)
    ranks = range(1, len(sorted_counts) + 1)
    ax2.loglog(ranks, sorted_counts, "o-", markersize=3, alpha=0.7, color="red")
    ax2.set_xlabel(f"Rank (by frequency)")
    ax2.set_ylabel(f"Frequency")
    ax2.set_title(f"{sequence_type} Rank-Frequency Plot")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Cumulative uniqueness curve
    cumulative_unique = []
    seen = set()
    for seq in sequences:
        seen.add(seq)
        cumulative_unique.append(len(seen))

    sample_indices = range(
        0, len(cumulative_unique), max(1, len(cumulative_unique) // 1000)
    )
    sampled_x = [i for i in sample_indices]
    sampled_y = [cumulative_unique[i] for i in sample_indices]

    ax3.plot(sampled_x, sampled_y, linewidth=2, color="green")
    ax3.set_xlabel(f"Number of {sequence_type} sequences sampled")
    ax3.set_ylabel("Number of unique sequences seen")
    ax3.set_title(f"Cumulative {sequence_type} Sequence Discovery")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Top sequence frequencies (bar chart)
    top_n = min(20, len(unique_sequences))
    top_seqs = [f"Seq_{i+1}" for i in range(top_n)]  # Anonymous labels
    top_counts = sorted_counts[:top_n]

    bars = ax4.bar(range(top_n), top_counts, color="orange", alpha=0.7)
    ax4.set_xlabel(f"Top {top_n} most frequent {sequence_type} sequences")
    ax4.set_ylabel("Frequency")
    ax4.set_title(f"Top {sequence_type} Sequence Frequencies")
    ax4.tick_params(axis="x", rotation=45)
    ax4.set_xticks(range(top_n))
    ax4.set_xticklabels(top_seqs)
    ax4.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    if verbose:
        log.info(f"Saved sequence diversity plot to {save_path}")

        # Print diversity stats
        print(f"  {sequence_type} Diversity Stats:")
        print(f"    Total sequences: {total_sequences:,}")
        print(f"    Unique sequences: {num_unique:,}")
        print(f"    Diversity ratio: {diversity_ratio:.3f}")
        print(f"    Most frequent sequence appears: {max_count:,} times")
        print(f"    Singleton sequences: {singletons:,} ({singletons/num_unique:.1%})")

    # Save cache
    _save_cache(
        {
            "sequences": sequences,
            "sequence_counts": dict(seq_counts),
            "total_sequences": total_sequences,
            "num_unique": num_unique,
            "diversity_ratio": diversity_ratio,
            "max_count": max_count,
            "singletons": singletons,
        },
        save_path,
    )


def plot_dna_and_peptide_diversity(
    dna_sequences, save_path, title="DNA and Peptide Diversity", verbose=True
):
    """
    Plot diversity for both DNA sequences and their corresponding peptides.

    Args:
        dna_sequences: List of DNA sequence strings
        save_path: Path to save the plot
        title: Title for the plot
        verbose: If True, print diversity stats to console
    """
    # Convert DNA to peptides
    peptide_sequences = []
    for dna_seq in dna_sequences:
        try:
            peptide = dna_to_protein(dna_seq)
            if peptide and peptide != "":  # Skip empty/invalid peptides
                peptide_sequences.append(peptide)
        except:
            continue  # Skip problematic sequences

    # Create individual diversity plots for DNA and peptides
    base_dir = os.path.dirname(save_path)
    base_name = os.path.splitext(os.path.basename(save_path))[0]
    extension = os.path.splitext(save_path)[1]

    # DNA diversity plot
    dna_save_path = os.path.join(base_dir, f"{base_name}_dna{extension}")
    plot_sequence_diversity(
        dna_sequences, dna_save_path, f"{title} - DNA", "DNA", verbose=verbose
    )

    # Peptide diversity plot
    if peptide_sequences:
        peptide_save_path = os.path.join(base_dir, f"{base_name}_peptide{extension}")
        plot_sequence_diversity(
            peptide_sequences,
            peptide_save_path,
            f"{title} - Peptide",
            "peptide",
            verbose=verbose,
        )
    elif verbose:
        print("  Warning: No valid peptide sequences found for diversity analysis")


def print_oligo_ordering_template(policy, log_to_file=None, verbose=True):
    """
    Print policy templates in oligo ordering format.

    Args:
        policy: Policy with to_text() method
        log_to_file: Optional file path to save output
        verbose: If True, log output to console
    """
    if not hasattr(policy, "to_text"):
        return

    try:
        template_text = policy.to_text()
    except (NotImplementedError, AttributeError):
        return

    output = f"\n{'='*80}\nOLIGO ORDERING TEMPLATE\n{'='*80}\n\n{template_text}\n\n{'='*80}\n"
    if verbose:
        log.info(output)

    if log_to_file:
        with open(log_to_file, "w") as f:
            f.write(output)


def print_theoretical_diversity(
    policy,
    alphabet,
    is_peptide_landscape=False,
    log_to_file=None,
    threshold=DEFAULT_DIVERSITY_THRESHOLD,
    verbose=True,
) -> Optional[Dict[int, float]]:
    """
    Print theoretical diversity for each template in the policy.

    Args:
        policy: Policy with get_template_dna_probs() method
        alphabet: DNA alphabet
        is_peptide_landscape: If True, compute amino acid diversity; if False, compute DNA diversity
        log_to_file: Optional file path to save output
        threshold: Probability threshold for considering an AA/base as possible
        verbose: If True, log output to console

    Returns:
        Dictionary mapping template index to diversity value, or None if not applicable.
    """
    if not hasattr(policy, "get_template_dna_probs"):
        return None
    if len(alphabet) != 4 or set(alphabet) != {"A", "C", "G", "T"}:
        return None

    template_dna_probs = policy.get_template_dna_probs()
    num_templates = template_dna_probs.shape[0]

    diversity_dict = {}
    for t in range(num_templates):
        if is_peptide_landscape:
            aa_probs = dna_to_amino_acid_probabilities(template_dna_probs[t])
            diversity_dict[t] = compute_theoretical_diversity_from_aa_probs(
                aa_probs, threshold
            )
        else:
            diversity = 1.0
            for pos in range(template_dna_probs.shape[1]):
                num_possible_bases = int(np.sum(template_dna_probs[t, pos] > threshold))
                diversity *= num_possible_bases
            diversity_dict[t] = diversity

    sequence_type = "Protein Sequences" if is_peptide_landscape else "DNA Sequences"
    lines = [
        f"\n{'='*80}",
        f"THEORETICAL DIVERSITY (Unique {sequence_type})",
        "=" * 80,
        "",
    ]
    for template_idx, diversity in diversity_dict.items():
        if diversity >= 1e6:
            lines.append(f"Template {template_idx}: {diversity:.2e} unique sequences")
        else:
            lines.append(
                f"Template {template_idx}: {int(diversity):,} unique sequences"
            )
    lines.append(f"\n{'='*80}\n")

    output = "\n".join(lines)
    if verbose:
        log.info(output)

    if log_to_file:
        with open(log_to_file, "w") as f:
            f.write(output)

    return diversity_dict


def plot_theoretical_diversity_histogram(
    diversity_dict: dict[int, float],
    save_path: str,
    title: str = "Template Diversity",
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """
    Plot histogram of theoretical diversity values across templates.

    Args:
        diversity_dict: Template index -> diversity value mapping
        save_path: Path to save the plot
        title: Plot title
        figsize: Figure size (width, height) in inches
        paper_ready: If True, use paper-ready styling
    """
    values = list(diversity_dict.values())
    if not values:
        log.warning("No diversity values to plot")
        return

    fig, ax = plt.subplots(figsize=figsize or (10, 6))

    # Log-spaced bins (one per order of magnitude)
    log_min = np.floor(np.log10(min(values)))
    log_max = np.ceil(np.log10(max(values)))
    bins = np.logspace(log_min, log_max, int(log_max - log_min) + 1)

    ax.hist(
        values,
        bins=bins,
        color=PAPER_COLORS["PGLD"],
        alpha=0.7,
        edgecolor="black" if not paper_ready else None,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Unique sequences")
    ax.set_ylabel("# Templates")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=5))

    if paper_ready:
        fig.subplots_adjust(**PAPER_MARGINS)
    else:
        fig.tight_layout()

    plt.savefig(save_path, dpi=300)
    log.info(f"Saved theoretical diversity histogram to {save_path}")
    plt.close()


def plot_correlation(preds, labs, title="Correlation", save_path="rsquared.png"):
    """
    Plot correlation between PLM log-likelihood predictions and reference values.

    Fits a linear regression to show the correlation strength. Used for evaluating
    PLM landscape predictions against ground truth (e.g., experimental binding data).
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()
    if isinstance(labs, torch.Tensor):
        labs = labs.numpy()

    fig, ax = plt.subplots()
    lm = sklearn.linear_model.LinearRegression().fit(labs.reshape(-1, 1), preds)
    lm_y = lm.predict(labs.reshape(-1, 1))

    ax.scatter(labs, preds, s=5, alpha=0.5)
    ax.plot(labs, lm_y, color="red")

    rsq = lm.score(labs.reshape(-1, 1), preds)

    ax.set_xlabel("Target")
    ax.set_ylabel("Predicted")
    ax.set_title(title + f" (R^2 = {rsq:.3f})", fontsize="small")

    plt.tight_layout()

    plt.savefig(save_path)
    plt.close()
    log.info(f"Saved correlation plot to {save_path}")

    _save_cache({"predictions": preds, "reference": labs, "rsquared": rsq}, save_path)


def plot_pred_vs_true_oracle(targets, preds, save_path):
    """
    Plot predicted vs true values for BO surrogate model evaluation.

    Shows deviation from perfect prediction (y=x line). Used for assessing
    how well the Bayesian optimization surrogate model predicts oracle scores.
    """
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.numpy()

    mse = mean_squared_error(targets, preds)
    r2 = r2_score(targets, preds)
    spearman_corr, _ = spearmanr(targets, preds)

    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, alpha=0.6)
    plt.plot(
        [targets.min(), targets.max()],
        [targets.min(), targets.max()],
        "r--",
        lw=2,
        label="Perfect Prediction (y=x)",
    )
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(
        f"Predicted vs True Values\nMSE={mse:.3f}, R²={r2:.3f}, Spearman={spearman_corr:.3f}"
    )
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    _save_cache(
        {
            "targets": targets,
            "preds": preds,
        },
        save_path,
    )

    return mse, r2, spearman_corr


def plot_r_squared(r_squared_history: List[float], save_path: str) -> None:
    """Plot R² history through BO iterations."""
    iterations = list(range(len(r_squared_history)))

    plt.plot(iterations, r_squared_history)

    plt.xlabel("Iteration")
    plt.ylabel("R Squared")

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.close()

    _save_cache(
        {"r_squared_history": r_squared_history},
        save_path,
    )


def _plot_hits_single(
    iterations: list[int],
    bo_data: Optional[list[int]],
    baselines: Optional[dict[str, list[int]]],
    save_path: str,
    ylabel: str,
) -> None:
    """Plot a single hits chart (either unique or total)."""
    if bo_data:
        cumulative = np.cumsum(bo_data)
        plt.plot(iterations, cumulative, label="BO", marker="o")

    if baselines:
        markers = ["s", "^", "D", "v"]
        for idx, (name, data) in enumerate(baselines.items()):
            marker = markers[idx % len(markers)]
            cumulative = np.cumsum(data)
            plt.plot(iterations, cumulative, label=name, marker=marker)

    plt.xlabel("Iteration")
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_hits(
    new_hits_history: list[int],
    hits_history: list[int],
    save_path: str,
    baselines: Optional[dict[str, tuple[list[int], list[int]]]] = None,
) -> None:
    """
    Plot cumulative hits through iterations as two separate plots.

    Args:
        new_hits_history: Per-round unique hits for the main method (can be empty).
        hits_history: Per-round total hits for the main method (can be empty).
        save_path: Base path to save plots (will generate _unique.png and _total.png).
        baselines: Optional dict mapping baseline name to (new_hits, hits) tuples.
    """
    if hits_history:
        iterations = list(range(len(hits_history)))
    elif baselines:
        first_baseline = next(iter(baselines.values()))
        iterations = list(range(len(first_baseline[0])))
    else:
        return

    base_path = save_path.replace(".png", "")
    unique_path = f"{base_path}_unique.png"
    total_path = f"{base_path}_total.png"

    unique_baselines = None
    total_baselines = None
    if baselines:
        unique_baselines = {name: data[0] for name, data in baselines.items()}
        total_baselines = {name: data[1] for name, data in baselines.items()}

    _plot_hits_single(
        iterations,
        new_hits_history if new_hits_history else None,
        unique_baselines,
        unique_path,
        "Cumulative Unique Hits",
    )

    _plot_hits_single(
        iterations,
        hits_history if hits_history else None,
        total_baselines,
        total_path,
        "Cumulative Total Hits",
    )

    cache_data = {
        "new_hits_history": new_hits_history,
        "hits_history": hits_history,
    }
    if baselines:
        for name, (baseline_new_hits, baseline_hits) in baselines.items():
            cache_data[f"{name}_new_hits"] = baseline_new_hits
            cache_data[f"{name}_hits"] = baseline_hits

    _save_cache(cache_data, save_path)


def plot_top_k_fitness(
    top_k_history: Dict[int, List[Optional[float]]],
    save_path: str,
    baselines_top_k: Optional[Dict[str, Dict[int, List[Optional[float]]]]] = None,
) -> None:
    """
    Plot top-k fitness across iterations for each k value.

    Args:
        top_k_history: Dict mapping k -> list of k-th best fitness per round for BO.
        save_path: Base path to save plots (will generate _top_1.png, _top_10.png, etc.).
        baselines_top_k: Optional dict mapping baseline name to top_k_history dicts.
    """
    if not top_k_history:
        return

    k_values = list(top_k_history.keys())
    first_k = k_values[0]
    iterations = list(range(len(top_k_history[first_k])))

    base_path = save_path.replace(".png", "")

    for k in k_values:
        fig_path = f"{base_path}_top_{k}.png"

        bo_data = top_k_history[k]
        if any(v is not None for v in bo_data):
            plt.plot(iterations, bo_data, label="BO", marker="o")

        if baselines_top_k:
            markers = ["s", "^", "D", "v"]
            for idx, (name, baseline_data) in enumerate(baselines_top_k.items()):
                marker = markers[idx % len(markers)]
                if k in baseline_data and any(v is not None for v in baseline_data[k]):
                    plt.plot(iterations, baseline_data[k], label=name, marker=marker)

        plt.xlabel("Iteration")
        plt.ylabel(f"Top-{k} Fitness")
        plt.legend()
        plt.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_landscape_histograms(
    landscape,
    objective_names,
    thresholds,
    vis_dir,
    policy,
    cfg,
    landscape_name,
):
    """Plot histograms for each objective in the landscape."""
    use_nnk = landscape_name in ["mhc_flurry", "plm_loglh"]

    for obj_name in objective_names:
        obj_threshold = thresholds[obj_name] if len(objective_names) > 1 else thresholds
        objective_arg = obj_name if len(objective_names) > 1 else None
        suffix = f"_{obj_name}" if len(objective_names) > 1 else ""

        if is_landscape_enumerable(landscape):
            save_path = os.path.join(vis_dir, f"landscape_histogram{suffix}.png")
            plot_landscape_histogram(
                landscape, obj_threshold, save_path, objective=objective_arg
            )
        else:
            log.info(
                f"Landscape too large to enumerate. Plotting MC histogram{f' for {obj_name}' if suffix else ''}."
            )
            n_mc = getattr(cfg, "nnk_mc_samples", 1000)

            if use_nnk:
                random_seqs = generate_nnk_sequences(landscape.str_length, n_mc)
            else:
                random_seqs = generate_uniform_random_sequences(
                    landscape.alphabet, landscape.str_length, n_mc
                )

            random_scores = landscape.get_fitness_values(
                random_seqs, objective=objective_arg
            )
            save_path = os.path.join(
                vis_dir,
                f"{'nnk' if use_nnk else 'uniform'}_random_sequences_histogram{suffix}.png",
            )
            title = f"{'NNK' if use_nnk else 'uniform'} random sequences histogram{f' ({obj_name})' if suffix else ''}"
            plot_score_histogram(random_scores, save_path, title)

        save_path = os.path.join(vis_dir, f"sampled_scores_histogram{suffix}.png")
        plot_sampled_scores_histogram(
            policy,
            landscape,
            num_sequences=1000,
            save_path=save_path,
            objective=objective_arg,
        )

        if (
            isinstance(landscape, PLMLogLikelihoodLandscape)
            and obj_name in landscape.models
        ):
            model = landscape.models[obj_name]
            if model.name == "linear":
                save_path = os.path.join(vis_dir, f"linear_plm_target{suffix}.png")
                plot_linear_plm_target(model, save_path)


def compute_empirical_aa_frequencies(
    protein_sequences: list[str], expected_length: Optional[int] = None
) -> np.ndarray:
    """
    Compute empirical amino acid frequencies at each position from sampled protein sequences.

    Args:
        protein_sequences: List of protein sequences (same length expected).
        expected_length: If provided, only sequences of this length are included.

    Returns:
        Array of shape (seq_length, 21) with AA frequencies at each position.
        The AA order follows AMINO_ACIDS from constants.py.
    """
    if expected_length is not None:
        protein_sequences = [s for s in protein_sequences if len(s) == expected_length]

    if not protein_sequences:
        raise ValueError("No valid protein sequences to compute frequencies")

    seq_length = len(protein_sequences[0])
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    counts = np.zeros((seq_length, len(AMINO_ACIDS)))

    for seq in protein_sequences:
        if len(seq) != seq_length:
            continue
        for pos, aa in enumerate(seq):
            if aa in aa_to_idx:
                counts[pos, aa_to_idx[aa]] += 1

    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    frequencies = counts / row_sums
    return frequencies


def plot_sequence_logo(
    protein_sequences: list[str],
    save_path: str,
    title: str = "Sequence Logo",
) -> None:
    """
    Plot sequence logos from protein sequences using logomaker.

    Saves two plots: one with information content (bits) and one with frequencies.

    Args:
        protein_sequences: List of equal-length protein sequences.
        save_path: Path to save the plot (will add _info and _freq suffixes).
        title: Title for the plot.
    """
    counts_matrix = logomaker.alignment_to_matrix(protein_sequences, to_type="counts")
    freq_matrix = logomaker.transform_matrix(
        counts_matrix, from_type="counts", to_type="probability"
    )
    info_matrix = logomaker.transform_matrix(
        counts_matrix, from_type="counts", to_type="information"
    )

    seq_length = len(freq_matrix)
    total_seqs = len(protein_sequences)
    log.info(f"Sequence logo AA counts ({total_seqs} samples):")
    deterministic_count = 0
    for pos in range(seq_length):
        counts = counts_matrix.iloc[pos]
        nonzero = counts[counts > 0].sort_values(ascending=False).astype(int)
        counts_str = " ".join([f"{aa}:{count}" for aa, count in nonzero.items()])
        is_deterministic = counts.max() == total_seqs
        if is_deterministic:
            deterministic_count += 1
        marker = " [DETERMINISTIC]" if is_deterministic else ""
        log.info(f"  Position {pos + 1:2d}: {counts_str}{marker}")
    log.info(f"  Summary: {deterministic_count}/{seq_length} positions deterministic")

    seq_length = len(freq_matrix)
    fig_width = max(12, seq_length * 0.5)

    base_path = save_path.rsplit(".", 1)[0]
    ext = save_path.rsplit(".", 1)[1] if "." in save_path else "png"

    fig, ax = plt.subplots(figsize=(fig_width, 3))
    logomaker.Logo(info_matrix, ax=ax, color_scheme="chemistry")
    ax.set_xlabel("Position")
    ax.set_ylabel("Information (bits)")
    ax.set_title(title)
    ax.set_xlim(-0.5, seq_length - 0.5)
    ax.set_xticks(range(seq_length))
    ax.set_xticklabels([str(i + 1) for i in range(seq_length)])
    plt.tight_layout()
    info_path = f"{base_path}_info.{ext}"
    plt.savefig(info_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved information sequence logo to {info_path}")

    fig, ax = plt.subplots(figsize=(fig_width, 3))
    logomaker.Logo(freq_matrix, ax=ax, color_scheme="chemistry")
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.set_xlim(-0.5, seq_length - 0.5)
    ax.set_xticks(range(seq_length))
    ax.set_xticklabels([str(i + 1) for i in range(seq_length)])
    plt.tight_layout()
    freq_path = f"{base_path}_freq.{ext}"
    plt.savefig(freq_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved frequency sequence logo to {freq_path}")

    _save_cache(
        {
            "frequency_matrix": freq_matrix.to_dict(),
            "information_matrix": info_matrix.to_dict(),
            "title": title,
            "num_sequences": len(protein_sequences),
        },
        save_path,
    )


def compute_tvd_per_position(
    sampled_freqs: np.ndarray,
    reference_freqs: np.ndarray,
) -> np.ndarray:
    """
    Compute Total Variation Distance between two distributions at each position.

    TVD = 0.5 * sum(|p - q|) for each position.

    Args:
        sampled_freqs: (seq_len, 20) AA frequencies from sampled sequences.
        reference_freqs: (seq_len, 20) AA frequencies from reference FASTA.

    Returns:
        (seq_len,) array of TVD values per position.
    """
    return 0.5 * np.abs(sampled_freqs - reference_freqs).sum(axis=1)


def _filter_seqs_for_tvd(
    seqs: list[str], expected_length: int, include_stops: bool
) -> list[str]:
    """Filter sequences by length and optionally exclude stop codons.

    When include_stops=True, sequences shorter than expected_length are padded
    with '*' (representing stop codons at all remaining positions).
    """
    if include_stops:
        result = []
        for s in seqs:
            if len(s) == expected_length:
                result.append(s)
            elif len(s) < expected_length:
                # Pad with stops to expected length
                result.append(s + "*" * (expected_length - len(s)))
        return result
    return [s for s in seqs if len(s) == expected_length and "*" not in s]


def _plot_tvd_core(
    sampled_seqs: list[str],
    reference_seqs: list[str],
    save_path: str,
    title: str,
    expected_length: int,
    max_samples: Optional[int],
    var_synth_seqs: Optional[list[str]],
    figsize: tuple[float, float],
    paper_ready: bool,
    include_stops: bool,
) -> None:
    """Core TVD plotting logic, called by plot_tvd_per_position for both variants."""
    num_aas = 21 if include_stops else 20

    # 1. Filter sequences
    sampled_filtered = _filter_seqs_for_tvd(
        sampled_seqs, expected_length, include_stops
    )
    reference_filtered = _filter_seqs_for_tvd(
        reference_seqs, expected_length, include_stops
    )

    if not sampled_filtered:
        log.warning(f"No sampled sequences match expected_length={expected_length}")
        return
    if not reference_filtered:
        log.warning(f"No reference sequences match expected_length={expected_length}")
        return

    # 2. Subsample if exceeds max_samples
    if max_samples is not None:
        random.seed(42)
        if len(sampled_filtered) > max_samples:
            sampled_filtered = random.sample(sampled_filtered, max_samples)
        if len(reference_filtered) > max_samples:
            reference_filtered = random.sample(reference_filtered, max_samples)

    # 3. Compute AA frequencies
    sampled_freqs = compute_empirical_aa_frequencies(sampled_filtered, expected_length)[
        :, :num_aas
    ]
    reference_freqs = compute_empirical_aa_frequencies(
        reference_filtered, expected_length
    )[:, :num_aas]

    # 4. Compute TVD at each position
    tvd_per_pos = compute_tvd_per_position(sampled_freqs, reference_freqs)

    # 5. Compute var_synth TVD if provided
    var_synth_tvd = None
    if var_synth_seqs is not None:
        var_synth_filtered = _filter_seqs_for_tvd(
            var_synth_seqs, expected_length, include_stops
        )
        if var_synth_filtered:
            if max_samples is not None and len(var_synth_filtered) > max_samples:
                var_synth_filtered = random.sample(var_synth_filtered, max_samples)
            var_synth_freqs = compute_empirical_aa_frequencies(
                var_synth_filtered, expected_length
            )[:, :num_aas]
            var_synth_tvd = compute_tvd_per_position(var_synth_freqs, reference_freqs)

    # 6. Plot
    fig, ax = plt.subplots(figsize=figsize or (max(10, expected_length * 0.4), 6))
    positions = np.arange(1, expected_length + 1)

    pgld_label = None if paper_ready else "PGLD"
    var_label = None if paper_ready else "Var-Synth"

    ax.scatter(
        positions,
        tvd_per_pos,
        marker="o",
        s=50,
        alpha=0.8,
        color=PAPER_COLORS["PGLD"],
        label=pgld_label,
    )

    if var_synth_tvd is not None:
        ax.scatter(
            positions,
            var_synth_tvd,
            marker="s",
            s=50,
            alpha=0.8,
            color=PAPER_COLORS["Var-Synth"],
            label=var_label,
        )

    ax.set_xlabel("Position")
    ax.set_ylabel("Total Var. Distance")
    ax.set_title(title)
    ax.set_xlim(0.5, expected_length + 0.5)
    max_tvd = tvd_per_pos.max()
    if var_synth_tvd is not None:
        max_tvd = max(max_tvd, var_synth_tvd.max())
    ax.set_ylim(0, min(1.0, max_tvd * 1.2 + 0.05))
    if paper_ready and expected_length > 15:
        ax.set_xticks(positions[::3])
    else:
        ax.set_xticks(positions)
    ax.grid(True, alpha=0.3)

    # Add mean TVD annotations
    mean_tvd = tvd_per_pos.mean()
    mean_label = None if paper_ready else f"PGLD Mean: {mean_tvd:.3f}"
    ax.axhline(
        mean_tvd,
        color=PAPER_COLORS["PGLD"],
        linestyle="--",
        alpha=0.5,
        label=mean_label,
    )
    if var_synth_tvd is not None:
        mean_var_synth_tvd = var_synth_tvd.mean()
        var_mean_label = (
            None if paper_ready else f"Var-Synth Mean: {mean_var_synth_tvd:.3f}"
        )
        ax.axhline(
            mean_var_synth_tvd,
            color=PAPER_COLORS["Var-Synth"],
            linestyle="--",
            alpha=0.5,
            label=var_mean_label,
        )
    if not paper_ready:
        ax.legend()

    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
        plt.savefig(save_path, dpi=300)
    else:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    log.info(f"Saved TVD per-position plot to {save_path}")

    _save_cache(
        {
            "tvd_per_position": tvd_per_pos.tolist(),
            "positions": positions.tolist(),
            "mean_tvd": float(mean_tvd),
            "sampled_count": len(sampled_filtered),
            "reference_count": len(reference_filtered),
            "expected_length": expected_length,
            "include_stops": include_stops,
        },
        save_path,
    )


def plot_tvd_per_position(
    sampled_seqs: list[str],
    reference_fasta_path: str,
    save_path: str,
    title: str = "Per-Position TVD",
    expected_length: Optional[int] = None,
    max_samples: Optional[int] = None,
    var_synth_seqs: Optional[list[str]] = None,
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """
    Plot TVD at each AA position between sampled and reference sequences.
    Generates two plots: one excluding stop codons, one including them.

    Args:
        sampled_seqs: List of AA sequences from model.
        reference_fasta_path: Path to FASTA file with reference distribution.
        save_path: Where to save the plot (without stops).
        title: Plot title.
        expected_length: Filter sequences to this length (removes early stops).
        max_samples: Subsample both sets to this size if they exceed it.
        var_synth_seqs: Optional list of var_synth AA sequences to compare.
    """
    # 1. Load reference sequences from FASTA
    reference_seqs = load_fasta(reference_fasta_path)
    if not reference_seqs:
        raise ValueError(
            f"No sequences found in reference FASTA: {reference_fasta_path}"
        )

    # 2. Determine expected length if not provided
    if expected_length is None:
        ref_lengths = [len(s) for s in reference_seqs]
        expected_length = max(set(ref_lengths), key=ref_lengths.count)
        log.info(
            f"TVD plot: using expected_length={expected_length} from reference FASTA"
        )

    # 3. Generate plot without stop codons
    _plot_tvd_core(
        sampled_seqs=sampled_seqs,
        reference_seqs=reference_seqs,
        save_path=save_path,
        title=title,
        expected_length=expected_length,
        max_samples=max_samples,
        var_synth_seqs=var_synth_seqs,
        figsize=figsize,
        paper_ready=paper_ready,
        include_stops=False,
    )

    # 4. Generate plot with stop codons
    base, ext = os.path.splitext(save_path)
    save_path_with_stops = f"{base}_with_stops{ext}"
    _plot_tvd_core(
        sampled_seqs=sampled_seqs,
        reference_seqs=reference_seqs,
        save_path=save_path_with_stops,
        title=title,
        expected_length=expected_length,
        max_samples=max_samples,
        var_synth_seqs=var_synth_seqs,
        figsize=figsize,
        paper_ready=paper_ready,
        include_stops=True,
    )


# Numba-compiled kernel distance for UMAP (module-level to avoid recompilation)
# Heavy-tailed Hamming kernel: k(x,y) = (C + d_H(x,y))^(-β), with C=1, β=0.5
# UMAP distance = 1/k - 1
@njit(fastmath=True)
def _hamming_kernel_distance(x, y):
    hamming = 0
    for i in range(x.shape[0]):
        if x[i] != y[i]:
            hamming += 1
    C, beta = 1.0, 0.5
    kernel = (C + hamming) ** (-beta)
    return 1.0 / kernel - 1.0


def plot_umap_sequence_space_grid(
    reference_seqs: list[str],
    var_synth_seqs: Optional[list[str]],
    policy_seqs: Optional[list[str]],
    output_dir: str,
    beta: float = 0.5,
    C: float = 1.0,
    max_plot_samples: Optional[int] = None,
    expected_length: Optional[int] = None,
    figsize: tuple[float, float] = None,
    plot_ext: str = ".png",
    paper_ready: bool = False,
) -> None:
    """Generate UMAP plots for sequence space visualization.

    Computes UMAP embedding using approximate nearest neighbors with a custom
    kernel distance metric, then generates:
    - umap_reference.png: Reference sequences only
    - umap_ref_vs_varsynth.png: Reference + Var-Synth (if provided)
    - umap_ref_vs_policy.png: Reference + PGLD (if provided)

    Uses heavy-tailed Hamming kernel: k(x,y) = (C + dH(x,y))^(-β)

    Args:
        reference_seqs: Reference sequences (e.g., from IgLM). All used for fitting.
        var_synth_seqs: Var-synth sequences (optional).
        policy_seqs: Policy/PGLD sequences (optional).
        output_dir: Directory to save plots.
        beta: Kernel parameter (default 0.5).
        C: Kernel parameter (default 1.0).
        max_plot_samples: Max reference points to plot (all used for fitting).
        expected_length: If provided, filter to sequences of this length with no stop codons.
            If None, use all sequences with padding for variable lengths.
    """
    if not reference_seqs:
        log.warning("No reference sequences provided for UMAP plot")
        return

    # Lazy import to avoid numba JIT compilation at module load (causes NFS issues)
    from umap import UMAP

    # 0. Optionally filter sequences by length and stop codons
    def _filter(seqs: list[str]) -> list[str]:
        if expected_length is None:
            return seqs
        return [s for s in seqs if len(s) == expected_length and "*" not in s]

    reference_seqs = _filter(reference_seqs)
    if var_synth_seqs:
        var_synth_seqs = _filter(var_synth_seqs)
    if policy_seqs:
        policy_seqs = _filter(policy_seqs)

    if not reference_seqs:
        log.warning("No reference sequences remain after filtering")
        return

    # 1. Track sequence counts
    n_ref = len(reference_seqs)
    n_var = len(var_synth_seqs) if var_synth_seqs else 0
    n_policy = len(policy_seqs) if policy_seqs else 0

    log.info(f"UMAP: {n_ref} reference, {n_var} var-synth, {n_policy} policy sequences")

    # 2. Encode all sequences together (ensures consistent padding length)
    all_seqs = list(reference_seqs)
    if var_synth_seqs:
        all_seqs.extend(var_synth_seqs)
    if policy_seqs:
        all_seqs.extend(policy_seqs)

    log.info("Encoding sequences...")
    encoded = encode_sequences(all_seqs)  # (N, L)

    # Split encodings
    ref_enc = encoded[:n_ref]
    var_enc = encoded[n_ref : n_ref + n_var] if n_var > 0 else None
    policy_enc = encoded[n_ref + n_var :] if n_policy > 0 else None

    # 3. Fit UMAP only on reference sequences (much faster)
    log.info("Fitting UMAP on reference sequences...")
    reducer = UMAP(
        metric=_hamming_kernel_distance,
        n_neighbors=30,  # more neighbors for smoother global structure
        min_dist=0.1,  # allow tight clusters
        n_epochs=500,  # ensure convergence (default 200 for large data)
        random_state=42,
    )
    ref_emb = reducer.fit_transform(ref_enc)

    # 4. Transform var_synth and policy sequences
    var_emb = None
    policy_emb = None
    if var_enc is not None:
        log.info("Transforming var-synth sequences...")
        var_emb = reducer.transform(var_enc)
    if policy_enc is not None:
        log.info("Transforming policy sequences...")
        policy_emb = reducer.transform(policy_enc)

    # Subsample reference for plotting (all used for fitting above)
    if max_plot_samples is not None and len(ref_emb) > max_plot_samples:
        plot_idx = np.random.default_rng(42).choice(
            len(ref_emb), max_plot_samples, replace=False
        )
        ref_emb_plot = ref_emb[plot_idx]
        log.info(
            f"Subsampled {len(ref_emb)} -> {max_plot_samples} reference points for plotting"
        )
    else:
        ref_emb_plot = ref_emb

    # 6. Generate plots
    from constants import PAPER_COLORS, PAPER_MARGINS

    def _plot_umap(
        ref_points: np.ndarray,
        overlay_points: Optional[np.ndarray],
        filename: str,
        title: str,
        overlay_label: str = "Overlay",
    ):
        fig, ax = plt.subplots(figsize=figsize or (10, 8))

        # Plot reference as background (gray, small, low alpha)
        ref_label = None if paper_ready else "Reference"
        ref_size = 1 if paper_ready else 5
        ref_alpha = 0.6 if paper_ready else 0.5
        ax.scatter(
            ref_points[:, 0],
            ref_points[:, 1],
            c=PAPER_COLORS["Reference"],
            s=ref_size,
            alpha=ref_alpha,
            label=ref_label,
        )

        # Plot overlay if provided (colored, slightly larger)
        if overlay_points is not None:
            # Use consistent colors based on label
            if overlay_label == "PGLD":
                color = PAPER_COLORS["PGLD"]
            elif overlay_label == "Var-Synth":
                color = PAPER_COLORS["Var-Synth"]
            else:
                color = "steelblue"
            display_label = None if paper_ready else overlay_label
            overlay_size = 1 if paper_ready else 10
            overlay_alpha = 0.4 if paper_ready else 0.55
            ax.scatter(
                overlay_points[:, 0],
                overlay_points[:, 1],
                c=color,
                s=overlay_size,
                alpha=overlay_alpha,
                label=display_label,
            )

        if paper_ready:
            # Set axis limits based on reference distribution percentiles to avoid outliers
            x_low, x_high = np.percentile(ref_points[:, 0], [1, 99])
            y_low, y_high = np.percentile(ref_points[:, 1], [1, 99])
            x_pad = (x_high - x_low) * 0.05
            y_pad = (y_high - y_low) * 0.05
            ax.set_xlim(x_low - x_pad, x_high + x_pad)
            ax.set_ylim(y_low - y_pad, y_high + y_pad)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.legend()
            ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(title)

        if paper_ready:
            plt.subplots_adjust(**PAPER_MARGINS)
        else:
            plt.tight_layout()
        save_path = os.path.join(output_dir, filename)
        plt.savefig(save_path, dpi=150)
        plt.close()
        log.info(f"Saved UMAP plot to {save_path}")

    # Plot 1: Reference only
    _plot_umap(
        ref_emb_plot, None, f"umap_reference{plot_ext}", "UMAP of Reference Sequences"
    )

    # Plot 2: Reference + Var-Synth
    if var_emb is not None:
        _plot_umap(
            ref_emb_plot,
            var_emb,
            f"umap_ref_vs_varsynth{plot_ext}",
            "UMAP Var-Synth",
            "Var-Synth",
        )

    # Plot 3: Reference + Policy
    if policy_emb is not None:
        _plot_umap(
            ref_emb_plot,
            policy_emb,
            f"umap_ref_vs_policy{plot_ext}",
            "UMAP PGLD",
            "PGLD",
        )


def plot_mmd_comparison(
    mmd_policy: float,
    mmd_var_synth: float,
    save_path: str,
    title: str = "MMD²",
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """Bar plot comparing MMD² scores."""
    from constants import PAPER_COLORS, PAPER_MARGINS

    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    x_positions = [0, 1]
    colors = [PAPER_COLORS["PGLD"], PAPER_COLORS["Var-Synth"]]
    bars = ax.bar(x_positions, [mmd_policy, mmd_var_synth], color=colors)
    if paper_ready:
        ax.set_title("MMD² (V-stat)")
    else:
        ax.set_ylabel("MMD² (V-stat)")
        ax.set_title(title)
    max_val = max(mmd_policy, mmd_var_synth)
    ax.set_ylim(0, max_val * 1.15)  # 15% headroom
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["PGLD", "Var-Synth"])
    # Always show value labels on top of bars (colored in paper_ready mode)
    label_fontsize = 20 if paper_ready else 10
    for bar, v, c in zip(bars, [mmd_policy, mmd_var_synth], colors):
        text_color = c if paper_ready else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max_val * 0.02,  # small offset above bar
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color=text_color,
        )
    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved MMD comparison plot to {save_path}")


def plot_early_stop_comparison(
    pct_policy: float,
    pct_var_synth: float,
    save_path: str,
    title: str = "Early Stop %",
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """Bar plot comparing early stop percentages."""
    from constants import PAPER_COLORS, PAPER_MARGINS

    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    x_positions = [0, 1]
    colors = [PAPER_COLORS["PGLD"], PAPER_COLORS["Var-Synth"]]
    bars = ax.bar(x_positions, [pct_policy, pct_var_synth], color=colors)
    if not paper_ready:
        ax.set_ylabel("Early Stop (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(pct_policy, pct_var_synth, 1) * 1.15)  # 15% headroom
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["PGLD", "Var-Synth"])
    # Always show value labels on top of bars (colored in paper_ready mode)
    label_fontsize = 20 if paper_ready else 10
    for bar, v, c in zip(bars, [pct_policy, pct_var_synth], colors):
        text_color = c if paper_ready else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.5,  # small offset above bar
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color=text_color,
        )
    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved early stop comparison plot to {save_path}")


def plot_witness_inlier_fraction(
    frac_policy: float,
    frac_var_synth: float,
    save_path: str,
    title: str = "Clearly-Realistic Fraction",
    ideal: float = 0.67,
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """Bar plot of the MMD witness-function 'clearly realistic' fraction.

    ideal = 1 - alpha is the fraction expected if the model matched the data.
    """
    from constants import PAPER_COLORS, PAPER_MARGINS

    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    x_positions = [0, 1]
    colors = [PAPER_COLORS["PGLD"], PAPER_COLORS["Var-Synth"]]
    pcts = [100 * frac_policy, 100 * frac_var_synth]
    bars = ax.bar(x_positions, pcts, color=colors)
    ax.axhline(
        100 * ideal,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label=f"data match ({100*ideal:.0f}%)",
    )
    if not paper_ready:
        ax.set_ylabel("Clearly realistic (%)")
    ax.set_title(title)
    ax.set_ylim(0, max(pcts + [100 * ideal]) * 1.15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["PGLD", "Var-Synth"])
    ax.legend(fontsize=9, loc="upper right")
    label_fontsize = 20 if paper_ready else 10
    for bar, v, c in zip(bars, pcts, colors):
        text_color = c if paper_ready else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + 0.5,
            f"{v:.1f}%",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color=text_color,
        )
    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved witness inlier fraction plot to {save_path}")


def plot_bear_bayes_factor(
    bf_policy: float,
    bf_var_synth: float,
    save_path: str,
    title: str = "BEAR Bayes Factor",
    figsize: tuple[float, float] = None,
    paper_ready: bool = False,
) -> None:
    """Bar plot of the BEAR two-sample per-sequence geometric-average Bayes factor.

    BF = 1 (dashed line) means the model and data distributions are indistinguishable;
    larger BF is stronger evidence they differ.
    """
    from constants import PAPER_COLORS, PAPER_MARGINS

    fig, ax = plt.subplots(figsize=figsize or (6, 5))
    x_positions = [0, 1]
    colors = [PAPER_COLORS["PGLD"], PAPER_COLORS["Var-Synth"]]
    vals = [bf_policy, bf_var_synth]
    bars = ax.bar(x_positions, vals, color=colors)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.5)
    if not paper_ready:
        ax.set_ylabel("Per-sequence Bayes factor")
    ax.set_title(title)
    ax.set_ylim(0, max(vals + [1.0]) * 1.15)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["PGLD", "Var-Synth"])
    label_fontsize = 20 if paper_ready else 10
    for bar, v, c in zip(bars, vals, colors):
        text_color = c if paper_ready else "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + max(vals) * 0.02,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=label_fontsize,
            color=text_color,
        )
    if paper_ready:
        plt.subplots_adjust(**PAPER_MARGINS)
    else:
        plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"Saved BEAR Bayes factor plot to {save_path}")
