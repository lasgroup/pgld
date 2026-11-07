"""
train_bo.py

Main script for running Batch Bayesian Optimization.
Uses Hydra for configuration management.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import csv
import logging
import os
import pickle
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from train_utils_policies import run_extra_eval
from copy import deepcopy

from bo import BBOAlgorithm, evaluate_ensemble_on_held_out
from generate_training_data import (
    create_data_from_policy,
    create_dataloaders,
    get_starting_data_for_bo,
)
from landscapes import (
    DHFRFitnessLandscape,
    GB1FitnessLandscape,
    MHCflurryLandscape,
    PhoQFitnessLandscape,
    RiboswitchLandscape,
    TFBindingLandscape,
    BoLandscape,
    dna_to_protein,
    enumerate_sequences_and_score,
    generate_iupac_sequences,
    generate_nnk_sequences,
    generate_uniform_random_sequences,
    get_top_proportion_threshold,
    is_landscape_enumerable,
)
from oracle import OracleCNNSmall, OracleCNNBig
from rational_design import (
    compute_aa_frequencies_from_hits,
    compute_nucleotide_frequencies_from_hits,
    design_degenerate_codons_simple,
    design_iupac_codes_simple,
)
from policies import (
    ArbitraryBaseOptimizer,
    BottleBasedOptimizer,
    FiniteBaseOptimizer,
    evaluate_on_batch,
)
from policy_training import train_policy, pooled_train_policy
from entropy import calculate_entropy, get_entropy_function
from rewards import get_reward_function
from vis import (
    plot_arbitrary_optimizer_results,
    plot_finite_optimizer_results,
    plot_reward_history,
    plot_training_history_oracle,
    plot_pred_vs_true_oracle,
    plot_sampled_scores_histogram,
    plot_r_squared,
    plot_hits,
    plot_top_k_fitness,
    print_oligo_ordering_template,
    print_theoretical_diversity,
)
from library_size_estimator import print_total_library_diversity
from setup import (
    build_invalid_scores,
    build_landscape,
    build_policy,
    build_thresholds,
    setup_device,
    setup_seeds,
)
from config_schemas import build_bo_config, build_training_config


log = logging.getLogger(__name__)

TOP_K_VALUES = [1, 10, 100]


def save_sampled_sequences(
    sequences: List[str],
    fitness_values: Dict[str, List[float]],
    output_path: str,
) -> None:
    """Save sampled sequences and their fitness values to a pickle file."""
    data = {
        seq: {obj: fitness_values[obj][i] for obj in fitness_values}
        for i, seq in enumerate(sequences)
    }
    with open(output_path, "wb") as f:
        pickle.dump(data, f)
    log.info(f"Saved {len(data)} sampled sequences to {output_path}")


def get_top_k(unique_fitness: Dict[str, float], k: int) -> Optional[float]:
    """Return k-th best fitness (1-indexed), or None if fewer than k sequences."""
    if len(unique_fitness) < k:
        return None
    return sorted(unique_fitness.values(), reverse=True)[k - 1]


def build_ensembles(
    objectives: list[str],
    num_ensembles: int,
    input_dim: int,
    input_channels: int,
    device: str,
) -> dict[str, nn.ModuleList]:
    """Build an ensemble of oracle models for each objective."""
    ensembles = {}
    for obj in objectives:
        ensemble = nn.ModuleList(
            [
                OracleCNNBig(input_dim=input_dim, input_channels=input_channels)
                for _ in range(num_ensembles)
            ]
        )
        ensembles[obj] = ensemble.to(device)
    return ensembles


def calculate_new_hits(
    sampled_sequences: list[str],
    new_observations: list[str],
    new_fitness_values: dict[str, list[float]],
    bo_thresholds: dict[str, float],
) -> tuple[int, int]:
    """
    Calculate hits using AND logic across all objectives.

    Args:
        sampled_sequences: Previously seen sequences.
        new_observations: New sequences to evaluate.
        new_fitness_values: Dict mapping objective -> list of fitness values.
        bo_thresholds: Dict mapping objective -> threshold.

    Returns:
        (new_hits, hits): New unique hits and total hits in batch.
    """
    sampled_sequences_set = set(sampled_sequences)
    new_hits = 0
    hits = 0

    objectives = list(new_fitness_values.keys())
    n_sequences = len(new_observations)

    for i in range(n_sequences):
        # AND logic: must pass ALL thresholds
        is_hit = all(
            new_fitness_values[obj][i] > bo_thresholds[obj] for obj in objectives
        )

        if is_hit:
            hits += 1
            if new_observations[i] not in sampled_sequences_set:
                new_hits += 1
                sampled_sequences_set.add(new_observations[i])

    return new_hits, hits


def run_random_baseline(
    landscape,
    bo_thresholds: Dict[str, float],
    bo_iters: int,
    samples_per_round: int,
    initial_observations: List[str],
    initial_fitness_values: Dict[str, List[float]],
    output_dir: Optional[str] = None,
) -> Tuple[List[int], List[int], Dict[int, List[Optional[float]]]]:
    """
    Run a random sampling baseline for comparison with BO.

    Uses the same initial observations as BO for round 0, then samples
    randomly for the remaining bo_iters rounds.

    Args:
        landscape: The landscape to sample from and evaluate.
        bo_thresholds: Dict mapping objective -> threshold for hit detection.
        bo_iters: Number of BO iterations after round 0.
        samples_per_round: Number of sequences to sample per round.
        initial_observations: Initial peptide sequences (shared with BO).
        initial_fitness_values: Dict mapping objective -> list of fitness values.

    Returns:
        (new_hits_history, hits_history, top_k_history): Per-round metrics.
    """
    sampled_sequences: List[str] = []
    new_hits_history: List[int] = []
    hits_history: List[int] = []

    first_obj = list(bo_thresholds.keys())[0]
    unique_fitness: Dict[str, float] = {}
    top_k_history: Dict[int, List[Optional[float]]] = {k: [] for k in TOP_K_VALUES}

    # Round 0: use shared initial observations
    new_hits, hits = calculate_new_hits(
        sampled_sequences, initial_observations, initial_fitness_values, bo_thresholds
    )
    sampled_sequences.extend(initial_observations)
    new_hits_history.append(new_hits)
    hits_history.append(hits)

    for seq, fit in zip(initial_observations, initial_fitness_values[first_obj]):
        unique_fitness[seq] = max(unique_fitness.get(seq, float("-inf")), fit)
    for k in TOP_K_VALUES:
        top_k_history[k].append(get_top_k(unique_fitness, k))

    # Remaining rounds: random sampling
    for round_idx in range(bo_iters):
        if landscape.is_peptide:
            # Use NNK codon sampling to match initial data generation
            dna_sequences = generate_nnk_sequences(
                landscape.str_length, samples_per_round
            )
        else:
            dna_sequences = generate_uniform_random_sequences(
                landscape.alphabet, landscape.str_length, samples_per_round
            )
        observations, fitness_values = landscape.annotate_sequences(dna_sequences)

        if output_dir:
            save_path = os.path.join(output_dir, f"random_{round_idx + 1}.pkl")
            save_sampled_sequences(observations, fitness_values, save_path)

        new_hits, hits = calculate_new_hits(
            sampled_sequences, observations, fitness_values, bo_thresholds
        )
        sampled_sequences.extend(observations)

        for seq, fit in zip(observations, fitness_values[first_obj]):
            unique_fitness[seq] = max(unique_fitness.get(seq, float("-inf")), fit)

        for k in TOP_K_VALUES:
            top_k_history[k].append(get_top_k(unique_fitness, k))

        new_hits_history.append(new_hits)
        hits_history.append(hits)

    return new_hits_history, hits_history, top_k_history


def _extract_hits(
    observations: list[str],
    fitness_values: dict[str, list[float]],
    bo_thresholds: dict[str, float],
) -> list[str]:
    """Extract hit peptides from observations using AND logic across objectives."""
    objectives = list(fitness_values.keys())
    hits = []
    for i in range(len(observations)):
        is_hit = all(fitness_values[obj][i] > bo_thresholds[obj] for obj in objectives)
        if is_hit:
            hits.append(observations[i])
    return hits


def _design_library_template(hit_peptides: list[str]) -> str:
    """Design IUPAC template from hit peptides using MSE-based codon selection."""
    unique_peptides = list(dict.fromkeys(hit_peptides))
    fraq_df = compute_aa_frequencies_from_hits(unique_peptides)
    selected_codons = design_degenerate_codons_simple(fraq_df)
    template = "".join(selected_codons[pos] for pos in fraq_df.index)
    return template


def _design_dna_library_template(hit_dna_seqs: list[str]) -> str:
    """Design IUPAC template from hit DNA sequences using MSE-based IUPAC selection."""
    unique_seqs = list(dict.fromkeys(hit_dna_seqs))
    log.info(f"Designing DNA template from {len(unique_seqs)} unique hit sequences")
    fraq_df = compute_nucleotide_frequencies_from_hits(unique_seqs)
    log.info(f"Nucleotide frequencies:\n{fraq_df.to_string()}")
    selected_codes = design_iupac_codes_simple(fraq_df)
    template = "".join(selected_codes[pos] for pos in fraq_df.index)
    return template


def run_rational_design_baseline(
    landscape,
    bo_thresholds: Dict[str, float],
    bo_iters: int,
    samples_per_round: int,
    initial_observations: List[str],
    initial_fitness_values: Dict[str, List[float]],
    output_dir: Optional[str] = None,
) -> Tuple[List[int], List[int], Dict[int, List[Optional[float]]]]:
    """
    Run adaptive rational design baseline for comparison with BO.

    Redesigns the library each round based on accumulated hits. For peptide landscapes,
    designs degenerate codons. For DNA landscapes, designs IUPAC codes directly.

    Args:
        landscape: The landscape to evaluate sequences.
        bo_thresholds: Dict mapping objective -> threshold for hit detection.
        bo_iters: Number of BO iterations (total rounds = bo_iters + 1).
        samples_per_round: Number of sequences to sample per round.
        initial_observations: Initial sequences (from annotate_sequences).
        initial_fitness_values: Dict mapping objective -> list of fitness values.

    Returns:
        (new_hits_history, hits_history, top_k_history): Per-round metrics.
    """
    is_peptide = landscape.is_peptide
    design_template_fn = (
        _design_library_template if is_peptide else _design_dna_library_template
    )

    sampled_sequences: List[str] = []
    new_hits_history: List[int] = []
    hits_history: List[int] = []

    first_obj = list(bo_thresholds.keys())[0]
    unique_fitness: Dict[str, float] = {}
    top_k_history: Dict[int, List[Optional[float]]] = {k: [] for k in TOP_K_VALUES}

    all_hit_seqs: List[str] = []

    initial_hits = _extract_hits(
        initial_observations, initial_fitness_values, bo_thresholds
    )
    all_hit_seqs.extend(initial_hits)

    if not all_hit_seqs:
        log.warning(
            "No hit sequences found in initial observations. Using all sequences."
        )
        all_hit_seqs = list(initial_observations)

    log.info(f"Round 0: Found {len(initial_hits)} hits from initial observations.")

    new_hits, hits = calculate_new_hits(
        sampled_sequences, initial_observations, initial_fitness_values, bo_thresholds
    )
    sampled_sequences.extend(initial_observations)
    new_hits_history.append(new_hits)
    hits_history.append(hits)

    for seq, fit in zip(initial_observations, initial_fitness_values[first_obj]):
        unique_fitness[seq] = max(unique_fitness.get(seq, float("-inf")), fit)
    for k in TOP_K_VALUES:
        top_k_history[k].append(get_top_k(unique_fitness, k))

    template = design_template_fn(all_hit_seqs)
    log.info(f"Round 0: Designed IUPAC template: {template}")

    for round_idx in range(1, bo_iters + 1):
        dna_sequences = generate_iupac_sequences(template, samples_per_round)
        observations, fitness_values = landscape.annotate_sequences(dna_sequences)

        if output_dir:
            save_path = os.path.join(output_dir, f"rational_{round_idx}.pkl")
            save_sampled_sequences(observations, fitness_values, save_path)

        new_hits, hits = calculate_new_hits(
            sampled_sequences, observations, fitness_values, bo_thresholds
        )
        sampled_sequences.extend(observations)
        new_hits_history.append(new_hits)
        hits_history.append(hits)

        for seq, fit in zip(observations, fitness_values[first_obj]):
            unique_fitness[seq] = max(unique_fitness.get(seq, float("-inf")), fit)
        for k in TOP_K_VALUES:
            top_k_history[k].append(get_top_k(unique_fitness, k))

        round_hits = _extract_hits(observations, fitness_values, bo_thresholds)
        all_hit_seqs.extend(round_hits)
        log.info(
            f"Round {round_idx}: {len(round_hits)} new hits, "
            f"{len(all_hit_seqs)} total accumulated hits."
        )

        if round_idx < bo_iters:
            template = design_template_fn(all_hit_seqs)
            log.info(f"Round {round_idx}: Redesigned IUPAC template: {template}")

    return new_hits_history, hits_history, top_k_history


def transform_thresholds_for_bo_landscape(
    thresholds: dict[str, float], landscape
) -> dict[str, float]:
    """
    Transform thresholds to landscape-specific hit thresholds.

    For MHCflurryLandscape, samples random sequences, scores them, and returns
    the -log10(affinity) value at the corresponding percentile for each objective.
    For landscapes using log10(1 + x) transformation in annotate_sequences,
    applies the same transformation to thresholds.
    For other landscapes, returns the thresholds unchanged.

    Args:
        thresholds: Dict mapping objective -> threshold.
        landscape: The landscape object to evaluate sequences on.

    Returns:
        Dict mapping objective -> transformed threshold.
    """
    if isinstance(landscape, MHCflurryLandscape):
        # Fix seed for deterministic threshold computation across runs
        np_rng_state = np.random.get_state()
        np.random.seed(42)

        n_samples = 10000
        random_sequences = generate_uniform_random_sequences(
            landscape.alphabet, landscape.str_length, n_samples
        )

        # Restore original random state
        np.random.set_state(np_rng_state)
        _, fitness_values = landscape.annotate_sequences(random_sequences)

        bo_thresholds = {}
        for obj_name, threshold in thresholds.items():
            percentile = threshold * 100
            bo_threshold = np.percentile(fitness_values[obj_name], percentile)
            log.info(
                f"[{obj_name}] threshold {threshold} -> {percentile}th percentile -> {bo_threshold:.4f}"
            )
            bo_thresholds[obj_name] = bo_threshold

        return bo_thresholds

    # Landscapes that use log10(1 + x) transformation in annotate_sequences
    log_transform_landscapes = (
        RiboswitchLandscape,
        GB1FitnessLandscape,
        PhoQFitnessLandscape,
        DHFRFitnessLandscape,
    )
    if isinstance(landscape, log_transform_landscapes):
        bo_thresholds = {}
        for obj_name, threshold in thresholds.items():
            bo_threshold = np.log10(1 + threshold)
            log.info(
                f"[{obj_name}] threshold {threshold} -> log10(1 + {threshold}) -> {bo_threshold:.4f}"
            )
            bo_thresholds[obj_name] = bo_threshold

        return bo_thresholds

    return thresholds.copy()


def update_threshold_based_on_observations(
    sequences: list[str],
    values: list[float],
    landscape,
    max_value: float,
    objective: str,
    percentile: float = 90,
) -> float:
    """
    Update threshold for a single objective based on oracle predictions.

    Args:
        sequences: Sequences to evaluate (peptides for peptide landscapes).
        values: True fitness values for these sequences.
        landscape: BoLandscape with trained oracle.
        max_value: Maximum allowed threshold.
        objective: Which objective to update threshold for.
        percentile: Percentile to use for threshold.
    """
    if len(sequences) == 0:
        log.info("No sequences to update on.")
        return max_value

    predictions = landscape.get_fitness_values(
        sequences,
        without_filtering=True,
        objective=objective,
    )
    valid_mask = predictions > float("-inf")
    predictions = predictions[valid_mask]
    target_percentile = np.percentile(predictions, percentile)
    target_percentile_values = np.percentile(values, percentile)
    log.info(
        f"[{objective}] Threshold update - predictions: {target_percentile:.4f}, real: {target_percentile_values:.4f}"
    )

    target_percentile = min(target_percentile, max_value)
    log.info(f"[{objective}] Threshold clamped to {target_percentile:.4f}")

    return target_percentile


def save_hits_csv(
    new_hits_history: List[int],
    hits_history: List[int],
    top_k_history: Dict[int, List[Optional[float]]],
    baselines_hits: Optional[Dict[str, Tuple[List[int], List[int]]]],
    baselines_top_k: Optional[Dict[str, Dict[int, List[Optional[float]]]]],
    output_dir: str,
) -> None:
    """
    Save hits and top-k fitness data to CSV in long format.

    Args:
        new_hits_history: Per-round unique hits for BO.
        hits_history: Per-round total hits for BO.
        top_k_history: Dict mapping k -> list of k-th best fitness per round for BO.
        baselines_hits: Dict mapping baseline name to (new_hits, hits) tuples.
        baselines_top_k: Dict mapping baseline name to top_k_history dicts.
        output_dir: Directory to save the CSV file.
    """
    csv_path = os.path.join(output_dir, "hits_data.csv")
    rows = []

    fieldnames = ["method", "iteration", "new_hits", "hits"]
    for k in TOP_K_VALUES:
        fieldnames.append(f"top_{k}")

    for i, (new_hits, hits) in enumerate(zip(new_hits_history, hits_history)):
        row = {"method": "BO", "iteration": i, "new_hits": new_hits, "hits": hits}
        for k in TOP_K_VALUES:
            row[f"top_{k}"] = top_k_history[k][i]
        rows.append(row)

    if baselines_hits:
        for name, (baseline_new_hits, baseline_hits) in baselines_hits.items():
            baseline_top_k = baselines_top_k.get(name, {}) if baselines_top_k else {}
            for i, (new_hits, hits) in enumerate(zip(baseline_new_hits, baseline_hits)):
                row = {
                    "method": name,
                    "iteration": i,
                    "new_hits": new_hits,
                    "hits": hits,
                }
                for k in TOP_K_VALUES:
                    row[f"top_{k}"] = baseline_top_k.get(k, [None] * (i + 1))[i]
                rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"Saved hits data to {csv_path}")


def run_baselines(
    bo_cfg,
    landscape,
    bo_thresholds: Dict[str, float],
    initial_observations: List[str],
    initial_fitness_values: Dict[str, List[float]],
    output_dir: str,
    sampled_dir: Optional[str] = None,
    plot_baselines: bool = False,
) -> Tuple[
    Dict[str, Tuple[List[int], List[int]]], Dict[str, Dict[int, List[Optional[float]]]]
]:
    """Run only the baselines without the full BO loop."""
    baselines_hits: Dict[str, Tuple[List[int], List[int]]] = {}
    baselines_top_k: Dict[str, Dict[int, List[Optional[float]]]] = {}

    run_random = bo_cfg.baseline in ("random", "all")
    run_rational = bo_cfg.baseline in ("rational", "all")

    if run_random:
        log.info("Running random baseline...")
        random_new_hits, random_hits, random_top_k = run_random_baseline(
            landscape,
            bo_thresholds,
            bo_cfg.bo_iters,
            bo_cfg.generate_per_round,
            initial_observations,
            initial_fitness_values,
            sampled_dir,
        )
        baselines_hits["Random"] = (random_new_hits, random_hits)
        baselines_top_k["Random"] = random_top_k
        log.info(
            f"Random baseline: {sum(random_new_hits)} unique hits, {sum(random_hits)} total hits."
        )

    if run_rational:
        log.info("Running rational design baseline...")
        rational_new_hits, rational_hits, rational_top_k = run_rational_design_baseline(
            landscape,
            bo_thresholds,
            bo_cfg.bo_iters,
            bo_cfg.generate_per_round,
            initial_observations,
            initial_fitness_values,
            sampled_dir,
        )
        baselines_hits["Rational"] = (rational_new_hits, rational_hits)
        baselines_top_k["Rational"] = rational_top_k
        log.info(
            f"Rational design baseline: {sum(rational_new_hits)} unique hits, {sum(rational_hits)} total hits."
        )

    if baselines_hits and plot_baselines:
        save_path = os.path.join(output_dir, "baselines_hits_plot.png")
        plot_hits([], [], save_path, baselines_hits)
        log.info(f"Saved baselines plot to {save_path}")

        empty_top_k: Dict[int, List[Optional[float]]] = {k: [] for k in TOP_K_VALUES}
        save_hits_csv([], [], empty_top_k, baselines_hits, baselines_top_k, output_dir)
    elif not baselines_hits:
        log.warning(
            "No baselines were run. Set bo.baseline to 'random', 'rational', or 'all'."
        )

    return baselines_hits, baselines_top_k


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("notes"):
        log.info("=== EXPERIMENT NOTES ===")
        log.info(cfg.notes)
        log.info("========================\n")

    log.info("--- Configuration ---")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("---------------------\n")

    log.info("--- Landscape Config ---")
    log.info(OmegaConf.to_yaml(cfg.landscape))
    log.info("------------------------\n")

    device = setup_device()

    # Set seeds for reproducibility
    setup_seeds(cfg.seed)

    objective_names = list(cfg.landscape.objectives.keys())
    invalid_scores = build_invalid_scores(cfg, objective_names)
    bo_invalid_scores = build_invalid_scores(cfg, objective_names, for_bo=True)

    # 1. Set up the true Landscape and BO landscape based on the config
    landscape = build_landscape(cfg, objective_names, device, invalid_scores)

    alphabet = landscape.alphabet
    alphabet_size = len(alphabet)

    thresholds = build_thresholds(cfg, objective_names, landscape)
    threshold = thresholds[objective_names[0]]

    log.info(f"Using threshold {threshold}")

    # 2. Set up the Policy (Agent) based on the config
    policy = build_policy(cfg, landscape, alphabet_size, device)

    bo_cfg = build_bo_config(cfg.bo)

    base_output_dir = HydraConfig.get().runtime.output_dir

    # Build structured training configuration
    training_cfg = build_training_config(cfg.training)
    # Set eval thresholds at runtime
    training_cfg.evaluation.eval_thresholds = thresholds
    training_cfg.evaluation.eval_threshold = (
        threshold if len(objective_names) == 1 else None
    )

    bo_thresholds = transform_thresholds_for_bo_landscape(thresholds, landscape)
    bo_threshold = bo_thresholds[objective_names[0]]

    bo_training_cfg = deepcopy(training_cfg)
    bo_training_cfg.evaluation.eval_thresholds = bo_thresholds
    bo_training_cfg.evaluation.eval_threshold = (
        bo_threshold if len(objective_names) == 1 else None
    )

    dna_sequences = get_starting_data_for_bo(landscape, bo_cfg.generate_per_round)
    new_observations, new_fitness_values = landscape.annotate_sequences(dna_sequences)
    # Save initial observations for rational design baseline
    initial_observations = new_observations.copy()
    initial_fitness_values = {k: v.copy() for k, v in new_fitness_values.items()}

    sampled_dir = None
    if bo_cfg.save_samples:
        sampled_dir = os.path.join(base_output_dir, "sampled")
        os.makedirs(sampled_dir, exist_ok=True)
        init_save_path = os.path.join(sampled_dir, "init.pkl")
        save_sampled_sequences(
            initial_observations, initial_fitness_values, init_save_path
        )

    log.info(f"First observation: {new_observations[0]}")
    log.info(
        f"First fitness values: {[new_fitness_values[obj][0] for obj in objective_names]}"
    )

    if bo_cfg.baselines_only:
        log.info("Running baselines only mode (skipping BO loop)...")
        baselines_sampled_dir = None
        if bo_cfg.save_samples:
            baselines_sampled_dir = os.path.join(base_output_dir, "sampled")
            os.makedirs(baselines_sampled_dir, exist_ok=True)
            init_save_path = os.path.join(baselines_sampled_dir, "init.pkl")
            save_sampled_sequences(
                initial_observations, initial_fitness_values, init_save_path
            )
        run_baselines(
            bo_cfg,
            landscape,
            bo_thresholds,
            initial_observations,
            initial_fitness_values,
            base_output_dir,
            baselines_sampled_dir,
            plot_baselines=True,
        )
        return

    num_ensembles = bo_cfg.oracle_training.n_ensembles

    if landscape.is_peptide:
        input_channels = 20  # amino acids
        input_dim = landscape.str_length // 3  # DNA to protein length
    else:
        input_channels = 4  # nucleotides
        input_dim = landscape.str_length  # DNA length

    ensembles = build_ensembles(
        objective_names, num_ensembles, input_dim, input_channels, device
    )

    bo_landscape = BoLandscape(
        str_length=cfg.landscape.str_length,
        objectives=objective_names,
        ensembles=ensembles,
        with_replacement=bo_cfg.with_replacement,
        device=device,
        min_peptide_len=cfg.landscape.get("min_peptide_len"),
        max_peptide_len=cfg.landscape.get("max_peptide_len"),
        is_peptide=landscape.is_peptide,
        invalid_scores=bo_invalid_scores,
        oracle_batch_size=bo_cfg.landscape_batch_size,
    )

    log.info(bo_landscape.active_objs)

    BBO = BBOAlgorithm(landscape, ensembles, objectives=objective_names)

    # Add 10000 NNK initialized sequences
    BBO.add_observations(new_observations, new_fitness_values)

    uniqueness_type = cfg.training.pooling.uniqueness
    reward_type = getattr(cfg.training, "reward_type", "hits")
    per_sequence_rewards = getattr(cfg.training, "per_sequence_rewards", False)

    if reward_type == "average" and landscape.is_peptide:
        for obj, score in bo_invalid_scores.items():
            if np.isinf(score) and score < 0:
                raise ValueError(
                    f"Objective '{obj}' has bo_invalid_score=-inf but reward_type='average'. "
                    "Set a finite bo_invalid_score (or invalid_score) in the landscape objectives config."
                )

    # Set up the reward function
    reward_function = get_reward_function(
        bo_landscape,
        objective_names,
        bo_thresholds,
        bo_threshold if len(objective_names) == 1 else None,
        cfg.training.sampling.mc_samples,
        cfg.training.sampling.batch_size,
        reward_type,
        cfg.training.reward_aggregation_unit,
        per_sequence_rewards,
        uniqueness_type=uniqueness_type,
    )

    # Set up entropy function if lambda > 0
    if training_cfg.entropy.lambda_ > 0:
        entropy_function = get_entropy_function(
            training_cfg.entropy.type,
            training_cfg.entropy.sequence_type,
            policy,
            training_cfg.entropy.baseline_per_template,
        )
    else:
        entropy_function = None

    extra_eval = True

    sampled_sequences = []
    new_hits, hits = calculate_new_hits(
        sampled_sequences,
        new_observations,
        new_fitness_values,
        bo_thresholds,
    )
    sampled_sequences.extend(new_observations)

    hits_history = [hits]
    new_hits_history = [new_hits]
    r_squared_history = []
    random_val_r2_history: List[float] = []

    # Generate random held-out validation set (unbiased w.r.t. data gathering)
    n_random_val = 2000
    log.info(f"Generating {n_random_val} random sequences for held-out validation...")
    if landscape.is_peptide:
        random_val_dna = generate_nnk_sequences(landscape.str_length, n_random_val)
    else:
        random_val_dna = generate_uniform_random_sequences(
            landscape.alphabet, landscape.str_length, n_random_val
        )
    random_val_seqs, random_val_fitness = landscape.annotate_sequences(random_val_dna)
    log.info(f"Random held-out set: {len(random_val_seqs)} sequences")

    # Create dataloaders for random validation (one per objective)
    random_val_loaders = {}
    for obj_name in objective_names:
        loader, _ = create_dataloaders(
            landscape,
            random_val_seqs,
            random_val_fitness[obj_name],
            mode="test",
            batch_size=64,
        )
        random_val_loaders[obj_name] = loader

    first_obj = objective_names[0]
    unique_fitness: Dict[str, float] = {}
    top_k_history: Dict[int, List[Optional[float]]] = {k: [] for k in TOP_K_VALUES}

    for seq, fit in zip(new_observations, new_fitness_values[first_obj]):
        unique_fitness[seq] = max(unique_fitness.get(seq, float("-inf")), fit)
    for k in TOP_K_VALUES:
        top_k_history[k].append(get_top_k(unique_fitness, k))

    for i in range(bo_cfg.bo_iters):
        log.info(f"=== BO Iteration {i} ===")
        log.info(f"Hit thresholds (fixed): {bo_thresholds}")
        first_ensemble = ensembles[objective_names[0]]
        log.info(f"Ensemble is on: {next(first_ensemble[0].parameters()).device}")
        hits = 0
        new_hits = 0
        current_output_dir = os.path.join(base_output_dir, f"iter_{i}")
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir)
        params_dir = os.path.join(current_output_dir, "oracle_params")
        if not os.path.exists(params_dir):
            os.makedirs(params_dir)

        # 1. training the oracle with the newly observed data
        oracle_training_results = BBO.train_all_ensembles(
            params_dir,
            device=device,
            n_epochs=bo_cfg.oracle_training.n_epochs,
            lr=bo_cfg.oracle_training.lr,
            patience=bo_cfg.oracle_training.patience,
        )

        # visualizing oracle training loss (one plot per objective, first model only)
        vis_dir = os.path.join(current_output_dir, cfg.vis.vis_dir)
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
            log.info(f"\nSaving visualizations to: {vis_dir}")
        for obj_name, training_result in oracle_training_results.items():
            save_path = os.path.join(vis_dir, f"oracle_training_history_{obj_name}.png")
            plot_training_history_oracle(
                training_result["history"], save_path=save_path
            )
            save_path = os.path.join(vis_dir, f"oracle_pred_vs_true_val_{obj_name}.png")
            mse, r2, spearman = plot_pred_vs_true_oracle(
                training_result["validation"]["targets"],
                training_result["validation"]["outputs"],
                save_path=save_path,
            )
            log.info(
                f"Objective {obj_name}: MSE={mse:.4f}, R²={r2:.4f}, Spearman={spearman:.4f} on validation."
            )
        r_squared_history.append(r2)
        log.info(f"Oracle training for iteration {i} complete!")

        # Evaluate on random held-out set (unbiased validation)
        for obj_name in objective_names:
            random_val_metrics = evaluate_ensemble_on_held_out(
                ensembles[obj_name],
                random_val_loaders[obj_name],
                device=device,
            )
            log.info(
                f"Objective {obj_name} (random val): "
                f"MSE={random_val_metrics['mse']:.4f}, "
                f"R²={random_val_metrics['r2']:.4f}, "
                f"Spearman={random_val_metrics['spearman']:.4f}"
            )
        random_val_r2_history.append(random_val_metrics["r2"])

        # Update thresholds for all objectives based on oracle predictions
        if bo_cfg.adaptive_threshold:
            bo_thresholds_new = {
                obj_name: update_threshold_based_on_observations(
                    new_observations,
                    new_fitness_values[obj_name],
                    bo_landscape,
                    bo_thresholds[obj_name],
                    objective=obj_name,
                )
                for obj_name in objective_names
            }
            if hasattr(reward_function, "thresholds"):
                reward_function.thresholds = bo_thresholds_new
            elif hasattr(reward_function, "threshold"):
                reward_function.threshold = bo_thresholds_new[objective_names[0]]
            log.info(f"Updated thresholds for scoring: {bo_thresholds_new}")

        bo_landscape.update_observed_sequences(new_observations)

        # 2. Train policy with the updated reward function and oracle
        log.info(f"Started policy training for iteration {i}")

        if training_cfg.pooling.enabled:
            log.info("Using pooled training with landscape query subsampling")
            reward_history, entropy_objective_history = pooled_train_policy(
                policy,
                bo_landscape,
                bo_training_cfg,
                base_output_dir,
                reward_function,
                entropy_function,
                device,
            )
        else:
            reward_history, entropy_objective_history = train_policy(
                policy,
                bo_landscape,
                bo_training_cfg,
                base_output_dir,
                reward_function,
                device,
            )

        log.info(f"Policy training finished for iteration {i}")
        log.info(f"Saving visualizations to {vis_dir}.")
        # Plot training progress
        save_path = os.path.join(vis_dir, f"policy_reward_history.png")
        plot_reward_history(reward_history, policy.name, save_path=save_path)

        # Plot policy-specific parameter visualizations
        if policy.name == "Arbitrary":
            save_path = os.path.join(vis_dir, "arbitrary_policy_params.png")
            plot_arbitrary_optimizer_results(policy, alphabet, save_path)
        elif isinstance(policy, BottleBasedOptimizer):
            # Works for both Finite and IUPAC (and any future bottle-based policies)
            save_path = os.path.join(
                vis_dir, f"{policy.name.lower()}_policy_params.png"
            )
            plot_finite_optimizer_results(policy, alphabet, save_path)

        # 3. Generate new data from the policy and add it to observed values
        dna_sequences = create_data_from_policy(
            policy, landscape, bo_cfg.generate_per_round
        )
        new_observations, new_fitness_values = landscape.annotate_sequences(
            dna_sequences
        )
        log.info(f"Generated {len(new_observations)} new sequences from the policy.")

        if sampled_dir:
            bo_save_path = os.path.join(sampled_dir, f"bo_{i + 1}.pkl")
            save_sampled_sequences(new_observations, new_fitness_values, bo_save_path)

        # Calculate new hits using AND logic across all objectives
        new_hits, hits = calculate_new_hits(
            sampled_sequences,
            new_observations,
            new_fitness_values,
            bo_thresholds,
        )

        sampled_sequences.extend(new_observations)
        log.info(f"We have observed {new_hits} new hits.")
        log.info(
            f"We have observed {hits} total hits."
        )  # Note: total means not observed before, still it is deduplicated
        hits_history.append(hits)
        new_hits_history.append(new_hits)

        for seq, fit in zip(new_observations, new_fitness_values[first_obj]):
            unique_fitness[seq] = max(unique_fitness.get(seq, float("-inf")), fit)
        for k in TOP_K_VALUES:
            top_k_history[k].append(get_top_k(unique_fitness, k))

        # Update BBO to include the newly generated data
        BBO.add_observations(new_observations, new_fitness_values)

        if extra_eval == True:
            log.info(
                "Evaluating the policy on the actual landscape with the true oracle."
            )
            run_extra_eval(policy, landscape, training_cfg, current_output_dir, i)

        policy_params_save_path = os.path.join(
            current_output_dir, "policy_parameters.pth"
        )
        log.info(f"Saving policy parameters to: {policy_params_save_path}")
        torch.save(policy.state_dict(), policy_params_save_path)

        # Print oligo ordering template
        oligo_template_path = os.path.join(vis_dir, "oligo_ordering_template.txt")
        print_oligo_ordering_template(policy, log_to_file=oligo_template_path)

        # Print theoretical diversity (per-template)
        diversity_path = os.path.join(vis_dir, "theoretical_diversity.txt")
        print_theoretical_diversity(
            policy,
            landscape.alphabet,
            is_peptide_landscape=landscape.is_peptide,
            log_to_file=diversity_path,
        )

        # Print total library diversity (union of all templates)
        print_total_library_diversity(
            policy,
            landscape.alphabet,
            is_peptide_landscape=landscape.is_peptide,
            log_to_file=diversity_path,
        )

        log.info("Rebuilding the ensembles")
        ensembles = build_ensembles(
            objective_names, num_ensembles, input_dim, input_channels, device
        )
        BBO.ensembles = ensembles
        bo_landscape.ensembles = ensembles

        if len(landscape.active_objs) == 1:
            save_path = os.path.join(vis_dir, "sampled_scores_histogram.png")
            plot_sampled_scores_histogram(
                policy, landscape, num_sequences=1000, save_path=save_path
            )

    total_unique_hits = sum(new_hits_history)
    total_all_hits = sum(hits_history)
    log.info(
        f"In all {bo_cfg.bo_iters} iterations, we have observed {total_unique_hits} new hits and {total_all_hits} all hits."
    )

    baselines_hits, baselines_top_k = run_baselines(
        bo_cfg,
        landscape,
        bo_thresholds,
        initial_observations,
        initial_fitness_values,
        base_output_dir,
        sampled_dir,
    )

    save_path = os.path.join(base_output_dir, "hits_plot.png")
    plot_hits(
        new_hits_history,
        hits_history,
        save_path,
        baselines_hits if baselines_hits else None,
    )

    save_path = os.path.join(base_output_dir, "top_k_fitness_plot.png")
    plot_top_k_fitness(
        top_k_history, save_path, baselines_top_k if baselines_top_k else None
    )

    save_hits_csv(
        new_hits_history,
        hits_history,
        top_k_history,
        baselines_hits,
        baselines_top_k,
        base_output_dir,
    )

    save_path = os.path.join(base_output_dir, "r_2_observed_val.png")
    plot_r_squared(r_squared_history, save_path)

    save_path = os.path.join(base_output_dir, "r_2_random_val.png")
    plot_r_squared(random_val_r2_history, save_path)


if __name__ == "__main__":
    main()
