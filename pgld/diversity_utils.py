"""Shared utilities for diversity computation."""

import os
from typing import Optional

import numpy as np
import torch

from landscapes import sequences_to_strings_vectorized
from policies import ArbitraryBaseOptimizer, IupacBaseOptimizer
from setup import build_invalid_scores, build_landscape, build_policy


def compute_mask_diversity(mask: np.ndarray) -> float:
    """
    Computes diversity of a single boolean mask.

    Args:
        mask: Array of shape (length, alphabet_size) where values > 0 indicate allowed

    Returns:
        Diversity as product of allowed chars per position (0 if any position is empty)
    """
    counts_per_pos = np.sum(mask > 0, axis=1)
    if np.any(counts_per_pos == 0):
        return 0.0
    return float(np.prod(counts_per_pos))


def compute_diversity_with_threshold(
    run_dir: str,
    cfg,
    device: torch.device,
    threshold: float,
) -> Optional[int]:
    """
    Load policy and compute theoretical diversity with custom threshold.

    Args:
        run_dir: Path to run directory containing parameters.pth
        cfg: Loaded OmegaConf config
        device: Torch device
        threshold: Probability threshold for diversity computation

    Returns:
        Union diversity, or None if loading fails
    """
    # Deferred: library_size_estimator imports diversity_utils (circular).
    from library_size_estimator import compute_total_library_diversity

    param_path = os.path.join(run_dir, "parameters.pth")
    if not os.path.exists(param_path):
        return None

    try:
        # Build policy directly (don't need landscape for diversity)
        str_length = cfg.landscape.str_length
        alphabet_size = 4  # DNA alphabet
        batch_size = 32  # Doesn't matter for diversity computation

        if cfg.policy.name == "arbitrary":
            policy = ArbitraryBaseOptimizer(
                str_length=str_length,
                alphabet_size=alphabet_size,
                batch_size=batch_size,
                num_templates=cfg.policy.num_templates,
                device=device,
                uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
            )
        elif cfg.policy.name == "iupac":
            policy = IupacBaseOptimizer(
                str_length=str_length,
                alphabet_size=alphabet_size,
                batch_size=batch_size,
                num_templates=cfg.policy.num_templates,
                device=device,
                uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
            )
        else:
            print(f"  Unsupported policy type: {cfg.policy.name}")
            return None

        policy.load_state_dict(
            torch.load(param_path, map_location=device, weights_only=True)
        )

        # For peptide landscapes, sum diversity over the admitted lengths (8-9)
        # rather than full-length only.
        is_peptide = cfg.landscape.get("seq_scored", "dna") == "peptide"
        peptide_lengths = None
        if is_peptide:
            min_len = cfg.landscape.get("min_peptide_len")
            max_len = cfg.landscape.get("max_peptide_len")
            if min_len is not None and max_len is not None:
                num_codons = str_length // 3
                peptide_lengths = list(
                    range(int(min_len), min(int(max_len), num_codons) + 1)
                )
        result = compute_total_library_diversity(
            policy,
            is_peptide_landscape=is_peptide,
            threshold=threshold,
            peptide_lengths=peptide_lengths,
        )

        # Return union diversity if available, else sum
        if result["union_diversity"] is not None:
            return int(result["union_diversity"])
        return int(result["sum_diversity"])

    except Exception as e:
        print(f"  Error computing diversity for {run_dir}: {e}")
        return None


def greedy_mean_fitness(
    run_dir: str,
    cfg,
    device: torch.device,
    n_samples: int = 10000,
) -> Optional[float]:
    """
    Greedy-sample the saved policy (the orderable library) and return its mean
    fitness under the landscape. Returns None if loading fails.

    Args:
        run_dir: Run directory containing parameters.pth and .hydra/config.yaml
        cfg: Loaded OmegaConf config for the run
        device: Torch device
        n_samples: Number of sequences to sample from the hardened library

    Returns:
        Mean fitness averaged over objectives and samples, or None on failure.
    """
    param_path = os.path.join(run_dir, "parameters.pth")
    if not os.path.exists(param_path):
        return None

    try:
        objective_names = list(cfg.landscape.objectives.keys())
        invalid_scores = build_invalid_scores(cfg, objective_names)
        landscape = build_landscape(cfg, objective_names, device, invalid_scores)

        policy = build_policy(cfg, landscape, len(landscape.alphabet), device)
        policy.load_state_dict(
            torch.load(param_path, map_location=device, weights_only=True)
        )

        sample_result = policy.sample(n_samples, greedy=True)
        seq_strings = sequences_to_strings_vectorized(
            sample_result.sequences, landscape.alphabet
        )

        obj_scores = [
            np.asarray(landscape.get_fitness_values(seq_strings, objective=obj))
            for obj in objective_names
        ]  # (n_objectives, n_samples)
        return float(np.mean(np.mean(obj_scores, axis=0)))

    except Exception as e:
        print(f"  Error computing greedy mean fitness for {run_dir}: {e}")
        return None
