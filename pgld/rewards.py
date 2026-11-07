"""
rewards.py

Defines rewards for policy training.
Includes rewards for BO settings and standard settings.
"""

from abc import ABC
from collections import Counter
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from landscapes import is_landscape_multi_objective

from landscapes import (
    MHCflurryLandscape,
    TFBindingLandscape,
    avg_unique_weighted,
    avg_weighted,
    dna_to_protein,
    hit_above_threshold_individual,
    hits_all_objectives,
    sequences_to_valid_peptides,
    filter_same_protein_encoding,
)

log = logging.getLogger(__name__)


class RewardFunction(ABC):
    """Base class for reward functions."""

    def __call__(self, sequences, landscape):
        raise NotImplementedError


class IndividualHitsReward(RewardFunction):
    """Returns a torch of the same shape as number of sequences with hit values for each sequence"""

    def __init__(self, threshold=0.995, uniqueness_type="dna"):
        self.threshold = threshold
        self.uniqueness_type = uniqueness_type

    def __call__(self, sequences, landscape):
        return hit_above_threshold_individual(landscape, sequences, self.threshold)


class AllModelsHitReward(RewardFunction):
    """
    Counts sequences as hits only if ALL objectives pass their thresholds (logical AND).
    Only counts unique sequences.
    For template-level aggregation, uses fractional credit assignment.
    """

    def __init__(self, thresholds, uniqueness_type="dna"):
        self.thresholds = thresholds
        self.uniqueness_type = uniqueness_type

    def __call__(self, sequences, landscape):
        if not sequences:
            return 0.0
        unique_hits, _ = hits_all_objectives(
            landscape, sequences, self.thresholds, self.uniqueness_type
        )
        return float(unique_hits) / len(sequences)

    def _is_hit(self, fitness_value, landscape) -> bool:
        """Check if a fitness value passes all thresholds."""
        if is_landscape_multi_objective(landscape):
            for i, obj_name in enumerate(landscape.active_objs):
                threshold = self.thresholds[obj_name]
                if fitness_value[i] <= threshold:
                    return False
            return True
        else:
            obj_name = landscape.active_objs[0]
            threshold = self.thresholds[obj_name]
            return fitness_value > threshold

    def compute_with_cache(self, batch_info: dict):
        """Compute reward using pooled cache with IS weighting."""
        keys_scored = batch_info["keys_scored"]
        weight = batch_info["importance_weight"]
        batch_size = batch_info["batch_size"]
        key_to_fitness = batch_info["key_to_fitness"]
        landscape = batch_info["landscape"]

        # Batch-level: single unit
        if isinstance(keys_scored, list) and len(keys_scored) == 1:
            keys_scored = keys_scored[0]
            weight = weight[0]

            if not keys_scored:
                return 0.0

            all_pass_count = sum(
                1
                for key in keys_scored
                if key_to_fitness.get(key) is not None
                and self._is_hit(key_to_fitness[key], landscape)
            )
            return (all_pass_count * weight) / batch_size

        # Template-level: fractional credit assignment
        unit_sizes = batch_info["unit_sizes"]
        reward_unit = batch_info["reward_unit"]
        key_counts = batch_info["key_counts"]
        n_reward_units = len(keys_scored)
        reward_per_unit = torch.zeros((n_reward_units,), dtype=torch.float32)

        # Step 1: Find all unique hits and which templates HAVE them
        hit_to_templates = {}
        for unit_idx, key_counts_unit in enumerate(key_counts):
            for key in key_counts_unit.keys():
                fitness_value = key_to_fitness.get(key)
                if fitness_value is None:
                    continue
                if self._is_hit(fitness_value, landscape):
                    hit_to_templates.setdefault(key, []).append(unit_idx)

        # Step 2: Split credit among templates that HAVE the hit, give to those that SAMPLED it
        for key, template_indices in hit_to_templates.items():
            fractional_credit = 1.0 / len(template_indices)
            for unit_idx in template_indices:
                if key in keys_scored[unit_idx]:
                    reward_per_unit[unit_idx] += fractional_credit

        # Step 3: Apply importance weights and normalize by batch_size
        for unit_idx in range(n_reward_units):
            reward_per_unit[unit_idx] *= weight[unit_idx] / batch_size

        return reward_per_unit[reward_unit]


class AverageMultiReward(RewardFunction):
    """
    Weighted sum across objectives, averaged over batch sequences.
    Returns single scalar reward.
    If unique=True: deduplicates sequences but normalizes by original batch size.
    If unique=False: averages over all sequences in batch.
    """

    def __init__(self, weights, unique: bool = True):
        self.weights = weights
        self.unique = unique

    def __call__(self, sequences, landscape):
        if self.unique:
            return avg_unique_weighted(landscape, sequences, self.weights)
        else:
            return avg_weighted(landscape, sequences, self.weights)

    def compute_with_cache(self, batch_info: dict) -> float:
        """Compute reward using pooled cache with IS weighting."""
        keys_scored = batch_info["keys_scored"]
        weight = batch_info["importance_weight"]
        batch_size = batch_info["batch_size"]
        landscape = batch_info["landscape"]
        key_counts = batch_info["key_counts"]
        key_to_fitness = batch_info["key_to_fitness"]

        # Handle nested structure from template-level cache (batch-level only)
        if isinstance(keys_scored, list) and len(keys_scored) == 1:
            keys_scored = keys_scored[0]
            weight = weight[0]
            key_counts = key_counts[0]
        elif isinstance(keys_scored, list) and len(keys_scored) > 1:
            raise ValueError(
                f"AverageMultiReward only supports batch-level aggregation, "
                f"but got {len(keys_scored)} units. Use AverageFitnessPerUnitReward for template-level."
            )

        weighted_sum = 0.0
        for key in keys_scored:
            fitness_data = key_to_fitness.get(key)
            if fitness_data is None:
                continue

            if is_landscape_multi_objective(landscape):
                weighted_value = np.sum(fitness_data * self.weights)
            else:
                weighted_value = fitness_data * self.weights[0]

            if self.unique:
                weighted_sum += weighted_value
            else:
                weighted_sum += weighted_value * key_counts[key]

        return (weighted_sum * weight) / batch_size


class AverageFitnessPerUnitReward(RewardFunction):
    """
    Weighted sum across objectives, averaged across each reward unit (e.g. template).
    Equivalent to AverageMultiReward if reward units are batches.
    """

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, sequences, landscape):
        # not implemented for now as we would
        # have to pass additional info about reward units
        raise NotImplementedError

    def compute_with_cache(self, batch_info: dict) -> torch.Tensor:
        """Compute reward using pooled cache with IS weighting."""
        keys_scored = batch_info["keys_scored"]
        key_counts = batch_info["key_counts"]
        weight = batch_info["importance_weight"]
        unit_sizes = batch_info["unit_sizes"]
        batch_size = batch_info["batch_size"]
        landscape = batch_info["landscape"]
        key_to_fitness = batch_info["key_to_fitness"]
        reward_unit = batch_info["reward_unit"]

        n_reward_units = len(keys_scored)
        reward_per_unit = torch.zeros((n_reward_units,), dtype=torch.float32)

        for unit, (
            keys_scored_unit,
            key_counts_unit,
            weight_unit,
            unit_size,
        ) in enumerate(zip(keys_scored, key_counts, weight, unit_sizes)):
            for key in keys_scored_unit:
                fitness_data = key_to_fitness.get(key)
                if fitness_data is None:
                    continue

                if is_landscape_multi_objective(landscape):
                    weighted_value = np.sum(fitness_data * self.weights)
                else:
                    weighted_value = fitness_data * self.weights[0]

                reward_per_unit[unit] += weighted_value * key_counts_unit[key]
            reward_per_unit[unit] *= weight_unit / batch_size

        return reward_per_unit[reward_unit]


class AllModelsHitPerSequenceReward(RewardFunction):
    """
    Per-sequence reward for multi-objective AND hits with variance reduction.
    Returns a tensor of rewards where each sequence gets:
    - 0 if it doesn't pass ALL thresholds
    - 1/count if it passes ALL thresholds (where count = number of times this sequence appears in batch)

    This is the per-sequence version of AllModelsHitReward.
    Only queries landscape for unique sequences for efficiency.
    """

    def __init__(self, thresholds):
        self.thresholds = thresholds

    def __call__(self, sequences, landscape):
        if not sequences:
            return torch.zeros(0)

        seq_counts = Counter(sequences)
        unique_seqs = list(seq_counts.keys())

        unique_all_objectives = landscape.get_all_objectives(unique_seqs)

        unique_all_pass = np.ones(len(unique_seqs), dtype=bool)
        for i, obj_name in enumerate(landscape.active_objs):
            threshold = self.thresholds[obj_name]
            unique_all_pass = unique_all_pass & (
                unique_all_objectives[:, i] > threshold
            )

        seq_to_hit = {seq: passes for seq, passes in zip(unique_seqs, unique_all_pass)}

        rewards = torch.zeros(len(sequences), dtype=torch.float32)
        for i, seq in enumerate(sequences):
            if seq_to_hit[seq]:
                rewards[i] = 1.0 / seq_counts[seq]

        return rewards


def get_reward_function(
    landscape,
    objective_names,
    thresholds,
    threshold,
    mc_samples,
    batch_size,
    reward_type="hits",
    reward_aggregation_unit="batch",
    per_sequence=False,
    uniqueness_type="dna",
):
    """
    Factory function to select and instantiate reward function.

    Args:
        landscape: Landscape object
        objective_names: List of objective names
        thresholds: Dict of thresholds (works for both single and multi-objective)
        threshold: Single threshold value (legacy, only used for mc_samples>1 edge case)
        mc_samples: Number of MC samples
        batch_size: Batch size
        reward_type: Type of reward ("hits" or "average")
        reward_aggregation_unit: "batch" or "template"
        per_sequence: If True, use per-sequence rewards for variance reduction

    Returns:
        Instantiated reward function
    """
    # Special case: MC sampling with batch_size=1 uses per-sequence rewards
    if mc_samples > 1 and batch_size == 1:
        reward_function = IndividualHitsReward(
            threshold, uniqueness_type=uniqueness_type
        )
        log.info("Using IndividualHitsReward (mc_samples > 1, batch_size = 1)")
        return reward_function

    # General case: Works for both single and multi-objective
    if reward_type == "average":
        # Use landscape.active_objs ordering (may differ from config ordering)
        weights = [1.0 for _ in landscape.active_objs]
        if reward_aggregation_unit == "batch":
            reward_function = AverageMultiReward(weights=weights, unique=False)
            log.info(
                f"Using AverageMultiReward with equal weights for {landscape.active_objs}"
            )
        else:
            reward_function = AverageFitnessPerUnitReward(weights=weights)
            log.info(
                f"Using AverageFitnessPerUnitReward with equal weights for {landscape.active_objs}"
            )
    elif reward_type == "hits":
        if per_sequence:
            reward_function = AllModelsHitPerSequenceReward(thresholds)
            log.info(
                f"Using AllModelsHitPerSequenceReward (per-sequence variance reduction) for {objective_names}"
            )
        else:
            reward_function = AllModelsHitReward(
                thresholds, uniqueness_type=uniqueness_type
            )
            log.info(f"Using AllModelsHitReward (logical AND) for {objective_names}")
    else:
        raise ValueError(
            f"Invalid reward_type '{reward_type}'. Use 'hits' or 'average'."
        )

    return reward_function
