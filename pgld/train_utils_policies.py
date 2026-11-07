"""
train_utils_policies.py

Utility functions for policy training: checkpointing, logging, and evaluation.
"""

import csv
import logging
import os
import random
from collections import Counter
from typing import Optional, Union

import numpy as np
import torch

from config_schemas import EntropyConfig, EvaluationConfig, TrainingConfig
from constants import AMINO_ACID_ALPHABET
from landscapes import (
    dna_to_protein,
    dna_to_protein_batch,
    filter_same_protein_encoding,
    hits_all_objectives,
    is_landscape_multi_objective,
    sequences_to_strings_vectorized,
)
from tensor_translation import tensor_to_protein_batch
from tensor_translation_gpu import (
    build_codon_lookup_gpu,
    translate_dna_to_aa_indices_gpu,
    dna_probs_to_aa_probs,
)
from train_utils import compute_peptide_length_stats, compute_edit_distance_stats
from entropy import (
    calculate_entropy,
    calculate_entropy_exact_bottle_based,
    calculate_entropy_gradient_estimate,
    calculate_entropy_mc_bottle_based,
    calculate_entropy_per_template,
    calculate_marginal_entropy_gain,
    calculate_peptide_entropy,
    calculate_peptide_entropy_gradient_estimate,
)
from policies import ArbitraryBaseOptimizer, BottleBasedOptimizer

log = logging.getLogger(__name__)

NATS_TO_BITS = 1.0 / np.log(2)


def _compute_policy_entropy_bits(policy, training_cfg: TrainingConfig) -> float:
    """Compute policy entropy in bits. Returns nan if not computable (e.g., multi-template)."""
    if policy.num_templates != 1:
        log.warning("Entropy computation not supported for num_templates != 1")
        return float("nan")

    peptide_level = training_cfg.entropy.sequence_type == "peptide"

    with torch.no_grad():
        if isinstance(policy, ArbitraryBaseOptimizer):
            if peptide_level:
                entropy_nats = calculate_peptide_entropy(policy).item()
            else:
                entropy_nats = calculate_entropy(policy).item()
        elif isinstance(policy, BottleBasedOptimizer):
            sample_result = policy.sample(batch_size=1)
            _, entropy_nats = calculate_entropy_exact_bottle_based(
                policy, sample_result, peptide_level=peptide_level
            )
        else:
            log.warning(
                f"Entropy computation not supported for policy type {type(policy)}"
            )
            return float("nan")

    entropy_bits = entropy_nats * NATS_TO_BITS
    log.info(f"Entropy: {entropy_bits:.4f} bits")
    return entropy_bits


def log_metrics_csv(save_dir: str, iteration: int, metrics: dict) -> None:
    """
    Log multiple metrics to a CSV file.

    Args:
        save_dir: Path to the directory where logs are stored.
        iteration: Current training iteration.
        metrics: Dictionary of metric names to values.
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "metrics.csv")

    row = {"iteration": iteration, **metrics}
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(policy, output_dir: str, iteration: int):
    """Save policy checkpoint to disk."""
    log.info(f"Saving model parameters for iteration {iteration}")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    params_path = os.path.join(checkpoint_dir, f"parameters_{iteration}.pth")
    log.info(f"Saving policy parameters to: {params_path}")
    torch.save(policy.state_dict(), params_path)


def log_peptide_stats(policy, landscape, alphabet: list[str]):
    """Log peptide length and edit distance statistics."""
    eval_sample = policy.sample(1000, greedy=True)
    eval_sequences_str = sequences_to_strings_vectorized(
        eval_sample.sequences, alphabet
    )

    stats = compute_peptide_length_stats(eval_sequences_str)
    log.info(
        f"  Peptide lengths - min: {stats['peptide_len_p0']}, "
        f"p25: {stats['peptide_len_p25']}, median: {stats['peptide_len_p50']}, "
        f"p75: {stats['peptide_len_p75']}, max: {stats['peptide_len_p100']}"
    )

    if hasattr(landscape, "wild_type") and landscape.wild_type is not None:
        edit_stats = compute_edit_distance_stats(
            eval_sequences_str, landscape.wild_type
        )
        log.info(
            f"  Edit distances from WT - min: {edit_stats['edit_dist_p0']}, "
            f"p25: {edit_stats['edit_dist_p25']}, median: {edit_stats['edit_dist_p50']}, "
            f"p75: {edit_stats['edit_dist_p75']}, max: {edit_stats['edit_dist_p100']}"
        )


def _evaluate_multi_objective(
    policy, landscape, thresholds: Optional[dict], batch_size: int
) -> tuple[int, int]:
    """
    Evaluate multi-objective hits where a sequence is a hit only if ALL objectives pass.

    Returns:
        (unique_hits, all_hits)
    """
    sample_result = policy.sample(batch_size)
    sequences_str = sequences_to_strings_vectorized(
        sample_result.sequences, landscape.alphabet
    )
    uniqueness_type = "dna"
    if landscape.is_peptide:
        uniqueness_type = "peptide"
    return hits_all_objectives(
        landscape, sequences_str, thresholds, uniqueness_type=uniqueness_type
    )


def evaluate_with_frequencies(
    policy,
    landscape,
    threshold: Optional[float],
    thresholds: Optional[dict],
    batch_size: int,
) -> tuple[int, int, Counter, float]:
    """
    Evaluate hits and return frequency counts for SGT estimation.

    Args:
        policy: Policy to sample from.
        landscape: Landscape for fitness evaluation.
        threshold: Single-objective threshold (or None for multi-objective).
        thresholds: Multi-objective thresholds dict (or None for single-objective).
        batch_size: Number of sequences to sample.

    Returns:
        (unique_hits, all_hits, hit_counts, mean_fitness) where hit_counts maps
        each unique hit sequence -> occurrence count in sample.
    """
    sample_result = policy.sample(batch_size)
    sequences_str = sequences_to_strings_vectorized(
        sample_result.sequences, landscape.alphabet
    )
    return count_hits_from_sequences(landscape, sequences_str, threshold, thresholds)


def count_hits_from_sequences(
    landscape,
    sequences_str: list,
    threshold: Optional[float],
    thresholds: Optional[dict],
) -> tuple[int, int, Counter, float]:
    """
    Count hits for an explicit set of sampled DNA sequences, scoring each unique
    sequence only once.

    Shared by the in-training soft eval (``evaluate_with_frequencies``) and the
    standalone greedy library eval, so both use identical hit semantics. Peptide
    landscapes count unique hits at the peptide (protein) level.

    Returns (unique_hits, all_hits, hit_counts, mean_fitness) where hit_counts maps
    each unique hit sequence -> occurrence count in the sample.
    """
    is_multi_objective = is_landscape_multi_objective(landscape)
    is_peptide = landscape.is_peptide
    unique_xs = list(set(sequences_str))

    if is_multi_objective:
        all_objectives = landscape.get_all_objectives(unique_xs)
        all_pass = np.ones(len(unique_xs), dtype=bool)
        for i, obj_name in enumerate(landscape.active_objs):
            obj_threshold = thresholds[obj_name]
            all_pass = all_pass & (all_objectives[:, i] > obj_threshold)
        fitness_values = all_objectives[:, 0]  # Use first objective for mean
    else:
        fitness_values = landscape.get_fitness_values(unique_xs)
        if isinstance(fitness_values, torch.Tensor):
            fitness_values = fitness_values.cpu().numpy()
        all_pass = np.array(fitness_values) > threshold

    seq_to_hit = {seq: passes for seq, passes in zip(unique_xs, all_pass)}

    # For peptide landscapes, count unique peptides rather than DNA sequences
    if is_peptide:
        dna_to_peptide = dna_to_protein_batch(sequences_str)
        hit_counts: Counter = Counter()
        for dna_seq in sequences_str:
            if seq_to_hit.get(dna_seq, False):
                peptide = dna_to_peptide[dna_seq]
                hit_counts[peptide] += 1

        hit_dna_seqs = [seq for seq, passes in seq_to_hit.items() if passes]
        unique_hits = len(filter_same_protein_encoding(hit_dna_seqs))
    else:
        hit_counts = Counter()
        for seq in sequences_str:
            if seq_to_hit.get(seq, False):
                hit_counts[seq] += 1
        unique_hits = sum(1 for passes in seq_to_hit.values() if passes)

    all_hits = sum(1 for seq in sequences_str if seq_to_hit.get(seq, False))

    # Compute mean fitness over all sampled sequences (not just unique)
    seq_to_fitness = {seq: fit for seq, fit in zip(unique_xs, np.array(fitness_values))}
    mean_fitness = float(np.mean([seq_to_fitness[seq] for seq in sequences_str]))

    return unique_hits, all_hits, hit_counts, mean_fitness


def run_extra_eval(
    policy,
    landscape,
    training_cfg: TrainingConfig,
    output_dir: str,
    iteration: int,
):
    """Run extra evaluation on a large batch and log threshold-based hit metrics."""
    eval_cfg = training_cfg.evaluation

    is_multi_objective = eval_cfg.eval_thresholds is not None
    has_valid_threshold = eval_cfg.eval_threshold is not None or is_multi_objective

    if is_landscape_multi_objective(landscape) and not is_multi_objective:
        log.warning(
            "Landscape has multiple objectives but eval_thresholds not set. "
            "Only evaluating single objective."
        )

    if not has_valid_threshold:
        log.info("No evaluation threshold specified, skipping threshold-based eval")
        return

    log.info(f"Evaluating on batch size {eval_cfg.eval_batch_size}")

    unique_hits, all_hits, _, mean_fitness = evaluate_with_frequencies(
        policy,
        landscape,
        eval_cfg.eval_threshold,
        eval_cfg.eval_thresholds,
        eval_cfg.eval_batch_size,
    )

    entropy = _compute_policy_entropy_bits(policy, training_cfg)

    log.info(f"Our policy evaluated on batch size {eval_cfg.eval_batch_size}")
    log.info(
        f"Unique hits: {unique_hits}, All hits: {all_hits}, Mean fitness: {mean_fitness:.4f}"
    )

    # Log exact evaluation metrics
    metrics_exact = {
        "all_hits": all_hits,
        "unique_hits": unique_hits,
        "mean_fitness": mean_fitness,
        "entropy": entropy,
        "eval_batch_size": eval_cfg.eval_batch_size,
        "eval_type": "exact",
    }
    log_metrics_csv(output_dir, iteration, metrics_exact)


def compute_reward_and_advantage(
    reward, baseline_reward: float, device: str, reward_unit=None
) -> tuple[torch.Tensor, Union[float, torch.Tensor]]:
    """Convert reward to torch tensor and compute advantage."""
    if isinstance(reward, torch.Tensor):
        reward_torch = reward.to(dtype=torch.float32, device=device)
        if reward_torch.dim() == 0:
            reward_val = reward_torch.item()
        else:
            if reward_unit is not None:
                reward_unit = reward_unit.to(device=device)
                reward_per_unit = torch.zeros(
                    reward_unit.max() + 1, device=device
                ).scatter_add_(0, reward_unit, reward_torch)
                reward_val = reward_per_unit / torch.bincount(reward_unit)
                baseline_reward = baseline_reward[reward_unit]
            else:
                reward_val = reward_torch.mean().item()
    else:
        reward_torch = torch.tensor(reward, dtype=torch.float32, device=device)
        reward_val = reward
    advantage = reward_torch - baseline_reward
    return advantage, reward_val


def compute_entropy_loss(
    entropy_cfg: EntropyConfig,
    policy,
    log_probs: torch.Tensor,
    landscape,
    sample_result,
    baseline,
    n_templates_considered=None,
) -> tuple[torch.Tensor, float]:
    """Compute entropy regularization term."""
    peptide_level = entropy_cfg.sequence_type == "peptide"
    if entropy_cfg.type == "exact":
        if policy.name == "Arbitrary":
            if peptide_level:
                return -calculate_peptide_entropy(policy), 0.0
            else:
                return -calculate_entropy(policy), 0.0
        else:
            if policy.num_templates == 1:
                entropy, cond_entropy = calculate_entropy_exact_bottle_based(
                    policy,
                    sample_result,
                    peptide_level=peptide_level,
                    baseline=baseline,
                )
            else:
                if n_templates_considered is None:
                    n_templates_considered = policy.num_templates
                entropy, cond_entropy = calculate_entropy_per_template(
                    policy,
                    sample_result,
                    peptide_level=peptide_level,
                    baseline=baseline,
                    n_lowest=n_templates_considered,
                )
            return -entropy, cond_entropy

    elif entropy_cfg.type == "gradient_estimate":
        if peptide_level:
            return (
                -calculate_peptide_entropy_gradient_estimate(
                    landscape, policy, sample_result
                ),
                0.0,
            )
        else:
            return -calculate_entropy_gradient_estimate(log_probs), 0.0
    elif entropy_cfg.type == "monte_carlo":
        entropy, cond_entropy = calculate_entropy_mc_bottle_based(
            policy,
            landscape,
            sample_result,
            log_probs,
            baseline=baseline,
            peptide_level=peptide_level,
        )
        return -entropy, cond_entropy
    elif entropy_cfg.type == "equal_share":
        entropy, baseline_update = calculate_equal_share_entropy(
            policy,
            landscape,
            sample_result,
            baseline=baseline,
            peptide_level=peptide_level,
        )
        return -entropy, baseline_update
    elif entropy_cfg.type == "marginal_gain":
        entropy, baseline_update = calculate_marginal_entropy_gain(
            policy,
            landscape,
            sample_result,
            baseline=baseline,
            peptide_level=peptide_level,
        )
        return -entropy, baseline_update
    else:
        raise ValueError(
            f"Unknown entropy_type: {entropy_cfg.type}. Must be 'exact', 'gradient_estimate', 'monte_carlo', 'equal_share', or 'marginal_gain'."
        )


def compute_policy_gradient_loss(
    sample_result,
    reward_function,
    landscape,
    baseline_reward: float,
    policy,
    entropy_cfg: EntropyConfig,
    alphabet: list[str],
    device: str,
    mc_samples_multiplier: int = 1,
    baseline_entropy=0.0,
    n_templates_considered=None,
) -> tuple[torch.Tensor, float, torch.Tensor, float]:
    """Compute policy gradient loss for a single sample."""
    sequences_str = sequences_to_strings_vectorized(sample_result.sequences, alphabet)
    reward = reward_function(sequences_str, landscape)
    advantage, reward_val = compute_reward_and_advantage(
        reward, baseline_reward, device
    )
    reward_val = reward_val * mc_samples_multiplier

    log_probs = policy.get_log_probs(sample_result)
    loss = -torch.sum(log_probs * advantage)

    if hasattr(policy, "get_batch_log_probs"):
        batch_log_probs = policy.get_batch_log_probs(sample_result)
        loss += -batch_log_probs * torch.sum(advantage)

    if entropy_cfg.lambda_ > 0:
        entropy_loss, entropy_reward = compute_entropy_loss(
            entropy_cfg,
            policy,
            log_probs,
            landscape,
            sample_result,
            baseline_entropy,
            n_templates_considered,
        )
        entropy_loss = entropy_loss * mc_samples_multiplier
        loss += entropy_loss * entropy_cfg.lambda_
    else:
        entropy_loss = torch.tensor(0.0, device=device)
        entropy_reward = 0.0

    return loss, reward_val, entropy_loss, entropy_reward


class PooledLandscapeCache:
    """
    Manages pooled landscape queries with subsampling across multiple MC samples.
    Supports both DNA and peptide-level uniqueness determination.

    Workflow:
    1) Collect all sequences from all mc_samples
    2) Determine unique sequences globally
    3) If uniqueness=peptide, convert to peptides and determine unique peptides
    4) Subsample unique sequences per batch and query the landscape for only these sequences
    5) Store rewards in a mapping from sequence to fitness
    """

    def __init__(
        self,
        sample_results: list,
        alphabet: list[str],
        policy,
        landscape,
        landscape_queries_per_batch: int = None,
        uniqueness: str = "dna",
        use_gpu: bool = False,
        unit_type: str = "batch",
    ):
        """
        Initialize pooled landscape cache.

        Args:
            sample_results: List of SampleResult objects from policy.sample()
            alphabet: DNA alphabet for sequence conversion
            landscape: Landscape object for fitness queries
            landscape_queries_per_batch: Max unique sequences to query per batch (None = no limit)
            uniqueness: "dna" or "peptide" - determines how sequences are deduplicated
            use_gpu: Use GPU-accelerated cache building (only for uniqueness="peptide")
            unit_type: On what level to aggregate the rewards. ("batch" | "template")
        """
        self.sample_results = sample_results
        self.alphabet = alphabet
        self.policy = policy
        self.landscape = landscape
        self.landscape_queries_per_batch = landscape_queries_per_batch
        self.uniqueness = uniqueness
        self.use_gpu = use_gpu
        self.unit_type = unit_type

        self.batch_unique_keys = []
        self.batch_sampled_keys = []
        self.batch_keys_scored = []
        self.importance_weights = []
        self.batch_key_counts = []
        self.batch_cond_log_probs = []
        self.batch_valid_peptides = []
        self.key_to_fitness = {}
        self.batch_reward_unit = []
        self.batch_unit_idx = []

        # Initialize GPU codon lookup if using GPU acceleration
        if self.use_gpu and torch.cuda.is_available() and self.uniqueness == "peptide":
            self.codon_lookup = build_codon_lookup_gpu(torch.device("cuda"))
        else:
            self.codon_lookup = None

        self._build_cache()

    def _build_cache(self):
        """Dispatcher for CPU vs GPU cache building."""
        if self.codon_lookup is not None and self.uniqueness == "peptide":
            self._build_cache_gpu()
        else:
            self._build_cache_cpu()

    def _prepare_batch_data(
        self,
    ) -> tuple[torch.Tensor, list[int], torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Common setup: concatenate sequences and deduplicate across all batches.

        Returns:
            Tuple of:
                - all_sequences: concatenated samples from all batches
                - batch_sizes: number of samples per batch
                - deduped_seqs: unique sequences across all batches
                - sample_to_dedup: maps each sample position to its deduped index
                - batch_sample_to_dedup: same mapping split per batch
                - reward_units: template/batch indices for reward aggregation
        """
        all_sequences = torch.cat([sr.sequences for sr in self.sample_results], dim=0)
        batch_sizes = [len(sr.sequences) for sr in self.sample_results]

        deduped_seqs, sample_to_dedup, _ = torch.unique(
            all_sequences, dim=0, return_inverse=True, return_counts=True
        )

        batch_sample_to_dedup = []
        offset = 0
        for batch_size in batch_sizes:
            batch_sample_to_dedup.append(sample_to_dedup[offset : offset + batch_size])
            offset += batch_size

        if self.unit_type == "template":
            reward_units = [sr.mixture_indices.cpu() for sr in self.sample_results]
        elif self.unit_type == "batch":
            reward_units = [
                torch.zeros_like(sr.mixture_indices).cpu() for sr in self.sample_results
            ]
        else:
            raise ValueError(f"Unknown reward aggregation unit type {self.unit_type}")

        return (
            all_sequences,
            batch_sizes,
            deduped_seqs,
            sample_to_dedup,
            batch_sample_to_dedup,
            reward_units,
        )

    def _apply_importance_sampling(self, unique_keys: list) -> tuple[set, float]:
        """
        Apply importance sampling if needed.

        Args:
            unique_keys: List of unique keys to potentially subsample

        Returns:
            Tuple of (sampled_keys_set, importance_weight)
        """
        n_unique = len(unique_keys)
        if (
            self.landscape_queries_per_batch is not None
            and n_unique > self.landscape_queries_per_batch
        ):
            sampled = set(random.sample(unique_keys, self.landscape_queries_per_batch))
            weight = n_unique / self.landscape_queries_per_batch
        else:
            sampled = set(unique_keys)
            weight = 1.0
        return sampled, weight

    def _store_batch_metadata(
        self,
        unique_keys: list,
        sampled: set,
        key_counts: dict,
        weight: float,
        cond_log_probs: torch.Tensor,
        valid_peptides: list,
        reward_unit: torch.Tensor,
        unit_idx: torch.Tensor,
    ) -> None:
        """Store per-batch metadata for reward computation."""
        self.batch_unique_keys.append(unique_keys)
        self.batch_sampled_keys.append(sampled)
        self.batch_keys_scored.append(sampled)
        self.batch_key_counts.append(key_counts)
        self.batch_reward_unit.append(reward_unit)
        self.batch_unit_idx.append(unit_idx)
        self.importance_weights.append(weight)
        self.batch_cond_log_probs.append(cond_log_probs)
        self.batch_valid_peptides.append(valid_peptides)

    def _get_template_dna_probs(self, sample_result):
        """
        Get DNA base probabilities for each template that was sampled.
        Handles both bottle-based and arbitrary policies.

        Returns:
            template_probs: Tensor of shape (num_unique_templates, str_length, alphabet_size)
            uq_templates: Tensor of unique template indices
            shifted_mix_idx: Tensor mapping sample_result.mixture_indices to 0-indexed positions
        """
        uq_templates = torch.unique(sample_result.mixture_indices)
        if hasattr(self.policy, "get_bottle_probs"):
            # Bottle-based policy: construct template probs from sampled bottles
            bottle_probs = self.policy.get_bottle_probs()
            first_idx = self.policy._find_first_occurrence_indices(
                sample_result.mixture_indices, uq_templates
            )
            template_bottles = sample_result.bottle_indices[first_idx]

            template_probs = bottle_probs[template_bottles.view(-1)].reshape(
                *template_bottles.shape, -1
            )
        else:
            # Arbitrary policy: directly get template probs
            template_probs = self.policy.get_template_probs()[uq_templates]

        # Create mapping from original template indices to 0-indexed positions
        shifted_mix_idx = torch.zeros_like(sample_result.mixture_indices, dtype=int)
        for i_new, i_old in enumerate(uq_templates):
            shifted_mix_idx[sample_result.mixture_indices == i_old] = i_new

        return template_probs, uq_templates, shifted_mix_idx

    def _query_cond_peptide_log_probs(
        self, sample_result, unique_peptides, peptide_inverse, uq_peptide_mix_idx
    ):
        """Compute conditional log probs of peptides given their templates."""
        device = self.policy.device

        # Get template DNA probs and convert to AA probs
        template_probs, uq_templates, shifted_mix_idx = self._get_template_dna_probs(
            sample_result
        )
        template_probs = torch.log(
            torch.stack([dna_probs_to_aa_probs(tp) for tp in template_probs], dim=0)
        ).clip(min=-1e8)

        # Ensure inputs are on the right device
        peptide_inverse = peptide_inverse.to(device)

        # Get log probs for each unique template:peptide combination
        if self.unit_type == "batch":
            uq_template_peptide, tp_inverse = torch.unique(
                torch.stack([shifted_mix_idx, peptide_inverse], dim=1),
                dim=0,
                return_inverse=True,
            )
            unique_peptides = unique_peptides[uq_template_peptide[:, 1]]
            uq_peptide_mix_idx = uq_template_peptide[:, 0]
        else:
            tp_inverse = peptide_inverse
            uq_peptide_mix_idx = torch.tensor(uq_peptide_mix_idx, device=device)

        n_peptides, padded_len = unique_peptides.shape
        log_probs = template_probs[
            uq_peptide_mix_idx[..., None].repeat(1, padded_len),
            torch.arange(padded_len, dtype=int, device=device).repeat(n_peptides, 1),
            unique_peptides,
        ]
        log_probs = torch.sum(log_probs * (unique_peptides != -1), dim=-1)
        return log_probs[tp_inverse]

    def _query_cond_dna_log_probs(self, sample_result):
        # Get template DNA probs
        template_probs, uq_templates, shifted_mix_idx = self._get_template_dna_probs(
            sample_result
        )

        # Get character probabilities for each sequence's template
        char_probs = template_probs[shifted_mix_idx]

        gathered_probs = char_probs.gather(
            2, sample_result.sequences.unsqueeze(-1)
        ).squeeze(-1)
        log_probs = torch.log(gathered_probs).sum(dim=1)

        return log_probs

    def _query_landscape_and_cache(
        self, dna_queries: list[str], result_keys: list
    ) -> None:
        """
        Query landscape and populate key_to_fitness cache.

        Args:
            dna_queries: DNA sequences to query
            result_keys: Keys to use in fitness mapping (peptides, DNA strings, or peptide IDs)
        """
        if not dna_queries:
            self.total_queries = 0
            return

        if is_landscape_multi_objective(self.landscape):
            # get_all_objectives returns np.ndarray or torch.Tensor depending on the
            # landscape implementation
            all_objectives = self.landscape.get_all_objectives(dna_queries)
            self.key_to_fitness = {
                key: all_objectives[i, :] for i, key in enumerate(result_keys)
            }
        else:
            objective = (
                self.landscape.active_objs[0] if self.landscape.active_objs else None
            )
            fitness_values = self.landscape.get_fitness_values(
                dna_queries, objective=objective
            )
            self.key_to_fitness = dict(zip(result_keys, fitness_values))

        self.total_queries = len(dna_queries)

    def _build_cache_cpu(self):
        """Build cache by pooling queries across all batches (CPU implementation)."""
        if self.uniqueness == "peptide":
            self._build_cache_cpu_peptide()
        else:
            self._build_cache_cpu_dna()

    def _build_cache_cpu_peptide(self):
        """CPU cache building for peptide-level uniqueness."""
        _, _, deduped_seqs, _, batch_sample_to_dedup, reward_units = (
            self._prepare_batch_data()
        )

        dedup_idx_to_peptide = tensor_to_protein_batch(deduped_seqs)
        peptide_lens = {
            idx: len(peptide) for idx, peptide in dedup_idx_to_peptide.items()
        }
        max_len = np.max(list(peptide_lens.values()))

        for i, (sample_to_dedup, reward_unit) in enumerate(
            zip(batch_sample_to_dedup, reward_units)
        ):
            unique_units, sample_to_unit = torch.unique(
                reward_unit, return_inverse=True
            )

            unit_peptide_strs = []  # Per-unit: unique peptide strings
            batch_peptide_strs_flat = []  # Flattened across all units
            sampled_batch = []
            peptide_counts_batch = []
            weights_batch = []
            sample_to_flat_peptide = torch.empty_like(sample_to_unit)
            peptide_to_unit_idx = []
            all_units_peptides_str = []

            for unit_idx in range(len(unique_units)):
                unit_dedup_indices = sample_to_dedup[sample_to_unit == unit_idx]
                unique_dedup_in_unit, sample_to_unit_dedup, counts = torch.unique(
                    unit_dedup_indices, return_inverse=True, return_counts=True
                )
                unique_dedup_list = unique_dedup_in_unit.tolist()
                unique_peptides = [
                    dedup_idx_to_peptide[idx] for idx in unique_dedup_list
                ]

                # Key by peptide strings, aggregating counts for DNA seqs encoding same peptide
                peptide_counts = {}
                for peptide_str, count in zip(unique_peptides, counts.tolist()):
                    peptide_counts[peptide_str] = (
                        peptide_counts.get(peptide_str, 0) + count
                    )

                peptide_strs_in_unit = list(peptide_counts.keys())
                sampled, weight = self._apply_importance_sampling(peptide_strs_in_unit)

                peptide_counts_batch.append(peptide_counts)
                sample_to_flat_peptide[sample_to_unit == unit_idx] = (
                    sample_to_unit_dedup.cpu() + len(all_units_peptides_str)
                )
                unit_peptide_strs.append(peptide_strs_in_unit)
                batch_peptide_strs_flat += peptide_strs_in_unit
                sampled_batch.append(list(sampled))
                weights_batch.append(weight)
                # Keep DNA-index-based length for log prob computation
                peptide_to_unit_idx += [unit_idx] * len(unique_peptides)

                all_units_peptides_str += unique_peptides

            all_units_peptides_int = [
                torch.tensor([AMINO_ACID_ALPHABET[aa] for aa in peptide], dtype=int)
                for peptide in all_units_peptides_str
            ]
            all_units_unique_peptides = torch.nn.utils.rnn.pad_sequence(
                all_units_peptides_int, batch_first=True, padding_value=-1
            ).to(self.policy.device)
            batch_valid_peptides = torch.tensor(
                [peptide_lens[idx.item()] == max_len for idx in sample_to_dedup]
            )
            log_probs_cond = self._query_cond_peptide_log_probs(
                self.sample_results[i],
                all_units_unique_peptides,
                sample_to_flat_peptide,
                peptide_to_unit_idx,
            )

            self._store_batch_metadata(
                batch_peptide_strs_flat,
                sampled_batch,
                peptide_counts_batch,
                weights_batch,
                log_probs_cond,
                batch_valid_peptides,
                sample_to_unit,
                unique_units,
            )

        all_sampled_peptides = set()
        for batch_keys in self.batch_sampled_keys:
            for unit_keys in batch_keys:
                all_sampled_peptides.update(unit_keys)

        # Find one deduped DNA index per unique peptide (for querying landscape)
        peptide_to_dedup_idx = {}
        for dedup_idx, peptide in dedup_idx_to_peptide.items():
            if peptide in all_sampled_peptides and peptide not in peptide_to_dedup_idx:
                peptide_to_dedup_idx[peptide] = dedup_idx

        sampled_dedup_indices = list(peptide_to_dedup_idx.values())
        sampled_tensors = deduped_seqs[sampled_dedup_indices]
        sampled_strs = sequences_to_strings_vectorized(sampled_tensors, self.alphabet)

        dedup_to_local = {
            dedup_idx: local_idx
            for local_idx, dedup_idx in enumerate(sampled_dedup_indices)
        }

        peptide_to_dna = {}
        for peptide, dedup_idx in peptide_to_dedup_idx.items():
            peptide_to_dna[peptide] = sampled_strs[dedup_to_local[dedup_idx]]

        # Query landscape and cache using peptide strings as keys
        dna_queries = [peptide_to_dna[p] for p in all_sampled_peptides]
        self._query_landscape_and_cache(dna_queries, list(all_sampled_peptides))

    def _build_cache_cpu_dna(self):
        """CPU cache building for DNA-level uniqueness."""
        _, _, deduped_seqs, _, batch_sample_to_dedup, reward_units = (
            self._prepare_batch_data()
        )

        # Build dedup_to_str mapping (DNA cache uses string keys)
        all_dedup_indices = set()
        for sample_to_dedup in batch_sample_to_dedup:
            all_dedup_indices.update(torch.unique(sample_to_dedup).tolist())

        all_dedup_indices_list = sorted(list(all_dedup_indices))
        all_dedup_tensors = deduped_seqs[all_dedup_indices_list]
        all_dedup_strs = sequences_to_strings_vectorized(
            all_dedup_tensors, self.alphabet
        )

        dedup_to_str = {
            idx: all_dedup_strs[i] for i, idx in enumerate(all_dedup_indices_list)
        }

        dedup_to_count_per_unit = []

        for i, (sample_to_dedup, reward_unit) in enumerate(
            zip(batch_sample_to_dedup, reward_units)
        ):
            unique_units, sample_to_unit = torch.unique(
                reward_unit, return_inverse=True
            )

            unit_dna_strs = []
            dna_counts = []
            weight = []
            sampled = []

            for unit_idx in range(len(unique_units)):
                unit_dedup_ids = sample_to_dedup[sample_to_unit == unit_idx]
                unique_dedup_in_unit, counts = torch.unique(
                    unit_dedup_ids, return_counts=True
                )

                unique_dedup_list = unique_dedup_in_unit.tolist()
                dedup_counts_unit = dict(zip(unique_dedup_list, counts.tolist()))

                sampled_dedup_unit, weight_unit = self._apply_importance_sampling(
                    unique_dedup_list
                )

                dedup_to_count_per_unit.append(
                    (unique_dedup_list, dedup_counts_unit, sampled_dedup_unit)
                )

                # Convert dedup indices to DNA strings
                dna_strs_in_unit = [dedup_to_str[idx] for idx in unique_dedup_list]
                sampled_unit = {dedup_to_str[idx] for idx in sampled_dedup_unit}
                key_counts_unit = {
                    dedup_to_str[idx]: count for idx, count in dedup_counts_unit.items()
                }

                weight.append(weight_unit)
                sampled.append(sampled_unit)
                dna_counts.append(key_counts_unit)
                unit_dna_strs.append(dna_strs_in_unit)

            log_probs_cond = self._query_cond_dna_log_probs(self.sample_results[i])

            self._store_batch_metadata(
                unit_dna_strs,
                sampled,
                dna_counts,
                weight,
                log_probs_cond,
                torch.ones_like(sample_to_dedup, dtype=bool),
                sample_to_unit,
                unique_units,
            )

        all_sampled_dedup = set().union(*[s for _, _, s in dedup_to_count_per_unit])

        dna_queries = [dedup_to_str[idx] for idx in all_sampled_dedup]
        self._query_landscape_and_cache(dna_queries, dna_queries)

    def _build_cache_gpu(self):
        """GPU-accelerated cache building with integer peptide IDs."""
        # GPU cache marks peptides as valid only if they have no stop codons (first_stop == -1).
        # This assumes fixed-length peptides where any stop codon makes the sequence invalid.
        # For landscapes with variable peptide lengths (e.g. MHC flurry), use CPU cache instead.
        if hasattr(self.landscape, "min_peptide_len") and hasattr(
            self.landscape, "max_peptide_len"
        ):
            if self.landscape.min_peptide_len != self.landscape.max_peptide_len:
                raise ValueError(
                    f"GPU cache requires fixed peptide length (min_peptide_len == max_peptide_len), "
                    f"but got min={self.landscape.min_peptide_len}, max={self.landscape.max_peptide_len}. "
                    f"Use gpu_accelerated=false for landscapes with variable peptide lengths."
                )

        _, _, deduped_seqs, _, batch_sample_to_dedup, reward_units = (
            self._prepare_batch_data()
        )

        # --- Translate DNA to amino acid indices and mask positions after stop codons ---
        aa_indices_gpu, first_stop_gpu = translate_dna_to_aa_indices_gpu(
            deduped_seqs, self.codon_lookup
        )
        torch.cuda.synchronize()

        n_seqs, max_codons = aa_indices_gpu.shape
        positions = torch.arange(max_codons, device="cuda").unsqueeze(0)
        stops = first_stop_gpu.unsqueeze(1)
        # Keep positions before stop codon, or all positions if no stop (stops == -1)
        mask = (positions < stops) | (stops == -1)
        masked_peptide_ids = torch.where(
            mask,
            aa_indices_gpu,
            torch.tensor(-1, device="cuda", dtype=aa_indices_gpu.dtype),
        )

        # --- Map each deduped DNA sequence to a unique peptide ID ---
        unique_peptides, dedup_to_peptide_id = torch.unique(
            masked_peptide_ids, dim=0, return_inverse=True
        )
        valid_peptides = first_stop_gpu == -1
        torch.cuda.synchronize()

        # --- Process each batch and its reward aggregation units ---
        for i, (sample_to_dedup, reward_unit) in enumerate(
            zip(batch_sample_to_dedup, reward_units)
        ):
            unique_units, sample_to_unit = torch.unique(
                reward_unit, return_inverse=True
            )
            batch_peptide_ids = dedup_to_peptide_id[sample_to_dedup]

            unit_peptide_ids = []  # Per-unit: unique peptide IDs in each unit
            batch_peptide_ids_flat = []  # Flattened across all units
            peptide_counts = []
            weight = []
            sampled = []
            sample_to_flat_peptide = torch.empty_like(batch_peptide_ids)
            peptide_to_unit_idx = []  # Maps each flat peptide ID to its unit

            for unit_idx in range(len(unique_units)):
                unit_seqs_peptide_ids = batch_peptide_ids[sample_to_unit == unit_idx]

                peptide_ids_in_unit, sample_to_unit_peptide, counts_in_unit = (
                    torch.unique(
                        unit_seqs_peptide_ids, return_counts=True, return_inverse=True
                    )
                )

                # Transfer to CPU (only unique IDs, not all sequences)
                peptide_ids_cpu = peptide_ids_in_unit.cpu().tolist()
                counts_cpu = counts_in_unit.cpu().tolist()

                peptide_counts.append(dict(zip(peptide_ids_cpu, counts_cpu)))

                sampled_unit, weight_unit = self._apply_importance_sampling(
                    peptide_ids_cpu
                )

                # Offset by number of peptides already processed in previous units
                sample_to_flat_peptide[sample_to_unit == unit_idx] = (
                    sample_to_unit_peptide + len(batch_peptide_ids_flat)
                )
                unit_peptide_ids.append(peptide_ids_cpu)
                batch_peptide_ids_flat += peptide_ids_cpu
                weight.append(weight_unit)
                sampled.append(sampled_unit)
                peptide_to_unit_idx += [unit_idx] * len(peptide_ids_cpu)

            batch_valid_peptides = valid_peptides[sample_to_dedup]
            all_units_unique_peptides = unique_peptides[batch_peptide_ids_flat].to(
                dtype=int
            )
            log_probs_cond = self._query_cond_peptide_log_probs(
                self.sample_results[i],
                all_units_unique_peptides,
                sample_to_flat_peptide,
                peptide_to_unit_idx,
            )

            self._store_batch_metadata(
                unit_peptide_ids,
                sampled,
                peptide_counts,
                weight,
                log_probs_cond,
                batch_valid_peptides,
                sample_to_unit,
                unique_units,
            )

        all_sampled_peptide_ids = set().union(
            *[sk for batch_sampled in self.batch_sampled_keys for sk in batch_sampled]
        )

        # --- Find one representative DNA sequence per sampled peptide for landscape queries ---
        # Sort by peptide ID and take the first deduped DNA index for each unique peptide
        sorted_peptide_ids, sort_indices = torch.sort(dedup_to_peptide_id, stable=True)
        sorted_dedup_indices = torch.arange(len(dedup_to_peptide_id), device="cuda")[
            sort_indices
        ]

        # Find first occurrence (where peptide ID changes)
        first_occurrence_mask = torch.cat(
            [
                torch.tensor([True], device="cuda"),
                sorted_peptide_ids[1:] != sorted_peptide_ids[:-1],
            ]
        )
        first_dedup_per_peptide = sorted_dedup_indices[first_occurrence_mask]

        sampled_peptide_ids_tensor = torch.tensor(
            list(all_sampled_peptide_ids), device="cuda"
        )
        sampled_dedup_indices = first_dedup_per_peptide[sampled_peptide_ids_tensor]
        torch.cuda.synchronize()

        sampled_dedup_indices_cpu = sampled_dedup_indices.cpu().tolist()
        sampled_tensors = deduped_seqs[sampled_dedup_indices_cpu]
        sampled_dna_strs = sequences_to_strings_vectorized(
            sampled_tensors, self.alphabet
        )

        peptide_id_to_dna = dict(zip(all_sampled_peptide_ids, sampled_dna_strs))
        dna_queries = [peptide_id_to_dna[pid] for pid in all_sampled_peptide_ids]

        self._query_landscape_and_cache(dna_queries, list(all_sampled_peptide_ids))

    def get_batch_info(self, batch_idx: int) -> dict:
        """
        Get info for a specific batch.

        Args:
            batch_idx: Index of the batch (0 to mc_samples-1)

        Returns:
            Dictionary with batch information for reward computation:
                - unique_keys: All unique keys in each aggregation unit of this batch (list[list])
                - sampled_keys: Subset sampled for landscape queries (list[set])
                - keys_scored: Same as sampled_keys (list[set])
                - importance_weight: Reweighting factor for each aggregation unit (list)
                - key_counts: How many times each key appeared in each aggregation unit of each batch (list[dict])
                - key_to_fitness: GLOBAL fitness cache across all batches (dict)
                    * Single-objective: {key: float, ...}
                    * Multi-objective: {key: np.ndarray or torch.Tensor, ...}
                - batch_size: Total sequences in this batch (int)
                - unit_sizes: Total sequences in each unit (list[int])
                - landscape: Landscape object reference
                - reward_unit: reward aggregation unit indices (torch.Tensor)
                - unit_idx: indices of units from which batch contains samples (torch.Tensor)

            Note: Key types depend on uniqueness mode and backend:
                  - CPU + peptide: peptide strings (e.g., "MSKGEELFT")
                  - CPU + DNA: DNA strings (e.g., "ATGCATGCATGC")
                  - GPU + peptide: integer peptide IDs (e.g., 42)
                  - GPU + DNA: not supported, falls back to CPU + DNA

        Example (CPU peptide mode):
            For 100 DNA seqs → 30 unique peptides, subsampled to 20:
            {
                "unique_keys": ["MSKGEELFT", ..., "ATGCATGC"],  # 30 peptides
                "sampled_keys": {"MSKGEELFT", ..., "QWERTY"},   # 20 peptides
                "keys_scored": {"MSKGEELFT", ..., "QWERTY"},    # 20 peptides
                "importance_weight": 1.5,  # 30 / 20
                "key_counts": {"MSKGEELFT": 15, "QWERTY": 8, ...},
                "key_to_fitness": {"MSKGEELFT": 0.98, ...},  # all mc_samples
                "batch_size": 100,
                "landscape": <landscape_obj>
            }
        """
        unit_sizes = [
            sum(unit_key_counts.values())
            for unit_key_counts in self.batch_key_counts[batch_idx]
        ]
        return {
            "unique_keys": self.batch_unique_keys[batch_idx],
            "sampled_keys": self.batch_sampled_keys[batch_idx],
            "keys_scored": self.batch_keys_scored[batch_idx],
            "importance_weight": self.importance_weights[batch_idx],
            "key_counts": self.batch_key_counts[batch_idx],
            "key_to_fitness": self.key_to_fitness,
            "batch_size": sum(unit_sizes),
            "unit_sizes": unit_sizes,
            "landscape": self.landscape,
            "sample_result": self.sample_results[batch_idx],
            "valid_peptides": self.batch_valid_peptides[batch_idx],
            "cond_log_probs": self.batch_cond_log_probs[batch_idx],
            "reward_unit": self.batch_reward_unit[batch_idx],
            "unit_idx": self.batch_unit_idx[batch_idx],
        }
