"""
policies.py

Defines the policy gradient optimizer classes for sequence generation.
Includes ArbitraryBaseOptimizer and FiniteBaseOptimizer.
"""

import csv
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from constants import (
    AMINO_ACID_ALPHABET,
    AMINO_ACIDS,
    CODON_TABLE,
    DNA_BASES,
    IUPAC_BOTTLES,
    IUPAC_MAP,
)
from landscapes import (
    generate_iupac_sequences,
    hit_above_threshold,
    hit_above_threshold_all,
    hit_above_threshold_individual,
    generate_iupac_sequences,
    sequences_to_strings_vectorized,
)
from train_utils import (
    compute_edit_distance_stats,
    compute_peptide_length_stats,
)

log = logging.getLogger(__name__)


def evaluate_on_batch(
    policy, landscape, threshold: Optional[float] = 0.995, batch_size: int = 100000
):
    sample_result = policy.sample(batch_size)
    sequences = sample_result.sequences
    sequences_str = sequences_to_strings_vectorized(sequences, landscape.alphabet)

    uniqueness_type = "dna"
    if landscape.is_peptide:
        uniqueness_type = "peptide"

    log.info(f"Using threshold {threshold} and uniqueness {uniqueness_type}")
    unique_hits = hit_above_threshold(
        landscape, sequences_str, threshold, uniqueness_type=uniqueness_type
    )
    all_hits = hit_above_threshold_all(landscape, sequences_str, threshold)

    # Log peptide length stats for DNA->peptide landscapes
    if hasattr(landscape, "min_peptide_len"):
        stats = compute_peptide_length_stats(sequences_str)
        log.info(
            f"Peptide lengths - min: {stats['peptide_len_p0']}, "
            f"p25: {stats['peptide_len_p25']}, median: {stats['peptide_len_p50']}, "
            f"p75: {stats['peptide_len_p75']}, max: {stats['peptide_len_p100']}"
        )

        # Log edit distance stats if wild-type is specified
        if hasattr(landscape, "wild_type") and landscape.wild_type is not None:
            edit_stats = compute_edit_distance_stats(sequences_str, landscape.wild_type)
            log.info(
                f"Edit distances from WT - min: {edit_stats['edit_dist_p0']}, "
                f"p25: {edit_stats['edit_dist_p25']}, median: {edit_stats['edit_dist_p50']}, "
                f"p75: {edit_stats['edit_dist_p75']}, max: {edit_stats['edit_dist_p100']}"
            )

    return unique_hits, all_hits


@dataclass
class SampleResult:
    """
    Bundle together all objects that might be in a .sample() object of an optimizer.

    Fields:
        sequences: Sampled sequences (batch_size, str_length)
        mixture_indices: Template assignments per sequence (batch_size,)
        bottle_indices: Bottle assignments (batch_size, str_length) or None
        selected_bottles: Selected non-AGCT bottles (max_non_agct_slots,) or None
    """

    sequences: torch.Tensor
    mixture_indices: torch.Tensor
    bottle_indices: Optional[torch.Tensor] = None
    selected_bottles: Optional[torch.Tensor] = None


def try_different_temperatures(
    policy, landscape, options, threshold=0.995, batch_size=120000
):
    # options = [1.1, 1.2, 1.3, 1.4, 1.5]
    for t in options:
        sample_result = policy.sample(batch_size, tau=t)
        sequences_str = [
            "".join([landscape.alphabet[c] for c in s]) for s in sample_result.sequences
        ]

        log.info(f"Our policy for tau = {t} :")

        if landscape.is_peptide:
            uniqueness_type = "peptide"
        else:
            uniqueness_type = "dna"

        unique_hits = hit_above_threshold(
            landscape, sequences_str, threshold, uniqueness_type=uniqueness_type
        )

        log.info(f"Unique hits, duplicate protein encodings excluded: {unique_hits}")
        log.info(
            f"All hits: {hit_above_threshold_all(landscape, sequences_str, threshold)}"
        )

    return


class ArbitraryBaseOptimizer(nn.Module):
    """Optimizes synthesis parameters using fully flexible per-position probabilities."""

    def __init__(
        self,
        str_length,
        alphabet_size,
        batch_size,
        num_templates,
        device="cpu",
        uniform_template_mix=False,
    ):
        super(ArbitraryBaseOptimizer, self).__init__()
        self.str_length = str_length
        self.alphabet_size = alphabet_size
        self.batch_size = batch_size
        self.num_templates = num_templates
        self.device = device
        self.uniform_template_mix = uniform_template_mix

        self.latent_w = nn.Parameter(
            torch.randn(num_templates, str_length, alphabet_size).to(device)
        )
        self.latent_w_mix = nn.Parameter(torch.randn(num_templates).to(device))

        if uniform_template_mix:
            self.latent_w_mix.data.zero_()
            self.latent_w_mix.requires_grad = False

        self.name = "Arbitrary"

    def get_probs(self, tau=1):
        template_probs = torch.softmax(self.latent_w / tau, dim=-1)
        mixture_probs = torch.softmax(self.latent_w_mix / tau, dim=0)
        return template_probs, mixture_probs

    def get_template_probs(self, tau=1):
        template_probs, mixture_probs = self.get_probs(tau)
        return template_probs

    def sample(self, batch_size=None, tau=1, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        with torch.no_grad():
            template_probs, mixture_probs = self.get_probs(tau)
            mixture_dist = torch.distributions.Categorical(probs=mixture_probs)
            mixture_indices = mixture_dist.sample(sample_shape=(batch_size,))
            selected_template_probs = template_probs[mixture_indices]
            seq_dist = torch.distributions.Categorical(probs=selected_template_probs)
            sequences = seq_dist.sample()

            return SampleResult(
                sequences=sequences,
                mixture_indices=mixture_indices,
                bottle_indices=None,
                selected_bottles=None,
            )

    def get_log_probs(self, sample_result: SampleResult):
        sequences = sample_result.sequences
        mixture_indices = sample_result.mixture_indices

        # Ensure input tensors are on the correct device if they weren't already
        sequences = sequences.to(self.device)
        mixture_indices = mixture_indices.to(self.device)

        template_probs, mixture_probs = self.get_probs()

        log_prob_mix = torch.log(mixture_probs[mixture_indices])
        char_probs = template_probs[mixture_indices]

        # Ensure gather operation correctly uses tensors on the same device
        gathered_probs = char_probs.gather(2, sequences.unsqueeze(-1)).squeeze(-1)
        log_prob_seq = torch.log(gathered_probs).sum(dim=1)
        return log_prob_mix + log_prob_seq

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def load_params(self, path, device=None):
        state_dict = torch.load(path, map_location=device or self.device)
        self.load_state_dict(state_dict)

    def get_template_dna_probs(self):
        """
        Get per-position DNA base probabilities for each template.

        Returns:
            numpy array of shape (num_templates, str_length, 4) with DNA base probabilities
        """
        template_probs, _ = self.get_probs()
        return template_probs.detach().cpu().numpy()


class BottleBasedOptimizer(nn.Module, ABC):
    """
    Abstract base class for bottle-based synthesis optimizers.

    This class cannot be instantiated directly. Use concrete subclasses like
    FiniteBaseOptimizer or IupacBaseOptimizer instead.

    Subclasses must implement:
        - get_bottle_probs(tau): Returns the bottle probability distributions
    """

    def __init__(
        self,
        str_length,
        alphabet_size,
        batch_size,
        num_templates,
        num_bottles,
        device="cpu",
        uniform_template_mix=False,
    ):
        super(BottleBasedOptimizer, self).__init__()
        self.str_length = str_length
        self.alphabet_size = alphabet_size
        self.batch_size = batch_size
        self.num_templates = num_templates
        self.num_bottles = num_bottles
        self.device = device
        self.uniform_template_mix = uniform_template_mix

        self.latent_phi = nn.Parameter(
            torch.randn(num_templates, str_length, num_bottles).to(device)
        )
        self.latent_w_mix = nn.Parameter(torch.randn(num_templates).to(device))

        if uniform_template_mix:
            self.latent_w_mix.data.zero_()
            self.latent_w_mix.requires_grad = False

        self.name = "BottleBased"

    @abstractmethod
    def get_bottle_probs(self, tau=1):
        """
        Returns the bottle probabilities v_probs.

        Subclasses must implement this method to define how bottle probabilities
        are computed (e.g., via softmax for learnable bottles, or returning
        fixed probabilities for IUPAC).

        Args:
            tau: Temperature parameter for sampling (may be ignored by some subclasses)

        Returns:
            torch.Tensor: Bottle probabilities of shape (num_bottles, alphabet_size)
        """
        pass

    def get_probs(self, tau=1):
        """Returns bottle probabilities, phi probabilities, and mixture probabilities."""
        v_probs = self.get_bottle_probs(tau)
        phi_probs = torch.softmax(self.latent_phi / tau, dim=-1)
        w_mix_probs = torch.softmax(self.latent_w_mix / tau, dim=0)
        return v_probs, phi_probs, w_mix_probs

    def _find_first_occurrence_indices(self, mixture_indices, unique_templates):
        """Finds the first batch index where each unique template appears.

        Uses scatter_reduce with 'min' to efficiently find first occurrences.

        Args:
            mixture_indices: Tensor (batch_size,) of template assignments
            unique_templates: Tensor of unique template indices that appear in mixture_indices

        Returns:
            Tensor (num_unique,) of batch indices for first occurrence of each template
        """
        batch_indices = torch.arange(
            len(mixture_indices), device=mixture_indices.device
        )
        # Initialize to 0 and use include_self=False to ignore default values.
        # This is safe because unique_templates only contains templates that actually
        # appear in mixture_indices, so every template will have at least one occurrence.
        first_occurrence = torch.zeros(
            (self.num_templates,),
            dtype=torch.long,
            device=mixture_indices.device,
        )
        # Use reduce="min" (not "amin") for compatibility with older torch versions.
        first_occurrence.scatter_reduce_(
            0, mixture_indices, batch_indices, reduce="min", include_self=False
        )
        return first_occurrence[unique_templates]

    def sample(self, batch_size=None, tau=1, greedy=False):
        """
        Sample sequences using the bottle-based policy.

        Args:
            greedy: If True, use argmax for bottle selection per template.
                    Template mixing and character sampling remain stochastic.
        """
        if batch_size is None:
            batch_size = self.batch_size
        with torch.no_grad():
            v_probs, phi_probs, w_mix_probs = self.get_probs(tau)

            mixture_dist = torch.distributions.Categorical(probs=w_mix_probs)
            mixture_indices = mixture_dist.sample(sample_shape=(batch_size,))

            # Sample bottles once per template (not per sequence)
            # All sequences from the same template use the same bottle configuration
            unique_templates, positions = torch.unique(
                mixture_indices, return_inverse=True
            )

            if greedy:
                greedy_bottles = self.sample_greedy_bottles()
                bottles_for_unique = torch.from_numpy(
                    greedy_bottles[unique_templates.cpu().numpy()]
                ).to(self.device)
            else:
                # Batch sample bottles for all unique templates at once
                # phi_probs[unique_templates] shape: (num_unique, str_length, num_bottles)
                phi_probs_for_unique = phi_probs[unique_templates]
                bottle_dist = torch.distributions.Categorical(
                    probs=phi_probs_for_unique
                )
                bottles_for_unique = bottle_dist.sample()  # (num_unique, str_length)

            bottle_indices = bottles_for_unique[positions]  # (batch_size, str_length)

            # Character sampling from selected bottles
            # v_probs[bottle_indices] shape: (batch_size, str_length, alphabet_size)
            v_probs_for_batch = v_probs[bottle_indices]
            char_dist = torch.distributions.Categorical(probs=v_probs_for_batch)
            sequences = char_dist.sample()  # (batch_size, str_length)

            return SampleResult(
                sequences=sequences,
                mixture_indices=mixture_indices,
                bottle_indices=bottle_indices,
                selected_bottles=None,
            )

    def get_batch_log_probs(self, sample_result: SampleResult, sum: bool = True):
        """
        Compute batch-level log probabilities (bottle selection).
        Bottles are sampled once per unique template, not per sequence.

        Returns:
            log_prob_phi_total: Scalar tensor with log p(bottles | templates)
            sum (bool): If False, returns batch log probs per template as a tensor.
        """
        mixture_indices = sample_result.mixture_indices
        bottle_indices = sample_result.bottle_indices
        assert (
            bottle_indices is not None
        ), "BottleBasedOptimizer requires bottle_indices"

        _, phi_probs, _ = self.get_probs()

        unique_templates = torch.unique(mixture_indices)
        first_occurrence_unique = self._find_first_occurrence_indices(
            mixture_indices, unique_templates
        )
        bottles_for_unique = bottle_indices[first_occurrence_unique]
        phi_probs_for_unique = phi_probs[unique_templates]
        gathered_phi_probs = phi_probs_for_unique.gather(
            2, bottles_for_unique.unsqueeze(-1)
        ).squeeze(-1)
        log_prob_phi_total = torch.log(gathered_phi_probs)

        if sum:
            return log_prob_phi_total.sum()

        return log_prob_phi_total.sum(dim=-1)

    def get_log_probs(self, sample_result: SampleResult):
        """
        Compute per-sequence log probabilities (template and character selection).

        Returns:
            log_probs: Tensor of shape (batch_size,) with log p(s_i | t_i, b_{t_i}) + log p(t_i)
        """
        sequences = sample_result.sequences
        mixture_indices = sample_result.mixture_indices
        bottle_indices = sample_result.bottle_indices
        assert (
            bottle_indices is not None
        ), "BottleBasedOptimizer requires bottle_indices"

        v_probs, _, w_mix_probs = self.get_probs()

        log_prob_mix = torch.log(w_mix_probs[mixture_indices])

        v_probs_for_batch = v_probs[bottle_indices]
        gathered_v_probs = v_probs_for_batch.gather(2, sequences.unsqueeze(-1)).squeeze(
            -1
        )
        log_prob_v = torch.log(gathered_v_probs).sum(dim=1)

        return log_prob_mix + log_prob_v

    def save_params(self, path):
        """Save model parameters to disk."""
        torch.save(self.state_dict(), path)

    def load_params(self, path, device=None):
        """Load model parameters from disk."""
        state_dict = torch.load(path, map_location=device or self.device)
        self.load_state_dict(state_dict)

    def sample_greedy_bottles(self):
        """
        Returns:
            numpy array of shape (num_templates, str_length) with bottle indices
        """
        _, phi_probs, _ = self.get_probs()
        phi_probs_np = phi_probs.detach().cpu().numpy()
        return np.argmax(phi_probs_np, axis=2)

    def get_template_dna_probs(self):
        """
        Returns:
            numpy array of shape (num_templates, str_length, 4) with DNA base probabilities
        """
        v_probs, _, _ = self.get_probs()
        v_probs_np = v_probs.detach().cpu().numpy()

        greedy_bottles = self.sample_greedy_bottles()

        template_dna_probs = np.zeros((self.num_templates, self.str_length, 4))
        for t in range(self.num_templates):
            for pos in range(self.str_length):
                bottle = greedy_bottles[t, pos]
                template_dna_probs[t, pos] = v_probs_np[bottle]

        return template_dna_probs


class FiniteBaseOptimizer(BottleBasedOptimizer):
    """Optimizes synthesis parameters using a finite set of learnable "bottles"."""

    def __init__(
        self,
        str_length,
        alphabet_size,
        batch_size,
        num_templates,
        num_bottles,
        device="cpu",
        uniform_template_mix=False,
    ):
        super(FiniteBaseOptimizer, self).__init__(
            str_length,
            alphabet_size,
            batch_size,
            num_templates,
            num_bottles,
            device,
            uniform_template_mix,
        )
        self.latent_v = nn.Parameter(torch.randn(num_bottles, alphabet_size).to(device))
        self.name = "Finite"

    def get_bottle_probs(self, tau=1):
        """Returns bottle probabilities using softmax over latent_v."""
        return torch.softmax(self.latent_v / tau, dim=-1)


class IupacBaseOptimizer(BottleBasedOptimizer):
    """Optimizes synthesis parameters over IUPAC bases (fixed bottle probabilities)."""

    def __init__(
        self,
        str_length,
        alphabet_size,
        batch_size,
        num_templates,
        device="cpu",
        uniform_template_mix=False,
    ):
        # IUPAC has a fixed number of bottles (15)
        iupac_bottles = self._create_iupac_bottles()
        num_bottles = iupac_bottles.shape[0]

        super(IupacBaseOptimizer, self).__init__(
            str_length,
            alphabet_size,
            batch_size,
            num_templates,
            num_bottles,
            device,
            uniform_template_mix,
        )

        # For IUPAC, bottle probabilities are fixed (not learnable parameters)
        self._iupac_bottle_probs = iupac_bottles.to(self.device)
        self._iupac_bottle_probs.requires_grad = False
        self.name = "IUPAC"

    def _create_iupac_bottles(self):
        """Creates the fixed IUPAC bottle probabilities."""
        iupac_bases = [probs for _, probs in IUPAC_BOTTLES]
        iupac_bases = torch.tensor(iupac_bases, dtype=torch.float32)
        iupac_bases = iupac_bases / iupac_bases.sum(dim=1, keepdim=True)
        return iupac_bases

    def get_bottle_probs(self, tau=1):
        """Returns fixed IUPAC bottle probabilities (tau parameter ignored for IUPAC)."""
        return self._iupac_bottle_probs

    def to_text(self):
        """
        Convert policy to IUPAC template text for oligo ordering.

        Returns:
            str: Formatted text with template sequences and weights
        """
        _, phi_probs, w_mix_probs = self.get_probs(tau=0.1)
        phi_probs = phi_probs.detach().cpu().numpy()
        w_mix_probs = w_mix_probs.detach().cpu().numpy()

        bottle_to_iupac = {idx: code for idx, (code, _) in enumerate(IUPAC_BOTTLES)}

        lines = []
        for t_idx in range(phi_probs.shape[0]):
            most_likely = np.argmax(phi_probs[t_idx], axis=1)
            template = "".join(bottle_to_iupac[b] for b in most_likely)
            weight = w_mix_probs[t_idx] * 100
            lines.append(f"Template {t_idx} (weight: {weight:.1f}%): {template}")

        return "\n".join(lines)


class IupacPlusOptimizer(IupacBaseOptimizer):
    """IUPAC bases (fixed, 15 bottles) + N learnable custom bottles."""

    def __init__(
        self,
        str_length: int,
        alphabet_size: int,
        batch_size: int,
        num_templates: int,
        num_custom_bottles: int = 5,
        device: str = "cpu",
        uniform_template_mix: bool = False,
    ):
        self.num_custom_bottles = num_custom_bottles
        super(IupacPlusOptimizer, self).__init__(
            str_length,
            alphabet_size,
            batch_size,
            num_templates,
            device,
            uniform_template_mix,
        )

        self.num_bottles = 15 + num_custom_bottles
        self.latent_phi = nn.Parameter(
            torch.randn(num_templates, str_length, self.num_bottles, device=device)
        )
        self.latent_v_custom = nn.Parameter(
            torch.randn(num_custom_bottles, alphabet_size, device=device)
        )
        self.name = f"IUPAC+{num_custom_bottles}"

    def get_bottle_probs(self, tau: float = 1) -> torch.Tensor:
        custom_probs = torch.softmax(self.latent_v_custom / tau, dim=-1)
        return torch.cat([self._iupac_bottle_probs, custom_probs], dim=0)


class FixedIupacPolicy:
    """Non-trainable policy for evaluating a fixed IUPAC template."""

    def __init__(self, iupac_template: str, alphabet: list):
        self.iupac_template = iupac_template
        self.alphabet = alphabet
        self.str_length = len(iupac_template)
        self.alphabet_size = len(alphabet)
        self.name = "FixedIUPAC"
        self.num_templates = 1
        self.alphabet_to_idx = {base: idx for idx, base in enumerate(alphabet)}

    def sample(self, batch_size: int, tau: float = 1.0, **kwargs) -> SampleResult:
        sequences_str = generate_iupac_sequences(self.iupac_template, batch_size)

        sequences = torch.tensor(
            [[self.alphabet_to_idx[base] for base in seq] for seq in sequences_str],
            dtype=torch.long,
        )

        mixture_indices = torch.zeros(batch_size, dtype=torch.long)

        return SampleResult(
            sequences=sequences,
            mixture_indices=mixture_indices,
            bottle_indices=None,
            selected_bottles=None,
        )

    def get_probs(self, tau: float = 1.0):
        """
        Returns base probabilities from IUPAC template and mixture probabilities.
        Returns same format as ArbitraryBaseOptimizer for compatibility.

        Returns:
            template_probs: Tensor of shape (1, str_length, alphabet_size)
            mixture_probs: Tensor of shape (1,) with value 1.0
        """
        base_probs = np.zeros((self.str_length, self.alphabet_size))

        for pos, iupac_code in enumerate(self.iupac_template):
            if iupac_code in IUPAC_MAP:
                allowed_bases = IUPAC_MAP[iupac_code]
                num_allowed = len(allowed_bases)
                for base in allowed_bases:
                    if base in self.alphabet:
                        base_idx = self.alphabet.index(base)
                        base_probs[pos, base_idx] = 1.0 / num_allowed
            else:
                # Not a valid IUPAC code, treat as uniform
                base_probs[pos, :] = 1.0 / self.alphabet_size

        template_probs = torch.tensor(base_probs, dtype=torch.float32).unsqueeze(0)
        mixture_probs = torch.tensor([1.0], dtype=torch.float32)

        return template_probs, mixture_probs

    def state_dict(self):
        return {"iupac_template": self.iupac_template}

    def load_state_dict(self, state_dict):
        self.iupac_template = state_dict["iupac_template"]

    def get_template_dna_probs(self):
        """
        Get per-position DNA base probabilities for the template.

        Returns:
            numpy array of shape (1, str_length, 4) with DNA base probabilities
        """
        template_probs, _ = self.get_probs()
        return template_probs.detach().cpu().numpy()


class FixedAAPolicy:
    """Fixed single-template policy over amino acids for evaluation (non-trainable)."""

    def __init__(self, aa_probs: np.ndarray):
        """
        Args:
            aa_probs: Shape (str_length, 20) - probability of each AA at each position.
                      Must be aligned with constants.py AMINO_ACIDS ordering (indices 0-19).
        """
        if aa_probs.ndim != 2 or aa_probs.shape[1] != 20:
            raise ValueError(
                f"aa_probs must be shape (str_length, 20), got {aa_probs.shape}"
            )

        self.aa_probs = aa_probs.astype(np.float32)
        self.str_length = aa_probs.shape[0]
        self.num_templates = 1
        self.alphabet_size = 20
        self.is_aa_level = True
        self.name = "FixedAA"

    def sample(self, batch_size: int, **kwargs) -> SampleResult:
        """Sample AA indices from the probability distribution."""
        probs = torch.tensor(self.aa_probs, dtype=torch.float32)
        dist = torch.distributions.Categorical(probs=probs)
        sequences = dist.sample((batch_size,))  # (batch_size, str_length)
        mixture_indices = torch.zeros(batch_size, dtype=torch.long)
        return SampleResult(
            sequences=sequences,
            mixture_indices=mixture_indices,
            bottle_indices=None,
            selected_bottles=None,
        )

    def get_probs(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (template_probs, mixture_probs) for vis compatibility."""
        template_probs = torch.tensor(self.aa_probs, dtype=torch.float32).unsqueeze(0)
        mixture_probs = torch.tensor([1.0], dtype=torch.float32)
        return template_probs, mixture_probs

    def get_template_aa_probs(self) -> np.ndarray:
        """Return (1, str_length, 20) AA probability array."""
        return self.aa_probs[np.newaxis, :, :]


class SlotConstrainedIupacOptimizer(IupacBaseOptimizer):
    """
    IUPAC optimizer with constrained non-AGCT base usage.

    Allows unlimited use of A, G, C, T (bottles 0-3), but constrains
    non-AGCT IUPAC bases (bottles 4-14) to a maximum of max_non_agct_slots
    bottles across all templates.

    The sampling process is now:
    1. Sample which non-AGCT bottles to enable (max_non_agct_slots)
    2. Get constrained phi probabilities with only selected bottles enabled
    3. Sample templates, bottles, and characters as usual
    """

    def __init__(
        self,
        str_length,
        alphabet_size,
        batch_size,
        num_templates,
        max_non_agct_slots=4,
        device="cpu",
        uniform_template_mix=True,
    ):
        super(SlotConstrainedIupacOptimizer, self).__init__(
            str_length,
            alphabet_size,
            batch_size,
            num_templates,
            device,
            uniform_template_mix,
        )

        # Number of non-AGCT bottles (bottles 4-14 = 11 bottles: M, S, K, W, R, Y, V, H, D, B, N)
        self.num_non_agct_bottles = 11

        assert max_non_agct_slots <= self.num_non_agct_bottles, (
            f"max_non_agct_slots ({max_non_agct_slots}) cannot exceed "
            f"the number of non-AGCT bottles ({self.num_non_agct_bottles})."
        )

        # Learnable parameter: logits for selecting which non-AGCT bottles to enable
        # Shape: (num_non_agct_bottles,) = (11,)
        # We will select max_non_agct_slots bottles from these 11
        self.slot_allocation_logits = nn.Parameter(
            torch.randn(self.num_non_agct_bottles, device=device)
        )

        self.max_non_agct_slots = max_non_agct_slots
        self.name = f"SlotConstrained_IUPAC_{max_non_agct_slots}"

    def sample_greedy_slot_allocation(self):
        """
        Returns:
            Tensor of shape (max_non_agct_slots,) with bottle indices (values in range 4-14)
        """
        selected_bottle_indices = torch.topk(
            self.slot_allocation_logits, k=self.max_non_agct_slots
        ).indices
        return selected_bottle_indices + 4

    def sample_slot_allocation(self, tau=1):
        """
        Sample which non-AGCT bottles to enable using Gumbel top-k.

        Args:
            tau: Temperature parameter for sampling

        Returns:
            selected_bottles: Tensor of shape (max_non_agct_slots,) with bottle indices (values in range 4-14)
        """
        gumbel_noise = -torch.log(
            -torch.log(torch.rand_like(self.slot_allocation_logits))
        )
        perturbed_logits = self.slot_allocation_logits / tau + gumbel_noise

        selected_bottle_indices = torch.topk(
            perturbed_logits, k=self.max_non_agct_slots
        ).indices

        # Offset by 4 to get actual bottle indices (bottles 4-14)
        selected_bottles = selected_bottle_indices + 4

        return selected_bottles

    def get_constrained_phi_probs(self, selected_bottles, tau=1):
        """
        Get phi_probs with non-AGCT bottles masked except for selected bottles.

        Args:
            selected_bottles: Tensor of shape (max_non_agct_slots,) with bottle indices (values in range 4-14)
            tau: Temperature parameter

        Returns:
            phi_probs_constrained: Tensor of shape (num_templates, str_length, num_bottles)
                with non-selected bottles mask to 0
        """
        phi_probs_unconstrained = torch.softmax(self.latent_phi / tau, dim=-1)

        mask = torch.zeros_like(
            phi_probs_unconstrained
        )  # (num_templates, str_length, num_bottles)
        mask[:, :, :4] = 1.0  # Always allow AGCT (bottles 0-3)

        for bottle_idx in selected_bottles:
            mask[:, :, bottle_idx] = 1.0

        phi_probs_constrained = phi_probs_unconstrained * mask
        phi_probs_constrained = phi_probs_constrained / phi_probs_constrained.sum(
            dim=-1, keepdim=True
        )

        return phi_probs_constrained

    def get_probs(self, selected_bottles=None, tau=1):
        """
        Returns bottle probabilities, constrained phi probabilities, and mixture probabilities.

        Args:
            selected_bottles: Tensor of shape (max_non_agct_slots,) with bottle indices (values in range 4-14).
                            If None, bottles are sampled automatically.
            tau: Temperature parameter

        Returns:
            v_probs: Bottle probabilities (num_bottles, alphabet_size)
            phi_probs: Constrained bottle selection probabilities (num_templates, str_length, num_bottles)
            w_mix_probs: Template mixture probabilities (num_templates,)
        """
        if selected_bottles is None:
            selected_bottles = self.sample_slot_allocation(tau)

        v_probs = self.get_bottle_probs(tau)
        phi_probs = self.get_constrained_phi_probs(selected_bottles, tau)
        w_mix_probs = torch.softmax(self.latent_w_mix / tau, dim=0)

        return v_probs, phi_probs, w_mix_probs

    def sample_greedy_bottles(self):
        """
        Returns:
            numpy array of shape (num_templates, str_length) with bottle indices
        """
        selected_bottles = self.sample_greedy_slot_allocation()
        phi_probs = self.get_constrained_phi_probs(selected_bottles, tau=1)
        phi_probs_np = phi_probs.detach().cpu().numpy()
        return np.argmax(phi_probs_np, axis=2)

    def sample(self, batch_size=None, tau=1, greedy=False):
        """
        Sample sequences using the slot-constrained bottle-based policy.

        Args:
            greedy: If True, use argmax for slot allocation and bottle selection per template.
                    Template mixing and character sampling remain stochastic.

        Returns:
            SampleResult with sequences, mixture_indices, bottle_indices, and selected_bottles
        """
        if batch_size is None:
            batch_size = self.batch_size

        with torch.no_grad():
            if greedy:
                selected_bottles = self.sample_greedy_slot_allocation()
            else:
                selected_bottles = self.sample_slot_allocation(tau)

            v_probs = self.get_bottle_probs(tau)
            phi_probs = self.get_constrained_phi_probs(selected_bottles, tau)
            w_mix_probs = torch.softmax(self.latent_w_mix / tau, dim=0)

            mixture_dist = torch.distributions.Categorical(probs=w_mix_probs)
            mixture_indices = mixture_dist.sample(sample_shape=(batch_size,))

            unique_templates, positions = torch.unique(
                mixture_indices, return_inverse=True
            )

            if greedy:
                greedy_bottles = self.sample_greedy_bottles()
                bottles_for_unique = torch.from_numpy(
                    greedy_bottles[unique_templates.cpu().numpy()]
                ).to(self.device)
            else:
                phi_probs_for_unique = phi_probs[unique_templates]
                bottle_dist = torch.distributions.Categorical(
                    probs=phi_probs_for_unique
                )
                bottles_for_unique = bottle_dist.sample()  # (num_unique, str_length)

            bottle_indices = bottles_for_unique[positions]  # (batch_size, str_length)

            v_probs_for_batch = v_probs[bottle_indices]
            char_dist = torch.distributions.Categorical(probs=v_probs_for_batch)
            sequences = char_dist.sample()  # (batch_size, str_length)

            return SampleResult(
                sequences=sequences,
                mixture_indices=mixture_indices,
                bottle_indices=bottle_indices,
                selected_bottles=selected_bottles,
            )

    def get_batch_log_probs(self, sample_result: SampleResult):
        """
        Compute batch-level log probabilities (slot allocation and bottle selection per template).

        Returns:
            Scalar tensor with log p(slot allocation) + log p(bottle config per template)
        """
        mixture_indices = sample_result.mixture_indices
        bottle_indices = sample_result.bottle_indices
        selected_bottles = sample_result.selected_bottles
        assert (
            bottle_indices is not None
        ), "SlotConstrainedIupacOptimizer requires bottle_indices"
        assert (
            selected_bottles is not None
        ), "SlotConstrainedIupacOptimizer requires selected_bottles"

        _, phi_probs, _ = self.get_probs(selected_bottles, tau=1)

        # Slot allocation probability
        bottle_selection_probs = torch.softmax(self.slot_allocation_logits, dim=0)
        selected_indices = selected_bottles - 4
        log_prob_bottles = torch.log(bottle_selection_probs[selected_indices]).sum()

        # Bottle selection per template probability
        unique_templates = torch.unique(mixture_indices)
        first_occurrence_unique = self._find_first_occurrence_indices(
            mixture_indices, unique_templates
        )
        bottles_for_unique = bottle_indices[first_occurrence_unique]
        phi_probs_for_unique = phi_probs[unique_templates]
        gathered_phi_probs = phi_probs_for_unique.gather(
            2, bottles_for_unique.unsqueeze(-1)
        ).squeeze(-1)
        log_prob_phi_total = torch.log(gathered_phi_probs).sum()

        return log_prob_bottles + log_prob_phi_total

    def get_log_probs(self, sample_result: SampleResult):
        """
        Compute per-sequence log probabilities (template and character selection).

        Returns:
            log_probs: Tensor of shape (batch_size,) with log p(s_i | t_i, b_{t_i}) + log p(t_i)
        """
        sequences = sample_result.sequences
        mixture_indices = sample_result.mixture_indices
        bottle_indices = sample_result.bottle_indices
        selected_bottles = sample_result.selected_bottles
        assert (
            bottle_indices is not None
        ), "SlotConstrainedIupacOptimizer requires bottle_indices"
        assert (
            selected_bottles is not None
        ), "SlotConstrainedIupacOptimizer requires selected_bottles"

        v_probs, _, w_mix_probs = self.get_probs(selected_bottles, tau=1)

        log_prob_mix = torch.log(w_mix_probs[mixture_indices])

        v_probs_for_batch = v_probs[bottle_indices]
        gathered_v_probs = v_probs_for_batch.gather(2, sequences.unsqueeze(-1)).squeeze(
            -1
        )
        log_prob_v = torch.log(gathered_v_probs).sum(dim=1)

        return log_prob_mix + log_prob_v
