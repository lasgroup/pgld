"""
entropy.py

Defines entropy variations for policy training.
"""

from abc import ABC
import logging

import numpy as np
import torch
import torch.nn as nn

from constants import (
    AMINO_ACID_ALPHABET,
    AMINO_ACIDS,
    CODON_TABLE,
    DNA_BASES,
    IUPAC_BOTTLES,
    MAX_AA_INDEX,
)
from landscapes import dna_to_protein
from policies import (
    ArbitraryBaseOptimizer,
    BottleBasedOptimizer,
    FiniteBaseOptimizer,
    SampleResult,
)
from dataclasses import dataclass
from typing import Union
from tensor_translation_gpu import (
    translate_dna_to_aa_indices_gpu,
    build_codon_lookup_gpu,
    dna_probs_to_aa_probs,
)

log = logging.getLogger(__name__)


def _get_probs_for_all_templates(
    valid_sample_result, policy, peptide_level: bool = False
) -> torch.Tensor:
    """
    Calculate log probabilities of sampling a sequence from a template for all sequences and unique templates in a valid sample result.
    Valid sample result is a sample result already filtered for valid peptides.

    Args:
        valid_sample_result, policy, peptide_level
    Returns:
        tensor of shape (unique_templates, unique_sequences)
    """
    uq_templates = torch.unique(valid_sample_result.mixture_indices)
    if hasattr(policy, "get_bottle_probs"):
        # Bottle-based policy: construct template probs from sampled bottles
        bottle_probs = policy.get_bottle_probs()  # (num_bottles, alphabet_size)
        first_idx = policy._find_first_occurrence_indices(
            valid_sample_result.mixture_indices, uq_templates
        )
        template_bottles = valid_sample_result.bottle_indices[first_idx]  # (T, L)

        template_probs = bottle_probs[template_bottles.view(-1)].reshape(
            *template_bottles.shape, -1
        )  # (T, L, alphabet_size)

    else:
        # Arbitrary policy: directly get template probs
        template_probs = policy.get_template_probs()[
            uq_templates
        ]  # (T, L, alphabet_size)

    N = valid_sample_result.sequences.shape[0]
    T = len(uq_templates)

    if peptide_level:
        num_codons = template_probs.shape[1] // 3
        log_template_probs = torch.log(
            torch.stack([dna_probs_to_aa_probs(tp) for tp in template_probs], dim=0)
        ).clip(
            min=-1e8
        )  # (T, num_codons, 20)

        dna_seqs = valid_sample_result.sequences  # (N, L)
        codon_lookup = build_codon_lookup_gpu(dna_seqs.device)
        aa_seqs, stop_codon_idx = translate_dna_to_aa_indices_gpu(
            dna_seqs.clone(), codon_lookup
        )  # aa_seqs: (N, num_codons), stop_codon_idx: (N,)

        positions = torch.arange(num_codons, device=dna_seqs.device).unsqueeze(
            0
        )  # (1, num_codons)
        stops = stop_codon_idx.unsqueeze(1)  # (N, 1)
        valid_mask = (positions < stops) | (stops == -1)  # (N, num_codons)

        sequences = aa_seqs.clamp(max=MAX_AA_INDEX).long()  # (N, num_codons)

    else:
        log_template_probs = torch.log(template_probs).clip(
            min=-1e8
        )  # (T, L, alphabet_size)
        sequences = valid_sample_result.sequences  # (N, L)

        # In the DNA case, there is no stop codon so all positions are valid
        valid_mask = torch.ones(
            N, sequences.shape[1], dtype=torch.bool, device=sequences.device
        )  # (N, L)

    # Gather log probs at sequence positions for all templates
    log_template_probs_exp = log_template_probs.unsqueeze(1).expand(
        -1, N, -1, -1
    )  # (T, N, L, alphabet)
    sequences_exp = (
        sequences.unsqueeze(0).expand(T, -1, -1).unsqueeze(-1)
    )  # (T, N, L, 1)
    log_gathered_probs = log_template_probs_exp.gather(3, sequences_exp).squeeze(
        -1
    )  # (T, N, L)

    # Mask out positions after stop codon
    valid_mask_exp = valid_mask.unsqueeze(0)  # (1, N, L)
    log_gathered_probs = torch.where(
        valid_mask_exp,
        log_gathered_probs,
        torch.zeros_like(log_gathered_probs),
    )

    log_probs_per_template = log_gathered_probs.sum(dim=-1)  # (T, N)

    return log_probs_per_template


def calculate_entropy(policy, tau=1):
    """
    Computes exact Shannon entropy of a policy.
    Only valid for num_templates = 1 (single template, no mixture).

    For ArbitraryBaseOptimizer with M=1:
        H = -Σ_{l,a} p(a|l) log p(a|l)
    """

    if policy.num_templates != 1:
        raise ValueError(
            f"calculate_entropy() only works for num_templates=1, "
            f"but policy has num_templates={policy.num_templates}. "
            f"Use calculate_entropy_gradient_estimate() for M>1 or set entropy_type='gradient_estimate'."
        )

    assert isinstance(
        policy, ArbitraryBaseOptimizer
    ), "calculate_entropy only supports ArbitraryBaseOptimizer with M=1."

    # Note: If you want M=1 for FiniteBaseOptimizer, then need to sample bottles, compute
    # exact entropy given bottle contents, and then do policy grads trick because you can't
    # backprop though sampling bottles (same as we do for eg unique hits)

    template_probs = policy.get_template_probs(tau)
    entropy_terms = -template_probs * torch.log(template_probs)
    return entropy_terms.sum()


def calculate_entropy_exact_bottle_based(
    policy, sample_result, peptide_level: bool = False, baseline: float = 0.0
):
    assert isinstance(
        policy, BottleBasedOptimizer
    ), "calculate_entropy_exact_bottle_based only supports BottleBasedOptimizer"
    if policy.num_templates != 1:
        # log.info(f"Using non-rigorous exact entropy calculation for M={policy.num_templates}")
        return calculate_entropy_per_template(policy, sample_result, peptide_level)

    batch_log_probs = policy.get_batch_log_probs(sample_result)
    bottle_probs = policy.get_bottle_probs()
    template_probs = bottle_probs[
        sample_result.bottle_indices[0]
    ]  # (str_length, alphabet_size)
    if peptide_level:
        # Convert to amino acid probs
        template_probs = dna_probs_to_aa_probs(template_probs)
    entropy_batch = -torch.sum(
        template_probs * torch.log(template_probs).clip(min=-1e8)
    )
    return (
        batch_log_probs * (entropy_batch.detach() - baseline)
        + (entropy_batch - baseline),
        entropy_batch.detach().cpu().item(),
    )


def calculate_entropy_per_template(
    policy,
    sample_result,
    peptide_level: bool = False,
    baseline: float = 0.0,
    n_lowest=20,
):
    n_lowest = min(n_lowest, policy.num_templates)
    batch_log_probs = policy.get_batch_log_probs(sample_result)
    bottle_probs = policy.get_bottle_probs()
    # get sampled bottle configuration of each template in current batch
    first_idx = policy._find_first_occurrence_indices(
        sample_result.mixture_indices, torch.unique(sample_result.mixture_indices)
    )
    template_bottles = sample_result.bottle_indices[first_idx]
    # assemble base probs at each position of each template
    template_probs = bottle_probs[template_bottles.view(-1)].reshape(
        *template_bottles.shape, -1
    )  # (num_templates, str_length, alphabet_size)
    if peptide_level:
        # Convert to amino acid probs
        template_probs = torch.stack(
            [dna_probs_to_aa_probs(template) for template in template_probs], dim=0
        )
    # batch reward is average of n_lowest templates with lowest entropy
    entropy_batch = -torch.sum(
        template_probs * torch.log(template_probs).clip(min=-1e8), dim=(1, 2)
    )
    entropy_batch = torch.mean(entropy_batch[torch.argsort(entropy_batch)[:n_lowest]])
    return (
        batch_log_probs * (entropy_batch.detach() - baseline)
        + entropy_batch
        - baseline,
        entropy_batch.detach().cpu().item(),
    )


def _setup_template_entropy_computation(
    policy,
    landscape,
    sample_result,
    peptide_level: bool = False,
) -> Union[dict, None]:
    """
    Common setup for equal share and marginal gain entropy calculations.

    Returns dict with: log_probs_cond, mixture_indices, uq_templates, template_cts,
    w_mix_probs, log_joint, batch_log_probs, N, sample_result.
    Returns None if no valid peptides (when peptide_level=True).
    """
    if policy.num_templates == 1:
        raise ValueError("Cannot compute template entropy for M=1")
    if not isinstance(policy, BottleBasedOptimizer):
        raise NotImplementedError("Template entropy only supports BottleBasedOptimizer")

    v_probs, _, w_mix_probs = (
        policy.get_probs()
    )  # v: (num_bottles, 4), w: (num_templates,)

    if peptide_level:
        peptides = []
        valid_peptides = []
        for i, seq in enumerate(sample_result.sequences):
            if hasattr(seq, "cpu"):
                seq = seq.cpu()
            dna_seq = "".join([landscape.alphabet[int(c)] for c in seq])
            peptide = dna_to_protein(dna_seq)
            if peptide:
                peptides.append(peptide)
                valid_peptides.append(i)
        if len(peptides) == 0:
            return None

        N = len(peptides)
        mixture_indices = sample_result.mixture_indices[valid_peptides]  # (N,)
        bottle_indices = sample_result.bottle_indices[valid_peptides]  # (N, L)
        uq_templates, template_cts = torch.unique(
            mixture_indices, return_counts=True
        )  # (T,), (T,)
        first_idx = policy._find_first_occurrence_indices(mixture_indices, uq_templates)
        template_bottles = bottle_indices[first_idx]  # (T, L)
        bottle_probs = policy.get_bottle_probs()  # (num_bottles, 4)
        template_dna_probs = bottle_probs[template_bottles.view(-1)].reshape(
            *template_bottles.shape, -1
        )  # (T, L, 4)
        log_template_aa_probs = torch.log(
            torch.stack([dna_probs_to_aa_probs(tp) for tp in template_dna_probs], dim=0)
        ).clip(
            min=-1e8
        )  # (T, num_codons, 20)

        # Map original template indices to 0..T-1 for indexing
        shifted_mix_idx = torch.zeros_like(mixture_indices)  # (N,)
        for i_new, i_old in enumerate(uq_templates):
            shifted_mix_idx[mixture_indices == i_old] = i_new

        num_codons = log_template_aa_probs.shape[1]
        peptide_int_enc = torch.tensor(
            [
                [AMINO_ACID_ALPHABET[aa] for aa in seq] + [-1] * (num_codons - len(seq))
                for seq in peptides
            ],
            dtype=int,
            device=policy.device,
        )  # (N, num_codons)

        # Gather log probs for each sequence from its template
        log_probs_cond = log_template_aa_probs[
            shifted_mix_idx[:, None].expand(-1, num_codons),  # (N, num_codons)
            torch.arange(num_codons, dtype=int).expand(N, -1),  # (N, num_codons)
            peptide_int_enc,  # (N, num_codons)
        ]  # (N, num_codons)
        log_probs_cond = torch.sum(
            log_probs_cond * (peptide_int_enc != -1), dim=-1
        )  # (N,)
        sample_result.mixture_indices = mixture_indices
        sample_result.bottle_indices = bottle_indices
    else:
        mixture_indices = sample_result.mixture_indices  # (N,)
        v_probs_for_batch = v_probs[sample_result.bottle_indices]  # (N, L, 4)
        gathered_v_probs = v_probs_for_batch.gather(
            2, sample_result.sequences.unsqueeze(-1)
        ).squeeze(
            -1
        )  # (N, L)
        log_probs_cond = torch.log(gathered_v_probs).sum(dim=1)  # (N,)
        uq_templates, template_cts = torch.unique(
            mixture_indices, return_counts=True
        )  # (T,), (T,)

    N = len(mixture_indices)
    log_prob_mix = torch.log(w_mix_probs[mixture_indices])  # (N,)
    batch_log_probs = policy.get_batch_log_probs(sample_result, sum=False)  # (N,)
    log_joint = log_probs_cond + log_prob_mix  # (N,)

    return {
        "log_probs_cond": log_probs_cond,  # (N,)
        "mixture_indices": mixture_indices,  # (N,)
        "uq_templates": uq_templates,  # (T,)
        "template_cts": template_cts,  # (T,)
        "w_mix_probs": w_mix_probs,  # (num_templates,)
        "log_joint": log_joint,  # (N,)
        "batch_log_probs": batch_log_probs,  # (N,)
        "N": N,
        "sample_result": sample_result,
    }


def calculate_equal_share_entropy(
    policy,
    landscape,
    sample_result,
    peptide_level: bool = False,
    baseline=0.0,
):
    """
    Calculate equal share entropy: contribution from each template's samples.
    Returns tuple of (reinforce loss contribution, baseline update value).
    """
    setup = _setup_template_entropy_computation(
        policy, landscape, sample_result, peptide_level
    )
    if setup is None:
        return torch.tensor(0.0, device=policy.device, requires_grad=True), 0.0

    w_mix_probs_uq = setup["w_mix_probs"][setup["uq_templates"]]
    equal_share = -torch.stack(
        [
            torch.sum(setup["log_probs_cond"][setup["mixture_indices"] == t])
            for t in setup["uq_templates"]
        ]
    )
    equal_share -= setup["template_cts"] * torch.log(w_mix_probs_uq)
    equal_share /= setup["N"]

    advantage = equal_share - baseline
    return (
        torch.sum(setup["batch_log_probs"] * advantage.detach() + advantage),
        torch.mean(equal_share).detach().cpu().item(),
    )


def calculate_marginal_entropy_gain(
    policy,
    landscape,
    sample_result,
    peptide_level: bool = False,
    baseline=0.0,
):
    """
    Calculate marginal entropy gain: H(full mixture) - H(mixture without template t).
    Returns tuple of (reinforce loss contribution, baseline update value).
    """
    raise ValueError(
        "Standalone calculate_marginal_entropy_gain is deprecated. "
        "Use MarginalEntropyGain.compute_with_cache() instead."
    )


def calculate_entropy_mc_bottle_based(
    policy,
    landscape,
    sample_result,
    log_probs: torch.Tensor,
    baseline: float = 0.0,
    peptide_level: bool = False,
):
    """Calculate the per-mc-sample contribution to the entropy for bottle-based optimizers"""
    assert isinstance(
        policy, BottleBasedOptimizer
    ), "calculate_entropy_mc_bottle_based only supports BottleBasedOptimizer."

    batch_log_probs = policy.get_batch_log_probs(sample_result)
    # get mc estimate of the entropy for the batch
    if peptide_level:
        # translate integer base to integer peptide encoding
        peptides = []
        valid_peptides = []
        for i, seq in enumerate(sample_result.sequences):
            if hasattr(seq, "cpu"):
                seq = seq.cpu()
            dna_seq = "".join([landscape.alphabet[int(c)] for c in seq])
            peptide = dna_to_protein(dna_seq)
            if peptide:
                peptides.append(peptide)
                valid_peptides.append(i)
        if len(peptides) == 0:
            return torch.tensor(0.0, device=policy.device, requires_grad=True)

        mixture_indices = sample_result.mixture_indices[valid_peptides]
        bottle_indices = sample_result.bottle_indices[valid_peptides]
        v_probs, _, w_mix_probs = policy.get_probs()
        log_prob_mix = torch.log(w_mix_probs[mixture_indices])
        # assemble base probs at each position of each template
        uq_templates = torch.unique(mixture_indices)
        first_idx = policy._find_first_occurrence_indices(mixture_indices, uq_templates)
        template_bottles = bottle_indices[first_idx]
        bottle_probs = policy.get_bottle_probs()
        template_probs = bottle_probs[template_bottles.view(-1)].reshape(
            *template_bottles.shape, -1
        )  # (num_templates, str_length, alphabet_size)
        # convert to residue-level probs
        template_probs = torch.log(
            torch.stack([dna_probs_to_aa_probs(tp) for tp in template_probs], dim=0)
        ).clip(min=-1e8)
        # some templates may not have been sampled so we need to make sure that the indices are correct
        shifted_mix_idx = torch.zeros_like(mixture_indices)
        for i_new, i_old in enumerate(uq_templates):
            shifted_mix_idx[mixture_indices == i_old] = i_new

        # compute log probs of peptides
        max_len = template_probs.shape[1]
        peptide_int_enc = torch.tensor(
            [
                [AMINO_ACID_ALPHABET[aa] for aa in seq] + [-1] * (max_len - len(seq))
                for seq in peptides
            ],
            dtype=int,
            device=policy.device,
        )  # (n_sequences, max_len)
        log_probs = template_probs[
            shifted_mix_idx[:, None].repeat(1, max_len),
            torch.arange(max_len, dtype=int).repeat(len(peptides), 1),
            peptide_int_enc,
        ]
        # filter out padded positions at the end
        log_probs = torch.sum(
            log_probs * (peptide_int_enc != -1), dim=-1
        )  # p(seq | template, bottles)
        log_probs += log_prob_mix

    entropy_batch = -torch.mean(log_probs, dim=-1)
    return (
        batch_log_probs * (entropy_batch.detach() - baseline)
        + entropy_batch
        - baseline,
        entropy_batch.detach().cpu().item(),
    )


def calculate_entropy_gradient_estimate(log_probs):
    """
    Computes a gradient estimate for entropy regularization using sampled sequences.
    Works for any number of templates.

    Implements the REINFORCE gradient estimator:
        ∇H ≈ -E_s~π[∇ log π(s) · (log π(s) + 1)]

    where s are sampled sequences and π is the policy.

    Args:
        log_probs: Log probabilities of sampled sequences, shape [batch_size]
                   Should have gradients attached.
    """
    return -torch.mean(log_probs * (log_probs.detach() + 1))


def calculate_peptide_entropy(policy, tau=1):
    """
    Computes exact Shannon entropy over peptide sequences (amino acid alphabet).
    Only valid for num_templates = 1 (single template, no mixture).

    Same as calculate_entropy but converts DNA probs to amino acid probs first.

    Args:
        policy: The policy (must be ArbitraryBaseOptimizer with num_templates=1)
        tau: Temperature parameter (default=1)

    Returns:
        torch.Tensor: Shannon entropy over peptide sequences
    """
    if policy.num_templates != 1:
        raise ValueError(
            f"calculate_peptide_entropy() only works for num_templates=1, "
            f"but policy has num_templates={policy.num_templates}. "
            f"Use calculate_peptide_entropy_gradient_estimate() for M>1."
        )

    assert isinstance(
        policy, ArbitraryBaseOptimizer
    ), "calculate_peptide_entropy only supports ArbitraryBaseOptimizer with M=1."

    # Get DNA probs: shape (num_templates, seq_len, 4), squeeze to (seq_len, 4)
    dna_probs = policy.get_template_probs(tau)[0]

    # Convert to amino acid probs
    aa_probs = dna_probs_to_aa_probs(dna_probs)

    # Same entropy formula as DNA version
    entropy_terms = -aa_probs * torch.log(aa_probs + 1e-8)
    return entropy_terms.sum()


def calculate_peptide_entropy_gradient_estimate(
    landscape, policy, sample_result: SampleResult
):
    """
    Computes gradient estimate for peptide entropy using REINFORCE.
    Works for any number of templates.

    Filters out sequences with stop codons, then applies same formula as DNA version.

    Args:
        landscape: The landscape (for alphabet)
        policy: The policy from which sequences were sampled
        sample_result: Holds the sequences, template indices and bottle indices

    Returns:
        torch.Tensor: Gradient estimate for peptide entropy regularization
    """
    # Filter to valid peptides only
    peptides = []
    valid_peptides = []
    for i, seq in enumerate(sample_result.sequences):
        if hasattr(seq, "cpu"):
            seq = seq.cpu()
        dna_seq = "".join([landscape.alphabet[int(c)] for c in seq])
        peptide = dna_to_protein(dna_seq)
        if peptide:
            peptides.append(peptide)
            valid_peptides.append(i)

    if len(peptides) == 0:
        return torch.tensor(0.0, device=policy.device, requires_grad=True)

    # update result
    sample_result.sequences = sample_result.sequences[valid_peptides]
    sample_result.mixture_indices = sample_result.mixture_indices[valid_peptides]
    # if a template was never sampled we need to adjust the indices
    uq_mixture_indices, mixture_indices = torch.unique(
        sample_result.mixture_indices, return_inverse=True
    )

    # get amino acid probs at each position of each template
    if policy.name == "Arbitrary":
        dna_probs = policy.get_template_probs()
        _, mixture_probs = policy.get_probs()
    else:
        sample_result.bottle_indices = sample_result.bottle_indices[valid_peptides]
        _, phi_probs, mixture_probs = policy.get_probs()
        # get base probabilities of all bottles
        bottle_probs = policy.get_bottle_probs()  # (num_bottles, alphabet_size)
        # get sampled bottle configuration of each template in current batch
        first_idx = policy._find_first_occurrence_indices(
            sample_result.mixture_indices, uq_mixture_indices
        )
        template_bottles = sample_result.bottle_indices[first_idx]
        # assemble base probs at each position of each template
        dna_probs = bottle_probs[template_bottles.view(-1)].reshape(
            *template_bottles.shape, -1
        )  # (num_templates, str_length, alphabet_size)

    aa_probs = torch.log(
        torch.stack(
            [dna_probs_to_aa_probs(temp_dna_probs) for temp_dna_probs in dna_probs],
            dim=0,
        )
    ).clip(
        min=-1e8
    )  # (num_templates, str_length // 3, 20)

    # compute log probs of peptides
    max_len = aa_probs.shape[1]
    peptide_int_enc = torch.tensor(
        [
            [AMINO_ACID_ALPHABET[aa] for aa in seq] + [-1] * (max_len - len(seq))
            for seq in peptides
        ],
        dtype=int,
        device=policy.device,
    )  # (n_sequences, max_len)
    log_probs_cond = aa_probs[
        mixture_indices[:, None].repeat(1, max_len),
        torch.arange(max_len, dtype=int).repeat(len(peptides), 1),
        peptide_int_enc,
    ]
    # filter out padded positions at the end
    log_probs_cond = torch.sum(
        log_probs_cond * (peptide_int_enc != -1), dim=-1
    )  # p(seq | template, bottles)
    log_probs_template = torch.log(
        mixture_probs[sample_result.mixture_indices]
    )  # p(template)

    # conditional entropy H(seq | template, bottles)
    return -torch.mean(
        log_probs_cond * (log_probs_cond.detach() + 1)
        + log_probs_template * log_probs_cond.detach()
    )


class EntropyFunction(ABC):
    """
    Base class for entropy functions.
    """

    def __init__(self, peptide_level: bool = False):
        self.peptide_level = peptide_level
        self.name = "Entropy"
        self.allowed_policies = []
        self.single_template = False
        self.multi_template = False

    def is_valid_policy(self, policy):
        valid_M = (policy.num_templates == 1 and self.single_template) or (
            policy.num_templates > 1 and self.multi_template
        )
        return (
            np.any(
                [isinstance(policy, policy_cls) for policy_cls in self.allowed_policies]
            )
            and valid_M
        )

    def __call__(self, policy, landscape, sample_result, log_probs, baseline):
        """
        Contribution of one batch (single MC sample) to the entropy
        """
        raise NotImplementedError

    def compute_with_cache(self, policy, batch_info, baseline):
        """
        Compute entropy for all MC samples and return the baseline update value
        Takes the average across MC samples if the computation is performed on the batch level.
        """
        raise NotImplementedError


class ShannonExactEntropy(EntropyFunction):
    """
    Exact Shannon's entropy
    """

    def __init__(self, peptide_level: bool = False):
        super(ShannonExactEntropy, self).__init__(peptide_level)

        self.name = "ShannonEntropy"
        self.single_template = True
        self.multi_template = False
        self.allowed_policies = [
            ArbitraryBaseOptimizer,
            BottleBasedOptimizer,
        ]

    def __call__(self, policy, landscape, sample_result, log_probs=None, baseline=None):
        if isinstance(policy, ArbitraryBaseOptimizer):
            baseline = torch.tensor(0.0, device=policy.device)
            if self.peptide_level:
                entropy = calculate_peptide_entropy(policy)
            else:
                entropy = calculate_entropy(policy)
        else:
            if baseline is None:
                baseline = torch.tensor(0.0, device=policy.device)
            entropy, baseline = calculate_entropy_exact_bottle_based(
                policy, sample_result, self.peptide_level, baseline
            )
        return entropy, baseline

    def compute_with_cache(
        self,
        policy,
        batch_info,
        baseline=None,
    ):
        return self(
            policy,
            landscape=None,
            sample_result=batch_info["sample_result"],
            baseline=baseline,
        )


class GradientEstimateEntropy(EntropyFunction):
    """
    REINFORCE estimate of the entropy.

    For bottle-based optimizers, calculates the bottle-conditioned entropy H(S|b)
    """

    def __init__(
        self, peptide_level: bool = False, baseline_per_template: bool = False
    ):
        super(GradientEstimateEntropy, self).__init__(peptide_level)

        self.name = "GradientEstimateEntropy"
        self.single_template = True
        self.multi_template = True
        self.allowed_policies = [ArbitraryBaseOptimizer, BottleBasedOptimizer]
        self.baseline_per_template = baseline_per_template

    def __call__(self, policy, landscape, sample_result, log_probs, baseline):

        if self.peptide_level:
            entropy = calculate_peptide_entropy_gradient_estimate(
                landscape, policy, sample_result
            )
        else:
            entropy = -torch.mean(log_probs * (log_probs.detach() + 1))
        return entropy, torch.tensor(0.0, device=policy.device)

    def compute_with_cache(
        self,
        policy,
        batch_info,
        baseline,
    ):

        valid_peptides = batch_info["valid_peptides"]
        sample_result = batch_info["sample_result"]
        log_probs = batch_info["cond_log_probs"][valid_peptides]

        mixture_indices = sample_result.mixture_indices[valid_peptides]

        if self.baseline_per_template:
            advantage = log_probs - baseline[mixture_indices]
        else:
            advantage = log_probs - baseline

        entropy = -torch.mean(log_probs * advantage.detach())

        if self.baseline_per_template:
            baseline_update = torch.zeros(policy.num_templates, device=policy.device)
            baseline_update.scatter_add_(
                dim=0, index=mixture_indices, src=log_probs.detach()
            )
            baseline_update /= torch.bincount(mixture_indices)
        else:
            baseline_update = log_probs.mean().detach()

        return entropy, baseline_update


@dataclass
class MixtureEntropyCache:
    """Cache of precomputed values for mixture entropy calculations."""

    H_full: torch.Tensor  # MC estimate of H(S)
    batch_log_probs: torch.Tensor  # (N,)
    uq_templates: torch.Tensor  # (T,)
    template_cts: torch.Tensor  # (T,)
    mixture_indices: torch.Tensor  # (N,)
    w_mix_probs: torch.Tensor  # (num_templates,)
    log_joint: torch.Tensor  # (T, N) - log p(s,t) for marginal gain computation


def _compute_mixture_entropy_cache(
    policy,
    batch_info: dict,
    peptide_level: bool,
) -> MixtureEntropyCache:
    """
    Compute full mixture entropy H(S) where p(s) = sum_t p(t) p(s|t).

    Args:
        policy: The policy
        batch_info: Dict with sample_result and valid_peptides
        peptide_level: Whether to compute at peptide level

    Returns:
        MixtureEntropyCache with H_full and intermediate values for marginal gain
    """
    sample_result = batch_info["sample_result"]

    if peptide_level:
        valid_peptides = batch_info["valid_peptides"]
    else:
        valid_peptides = torch.ones(
            len(sample_result.sequences), dtype=bool, device=policy.device
        )

    mixture_indices = sample_result.mixture_indices[valid_peptides]
    sample_result_valid = SampleResult(
        sample_result.sequences[valid_peptides],
        mixture_indices,
        sample_result.bottle_indices[valid_peptides],
    )

    uq_templates, template_cts = torch.unique(mixture_indices, return_counts=True)
    _, __, w_mix_probs = policy.get_probs()

    # log p(s|t) for all templates and sequences
    all_templates_log_probs = _get_probs_for_all_templates(
        sample_result_valid, policy, peptide_level
    )  # (T, N)

    # log p(s,t) = log p(s|t) + log p(t)
    log_prob_mix = torch.log(w_mix_probs[uq_templates]).unsqueeze(1)  # (T, 1)
    log_joint = all_templates_log_probs + log_prob_mix  # (T, N)

    # log p(s) = logsumexp_t log p(s,t)
    log_marginal = torch.logsumexp(log_joint, dim=0)  # (N,)

    N = len(mixture_indices)
    H_full = -(1 / N) * torch.sum(log_marginal)

    batch_log_probs = policy.get_batch_log_probs(sample_result_valid, sum=False)

    return MixtureEntropyCache(
        H_full=H_full,
        batch_log_probs=batch_log_probs,
        uq_templates=uq_templates,
        template_cts=template_cts,
        mixture_indices=mixture_indices,
        w_mix_probs=w_mix_probs,
        log_joint=log_joint,
    )


class MarginalEntropyGain(EntropyFunction):
    """
    Marginal entropy gain: H(full mixture) - H(mixture without template t).
    """

    # TODO: in this case we only calculate the marginal gain over templates that were sampled and not all templates

    def __init__(
        self, peptide_level: bool = False, baseline_per_template: bool = False
    ):
        super(MarginalEntropyGain, self).__init__(peptide_level)

        self.name = "MarginalEntropyGain"
        self.single_template = False
        self.multi_template = True
        self.allowed_policies = [BottleBasedOptimizer]
        self.baseline_per_template = baseline_per_template

    def __call__(self, policy, landscape, sample_result, log_probs, baseline):
        return calculate_marginal_entropy_gain(
            policy, landscape, sample_result, self.peptide_level, baseline
        )

    def compute_with_cache(
        self,
        policy,
        batch_info,
        baseline,
    ):
        cache = _compute_mixture_entropy_cache(policy, batch_info, self.peptide_level)

        # Compute marginal gain for each template: H_full - H_without_t
        N = len(cache.mixture_indices)
        marginal_gain = torch.zeros(len(cache.uq_templates), device=policy.device)
        for t_idx, t in enumerate(cache.uq_templates):
            N_t = cache.template_cts[t_idx]
            mask_not_t = cache.mixture_indices != t
            template_mask = (
                torch.arange(len(cache.uq_templates), device=policy.device) != t_idx
            )

            log_renorm = torch.log(1 - cache.w_mix_probs[t])
            log_marginal_without_t = torch.logsumexp(
                cache.log_joint[template_mask, :][:, mask_not_t] - log_renorm, dim=0
            )

            H_without_t = -(1 / (N - N_t)) * torch.sum(log_marginal_without_t)
            marginal_gain[t_idx] = cache.H_full - H_without_t

        if self.baseline_per_template:
            advantage = marginal_gain - baseline[cache.uq_templates]
            baseline_update = torch.zeros(policy.num_templates, device=policy.device)
            baseline_update[cache.uq_templates] = marginal_gain.detach()
        else:
            advantage = marginal_gain - baseline
            baseline_update = torch.mean(marginal_gain).detach()

        return (
            torch.sum(cache.batch_log_probs * advantage.detach() + advantage),
            baseline_update,
        )


def get_entropy_function(
    entropy_type: str, sequence_type: str, policy, baseline_per_template: bool = False
):
    peptide_level = sequence_type == "peptide"
    if entropy_type == "exact":
        entropy_function = ShannonExactEntropy(peptide_level=peptide_level)
        log.info(f"Using ShannonExactEntropy")
    elif entropy_type == "gradient_estimate":
        entropy_function = GradientEstimateEntropy(
            peptide_level=peptide_level, baseline_per_template=baseline_per_template
        )
        log.info(f"Using GradientEstimateEntropy")
    elif entropy_type == "marginal_gain":
        entropy_function = MarginalEntropyGain(
            peptide_level=peptide_level, baseline_per_template=baseline_per_template
        )
        log.info(f"Using MarginalEntropyGain")
    else:
        raise ValueError(f"Unknown entropy type {entropy_type}")

    if not entropy_function.is_valid_policy(policy):
        raise ValueError(
            f"{policy.name} policy with {policy.num_templates} template(s) not compatible "
            f"with {entropy_function.name} (single_template={entropy_function.single_template}, "
            f"multi_template={entropy_function.multi_template})"
        )

    return entropy_function
