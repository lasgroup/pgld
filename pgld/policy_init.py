"""Policy initialization utilities."""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from constants import AA_TO_CODONS, DNA_ALPHABET, IUPAC_MAP, IUPAC_TO_BOTTLE_IDX
from plm_utils import load_fasta


def init_from_iupac_sequence(policy, iupac_sequence: str, epsilon: float = 0.1):
    """
    Initialize policy templates from an IUPAC sequence with noise.

    Args:
        policy: Policy instance (any policy with latent_phi or latent_w)
        iupac_sequence: IUPAC sequence string (e.g., "NNNNNNN...")
        epsilon: Probability of random base/bottle instead of correct one (noise level)
    """
    if len(iupac_sequence) != policy.str_length:
        raise ValueError(
            f"IUPAC sequence length {len(iupac_sequence)} doesn't match policy str_length {policy.str_length}"
        )

    if hasattr(policy, "latent_phi"):
        _init_bottle_based_policy(policy, iupac_sequence, epsilon)
    elif hasattr(policy, "latent_w"):
        _init_arbitrary_base_policy(policy, iupac_sequence, epsilon)
    else:
        raise ValueError(
            f"Don't know how to initialize {type(policy).__name__}. "
            f"Policy must have either 'latent_phi' (bottle-based) or 'latent_w' (arbitrary base) attribute."
        )


def _init_arbitrary_base_policy(policy, iupac_sequence: str, epsilon: float):
    """Initialize ArbitraryBaseOptimizer from IUPAC sequence."""
    for template_idx in range(policy.num_templates):
        for pos_idx, iupac_base in enumerate(iupac_sequence):
            if iupac_base not in IUPAC_MAP:
                raise ValueError(
                    f"Invalid IUPAC base '{iupac_base}' at position {pos_idx}"
                )

            allowed_bases = IUPAC_MAP[iupac_base]

            logits = torch.ones(policy.alphabet_size) * (-10.0)

            for base in allowed_bases:
                base_idx = DNA_ALPHABET[base]
                logits[base_idx] = torch.log(
                    torch.tensor((1 - epsilon) / len(allowed_bases))
                )

            noise = torch.log(torch.tensor(epsilon / policy.alphabet_size))
            logits = torch.logaddexp(logits, noise * torch.ones(policy.alphabet_size))

            policy.latent_w.data[template_idx, pos_idx] = logits.to(policy.device)


def _init_bottle_based_policy(policy, iupac_sequence: str, epsilon: float):
    """
    Initialize bottle-based optimizer (IupacBaseOptimizer or IupacPlusOptimizer) from IUPAC sequence.

    For each position, does epsilon-greedy around the bottle selection probability
    for the corresponding IUPAC base. For IupacPlusOptimizer (has latent_v_custom),
    custom bottles get uniform initialization.
    """
    for template_idx in range(policy.num_templates):
        for pos_idx, iupac_base in enumerate(iupac_sequence):
            if iupac_base not in IUPAC_TO_BOTTLE_IDX:
                raise ValueError(
                    f"Invalid IUPAC base '{iupac_base}' at position {pos_idx}"
                )

            correct_bottle_idx = IUPAC_TO_BOTTLE_IDX[iupac_base]

            logits = torch.ones(policy.num_bottles) * (-10.0)
            logits[correct_bottle_idx] = torch.log(torch.tensor(1 - epsilon))

            noise = torch.log(torch.tensor(epsilon / policy.num_bottles))
            logits = torch.logaddexp(logits, noise * torch.ones(policy.num_bottles))

            policy.latent_phi.data[template_idx, pos_idx] = logits.to(policy.device)

    if hasattr(policy, "latent_v_custom"):
        nn.init.uniform_(policy.latent_v_custom.data, -1.0, 1.0)


def aa_to_dna_random(aa_sequence: str, rng: np.random.Generator) -> str:
    """Convert amino acid sequence to DNA using random codon selection."""
    dna_bases = []
    for aa in aa_sequence:
        if aa not in AA_TO_CODONS:
            raise ValueError(f"Unknown amino acid: '{aa}'")
        codons = AA_TO_CODONS[aa]
        codon = rng.choice(codons)
        dna_bases.append(codon)
    return "".join(dna_bases)


def init_from_fasta_sequences(
    policy,
    fasta_path: str,
    epsilon: float = 0.1,
    seed: Optional[int] = None,
) -> list[str]:
    """
    Initialize each template from a different AA sequence sampled from FASTA.

    Args:
        policy: Policy with latent_phi (bottle-based) or latent_w (arbitrary base)
        fasta_path: Path to FASTA file with AA sequences
        epsilon: Noise level for initialization
        seed: Random seed for reproducible sampling

    Returns:
        List of AA sequences used for each template (for logging)
    """
    rng = np.random.default_rng(seed)

    # 1. Load and validate sequences
    aa_sequences = load_fasta(fasta_path)
    if not aa_sequences:
        raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

    expected_aa_len = policy.str_length // 3
    valid_sequences = [s for s in aa_sequences if len(s) == expected_aa_len]
    if not valid_sequences:
        lengths = set(len(s) for s in aa_sequences)
        raise ValueError(
            f"No sequences match expected AA length {expected_aa_len}. "
            f"Found lengths: {sorted(lengths)}"
        )

    # 2. Sample sequences for each template (with replacement)
    selected_indices = rng.choice(len(valid_sequences), size=policy.num_templates)
    selected_aa_seqs = [valid_sequences[i] for i in selected_indices]

    # 3. Initialize each template
    for template_idx, aa_seq in enumerate(selected_aa_seqs):
        # Convert AA -> DNA using random codons
        dna_seq = aa_to_dna_random(aa_seq, rng)

        # Initialize this specific template from DNA sequence
        _init_single_template_from_dna(policy, template_idx, dna_seq, epsilon)

    # Initialize custom bottles if present
    if hasattr(policy, "latent_v_custom"):
        nn.init.uniform_(policy.latent_v_custom.data, -1.0, 1.0)

    return selected_aa_seqs


def _init_single_template_from_dna(
    policy, template_idx: int, dna_sequence: str, epsilon: float
):
    """Initialize a single template from a DNA sequence (A, C, G, T only)."""
    if len(dna_sequence) != policy.str_length:
        raise ValueError(
            f"DNA sequence length {len(dna_sequence)} doesn't match "
            f"policy str_length {policy.str_length}"
        )

    if hasattr(policy, "latent_phi"):
        # Bottle-based policy: use bottle indices 0-3 for A, C, G, T
        for pos_idx, base in enumerate(dna_sequence):
            bottle_idx = IUPAC_TO_BOTTLE_IDX[base]  # A=0, C=1, G=2, T=3

            logits = torch.ones(policy.num_bottles) * (-10.0)
            logits[bottle_idx] = torch.log(torch.tensor(1 - epsilon))

            noise = torch.log(torch.tensor(epsilon / policy.num_bottles))
            logits = torch.logaddexp(logits, noise * torch.ones(policy.num_bottles))

            policy.latent_phi.data[template_idx, pos_idx] = logits.to(policy.device)

    elif hasattr(policy, "latent_w"):
        # Arbitrary base policy: set probabilities directly
        for pos_idx, base in enumerate(dna_sequence):
            base_idx = DNA_ALPHABET[base]

            logits = torch.ones(policy.alphabet_size) * (-10.0)
            logits[base_idx] = torch.log(torch.tensor(1 - epsilon))

            noise = torch.log(torch.tensor(epsilon / policy.alphabet_size))
            logits = torch.logaddexp(logits, noise * torch.ones(policy.alphabet_size))

            policy.latent_w.data[template_idx, pos_idx] = logits.to(policy.device)

    else:
        raise ValueError(
            f"Policy {type(policy).__name__} doesn't support DNA initialization"
        )
