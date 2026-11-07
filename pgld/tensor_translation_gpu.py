"""GPU-accelerated DNA to protein translation and probability conversion."""

import torch
import numpy as np
from constants import DNA_BASES, CODON_TABLE, AMINO_ACIDS


def build_codon_lookup_gpu(device: torch.device) -> torch.Tensor:
    """Build codon→amino acid lookup table on GPU."""

    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    codon_to_aa_idx = np.zeros(64, dtype=np.int8)

    for i, b1 in enumerate(DNA_BASES):
        for j, b2 in enumerate(DNA_BASES):
            for k, b3 in enumerate(DNA_BASES):
                codon = b1 + b2 + b3
                codon_idx = i * 16 + j * 4 + k
                aa = CODON_TABLE[codon]
                codon_to_aa_idx[codon_idx] = aa_to_idx[aa]

    return torch.from_numpy(codon_to_aa_idx).to(device)


def translate_dna_to_aa_indices_gpu(
    dna_tensor: torch.Tensor, codon_lookup: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Translate DNA sequences to amino acid indices on GPU.

    Args:
        dna_tensor: (n_sequences, seq_length) with integer indices 0-3
        codon_lookup: (64,) tensor mapping codon indices to AA indices

    Returns:
        aa_indices: (n_sequences, n_codons) amino acid indices
        first_stop_positions: (n_sequences,) position of first stop codon (-1 if none)
    """
    n_seqs, seq_len = dna_tensor.shape

    if seq_len % 3 != 0:
        seq_len = seq_len - (seq_len % 3)
        dna_tensor = dna_tensor[:, :seq_len]

    n_codons = seq_len // 3

    dna_codons = dna_tensor.reshape(n_seqs, n_codons, 3)
    codon_indices = (
        dna_codons[:, :, 0] * 16 + dna_codons[:, :, 1] * 4 + dna_codons[:, :, 2]
    )

    aa_indices = codon_lookup[codon_indices]

    stop_idx = len(AMINO_ACIDS) - 1
    stop_mask = aa_indices == stop_idx

    has_stop = stop_mask.any(dim=1)
    first_stop_positions = torch.where(
        has_stop,
        stop_mask.int().argmax(dim=1),
        torch.full((n_seqs,), -1, dtype=torch.long, device=dna_tensor.device),
    )

    return aa_indices, first_stop_positions


def aa_indices_to_strings(
    aa_indices: torch.Tensor, first_stop_positions: torch.Tensor
) -> list[str]:
    """Convert amino acid indices to protein strings (on CPU)."""

    aa_indices_cpu = aa_indices.cpu().numpy()
    first_stop_cpu = first_stop_positions.cpu().numpy()
    n_seqs, n_codons = aa_indices_cpu.shape

    idx_to_aa = np.array(AMINO_ACIDS)
    aa_chars = idx_to_aa[aa_indices_cpu]

    results = []
    for i in range(n_seqs):
        end_idx = first_stop_cpu[i] if first_stop_cpu[i] != -1 else n_codons
        protein = "".join(aa_chars[i, :end_idx])
        results.append(protein)

    return results


def dna_probs_to_aa_probs(dna_probs: torch.Tensor) -> torch.Tensor:
    """
    Convert DNA base probabilities to amino acid probabilities.
    Differentiable PyTorch implementation.

    Args:
        dna_probs: Tensor of shape (seq_len, 4) with DNA base probabilities (A, C, G, T order)

    Returns:
        Tensor of shape (num_codons, 20) with amino acid probabilities (excluding stop codons)
    """
    seq_len = dna_probs.shape[0]
    if seq_len % 3 != 0:
        dna_probs = dna_probs[: seq_len - (seq_len % 3)]
        seq_len = dna_probs.shape[0]

    num_codons = seq_len // 3
    device = dna_probs.device
    dtype = dna_probs.dtype

    # Build codon to AA index mapping
    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    codon_to_aa = torch.zeros(64, dtype=torch.long, device=device)
    for i, b1 in enumerate(DNA_BASES):
        for j, b2 in enumerate(DNA_BASES):
            for k, b3 in enumerate(DNA_BASES):
                codon = b1 + b2 + b3
                codon_idx = i * 16 + j * 4 + k
                codon_to_aa[codon_idx] = aa_to_idx[CODON_TABLE[codon]]

    # Compute amino acid probabilities for each codon position
    aa_probs = torch.zeros(num_codons, len(AMINO_ACIDS), device=device, dtype=dtype)

    for codon_pos in range(num_codons):
        pos1_probs = dna_probs[codon_pos * 3]
        pos2_probs = dna_probs[codon_pos * 3 + 1]
        pos3_probs = dna_probs[codon_pos * 3 + 2]

        # Compute all 64 codon probabilities via outer products
        codon_probs = (
            pos1_probs[:, None, None]
            * pos2_probs[None, :, None]
            * pos3_probs[None, None, :]
        )
        codon_probs_flat = codon_probs.reshape(64)

        # Accumulate into amino acid probs
        aa_probs[codon_pos].scatter_add_(0, codon_to_aa, codon_probs_flat)

    # Exclude stop codons (last column) and renormalize
    aa_probs_no_stop = aa_probs[:, :-1]
    aa_probs_normalized = aa_probs_no_stop / aa_probs_no_stop.sum(
        dim=1, keepdim=True
    ).clamp(min=1e-8)

    return aa_probs_normalized
