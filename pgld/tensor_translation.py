"""Optimized tensor-based DNA to protein translation."""

from typing import Union, Optional, Dict, Any
import numpy as np
import torch
from constants import DNA_BASES, CODON_TABLE, AMINO_ACIDS


def tensor_to_protein_batch(
    dna_tensor: Union[torch.Tensor, np.ndarray], dna_strings: Optional[list] = None
) -> Dict[Any, str]:
    """
    Translate DNA sequences from tensor format directly to proteins.

    This runs on the cpu. For a gpu implementation see tensor_translation_gpu.

    Args:
        dna_tensor: Tensor of shape (n_sequences, seq_length) with integer indices 0-3.
        dna_strings: Keys for result dict. If None, uses indices.

    Returns:
        Dictionary mapping keys to protein sequences.
    """
    dna_arr = (
        dna_tensor.cpu().numpy()
        if hasattr(dna_tensor, "cpu")
        else np.asarray(dna_tensor)
    )
    n_seqs, seq_len = dna_arr.shape

    if seq_len % 3 != 0:
        seq_len = seq_len - (seq_len % 3)
        dna_arr = dna_arr[:, :seq_len]

    n_codons = seq_len // 3

    aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
    codon_to_aa_idx = np.zeros(64, dtype=np.int8)
    for i, b1 in enumerate(DNA_BASES):
        for j, b2 in enumerate(DNA_BASES):
            for k, b3 in enumerate(DNA_BASES):
                codon = b1 + b2 + b3
                codon_idx = i * 16 + j * 4 + k
                aa = CODON_TABLE[codon]
                codon_to_aa_idx[codon_idx] = aa_to_idx[aa]

    dna_codons = dna_arr.reshape(n_seqs, n_codons, 3)
    codon_indices = (
        dna_codons[:, :, 0] * 16 + dna_codons[:, :, 1] * 4 + dna_codons[:, :, 2]
    )
    aa_indices = codon_to_aa_idx[codon_indices]
    idx_to_aa = np.array(AMINO_ACIDS)

    stop_idx = aa_to_idx["*"]
    stop_mask = aa_indices == stop_idx
    first_stop = np.where(stop_mask.any(axis=1), stop_mask.argmax(axis=1), -1)

    aa_chars = idx_to_aa[aa_indices]

    if dna_strings is None:
        dna_strings = range(n_seqs)

    results = {}
    for i, dna_key in enumerate(dna_strings):
        end_idx = first_stop[i] if first_stop[i] != -1 else n_codons
        results[dna_key] = "".join(aa_chars[i, :end_idx])

    return results
