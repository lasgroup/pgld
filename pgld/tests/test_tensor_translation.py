"""Test tensor-based DNA to protein translation."""

import os
import sys

sys.path.insert(
    0, os.environ.get("PGLD_PKG_ROOT", os.path.dirname(os.path.dirname(__file__)))
)

import torch
import numpy as np
import pytest
from tensor_translation import tensor_to_protein_batch
from tensor_translation_gpu import (
    build_codon_lookup_gpu,
    translate_dna_to_aa_indices_gpu,
    aa_indices_to_strings,
)
from landscapes import dna_to_protein_batch, sequences_to_strings_vectorized


def test_translation_correctness():
    """Verify tensor translation matches string-based translation."""
    alphabet = ["A", "C", "G", "T"]

    torch.manual_seed(42)
    n_seqs = 100
    seq_len = 75
    dna_tensor = torch.randint(0, 4, (n_seqs, seq_len))

    # Method 1: Tensor-based (new)
    idx_to_peptide_tensor = tensor_to_protein_batch(dna_tensor)

    # Method 2: String-based (original)
    dna_strings = sequences_to_strings_vectorized(dna_tensor, alphabet)
    dna_to_peptide_string = dna_to_protein_batch(dna_strings)

    # Compare results
    for idx in range(n_seqs):
        peptide_tensor = idx_to_peptide_tensor[idx]
        peptide_string = dna_to_peptide_string[dna_strings[idx]]
        assert (
            peptide_tensor == peptide_string
        ), f"Mismatch at idx {idx}: tensor={peptide_tensor}, string={peptide_string}, dna={dna_strings[idx]}"


def test_gpu_translation_cpu_fallback():
    """Test GPU translation functions work on CPU as fallback."""
    torch.manual_seed(42)
    n_seqs = 50
    seq_len = 75

    # Use CPU device explicitly
    device = torch.device("cpu")
    dna_tensor = torch.randint(0, 4, (n_seqs, seq_len), device=device)

    # Build codon lookup on CPU
    codon_lookup = build_codon_lookup_gpu(device)
    assert codon_lookup.device.type == "cpu", "Codon lookup should be on CPU"

    # Translate to AA indices
    aa_indices, first_stops = translate_dna_to_aa_indices_gpu(dna_tensor, codon_lookup)
    assert aa_indices.device.type == "cpu", "AA indices should be on CPU"
    assert first_stops.device.type == "cpu", "First stops should be on CPU"

    # Convert to strings
    peptide_strings = aa_indices_to_strings(aa_indices, first_stops)
    assert (
        len(peptide_strings) == n_seqs
    ), f"Expected {n_seqs} peptides, got {len(peptide_strings)}"

    # Verify against string-based translation
    alphabet = ["A", "C", "G", "T"]
    dna_strings = sequences_to_strings_vectorized(dna_tensor, alphabet)
    dna_to_peptide = dna_to_protein_batch(dna_strings)

    for i, dna_str in enumerate(dna_strings):
        expected_peptide = dna_to_peptide[dna_str]
        actual_peptide = peptide_strings[i]
        assert (
            actual_peptide == expected_peptide
        ), f"Mismatch at sequence {i}: expected={expected_peptide}, actual={actual_peptide}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_translation_on_gpu():
    """Test GPU translation functions on actual GPU (if available)."""
    torch.manual_seed(42)
    n_seqs = 100
    seq_len = 75

    device = torch.device("cuda")
    dna_tensor = torch.randint(0, 4, (n_seqs, seq_len), device=device)

    # Build codon lookup on GPU
    codon_lookup = build_codon_lookup_gpu(device)
    assert codon_lookup.device.type == "cuda", "Codon lookup should be on GPU"

    # Translate to AA indices
    aa_indices, first_stops = translate_dna_to_aa_indices_gpu(dna_tensor, codon_lookup)
    assert aa_indices.device.type == "cuda", "AA indices should be on GPU"
    assert first_stops.device.type == "cuda", "First stops should be on GPU"

    # Convert to strings (transfers to CPU internally)
    peptide_strings = aa_indices_to_strings(aa_indices, first_stops)
    assert (
        len(peptide_strings) == n_seqs
    ), f"Expected {n_seqs} peptides, got {len(peptide_strings)}"

    # Verify against string-based translation
    alphabet = ["A", "C", "G", "T"]
    dna_strings = sequences_to_strings_vectorized(dna_tensor.cpu(), alphabet)
    dna_to_peptide = dna_to_protein_batch(dna_strings)

    for i, dna_str in enumerate(dna_strings):
        expected_peptide = dna_to_peptide[dna_str]
        actual_peptide = peptide_strings[i]
        assert (
            actual_peptide == expected_peptide
        ), f"Mismatch at sequence {i}: expected={expected_peptide}, actual={actual_peptide}"


def test_gpu_translation_handles_stop_codons():
    """Test that GPU translation correctly handles stop codons."""
    device = torch.device("cpu")  # Works on CPU too

    # Create sequences with known stop codons
    # Alphabet: ['A', 'C', 'G', 'T'] -> indices [0, 1, 2, 3]
    # Stop codons: TAA=[3,0,0], TAG=[3,0,2], TGA=[3,2,0]
    test_cases = [
        # DNA sequence (as indices), expected truncation point
        (torch.tensor([[3, 0, 0, 1, 1, 1]], device=device), 0),  # TAA at start
        (torch.tensor([[1, 1, 1, 3, 0, 0]], device=device), 1),  # TAA after 1 codon
        (
            torch.tensor([[1, 1, 1, 2, 2, 2, 3, 0, 2]], device=device),
            2,
        ),  # TAG after 2 codons
    ]

    codon_lookup = build_codon_lookup_gpu(device)

    for dna_tensor, expected_len in test_cases:
        aa_indices, first_stops = translate_dna_to_aa_indices_gpu(
            dna_tensor, codon_lookup
        )

        if expected_len == 0:
            # Stop at position 0 means empty peptide
            assert (
                first_stops[0].item() == 0
            ), f"Expected stop at position 0, got {first_stops[0].item()}"
        else:
            # Stop should be at expected position
            assert (
                first_stops[0].item() == expected_len
            ), f"Expected stop at position {expected_len}, got {first_stops[0].item()}"

        # Convert to string and verify length
        peptide_strings = aa_indices_to_strings(aa_indices, first_stops)
        alphabet = ["A", "C", "G", "T"]
        dna_strings = sequences_to_strings_vectorized(dna_tensor, alphabet)
        dna_to_peptide = dna_to_protein_batch(dna_strings)

        expected_peptide = dna_to_peptide[dna_strings[0]]
        actual_peptide = peptide_strings[0]
        assert (
            actual_peptide == expected_peptide
        ), f"Peptide mismatch: expected={expected_peptide}, actual={actual_peptide}"
