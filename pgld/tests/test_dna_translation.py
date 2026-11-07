"""Tests for DNA to protein translation functions."""

import os
import sys

import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from landscapes import (
    dna_to_protein_batch_loop,
    dna_to_protein_vectorized,
    dna_to_protein_batch,
)


def test_loop_vs_vectorized_basic():
    """Test that vectorized and loop versions produce identical results for basic sequences."""
    sequences = [
        "ATGATGATG",  # MMM
        "GCGGCGGCG",  # AAA
        "TTTAAATAG",  # FK (stop)
        "AAAGGGCCC",  # KGP
    ]

    loop_result = dna_to_protein_batch_loop(sequences)
    vec_result = dna_to_protein_vectorized(sequences)

    assert (
        loop_result == vec_result
    ), f"Loop and vectorized results differ:\nLoop: {loop_result}\nVec: {vec_result}"


def test_loop_vs_vectorized_stop_codons():
    """Test that stop codons are handled identically."""
    sequences = [
        "ATGTAAATG",  # M* (stop after M)
        "ATGATGATG",  # MMM (no stop)
        "TAATGATAG",  # *M* (immediate stop)
        "GCGTGATTT",  # A*F (stop in middle)
    ]

    loop_result = dna_to_protein_batch_loop(sequences)
    vec_result = dna_to_protein_vectorized(sequences)

    assert (
        loop_result == vec_result
    ), f"Stop codon handling differs:\nLoop: {loop_result}\nVec: {vec_result}"


def test_loop_vs_vectorized_large_batch():
    """Test that vectorized works correctly on large batches."""
    # Generate a large batch with same-length sequences
    base_sequences = [
        "ATGATGATGATGATGATGATGATG",  # 8 codons
        "GCGGCGGCGGCGGCGGCGGCGGCG",
        "TTTAAACCCTTTAAACCCTTTAAA",
        "CCCGGGAAACCCGGGAAACCCGGG",
    ]

    # Repeat to make a large batch
    sequences = base_sequences * 50  # 200 sequences

    loop_result = dna_to_protein_batch_loop(sequences)
    vec_result = dna_to_protein_vectorized(sequences)

    assert loop_result == vec_result, "Large batch results differ"
    # Result is a dict, so check all input sequences are present as keys
    assert all(seq in loop_result for seq in sequences), "Missing sequences in result"
    assert len(loop_result) == len(
        set(sequences)
    ), "Should have one entry per unique sequence"


def test_loop_vs_vectorized_realistic_length():
    """Test with realistic peptide-length sequences (72 nucleotides = 24 amino acids)."""
    np.random.seed(42)
    nucleotides = ["A", "C", "G", "T"]

    # Generate 1000 random 72-nucleotide sequences
    sequences = []
    for _ in range(1000):
        seq = "".join(np.random.choice(nucleotides, 72))
        sequences.append(seq)

    loop_result = dna_to_protein_batch_loop(sequences)
    vec_result = dna_to_protein_vectorized(sequences)

    assert loop_result == vec_result, "Realistic sequences differ"

    # Verify some sequences have the right length peptides
    for seq in list(sequences)[:10]:
        peptide = loop_result[seq]
        assert len(peptide) <= 24, f"Peptide too long: {len(peptide)}"


def test_dna_to_protein_batch_dispatch():
    """Test that dna_to_protein_batch correctly dispatches to loop vs vectorized."""
    # Small batch should use loop
    small_batch = ["ATGATGATG", "GCGGCGGCG", "TTTAAATAG"]
    small_result = dna_to_protein_batch(small_batch)
    loop_result = dna_to_protein_batch_loop(small_batch)
    assert small_result == loop_result

    # Large batch with same length should use vectorized
    large_batch = ["ATGATGATGATGATGATGATGATG"] * 200
    large_result = dna_to_protein_batch(large_batch)
    vec_result = dna_to_protein_vectorized(large_batch)
    assert large_result == vec_result


def test_empty_input():
    """Test that empty inputs are handled correctly."""
    assert dna_to_protein_batch_loop([]) == {}
    assert dna_to_protein_vectorized([]) == {}
    assert dna_to_protein_batch([]) == {}


def test_sequences_not_divisible_by_three():
    """Test that sequences not divisible by 3 are truncated correctly."""
    sequences = [
        "ATGATGATGA",  # 10 nucleotides, should use first 9
        "ATGATGATGAT",  # 11 nucleotides, should use first 9
    ]

    loop_result = dna_to_protein_batch_loop(sequences)
    vec_result = dna_to_protein_vectorized(sequences)

    assert loop_result == vec_result
    # Both should produce "MMM" (using first 9 nucleotides)
    assert loop_result["ATGATGATGA"] == "MMM"
    assert loop_result["ATGATGATGAT"] == "MMM"
