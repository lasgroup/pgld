"""Tests for BO-related functions in generate_training_data.py, bo.py, and train_bo.py."""

from __future__ import annotations

import os
import sys

import pytest
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import (
    one_hot_encode_peptide,
    one_hot_encode_dna,
    AA_VOCAB_NO_STOP,
    DNA_BASES,
)
from generate_training_data import (
    SequenceDataset,
    PeptideDataset,
    DNADataset,
    get_starting_data_for_bo,
)
from landscapes import TFBindingLandscape, MHCflurryLandscape
from train_bo import calculate_new_hits


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tf_landscape():
    """Create a TFBindingLandscape for testing."""
    return TFBindingLandscape(str_length=8, objectives=["SIX6_REF_R1"])


@pytest.fixture
def mhc_landscape():
    """Create an MHCflurryLandscape for testing."""
    try:
        return MHCflurryLandscape(
            str_length=27,
            objectives=["HLA-A*02:01"],
            invalid_scores={"HLA-A*02:01": 0.0},
        )
    except ImportError:
        pytest.skip("MHCflurry not available")


# =============================================================================
# Tests for one_hot_encode_peptide
# =============================================================================


def test_peptide_encoding_basic_dimensions():
    """Test basic one-hot encoding dimensions of a simple peptide."""
    seq = "ACE"
    length = 5
    encoded = one_hot_encode_peptide(seq, length)

    assert encoded.shape == (length, len(AA_VOCAB_NO_STOP))
    assert encoded.shape == (5, 20)


def test_peptide_encoding_padding():
    """Test that peptide sequences shorter than length are padded with zeros."""
    seq = "AC"
    length = 5
    encoded = one_hot_encode_peptide(seq, length)

    # First two positions should have encoding
    assert encoded[0].sum() == 1.0
    assert encoded[1].sum() == 1.0
    # Last three positions should be zeros (padding)
    assert encoded[2].sum() == 0.0
    assert encoded[3].sum() == 0.0
    assert encoded[4].sum() == 0.0


def test_peptide_encoding_all_amino_acids():
    """Test encoding of all 20 standard amino acids."""
    seq = AA_VOCAB_NO_STOP  # "ACDEFGHIKLMNPQRSTVWY"
    length = len(AA_VOCAB_NO_STOP)
    encoded = one_hot_encode_peptide(seq, length)

    # Each position should have exactly one 1
    for i in range(length):
        assert encoded[i].sum() == 1.0
        assert encoded[i, i] == 1.0  # Diagonal should be 1


# =============================================================================
# Tests for one_hot_encode_dna
# =============================================================================


def test_dna_encoding_basic_dimensions():
    """Test basic one-hot encoding dimensions of a simple DNA sequence."""
    seq = "ACGT"
    length = 6
    encoded = one_hot_encode_dna(seq, length)

    assert encoded.shape == (length, len(DNA_BASES))
    assert encoded.shape == (6, 4)


def test_dna_encoding_all_nucleotides():
    """Test encoding of all 4 nucleotides."""
    seq = "ACGT"
    length = 4
    encoded = one_hot_encode_dna(seq, length)

    expected_positions = {"A": 0, "C": 1, "G": 2, "T": 3}
    for i, nucleotide in enumerate(seq):
        assert encoded[i, expected_positions[nucleotide]] == 1.0
        assert encoded[i].sum() == 1.0


def test_dna_encoding_padding():
    """Test that DNA sequences shorter than length are padded with zeros."""
    seq = "AC"
    length = 5
    encoded = one_hot_encode_dna(seq, length)

    # First two positions should have encoding
    assert encoded[0].sum() == 1.0
    assert encoded[1].sum() == 1.0
    # Last three positions should be zeros (padding)
    for i in range(2, 5):
        assert encoded[i].sum() == 0.0


def test_dna_encoding_specific_example():
    """Test a specific DNA encoding example with known expected output."""
    seq = "ACG"
    length = 3
    encoded = one_hot_encode_dna(seq, length)

    expected = torch.zeros(3, 4)
    expected[0, 0] = 1.0  # A
    expected[1, 1] = 1.0  # C
    expected[2, 2] = 1.0  # G

    assert torch.allclose(encoded, expected)


# =============================================================================
# Tests for SequenceDataset
# =============================================================================


def test_peptide_dataset_length():
    """Test that PeptideDataset returns correct length."""
    sequences = ["ACE", "DEF", "GHI"]
    targets = [1.0, 2.0, 3.0]
    length = 5

    dataset = PeptideDataset(sequences, targets, length)
    assert len(dataset) == 3


def test_peptide_dataset_getitem():
    """Test that PeptideDataset returns correct items."""
    sequences = ["ACE", "DEF"]
    targets = [1.0, 2.0]
    length = 5

    dataset = PeptideDataset(sequences, targets, length)

    seq, target = dataset[0]
    assert seq.shape == (length, 20)
    assert target.item() == 1.0

    seq, target = dataset[1]
    assert target.item() == 2.0


def test_dna_dataset_length():
    """Test that DNADataset returns correct length."""
    sequences = ["ACGT", "GCTA", "TTAA"]
    targets = [1.0, 2.0, 3.0]
    length = 6

    dataset = DNADataset(sequences, targets, length)
    assert len(dataset) == 3


def test_dna_dataset_getitem():
    """Test that DNADataset returns correct items."""
    sequences = ["ACGT", "GCTA"]
    targets = [1.5, 2.5]
    length = 6

    dataset = DNADataset(sequences, targets, length)

    seq, target = dataset[0]
    assert seq.shape == (length, 4)
    assert target.item() == 1.5


def test_dataset_encoded_sequences_shape():
    """Test that encoded sequences have correct shape."""
    sequences = ["ACE", "DEF", "GHI", "KLM"]
    targets = [1.0, 2.0, 3.0, 4.0]
    length = 7

    dataset = PeptideDataset(sequences, targets, length)

    # encoded_sequences should be (n_samples, length, vocab_size)
    assert dataset.encoded_sequences.shape == (4, 7, 20)


def test_dataset_targets_type():
    """Test that dataset targets are float32 tensors."""
    sequences = ["ACE"]
    targets = [1.0]
    length = 5

    dataset = PeptideDataset(sequences, targets, length)
    assert dataset.targets.dtype == torch.float32


def test_sequence_dataset_base_class():
    """Test SequenceDataset base class directly."""
    sequences = ["ACG", "TGC"]
    targets = [0.5, 1.5]
    length = 4

    dataset = SequenceDataset(sequences, targets, length, one_hot_encode_dna)

    assert len(dataset) == 2
    seq, target = dataset[0]
    assert seq.shape == (4, 4)


# =============================================================================
# Tests for get_starting_data_for_bo
# =============================================================================


def test_tf_binding_returns_dna_sequences(tf_landscape):
    """Test that get_starting_data_for_bo returns DNA sequences for TFBinding."""
    sequences = get_starting_data_for_bo(tf_landscape, n_seq=10)

    assert isinstance(sequences, list)
    assert len(sequences) == 10


def test_tf_binding_sequence_length(tf_landscape):
    """Test that TFBinding generated sequences have correct length."""
    sequences = get_starting_data_for_bo(tf_landscape, n_seq=10)

    for seq in sequences:
        assert len(seq) == tf_landscape.str_length


def test_tf_binding_sequences_are_valid_dna(tf_landscape):
    """Test that TFBinding generated sequences contain only valid DNA characters."""
    sequences = get_starting_data_for_bo(tf_landscape, n_seq=10)

    valid_chars = set("ACGT")
    for seq in sequences:
        assert all(c in valid_chars for c in seq), f"Invalid sequence: {seq}"


def test_mhc_returns_dna_sequences(mhc_landscape):
    """Test that get_starting_data_for_bo returns DNA sequences for MHC."""
    sequences = get_starting_data_for_bo(mhc_landscape, n_seq=10)

    assert isinstance(sequences, list)
    # Should be DNA sequences (NNK), not peptides
    valid_chars = set("ACGT")
    for seq in sequences:
        assert all(c in valid_chars for c in seq), f"Expected DNA, got: {seq}"


def test_mhc_dna_sequence_length(mhc_landscape):
    """Test that MHC generated DNA sequences have correct length."""
    sequences = get_starting_data_for_bo(mhc_landscape, n_seq=10)

    for seq in sequences:
        assert len(seq) == mhc_landscape.str_length


# =============================================================================
# Tests for annotate_sequences
# =============================================================================


def test_tf_annotate_returns_dict(tf_landscape):
    """Test that annotate_sequences returns sequences and dict of fitness values."""
    dna_sequences = get_starting_data_for_bo(tf_landscape, n_seq=10)
    sequences, fitness_values = tf_landscape.annotate_sequences(dna_sequences)

    assert isinstance(sequences, list)
    assert isinstance(fitness_values, dict)
    assert len(fitness_values) == 1  # Single objective
    assert "SIX6_REF_R1" in fitness_values
    assert len(sequences) == len(fitness_values["SIX6_REF_R1"])


def test_mhc_annotate_returns_peptides(mhc_landscape):
    """Test that MHC annotate_sequences returns peptides, not DNA."""
    dna_sequences = get_starting_data_for_bo(mhc_landscape, n_seq=20)
    peptides, fitness_values = mhc_landscape.annotate_sequences(dna_sequences)

    assert isinstance(peptides, list)
    assert isinstance(fitness_values, dict)

    # Peptides should contain amino acids, not just ACGT
    amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
    for seq in peptides:
        assert all(c in amino_acids for c in seq), f"Invalid peptide: {seq}"


def test_mhc_annotate_fitness_is_negative_log10(mhc_landscape):
    """Test that MHC fitness values are -log10 of raw predictions."""
    dna_sequences = get_starting_data_for_bo(mhc_landscape, n_seq=20)
    peptides, fitness_values = mhc_landscape.annotate_sequences(dna_sequences)

    obj_name = "HLA-A*02:01"
    # Get raw predictions directly from peptides
    raw_predictions = mhc_landscape._predict_from_peptides(
        peptides, percentiles=False, objective=obj_name
    )

    # Expected: -log10(raw_predictions)
    expected_fitness = -np.log10(raw_predictions)

    np.testing.assert_array_almost_equal(
        fitness_values[obj_name], expected_fitness, decimal=5
    )


# =============================================================================
# Tests for calculate_new_hits (multi-objective)
# =============================================================================


def test_calculate_new_hits_single_objective():
    """Test calculate_new_hits with single objective."""
    sampled_sequences = []
    new_observations = ["AAAA", "BBBB", "CCCC"]
    new_fitness_values = {"obj1": [1.5, 0.5, 2.0]}
    bo_thresholds = {"obj1": 1.0}

    new_hits, hits = calculate_new_hits(
        sampled_sequences, new_observations, new_fitness_values, bo_thresholds
    )

    # Hits: AAAA (1.5 > 1.0), CCCC (2.0 > 1.0) = 2
    assert hits == 2
    # New hits: same as hits since sampled_sequences is empty
    assert new_hits == 2


def test_calculate_new_hits_multi_objective_and_logic():
    """Test that multi-objective uses AND logic."""
    sampled_sequences = []
    new_observations = ["AAAA", "BBBB", "CCCC", "DDDD"]
    new_fitness_values = {
        "obj1": [1.5, 1.5, 0.5, 2.0],  # AAAA, BBBB pass obj1
        "obj2": [2.0, 0.5, 1.5, 1.5],  # AAAA, CCCC, DDDD pass obj2
    }
    bo_thresholds = {"obj1": 1.0, "obj2": 1.0}

    new_hits, hits = calculate_new_hits(
        sampled_sequences, new_observations, new_fitness_values, bo_thresholds
    )

    # Only AAAA (1.5 > 1.0 AND 2.0 > 1.0) and DDDD (2.0 > 1.0 AND 1.5 > 1.0) pass both
    assert hits == 2
    assert new_hits == 2


def test_calculate_new_hits_duplicates_in_batch():
    """Test that duplicates within a batch are only counted once as new hits."""
    sampled_sequences = []
    new_observations = ["AAAA", "AAAA", "AAAA", "BBBB"]
    new_fitness_values = {"obj1": [1.5, 1.5, 1.5, 2.0]}
    bo_thresholds = {"obj1": 1.0}

    new_hits, hits = calculate_new_hits(
        sampled_sequences, new_observations, new_fitness_values, bo_thresholds
    )

    # Hits: all 4 observations are above threshold
    assert hits == 4
    # New hits: AAAA counted once + BBBB = 2
    assert new_hits == 2


def test_calculate_new_hits_empty_new_observations():
    """Test with empty new observations."""
    sampled_sequences = ["AAAA", "BBBB"]
    new_observations = []
    new_fitness_values = {"obj1": []}
    bo_thresholds = {"obj1": 1.0}

    new_hits, hits = calculate_new_hits(
        sampled_sequences, new_observations, new_fitness_values, bo_thresholds
    )

    assert hits == 0
    assert new_hits == 0


def test_calculate_new_hits_previously_observed():
    """Test that previously observed sequences don't count as new hits."""
    sampled_sequences = ["AAAA", "BBBB"]
    new_observations = ["AAAA", "CCCC", "DDDD"]
    new_fitness_values = {"obj1": [2.0, 1.5, 0.5]}
    bo_thresholds = {"obj1": 1.0}

    new_hits, hits = calculate_new_hits(
        sampled_sequences, new_observations, new_fitness_values, bo_thresholds
    )

    # Hits: AAAA (2.0), CCCC (1.5) = 2
    assert hits == 2
    # New hits: only CCCC (AAAA was already observed)
    assert new_hits == 1


def test_calculate_new_hits_mixed_scenario():
    """Test a realistic mixed scenario with various combinations."""
    sampled_sequences = ["SEQ1", "SEQ2", "SEQ3"]
    new_observations = [
        "SEQ1",  # already observed, above threshold
        "SEQ4",  # new, above threshold
        "SEQ5",  # new, below threshold
        "SEQ4",  # duplicate in batch, above threshold
        "SEQ6",  # new, above threshold
        "SEQ2",  # already observed, below threshold
    ]
    new_fitness_values = {"obj1": [2.0, 1.5, 0.5, 1.5, 3.0, 0.3]}
    bo_thresholds = {"obj1": 1.0}

    new_hits, hits = calculate_new_hits(
        sampled_sequences, new_observations, new_fitness_values, bo_thresholds
    )

    # Hits: SEQ1 (2.0), SEQ4 (1.5), SEQ4 again (1.5), SEQ6 (3.0) = 4
    assert hits == 4
    # New hits: SEQ4 (first occurrence) + SEQ6 = 2
    # SEQ1 was already observed, SEQ4 duplicate doesn't count again
    assert new_hits == 2
