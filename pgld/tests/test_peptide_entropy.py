"""Tests for peptide entropy functionality."""

import os
import sys

import pytest
import torch
import numpy as np

# Add parent directory to path to import from pgld module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from entropy import (
    calculate_entropy,
    calculate_peptide_entropy,
    calculate_entropy_gradient_estimate,
    calculate_peptide_entropy_gradient_estimate,
)
from tensor_translation_gpu import dna_probs_to_aa_probs
from policies import (
    ArbitraryBaseOptimizer,
    FiniteBaseOptimizer,
)
from constants import DNA_BASES, CODON_TABLE, AMINO_ACIDS
from landscapes import dna_to_protein
from vis import dna_to_amino_acid_probabilities


class MockLandscape:
    """Mock landscape for testing."""

    def __init__(self):
        self.alphabet = DNA_BASES


@pytest.fixture
def simple_policy():
    """Create a simple test policy."""
    policy = ArbitraryBaseOptimizer(
        str_length=12,  # 4 codons
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )
    # Initialize with random logits to get diverse probabilities
    with torch.no_grad():
        policy.latent_w.normal_(0, 1)
    return policy


@pytest.fixture
def mock_landscape():
    """Create a mock landscape."""
    return MockLandscape()


def test_dna_to_protein_basic():
    """Test basic DNA to protein translation."""
    test_cases = [
        ("ATG", "M"),  # Methionine
        ("ATGATGATG", "MMM"),  # 3 methionines
        ("ATGGCGGGG", "MAG"),  # Met-Ala-Gly
        ("AAACCC", "KP"),  # Lys-Pro
        ("AAACCCTTT", "KPF"),  # Lys-Pro-Phe (truncate last base)
    ]

    for dna, expected in test_cases:
        protein = dna_to_protein(dna)
        assert protein == expected, f"Expected {expected}, got {protein}"


def test_dna_to_protein_stop_codons():
    """Test that stop codons terminate translation."""
    # TAA is a stop codon
    protein = dna_to_protein("ATGTAA")  # Met-Stop
    assert protein == "M", f"Expected 'M', got '{protein}'"

    # Stop codon in middle terminates
    protein = dna_to_protein("ATGTAAGGG")  # Met-Stop-Gly
    assert protein == "M", f"Expected 'M', got '{protein}'"


def test_calculate_peptide_entropy_basic(simple_policy, mock_landscape):
    """Test basic peptide entropy calculation."""
    peptide_entropy = calculate_peptide_entropy(simple_policy)

    # Should be a positive scalar tensor
    assert isinstance(peptide_entropy, torch.Tensor)
    assert peptide_entropy.numel() == 1  # scalar
    assert peptide_entropy.item() >= 0, "Peptide entropy should be non-negative"


def test_calculate_entropy_vs_peptide_entropy(simple_policy, mock_landscape):
    """Test that DNA and peptide entropy give different values."""
    dna_entropy = calculate_entropy(simple_policy)
    peptide_entropy = calculate_peptide_entropy(simple_policy)

    # Both should be positive
    assert dna_entropy.item() > 0
    assert peptide_entropy.item() > 0

    # They should generally be different (not a strict requirement, but likely)
    # This test might occasionally fail due to randomness, but should usually pass
    assert abs(dna_entropy.item() - peptide_entropy.item()) > 1e-6


def test_peptide_entropy_wrong_num_templates(mock_landscape):
    """Test that peptide entropy fails with wrong number of templates."""
    policy = ArbitraryBaseOptimizer(
        str_length=12,
        alphabet_size=4,
        batch_size=10,
        num_templates=2,  # More than 1
        device="cpu",
    )

    with pytest.raises(ValueError, match="only works for num_templates=1"):
        calculate_peptide_entropy(policy)


def test_peptide_entropy_sequence_length_not_divisible_by_3(mock_landscape):
    """Test that peptide entropy handles sequence length not divisible by 3."""
    # 13 is not divisible by 3
    policy = ArbitraryBaseOptimizer(
        str_length=13,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )
    with torch.no_grad():
        policy.latent_w.normal_(0, 1)

    # Should work by truncating to nearest multiple of 3
    peptide_entropy = calculate_peptide_entropy(policy)
    assert isinstance(peptide_entropy, torch.Tensor)
    assert peptide_entropy.item() >= 0


def test_calculate_peptide_entropy_gradient_estimate(simple_policy, mock_landscape):
    """Test peptide entropy gradient estimate calculation."""
    # Sample sequences and get log probs
    sample_result = simple_policy.sample(batch_size=10)
    log_probs = simple_policy.get_log_probs(sample_result)

    # Calculate gradient estimate
    grad_entropy = calculate_peptide_entropy_gradient_estimate(
        mock_landscape, simple_policy, sample_result
    )

    # Should be a scalar tensor with gradients
    assert isinstance(grad_entropy, torch.Tensor)
    assert grad_entropy.numel() == 1
    # Note: grad_entropy can be negative (it's a gradient estimate)


def test_peptide_entropy_gradient_estimate_no_valid_peptides(mock_landscape):
    """Test gradient estimate when no peptides are valid."""
    # Create policy that generates stop codons
    policy = ArbitraryBaseOptimizer(
        str_length=3,  # 1 codon
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    # Force it to generate stop codons by setting high probability for TAA
    with torch.no_grad():
        policy.latent_w[0, 0, :] = torch.log(torch.tensor([0.0, 0.0, 0.0, 1.0]))  # T
        policy.latent_w[0, 1, :] = torch.log(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # A
        policy.latent_w[0, 2, :] = torch.log(torch.tensor([1.0, 0.0, 0.0, 0.0]))  # A

    # Sample sequences (should mostly be TAA = stop codon)
    sample_result = policy.sample(batch_size=10)
    log_probs = policy.get_log_probs(sample_result)

    # Should return zero when no valid peptides
    grad_entropy = calculate_peptide_entropy_gradient_estimate(
        mock_landscape, policy, sample_result
    )

    assert grad_entropy.item() == 0.0


def test_entropy_gradient_estimates_comparison(simple_policy, mock_landscape):
    """Compare DNA vs peptide gradient estimates."""
    sample_result = simple_policy.sample(batch_size=20)
    log_probs = simple_policy.get_log_probs(sample_result)

    # DNA gradient estimate
    dna_grad = calculate_entropy_gradient_estimate(log_probs)

    # Peptide gradient estimate
    peptide_grad = calculate_peptide_entropy_gradient_estimate(
        mock_landscape, simple_policy, sample_result
    )

    # Both should be scalar tensors
    assert isinstance(dna_grad, torch.Tensor)
    assert isinstance(peptide_grad, torch.Tensor)
    assert dna_grad.numel() == 1
    assert peptide_grad.numel() == 1


def test_peptide_entropy_gradient_estimate_bottle_based(mock_landscape):
    """Test gradient estimate with bottle-based optimizer"""
    # Create bottle-based policy with multiple templates
    policy = FiniteBaseOptimizer(
        str_length=9,  # 3 codons
        alphabet_size=4,
        batch_size=10,
        num_templates=2,  # 2 templates
        num_bottles=3,  # 3 bottles
        device="cpu",
    )

    # Sample sequences
    sample_result = policy.sample(batch_size=10)

    # Should return zero when no valid peptides
    grad_entropy = calculate_peptide_entropy_gradient_estimate(
        mock_landscape, policy, sample_result
    )
    # Should be a scalar tensor with gradients
    assert isinstance(grad_entropy, torch.Tensor)
    assert grad_entropy.numel() == 1


def test_codon_indexing_formula():
    """
    Test that codon indexing formula i*16 + j*4 + k correctly maps
    to codon strings and CODON_TABLE lookups.
    """
    # Verify all 64 codons have correct index mapping
    for i, b1 in enumerate(DNA_BASES):
        for j, b2 in enumerate(DNA_BASES):
            for k, b3 in enumerate(DNA_BASES):
                codon = b1 + b2 + b3
                codon_idx = i * 16 + j * 4 + k

                # Verify index is in valid range
                assert (
                    0 <= codon_idx < 64
                ), f"Invalid index {codon_idx} for codon {codon}"

                # Verify codon exists in table
                assert codon in CODON_TABLE, f"Codon {codon} not in CODON_TABLE"

                # Verify specific known mappings
                if codon == "AAA":
                    assert codon_idx == 0
                    assert CODON_TABLE[codon] == "K"
                elif codon == "TTT":
                    assert codon_idx == 63
                    assert CODON_TABLE[codon] == "F"
                elif codon == "ATG":
                    assert codon_idx == 0 * 16 + 3 * 4 + 2  # A=0, T=3, G=2
                    assert CODON_TABLE[codon] == "M"
                elif codon == "TAA":
                    assert codon_idx == 3 * 16 + 0 * 4 + 0  # T=3, A=0, A=0
                    assert CODON_TABLE[codon] == "*"

    # Verify we covered all 64 codons
    all_indices = set()
    for i in range(4):
        for j in range(4):
            for k in range(4):
                all_indices.add(i * 16 + j * 4 + k)
    assert len(all_indices) == 64, "Should have exactly 64 unique indices"
    assert all_indices == set(range(64)), "Indices should be 0-63"


def test_torch_vs_numpy_amino_acid_probs():
    """
    Test that PyTorch (dna_probs_to_aa_probs) and numpy (dna_to_amino_acid_probabilities)
    implementations give the same results for DNA to amino acid probability conversion.
    """
    # Create test DNA probabilities - uniform for simplicity
    seq_len = 12  # 4 codons
    dna_probs_np = np.ones((seq_len, 4)) * 0.25

    # Add some variation to make it non-trivial
    dna_probs_np[0] = [0.7, 0.1, 0.1, 0.1]  # First position biased toward A
    dna_probs_np[3] = [0.1, 0.7, 0.1, 0.1]  # Fourth position biased toward C

    dna_probs_torch = torch.from_numpy(dna_probs_np).float()

    aa_probs_torch = dna_probs_to_aa_probs(dna_probs_torch)
    aa_probs_np = dna_to_amino_acid_probabilities(dna_probs_np)

    aa_probs_torch_np = aa_probs_torch.detach().cpu().numpy()

    # Exclude stop codons from numpy version to match torch output
    aa_probs_np_no_stop = aa_probs_np[:, :-1]
    aa_probs_np_normalized = aa_probs_np_no_stop / aa_probs_np_no_stop.sum(
        axis=1, keepdims=True
    )

    # Compare shapes
    assert (
        aa_probs_torch_np.shape == aa_probs_np_normalized.shape
    ), f"Shape mismatch: torch {aa_probs_torch_np.shape} vs numpy {aa_probs_np_normalized.shape}"

    # Compare values with tolerance for floating point errors
    np.testing.assert_allclose(
        aa_probs_torch_np,
        aa_probs_np_normalized,
        rtol=1e-5,
        atol=1e-7,
        err_msg="Torch and numpy amino acid probabilities don't match",
    )

    # Verify probabilities sum to 1 for each codon position
    for i in range(aa_probs_torch_np.shape[0]):
        assert (
            abs(aa_probs_torch_np[i].sum() - 1.0) < 1e-6
        ), f"Probabilities don't sum to 1 at position {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
