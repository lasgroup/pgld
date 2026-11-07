"""Test script to verify bottle sampling of BottleBasedOptimizer."""

import os
import sys

import pytest
import torch

# Add parent directory to path to import from pgld module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies import FiniteBaseOptimizer, IupacBaseOptimizer


@pytest.fixture
def iupac_optimizer():
    """Create an IupacBaseOptimizer for testing."""
    return IupacBaseOptimizer(
        str_length=20,
        alphabet_size=4,
        batch_size=100,
        num_templates=3,
        device="cpu",
    )


@pytest.fixture
def finite_optimizer():
    """Create a FiniteBaseOptimizer for testing."""
    return FiniteBaseOptimizer(
        str_length=20,
        alphabet_size=4,
        batch_size=100,
        num_templates=3,
        num_bottles=10,
        device="cpu",
    )


@pytest.mark.parametrize("optimizer_fixture", ["iupac_optimizer", "finite_optimizer"])
@pytest.mark.parametrize("batch_size", [50, 100])
def test_same_template_uses_same_bottles(optimizer_fixture, batch_size, request):
    """Test that sequences from the same template use identical bottle configurations.

    Bottles should be sampled once per template, not once per sequence, to match the physical DNA synthesis process.
    """
    optimizer = request.getfixturevalue(optimizer_fixture)

    sample_result = optimizer.sample(batch_size=batch_size)

    unique_templates = torch.unique(sample_result.mixture_indices)
    print(f"\nSampled {batch_size} sequences from {len(unique_templates)} templates")

    # Check consistency: sequences from the same template should have identical bottle_indices
    for template_idx in unique_templates:
        mask = sample_result.mixture_indices == template_idx
        sequences_from_template = mask.sum().item()
        bottles_from_template = sample_result.bottle_indices[mask]

        # All rows should be identical since a given template uses the same bottles
        first_bottles = bottles_from_template[0]
        all_same = torch.all(bottles_from_template == first_bottles, dim=0).all()

        assert all_same, (
            f"Template {template_idx.item()}: {sequences_from_template} sequences "
            f"should use same bottles but don't!"
        )
        print(
            f"✓ Template {template_idx.item()}: {sequences_from_template} sequences use same bottles"
        )


@pytest.mark.parametrize("optimizer_fixture", ["iupac_optimizer", "finite_optimizer"])
def test_same_template_produces_diverse_sequences(optimizer_fixture, request):
    """Test that sequences from the same template can produce different sequences.

    While bottles are fixed per template, the character sampling from those bottles
    should still produce diverse sequences (not all identical).
    """
    optimizer = request.getfixturevalue(optimizer_fixture)

    # Use larger batch to increase chance of multiple sequences per template
    batch_size = 200
    sample_result = optimizer.sample(batch_size=batch_size)

    unique_templates = torch.unique(sample_result.mixture_indices)

    found_diversity = False
    for template_idx in unique_templates:
        mask = sample_result.mixture_indices == template_idx
        sequences_from_template = sample_result.sequences[mask]

        if len(sequences_from_template) > 1:
            # Check that not all sequences are identical
            first_seq = sequences_from_template[0]
            all_identical = torch.all(sequences_from_template == first_seq, dim=0).all()

            if not all_identical:
                found_diversity = True
                print(f"✓ Template {template_idx.item()}: Sequences are diverse")

    # We expect at least some diversity across all templates
    # (could fail with very low probability if all draws happen to be identical)
    assert found_diversity, "Expected some sequence diversity within templates"


def test_bottle_indices_have_correct_shape():
    """Test that bottle_indices tensor has the expected shape."""
    optimizer = IupacBaseOptimizer(
        str_length=20,
        alphabet_size=4,
        batch_size=100,
        num_templates=3,
        device="cpu",
    )

    batch_size = 50
    sample_result = optimizer.sample(batch_size=batch_size)

    assert sample_result.bottle_indices.shape == (batch_size, optimizer.str_length)
    assert sample_result.sequences.shape == (batch_size, optimizer.str_length)
    assert sample_result.mixture_indices.shape == (batch_size,)


def test_find_first_occurrence_indices():
    """Test _find_first_occurrence_indices with handcrafted examples."""
    optimizer = IupacBaseOptimizer(
        str_length=10,
        alphabet_size=4,
        batch_size=100,
        num_templates=10,
        device="cpu",
    )

    # Case 1: Multiple occurrences of templates
    mixture_indices = torch.tensor([2, 5, 2, 7, 5, 2, 7])
    unique_templates = torch.tensor([2, 5, 7])
    first_indices = optimizer._find_first_occurrence_indices(
        mixture_indices, unique_templates
    )

    expected = torch.tensor([0, 1, 3])
    assert torch.all(
        first_indices == expected
    ), f"Expected {expected}, got {first_indices}"

    # Case 2: Each template appears once
    mixture_indices = torch.tensor([3, 1, 4])
    unique_templates = torch.tensor([1, 3, 4])
    first_indices = optimizer._find_first_occurrence_indices(
        mixture_indices, unique_templates
    )

    expected = torch.tensor([1, 0, 2])
    assert torch.all(
        first_indices == expected
    ), f"Expected {expected}, got {first_indices}"

    # Case 3: Templates in reverse order of appearance
    mixture_indices = torch.tensor([9, 9, 5, 5, 5, 1])
    unique_templates = torch.tensor([1, 5, 9])
    first_indices = optimizer._find_first_occurrence_indices(
        mixture_indices, unique_templates
    )

    expected = torch.tensor([5, 2, 0])
    assert torch.all(
        first_indices == expected
    ), f"Expected {expected}, got {first_indices}"

    # Case 4: Single template repeated
    mixture_indices = torch.tensor([7, 7, 7, 7])
    unique_templates = torch.tensor([7])
    first_indices = optimizer._find_first_occurrence_indices(
        mixture_indices, unique_templates
    )

    expected = torch.tensor([0])
    assert torch.all(
        first_indices == expected
    ), f"Expected {expected}, got {first_indices}"

    print("✓ _find_first_occurrence_indices tests passed")


def test_find_first_occurrence_with_more_templates_than_batch():
    """Test _find_first_occurrence_indices when num_templates > batch_size.

    This is an edge case where the number of possible templates exceeds the batch size.
    Only a subset of templates will appear in the batch, but the function should still
    correctly identify first occurrences without being affected by uninitialized default values.
    """
    optimizer = IupacBaseOptimizer(
        str_length=10,
        alphabet_size=4,
        batch_size=8,
        num_templates=50,  # More templates than batch size
        device="cpu",
    )

    # Create a small batch that only uses a few templates
    mixture_indices = torch.tensor([5, 12, 5, 30, 12, 5, 30, 45])
    unique_templates = torch.tensor([5, 12, 30, 45])
    first_indices = optimizer._find_first_occurrence_indices(
        mixture_indices, unique_templates
    )

    expected = torch.tensor([0, 1, 3, 7])
    assert torch.all(
        first_indices == expected
    ), f"Expected {expected}, got {first_indices}"

    for i, template_idx in enumerate(unique_templates):
        expected_first = expected[i].item()
        assert (
            mixture_indices[expected_first] == template_idx
        ), f"Template {template_idx} should first appear at index {expected_first}"

    print(
        "✓ _find_first_occurrence_indices with num_templates > batch_size test passed"
    )
