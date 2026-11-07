"""Test script for IupacPlusOptimizer."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies import IupacPlusOptimizer, IupacBaseOptimizer


@pytest.fixture
def iupac_plus_optimizer():
    """Create an IupacPlusOptimizer for testing."""
    return IupacPlusOptimizer(
        str_length=20,
        alphabet_size=4,
        batch_size=100,
        num_templates=3,
        num_custom_bottles=5,
        device="cpu",
    )


def test_iupac_plus_has_correct_num_bottles(iupac_plus_optimizer):
    """Test that IupacPlusOptimizer has 15 IUPAC + N custom bottles."""
    assert iupac_plus_optimizer.num_bottles == 20  # 15 + 5
    assert iupac_plus_optimizer.num_custom_bottles == 5


def test_iupac_plus_bottle_probs_shape(iupac_plus_optimizer):
    """Test that get_bottle_probs returns correct shape."""
    bottle_probs = iupac_plus_optimizer.get_bottle_probs(tau=1.0)
    assert bottle_probs.shape == (20, 4)  # 20 bottles, 4 bases


def test_iupac_plus_fixed_bottles_unchanged(iupac_plus_optimizer):
    """Test that first 15 bottles match fixed IUPAC probabilities."""
    bottle_probs = iupac_plus_optimizer.get_bottle_probs(tau=1.0)
    fixed_iupac_probs = iupac_plus_optimizer._iupac_bottle_probs

    assert torch.allclose(bottle_probs[:15], fixed_iupac_probs)


def test_iupac_plus_custom_bottles_are_learnable(iupac_plus_optimizer):
    """Test that custom bottles have learnable parameters."""
    assert iupac_plus_optimizer.latent_v_custom.requires_grad
    assert iupac_plus_optimizer._iupac_bottle_probs.requires_grad is False


def test_iupac_plus_custom_bottles_are_probabilities(iupac_plus_optimizer):
    """Test that custom bottles are valid probability distributions."""
    bottle_probs = iupac_plus_optimizer.get_bottle_probs(tau=1.0)
    custom_probs = bottle_probs[15:]  # Last 5 bottles

    assert torch.allclose(custom_probs.sum(dim=1), torch.ones(5))
    assert torch.all(custom_probs >= 0)
    assert torch.all(custom_probs <= 1)


def test_iupac_plus_samples_correctly(iupac_plus_optimizer):
    """Test that sampling produces valid results."""
    sample_result = iupac_plus_optimizer.sample(batch_size=50)

    assert sample_result.sequences.shape == (50, 20)
    assert sample_result.mixture_indices.shape == (50,)
    assert sample_result.bottle_indices.shape == (50, 20)
    assert torch.all(sample_result.bottle_indices >= 0)
    assert torch.all(sample_result.bottle_indices < 20)


def test_iupac_plus_can_use_custom_bottles(iupac_plus_optimizer):
    """Test that sampling actually uses custom bottles (not just IUPAC)."""
    sample_result = iupac_plus_optimizer.sample(batch_size=1000)

    uses_custom_bottles = torch.any(sample_result.bottle_indices >= 15)
    assert uses_custom_bottles, "Expected some sequences to use custom bottles"


def test_iupac_plus_temperature_affects_custom_bottles():
    """Test that temperature parameter affects custom bottle probabilities."""
    optimizer = IupacPlusOptimizer(
        str_length=10,
        alphabet_size=4,
        batch_size=100,
        num_templates=2,
        num_custom_bottles=3,
        device="cpu",
    )

    probs_low_temp = optimizer.get_bottle_probs(tau=0.1)
    probs_high_temp = optimizer.get_bottle_probs(tau=10.0)

    custom_low = probs_low_temp[15:]
    custom_high = probs_high_temp[15:]

    entropy_low = -torch.sum(custom_low * torch.log(custom_low + 1e-10), dim=1).mean()
    entropy_high = -torch.sum(
        custom_high * torch.log(custom_high + 1e-10), dim=1
    ).mean()

    assert entropy_high > entropy_low, "Higher temperature should increase entropy"


def test_iupac_plus_vs_iupac_base():
    """Test that IupacPlusOptimizer extends IupacBaseOptimizer correctly."""
    iupac_base = IupacBaseOptimizer(
        str_length=15,
        alphabet_size=4,
        batch_size=50,
        num_templates=2,
        device="cpu",
    )

    iupac_plus = IupacPlusOptimizer(
        str_length=15,
        alphabet_size=4,
        batch_size=50,
        num_templates=2,
        num_custom_bottles=0,
        device="cpu",
    )

    base_bottles = iupac_base.get_bottle_probs()
    plus_bottles = iupac_plus.get_bottle_probs()[:15]

    assert torch.allclose(base_bottles, plus_bottles)


def test_iupac_plus_different_num_custom_bottles():
    """Test IupacPlusOptimizer with different numbers of custom bottles."""
    for num_custom in [1, 3, 10]:
        optimizer = IupacPlusOptimizer(
            str_length=10,
            alphabet_size=4,
            batch_size=50,
            num_templates=2,
            num_custom_bottles=num_custom,
            device="cpu",
        )

        assert optimizer.num_bottles == 15 + num_custom
        assert optimizer.get_bottle_probs().shape == (15 + num_custom, 4)

        sample_result = optimizer.sample(batch_size=50)
        assert torch.all(sample_result.bottle_indices < 15 + num_custom)
