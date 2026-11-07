"""Test script to verify SlotConstrainedIupacOptimizer functionality."""

import os
import sys

import pytest
import torch

# Add parent directory to path to import from pgld module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies import SlotConstrainedIupacOptimizer


@pytest.fixture
def slot_constrained_optimizer():
    """Create a SlotConstrainedIupacOptimizer for testing."""
    return SlotConstrainedIupacOptimizer(
        str_length=20,
        alphabet_size=4,
        batch_size=100,
        num_templates=3,
        max_non_agct_slots=4,
        device="cpu",
    )


def test_slot_allocation_returns_correct_shape(slot_constrained_optimizer):
    """Test that slot allocation returns the correct shapes."""
    optimizer = slot_constrained_optimizer

    selected_bottles = optimizer.sample_slot_allocation()

    assert selected_bottles.shape == (optimizer.max_non_agct_slots,)

    # Verify that bottle indices are valid (should be in range 4-14)
    assert torch.all(selected_bottles >= 4)
    assert torch.all(selected_bottles <= 14)

    print(f"✓ Slot allocation shape test passed")


def test_slot_allocation_is_unique(slot_constrained_optimizer):
    """Test that selected bottles are unique (no duplicates)."""
    optimizer = slot_constrained_optimizer

    selected_bottles = optimizer.sample_slot_allocation()

    # Check uniqueness
    unique_bottles = torch.unique(selected_bottles)
    assert len(unique_bottles) == optimizer.max_non_agct_slots, (
        f"Expected {optimizer.max_non_agct_slots} unique bottles, "
        f"got {len(unique_bottles)}"
    )

    print(f"✓ Slot allocation uniqueness test passed")


def test_constrained_phi_probs_masks_non_agct(slot_constrained_optimizer):
    """Test that phi_probs properly masks non-selected bottles everywhere."""
    optimizer = slot_constrained_optimizer

    # Manually select some bottles (e.g., bottles 4, 7, 9, 10)
    selected_bottles = torch.tensor([4, 7, 9, 10])

    phi_probs = optimizer.get_constrained_phi_probs(selected_bottles)

    assert phi_probs.shape == (
        optimizer.num_templates,
        optimizer.str_length,
        optimizer.num_bottles,
    )

    sums = phi_probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    # Check that only selected bottles and AGCT have non-zero probability
    # at ALL templates and ALL positions
    for t in range(optimizer.num_templates):
        for p in range(optimizer.str_length):
            agct_probs = phi_probs[t, p, :4]
            assert agct_probs.sum() > 0, f"Position ({t},{p}) should allow AGCT"

            for bottle_idx in selected_bottles:
                assert (
                    phi_probs[t, p, bottle_idx] >= 0
                ), f"Selected bottle {bottle_idx} should be allowed at ({t},{p})"

            for bottle_idx in range(4, 15):
                if bottle_idx not in selected_bottles:
                    assert torch.allclose(
                        phi_probs[t, p, bottle_idx], torch.tensor(0.0), atol=1e-5
                    ), f"Non-selected bottle {bottle_idx} should be zero at ({t},{p})"

    print(f"✓ Constrained phi_probs masking test passed")


def test_sample_returns_correct_shape(slot_constrained_optimizer):
    """Test that sample() returns correct shapes including selected_bottles."""
    optimizer = slot_constrained_optimizer

    batch_size = 50
    sample_result = optimizer.sample(batch_size=batch_size)

    assert sample_result.sequences.shape == (batch_size, optimizer.str_length)
    assert sample_result.mixture_indices.shape == (batch_size,)
    assert sample_result.bottle_indices.shape == (batch_size, optimizer.str_length)
    assert sample_result.selected_bottles.shape == (optimizer.max_non_agct_slots,)
    assert torch.all(sample_result.selected_bottles >= 4) and torch.all(
        sample_result.selected_bottles <= 14
    )

    print(f"✓ Sample return shape test passed")


def test_constraint_is_enforced_in_sampling(slot_constrained_optimizer):
    """Test that only selected bottles appear in sampling."""
    optimizer = slot_constrained_optimizer

    batch_size = 200
    sample_result = optimizer.sample(batch_size=batch_size)

    allowed_bottles = set(sample_result.selected_bottles.tolist())
    allowed_bottles.update([0, 1, 2, 3])

    for seq_idx in range(batch_size):
        for pos in range(optimizer.str_length):
            bottle = sample_result.bottle_indices[seq_idx, pos].item()
            assert bottle in allowed_bottles, (
                f"Bottle {bottle} used at sequence {seq_idx}, position {pos} "
                f"but is not in allowed set {allowed_bottles}"
            )

    print(f"✓ Constraint enforcement test passed")
    print(f"  Selected bottles: {sample_result.selected_bottles.tolist()}")


def test_log_probs_computation(slot_constrained_optimizer):
    """Test that log_probs can be computed and have correct shape."""
    optimizer = slot_constrained_optimizer

    batch_size = 50
    sample_result = optimizer.sample(batch_size=batch_size)

    log_probs = optimizer.get_log_probs(sample_result)

    assert log_probs.shape == (batch_size,)

    assert torch.all(torch.isfinite(log_probs))

    assert torch.all(log_probs < 0)

    print(f"✓ Log probs computation test passed")


def test_gradients_flow_through_slot_allocation():
    """Test that gradients can flow through the slot allocation parameters."""
    optimizer = SlotConstrainedIupacOptimizer(
        str_length=10,
        alphabet_size=4,
        batch_size=50,
        num_templates=2,
        max_non_agct_slots=3,
        device="cpu",
    )

    assert optimizer.slot_allocation_logits.requires_grad

    assert not optimizer.latent_w_mix.requires_grad

    sample_result = optimizer.sample()
    log_probs = optimizer.get_log_probs(sample_result)
    batch_log_probs = optimizer.get_batch_log_probs(sample_result)

    # Create a dummy loss
    loss = -log_probs.sum() - batch_log_probs
    loss.backward()

    assert optimizer.slot_allocation_logits.grad is not None
    assert not torch.all(optimizer.slot_allocation_logits.grad == 0)

    assert optimizer.latent_phi.grad is not None

    assert optimizer.latent_w_mix.grad is None

    print(f"✓ Gradient flow test passed")
