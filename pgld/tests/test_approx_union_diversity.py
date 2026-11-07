"""Tests for approximate union diversity using Karp-Luby algorithm."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from approx_union_diversity import (
    sample_from_mask,
    sequence_in_mask,
    estimate_union_diversity_karp_luby,
)
from diversity_utils import compute_mask_diversity
from library_size_estimator import (
    compute_union_diversity,
    compute_diversity_exhaustive,
)


class TestSampleFromMask:
    """Tests for sample_from_mask function."""

    def test_samples_only_allowed_chars(self):
        """Sampled sequences should only contain allowed characters."""
        rng = np.random.default_rng(42)
        mask = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]])

        for _ in range(100):
            seq = sample_from_mask(mask, rng)
            assert seq[0] == 0  # Only A allowed
            assert seq[1] in [1, 2]  # Only C or G allowed
            assert seq[2] in [0, 1, 2, 3]  # All allowed

    def test_uniform_sampling(self):
        """Sampling should be approximately uniform over allowed characters."""
        rng = np.random.default_rng(42)
        mask = np.array([[1, 1, 0, 0]])  # Only A and C allowed

        counts = {0: 0, 1: 0}
        num_samples = 10000
        for _ in range(num_samples):
            seq = sample_from_mask(mask, rng)
            counts[seq[0]] += 1

        ratio = counts[0] / counts[1]
        assert 0.9 < ratio < 1.1, f"Expected ~50/50 split, got {counts}"


class TestSequenceInMask:
    """Tests for sequence_in_mask function."""

    def test_valid_sequence(self):
        """Valid sequences should return True."""
        mask = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]])
        seq = np.array([0, 1, 2])  # A, C, G
        assert sequence_in_mask(seq, mask) is True

    def test_invalid_sequence(self):
        """Invalid sequences should return False."""
        mask = np.array([[1, 0, 0, 0], [0, 1, 1, 0], [1, 1, 1, 1]])
        seq = np.array([1, 1, 2])  # C at pos 0 is not allowed
        assert sequence_in_mask(seq, mask) is False

    def test_boundary_cases(self):
        """Test edge cases."""
        mask = np.array([[1, 0, 0, 0]])
        assert sequence_in_mask(np.array([0]), mask) is True
        assert sequence_in_mask(np.array([1]), mask) is False
        assert sequence_in_mask(np.array([2]), mask) is False
        assert sequence_in_mask(np.array([3]), mask) is False


def _generate_random_mask(
    length: int, alphabet_size: int, density: float, rng: np.random.Generator
) -> np.ndarray:
    """Generate a random boolean mask with at least one allowed char per position."""
    mask = (rng.random((length, alphabet_size)) < density).astype(int)
    for pos in range(length):
        if mask[pos].sum() == 0:
            mask[pos, rng.integers(0, alphabet_size)] = 1
    return mask


class TestKarpLubyAccuracy:
    """Tests comparing Karp-Luby approximation against exact computation."""

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
    def test_vs_exact_5_templates(self, seed):
        """Karp-Luby should be within 2x of exact for 5 templates."""
        rng = np.random.default_rng(seed)
        masks = [_generate_random_mask(4, 4, density=0.5, rng=rng) for _ in range(5)]

        exact = compute_union_diversity(masks)
        approx, std_err = estimate_union_diversity_karp_luby(
            masks, min_samples=50000, seed=seed
        )

        if exact > 0:
            ratio = approx / exact
            assert (
                0.5 <= ratio <= 2.0
            ), f"Seed {seed}: approx={approx:.1f}, exact={exact:.1f}, ratio={ratio:.2f}"

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
    def test_vs_exact_10_templates(self, seed):
        """Karp-Luby should be within 2x of exact for 10 templates."""
        rng = np.random.default_rng(seed)
        masks = [_generate_random_mask(4, 4, density=0.5, rng=rng) for _ in range(10)]

        exact = compute_union_diversity(masks)
        approx, std_err = estimate_union_diversity_karp_luby(
            masks, min_samples=50000, seed=seed
        )

        if exact > 0:
            ratio = approx / exact
            assert (
                0.5 <= ratio <= 2.0
            ), f"Seed {seed}: approx={approx:.1f}, exact={exact:.1f}, ratio={ratio:.2f}"

    def test_vs_exhaustive_enumeration(self):
        """Compare against exhaustive enumeration for ground truth."""
        rng = np.random.default_rng(999)
        masks = [_generate_random_mask(3, 4, density=0.4, rng=rng) for _ in range(5)]

        exhaustive = compute_diversity_exhaustive(masks)
        approx, std_err = estimate_union_diversity_karp_luby(
            masks, min_samples=50000, seed=999
        )

        if exhaustive > 0:
            ratio = approx / exhaustive
            assert (
                0.5 <= ratio <= 2.0
            ), f"approx={approx:.1f}, exhaustive={exhaustive}, ratio={ratio:.2f}"

    def test_accuracy_over_many_trials(self):
        """At least 90% of trials should be within 2x of exact."""
        rng = np.random.default_rng(12345)
        masks = [_generate_random_mask(4, 4, density=0.5, rng=rng) for _ in range(8)]
        exact = compute_union_diversity(masks)

        if exact == 0:
            pytest.skip("Exact is zero, cannot test ratios")

        num_trials = 50
        within_2x = 0
        for trial in range(num_trials):
            approx, _ = estimate_union_diversity_karp_luby(
                masks, min_samples=20000, seed=trial
            )
            ratio = approx / exact
            if 0.5 <= ratio <= 2.0:
                within_2x += 1

        success_rate = within_2x / num_trials
        assert (
            success_rate >= 0.90
        ), f"Only {within_2x}/{num_trials} ({success_rate:.0%}) within 2x of exact"


class TestKarpLubyEdgeCases:
    """Edge case tests for Karp-Luby algorithm."""

    def test_empty_masks_list(self):
        """Empty list should return 0."""
        estimate, std_err = estimate_union_diversity_karp_luby([])
        assert estimate == 0.0
        assert std_err == 0.0

    def test_single_mask(self):
        """Single mask should approximate its exact diversity."""
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])  # 2 * 4 = 8
        exact = compute_mask_diversity(mask)

        estimate, std_err = estimate_union_diversity_karp_luby(
            [mask], min_samples=50000, seed=42
        )

        ratio = estimate / exact
        assert 0.8 <= ratio <= 1.2, f"Single mask: expected ~{exact}, got {estimate}"

    def test_identical_masks(self):
        """Identical masks should have union = individual diversity."""
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
        exact = compute_mask_diversity(mask)  # Should equal union

        estimate, std_err = estimate_union_diversity_karp_luby(
            [mask, mask.copy(), mask.copy()], min_samples=50000, seed=42
        )

        ratio = estimate / exact
        assert (
            0.8 <= ratio <= 1.2
        ), f"Identical masks: expected ~{exact}, got {estimate}"

    def test_disjoint_masks(self):
        """Disjoint masks should have union = sum of individuals."""
        mask1 = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])  # A at pos 0
        mask2 = np.array([[0, 1, 0, 0], [1, 1, 1, 1]])  # C at pos 0
        mask3 = np.array([[0, 0, 1, 0], [1, 1, 1, 1]])  # G at pos 0

        sum_div = sum(compute_mask_diversity(m) for m in [mask1, mask2, mask3])
        exact = compute_union_diversity([mask1, mask2, mask3])
        assert exact == sum_div  # Disjoint, so union = sum

        estimate, std_err = estimate_union_diversity_karp_luby(
            [mask1, mask2, mask3], min_samples=50000, seed=42
        )

        ratio = estimate / exact
        assert 0.8 <= ratio <= 1.2, f"Disjoint masks: expected {exact}, got {estimate}"

    def test_zero_diversity_mask(self):
        """Mask with zero diversity (empty position) should be handled."""
        mask1 = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
        mask2 = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])  # Zero diversity

        estimate, std_err = estimate_union_diversity_karp_luby(
            [mask1, mask2], min_samples=10000, seed=42
        )

        exact = compute_union_diversity([mask1, mask2])
        if exact > 0:
            ratio = estimate / exact
            assert 0.5 <= ratio <= 2.0


class TestLargeTemplateCount:
    """Tests for large numbers of templates (>20)."""

    def test_50_templates_runs(self):
        """Should complete for 50 templates without error."""
        rng = np.random.default_rng(42)
        masks = [_generate_random_mask(4, 4, density=0.5, rng=rng) for _ in range(50)]

        estimate, std_err = estimate_union_diversity_karp_luby(
            masks, min_samples=20000, seed=42
        )

        assert estimate >= 0
        assert std_err >= 0

    def test_100_templates_runs(self):
        """Should complete for 100 templates without error."""
        rng = np.random.default_rng(42)
        masks = [_generate_random_mask(4, 4, density=0.5, rng=rng) for _ in range(100)]

        estimate, std_err = estimate_union_diversity_karp_luby(
            masks, min_samples=20000, seed=42
        )

        assert estimate >= 0
        assert std_err >= 0
        assert estimate <= sum(compute_mask_diversity(m) for m in masks)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
