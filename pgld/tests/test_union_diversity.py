"""Tests for union diversity calculations."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diversity_utils import compute_mask_diversity
from library_size_estimator import (
    compute_union_diversity,
    compute_diversity_exhaustive,
    compute_total_library_diversity,
    print_total_library_diversity,
)
from policies import IupacBaseOptimizer
from policy_init import init_from_iupac_sequence


class TestComputeMaskDiversity:
    """Tests for compute_mask_diversity function."""

    def test_single_char_per_position(self):
        """Mask with exactly 1 allowed char per position -> diversity = 1."""
        mask = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        assert compute_mask_diversity(mask) == 1.0

    def test_all_chars_allowed(self):
        """All 4 chars allowed at each of 3 positions -> 4^3 = 64."""
        mask = np.ones((3, 4))
        assert compute_mask_diversity(mask) == 64.0

    def test_partial_degeneracy(self):
        """Mixed: 1*2*4 = 8."""
        mask = np.array([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]])
        assert compute_mask_diversity(mask) == 8.0

    def test_zero_diversity(self):
        """If any position has no allowed chars, diversity = 0."""
        mask = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]])
        assert compute_mask_diversity(mask) == 0.0

    def test_empty_mask(self):
        """Empty mask has diversity 1 (product of empty list)."""
        mask = np.array([]).reshape(0, 4)
        assert compute_mask_diversity(mask) == 1.0


class TestComputeUnionDiversity:
    """Tests for compute_union_diversity function."""

    def test_single_mask(self):
        """Union of single mask equals its diversity."""
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])  # 2 * 4 = 8
        assert compute_union_diversity([mask]) == 8.0

    def test_identical_masks(self):
        """Union of identical masks equals single mask diversity (not 2x)."""
        mask = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
        assert compute_union_diversity([mask, mask.copy()]) == 8.0

    def test_disjoint_masks(self):
        """Disjoint masks: union = sum."""
        # Mask 1: only A allowed at position 0
        mask1 = np.array([[1, 0, 0, 0], [1, 1, 1, 1]])
        # Mask 2: only C allowed at position 0
        mask2 = np.array([[0, 1, 0, 0], [1, 1, 1, 1]])
        # No overlap at position 0, so union = 4 + 4 = 8
        assert compute_union_diversity([mask1, mask2]) == 8.0

    def test_overlapping_masks(self):
        """Overlapping masks: union < sum."""
        # Mask 1: A or C at pos 0, all at pos 1 -> 2*4=8
        mask1 = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])
        # Mask 2: C or G at pos 0, all at pos 1 -> 2*4=8
        mask2 = np.array([[0, 1, 1, 0], [1, 1, 1, 1]])
        # Intersection: C at pos 0, all at pos 1 -> 1*4=4
        # Union = 8 + 8 - 4 = 12
        assert compute_union_diversity([mask1, mask2]) == 12.0

    def test_three_overlapping_masks(self):
        """Three masks with various overlaps."""
        # All masks: 2 options at pos 0, 4 options at pos 1
        mask_a = np.array([[1, 1, 0, 0], [1, 1, 1, 1]])  # A,C at pos 0
        mask_b = np.array([[0, 1, 1, 0], [1, 1, 1, 1]])  # C,G at pos 0
        mask_c = np.array([[0, 0, 1, 1], [1, 1, 1, 1]])  # G,T at pos 0

        # |A| = |B| = |C| = 8
        # |A ∩ B| = C only at pos 0 = 4
        # |B ∩ C| = G only at pos 0 = 4
        # |A ∩ C| = empty (no common char at pos 0) = 0
        # |A ∩ B ∩ C| = 0
        # Union = 8 + 8 + 8 - 4 - 4 - 0 + 0 = 16
        assert compute_union_diversity([mask_a, mask_b, mask_c]) == 16.0

    def test_empty_list(self):
        """Empty list has diversity 0."""
        assert compute_union_diversity([]) == 0.0

    def test_all_zero_mask(self):
        """Mask with all zeros has diversity 0."""
        mask = np.zeros((3, 4))
        assert compute_union_diversity([mask]) == 0.0


class TestComputeTotalLibraryDiversity:
    """Tests for compute_total_library_diversity with actual policies."""

    def test_single_template_skips_union(self):
        """N=1 should skip union computation."""
        policy = IupacBaseOptimizer(
            str_length=6,
            alphabet_size=4,
            batch_size=10,
            num_templates=1,
        )
        init_from_iupac_sequence(policy, "ACGTNN", epsilon=0.0)

        result = compute_total_library_diversity(policy, is_peptide_landscape=False)

        assert result["union_diversity"] is None
        assert "N=1" in result["union_skipped_reason"]
        assert result["sum_diversity"] == 16.0  # ACGT (1*1*1*1) * NN (4*4) = 16

    def test_two_identical_templates(self):
        """Two identical templates should have union = individual."""
        policy = IupacBaseOptimizer(
            str_length=4,
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
        )
        init_from_iupac_sequence(policy, "NNNN", epsilon=0.0)

        result = compute_total_library_diversity(policy, is_peptide_landscape=False)

        # Each template has diversity 256, sum = 512
        assert result["sum_diversity"] == 512.0
        # But they're identical, so union = 256
        assert result["union_diversity"] == 256.0

    def test_two_disjoint_templates(self):
        """Two templates with no overlap should have union = sum."""
        policy = IupacBaseOptimizer(
            str_length=4,
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
        )
        # Initialize both templates the same first
        init_from_iupac_sequence(policy, "AAAA", epsilon=0.0)

        # Manually modify template 1 to be TTTT (completely disjoint from AAAA)
        with torch.no_grad():
            # Template 1: Make position 0 select T bottle (index 3) instead of A (index 0)
            policy.latent_phi[1, :, :] = -100.0  # Low prob for all bottles
            policy.latent_phi[1, :, 3] = 100.0  # High prob for T bottle

        result = compute_total_library_diversity(policy, is_peptide_landscape=False)

        # Each template has diversity 1, sum = 2, union = 2 (disjoint)
        assert result["sum_diversity"] == 2.0
        assert result["union_diversity"] == 2.0

    def test_peptide_mode(self):
        """Test with amino acid diversity (excludes stop codons)."""
        policy = IupacBaseOptimizer(
            str_length=6,  # 2 codons
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
        )
        # ATG = Met (1 AA), NNN = 20 amino acids (stop codon excluded)
        init_from_iupac_sequence(policy, "ATGNNN", epsilon=0.0)

        result = compute_total_library_diversity(policy, is_peptide_landscape=True)

        # Each template: 1 * 20 = 20, sum = 40
        # But identical, so union = 20
        assert result["sum_diversity"] == 40.0
        assert result["union_diversity"] == 20.0


class TestPrintTotalLibraryDiversity:
    """Tests for print_total_library_diversity function."""

    def test_non_dna_alphabet_skipped(self):
        """Non-DNA alphabet should be skipped."""
        policy = IupacBaseOptimizer(
            str_length=4,
            alphabet_size=4,
            batch_size=10,
            num_templates=1,
        )
        result = print_total_library_diversity(policy, alphabet=["X", "Y", "Z", "W"])
        assert result["union_skipped_reason"] == "Non-DNA alphabet"

    def test_output_includes_sum(self, capsys):
        """Output should always include sum diversity."""
        policy = IupacBaseOptimizer(
            str_length=4,
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
        )
        init_from_iupac_sequence(policy, "NNNN", epsilon=0.0)

        result = print_total_library_diversity(policy, alphabet=["A", "C", "G", "T"])

        assert result["sum_diversity"] == 512.0
        assert result["union_diversity"] == 256.0


class TestStopCodonExclusion:
    """Tests verifying stop codons are excluded from peptide diversity."""

    def test_nnn_gives_20_not_21(self):
        """NNN should give 20 amino acids, not 21 (stop excluded)."""
        policy = IupacBaseOptimizer(
            str_length=3,  # 1 codon
            alphabet_size=4,
            batch_size=10,
            num_templates=1,
        )
        init_from_iupac_sequence(policy, "NNN", epsilon=0.0)

        result = compute_total_library_diversity(policy, is_peptide_landscape=True)

        # NNN encodes all 64 codons -> 20 amino acids (stop excluded)
        assert result["individual_diversities"][0] == 20.0

    def test_dna_mode_unaffected(self):
        """DNA mode should still count all 4 bases."""
        policy = IupacBaseOptimizer(
            str_length=3,
            alphabet_size=4,
            batch_size=10,
            num_templates=1,
        )
        init_from_iupac_sequence(policy, "NNN", epsilon=0.0)

        result = compute_total_library_diversity(policy, is_peptide_landscape=False)

        # NNN in DNA mode = 4^3 = 64
        assert result["individual_diversities"][0] == 64.0


class TestEdgeCases:
    """Edge case tests."""

    def test_very_long_sequence(self):
        """Test with a longer sequence."""
        policy = IupacBaseOptimizer(
            str_length=30,
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
        )
        # NNNNNN...NNN (10 N's) followed by fixed bases
        init_from_iupac_sequence(policy, "N" * 10 + "A" * 20, epsilon=0.0)

        result = compute_total_library_diversity(policy, is_peptide_landscape=False)

        # Each template: 4^10 * 1^20
        expected_individual = 4**10
        assert result["individual_diversities"][0] == expected_individual
        assert result["sum_diversity"] == 2 * expected_individual
        # Identical templates, so union = individual
        assert result["union_diversity"] == expected_individual


class TestExhaustiveComparison:
    """Compare inclusion-exclusion against exhaustive enumeration."""

    def _generate_random_mask(
        self, length: int, alphabet_size: int, density: float, rng: np.random.Generator
    ) -> np.ndarray:
        """
        Generate a random boolean mask.

        Args:
            length: Sequence length
            alphabet_size: Size of alphabet (4 for DNA, 20 for AA)
            density: Probability each position allows each character
            rng: Random number generator

        Returns:
            Boolean mask of shape (length, alphabet_size)
        """
        mask = (rng.random((length, alphabet_size)) < density).astype(int)
        # Ensure at least one character is allowed at each position
        for pos in range(length):
            if mask[pos].sum() == 0:
                mask[pos, rng.integers(0, alphabet_size)] = 1
        return mask

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
    def test_random_dna_5_templates(self, seed):
        """Test 5 random DNA templates against exhaustive enumeration."""
        rng = np.random.default_rng(seed)
        length = 4  # Keep short for exhaustive enumeration
        alphabet_size = 4  # DNA

        masks = [
            self._generate_random_mask(length, alphabet_size, density=0.5, rng=rng)
            for _ in range(5)
        ]

        inclusion_exclusion_result = compute_union_diversity(masks)
        exhaustive_result = compute_diversity_exhaustive(masks)

        assert inclusion_exclusion_result == exhaustive_result, (
            f"Seed {seed}: inclusion-exclusion gave {inclusion_exclusion_result}, "
            f"exhaustive gave {exhaustive_result}"
        )

    @pytest.mark.parametrize("seed", [42, 123, 456, 789, 1337])
    def test_random_aa_5_templates(self, seed):
        """Test 5 random amino acid templates against exhaustive enumeration."""
        rng = np.random.default_rng(seed)
        length = 3  # Keep short for exhaustive enumeration (20^3 = 8000 max)
        alphabet_size = 20  # Amino acids (excluding stop)

        masks = [
            self._generate_random_mask(length, alphabet_size, density=0.3, rng=rng)
            for _ in range(5)
        ]

        inclusion_exclusion_result = compute_union_diversity(masks)
        exhaustive_result = compute_diversity_exhaustive(masks)

        assert inclusion_exclusion_result == exhaustive_result, (
            f"Seed {seed}: inclusion-exclusion gave {inclusion_exclusion_result}, "
            f"exhaustive gave {exhaustive_result}"
        )

    @pytest.mark.parametrize("num_templates", [2, 3, 5, 7, 10])
    def test_varying_template_counts(self, num_templates):
        """Test different numbers of templates."""
        rng = np.random.default_rng(42 + num_templates)
        length = 4
        alphabet_size = 4

        masks = [
            self._generate_random_mask(length, alphabet_size, density=0.5, rng=rng)
            for _ in range(num_templates)
        ]

        inclusion_exclusion_result = compute_union_diversity(masks)
        exhaustive_result = compute_diversity_exhaustive(masks)

        assert inclusion_exclusion_result == exhaustive_result, (
            f"{num_templates} templates: inclusion-exclusion gave "
            f"{inclusion_exclusion_result}, exhaustive gave {exhaustive_result}"
        )

    @pytest.mark.parametrize("density", [0.2, 0.4, 0.6, 0.8])
    def test_varying_densities(self, density):
        """Test different mask densities (sparsity of allowed chars)."""
        rng = np.random.default_rng(int(density * 1000))
        length = 4
        alphabet_size = 4
        num_templates = 5

        masks = [
            self._generate_random_mask(length, alphabet_size, density=density, rng=rng)
            for _ in range(num_templates)
        ]

        inclusion_exclusion_result = compute_union_diversity(masks)
        exhaustive_result = compute_diversity_exhaustive(masks)

        assert inclusion_exclusion_result == exhaustive_result, (
            f"Density {density}: inclusion-exclusion gave {inclusion_exclusion_result}, "
            f"exhaustive gave {exhaustive_result}"
        )

    def test_high_overlap_templates(self):
        """Test templates with high overlap (many shared allowed chars)."""
        rng = np.random.default_rng(999)
        length = 5
        alphabet_size = 4
        num_templates = 5

        # Create a base mask with high density
        base_mask = self._generate_random_mask(
            length, alphabet_size, density=0.8, rng=rng
        )

        # Create templates that are slight variations of the base
        masks = []
        for _ in range(num_templates):
            mask = base_mask.copy()
            # Randomly flip a few bits
            for _ in range(2):
                pos = rng.integers(0, length)
                char = rng.integers(0, alphabet_size)
                mask[pos, char] = 1 - mask[pos, char]
            # Ensure at least one char per position
            for pos in range(length):
                if mask[pos].sum() == 0:
                    mask[pos, rng.integers(0, alphabet_size)] = 1
            masks.append(mask)

        inclusion_exclusion_result = compute_union_diversity(masks)
        exhaustive_result = compute_diversity_exhaustive(masks)

        assert inclusion_exclusion_result == exhaustive_result

    def test_low_overlap_templates(self):
        """Test templates with low overlap (mostly disjoint)."""
        rng = np.random.default_rng(888)
        length = 4
        alphabet_size = 4
        num_templates = 4

        masks = []
        for t in range(num_templates):
            mask = np.zeros((length, alphabet_size), dtype=int)
            # Each template primarily uses one character at position 0
            mask[0, t % alphabet_size] = 1
            # Random chars at other positions
            for pos in range(1, length):
                num_allowed = rng.integers(1, alphabet_size + 1)
                allowed = rng.choice(alphabet_size, size=num_allowed, replace=False)
                mask[pos, allowed] = 1
            masks.append(mask)

        inclusion_exclusion_result = compute_union_diversity(masks)
        exhaustive_result = compute_diversity_exhaustive(masks)

        assert inclusion_exclusion_result == exhaustive_result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
