"""Tests for policy initialization."""

import os
import sys

import numpy as np
import pytest
import torch

# Add parent directory to path to import from pgld module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import AA_TO_CODONS, AMINO_ACIDS, CODON_TABLE
from policies import ArbitraryBaseOptimizer, IupacBaseOptimizer, IupacPlusOptimizer
from policy_init import (
    aa_to_dna_random,
    init_from_fasta_sequences,
    init_from_iupac_sequence,
)


def test_init_from_iupac_basic():
    """Test basic IUPAC initialization with simple sequence."""
    policy = ArbitraryBaseOptimizer(
        str_length=6,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    # Initialize with "AACCGG" (all definite bases)
    init_from_iupac_sequence(policy, "AACCGG", epsilon=0.1)

    # Get probabilities
    probs, _ = policy.get_probs(tau=1.0)
    probs = probs[0]  # First template

    # Check that positions are dominated by correct base
    # Position 0 and 1 should be ~90% A
    assert probs[0, 0] > 0.7  # A at position 0
    assert probs[1, 0] > 0.7  # A at position 1

    # Position 2 and 3 should be ~90% C
    assert probs[2, 1] > 0.7  # C at position 2
    assert probs[3, 1] > 0.7  # C at position 3

    # Position 4 and 5 should be ~90% G
    assert probs[4, 2] > 0.7  # G at position 4
    assert probs[5, 2] > 0.7  # G at position 5


def test_init_from_iupac_degenerate():
    """Test IUPAC initialization with degenerate codes."""
    policy = ArbitraryBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    # Initialize with "NNKK" (N = any base, K = G or T)
    init_from_iupac_sequence(policy, "NNKK", epsilon=0.1)

    probs, _ = policy.get_probs(tau=1.0)
    probs = probs[0]

    # Position 0 and 1: N means all 4 bases roughly equal
    # With epsilon=0.1, should be ~(0.9/4 + 0.1/4) = 0.25 for each
    for base_idx in range(4):
        assert 0.15 < probs[0, base_idx] < 0.35
        assert 0.15 < probs[1, base_idx] < 0.35

    # Position 2 and 3: K means G or T should dominate (A and C should be low)
    # G (idx 2) and T (idx 3) should be ~45% each
    assert probs[2, 2] > 0.3  # G at position 2
    assert probs[2, 3] > 0.3  # T at position 2
    assert probs[2, 0] < 0.2  # A should be low
    assert probs[2, 1] < 0.2  # C should be low


def test_init_multiple_templates():
    """Test that all templates get initialized identically."""
    policy = ArbitraryBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=3,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "ACGT", epsilon=0.1)

    probs, _ = policy.get_probs(tau=1.0)

    # All templates should have same initialization
    assert torch.allclose(probs[0], probs[1], atol=1e-6)
    assert torch.allclose(probs[1], probs[2], atol=1e-6)


def test_init_epsilon_zero():
    """Test with epsilon=0 gives deterministic initialization."""
    policy = ArbitraryBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "ACGT", epsilon=0.0)

    probs, _ = policy.get_probs(tau=1.0)
    probs = probs[0]

    # Should be ~100% on correct base (allowing for softmax numerical issues)
    assert probs[0, 0] > 0.95  # A
    assert probs[1, 1] > 0.95  # C
    assert probs[2, 2] > 0.95  # G
    assert probs[3, 3] > 0.95  # T


def test_init_wrong_length():
    """Test that wrong sequence length raises error."""
    policy = ArbitraryBaseOptimizer(
        str_length=6,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="doesn't match"):
        init_from_iupac_sequence(policy, "ACGT", epsilon=0.1)


def test_init_invalid_base():
    """Test that invalid IUPAC base raises error."""
    policy = ArbitraryBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    with pytest.raises(ValueError, match="Invalid IUPAC base"):
        init_from_iupac_sequence(policy, "ACGX", epsilon=0.1)


def test_init_iupac_base_optimizer():
    """Test initialization of IupacBaseOptimizer with bottle selection."""
    policy = IupacBaseOptimizer(
        str_length=6,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "AACGTN", epsilon=0.1)

    phi_probs = torch.softmax(policy.latent_phi / 1.0, dim=-1)
    phi_probs = phi_probs[0]

    assert phi_probs[0, 0] > 0.7
    assert phi_probs[1, 0] > 0.7
    assert phi_probs[2, 1] > 0.7
    assert phi_probs[3, 2] > 0.7
    assert phi_probs[4, 3] > 0.7

    assert phi_probs[5, 14] > 0.7
    for bottle_idx in range(14):
        expected_low_prob = 0.1 / 15
        assert phi_probs[5, bottle_idx] < 0.02
        assert phi_probs[5, bottle_idx] > 0.001


def test_init_iupac_base_degenerate_codes():
    """Test IupacBaseOptimizer with degenerate IUPAC codes."""
    policy = IupacBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "RWSM", epsilon=0.1)

    phi_probs = torch.softmax(policy.latent_phi / 1.0, dim=-1)
    phi_probs = phi_probs[0]

    assert phi_probs[0, 8] > 0.7
    assert phi_probs[1, 7] > 0.7
    assert phi_probs[2, 5] > 0.7
    assert phi_probs[3, 4] > 0.7


def test_init_iupac_plus_optimizer():
    """Test initialization of IupacPlusOptimizer with custom bottles."""
    policy = IupacPlusOptimizer(
        str_length=6,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        num_custom_bottles=5,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "AACGTN", epsilon=0.1)

    phi_probs = torch.softmax(policy.latent_phi / 1.0, dim=-1)
    phi_probs = phi_probs[0]

    assert phi_probs[0, 0] > 0.7
    assert phi_probs[1, 0] > 0.7
    assert phi_probs[2, 1] > 0.7
    assert phi_probs[3, 2] > 0.7
    assert phi_probs[4, 3] > 0.7

    custom_probs_initialized = not torch.allclose(
        policy.latent_v_custom, torch.zeros_like(policy.latent_v_custom)
    )
    assert custom_probs_initialized


def test_init_iupac_plus_custom_bottles_uniform():
    """Test that IupacPlusOptimizer custom bottles get uniform init."""
    policy = IupacPlusOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        num_custom_bottles=3,
        device="cpu",
    )

    policy.latent_v_custom.data.zero_()

    init_from_iupac_sequence(policy, "ACGT", epsilon=0.1)

    assert torch.all(policy.latent_v_custom >= -1.0)
    assert torch.all(policy.latent_v_custom <= 1.0)
    assert not torch.allclose(
        policy.latent_v_custom, torch.zeros_like(policy.latent_v_custom)
    )


def test_init_bottle_based_epsilon_zero():
    """Test bottle-based init with epsilon=0 gives deterministic bottles."""
    policy = IupacBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "ACGT", epsilon=0.0)

    phi_probs = torch.softmax(policy.latent_phi / 1.0, dim=-1)
    phi_probs = phi_probs[0]

    assert phi_probs[0, 0] > 0.95
    assert phi_probs[1, 1] > 0.95
    assert phi_probs[2, 2] > 0.95
    assert phi_probs[3, 3] > 0.95


def test_init_bottle_based_multiple_templates():
    """Test that all templates get initialized identically for bottle-based."""
    policy = IupacBaseOptimizer(
        str_length=4,
        alphabet_size=4,
        batch_size=10,
        num_templates=3,
        device="cpu",
    )

    init_from_iupac_sequence(policy, "ACGT", epsilon=0.1)

    assert torch.allclose(policy.latent_phi[0], policy.latent_phi[1], atol=1e-6)
    assert torch.allclose(policy.latent_phi[1], policy.latent_phi[2], atol=1e-6)


def test_init_unknown_policy_type():
    """Test that initializing unsupported policy type raises error."""

    class UnknownPolicy:
        def __init__(self):
            self.str_length = 4
            self.num_templates = 1

    policy = UnknownPolicy()

    with pytest.raises(ValueError, match="Don't know how to initialize"):
        init_from_iupac_sequence(policy, "ACGT", epsilon=0.1)


def test_iupac_to_text_simple():
    """Test that to_text() recovers the input IUPAC sequence with epsilon=0."""
    test_sequence = "AACGTN"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    output = policy.to_text()

    # Should contain the input sequence
    assert test_sequence in output
    # Should have template info
    assert "Template 0" in output
    assert "weight:" in output


def test_iupac_to_text_degenerate():
    """Test to_text() with degenerate IUPAC codes."""
    test_sequence = "RWSMKYBDHVN"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    output = policy.to_text()

    # Should contain the input sequence with all degenerate codes
    assert test_sequence in output


def test_iupac_to_text_multiple_templates():
    """Test to_text() with multiple templates."""
    test_sequence = "ACGTN"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=3,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    output = policy.to_text()

    # Should have all three templates
    assert "Template 0" in output
    assert "Template 1" in output
    assert "Template 2" in output
    # All should be the same sequence
    assert output.count(test_sequence) == 3


def test_theoretical_diversity_single_codon():
    """Test theoretical diversity for a single specific codon (ATG = Met)."""
    test_sequence = "ATG"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    template_dna_probs = policy.get_template_dna_probs()

    from vis import (
        dna_to_amino_acid_probabilities,
        compute_theoretical_diversity_from_aa_probs,
    )

    aa_probs = dna_to_amino_acid_probabilities(template_dna_probs[0])
    diversity = compute_theoretical_diversity_from_aa_probs(aa_probs)

    assert (
        diversity == 1
    ), f"ATG should encode exactly 1 amino acid (Met), got {diversity}"


def test_theoretical_diversity_nnn():
    """Test theoretical diversity for NNN (any amino acid)."""
    test_sequence = "NNN"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    template_dna_probs = policy.get_template_dna_probs()

    from vis import (
        dna_to_amino_acid_probabilities,
        compute_theoretical_diversity_from_aa_probs,
    )

    aa_probs = dna_to_amino_acid_probabilities(template_dna_probs[0])
    diversity = compute_theoretical_diversity_from_aa_probs(aa_probs)

    assert (
        diversity == 21
    ), f"NNN should encode 21 possibilities (20 AAs + stop), got {diversity}"


def test_theoretical_diversity_multiple_codons():
    """Test theoretical diversity for multiple codons."""
    test_sequence = "ATGNNN"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    template_dna_probs = policy.get_template_dna_probs()

    from vis import (
        dna_to_amino_acid_probabilities,
        compute_theoretical_diversity_from_aa_probs,
    )

    aa_probs = dna_to_amino_acid_probabilities(template_dna_probs[0])
    diversity = compute_theoretical_diversity_from_aa_probs(aa_probs)

    assert (
        diversity == 21
    ), f"ATGNNN should encode 1 * 21 = 21 possibilities, got {diversity}"


def test_theoretical_diversity_fixed_iupac():
    """Test theoretical diversity using FixedIupacPolicy."""
    from policies import FixedIupacPolicy

    test_sequence = "GCNGCN"
    alphabet = ["A", "C", "G", "T"]

    policy = FixedIupacPolicy(test_sequence, alphabet)

    template_dna_probs = policy.get_template_dna_probs()

    from vis import (
        dna_to_amino_acid_probabilities,
        compute_theoretical_diversity_from_aa_probs,
    )

    aa_probs = dna_to_amino_acid_probabilities(template_dna_probs[0])
    diversity = compute_theoretical_diversity_from_aa_probs(aa_probs)

    # GCN codes for Alanine (4 codons: GCA, GCC, GCG, GCT all -> Ala)
    # So GCNGCN = Ala, Ala = 1 * 1 = 1
    assert (
        diversity == 1
    ), f"GCNGCN should encode exactly 1 dipeptide (Ala-Ala), got {diversity}"


def test_dna_diversity_specific_sequence():
    """Test DNA sequence diversity for a specific DNA sequence."""
    test_sequence = "ACGT"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    template_dna_probs = policy.get_template_dna_probs()

    threshold = 1e-9
    diversity = 1.0
    for pos in range(template_dna_probs.shape[1]):
        num_possible_bases = int(np.sum(template_dna_probs[0, pos] > threshold))
        diversity *= num_possible_bases

    assert diversity == 1, f"ACGT should encode exactly 1 DNA sequence, got {diversity}"


def test_dna_diversity_degenerate():
    """Test DNA sequence diversity for degenerate IUPAC codes."""
    test_sequence = "NNNN"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    template_dna_probs = policy.get_template_dna_probs()

    threshold = 1e-9
    diversity = 1.0
    for pos in range(template_dna_probs.shape[1]):
        num_possible_bases = int(np.sum(template_dna_probs[0, pos] > threshold))
        diversity *= num_possible_bases

    assert (
        diversity == 256
    ), f"NNNN should encode 4^4 = 256 DNA sequences, got {diversity}"


def test_dna_diversity_partial_degenerate():
    """Test DNA sequence diversity with partial degeneracy."""
    test_sequence = "ANNK"

    policy = IupacBaseOptimizer(
        str_length=len(test_sequence),
        alphabet_size=4,
        batch_size=10,
        num_templates=1,
        device="cpu",
    )

    init_from_iupac_sequence(policy, test_sequence, epsilon=0.0)

    template_dna_probs = policy.get_template_dna_probs()

    threshold = 1e-9
    diversity = 1.0
    for pos in range(template_dna_probs.shape[1]):
        num_possible_bases = int(np.sum(template_dna_probs[0, pos] > threshold))
        diversity *= num_possible_bases

    assert (
        diversity == 32
    ), f"ANNK should encode 1*4*4*2 = 32 DNA sequences, got {diversity}"


class TestAAToCodonTable:
    """Tests for the AA_TO_CODONS reverse codon table."""

    def test_all_amino_acids_have_codons(self):
        """Every amino acid (including stop) should have at least one codon."""
        for aa in AMINO_ACIDS:
            assert aa in AA_TO_CODONS, f"Missing AA: {aa}"
            assert len(AA_TO_CODONS[aa]) >= 1, f"No codons for {aa}"

    def test_codon_count(self):
        """Should have 64 total codons across all amino acids."""
        total_codons = sum(len(codons) for codons in AA_TO_CODONS.values())
        assert total_codons == 64, f"Expected 64 codons, got {total_codons}"

    def test_roundtrip(self):
        """AA_TO_CODONS should be consistent with CODON_TABLE."""
        for codon, aa in CODON_TABLE.items():
            assert codon in AA_TO_CODONS[aa], f"{codon} not in AA_TO_CODONS[{aa}]"


class TestAAToRandomDNA:
    """Tests for aa_to_dna_random function."""

    def test_produces_valid_codons(self):
        """Each AA should map to a valid codon."""
        rng = np.random.default_rng(42)
        for aa in AMINO_ACIDS:
            if aa == "*":
                continue  # skip stop codon for this test
            dna = aa_to_dna_random(aa, rng)
            assert len(dna) == 3, f"Expected 3 bases, got {len(dna)}"
            assert dna in AA_TO_CODONS[aa], f"{dna} not valid for {aa}"

    def test_randomness(self):
        """Different calls should produce different codons (for AAs with multiple)."""
        rng = np.random.default_rng(42)
        # Leucine has 6 codons, so should see variety
        codons = set()
        for _ in range(100):
            dna = aa_to_dna_random("L", rng)
            codons.add(dna)
        assert len(codons) > 1, "Expected multiple codons for Leucine"

    def test_invalid_aa_raises(self):
        """Unknown amino acid should raise error."""
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="Unknown amino acid"):
            aa_to_dna_random("X", rng)


class TestFastaInit:
    """Tests for init_from_fasta_sequences function."""

    @pytest.fixture
    def temp_fasta(self, tmp_path):
        """Create a temporary FASTA file with test sequences."""
        fasta_path = tmp_path / "test.fasta"
        # 4 AA sequences, each will become 12 DNA bases
        content = """>seq1
MAST
>seq2
KLPV
>seq3
WQRN
>seq4
EFGH
"""
        fasta_path.write_text(content)
        return str(fasta_path)

    def test_basic_fasta_init(self, temp_fasta):
        """Test basic FASTA initialization."""
        policy = IupacBaseOptimizer(
            str_length=12,  # 4 AAs * 3 bases
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
            device="cpu",
        )

        selected = init_from_fasta_sequences(policy, temp_fasta, epsilon=0.1, seed=42)

        assert len(selected) == 2
        assert all(len(s) == 4 for s in selected)  # 4 AAs each

    def test_seed_reproducibility(self, temp_fasta):
        """Same seed should produce same initialization."""
        policy1 = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=3,
            device="cpu",
        )
        policy2 = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=3,
            device="cpu",
        )

        selected1 = init_from_fasta_sequences(policy1, temp_fasta, epsilon=0.1, seed=42)
        selected2 = init_from_fasta_sequences(policy2, temp_fasta, epsilon=0.1, seed=42)

        assert selected1 == selected2
        assert torch.allclose(policy1.latent_phi, policy2.latent_phi)

    def test_different_seeds_produce_different_results(self, temp_fasta):
        """Different seeds should produce different initialization."""
        policy1 = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=3,
            device="cpu",
        )
        policy2 = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=3,
            device="cpu",
        )

        init_from_fasta_sequences(policy1, temp_fasta, epsilon=0.1, seed=42)
        init_from_fasta_sequences(policy2, temp_fasta, epsilon=0.1, seed=123)

        # At least one template should be different
        assert not torch.allclose(policy1.latent_phi, policy2.latent_phi)

    def test_template_diversity(self, temp_fasta):
        """With enough templates and sequences, templates should differ."""
        policy = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=4,
            device="cpu",
        )

        selected = init_from_fasta_sequences(policy, temp_fasta, epsilon=0.0, seed=42)

        # Check that not all templates have identical latent_phi
        # (since they get different sequences)
        unique_seqs = set(selected)
        if len(unique_seqs) > 1:
            # If sequences differ, templates should differ
            all_same = all(
                torch.allclose(policy.latent_phi[0], policy.latent_phi[i])
                for i in range(1, policy.num_templates)
            )
            assert not all_same, "Templates should differ when sequences differ"

    def test_wrong_length_raises(self, tmp_path):
        """Should raise error if sequences don't match expected length."""
        fasta_path = tmp_path / "wrong_len.fasta"
        fasta_path.write_text(">seq1\nMASTKL\n")  # 6 AAs = 18 DNA bases

        policy = IupacBaseOptimizer(
            str_length=12,  # expects 4 AAs
            alphabet_size=4,
            batch_size=10,
            num_templates=1,
            device="cpu",
        )

        with pytest.raises(ValueError, match="No sequences match expected AA length"):
            init_from_fasta_sequences(policy, str(fasta_path), epsilon=0.1)

    def test_empty_fasta_raises(self, tmp_path):
        """Should raise error for empty FASTA."""
        fasta_path = tmp_path / "empty.fasta"
        fasta_path.write_text("")

        policy = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=1,
            device="cpu",
        )

        with pytest.raises(ValueError, match="No sequences found"):
            init_from_fasta_sequences(policy, str(fasta_path), epsilon=0.1)

    def test_sampling_with_replacement(self, tmp_path):
        """Should work when num_templates > num_sequences (sampling with replacement)."""
        fasta_path = tmp_path / "small.fasta"
        fasta_path.write_text(">seq1\nMAST\n>seq2\nKLPV\n")  # Only 2 sequences

        policy = IupacBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=5,  # More templates than sequences
            device="cpu",
        )

        selected = init_from_fasta_sequences(
            policy, str(fasta_path), epsilon=0.1, seed=42
        )

        assert len(selected) == 5
        # All should be from the 2 available sequences
        assert all(s in ["MAST", "KLPV"] for s in selected)

    def test_arbitrary_base_optimizer(self, temp_fasta):
        """Should work with ArbitraryBaseOptimizer too."""
        policy = ArbitraryBaseOptimizer(
            str_length=12,
            alphabet_size=4,
            batch_size=10,
            num_templates=2,
            device="cpu",
        )

        selected = init_from_fasta_sequences(policy, temp_fasta, epsilon=0.1, seed=42)

        assert len(selected) == 2

        # Verify initialization changed latent_w
        probs, _ = policy.get_probs(tau=1.0)
        # Probs should not be uniform
        assert not torch.allclose(probs[0], torch.ones_like(probs[0]) / 4, atol=0.1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
