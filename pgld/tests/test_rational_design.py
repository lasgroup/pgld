"""Tests for rational design baseline functions."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rational_design import (
    AA_LIST,
    NUCLEOTIDE_LIST,
    IUPAC_NUCLEOTIDE_CODES,
    compute_aa_frequencies_from_hits,
    compute_nucleotide_frequencies_from_hits,
    design_degenerate_codons,
    design_degenerate_codons_simple,
    design_iupac_codes_simple,
    _expand_degenerate_codon,
    FREQUENCIES_AAS_DEGENERATE_CODONS,
)


class TestComputeAAFrequencies:
    """Tests for compute_aa_frequencies_from_hits function."""

    def test_uniform_single_aa(self):
        """All positions have the same AA - should get frequency 1.0."""
        hit_proteins = ["AAAA", "AAAA", "AAAA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)

        assert fraq_df.shape == (4, len(AA_LIST))
        assert list(fraq_df.index) == [1, 2, 3, 4]

        for pos in range(1, 5):
            assert fraq_df.loc[pos, "A"] == 1.0
            assert fraq_df.loc[pos, "C"] == 0.0

    def test_mixed_frequencies(self):
        """Test with mixed amino acids at each position."""
        hit_proteins = [
            "FLWG",
            "FLWG",
            "LLWG",
            "MLWA",
        ]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)

        # Position 1: F=2, L=1, M=1 -> F=0.5, L=0.25, M=0.25
        assert fraq_df.loc[1, "F"] == 0.5
        assert fraq_df.loc[1, "L"] == 0.25
        assert fraq_df.loc[1, "M"] == 0.25

        # Position 2: L=4 -> L=1.0
        assert fraq_df.loc[2, "L"] == 1.0

        # Position 3: W=4 -> W=1.0
        assert fraq_df.loc[3, "W"] == 1.0

        # Position 4: G=3, A=1 -> G=0.75, A=0.25
        assert fraq_df.loc[4, "G"] == 0.75
        assert fraq_df.loc[4, "A"] == 0.25

    def test_rows_sum_to_one(self):
        """Each row should sum to 1.0."""
        hit_proteins = ["ACDE", "FGHI", "KLMN", "PQRS"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)

        for pos in fraq_df.index:
            row_sum = fraq_df.loc[pos].sum()
            assert np.isclose(row_sum, 1.0), f"Row {pos} sums to {row_sum}"

    def test_empty_input_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="No hit proteins provided"):
            compute_aa_frequencies_from_hits([])

    def test_single_protein(self):
        """Single protein should give 1.0 at each position."""
        hit_proteins = ["MWFC"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)

        assert fraq_df.loc[1, "M"] == 1.0
        assert fraq_df.loc[2, "W"] == 1.0
        assert fraq_df.loc[3, "F"] == 1.0
        assert fraq_df.loc[4, "C"] == 1.0


class TestExpandDegenerateCodon:
    """Tests for _expand_degenerate_codon helper."""

    def test_concrete_codon(self):
        """Concrete codon should expand to itself."""
        result = _expand_degenerate_codon("ATG")
        assert result == ["ATG"]

    def test_nnk_codon(self):
        """NNK codon should expand to 32 codons."""
        result = _expand_degenerate_codon("NNK")
        assert len(result) == 32  # 4 * 4 * 2

    def test_nnn_codon(self):
        """NNN codon should expand to 64 codons."""
        result = _expand_degenerate_codon("NNN")
        assert len(result) == 64  # 4 * 4 * 4


class TestDesignDegenrateCodons:
    """Tests for design_degenerate_codons function."""

    def test_without_wild_type(self):
        """Function should work without wild type."""
        hit_proteins = ["FLWG", "FLWG", "LLWG", "MLWA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        peptide_length = 4
        shannon_entropies = {pos: 0.0 for pos in range(1, peptide_length + 1)}

        result = design_degenerate_codons(
            fraq_df=fraq_df,
            shannon_entropies_norm_dict=shannon_entropies,
            wt=None,
            entropy_threshold_scale=0.0,
        )

        assert "candidates_dict" in result
        assert len(result["candidates_dict"]) == peptide_length

        # Each position should have a codon selected
        for pos in range(1, peptide_length + 1):
            assert pos in result["candidates_dict"]
            assert "codon" in result["candidates_dict"][pos]

    def test_with_wild_type(self):
        """Function should work with wild type."""
        hit_proteins = ["FLWG", "FLWG", "LLWG", "MLWA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        peptide_length = 4
        shannon_entropies = {pos: 0.0 for pos in range(1, peptide_length + 1)}

        result = design_degenerate_codons(
            fraq_df=fraq_df,
            shannon_entropies_norm_dict=shannon_entropies,
            wt="FLWG",
            entropy_threshold_scale=0.0,
        )

        assert "candidates_dict" in result
        assert len(result["candidates_dict"]) == peptide_length

    def test_selected_codons_are_valid(self):
        """Selected codons should be valid IUPAC codons."""
        hit_proteins = ["AAAA", "AAAA", "AAAA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        peptide_length = 4
        shannon_entropies = {pos: 0.0 for pos in range(1, peptide_length + 1)}

        result = design_degenerate_codons(
            fraq_df=fraq_df,
            shannon_entropies_norm_dict=shannon_entropies,
            wt=None,
            entropy_threshold_scale=0.0,
        )

        for pos, candidate in result["candidates_dict"].items():
            codon = candidate["codon"]
            # Codon should be in our precomputed frequencies
            assert codon in FREQUENCIES_AAS_DEGENERATE_CODONS, f"Invalid codon: {codon}"

    def test_conserved_position_single_aa(self):
        """Position with single AA should get a specific codon."""
        # All positions have only Alanine
        hit_proteins = ["AAAA"] * 10
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        peptide_length = 4
        shannon_entropies = {pos: 0.0 for pos in range(1, peptide_length + 1)}

        result = design_degenerate_codons(
            fraq_df=fraq_df,
            shannon_entropies_norm_dict=shannon_entropies,
            wt=None,
            entropy_threshold_scale=0.0,
        )

        for pos, candidate in result["candidates_dict"].items():
            codon = candidate["codon"]
            aa_dist = FREQUENCIES_AAS_DEGENERATE_CODONS[codon]
            # Should produce Alanine
            assert "A" in aa_dist, f"Codon {codon} doesn't produce Alanine"


class TestDesignDegenerateCodonsSimple:
    """Tests for design_degenerate_codons_simple function."""

    def test_uniform_single_aa(self):
        """When all hits have same AA, should select a codon producing that AA."""
        hit_proteins = ["AAAA"] * 10
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        selected = design_degenerate_codons_simple(fraq_df)

        for pos, codon in selected.items():
            aa_dist = FREQUENCIES_AAS_DEGENERATE_CODONS[codon]
            assert "A" in aa_dist and aa_dist.get("A", 0) > 0.5

    def test_returns_dict_of_codons(self):
        """Should return dict mapping positions to codons."""
        hit_proteins = ["FLWG", "LLWG", "MLWA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        selected = design_degenerate_codons_simple(fraq_df)

        assert len(selected) == 4
        for pos, codon in selected.items():
            assert codon in FREQUENCIES_AAS_DEGENERATE_CODONS

    def test_template_generation(self):
        """Test building IUPAC template from selected codons."""
        hit_proteins = ["FLWG", "LLWG", "MLWA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        selected = design_degenerate_codons_simple(fraq_df)
        template = "".join(selected[pos] for pos in fraq_df.index)

        assert len(template) == 4 * 3  # 4 positions * 3 nucleotides per codon

    def test_perfect_mse_zero_match(self):
        """When target frequencies match a codon exactly, MSE should be zero.

        M (Methionine) is encoded only by ATG.
        W (Tryptophan) is encoded only by TGG.
        So 100% M or W at a position should select ATG or TGG respectively.
        """
        hit_proteins = ["MWMW"] * 10
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        selected = design_degenerate_codons_simple(fraq_df)

        # Position 1 and 3: 100% M -> ATG
        assert selected[1] == "ATG"
        assert selected[3] == "ATG"

        # Position 2 and 4: 100% W -> TGG
        assert selected[2] == "TGG"
        assert selected[4] == "TGG"


class TestIUPACTemplateGeneration:
    """Test that we can generate a valid IUPAC template from designed codons."""

    def test_template_length(self):
        """Template should be 3x peptide length."""
        hit_proteins = ["FLWG", "LLWG", "MLWA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        peptide_length = 4
        shannon_entropies = {pos: 0.0 for pos in range(1, peptide_length + 1)}

        result = design_degenerate_codons(
            fraq_df=fraq_df,
            shannon_entropies_norm_dict=shannon_entropies,
            wt=None,
            entropy_threshold_scale=0.0,
        )

        # Build template like in run_rational_design_baseline
        template = ""
        for pos in range(1, peptide_length + 1):
            codon = result["candidates_dict"][pos]["codon"]
            template += codon

        assert len(template) == peptide_length * 3

    def test_template_has_valid_iupac_chars(self):
        """Template should only contain valid IUPAC characters."""
        valid_iupac = set("ACGTRYSWKMBDHVN")

        hit_proteins = ["FLWG", "LLWG", "MLWA"]
        fraq_df = compute_aa_frequencies_from_hits(hit_proteins)
        peptide_length = 4
        shannon_entropies = {pos: 0.0 for pos in range(1, peptide_length + 1)}

        result = design_degenerate_codons(
            fraq_df=fraq_df,
            shannon_entropies_norm_dict=shannon_entropies,
            wt=None,
            entropy_threshold_scale=0.0,
        )

        template = ""
        for pos in range(1, peptide_length + 1):
            codon = result["candidates_dict"][pos]["codon"]
            template += codon

        for char in template:
            assert char in valid_iupac, f"Invalid IUPAC character: {char}"


class TestComputeNucleotideFrequencies:
    """Tests for compute_nucleotide_frequencies_from_hits function."""

    def test_uniform_single_nucleotide(self):
        """All positions have the same nucleotide - should get frequency 1.0."""
        hit_seqs = ["AAAA", "AAAA", "AAAA"]
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)

        assert fraq_df.shape == (4, len(NUCLEOTIDE_LIST))
        assert list(fraq_df.index) == [1, 2, 3, 4]

        for pos in range(1, 5):
            assert fraq_df.loc[pos, "A"] == 1.0
            assert fraq_df.loc[pos, "C"] == 0.0

    def test_mixed_frequencies(self):
        """Test with mixed nucleotides at each position."""
        hit_seqs = ["ACGT", "ACGT", "TCGT", "GCGA"]
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)

        # Position 1: A=2, T=1, G=1 -> A=0.5, T=0.25, G=0.25
        assert fraq_df.loc[1, "A"] == 0.5
        assert fraq_df.loc[1, "T"] == 0.25
        assert fraq_df.loc[1, "G"] == 0.25

        # Position 2: C=4 -> C=1.0
        assert fraq_df.loc[2, "C"] == 1.0

    def test_rows_sum_to_one(self):
        """Each row should sum to 1.0."""
        hit_seqs = ["ACGT", "TGCA", "GATC", "CTAG"]
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)

        for pos in fraq_df.index:
            row_sum = fraq_df.loc[pos].sum()
            assert np.isclose(row_sum, 1.0), f"Row {pos} sums to {row_sum}"

    def test_empty_input_raises(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError, match="No hit sequences provided"):
            compute_nucleotide_frequencies_from_hits([])


class TestDesignIUPACCodesSimple:
    """Tests for design_iupac_codes_simple function."""

    def test_uniform_frequencies_select_single_base(self):
        """When all hits have same base at position, should select that base."""
        hit_seqs = ["AAAA"] * 10
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)
        selected = design_iupac_codes_simple(fraq_df)

        for pos in range(1, 5):
            assert selected[pos] == "A"

    def test_equal_two_bases_select_degenerate(self):
        """When two bases are equally frequent, should select appropriate IUPAC code."""
        hit_seqs = ["ACGT", "TCGT"]  # Position 1: A=0.5, T=0.5
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)
        selected = design_iupac_codes_simple(fraq_df)

        # Position 1 should be W (A or T)
        assert selected[1] == "W"

    def test_selected_codes_are_valid_iupac(self):
        """Selected codes should be valid IUPAC codes."""
        hit_seqs = ["ACGT", "TGCA", "GATC", "CTAG"]
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)
        selected = design_iupac_codes_simple(fraq_df)

        for pos, code in selected.items():
            assert code in IUPAC_NUCLEOTIDE_CODES, f"Invalid IUPAC code: {code}"

    def test_template_length_matches_input(self):
        """Template built from selected codes should match input length."""
        hit_seqs = ["ACGTACGT"] * 5
        fraq_df = compute_nucleotide_frequencies_from_hits(hit_seqs)
        selected = design_iupac_codes_simple(fraq_df)
        template = "".join(selected[pos] for pos in fraq_df.index)

        assert len(template) == 8


class TestIUPACSequenceGeneration:
    """Tests for generate_iupac_sequences and related functions."""

    def test_iupac_sampling_is_deterministic_with_seed(self):
        """Sampling should be deterministic when numpy seed is set."""
        from landscapes import generate_iupac_sequences

        template = "NNKNNKNNK"
        n = 100

        np.random.seed(42)
        seqs1 = generate_iupac_sequences(template, n)

        np.random.seed(42)
        seqs2 = generate_iupac_sequences(template, n)

        assert seqs1 == seqs2, "Sequences should be identical with same seed"

    def test_iupac_sampling_differs_with_different_seed(self):
        """Sampling should differ with different seeds."""
        from landscapes import generate_iupac_sequences

        template = "NNKNNKNNK"
        n = 100

        np.random.seed(42)
        seqs1 = generate_iupac_sequences(template, n)

        np.random.seed(123)
        seqs2 = generate_iupac_sequences(template, n)

        assert seqs1 != seqs2, "Sequences should differ with different seeds"

    def test_nnk_sampling_is_deterministic_with_seed(self):
        """NNK sampling should be deterministic when numpy seed is set."""
        from landscapes import generate_nnk_sequences

        length = 12  # 4 codons
        n = 100

        np.random.seed(42)
        seqs1 = generate_nnk_sequences(length, n)

        np.random.seed(42)
        seqs2 = generate_nnk_sequences(length, n)

        assert seqs1 == seqs2, "Sequences should be identical with same seed"

    def test_uniform_random_sampling_is_deterministic_with_seed(self):
        """Uniform random sampling should be deterministic when numpy seed is set."""
        from landscapes import generate_uniform_random_sequences

        alphabet = ["A", "C", "G", "T"]
        length = 12
        n = 100

        np.random.seed(42)
        seqs1 = generate_uniform_random_sequences(alphabet, length, n)

        np.random.seed(42)
        seqs2 = generate_uniform_random_sequences(alphabet, length, n)

        assert seqs1 == seqs2, "Sequences should be identical with same seed"
