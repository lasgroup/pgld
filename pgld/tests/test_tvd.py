"""Test TVD (Total Variation Distance) computation."""

import os
import sys
import tempfile

sys.path.insert(
    0, os.environ.get("PGLD_PKG_ROOT", os.path.dirname(os.path.dirname(__file__)))
)

import numpy as np
import pytest
from vis import compute_tvd_per_position, plot_tvd_per_position, _filter_seqs_for_tvd


def test_identical_distributions_zero_tvd():
    """TVD of identical distributions should be 0."""
    freqs = np.array(
        [
            [0.5, 0.3, 0.2, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [1.0, 0.0, 0.0, 0.0],
        ]
    )
    tvd = compute_tvd_per_position(freqs, freqs)
    np.testing.assert_array_almost_equal(tvd, np.zeros(3))


def test_completely_different_distributions_max_tvd():
    """TVD of completely disjoint distributions should be 1."""
    # Position 0: all mass on AA 0 vs all mass on AA 1
    sampled = np.array([[1.0, 0.0, 0.0, 0.0]])
    reference = np.array([[0.0, 1.0, 0.0, 0.0]])
    tvd = compute_tvd_per_position(sampled, reference)
    np.testing.assert_array_almost_equal(tvd, np.array([1.0]))


def test_half_overlap_distributions():
    """TVD of distributions with half overlap should be 0.5."""
    # p = [0.5, 0.5, 0, 0], q = [0, 0.5, 0.5, 0]
    # |p - q| = [0.5, 0, 0.5, 0], sum = 1.0, TVD = 0.5
    sampled = np.array([[0.5, 0.5, 0.0, 0.0]])
    reference = np.array([[0.0, 0.5, 0.5, 0.0]])
    tvd = compute_tvd_per_position(sampled, reference)
    np.testing.assert_array_almost_equal(tvd, np.array([0.5]))


def test_multiple_positions():
    """TVD computed independently at each position."""
    sampled = np.array(
        [
            [1.0, 0.0],  # Position 0: disjoint from ref
            [0.5, 0.5],  # Position 1: identical to ref
            [0.7, 0.3],  # Position 2: partial overlap
        ]
    )
    reference = np.array(
        [
            [0.0, 1.0],  # Position 0: disjoint
            [0.5, 0.5],  # Position 1: identical
            [0.3, 0.7],  # Position 2: |0.4| + |0.4| = 0.8, TVD = 0.4
        ]
    )
    tvd = compute_tvd_per_position(sampled, reference)
    expected = np.array([1.0, 0.0, 0.4])
    np.testing.assert_array_almost_equal(tvd, expected)


def test_tvd_symmetry():
    """TVD(P, Q) == TVD(Q, P)."""
    np.random.seed(42)
    sampled = np.random.dirichlet([1, 1, 1, 1], size=5)
    reference = np.random.dirichlet([1, 1, 1, 1], size=5)

    tvd_forward = compute_tvd_per_position(sampled, reference)
    tvd_backward = compute_tvd_per_position(reference, sampled)
    np.testing.assert_array_almost_equal(tvd_forward, tvd_backward)


def test_tvd_bounds():
    """TVD should always be in [0, 1] for probability distributions."""
    np.random.seed(123)
    for _ in range(10):
        sampled = np.random.dirichlet([1] * 20, size=11)  # (11, 20)
        reference = np.random.dirichlet([1] * 20, size=11)  # (11, 20)
        tvd = compute_tvd_per_position(sampled, reference)
        assert np.all(tvd >= 0), "TVD should be non-negative"
        assert np.all(tvd <= 1), "TVD should be at most 1"


def test_plot_tvd_generates_both_files():
    """plot_tvd_per_position should generate both with and without stop codon plots."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a temporary reference FASTA
        fasta_path = os.path.join(tmpdir, "reference.fasta")
        with open(fasta_path, "w") as f:
            for i in range(100):
                f.write(f">seq{i}\n")
                f.write("ACDEFGHIKL\n")  # 10 AA sequence

        sampled_seqs = ["ACDEFGHIKL"] * 50 + ["ACDEFGHIKM"] * 50
        save_path = os.path.join(tmpdir, "tvd_per_position.png")

        plot_tvd_per_position(
            sampled_seqs=sampled_seqs,
            reference_fasta_path=fasta_path,
            save_path=save_path,
            expected_length=10,
        )

        assert os.path.exists(save_path), "Primary plot (no stops) not created"
        with_stops_path = os.path.join(tmpdir, "tvd_per_position_with_stops.png")
        assert os.path.exists(
            with_stops_path
        ), "Secondary plot (with stops) not created"


def test_filter_seqs_for_tvd():
    """Test the sequence filtering helper function."""
    seqs = ["ACDEF", "ACDE*", "ACDEFG", "AC*EF"]

    # Without stops: only full-length seqs without *
    filtered = _filter_seqs_for_tvd(seqs, expected_length=5, include_stops=False)
    assert filtered == ["ACDEF"]

    # With stops: full-length seqs including those with *
    filtered = _filter_seqs_for_tvd(seqs, expected_length=5, include_stops=True)
    assert filtered == ["ACDEF", "ACDE*", "AC*EF"]


def test_filter_seqs_for_tvd_padding():
    """Test that truncated sequences are padded with stops."""
    seqs = ["ACDEFGHIJK", "ACD", "ACDEF*HIJK", "AC"]  # expected_length=10

    # Without stops: only full-length without *
    filtered = _filter_seqs_for_tvd(seqs, expected_length=10, include_stops=False)
    assert filtered == ["ACDEFGHIJK"]

    # With stops: pad short sequences with *
    filtered = _filter_seqs_for_tvd(seqs, expected_length=10, include_stops=True)
    assert filtered == [
        "ACDEFGHIJK",  # full length, kept
        "ACD*******",  # padded from 3 to 10
        "ACDEF*HIJK",  # full length with *, kept
        "AC********",  # padded from 2 to 10
    ]
