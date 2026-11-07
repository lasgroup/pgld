"""Tests for the library-entropy core estimator (compute_library_entropy)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

from compute_library_entropy import (
    condition_full_length,
    entropy_report,
    log_p_within_ed_ball,
    mc_entropy,
)


def _uniform_template(n_templates: int, lc: int, stop_prob: float = 0.0) -> np.ndarray:
    """(T, Lc, 21) with uniform mass over the 20 AAs and a fixed stop prob."""
    aa = np.zeros((n_templates, lc, 21))
    aa[:, :, :20] = (1.0 - stop_prob) / 20.0
    aa[:, :, 20] = stop_prob
    return aa


def test_condition_no_stop_is_identity():
    aa = _uniform_template(1, 2, stop_prob=0.0)
    log_ptilde, log_wtilde, p_no_stop = condition_full_length(aa, np.array([1.0]))
    assert p_no_stop == pytest.approx(1.0)
    assert log_wtilde[0] == pytest.approx(0.0)
    # each position renormalized to uniform over 20
    assert np.exp(log_ptilde).sum(axis=-1) == pytest.approx(np.ones((1, 2)))


def test_p_no_stop_product_over_positions():
    aa = _uniform_template(1, 3, stop_prob=0.5)
    _, _, p_no_stop = condition_full_length(aa, np.array([1.0]))
    assert p_no_stop == pytest.approx(0.5**3)


def test_template_reweighting_by_stop_free_prob():
    # Template 0: no stop; template 1: stop 0.5 at each of 2 positions -> surv 0.25.
    aa = np.concatenate(
        [_uniform_template(1, 2, 0.0), _uniform_template(1, 2, 0.5)], axis=0
    )
    _, log_wtilde, _ = condition_full_length(aa, np.array([0.5, 0.5]))
    wtilde = np.exp(log_wtilde)
    assert wtilde == pytest.approx([0.8, 0.2])


def test_single_uniform_template_entropy_is_exact():
    # A single uniform stop-free template puts equal mass on every full-length
    # peptide, so -log p is constant and MC recovers H = Lc*ln(20) exactly.
    lc = 4
    aa = _uniform_template(1, lc, stop_prob=0.0)
    rep = entropy_report(aa, np.array([1.0]), n_samples=10_000, batch_size=2048,
                         device="cpu", seed=0)
    assert rep["H_nats"] == pytest.approx(lc * np.log(20.0), abs=1e-4)
    assert rep["eff_seq_space"] == pytest.approx(20.0**lc, rel=1e-4)
    assert rep["mc_stderr_nats"] == pytest.approx(0.0, abs=1e-6)


def test_identical_templates_equal_single_template():
    # A mixture of identical templates has the same entropy as one template.
    lc = 3
    aa = _uniform_template(5, lc, stop_prob=0.0)
    w = np.full(5, 0.2)
    rep = entropy_report(aa, w, n_samples=10_000, batch_size=2048, device="cpu", seed=1)
    assert rep["H_nats"] == pytest.approx(lc * np.log(20.0), abs=1e-4)


def test_ed_ball_prob_matches_binomial():
    # Single uniform template: #mismatches ~ Binomial(Lc, 19/20); check P(<=D).
    lc, d = 4, 1
    aa = _uniform_template(1, lc, stop_prob=0.0)
    log_ptilde, log_wtilde, _ = condition_full_length(aa, np.array([1.0]))
    wt_idx = np.zeros(lc, dtype=int)  # WT = residue 0 at every position
    log_p = log_p_within_ed_ball(log_ptilde, log_wtilde, wt_idx, d)
    pm = 1.0 / 20.0
    expected = (pm**lc) + lc * (1 - pm) * (pm ** (lc - 1))  # P(0)+P(1) mismatches
    assert np.exp(log_p) == pytest.approx(expected)


def test_ed_ball_full_radius_equals_full_entropy():
    # With max_ed = Lc the ball is the whole space, so H_ball == H_full.
    lc = 4
    aa = _uniform_template(1, lc, stop_prob=0.0)
    wt_idx = np.zeros(lc, dtype=int)
    rep = entropy_report(aa, np.array([1.0]), n_samples=5_000, batch_size=2048,
                         device="cpu", seed=0, wt_idx=wt_idx, max_ed=lc)
    assert rep["p_within_ed"] == pytest.approx(1.0)
    assert rep["H_ball_nats"] == pytest.approx(lc * np.log(20.0), abs=1e-4)
    assert rep["eff_seq_space_ed"] == pytest.approx(20.0**lc, rel=1e-4)
