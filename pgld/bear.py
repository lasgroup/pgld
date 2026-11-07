"""BEAR nonparametric two-sample Bayes factor (Amin et al. 2021; Weinstein 2024, C.3).

Model (Amin 2021, Eqn 2): L ~ pi(l); v_k ~ Dirichlet((1/h) f_k(theta));
X_i | X_{i-L:i-1} ~ Categorical(v_{X_{i-L:i-1}}). Uses the vanilla Jeffreys prior
(f(theta)=1, h=2 => Dirichlet(1/2,...,1/2), i.e. alpha = (1/h) f = 0.5 per symbol) and
marginalises over the lag L with a prior pi(L) truncated at max_lag.

Per-lag log evidence (Amin 2021, Eqn 3) over length-L k-mer contexts k with next-symbol
counts n_k (sum N_k), start-padded so every position has a length-L context:
    logML_L = sum_k [ gammaln(A a) - A gammaln(a)
                      + sum_b gammaln(a + n_kb) - gammaln(A a + N_k) ]
Total evidence marginalises over L: logE = logsumexp_L [ log pi(L) + logML_L ]. The
two-sample Bayes factor (Weinstein 2024, C.3) is BF = E(X) E(Y) / E(X u Y); the reported
quantity is the per-sequence geometric mean BF^(1/(n+m)) (n=m): ~1 indistinguishable, >1 differ.

pi(L) is uniform over positive lags truncated at max_lag.
Cost is O(N * max_lag * len) with one dict entry per observed k-mer, so a large max_lag
on long sequences is expensive.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp

AMINO_ACIDS = "ARNDCEQGHILKMFPSTWYV"
STOP = "$"
START = "^"
JEFFREYS_ALPHA = 0.5

_NEXT_SYMBOLS = AMINO_ACIDS + STOP  # predicted symbols (start is context-only)
_NEXT_INDEX = {c: i for i, c in enumerate(_NEXT_SYMBOLS)}


def _all_lag_counts(sequences: list[str], max_lag: int) -> list[dict[str, np.ndarray]]:
    """Per-lag {length-L context -> next-symbol counts} for L in 1..max_lag.

    Sequences are truncated at their first stop ('*'), start-padded, and stop-terminated
    so every prediction has a length-L context. Sequences with non-amino-acid symbols
    are skipped. Returns a list indexed by L (index 0 unused).
    """
    A = len(_NEXT_SYMBOLS)
    per_lag: list[dict[str, np.ndarray]] = [dict() for _ in range(max_lag + 1)]
    for raw in sequences:
        core = raw.split("*")[0]
        if any(c not in _NEXT_INDEX or c == STOP for c in core):
            continue
        padded = START * max_lag + core + STOP
        for i in range(max_lag, len(padded)):  # predict each real symbol, then stop
            nxt = _NEXT_INDEX[padded[i]]
            for L in range(1, max_lag + 1):
                ctx = padded[i - L : i]
                counts = per_lag[L]
                arr = counts.get(ctx)
                if arr is None:
                    arr = np.zeros(A, dtype=np.int64)
                    counts[ctx] = arr
                arr[nxt] += 1
    return per_lag


def _log_evidence_fixed_lag(counts: dict[str, np.ndarray], alpha: float) -> float:
    """BEAR log marginal evidence at a fixed lag (Amin 2021, Eqn 3)."""
    A = len(_NEXT_SYMBOLS)
    a0 = A * alpha
    const = gammaln(a0) - A * gammaln(alpha)
    total = 0.0
    for n in counts.values():
        total += const - gammaln(n.sum() + a0) + gammaln(n + alpha).sum()
    return total


def _marginal_log_evidence(
    per_lag: list[dict[str, np.ndarray]], max_lag: int, alpha: float
) -> float:
    """logsumexp_L [ log pi(L) + logML_L ] with uniform pi(L) over 1..max_lag."""
    per_lag_logml = np.array(
        [_log_evidence_fixed_lag(per_lag[L], alpha) for L in range(1, max_lag + 1)]
    )
    return float(logsumexp(per_lag_logml - np.log(max_lag)))


def _merge_lag_counts(
    cx: list[dict[str, np.ndarray]], cy: list[dict[str, np.ndarray]]
) -> list[dict[str, np.ndarray]]:
    """Pooled per-lag counts for the 'same distribution' hypothesis."""
    merged = [dict() for _ in cx]
    for L in range(len(cx)):
        pooled = {k: v.copy() for k, v in cx[L].items()}
        for k, v in cy[L].items():
            pooled[k] = pooled[k] + v if k in pooled else v.copy()
        merged[L] = pooled
    return merged


def bear_two_sample_bayes_factor(
    model_sequences: list[str],
    data_sequences: list[str],
    max_lag: int | None = None,
    alpha: float = JEFFREYS_ALPHA,
) -> dict:
    """BEAR two-sample Bayes factor between model samples and held-out data.

    Args:
        model_sequences: samples from the generative model (X ~ p_theta).
        data_sequences: held-out reference sequences (Y ~ p0).
        max_lag: lag prior truncation; defaults to the longest sequence.
        alpha: Dirichlet concentration per symbol; 0.5 = Jeffreys.

    Returns the log Bayes factor and the per-sequence geometric-mean Bayes factor
    (~1 => indistinguishable, >1 => evidence the distributions differ).
    """
    n, m = len(model_sequences), len(data_sequences)
    if n == 0 or m == 0:
        raise ValueError("Both sequence sets must be non-empty.")
    if max_lag is None:
        max_lag = max(len(s.split("*")[0]) for s in model_sequences + data_sequences)

    cx = _all_lag_counts(model_sequences, max_lag)
    cy = _all_lag_counts(data_sequences, max_lag)
    cpool = _merge_lag_counts(cx, cy)

    log_bf = (
        _marginal_log_evidence(cx, max_lag, alpha)
        + _marginal_log_evidence(cy, max_lag, alpha)
        - _marginal_log_evidence(cpool, max_lag, alpha)
    )
    return {
        "log_bayes_factor": log_bf,
        "bayes_factor_per_sequence": float(np.exp(log_bf / (n + m))),
        "n_model": n,
        "n_data": m,
        "max_lag": max_lag,
    }
