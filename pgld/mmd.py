"""MMD and witness-function computation with a heavy-tailed Hamming kernel.

Follows Weinstein et al. (2024), Manufacturing-Aware Generative Model
Architectures (bioRxiv 2024.09.13.612900), Appendix C.3-C.4:
  - kernel  k(x, y) = (C + d_H(x, y))^(-beta),  C = 1, beta = 0.5
  - MMD     reported as the V-statistic  E[k(X,X')] - 2 E[k(X,Y)] + E[k(Y,Y')]
  - witness f(x) = E_{Y~p0}[k(x, Y)] - E_{X~p_theta}[k(x, X)]   (data p0, model p_theta)
"""

from __future__ import annotations

import numpy as np

MMD_BLOCK_SIZE = 5000  # processes ALL pairs, just in memory-efficient chunks
KERNEL_C = 1.0
KERNEL_BETA = 0.5
WITNESS_ALPHA = 0.33  # quantile threshold; "clearly realistic" = top (1 - alpha)


def encode_sequences(sequences: list[str], max_len: int | None = None) -> np.ndarray:
    """Encode sequences as int8 arrays, right-padded to a common length.

    Padded with the stop symbol '*' so length differences add to the Hamming distance
    (Weinstein 2024, C.3). Pass max_len to pad two sets to the SAME width before
    comparing them; otherwise the per-call max length is used.
    """
    if not sequences:
        return np.array([], dtype=np.int8)
    if max_len is None:
        max_len = max(len(s) for s in sequences)
    aa_to_int = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY*-")}
    return np.array(
        [[aa_to_int.get(c, 21) for c in s.ljust(max_len, "*")[:max_len]] for s in sequences],
        dtype=np.int8,
    )  # (N, max_len)


def _hamming_kernel_block(seqs_a: np.ndarray, seqs_b: np.ndarray) -> np.ndarray:
    """Heavy-tailed Hamming kernel matrix for two blocks: (C + d_H)^(-beta)."""
    hamming = (seqs_a[:, None, :] != seqs_b[None, :, :]).sum(axis=2)  # (n, m)
    return (KERNEL_C + hamming.astype(np.float32)) ** (-KERNEL_BETA)  # (n, m)


def _kernel_row_means(
    queries: np.ndarray, targets: np.ndarray, block_size: int = MMD_BLOCK_SIZE
) -> np.ndarray:
    """Mean kernel from each query to the target set: (1/m) sum_j k(q_i, t_j)."""
    n, m = len(queries), len(targets)
    row_sums = np.zeros(n, dtype=np.float64)  # (n,)
    for i in range(0, n, block_size):
        qb = queries[i : i + block_size]
        acc = np.zeros(len(qb), dtype=np.float64)
        for j in range(0, m, block_size):
            acc += _hamming_kernel_block(qb, targets[j : j + block_size]).sum(axis=1)
        row_sums[i : i + len(qb)] = acc
    return row_sums / m


def compute_mmd_v_statistic(
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    block_size: int = MMD_BLOCK_SIZE,
    verbose: bool = True,
) -> float:
    """Compute exact MMD^2 (V-statistic) over ALL n x m pairs, processed in blocks.

    MMD^2(P, Q) = E[k(X,X')] - 2*E[k(X,Y)] + E[k(Y,Y')]
    """

    def _block_mean(seqs_a: np.ndarray, seqs_b: np.ndarray, label: str) -> float:
        total_sum, total_count = 0.0, 0
        n_blocks = (
            int(np.ceil(len(seqs_a) / block_size))
            * int(np.ceil(len(seqs_b) / block_size))
        )
        block_idx = 0
        for i in range(0, len(seqs_a), block_size):
            for j in range(0, len(seqs_b), block_size):
                k = _hamming_kernel_block(
                    seqs_a[i : i + block_size], seqs_b[j : j + block_size]
                )
                total_sum += float(k.sum())
                total_count += k.size
                block_idx += 1
                if verbose and block_idx % 50 == 0:
                    print(
                        f"      {label}: {block_idx}/{n_blocks} blocks "
                        f"({100*block_idx/n_blocks:.0f}%)",
                        end="\r",
                    )
        if verbose:
            print(f"      {label}: {n_blocks}/{n_blocks} blocks (100%)    ")
        return total_sum / total_count

    kxx = _block_mean(samples_x, samples_x, "K(X,X)")
    kxy = _block_mean(samples_x, samples_y, "K(X,Y)")
    kyy = _block_mean(samples_y, samples_y, "K(Y,Y)")

    return kxx - 2 * kxy + kyy


def witness_scores(
    queries: np.ndarray,
    data: np.ndarray,
    model: np.ndarray,
    block_size: int = MMD_BLOCK_SIZE,
) -> np.ndarray:
    """Witness f(x) = mean_i k(x, data_i) - mean_j k(x, model_j) at each query point.

    Low where x is a data outlier or a region the model over-represents, so it also
    penalises concentration.
    """
    return _kernel_row_means(queries, data, block_size) - _kernel_row_means(
        queries, model, block_size
    )


def witness_inlier_fraction(
    model_samples: np.ndarray,
    data_samples: np.ndarray,
    alpha: float = WITNESS_ALPHA,
    block_size: int = MMD_BLOCK_SIZE,
) -> tuple[float, float]:
    """Fraction of model samples that are "clearly realistic" (Weinstein 2024, C.4).

    Threshold tau = alpha-quantile of the witness scores of the data points
    themselves; the reported fraction is the share of model samples with witness
    score >= tau. A model matching the data yields ~ (1 - alpha) on expectation.

    Returns (fraction, tau).
    """
    # Self-pairing (k(x,x)=1) is included in both terms.
    data_witness = witness_scores(data_samples, data_samples, model_samples, block_size)
    tau = float(np.quantile(data_witness, alpha))
    model_witness = witness_scores(
        model_samples, data_samples, model_samples, block_size
    )
    fraction = float(np.mean(model_witness >= tau))
    return fraction, tau
