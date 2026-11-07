"""
approx_union_diversity.py

Approximate union diversity estimation using the Karp-Luby FPRAS algorithm.

This provides polynomial-time (1±ε) approximation for computing union diversity
when exact inclusion-exclusion is infeasible (>20 templates).
"""

from typing import List, Optional, Tuple

import numpy as np

from diversity_utils import compute_mask_diversity


def sample_from_mask(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Samples one sequence uniformly from the set defined by a boolean mask.

    Args:
        mask: Boolean array of shape (length, alphabet_size) where values > 0
            indicate allowed characters
        rng: NumPy random generator

    Returns:
        Array of shape (length,) with character indices for each position
    """
    length = mask.shape[0]
    sequence = np.empty(length, dtype=np.int32)
    for pos in range(length):
        allowed = np.where(mask[pos] > 0)[0]
        sequence[pos] = rng.choice(allowed)
    return sequence


def sequence_in_mask(sequence: np.ndarray, mask: np.ndarray) -> bool:
    """
    Checks if a sequence is allowed by a mask.

    Args:
        sequence: Array of shape (length,) with character indices
        mask: Boolean array of shape (length, alphabet_size)

    Returns:
        True if the sequence is valid under the mask
    """
    for pos, char_idx in enumerate(sequence):
        if mask[pos, char_idx] == 0:
            return False
    return True


def estimate_union_diversity_karp_luby(
    masks: List[np.ndarray],
    target_relative_error: float = 0.5,
    confidence: float = 0.95,
    min_samples: int = 10000,
    max_samples: int = 1000000,
    seed: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Estimates union diversity using the Karp-Luby FPRAS algorithm.

    The algorithm samples from the weighted union of templates and corrects
    for overcounting by only accepting samples that belong to their "first"
    (lowest-indexed) template.

    Args:
        masks: List of boolean masks, each shape (length, alphabet_size)
        target_relative_error: Target ε for (1±ε) approximation (default 0.5 = 2x)
        confidence: Confidence level (default 0.95)
        min_samples: Minimum number of samples
        max_samples: Maximum number of samples
        seed: Random seed for reproducibility

    Returns:
        Tuple of (estimated_diversity, standard_error)
    """
    if len(masks) == 0:
        return 0.0, 0.0

    rng = np.random.default_rng(seed)
    num_templates = len(masks)

    individual_divs = np.array([compute_mask_diversity(m) for m in masks])
    sum_diversity = individual_divs.sum()

    if sum_diversity == 0:
        return 0.0, 0.0

    # Compute number of samples: N ≈ M × log(2/δ) / ε²
    delta = 1.0 - confidence
    theoretical_samples = int(
        np.ceil(num_templates * np.log(2.0 / delta) / (target_relative_error**2))
    )
    num_samples = max(min_samples, min(max_samples, theoretical_samples))

    # Sampling probabilities proportional to individual diversities
    probs = individual_divs / sum_diversity

    accepts = 0
    for _ in range(num_samples):
        # Pick template proportional to its diversity
        template_idx = rng.choice(num_templates, p=probs)

        # Sample a sequence from this template
        sequence = sample_from_mask(masks[template_idx], rng)

        # Check if this sequence belongs to any earlier template
        in_earlier = False
        for j in range(template_idx):
            if sequence_in_mask(sequence, masks[j]):
                in_earlier = True
                break

        if not in_earlier:
            accepts += 1

    # Estimate: Union = S × (accepts / N)
    accept_ratio = accepts / num_samples
    estimate = sum_diversity * accept_ratio

    # Standard error using binomial variance
    std_error = sum_diversity * np.sqrt(accept_ratio * (1 - accept_ratio) / num_samples)

    return estimate, std_error
