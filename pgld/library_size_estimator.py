"""
library_size_estimator.py

Computes the total theoretical diversity of a sequence library as the union
of all templates using the Inclusion-Exclusion principle.

For peptide landscapes, diversity is computed for FULL-LENGTH peptides only
(sequences without internal stop codons). This is the relevant metric for
libraries designed for full-length protein expression.
"""

import logging
import time
from typing import Optional

import numpy as np

from approx_union_diversity import estimate_union_diversity_karp_luby
from constants import DEFAULT_DIVERSITY_THRESHOLD
from diversity_utils import compute_mask_diversity
from vis import dna_to_amino_acid_probabilities

log = logging.getLogger(__name__)


def _enumerate_sequences_from_mask(mask: np.ndarray) -> set[tuple[int, ...]]:
    """
    Enumerates all actual sequences allowed by a boolean mask.

    The mask indicates which characters are allowed at each position. This
    function returns the Cartesian product of allowed characters, i.e., all
    concrete sequences that could be generated from the template.

    Used for testing only - exponential in sequence length.

    Complexity: O(A^L * L) where L is sequence length and A is alphabet size.
    Each of the up to A^L sequences requires O(L) to construct and hash.

    Args:
        mask: Array of shape (length, alphabet_size) where values > 0 indicate allowed

    Returns:
        Set of tuples, each tuple is a concrete sequence (alphabet indices)
    """
    if mask.shape[0] == 0:
        return {()}

    sequences = {()}
    for pos in range(mask.shape[0]):
        allowed_indices = np.where(mask[pos] > 0)[0]
        if len(allowed_indices) == 0:
            return set()  # No valid sequences possible
        new_sequences = set()
        for seq in sequences:
            for idx in allowed_indices:
                new_sequences.add(seq + (idx,))
        sequences = new_sequences

    return sequences


def compute_diversity_exhaustive(masks: list[np.ndarray]) -> int:
    """
    Computes union diversity by exhaustively enumerating all concrete sequences.

    For each template mask, generates all actual sequences (Cartesian product of
    allowed characters), then takes the set union and counts. This is the ground
    truth implementation used for testing the inclusion-exclusion algorithm.

    Complexity: O(M * A^L * L) where M is number of templates, L is sequence
    length, and A is alphabet size. We enumerate up to A^L sequences per
    template, each taking O(L) to construct and hash, across M templates.

    Args:
        masks: List of boolean masks, each shape (length, alphabet_size)

    Returns:
        Number of unique concrete sequences in the union of all templates
    """
    all_sequences = set()
    for mask in masks:
        all_sequences |= _enumerate_sequences_from_mask(mask)
    return len(all_sequences)


def compute_union_diversity(masks: list[np.ndarray]) -> float:
    """
    Computes |A ∪ B ∪ C ...| using Inclusion-Exclusion with pruning.

    Uses a recursive approach where branches with zero intersection diversity
    are pruned, since all supersets also have zero diversity.

    Complexity: O(2^M * L * A) worst case, where M is number of templates,
    L is sequence length, and A is alphabet size. The 2^M comes from
    inclusion-exclusion requiring all subsets; each subset intersection
    takes O(L * A) to compute. Pruning often reduces this significantly
    when templates have disjoint positions.

    Args:
        masks: List of boolean masks, each shape (length, alphabet_size)

    Returns:
        Total union diversity
    """
    if len(masks) == 0:
        return 0.0
    if len(masks) == 1:
        return compute_mask_diversity(masks[0])

    total_diversity = 0.0

    def _recurse(idx: int, current_intersection: Optional[np.ndarray], sign: float):
        nonlocal total_diversity

        for i in range(idx, len(masks)):
            new_mask = masks[i]

            if current_intersection is None:
                next_intersection = new_mask
            else:
                next_intersection = np.minimum(current_intersection, new_mask)

            div = compute_mask_diversity(next_intersection)

            if div > 0:
                total_diversity += sign * div
                _recurse(i + 1, next_intersection, -sign)

    _recurse(0, None, 1.0)
    return total_diversity


def _build_template_masks(
    template_dna_probs: np.ndarray,
    is_peptide_landscape: bool,
    threshold: float,
) -> list[np.ndarray]:
    """
    Builds boolean masks from template DNA probabilities.

    Args:
        template_dna_probs: Array of shape (num_templates, length, 4)
        is_peptide_landscape: If True, convert to AA probs and exclude stop codons
        threshold: Probability threshold for considering a base/AA as possible

    Returns:
        List of boolean masks, one per template
    """
    num_templates = template_dna_probs.shape[0]
    masks = []

    for t in range(num_templates):
        if is_peptide_landscape:
            probs = dna_to_amino_acid_probabilities(template_dna_probs[t])
            probs = probs[:, :-1]  # Exclude stop codon (last column)
        else:
            probs = template_dna_probs[t]

        mask = (probs > threshold).astype(int)
        masks.append(mask)

    return masks


def _empty_result(reason: str) -> dict:
    """Returns an empty diversity result with the given skip reason."""
    return {
        "individual_diversities": [],
        "sum_diversity": 0.0,
        "union_diversity": None,
        "union_skipped_reason": reason,
        "union_is_approximate": False,
        "union_std_error": None,
    }


def _union_of_masks(masks: list[np.ndarray]) -> tuple[float, bool, Optional[float]]:
    """Union diversity of a list of same-shape boolean masks.

    Encapsulates the N-dependent estimator choice: exact for <=20 masks,
    Karp-Luby Monte-Carlo approximation above that.

    Returns:
        (union_diversity, is_approximate, std_error)
    """
    n = len(masks)
    if n == 0:
        return 0.0, False, None
    if n == 1:
        return compute_mask_diversity(masks[0]), False, None
    if n <= 20:
        return compute_union_diversity(masks), False, None
    estimate, std_error = estimate_union_diversity_karp_luby(masks)
    return estimate, True, std_error


def _multilength_peptide_diversity(
    template_dna_probs: np.ndarray,
    peptide_lengths: list[int],
    threshold: float,
) -> dict:
    """
    Union diversity summed over admitted peptide lengths (e.g. 8 and 9).

    Per length L, count distinct L-mers from templates whose codon L can be a
    stop (prob > threshold); L == num_codons needs no terminating stop. Lengths
    are disjoint, so the total is the sum of per-length unions.
    """
    num_templates = template_dna_probs.shape[0]
    num_codons = template_dna_probs.shape[1] // 3
    aa_probs_list = [
        dna_to_amino_acid_probabilities(template_dna_probs[t])  # (num_codons, 21)
        for t in range(num_templates)
    ]
    lengths = sorted({L for L in peptide_lengths if 1 <= L <= num_codons})

    per_template = [0.0] * num_templates
    union_total = 0.0
    any_approx = False
    var_total = 0.0
    for length in lengths:
        masks = []
        for t, aa_probs in enumerate(aa_probs_list):
            if length < num_codons and aa_probs[length, -1] <= threshold:
                continue  # this template can't terminate at exactly `length`
            mask = (aa_probs[:length, :-1] > threshold).astype(int)  # (length, 20)
            masks.append(mask)
            per_template[t] += compute_mask_diversity(mask)
        union_len, approx, std_error = _union_of_masks(masks)
        union_total += union_len
        any_approx = any_approx or approx
        if std_error is not None:
            var_total += std_error ** 2

    return {
        "individual_diversities": per_template,
        "sum_diversity": float(sum(per_template)),
        "union_diversity": float(union_total),
        "union_skipped_reason": None,
        "union_is_approximate": any_approx,
        "union_std_error": (float(var_total ** 0.5) if any_approx else None),
    }


def compute_total_library_diversity(
    policy,
    is_peptide_landscape: bool = False,
    threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    peptide_lengths: Optional[list[int]] = None,
) -> dict:
    """
    Computes total library diversity (union of all templates).

    Args:
        policy: Policy with get_template_dna_probs() method
        is_peptide_landscape: If True, compute amino acid diversity (see note below)
        threshold: Probability threshold for considering a base/AA as possible
        peptide_lengths: If given (peptide landscapes only), sum the union
            diversity over these admitted peptide lengths, counting shorter
            peptides that terminate at an internal stop (e.g. [8, 9]). If None,
            only full-length peptides are counted (see note below).

    Returns:
        Dict with:
          - 'individual_diversities': List of per-template diversities
          - 'sum_diversity': Sum of individual diversities
          - 'union_diversity': Union diversity (None if skipped for N=1)
          - 'union_skipped_reason': Reason if union was skipped (None otherwise)
          - 'union_is_approximate': True if Karp-Luby approximation was used (>20 templates)
          - 'union_std_error': Standard error of estimate (None if exact or skipped)

    Note on peptide diversity:
        When is_peptide_landscape=True, this computes diversity of FULL-LENGTH
        peptides only (sequences with no internal stop codons). Sequences that
        would terminate early due to stop codons are not counted. This is the
        relevant metric when designing libraries for full-length protein expression.

        To compute diversity including truncated peptides would require tracking
        all possible termination points, which is combinatorially complex.
    """
    if not hasattr(policy, "get_template_dna_probs"):
        return _empty_result("Policy does not support get_template_dna_probs()")

    template_dna_probs = policy.get_template_dna_probs()
    num_templates = template_dna_probs.shape[0]

    if is_peptide_landscape and peptide_lengths is not None:
        return _multilength_peptide_diversity(
            template_dna_probs, peptide_lengths, threshold
        )

    masks = _build_template_masks(template_dna_probs, is_peptide_landscape, threshold)
    individual_diversities = [compute_mask_diversity(m) for m in masks]
    sum_diversity = sum(individual_diversities)

    result = {
        "individual_diversities": individual_diversities,
        "sum_diversity": sum_diversity,
        "union_diversity": None,
        "union_skipped_reason": None,
        "union_is_approximate": False,
        "union_std_error": None,
    }

    if num_templates == 1:
        result["union_skipped_reason"] = "N=1 (trivial, same as sum)"
    elif num_templates > 20:
        start_time = time.perf_counter()
        estimate, std_error = estimate_union_diversity_karp_luby(masks)
        result["union_diversity"] = estimate
        result["union_is_approximate"] = True
        result["union_std_error"] = std_error
        elapsed_time = time.perf_counter() - start_time
        log.info(
            f"Approximate diversity computation (Karp-Luby) took {elapsed_time:.3f}s"
        )
    else:
        start_time = time.perf_counter()
        result["union_diversity"] = compute_union_diversity(masks)
        elapsed_time = time.perf_counter() - start_time
        log.info(f"Theoretical diversity computation took {elapsed_time:.3f}s")

    return result


def print_total_library_diversity(
    policy,
    alphabet: list[str],
    is_peptide_landscape: bool = False,
    log_to_file: Optional[str] = None,
    threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    verbose: bool = True,
) -> dict:
    """
    Computes and prints the total theoretical diversity (union of all templates).

    Args:
        policy: Policy with get_template_dna_probs() method
        alphabet: DNA alphabet (used for validation)
        is_peptide_landscape: If True, compute amino acid diversity
        log_to_file: Optional file path to save output
        threshold: Probability threshold for considering a base/AA as possible
        verbose: If True, log output to console

    Returns:
        Dict with diversity results (see compute_total_library_diversity)
    """
    if len(alphabet) != 4 or set(alphabet) != {"A", "C", "G", "T"}:
        return _empty_result("Non-DNA alphabet")

    result = compute_total_library_diversity(
        policy,
        is_peptide_landscape=is_peptide_landscape,
        threshold=threshold,
    )

    if not result["individual_diversities"]:
        return result

    num_templates = len(result["individual_diversities"])
    if is_peptide_landscape:
        sequence_type = "Protein Sequences (full-length only, no stop codons)"
    else:
        sequence_type = "DNA Sequences"

    def format_diversity(d: float) -> str:
        if d >= 1e6:
            return f"{d:.2e}"
        return f"{int(d):,}"

    lines = [
        f"\n{'='*80}",
        f"TOTAL LIBRARY THEORETICAL DIVERSITY ({sequence_type})",
        f"{'='*80}",
        f"Number of templates: {num_templates}",
        "",
        f"Sum of individual diversities: {format_diversity(result['sum_diversity'])}",
    ]

    if result["union_diversity"] is not None:
        if result["union_is_approximate"]:
            std_err = result["union_std_error"]
            lines.append(
                f"Union diversity (approx):      {format_diversity(result['union_diversity'])} "
                f"(± {format_diversity(std_err)})"
            )
        else:
            lines.append(
                f"Union diversity (exact):       {format_diversity(result['union_diversity'])}"
            )
        overlap = result["sum_diversity"] - result["union_diversity"]
        if overlap > 0:
            lines.append(f"Overlap (double-counted):      {format_diversity(overlap)}")
    elif result["union_skipped_reason"]:
        lines.append(f"Union diversity: skipped ({result['union_skipped_reason']})")

    lines.append(f"{'='*80}\n")

    output = "\n".join(lines)
    if verbose:
        log.info(output)

    if log_to_file:
        with open(log_to_file, "a") as f:
            f.write(output)

    return result
