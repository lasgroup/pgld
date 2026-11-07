"""Exhaustive checks for the multi-length peptide diversity (lengths 8 and 9).

Each library is small enough to enumerate every plausible peptide by hand, so
`compute_total_library_diversity(..., peptide_lengths=...)` is compared against a
brute-force set-union count. Codons are built so their >3% amino-acid support is
known: GCC=A, TTT=F, TTA=L, TGG=W, TTG=L, TAC=Y, TAA=stop.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from library_size_estimator import compute_total_library_diversity
from vis import dna_to_amino_acid_probabilities

THRESHOLD = 0.03
N_AA = 20  # non-stop amino acids; stop is column 20

# base indices for DNA_BASES = [A, C, G, T]
A, C, G, T = 0, 1, 2, 3

GCC = ((G,), (C,), (C,))  # -> Ala only
FL = ((T,), (T,), (A, T))  # -> {Phe, Leu}
LW = ((T,), (G, T), (G,))  # -> {Trp, Leu}
STOP_Y = ((T,), (A,), (A, C))  # -> {stop, Tyr}
DET_Y = ((T,), (A,), (C,))  # -> Tyr only


def _position(bases: tuple) -> np.ndarray:
    """Uniform base distribution over the given base indices."""
    v = np.zeros(4)
    for b in bases:
        v[b] = 1.0 / len(bases)
    return v


def _template(codons: list) -> np.ndarray:
    """Stack 9 codon specs into a (27, 4) DNA-probability array."""
    rows = []
    for b1, b2, b3 in codons:
        rows += [_position(b1), _position(b2), _position(b3)]
    return np.stack(rows, axis=0)


class _StubPolicy:
    def __init__(self, templates: list):
        self._probs = np.stack(templates, axis=0)  # (T, 27, 4)

    def get_template_dna_probs(self) -> np.ndarray:
        return self._probs


def _bruteforce_union(templates: list, lengths: list, threshold: float) -> int:
    """Count distinct plausible peptides across templates and lengths by hand."""
    seqs = set()
    for probs in templates:
        aa = dna_to_amino_acid_probabilities(probs)  # (num_codons, 21)
        num_codons = aa.shape[0]
        support = [
            frozenset(i for i in range(N_AA) if aa[pos, i] > threshold)
            for pos in range(num_codons)
        ]
        stop_ok = [aa[pos, N_AA] > threshold for pos in range(num_codons)]
        for length in lengths:
            if length > num_codons:
                continue
            if length < num_codons and not stop_ok[length]:
                continue
            pools = [sorted(support[p]) for p in range(length)]
            if any(len(p) == 0 for p in pools):
                continue
            for combo in product(*pools):
                seqs.add((length,) + combo)  # tag by length: 8-mers and 9-mers disjoint
    return len(seqs)


def _union(templates: list, peptide_lengths, threshold: float = THRESHOLD):
    result = compute_total_library_diversity(
        _StubPolicy(templates),
        is_peptide_landscape=True,
        threshold=threshold,
        peptide_lengths=peptide_lengths,
    )
    assert not result["union_is_approximate"]
    return result


def test_single_template_8_and_9():
    # codon 9 can be a stop -> both an 8-mer (drop pos-9) and 9-mers exist.
    # 9-mers: {F,L} x {Y} = 2 ; 8-mers: {F,L} = 2 ; total 4.
    templates = [_template([GCC] * 7 + [FL, STOP_Y])]
    result = _union(templates, [8, 9])
    assert result["union_diversity"] == _bruteforce_union(templates, [8, 9], THRESHOLD)
    assert result["union_diversity"] == 4


def test_stop_gate_excludes_8mers():
    # codon 9 is a fixed non-stop AA -> no 8-mers; 8+9 must equal 9-only.
    templates = [_template([GCC] * 7 + [FL, DET_Y])]
    both = _union(templates, [8, 9])
    nine = _union(templates, [9])
    assert both["union_diversity"] == nine["union_diversity"]
    assert both["union_diversity"] == _bruteforce_union(templates, [8, 9], THRESHOLD)
    assert both["union_diversity"] == 2


def test_length9_reduces_to_full_length_path():
    # peptide_lengths=[9] must equal the original peptide_lengths=None path.
    templates = [
        _template([GCC] * 7 + [FL, DET_Y]),
        _template([GCC] * 7 + [LW, DET_Y]),
    ]
    nine = _union(templates, [9])
    default = compute_total_library_diversity(
        _StubPolicy(templates), is_peptide_landscape=True, threshold=THRESHOLD
    )
    assert nine["union_diversity"] == default["union_diversity"]
    assert nine["union_diversity"] == _bruteforce_union(templates, [9], THRESHOLD)


def test_two_templates_union_below_sum():
    # Overlapping 9-mers across templates: union must dedup (union < sum).
    # 9-mers: {(F,Y),(L,Y)} U {(L,Y),(W,Y)} = 3 ; 8-mers: {F,L} (only template 0) = 2.
    templates = [
        _template([GCC] * 7 + [FL, STOP_Y]),
        _template([GCC] * 7 + [LW, DET_Y]),
    ]
    result = _union(templates, [8, 9])
    assert result["union_diversity"] == _bruteforce_union(templates, [8, 9], THRESHOLD)
    assert result["union_diversity"] == 5
    assert result["union_diversity"] < result["sum_diversity"]
