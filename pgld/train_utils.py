"""Functions used in train.py"""

import logging

import editdistance
import numpy as np

from landscapes import dna_to_protein

log = logging.getLogger(__name__)


def compute_peptide_length_stats(sequences):
    """
    Compute peptide length statistics for DNA sequences.

    Args:
        sequences: List of DNA sequence strings

    Returns:
        Dict with percentiles (0, 25, 50, 75, 100) of peptide lengths
    """
    peptides = [dna_to_protein(seq) for seq in sequences]
    lengths = np.array([len(p) for p in peptides])

    if len(lengths) == 0:
        return {
            "peptide_len_p0": 0,
            "peptide_len_p25": 0,
            "peptide_len_p50": 0,
            "peptide_len_p75": 0,
            "peptide_len_p100": 0,
        }

    return {
        "peptide_len_p0": int(np.percentile(lengths, 0)),
        "peptide_len_p25": int(np.percentile(lengths, 25)),
        "peptide_len_p50": int(np.percentile(lengths, 50)),
        "peptide_len_p75": int(np.percentile(lengths, 75)),
        "peptide_len_p100": int(np.percentile(lengths, 100)),
    }


def compute_edit_distance_stats(sequences, wild_type):
    """
    Compute edit distance statistics from wild-type for DNA sequences.
    For sequences shorter than wild-type, count missing positions as edits.

    Args:
        sequences: List of DNA sequence strings
        wild_type: Wild-type peptide sequence string

    Returns:
        Dict with percentiles (0, 25, 50, 75, 100) of edit distances
    """
    peptides = [dna_to_protein(seq) for seq in sequences]

    distances = []
    for pep in peptides:
        if len(pep) < len(wild_type):
            # Count missing positions as edits
            dist = editdistance.eval(pep, wild_type[: len(pep)]) + (
                len(wild_type) - len(pep)
            )
        else:
            dist = editdistance.eval(pep, wild_type)
        distances.append(dist)

    distances = np.array(distances)

    if len(distances) == 0:
        return {
            "edit_dist_p0": 0,
            "edit_dist_p25": 0,
            "edit_dist_p50": 0,
            "edit_dist_p75": 0,
            "edit_dist_p100": 0,
        }

    return {
        "edit_dist_p0": int(np.percentile(distances, 0)),
        "edit_dist_p25": int(np.percentile(distances, 25)),
        "edit_dist_p50": int(np.percentile(distances, 50)),
        "edit_dist_p75": int(np.percentile(distances, 75)),
        "edit_dist_p100": int(np.percentile(distances, 100)),
    }
