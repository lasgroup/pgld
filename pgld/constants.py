"""
Biological and genetic constants used across the codebase.
"""

from typing import List

import numpy as np
import torch
import torch.nn.functional as F

# DNA bases in canonical order (A, C, G, T)
DNA_BASES = ["A", "C", "G", "T"]

# DNA base to index mapping (derived from DNA_BASES order)
DNA_ALPHABET = {base: idx for idx, base in enumerate(DNA_BASES)}

# Amino acid vocabulary and mapping
AA_VOCAB_NO_STOP = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX_NO_STOP = {aa: i for i, aa in enumerate(AA_VOCAB_NO_STOP)}
MAX_AA_INDEX = len(AA_VOCAB_NO_STOP) - 1  # = 19, excludes stop codon at index 20

# Standard codon table mapping DNA codons to amino acids
CODON_TABLE = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGT": "C",
    "TGC": "C",
    "TGA": "*",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
}

# Reverse codon table: amino acid -> list of codons that encode it
AA_TO_CODONS: dict[str, list[str]] = {}
for _codon, _aa in CODON_TABLE.items():
    if _aa not in AA_TO_CODONS:
        AA_TO_CODONS[_aa] = []
    AA_TO_CODONS[_aa].append(_codon)

# Standard amino acids (20 + stop)
AMINO_ACIDS = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
    "*",
]

# amino acid to index mapping (derived from the order of AMINO ACIDS)
AMINO_ACID_ALPHABET = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}

# IUPAC nucleotide ambiguity codes
IUPAC_MAP = {
    "A": ["A"],
    "C": ["C"],
    "G": ["G"],
    "T": ["T"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
    "N": ["A", "C", "G", "T"],
}

# IUPAC bottle definitions (bottle_idx: (code, base_probs))
# base_probs are in ACGT order
IUPAC_BOTTLES = [
    ("A", [1, 0, 0, 0]),  # 0: A
    ("C", [0, 1, 0, 0]),  # 1: C
    ("G", [0, 0, 1, 0]),  # 2: G
    ("T", [0, 0, 0, 1]),  # 3: T
    ("M", [1, 1, 0, 0]),  # 4: M (A or C)
    ("S", [0, 1, 1, 0]),  # 5: S (C or G)
    ("K", [0, 0, 1, 1]),  # 6: K (G or T)
    ("W", [1, 0, 0, 1]),  # 7: W (A or T)
    ("R", [1, 0, 1, 0]),  # 8: R (A or G)
    ("Y", [0, 1, 0, 1]),  # 9: Y (C or T)
    ("V", [1, 1, 1, 0]),  # 10: V (A or C or G)
    ("H", [1, 1, 0, 1]),  # 11: H (A or C or T)
    ("D", [1, 0, 1, 1]),  # 12: D (A or G or T)
    ("B", [0, 1, 1, 1]),  # 13: B (C or G or T)
    ("N", [1, 1, 1, 1]),  # 14: N (A or C or G or T)
]

# IUPAC code to bottle index mapping
IUPAC_TO_BOTTLE_IDX = {code: idx for idx, (code, _) in enumerate(IUPAC_BOTTLES)}

# Probability threshold for counting a base/amino acid as "possible" in diversity calculations
DEFAULT_DIVERSITY_THRESHOLD = 0.03


def one_hot_encode_peptide(seq, length):
    """
    For a given sequence of amino-acids returns the one-hot encoding of it.
    The one hot encoding is of length 'length', which means that if the
    sequence is shorter we are padding.

    Args:
        seq: Amino acid sequence string
        length: Target length for padding

    Returns:
        One-hot encoded tensor of shape (length, vocab_size)
    """
    seq = seq.upper()
    one_hot = torch.zeros(length, len(AA_TO_IDX_NO_STOP))
    for i, aa in enumerate(seq):
        if aa in AA_TO_IDX_NO_STOP:
            one_hot[i, AA_TO_IDX_NO_STOP[aa]] = 1.0
    return one_hot


def one_hot_encode_dna(seq, length):
    """
    For a given sequence of nucleotides returns the one-hot encoding of it.
    The one hot encoding is of length 'length', which means that if the
    sequence is shorter we are padding.

    Args:
        seq: sequence of nucleotides string
        length: Target length for padding

    Returns:
        One-hot encoded tensor of shape (length, vocab_size)
    """
    seq = seq.upper()
    one_hot = torch.zeros(length, len(DNA_ALPHABET))
    for i, nucleotide in enumerate(seq):
        if nucleotide in DNA_ALPHABET:
            one_hot[i, DNA_ALPHABET[nucleotide]] = 1.0
    return one_hot


_AA_ORD_TO_IDX: np.ndarray = np.full(128, -1, dtype=np.int64)
for _aa, _idx in AA_TO_IDX_NO_STOP.items():
    _AA_ORD_TO_IDX[ord(_aa)] = _idx

_DNA_ORD_TO_IDX: np.ndarray = np.full(128, -1, dtype=np.int64)
for _base, _idx in DNA_ALPHABET.items():
    _DNA_ORD_TO_IDX[ord(_base)] = _idx


def batch_one_hot_encode_peptide(
    sequences: List[str],
    length: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Vectorized one-hot encoding for a batch of peptide sequences."""
    num_classes = len(AA_TO_IDX_NO_STOP)

    seq_array = np.array([list(s.upper().ljust(length, "X")) for s in sequences])
    ord_array = np.vectorize(ord)(seq_array)
    idx_array = _AA_ORD_TO_IDX[ord_array]

    valid_mask = idx_array >= 0
    idx_array = np.clip(idx_array, 0, num_classes - 1)

    indices = torch.from_numpy(idx_array).to(device)
    one_hot = F.one_hot(indices, num_classes=num_classes).float()

    mask = torch.from_numpy(valid_mask).to(device).unsqueeze(-1)
    one_hot = one_hot * mask

    return one_hot


def batch_one_hot_encode_dna(
    sequences: List[str],
    length: int,
    device: str = "cpu",
) -> torch.Tensor:
    """Vectorized one-hot encoding for a batch of DNA sequences."""
    num_classes = len(DNA_ALPHABET)

    seq_array = np.array([list(s.upper().ljust(length, "X")) for s in sequences])
    ord_array = np.vectorize(ord)(seq_array)
    idx_array = _DNA_ORD_TO_IDX[ord_array]

    valid_mask = idx_array >= 0
    idx_array = np.clip(idx_array, 0, num_classes - 1)

    indices = torch.from_numpy(idx_array).to(device)
    one_hot = F.one_hot(indices, num_classes=num_classes).float()

    mask = torch.from_numpy(valid_mask).to(device).unsqueeze(-1)
    one_hot = one_hot * mask

    return one_hot


# IgLM tokenizer vocab (from github.com/Graylab/IgLM/blob/main/iglm/trained_models/vocab.txt)
# Token ID = line number in vocab file (0-indexed)
IGLM_VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "[MASK]": 4,
    "A": 5,
    "C": 6,
    "D": 7,
    "E": 8,
    "F": 9,
    "G": 10,
    "H": 11,
    "I": 12,
    "K": 13,
    "L": 14,
    "M": 15,
    "N": 16,
    "P": 17,
    "Q": 18,
    "R": 19,
    "S": 20,
    "T": 21,
    "V": 22,
    "W": 23,
    "Y": 24,
    "[CAMEL]": 25,
    "[HUMAN]": 26,
    "[MOUSE]": 27,
    "[RABBIT]": 28,
    "[RAT]": 29,
    "[RHESUS]": 30,
    "[HEAVY]": 31,
    "[LIGHT]": 32,
}


# Paper plot colors (consistent across all comparison plots)
PAPER_COLORS = {
    "PGLD": "#1f77b4",  # blue
    "Var-Synth": "#ff7f0e",  # orange
    "PLM Samples": "#888888",  # grey
    "Reference": "#888888",  # grey (same as PLM Samples)
    "BO": "#1f77b4",  # blue
    "Rational": "#ff7f0e",  # orange
    "Random": "#888888",  # grey
    # Policy comparison colors
    "IUPAC": "#1f77b4",  # blue (same as PGLD)
    "IUPAC_8": "#2ca02c",  # green - 8 template IUPAC
    "Arbitrary": "#ff7f0e",  # orange (same as Var-Synth)
    "NNK": "#888888",  # grey (same as Reference)
}

# Paper plot margins (tight, minimal whitespace)
PAPER_MARGINS = {
    "left": 0.20,
    "right": 0.95,
    "top": 0.88,
    "bottom": 0.16,
}
