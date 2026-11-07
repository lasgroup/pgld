from collections import defaultdict
from itertools import combinations_with_replacement
from typing import Optional
import itertools

import numpy as np
import pandas as pd


AA_LIST = list("ACDEFGHIKLMNPQRSTVWY*")
NUCLEOTIDE_LIST = ["A", "C", "G", "T"]

IUPAC_NUCLEOTIDE_CODES = {
    "A": {"A": 1.0},
    "C": {"C": 1.0},
    "G": {"G": 1.0},
    "T": {"T": 1.0},
    "M": {"A": 0.5, "C": 0.5},
    "S": {"C": 0.5, "G": 0.5},
    "K": {"G": 0.5, "T": 0.5},
    "W": {"A": 0.5, "T": 0.5},
    "R": {"A": 0.5, "G": 0.5},
    "Y": {"C": 0.5, "T": 0.5},
    "V": {"A": 1 / 3, "C": 1 / 3, "G": 1 / 3},
    "H": {"A": 1 / 3, "C": 1 / 3, "T": 1 / 3},
    "D": {"A": 1 / 3, "G": 1 / 3, "T": 1 / 3},
    "B": {"C": 1 / 3, "G": 1 / 3, "T": 1 / 3},
    "N": {"A": 0.25, "C": 0.25, "G": 0.25, "T": 0.25},
}

GENETIC_CODE = {
    "ATA": "I",
    "ATC": "I",
    "ATT": "I",
    "ATG": "M",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AAC": "N",
    "AAT": "N",
    "AAA": "K",
    "AAG": "K",
    "AGC": "S",
    "AGT": "S",
    "AGA": "R",
    "AGG": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CAC": "H",
    "CAT": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GAC": "D",
    "GAT": "D",
    "GAA": "E",
    "GAG": "E",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TTC": "F",
    "TTT": "F",
    "TTA": "L",
    "TTG": "L",
    "TAC": "Y",
    "TAT": "Y",
    "TAA": "*",
    "TAG": "*",
    "TGC": "C",
    "TGT": "C",
    "TGA": "*",
    "TGG": "W",
}

DEGENERATE_BASES = {
    "A": ["A"],
    "T": ["T"],
    "C": ["C"],
    "G": ["G"],
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
    "N": ["A", "T", "C", "G"],
}

GENETIC_CODE_CEREVISIAE_USAGE = {
    "F": ["TTT", "TTC"],
    "L": ["TTG", "TTA", "CTA", "CTT", "CTG", "CTC"],
    "Y": ["TAT", "TAC"],
    "H": ["CAT", "CAC"],
    "Q": ["CAA", "CAG"],
    "I": ["ATT", "ATA", "ATC"],
    "M": ["ATG"],
    "N": ["AAT", "AAC"],
    "K": ["AAA", "AAG"],
    "V": ["GTT", "GTC", "GTA", "GTG"],
    "D": ["GAT", "GAC"],
    "E": ["GAA", "GAG"],
    "S": ["TCT", "TCA", "TCC", "AGT", "AGC", "TCG"],
    "C": ["TGT", "TGC"],
    "W": ["TGG"],
    "P": ["CCA", "CCT", "CCC", "CCG"],
    "R": ["AGA", "AGG", "CGT", "CGA", "CGC", "CGG"],
    "A": ["GCT", "GCA", "GCC", "GCG"],
    "G": ["GGT", "GGA", "GGC", "GGG"],
    "T": ["ACT", "ACA", "ACC", "ACG"],
    "*": ["TAA", "TGA", "TAG"],
}


def _expand_degenerate_codon(codon: str) -> list[str]:
    """Expand a degenerate codon into all concrete DNA codons."""
    return [
        "".join(x)
        for x in itertools.product(
            DEGENERATE_BASES[codon[0]],
            DEGENERATE_BASES[codon[1]],
            DEGENERATE_BASES[codon[2]],
        )
    ]


def _generate_all_degenerate_codons() -> list[str]:
    """Generate all possible 3-letter degenerate codons."""
    bases = list(DEGENERATE_BASES.keys())
    return ["".join(codon) for codon in itertools.product(bases, repeat=3)]


def _compute_degenerate_codon_aa_frequencies(
    usage_dict: dict[str, list[str]] = GENETIC_CODE_CEREVISIAE_USAGE,
) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, float]]:
    """
    Compute amino acid frequencies for all degenerate codons.

    Returns:
        (frequencies, codon_num, usage_ranks): Dicts mapping degenerate codons to
        their AA frequency distributions, number of concrete codons, and usage ranks.
    """
    frequencies = {}
    degenerated_codon_num = {}
    codon_usage_ranks = {}

    for deg_codon in _generate_all_degenerate_codons():
        expanded_codons = _expand_degenerate_codon(deg_codon)
        aa_counts = defaultdict(int)
        total_valid_codons = 0
        usage_ranks = []
        degenerated_codon_num[deg_codon] = len(expanded_codons)

        for c in expanded_codons:
            if c in GENETIC_CODE:
                aa = GENETIC_CODE[c]
                aa_counts[aa] += 1
                total_valid_codons += 1
                if aa in usage_dict:
                    codon_list = usage_dict[aa]
                    if c in codon_list:
                        usage_ranks.append(codon_list.index(c))

        if total_valid_codons > 0:
            freq_dict = {aa: aa_counts[aa] / total_valid_codons for aa in aa_counts}
            frequencies[deg_codon] = freq_dict
            codon_usage_ranks[deg_codon] = (
                sum(usage_ranks) if usage_ranks else float("inf")
            )

    return frequencies, degenerated_codon_num, codon_usage_ranks


# Precompute codon data at module load
FREQUENCIES_AAS_DEGENERATE_CODONS, DEGENERATED_CODON_NUM, CODON_USAGE_RANKS = (
    _compute_degenerate_codon_aa_frequencies()
)


def compute_aa_frequencies_from_hits(hit_proteins: list[str]) -> pd.DataFrame:
    """
    Compute position-wise amino acid frequencies from a list of hit protein sequences.

    Args:
        hit_proteins: List of protein sequences (must all be same length).

    Returns:
        DataFrame with index=positions (1-indexed), columns=AA_LIST, values=frequencies.
    """
    if not hit_proteins:
        raise ValueError("No hit proteins provided")

    peptide_length = len(hit_proteins[0])
    n_hits = len(hit_proteins)

    freq_matrix = np.zeros((peptide_length, len(AA_LIST)))
    aa_to_idx = {aa: i for i, aa in enumerate(AA_LIST)}

    for protein in hit_proteins:
        for pos, aa in enumerate(protein):
            if aa in aa_to_idx:
                freq_matrix[pos, aa_to_idx[aa]] += 1

    freq_matrix = freq_matrix / n_hits

    fraq_df = pd.DataFrame(
        freq_matrix, index=range(1, peptide_length + 1), columns=AA_LIST
    )

    return fraq_df


def _find_best_two_codon_mix(
    row_i: pd.Series,
    must_include: list[str],
    avoid_list: list[str],
    penalty_appearance: float = 5,
    penalty_missing: float = 2,
    penalty_stop_codon: float = 5,
    min_coverage: float = 0.8,
    max_error_ratio: float = 1.03,
    hard_avoid_stop: bool = False,
    hard_avoid: bool = False,
) -> Optional[pd.Series]:
    """
    Fallback function to find the best pair of degenerate codons that together
    approximate the target amino acid distribution in equal molar mix.
    """
    candidates = []
    keys = row_i.index.tolist()
    real_residues = list(set(list(row_i[row_i > 0].index) + must_include))

    def freq_vector(freq_dict):
        return np.array([freq_dict.get(k, 0.0) for k in keys])

    target_vec = freq_vector(row_i)

    all_codons = list(FREQUENCIES_AAS_DEGENERATE_CODONS.keys())
    for cod1, cod2 in combinations_with_replacement(all_codons, 2):
        aa_dist1 = FREQUENCIES_AAS_DEGENERATE_CODONS[cod1]
        aa_dist2 = FREQUENCIES_AAS_DEGENERATE_CODONS[cod2]

        combined_keys = set(aa_dist1.keys()) | set(aa_dist2.keys())

        if not all(aa in combined_keys for aa in must_include):
            continue

        if "*" in combined_keys and hard_avoid_stop:
            continue

        if any(aa in avoid_list for aa in combined_keys) and hard_avoid:
            continue

        vec1 = freq_vector(aa_dist1)
        vec2 = freq_vector(aa_dist2)
        avg_vec = 0.5 * (vec1 + vec2)

        total_error = 0
        for i, aa in enumerate(keys):
            if row_i[aa] == 0 and aa in combined_keys:
                if aa == "*":
                    error = avg_vec[i] ** 2 * penalty_stop_codon
                elif aa in avoid_list:
                    error = avg_vec[i] ** 2 * penalty_appearance
                else:
                    error = avg_vec[i] ** 2
            elif row_i[aa] > 0 and aa not in combined_keys:
                error = (row_i[aa]) ** 2 * penalty_missing
            else:
                error = (row_i[aa] - avg_vec[i]) ** 2
            total_error += error
        total_error /= len(AA_LIST)
        all_in_real = set(combined_keys) == set(real_residues)
        coverage = len(set(real_residues) & combined_keys) / len(real_residues)

        candidates.append(
            {
                "position": row_i.name,
                "codon": f"{cod1}+{cod2}",
                "error": total_error,
                "all_in_real": all_in_real,
                "real_aa_coverage": coverage,
                "has_stop": "*" in combined_keys,
                "num_codon": DEGENERATED_CODON_NUM[cod1] + DEGENERATED_CODON_NUM[cod2],
                "num_aa": len(combined_keys),
                "codon_usage_rank": CODON_USAGE_RANKS[cod1] + CODON_USAGE_RANKS[cod2],
            }
        )

    if not candidates:
        return None

    df = pd.DataFrame(candidates)
    try:
        filtered_df = df[df.all_in_real == True]
        if filtered_df.empty:
            raise IndexError
        best_mix = filtered_df.sort_values(
            ["error", "real_aa_coverage", "num_codon", "codon_usage_rank"],
            ascending=[True, False, True, True],
        ).iloc[0]
        return best_mix
    except IndexError:
        print(f"[WARN] No exact match at position {row_i.name}. Relaxing conditions...")

    filtered_df = df[df.real_aa_coverage >= min_coverage]
    if not filtered_df.empty:
        min_error = filtered_df.error.min()
        filtered_df = filtered_df[filtered_df.error <= min_error * max_error_ratio]
        best_mix = filtered_df.sort_values(
            ["real_aa_coverage", "num_codon", "codon_usage_rank"],
            ascending=[False, True, True],
        ).iloc[0]
        return best_mix

    return None


def design_degenerate_codons(
    fraq_df: pd.DataFrame,
    shannon_entropies_norm_dict: dict[int, float],
    wt: Optional[str] = None,
    entropy_threshold_scale: float = 0.1,
    group_threshold: float = 0.3,
    conserve_position_threshold: float = 0.2,
    penalty_appearance: float = 2,
    penalty_missing: float = 2,
    penalty_stop_codon: float = 5,
    hard_avoid_stop: bool = True,
    hard_avoid: bool = True,
    allow_two_codon_mix: bool = False,
    aminoacid_groups: Optional[dict[str, list[str]]] = None,
) -> dict:
    """
    Design optimal degenerate codons to approximate target amino acid frequencies.

    Args:
        fraq_df: DataFrame with AA frequencies (1-indexed positions, AA_LIST columns).
        shannon_entropies_norm_dict: Normalized entropy at each position (1-indexed).
        wt: Optional wild-type protein sequence. If provided, wild-type AAs are
            excluded from avoid list and included in must_include.
        entropy_threshold_scale: Scale for entropy-based filtering of low-freq AAs.
        group_threshold: Threshold for including AA group representatives.
        conserve_position_threshold: Entropy threshold below which position is conserved.
        penalty_appearance: Penalty for unwanted AAs in codon.
        penalty_missing: Penalty for missing wanted AAs in codon.
        penalty_stop_codon: Penalty for stop codons.
        hard_avoid_stop: If True, completely exclude codons with stop codons.
        hard_avoid: If True, completely exclude codons with avoided AAs.
        allow_two_codon_mix: If True, try two-codon mixes when single codon is wasteful.
        aminoacid_groups: Optional dict of AA property groups for constraints.

    Returns:
        Dict with keys: fraq_modify_df, must_include_dict, codon_error,
        candidates_dict (selected codons), avoid_pos_aa.
    """
    if aminoacid_groups is None:
        aminoacid_groups = {}

    positions = list(fraq_df.index)
    fraq_modify_df = fraq_df.copy()
    must_include_dict = {}
    candidates_dict = {}
    codon_error = {}
    avoid_pos_aa = {pos: ["C"] for pos in positions}

    for i, row in fraq_df.iterrows():
        codon_error[i] = {}
        position_entropy = shannon_entropies_norm_dict[i]
        row = row.copy()
        row /= row.sum()

        wt_aa = wt[i - 1] if wt is not None else None
        for aa in row.index:
            if row[aa] == 0 and aa != wt_aa:
                avoid_pos_aa[i].append(aa)
        print(f"[INFO] The aa to avoid at pos {i}: {avoid_pos_aa[i]}")

        row_i = row.copy()
        for aa in row_i.index:
            if aa in avoid_pos_aa[i] or aa == "*":
                row_i[aa] = 0
            elif row_i[aa] < (1 - position_entropy) * entropy_threshold_scale:
                row_i[aa] = 0
        row_i /= row_i.sum()
        fraq_modify_df.loc[i] = row_i

        must_include = [wt_aa] if wt_aa is not None and wt_aa in row_i.index else []
        if position_entropy < conserve_position_threshold:
            must_include.append(row_i.idxmax())

        for group, aa_list in aminoacid_groups.items():
            total_group_freq = row_i[aa_list].sum()
            if total_group_freq > group_threshold and not any(
                aa in must_include for aa in aa_list
            ):
                most_freq_aa = row_i[aa_list].idxmax()
                must_include.append(most_freq_aa)
        must_include_dict[i] = must_include
        print(f"[INFO] The aa must be included at pos {i}: {must_include_dict[i]}")

        candidates_pos = []
        for codon, aa_dist in FREQUENCIES_AAS_DEGENERATE_CODONS.items():
            if not all(aa in aa_dist for aa in must_include):
                continue
            total_error = 0
            for aa in AA_LIST:
                if aa in aa_dist:
                    if row_i[aa] == 0:
                        if aa == "*":
                            error = aa_dist[aa] ** 2 * penalty_stop_codon
                        elif aa in avoid_pos_aa[i]:
                            error = aa_dist[aa] ** 2 * penalty_appearance
                        else:
                            error = aa_dist[aa] ** 2
                    else:
                        error = (row_i[aa] - aa_dist[aa]) ** 2
                else:
                    error = (row_i[aa]) ** 2 * penalty_missing if row_i[aa] != 0 else 0
                total_error += error
            codon_error[i][codon] = total_error / len(AA_LIST)

            real_residues = list(set(list(row_i[row_i > 0].index) + must_include))
            coverage = len(set(aa_dist.keys()) & set(real_residues)) / len(
                set(real_residues)
            )
            all_in_real = set(aa_dist.keys()) == set(real_residues)
            has_stop = "*" in aa_dist
            candidates_pos.append(
                {
                    "position": i,
                    "codon": codon,
                    "error": codon_error[i][codon],
                    "all_in_real": all_in_real,
                    "real_aa_coverage": coverage,
                    "has_stop": has_stop,
                    "num_codon": DEGENERATED_CODON_NUM[codon],
                    "num_aa": len(aa_dist),
                    "codon_usage_rank": CODON_USAGE_RANKS[codon],
                }
            )

        if not candidates_pos:
            print(f"[ERROR] No valid codon candidates found at position {i}")
            continue

        candidates_df = pd.DataFrame(candidates_pos)
        try:
            filtered_df = candidates_df[candidates_df.all_in_real == True]
            if filtered_df.empty:
                raise IndexError
            optimal_deg = filtered_df.sort_values(
                ["error", "real_aa_coverage", "num_codon", "codon_usage_rank"],
                ascending=[True, False, True, True],
            ).iloc[0:5]
        except IndexError:
            print(f"[WARN] No exact match at position {i}. Relaxing conditions...")
            try:
                filtered_df = candidates_df[
                    (candidates_df.real_aa_coverage >= 0.8) & (~candidates_df.has_stop)
                ]
                if filtered_df.empty:
                    raise IndexError
                optimal_deg = filtered_df.sort_values(
                    ["error", "real_aa_coverage", "num_codon", "codon_usage_rank"],
                    ascending=[True, False, True, True],
                ).iloc[0:5]
            except IndexError:
                optimal_deg = candidates_df.sort_values(
                    ["error", "real_aa_coverage", "num_codon", "codon_usage_rank"],
                    ascending=[True, False, True, True],
                ).iloc[0:5]

        optimal_deg.reset_index(inplace=True, drop=True)
        if optimal_deg.empty:
            print(f"[ERROR] No candidates found at position {i}")
            raise ValueError(f"No candidates found at position {i}")
        optimal_one_deg = (
            optimal_deg[optimal_deg.error <= optimal_deg.error.min() * 1.05]
            .sort_values(
                ["real_aa_coverage", "num_codon", "codon_usage_rank"],
                ascending=[False, True, True],
            )
            .iloc[0]
        )

        candidates_dict[i] = optimal_one_deg
        print(
            f"[INFO] Selected optimal codons at position {i}:\n{optimal_deg} "
            f"and final selection is {optimal_one_deg.codon}"
        )

        real_residues = list(set(list(row_i[row_i > 0].index) + must_include))
        if allow_two_codon_mix and optimal_one_deg.num_aa / len(real_residues) > 2:
            print(f"[INFO] Trying 2-codon fallback for position {i}...")
            two_codon_result = _find_best_two_codon_mix(
                row_i=row_i,
                must_include=must_include_dict[i],
                avoid_list=avoid_pos_aa[i],
                hard_avoid_stop=hard_avoid_stop,
                hard_avoid=hard_avoid,
            )
            if two_codon_result is not None:
                print(
                    f"[INFO] Selected 2-codon mix at position {i}: {two_codon_result.codon}"
                )
                candidates_dict[i] = two_codon_result

    return {
        "fraq_modify_df": fraq_modify_df,
        "must_include_dict": must_include_dict,
        "codon_error": codon_error,
        "candidates_dict": candidates_dict,
        "avoid_pos_aa": avoid_pos_aa,
    }


def design_degenerate_codons_simple(
    fraq_df: pd.DataFrame,
    penalty_stop_codon: float = 5.0,
) -> dict[int, str]:
    """
    Simplified codon design using pure MSE optimization.

    For each position, selects the degenerate codon that minimizes MSE between
    target AA frequencies and codon's AA distribution, with a penalty for stop codons.

    Args:
        fraq_df: DataFrame with AA frequencies (1-indexed positions, AA_LIST columns).
        penalty_stop_codon: Penalty multiplier for stop codon probability.

    Returns:
        Dict mapping position (1-indexed) to selected degenerate codon.
    """
    selected_codons: dict[int, str] = {}

    for pos, row in fraq_df.iterrows():
        target = row.values
        best_codon = None
        best_error = float("inf")

        for codon, aa_dist in FREQUENCIES_AAS_DEGENERATE_CODONS.items():
            total_error = 0.0
            for i, aa in enumerate(AA_LIST):
                codon_freq = aa_dist.get(aa, 0.0)
                target_freq = target[i]

                if aa == "*":
                    error = (codon_freq**2) * penalty_stop_codon
                else:
                    error = (target_freq - codon_freq) ** 2

                total_error += error

            mse = total_error / len(AA_LIST)

            if mse < best_error:
                best_error = mse
                best_codon = codon

        selected_codons[pos] = best_codon

    return selected_codons


def compute_nucleotide_frequencies_from_hits(hit_dna_seqs: list[str]) -> pd.DataFrame:
    """
    Compute position-wise nucleotide frequencies from hit DNA sequences.

    Args:
        hit_dna_seqs: List of DNA sequences (must all be same length).

    Returns:
        DataFrame with 1-indexed positions as rows, NUCLEOTIDE_LIST as columns.
    """
    if not hit_dna_seqs:
        raise ValueError("No hit sequences provided")

    seq_length = len(hit_dna_seqs[0])
    n_hits = len(hit_dna_seqs)

    freq_matrix = np.zeros((seq_length, len(NUCLEOTIDE_LIST)))
    nt_to_idx = {nt: i for i, nt in enumerate(NUCLEOTIDE_LIST)}

    for seq in hit_dna_seqs:
        for pos, nt in enumerate(seq.upper()):
            if nt in nt_to_idx:
                freq_matrix[pos, nt_to_idx[nt]] += 1

    freq_matrix = freq_matrix / n_hits

    return pd.DataFrame(
        freq_matrix, index=range(1, seq_length + 1), columns=NUCLEOTIDE_LIST
    )


def design_iupac_codes_simple(fraq_df: pd.DataFrame) -> dict[int, str]:
    """
    Select IUPAC codes to approximate target nucleotide frequencies via MSE.

    Args:
        fraq_df: DataFrame with nucleotide frequencies (1-indexed, NUCLEOTIDE_LIST columns).

    Returns:
        Dict mapping position (1-indexed) to selected IUPAC code.
    """
    selected_codes: dict[int, str] = {}

    for pos, row in fraq_df.iterrows():
        target = row.values
        best_code = None
        best_error = float("inf")

        for code, nt_dist in IUPAC_NUCLEOTIDE_CODES.items():
            total_error = 0.0
            for i, nt in enumerate(NUCLEOTIDE_LIST):
                code_freq = nt_dist.get(nt, 0.0)
                error = (target[i] - code_freq) ** 2
                total_error += error

            mse = total_error / len(NUCLEOTIDE_LIST)

            if mse < best_error:
                best_error = mse
                best_code = code

        selected_codes[pos] = best_code

    return selected_codes


def sample_peptides_from_frequencies(
    fraq_df: pd.DataFrame,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> list[str]:
    """
    Sample peptide sequences directly from position-wise AA frequencies.

    Args:
        fraq_df: DataFrame with AA frequencies (1-indexed positions, AA_LIST columns).
        n_samples: Number of peptide sequences to generate.
        rng: Optional random generator for reproducibility.

    Returns:
        List of sampled peptide sequences.
    """
    if rng is None:
        rng = np.random.default_rng()

    peptides = []
    positions = list(fraq_df.index)
    aa_list = list(fraq_df.columns)

    for _ in range(n_samples):
        peptide = ""
        for pos in positions:
            probs = fraq_df.loc[pos].values
            probs = probs / probs.sum()
            aa = rng.choice(aa_list, p=probs)
            peptide += aa
        peptides.append(peptide)

    return peptides


def peptide_to_dna(
    peptide: str,
    codon_table: dict[str, list[str]] = GENETIC_CODE_CEREVISIAE_USAGE,
) -> str:
    """
    Convert a peptide sequence to DNA using preferred codons.

    Args:
        peptide: Amino acid sequence.
        codon_table: Dict mapping AA to list of codons (first is most preferred).

    Returns:
        DNA sequence.
    """
    dna = ""
    for aa in peptide:
        if aa in codon_table:
            dna += codon_table[aa][0]
        else:
            raise ValueError(f"Unknown amino acid: {aa}")
    return dna


def sample_dna_from_frequencies(
    fraq_df: pd.DataFrame,
    n_samples: int,
    rng: Optional[np.random.Generator] = None,
) -> list[str]:
    """
    Sample DNA sequences by sampling AAs from frequencies and converting to DNA.

    Args:
        fraq_df: DataFrame with AA frequencies (1-indexed positions, AA_LIST columns).
        n_samples: Number of DNA sequences to generate.
        rng: Optional random generator for reproducibility.

    Returns:
        List of sampled DNA sequences.
    """
    peptides = sample_peptides_from_frequencies(fraq_df, n_samples, rng)
    return [peptide_to_dna(p) for p in peptides]
