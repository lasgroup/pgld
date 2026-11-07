"""
landscapes.py

Defines the sequence-to-function landscapes used for the optimization experiments.
Includes helper functions for enumerating and scoring sequences.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import contextlib
import copy
import os
import sys
import warnings
from itertools import product

import numpy as np
from omegaconf import OmegaConf
import pandas as pd
import torch
import torch.nn.functional as F
import flexs_wrapper as flexs

from constants import (
    DNA_BASES,
    DNA_ALPHABET,
    CODON_TABLE,
    AMINO_ACIDS,
    AMINO_ACID_ALPHABET,
    IUPAC_MAP,
    batch_one_hot_encode_dna,
    batch_one_hot_encode_peptide,
)
# Conditionally import mhcflurry (must be before logger due to a bug)
# This allows running in the pgld_plm environment (new numpy) without mhcflurry
try:
    from mhcflurry import Class1AffinityPredictor, Class1ProcessingPredictor

    MHCFLURRY_AVAILABLE = True
except ImportError:
    warnings.filterwarnings("ignore")
    MHCFLURRY_AVAILABLE = False
    Class1AffinityPredictor = None
    Class1ProcessingPredictor = None

# Ignore MHCflurry warnings about updates
warnings.filterwarnings("ignore", category=FutureWarning)

log = logging.getLogger(__name__)


def sequences_to_strings_vectorized(sequences, alphabet):
    """
    Vectorized conversion of sequence tensors to strings.
    ~5-10x faster than list comprehension.

    Args:
        sequences: torch.Tensor or numpy array of shape (n_sequences, seq_length) with integer indices
        alphabet: list of characters (e.g., ['A', 'C', 'G', 'T'])

    Returns:
        list of strings
    """
    alphabet_arr = np.array(list(alphabet))
    seqs_np = (
        sequences.cpu().numpy() if hasattr(sequences, "cpu") else np.asarray(sequences)
    )
    chars = alphabet_arr[seqs_np]
    return ["".join(row) for row in chars]


def dna_to_protein_batch_loop(dna_sequences):
    """
    Loop-based DNA to protein translation. Use for small batches or variable-length sequences.

    Args:
        dna_sequences: iterable of DNA sequence strings

    Returns:
        dict mapping DNA sequence to protein sequence
    """
    results = {}
    for dna_seq in dna_sequences:
        protein = ""
        dna_upper = dna_seq.upper()
        for i in range(0, len(dna_upper) - (len(dna_upper) % 3), 3):
            codon = dna_upper[i : i + 3]
            aa = CODON_TABLE.get(codon, "")
            if aa == "*":
                break
            protein += aa
        results[dna_seq] = protein
    return results


def dna_to_protein_batch(dna_sequences):
    """
    Batch translate DNA sequences to protein.
    Uses vectorized numpy for large batches (>=100 seqs, same length).
    Falls back to loop for small/variable-length batches.

    Args:
        dna_sequences: iterable of DNA sequence strings

    Returns:
        dict mapping DNA sequence to protein sequence
    """
    dna_list = list(dna_sequences)
    if not dna_list:
        return {}

    n_seqs = len(dna_list)

    # Use vectorized for large uniform-length batches
    if n_seqs >= 100:
        first_len = len(dna_list[0])
        all_same_length = all(len(seq) == first_len for seq in dna_list)
        if all_same_length:
            return dna_to_protein_vectorized(dna_list)

    # Use loop for small or variable-length batches
    return dna_to_protein_batch_loop(dna_list)


def dna_to_protein_vectorized(dna_sequences):
    """
    Vectorized batch translation of DNA sequences to protein using numpy.
    ~10-50x faster than dna_to_protein_batch for large batches (1000+ sequences).
    Handles stop codons by truncating at first occurrence.

    Args:
        dna_sequences: list of DNA sequence strings (all same length)

    Returns:
        dict mapping DNA sequence to protein sequence
    """
    if not dna_sequences:
        return {}

    dna_list = list(dna_sequences)
    n_seqs = len(dna_list)

    # All sequences must be same length for vectorization
    seq_len = len(dna_list[0])
    if seq_len % 3 != 0:
        seq_len = seq_len - (seq_len % 3)

    n_codons = seq_len // 3

    # Build codon to AA lookup table (64 codons)
    codon_to_aa_char = [""] * 64
    for i, b1 in enumerate(DNA_BASES):
        for j, b2 in enumerate(DNA_BASES):
            for k, b3 in enumerate(DNA_BASES):
                codon = b1 + b2 + b3
                codon_idx = i * 16 + j * 4 + k
                codon_to_aa_char[codon_idx] = CODON_TABLE[codon]

    # Convert DNA strings to nucleotide indices (N, seq_len)
    dna_arr = np.zeros((n_seqs, seq_len), dtype=np.int8)
    for i, seq in enumerate(dna_list):
        seq_upper = seq[:seq_len].upper()
        for j, char in enumerate(seq_upper):
            dna_arr[i, j] = DNA_ALPHABET.get(char, 0)

    # Reshape to codons (N, n_codons, 3)
    dna_codons = dna_arr.reshape(n_seqs, n_codons, 3)

    # Convert codon triplets to codon indices (0-63)
    codon_indices = (
        dna_codons[:, :, 0] * 16 + dna_codons[:, :, 1] * 4 + dna_codons[:, :, 2]
    )

    # Vectorized lookup to get amino acids
    aa_chars = np.array([codon_to_aa_char[idx] for idx in codon_indices.flat]).reshape(
        n_seqs, n_codons
    )

    # Handle stop codons: find first stop in each sequence
    results = {}
    for i, seq in enumerate(dna_list):
        aas = aa_chars[i]
        # Find first stop codon
        stop_indices = np.where(aas == "*")[0]
        if len(stop_indices) > 0:
            protein = "".join(aas[: stop_indices[0]])
        else:
            protein = "".join(aas)
        results[seq] = protein

    return results


# Suppress the very verbose standard out of some landscape's predictors
@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class DummyTFBinding:
    def __init__(self, **kwargs):
        pass

    def _fitness_function(self, xs):
        # Simple scoring rule: Gs and Cs are good, As and Ts are bad.
        scores = []
        for seq_list in xs:  # flexs often wraps inputs in an extra list
            for seq in seq_list:
                score = seq.count("G") * 0.2
                score += seq.count("C") * 0.15
                score -= seq.count("A") * 0.1
                score -= seq.count("T") * 0.1
                scores.append(score)
        return np.array(scores) + np.random.normal(0, 0.05, len(scores))


class TFBindingLandscape:
    """
    A Landscape class that returns raw fitness scores for any batch of sequences.
    This acts as the environment for the policy gradient agent.
    """

    def __init__(self, str_length, objectives):
        tf_problems = flexs.landscapes.tf_binding.registry()
        for assay_name in objectives:
            if assay_name not in tf_problems:
                raise ValueError(
                    f"Assay name '{assay_name}' not found in the registry."
                )

        self.str_length = str_length
        self.alphabet = DNA_BASES
        self.active_objs = objectives
        self.problem_info = tf_problems[assay_name]
        self.landscapes = {
            assay: flexs.landscapes.TFBinding(**tf_problems[assay]["params"])
            for assay in self.active_objs
        }
        self.is_peptide = False

    @staticmethod
    def list_objectives():
        return list(flexs.landscapes.tf_binding.registry().keys())

    def get_all_objectives(self, xs: list):
        return np.stack(
            [landscape._fitness_function(xs) for landscape in self.landscapes.values()],
            axis=-1,
        )

    def get_fitness_values(self, xs: list, objective=None):
        """Returns fitness values for a list of sequences."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]
        # The landscape model might expect a list of lists.
        return self.landscapes[objective]._fitness_function(xs)

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Args:
            xs: List of sequences.

        Returns:
            Tuple of (sequences, {objective: fitness_values}).
        """
        if not isinstance(xs, list):
            raise TypeError("xs must be a list")

        fitness_values = {}
        for obj in self.active_objs:
            fitness_values[obj] = list(self.get_fitness_values(xs, objective=obj))

        return xs, fitness_values


class MHCflurryLandscape:
    """
    A landscape that wraps MHCflurry and returns scores for peptide sequences.
    """

    def __init__(
        self,
        str_length,
        objectives,
        min_peptide_len=None,
        max_peptide_len=None,
        invalid_scores: dict[str, float] = None,
    ):
        if not MHCFLURRY_AVAILABLE:
            raise ImportError(
                "MHCflurry is not available. Please use the MHC-specific conda environment to run this landscape."
            )
        self.str_length = str_length
        self.alphabet = (
            DNA_BASES  # Uses the DNA alphabet and the scorer translates to protein
        )
        self.alleles = [a for a in objectives if a != "Processing"]
        self.include_processing = "Processing" in objectives
        self.active_objs = objectives
        self.affinity_predictor = Class1AffinityPredictor.load()
        if self.include_processing:
            self.processing_predictor = Class1ProcessingPredictor.load()
        else:
            self.processing_predictor = None

        # Set peptide length constraints - default to expected length from DNA
        expected_peptide_len = str_length // 3
        self.min_peptide_len = (
            min_peptide_len if min_peptide_len is not None else expected_peptide_len
        )
        self.max_peptide_len = (
            max_peptide_len if max_peptide_len is not None else expected_peptide_len
        )
        self.is_peptide = True
        self.invalid_scores = invalid_scores or {}

    def list_objectives():
        if not MHCFLURRY_AVAILABLE:
            raise ImportError(
                "MHCflurry is not available. Please use the MHC-specific conda environment."
            )
        return Class1AffinityPredictor.load().supported_alleles + ["Processing"]

    def get_all_objectives(self, xs: list, percentiles=True) -> np.ndarray:
        """Returns shape (len(xs), n_objectives)."""
        return np.stack(
            [
                self.get_fitness_values(xs, percentiles=percentiles, objective=obj)
                for obj in self.active_objs
            ],
            axis=-1,
        )

    def _predict_from_peptides(self, peptides: list, percentiles=True, objective=None):
        """
        Internal method that predicts fitness values directly from peptide sequences.
        Assumes peptides are already validated (correct length, valid amino acids).

        Args:
            peptides: List of amino acid sequences
            percentiles: If True, return percentile ranks; if False, return raw predictions
            objective: Which objective to predict for

        Returns:
            numpy array of fitness values for each peptide
        """
        if len(self.active_objs) == 1:
            objective = self.active_objs[0]

        if not peptides:
            return np.zeros((0,))

        with suppress_stdout():
            if objective in self.alleles:
                df = self.affinity_predictor.predict_to_dataframe(
                    peptides, allele=objective
                )
                if percentiles:
                    return 1 - df.prediction_percentile.values / 100
                else:
                    return df.prediction.values
            elif self.include_processing:
                return self.processing_predictor.predict(peptides)
            else:
                raise ValueError(f"Invalid objective {objective}")

    def get_fitness_values(self, xs: list, percentiles=True, objective=None):
        """Returns fitness values for a list of DNA sequences."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]

        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        scores = np.full((len(xs),), invalid_score, dtype=np.float32)
        valid_idx, valid_peptides = sequences_to_valid_peptides(
            xs, min_len=self.min_peptide_len, max_len=self.max_peptide_len
        )
        if valid_peptides:
            fitness_values = self._predict_from_peptides(
                valid_peptides, percentiles=percentiles, objective=objective
            )
            scores[valid_idx] = fitness_values
        return scores

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Returns only valid peptides (ie min_length <= len(peptide) <= max_length)
        and their fitness values. Uses -log_10(affinity) as fitness for BO.

        Args:
            xs: List of nucleotide sequences.

        Returns:
            Tuple of (valid_peptides, {objective: fitness_values}).
        """
        valid_idx, valid_peptides = sequences_to_valid_peptides(
            xs, self.min_peptide_len, self.max_peptide_len
        )
        if not valid_peptides:
            return [], {obj: [] for obj in self.active_objs}

        fitness_values = {}
        for obj in self.active_objs:
            values = self._predict_from_peptides(
                valid_peptides, percentiles=False, objective=obj
            )
            values = -np.log10(values)
            fitness_values[obj] = list(values)

        return valid_peptides, fitness_values


class RiboswitchLandscape:
    """
    A landscape that uses oracles to predict a switch sequence's ability to transform an aptamer into an ATP-activated riboswitch.
    """

    def __init__(self, str_length, objectives, device, oracle_batch_size, oracle_dir):
        try:
            from grelu.lightning import LightningModel
        except ImportError:
            raise ImportError(
                "gReLU v1.0.2 is not installed. Please install to use riboswitch landscape."
            )

        if str_length != 10:
            raise ValueError(f"Sequence length has to be 10")

        self.str_length = str_length
        self.oracle_batch_size = oracle_batch_size

        if np.any([o not in RiboswitchLandscape.list_objectives() for o in objectives]):
            raise ValueError(
                f"Objectives have to be in {RiboswitchLandscape.list_objectives()}"
            )
        # always use the same order
        self.active_objs = objectives

        # for standardization
        self.obj_mean = np.zeros(len(self.active_objs))
        self.obj_std = np.ones(len(self.active_objs))

        self.alphabet = DNA_BASES
        self.device = device  # torch device for oracles

        self.atp_oracle = LightningModel.load_from_checkpoint(
            os.path.join(oracle_dir, f"atp_oracle.ckpt"), map_location=self.device
        )
        self.atp_oracle.eval()

        self.is_peptide = False

    @staticmethod
    def list_objectives():
        return ["ATP"]

    def prepare_seqs(self, xs):
        # tokenize and one-hot encode sequences
        xs = tokenize_seqs(xs)
        xs = torch.tensor(xs).long().to(self.device)
        xs = F.one_hot(xs, num_classes=len(self.alphabet)).float().transpose(1, 2)
        if self.oracle_batch_size >= len(xs):
            return (xs,)
        # split into batches
        return torch.split(xs, self.oracle_batch_size, dim=0)

    def get_fitness_values(self, xs: list, objective=None):
        """
        Slow access to individual fitness objectives.
        Use get_all_objectives when querying multiple objectives.
        """
        if len(self.active_objs) == 1:
            # support single-objective training
            objective = self.active_objs[0]

        if objective == "ATP":
            return self.predict_atp(xs)
        else:
            raise ValueError(f"Unknown objective {objective}")

    def get_all_objectives(self, xs: list, normalize=True):
        """
        Get the values of all fitness objectives

        Returns:
            preds (np.ndarray): Values of all fitness objectives. Shape (len(xs), n_objectives)
        """
        preds = np.expand_dims(self.predict_atp(xs), axis=1)
        if normalize:
            preds = (preds - self.obj_mean) / self.obj_std
        return preds

    def predict_atp(self, xs):
        """
        Get raw predictions from the ATP oracle
        """
        xs_batched = self.prepare_seqs(xs)
        results = []
        with torch.no_grad():
            for batch in xs_batched:
                results.append(
                    self.atp_oracle(batch.to(self.device)).cpu().numpy().reshape(-1)
                )
        return np.concatenate(results, axis=0)

    def compute_normalizer(self, xs: list):
        """
        Set mean and stdev used for normalization based on xs
        """
        values = self.get_all_objectives(xs, normalize=False)
        self.obj_mean = values.mean(axis=0)
        self.obj_std = values.std(axis=0)

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Applies log10(1 + score) transformation to reduce skewness for oracle training.

        Args:
            xs: List of sequences.

        Returns:
            Tuple of (sequences, {objective: fitness_values}).
        """
        if not isinstance(xs, list):
            raise TypeError("xs must be a list")

        fitness_values = {}
        for obj in self.active_objs:
            raw_scores = np.array(self.get_fitness_values(xs, objective=obj))
            fitness_values[obj] = list(np.log10(1 + raw_scores))

        return xs, fitness_values


class BoLandscape:
    """
    A unified landscape that is instantiated in BO runs.
    This landscape uses doesn't return true fitness values, but rather oracle predictions of fitness values based on data observed in previous runs.
    """

    def __init__(
        self,
        str_length: int,
        ensembles: dict[str, list[torch.nn.Module]],
        with_replacement: bool = True,
        device: str = "cuda",
        objectives: Optional[list[str]] = None,
        min_peptide_len: Optional[int] = None,
        max_peptide_len: Optional[int] = None,
        oracle_batch_size: int = 1024,
        is_peptide: bool = False,
        invalid_scores: dict[str, float] = None,
    ) -> None:
        self.oracle_batch_size = oracle_batch_size
        self.ensembles = ensembles

        self.str_length = str_length
        self.alphabet = DNA_BASES
        self.active_objs = objectives or list(ensembles.keys())
        self.device = device

        # Set peptide length constraints - default to expected length from DNA
        expected_peptide_len = str_length // 3
        self.min_peptide_len = (
            min_peptide_len if min_peptide_len is not None else expected_peptide_len
        )
        self.max_peptide_len = (
            max_peptide_len if max_peptide_len is not None else expected_peptide_len
        )

        self.is_peptide = is_peptide

        self._observed_set = set()
        self.with_replacement = with_replacement
        self.invalid_scores = invalid_scores or {}

    @staticmethod
    def list_objectives() -> list[str]:
        return []

    def oh_encode(self, sequences: list[str]) -> torch.Tensor:
        """One-hot encode sequences directly on device."""
        if self.is_peptide:
            seq_length = self.str_length // 3
            return batch_one_hot_encode_peptide(sequences, seq_length, self.device)
        else:
            seq_length = self.str_length
            return batch_one_hot_encode_dna(sequences, seq_length, self.device)

    def update_observed_sequences(self, sequences: list[str]) -> None:
        self._observed_set.update(sequences)

    def filter_and_transform(self, sequences: list[str]) -> tuple[list[int], list[str]]:
        """
        Filter invalid/duplicate sequences and return indices mapping to original input.

        Filters: invalid peptide lengths (if is_peptide), already-observed sequences.
        """
        if self.is_peptide:
            valid_idx, transformed = sequences_to_valid_peptides(
                sequences, self.min_peptide_len, self.max_peptide_len
            )
        else:
            valid_idx = list(range(len(sequences)))
            transformed = list(sequences)

        if not self.with_replacement:
            filtered_idx = []
            filtered_seqs = []
            for i, seq in zip(valid_idx, transformed):
                if seq not in self._observed_set:
                    filtered_idx.append(i)
                    filtered_seqs.append(seq)
            valid_idx, transformed = filtered_idx, filtered_seqs

        return valid_idx, transformed

    def get_ensemble_predictions(
        self, one_hot_sequences: torch.Tensor, objective: str
    ) -> torch.Tensor:
        """Get predictions from ensemble for a specific objective."""
        ensemble = self.ensembles[objective]
        num_sequences = one_hot_sequences.shape[0]

        with torch.no_grad():
            predictions = torch.zeros(
                (num_sequences, len(ensemble)),
                dtype=torch.float32,
                device=self.device,
            )

            for i, model in enumerate(ensemble):
                model.eval()
                outputs = model.predict(one_hot_sequences).squeeze(-1)
                predictions[:, i] = outputs

        return predictions

    def get_fitness_values(
        self,
        xs: list[str],
        without_filtering: bool = False,
        objective: Optional[str] = None,
    ) -> np.ndarray:
        """Returns raw enrichment predictions for a specific objective."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]

        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        scores = torch.full(
            (len(xs),), invalid_score, dtype=torch.float32, device=self.device
        )

        if without_filtering:
            valid_sequences = xs
            valid_idx = np.arange(len(valid_sequences))
        else:
            valid_idx, valid_sequences = self.filter_and_transform(xs)

        if not valid_sequences:
            return scores.cpu().numpy()

        one_hot_all = self.oh_encode(valid_sequences)

        with torch.no_grad():
            preds = self.get_ensemble_predictions(one_hot_all, objective).mean(dim=1)

        scores[valid_idx] = preds.clip(min=invalid_score)
        return scores.cpu().numpy()

    def get_all_objectives(self, xs: list[str]) -> np.ndarray:
        """Returns shape (len(xs), n_objectives)."""
        return np.stack(
            [self.get_fitness_values(xs, objective=obj) for obj in self.active_objs],
            axis=-1,
        )


class PLMLogLikelihoodLandscape:
    """
    A landscape that computes the (pseudo-)loglikelihood under protein language models (PLMs).
    Includes general and antibody-specific PLMs.

    Multi-GPU Support (ESM2 only):
        When n_gpus > 1, ESM2 parallelizes across masked positions (24 positions / n_gpus).
        Each GPU handles a subset of positions, providing near-linear speedup.
    """

    def __init__(
        self,
        str_length,
        objectives,
        device,
        min_peptide_len,
        max_peptide_len,
        models=None,
        oracle_batch_size_per_gpu: int = 1024,
        wild_type: str | None = None,
        max_edit_distance: int | None = None,
    ):
        try:
            from plm import (
                AbLang2Wrapper,
                FastESM2Wrapper,
                IgLMWrapper,
                LinearLogLH,
            )
        except ImportError:
            raise ImportError(
                f"Couldn't import PLMs from plm. Make sure all necessary dependencies are installed."
            )

        self.str_length = str_length
        self.min_peptide_len = min_peptide_len
        self.max_peptide_len = max_peptide_len
        self._oracle_batch_size_per_gpu = (
            oracle_batch_size_per_gpu  # multiplied by n_gpus below
        )
        self.active_objs = objectives
        self.alphabet = ["A", "C", "G", "T"]
        self.is_peptide = True
        self.wild_type = wild_type
        self.max_edit_distance = max_edit_distance
        if max_edit_distance is not None and wild_type is None:
            raise ValueError("wild_type must be provided when max_edit_distance is set")
        self.wild_type_indices = (
            torch.tensor(
                [AMINO_ACID_ALPHABET[aa] for aa in wild_type], dtype=torch.long
            )
            if wild_type
            else None
        )

        if not (str_length // 3):
            raise ValueError(f"str_length has to be divisible by 3")

        self.name_to_class = {
            "AbLang2": AbLang2Wrapper,
            "ESM2": FastESM2Wrapper,
            "IgLM": IgLMWrapper,
            "IgLM-S": IgLMWrapper,
            "Linear": LinearLogLH,
        }

        self.models = {}
        self.right_context = {}
        self.left_context = {}
        self.invalid_sequence_scores = {}
        for obj_name in objectives:
            if obj_name not in models:
                raise ValueError(
                    f"Objective '{obj_name}' not found in models configuration"
                )
            self.right_context[obj_name] = models[obj_name].get("right_context", "")
            self.left_context[obj_name] = models[obj_name].get("left_context", "")
            generated_pos = np.arange(str_length // 3) + len(
                self.left_context[obj_name]
            )

            model_args = OmegaConf.to_container(
                models[obj_name].get("args", {}), resolve=True
            )
            model_args["unmasked_pos"] = np.array(
                [
                    i
                    for i in range(
                        self.str_length // 3
                        + len(self.left_context[obj_name])
                        + len(self.right_context[obj_name])
                    )
                    if i not in generated_pos
                ]
            )

            if not model_args.get("device"):
                model_args["device"] = device

            if obj_name == "Linear":
                model_args["str_len"] = (
                    self.str_length // 3
                    + len(self.left_context[obj_name])
                    + len(self.right_context[obj_name])
                )
                model_args["alphabet"] = AMINO_ACIDS[:-1]

            model = self.name_to_class[obj_name](**model_args)
            self.models[obj_name] = model
            self.invalid_sequence_scores[obj_name] = models[obj_name].get(
                "invalid_sequence_score", -100
            )

        # Multi-GPU: use all available GPUs, scale batch size accordingly
        n_gpus = max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        self.n_gpus = n_gpus
        self.oracle_batch_size = self._oracle_batch_size_per_gpu * n_gpus

        # Configure ESM2 for multi-GPU
        for obj_name in objectives:
            if obj_name == "ESM2" and n_gpus > 1:
                model = self.models[obj_name]
                if hasattr(model, "set_n_gpus"):
                    model.set_n_gpus(n_gpus)

    @staticmethod
    def list_objectives():
        return ["AbLang2", "ESM2", "IgLM", "IgLM-S", "Linear"]

    def add_context(self, xs: list, objective: str):
        return [
            self.left_context[objective] + x + self.right_context[objective] for x in xs
        ]

    def _add_context_tensor(
        self, peptide_indices: torch.Tensor, left_ctx: str, right_ctx: str
    ) -> torch.Tensor:
        """Prepend/append context AA indices to peptide tensor."""
        device = peptide_indices.device
        U = peptide_indices.shape[0]

        parts = []
        if left_ctx:
            left_idx = (
                torch.tensor(
                    [AMINO_ACID_ALPHABET[aa] for aa in left_ctx],
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .expand(U, -1)
            )
            parts.append(left_idx)

        parts.append(peptide_indices)

        if right_ctx:
            right_idx = (
                torch.tensor(
                    [AMINO_ACID_ALPHABET[aa] for aa in right_ctx],
                    dtype=torch.long,
                    device=device,
                )
                .unsqueeze(0)
                .expand(U, -1)
            )
            parts.append(right_idx)

        return torch.cat(parts, dim=1)

    def get_fitness_values(self, xs: list, objective=None):
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]

        if objective is None:
            raise ValueError(
                "objective parameter is required for multi-objective landscapes"
            )

        invalid_score = self.invalid_sequence_scores[objective]
        scores = torch.full((len(xs),), invalid_score, dtype=torch.float32)
        valid_idx, valid_peptides = sequences_to_valid_peptides(
            xs, min_len=self.min_peptide_len, max_len=self.max_peptide_len
        )

        if len(valid_peptides) == 0:
            return scores.numpy()

        model = self.models[objective]
        valid_peptides_with_ctx = self.add_context(valid_peptides, objective)

        if hasattr(model, "score_tensor_batch"):
            # IgLM/ESM2 - use score_tensor_batch to respect use_infill
            peptide_indices = torch.tensor(
                [
                    [AMINO_ACID_ALPHABET[aa] for aa in pep]
                    for pep in valid_peptides_with_ctx
                ],
                dtype=torch.long,
                device=model.device,
            )
            # Batch to avoid OOM (same pattern as landscape.score_tensor_batch)
            n_valid = peptide_indices.shape[0]
            with torch.no_grad():
                if n_valid <= self.oracle_batch_size:
                    preds = model.score_tensor_batch(peptide_indices).cpu()
                else:
                    all_preds = []
                    for i in range(0, n_valid, self.oracle_batch_size):
                        batch = peptide_indices[i : i + self.oracle_batch_size]
                        all_preds.append(model.score_tensor_batch(batch).cpu())
                    preds = torch.cat(all_preds, dim=0)
        else:
            xs_enc = model.prepare(valid_peptides_with_ctx)
            with torch.no_grad():
                preds = model.get_log_probs(xs_enc).cpu()

        scores[valid_idx] = preds.clip(min=invalid_score)
        return scores.numpy()

    def get_all_objectives(self, xs: list):
        """Returns shape (len(xs), n_objectives)."""
        return np.stack(
            [self.get_fitness_values(xs, objective=obj) for obj in self.active_objs],
            axis=-1,
        )

    def score_tensor_batch(
        self, peptide_indices: torch.Tensor, objective: str = None
    ) -> torch.Tensor:
        """
        Score peptides given as tensor of AA indices (vectorized, for fast training).

        Args:
            peptide_indices: (U, L) int tensor, values 0-19 for AAs, -1 for masked/invalid
            objective: Objective name (required)

        Returns:
            (U,) float tensor of log-likelihood scores
        """
        if objective is None:
            raise ValueError(
                "objective parameter required for multi-objective landscapes"
            )

        invalid_score = self.invalid_sequence_scores[objective]
        device = peptide_indices.device

        # Count valid positions (non -1) to get peptide lengths
        peptide_lengths = (peptide_indices >= 0).sum(dim=1)

        # Check length constraints
        valid_length = (peptide_lengths >= self.min_peptide_len) & (
            peptide_lengths <= self.max_peptide_len
        )

        # Check edit distance constraint
        if self.max_edit_distance is not None:
            wt = self.wild_type_indices.to(device)
            valid_mask = peptide_indices >= 0  # (U, L)
            matches = (peptide_indices == wt.unsqueeze(0)) & valid_mask  # (U, L)
            edit_distances = valid_mask.sum(dim=1) - matches.sum(dim=1)  # (U,)
            valid_length = valid_length & (edit_distances <= self.max_edit_distance)

        # Initialize all scores to invalid
        scores = torch.full(
            (peptide_indices.shape[0],),
            invalid_score,
            dtype=torch.float32,
            device=device,
        )

        # Only score peptides with valid lengths
        if valid_length.any():
            model = self.models[objective]
            if not hasattr(model, "score_tensor_batch"):
                raise NotImplementedError(
                    f"Model {objective} does not support score_tensor_batch"
                )

            valid_peptides = peptide_indices[valid_length]

            # Add context if present
            left_ctx = self.left_context.get(objective, "")
            right_ctx = self.right_context.get(objective, "")
            if left_ctx or right_ctx:
                valid_peptides = self._add_context_tensor(
                    valid_peptides, left_ctx, right_ctx
                )

            # Score peptides (any multi-GPU parallelism handled inside ESM2 model, other landscapes only need single GPU)
            n_valid = valid_peptides.shape[0]
            if n_valid <= self.oracle_batch_size:
                valid_scores = model.score_tensor_batch(valid_peptides)
            else:
                all_scores = []
                for i in range(0, n_valid, self.oracle_batch_size):
                    batch = valid_peptides[i : i + self.oracle_batch_size]
                    all_scores.append(model.score_tensor_batch(batch))
                valid_scores = torch.cat(all_scores, dim=0)

            # Don't let the landscape score something lower than an invalid sequence
            scores[valid_length] = valid_scores.clamp(min=invalid_score)

        return scores


def _is_valid_dna_seq(seq: str) -> bool:
    """Check if sequence contains only valid DNA bases."""
    return all(c in DNA_BASES for c in seq.upper())


def _annotate_peptide_sequences(
    xs: list[str],
    fitness_lookup: dict[str, float],
    min_peptide_len: int,
    max_peptide_len: int,
    missing_score: float,
) -> tuple[list[str], list[float]]:
    """
    Translate DNA sequences to peptides and look up fitness scores.

    Args:
        xs: DNA sequences to annotate.
        fitness_lookup: Peptide -> fitness mapping.
        min_peptide_len: Minimum valid peptide length.
        max_peptide_len: Maximum valid peptide length.
        missing_score: Score for valid peptides missing from lookup.

    Returns:
        (valid_peptides, scores) for valid sequences only.
    """
    valid_dna = [seq for seq in xs if _is_valid_dna_seq(seq)]

    if not valid_dna:
        return [], []

    valid_idx, valid_peptides = sequences_to_valid_peptides(
        valid_dna, min_len=min_peptide_len, max_len=max_peptide_len
    )

    if not valid_peptides:
        return [], []

    scores = [fitness_lookup.get(p, missing_score) for p in valid_peptides]
    return valid_peptides, scores


def _peptide_lookup_fitness(
    xs: list[str],
    fitness_lookup: dict[str, float],
    invalid_score: float,
    min_peptide_len: int,
    max_peptide_len: int,
    missing_score: float | None = None,
) -> torch.Tensor:
    """
    Score DNA sequences by translating to peptides and looking up fitness.

    Args:
        xs: DNA sequences to score
        fitness_lookup: Peptide -> fitness mapping
        invalid_score: Score for invalid sequences (non-DNA chars, stop codons, wrong length)
        min_peptide_len: Minimum valid peptide length
        max_peptide_len: Maximum valid peptide length
        missing_score: Score for valid peptides missing from lookup. If None, uses invalid_score.
    """
    if missing_score is None:
        missing_score = invalid_score
    scores = torch.full((len(xs),), invalid_score, dtype=torch.float32)

    valid_dna_mask = [_is_valid_dna_seq(seq) for seq in xs]
    valid_dna_seqs = [seq for seq, valid in zip(xs, valid_dna_mask) if valid]
    valid_dna_indices = [i for i, valid in enumerate(valid_dna_mask) if valid]

    if not valid_dna_seqs:
        return scores

    valid_idx, valid_peptides = sequences_to_valid_peptides(
        valid_dna_seqs, min_len=min_peptide_len, max_len=max_peptide_len
    )

    if valid_peptides:
        # valid_idx is relative to valid_dna_seqs, map back to original xs indices
        original_indices = [valid_dna_indices[i] for i in valid_idx]
        scores[original_indices] = torch.tensor(
            [fitness_lookup.get(p, missing_score) for p in valid_peptides],
            dtype=torch.float32,
        )
    return scores


class GB1FitnessLandscape:
    """
    Lookup-based landscape using GB1 exhaustive SSM fitness data.
    Based on data from Wu et al. 2016.
    Accepts DNA input (12 bases = 4 codons), translates to 4 amino acid peptides.
    No missingness - all 160,000 variants present.
    """

    def __init__(
        self,
        str_length: int,
        objectives: list[str],
        data_path: str,
        min_peptide_len: int = 4,
        max_peptide_len: int = 4,
        wild_type: str | None = None,
        invalid_scores: dict[str, float] | None = None,
    ):
        if str_length != 12:
            raise ValueError(
                f"GB1 landscape expects str_length=12 (4 codons), got {str_length}"
            )

        self.str_length = str_length
        self.alphabet = DNA_BASES
        self.active_objs = objectives
        self.min_peptide_len = min_peptide_len
        self.max_peptide_len = max_peptide_len
        self.wild_type = wild_type
        self.is_peptide = True
        self.invalid_scores = invalid_scores or {}

        self.fitness_lookup = self._load_fitness_data(data_path)
        log.info(f"Loaded {len(self.fitness_lookup)} GB1 variants from {data_path}")

    def _load_fitness_data(self, data_path: str) -> dict[str, float]:
        """Load fitness lookup table from CSV file."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"GB1 fitness data not found at {data_path}. "
                "Run pgld/scripts/explore_gb1_data.py to generate it."
            )
        df = pd.read_csv(data_path)
        return dict(zip(df["variant"], df["fitness"]))

    @staticmethod
    def list_objectives() -> list[str]:
        return ["GB1_fitness"]

    def get_fitness_values(
        self, xs: list[str], objective: str | None = None
    ) -> torch.Tensor:
        """Score DNA sequences by translating to peptides and looking up fitness."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]
        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        return _peptide_lookup_fitness(
            xs,
            self.fitness_lookup,
            invalid_score,
            self.min_peptide_len,
            self.max_peptide_len,
        )

    def get_all_objectives(self, xs: list[str]) -> np.ndarray:
        """Get fitness values for all objectives (shape: len(xs), n_objectives)."""
        return self.get_fitness_values(xs).numpy().reshape(-1, 1)

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Args:
            xs: List of DNA sequences.

        Returns:
            Tuple of (valid_peptides, {objective: fitness_values}).
        """
        objective = self.active_objs[0]
        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        valid_peptides, scores = _annotate_peptide_sequences(
            xs,
            self.fitness_lookup,
            self.min_peptide_len,
            self.max_peptide_len,
            invalid_score,
        )
        transformed = [math.log10(s + 1) for s in scores]
        return valid_peptides, {objective: transformed}


class PhoQFitnessLandscape:
    """
    Lookup-based landscape using PhoQ kinase exhaustive SSM fitness data.
    Based on data from Podgornaia & Laub 2015.
    Accepts DNA input (12 bases = 4 codons), translates to 4 amino acid peptides.
    Has missingness - missing variants imputed with min observed fitness.
    """

    def __init__(
        self,
        str_length: int,
        objectives: list[str],
        data_path: str,
        min_peptide_len: int = 4,
        max_peptide_len: int = 4,
        wild_type: str | None = None,
        invalid_scores: dict[str, float] | None = None,
    ):
        if str_length != 12:
            raise ValueError(
                f"PhoQ landscape expects str_length=12 (4 codons), got {str_length}"
            )

        self.str_length = str_length
        self.alphabet = DNA_BASES
        self.active_objs = objectives
        self.min_peptide_len = min_peptide_len
        self.max_peptide_len = max_peptide_len
        self.wild_type = wild_type
        self.is_peptide = True
        self.invalid_scores = invalid_scores or {}

        self.fitness_lookup, self.min_score = self._load_fitness_data(data_path)
        log.info(
            f"Loaded {len(self.fitness_lookup)} PhoQ variants from {data_path}, "
            f"min_score={self.min_score:.4f}"
        )

    def _load_fitness_data(self, data_path: str) -> tuple[dict[str, float], float]:
        """Load fitness lookup table from xlsx file and compute min score."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"PhoQ fitness data not found at {data_path}.")
        df = pd.read_excel(data_path)
        min_score = df["Fitness"].min()
        return dict(zip(df["Variants"], df["Fitness"])), min_score

    @staticmethod
    def list_objectives() -> list[str]:
        return ["PhoQ_fitness"]

    def get_fitness_values(
        self, xs: list[str], objective: str | None = None
    ) -> torch.Tensor:
        """Score DNA sequences by translating to peptides and looking up fitness."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]
        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        return _peptide_lookup_fitness(
            xs,
            self.fitness_lookup,
            invalid_score,
            self.min_peptide_len,
            self.max_peptide_len,
            missing_score=self.min_score,
        )

    def get_all_objectives(self, xs: list[str]) -> np.ndarray:
        """Get fitness values for all objectives (shape: len(xs), n_objectives)."""
        return self.get_fitness_values(xs).numpy().reshape(-1, 1)

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Args:
            xs: List of DNA sequences.

        Returns:
            Tuple of (valid_peptides, {objective: fitness_values}).
        """
        valid_peptides, scores = _annotate_peptide_sequences(
            xs,
            self.fitness_lookup,
            self.min_peptide_len,
            self.max_peptide_len,
            self.min_score,
        )
        transformed = [math.log10(s + 1) for s in scores]
        return valid_peptides, {self.active_objs[0]: transformed}


class DHFRFitnessLandscape:
    """
    Lookup-based landscape using DHFR folate metabolism fitness data.
    Based on data from Papkou et al. 2023.
    Direct DNA landscape (9 nucleotides), no translation.
    Has missingness - missing variants imputed with min observed fitness.
    """

    def __init__(
        self,
        str_length: int,
        objectives: list[str],
        data_path: str,
        wild_type: str | None = None,
        invalid_scores: dict[str, float] | None = None,
    ):
        if str_length != 9:
            raise ValueError(f"DHFR landscape expects str_length=9, got {str_length}")

        self.str_length = str_length
        self.alphabet = DNA_BASES
        self.active_objs = objectives
        self.wild_type = wild_type
        self.is_peptide = False
        self.invalid_scores = invalid_scores or {}

        self.fitness_lookup, self.min_score = self._load_fitness_data(data_path)
        log.info(
            f"Loaded {len(self.fitness_lookup)} DHFR variants from {data_path}, "
            f"min_score={self.min_score:.4f}"
        )

    def _load_fitness_data(self, data_path: str) -> tuple[dict[str, float], float]:
        """Load fitness lookup table from CSV file and compute min score."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"DHFR fitness data not found at {data_path}.")
        df = pd.read_csv(data_path)
        df = df[df["SV"].notna()]
        df = df[df["SV"].str.len() == 9]
        min_score = df["fitness"].min()
        return dict(zip(df["SV"], df["fitness"])), min_score

    @staticmethod
    def list_objectives() -> list[str]:
        return ["DHFR_fitness"]

    def get_fitness_values(
        self, xs: list[str], objective: str | None = None
    ) -> torch.Tensor:
        """Score DNA sequences directly via lookup."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]
        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        scores = []
        for seq in xs:
            seq_upper = seq.upper()
            if len(seq_upper) != self.str_length or not _is_valid_dna_seq(seq_upper):
                scores.append(invalid_score)
            else:
                scores.append(self.fitness_lookup.get(seq_upper, self.min_score))
        return torch.tensor(scores, dtype=torch.float32)

    def get_all_objectives(self, xs: list[str]) -> np.ndarray:
        """Get fitness values for all objectives (shape: len(xs), n_objectives)."""
        return self.get_fitness_values(xs).numpy().reshape(-1, 1)

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Args:
            xs: List of DNA sequences.

        Returns:
            Tuple of (valid_sequences, {objective: fitness_values}).
        """
        valid_seqs = []
        for seq in xs:
            seq_upper = seq.upper()
            if len(seq_upper) == self.str_length and _is_valid_dna_seq(seq_upper):
                valid_seqs.append(seq_upper)

        if not valid_seqs:
            return [], {obj: [] for obj in self.active_objs}

        fitness_values = {}
        for obj in self.active_objs:
            scores = [
                self.fitness_lookup.get(seq, self.min_score) for seq in valid_seqs
            ]
            fitness_values[obj] = [math.log10(s + 1) for s in scores]
        return valid_seqs, fitness_values


class TrpBFitnessLandscape:
    """
    Lookup-based landscape using TrpB tryptophan synthase fitness data.
    Based on data from Johnston et al. 2024.
    Accepts DNA input (12 bases = 4 codons), translates to 4 amino acid peptides.
    Has missingness - missing variants imputed with min observed fitness.
    """

    def __init__(
        self,
        str_length: int,
        objectives: list[str],
        data_path: str,
        min_peptide_len: int = 4,
        max_peptide_len: int = 4,
        wild_type: str | None = None,
        invalid_scores: dict[str, float] | None = None,
    ):
        if str_length != 12:
            raise ValueError(
                f"TrpB landscape expects str_length=12 (4 codons), got {str_length}"
            )

        self.str_length = str_length
        self.alphabet = DNA_BASES
        self.active_objs = objectives
        self.min_peptide_len = min_peptide_len
        self.max_peptide_len = max_peptide_len
        self.wild_type = wild_type
        self.is_peptide = True
        self.invalid_scores = invalid_scores or {}

        self.fitness_lookup, self.min_score = self._load_fitness_data(data_path)
        log.info(
            f"Loaded {len(self.fitness_lookup)} TrpB variants from {data_path}, "
            f"min_score={self.min_score:.4f}"
        )

    def _load_fitness_data(self, data_path: str) -> tuple[dict[str, float], float]:
        """Load fitness lookup table from CSV file and compute min score."""
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"TrpB fitness data not found at {data_path}.")
        df = pd.read_csv(data_path)
        # Filter out sequences with stop codons (contain '*')
        df = df[~df["AAs"].str.contains(r"\*", regex=True)]
        min_score = df["fitness"].min()
        return dict(zip(df["AAs"], df["fitness"])), min_score

    @staticmethod
    def list_objectives() -> list[str]:
        return ["TrpB_fitness"]

    def get_fitness_values(
        self, xs: list[str], objective: str | None = None
    ) -> torch.Tensor:
        """Score DNA sequences by translating to peptides and looking up fitness."""
        if objective is None and len(self.active_objs) == 1:
            objective = self.active_objs[0]
        invalid_score = self.invalid_scores.get(objective, float("-inf"))
        return _peptide_lookup_fitness(
            xs,
            self.fitness_lookup,
            invalid_score,
            self.min_peptide_len,
            self.max_peptide_len,
            missing_score=self.min_score,
        )

    def get_all_objectives(self, xs: list[str]) -> np.ndarray:
        """Get fitness values for all objectives (shape: len(xs), n_objectives)."""
        return self.get_fitness_values(xs).numpy().reshape(-1, 1)

    def annotate_sequences(
        self, xs: list[str]
    ) -> tuple[list[str], dict[str, list[float]]]:
        """
        Annotate DNA sequences for all active objectives.

        Args:
            xs: List of DNA sequences.

        Returns:
            Tuple of (valid_peptides, {objective: fitness_values}).
        """
        valid_peptides, scores = _annotate_peptide_sequences(
            xs,
            self.fitness_lookup,
            self.min_peptide_len,
            self.max_peptide_len,
            self.min_score,
        )
        return valid_peptides, {self.active_objs[0]: scores}


def hit_above_threshold(
    landscape, xs: list, threshold=0.5, objective=None, uniqueness_type="dna", **kwargs
):
    """
    Calculates the number of unique sequences that score above a given threshold.
    This serves as the reward function for the policy gradient algorithm.
    """
    if not xs:
        return 0
    # Score only unique sequences to save computation and get a true count
    unique_xs = list(set(xs))
    if landscape.is_peptide and uniqueness_type == "peptide":
        unique_xs = filter_same_protein_encoding(unique_xs)
    fitness_values = landscape.get_fitness_values(unique_xs, objective=objective)
    # Count how many unique sequences exceed the threshold
    above_threshold = [
        x for x, val in zip(unique_xs, fitness_values) if val > threshold
    ]
    return len(above_threshold)


def hit_above_threshold_weighted(landscape, xs: list, threshold: list, weights):
    """
    Calculates number of unique sequences in the batch above a threshold for each objective.
    Aggregates the number of hits over objectives as a weighted sum.
    """
    assert len(threshold) == len(weights)
    if not is_landscape_multi_objective(landscape):
        return hit_above_threshold(landscape, xs, threshold[0]) * weights[0]
    else:
        unique_xs = list(np.unique(xs))
        values = landscape.get_all_objectives(
            unique_xs
        )  # (n_unique_seqs, n_objectives)
        hits = (values > np.array(threshold)).astype(float)
        return np.sum(hits * weights)


def sum_weighted(landscape, xs: list, weights):
    """
    Calculate weighted sum across objectives, summed over all sequences in batch.
    Returns: scalar = sum_over_sequences(sum_over_objectives(fitness[seq, obj] * weight[obj]))
    """
    assert len(weights) == len(landscape.active_objs)
    if not is_landscape_multi_objective(landscape):
        vals = landscape.get_fitness_values(xs)
        return np.sum(landscape.get_fitness_values(xs)) * weights[0]
    else:
        values = landscape.get_all_objectives(xs)
        return np.sum(values * weights)


def avg_weighted(landscape, xs: list, weights):
    """
    Calculate weighted sum across objectives, averaged over sequences in batch.
    Returns: scalar = mean_over_sequences(sum_over_objectives(fitness[seq, obj] * weight[obj]))
    """
    return sum_weighted(landscape, xs, weights) / len(xs)


def avg_unique_weighted(landscape, xs: list, weights):
    """
    Calculate weighted sum across objectives for unique sequences, averaged over batch size.
    Deduplicates sequences but normalizes by original batch size.
    Returns: scalar = sum_over_unique_sequences(sum_over_objectives(fitness[seq, obj] * weight[obj])) / batch_size
    """
    # remove duplicate sequences
    unique_xs = list(np.unique(xs))
    return sum_weighted(landscape, unique_xs, weights) / len(xs)


def hit_above_threshold_all(landscape, xs: list, threshold=0.5, objective=None):
    """
    Calculates the number of all sequences that score above a given threshold.
    """
    if not xs:
        return 0
    fitness_values = landscape.get_fitness_values(xs, objective=objective)
    return (fitness_values > threshold).sum()


def hits_all_objectives(
    landscape, xs: list, thresholds: dict, uniqueness_type="dna"
) -> tuple[int, int]:
    """
    Count hits where ALL objectives pass their thresholds (AND logic).

    Args:
        landscape: Multi-objective landscape
        xs: List of sequences
        thresholds: Dict mapping objective name to threshold

    Returns:
        (unique_hits, all_hits): Unique sequences passing all thresholds,
                                 and total count including duplicates
    """
    if not xs:
        return 0, 0

    missing_objs = set(landscape.active_objs) - set(thresholds.keys())
    if missing_objs:
        raise ValueError(f"Thresholds missing for active objectives: {missing_objs}")

    unique_xs = list(set(xs))
    all_objectives = landscape.get_all_objectives(unique_xs)

    all_pass = np.ones(len(unique_xs), dtype=bool)
    for i, obj_name in enumerate(landscape.active_objs):
        threshold = thresholds[obj_name]
        all_pass = all_pass & (all_objectives[:, i] > threshold)

    seq_to_hit = {seq: passes for seq, passes in zip(unique_xs, all_pass)}
    all_hits = sum(1 for seq in xs if seq_to_hit.get(seq, False))

    if landscape.is_peptide and uniqueness_type == "peptide":
        # Count unique proteins that pass, not unique DNA sequences
        hit_seqs = [seq for seq, passes in seq_to_hit.items() if passes]
        unique_hits = len(filter_same_protein_encoding(hit_seqs))
    else:
        unique_hits = int(all_pass.sum())

    return unique_hits, all_hits


def hit_above_threshold_individual(landscape, xs: list, threshold=0.5):
    """
    Calculates the number of all sequences that score above a given threshold.
    """
    if not xs:
        return 0
    fitness_values = landscape.get_fitness_values(xs)
    fitness_values = torch.tensor(fitness_values)
    fitness_values = fitness_values > threshold
    return fitness_values


def hit_above_threshold_batched(landscape, xs: list, threshold=0.5, mc_samples=1):
    """
    Splits the sequences into mc_samples batches and then calculates number of unique hits for each batch
    Returns a torch tensor of shape mc_samples * batch_size
    """
    if not xs:
        return 0
    fitness_values = landscape.get_fitness_values(xs)
    fitness_values = np.array(fitness_values)
    batch_size = len(xs) // mc_samples
    scores = torch.zeros(mc_samples * batch_size)
    for b in range(mc_samples):
        start, end = b * batch_size, (b + 1) * batch_size
        current_sequences = xs[start:end]

        _, unique_idx = np.unique(current_sequences, return_index=True)

        unique_idx = unique_idx + start

        current_score = (fitness_values[unique_idx] > threshold).sum()
        scores[start:end] = current_score

    return scores


# --- Helper Functions ---


def enumerate_sequences(alphabet, length):
    """
    Enumerate all sequences of a given length using the specified alphabet.
    """
    return ["".join(seq) for seq in product(alphabet, repeat=length)]


def enumerate_sequences_and_score(landscape, objective=None):
    """
    Enumerate all sequences in the landscape and return their scores as a DataFrame.
    Assumes a fixed sequence length.
    """
    all_sequences = enumerate_sequences(landscape.alphabet, landscape.str_length)
    fitness_values = landscape.get_fitness_values(all_sequences, objective=objective)
    sequence_scores = pd.DataFrame({"sequence": all_sequences, "score": fitness_values})
    sequence_scores = sequence_scores.sort_values(by="score", ascending=False)
    return sequence_scores.reset_index(drop=True)


def get_top_proportion_threshold(sequence_scores, top_proportion=0.1):
    """
    Compute the score required to be in the top proportion of sequences.
    """
    threshold_value = sequence_scores["score"].quantile(1 - top_proportion)
    return threshold_value


def is_landscape_enumerable(landscape):
    """Returns True if the landscape is small enough to enumerate all sequences."""
    return landscape.str_length <= 8 and len(landscape.alphabet) == 4


def is_landscape_multi_objective(landscape):
    """Returns True if the landscape has multiple active objectives"""
    return len(landscape.active_objs) > 1


def dna_to_protein(dna_sequence):
    """
    Translate a DNA sequence to protein using the standard genetic code.
    Truncates extra nucleotides if sequence length is not a multiple of 3.
    Stops at stop codons.

    Args:
        dna_sequence: DNA sequence string (A, C, G, T characters)

    Returns:
        Protein sequence string, or empty string if input is invalid.
    """
    protein_sequence = ""
    if not isinstance(dna_sequence, str) or not all(
        c in DNA_BASES for c in dna_sequence.upper()
    ):
        print(
            "Error: Invalid DNA sequence. Please use a string containing only 'A', 'G', 'C', 'T'."
        )
        return ""

    dna_sequence = dna_sequence.upper()

    for i in range(0, len(dna_sequence) - (len(dna_sequence) % 3), 3):
        codon = dna_sequence[i : i + 3]
        amino_acid = CODON_TABLE.get(codon, "")
        if amino_acid == "*":
            break
        protein_sequence += amino_acid

    return protein_sequence


def sequences_to_valid_peptides(sequences, min_len=5, max_len=15):
    """
    Recieves a string of DNA sequences, filters the invalid ones based on length.
    Returns the indices of valid peptides and the valid peptides.
    """
    dna_to_peptide = dna_to_protein_batch(sequences)
    peptides = [dna_to_peptide[seq] for seq in sequences]
    # Find which indices are valid
    valid_idx = [i for i, p in enumerate(peptides) if min_len <= len(p) <= max_len]
    valid_peptides = [peptides[i] for i in valid_idx]
    return valid_idx, valid_peptides


def generate_uniform_random_sequences(alphabet, str_length, n):
    """Generate n uniform random sequences of given length from the alphabet."""
    idxs = np.random.randint(0, len(alphabet), size=(n, str_length))
    return ["".join([alphabet[i] for i in row]) for row in idxs]


def tokenize_seqs(xs):
    return np.array([[DNA_ALPHABET[c] for c in seq] for seq in xs])


def generate_iupac_sequences(template: str, n: int):
    """
    Generate n random DNA sequences consistent with a template string
    containing IUPAC codes.

    Args:
        template (str): IUPAC sequence (e.g. "NNKATG")
        n (int): number of sequences to generate

    Returns:
        list[str]: list of generated sequences
    """
    # Ensure the input is a string and contains valid characters
    if not isinstance(template, str) or not all(
        c in IUPAC_MAP.keys() for c in template.upper()
    ):
        print(
            "Error: Invalid template sequence. Please use a string containing only valid IUPAC symbols."
        )
        return []

    choices = [np.array(IUPAC_MAP[base]) for base in template]
    sampled = [np.random.choice(opt, size=n) for opt in choices]
    sequences = ["".join(row) for row in zip(*sampled)]

    return sequences


def generate_nnk_sequences(sequence_length: int, n: int):
    """
    Generate n random DNA sequences of length sequence_length such that each codon is sampled from NNK degenerate codon.
    """
    # Ensure sequence_length is divisible by three
    if sequence_length % 3 != 0:
        print(
            "Error: the sequence length is not divisible into codons. Please use a sequence length divisible by 3."
        )
        return []

    template = "NNK" * (sequence_length // 3)

    return generate_iupac_sequences(template, n)


def filter_same_protein_encoding(sequences):
    """
    Receieves a a list of string DNA sequences, and deduplicates the one that encode for the same protein.
    """
    seen = {}
    for seq in sequences:
        peptide = dna_to_protein(seq)
        if peptide not in seen:
            seen[peptide] = seq
    return list(seen.values())
