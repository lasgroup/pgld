"""
generate_training_data.py

Helper functions for generating sequences and creating dataloaders for oracle training in BO.
"""

from __future__ import annotations

import logging
from typing import Literal

import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from landscapes import generate_nnk_sequences, generate_uniform_random_sequences
from constants import one_hot_encode_dna, one_hot_encode_peptide

log = logging.getLogger(__name__)


class SequenceDataset(Dataset):
    """Base class for sequence datasets used in oracle training."""

    def __init__(
        self,
        sequences: list[str],
        targets: list[float],
        length: int,
        encode_fn: callable,
    ):
        self.encoded_sequences = torch.stack(
            [encode_fn(seq, length) for seq in sequences]
        )
        self.targets = torch.tensor(targets, dtype=torch.float32)
        self.length = length

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_sequences[idx], self.targets[idx]

    def __len__(self) -> int:
        return self.encoded_sequences.shape[0]


class PeptideDataset(SequenceDataset):
    """Dataset for peptide sequences and their fitness values."""

    def __init__(self, sequences: list[str], targets: list[float], length: int):
        super().__init__(sequences, targets, length, one_hot_encode_peptide)


class DNADataset(SequenceDataset):
    """Dataset for DNA sequences and their fitness values."""

    def __init__(self, sequences: list[str], targets: list[float], length: int):
        super().__init__(sequences, targets, length, one_hot_encode_dna)


def get_starting_data_for_bo(landscape, n_seq: int) -> list[str]:
    """
    Generate starting DNA sequences for BO. Always generates fresh data.

    Args:
        landscape: The landscape to generate sequences for.
        n_seq: Number of sequences to generate.

    Returns:
        List of DNA sequences.
    """
    log.info(f"Generating {n_seq} starting sequences for BO")

    if landscape.is_peptide:
        # Use NNK codon sampling instead of uniform random to avoid stop codons
        # and better match experimental peptide library distributions
        sequences = generate_nnk_sequences(landscape.str_length, n_seq)
    else:
        sequences = generate_uniform_random_sequences(
            landscape.alphabet, landscape.str_length, n_seq
        )

    log.info(f"Generated {len(sequences)} DNA sequences.")
    return sequences


def create_data_from_policy(policy, landscape, n_seq: int) -> list[str]:
    """
    Sample DNA sequences from policy.

    Args:
        policy: Policy to sample from.
        landscape: Landscape (used for alphabet).
        n_seq: Number of sequences to sample.

    Returns:
        List of DNA sequences.
    """
    sample_result = policy.sample(n_seq)
    all_sequences = sample_result.sequences
    dna_sequences = ["".join([landscape.alphabet[c] for c in s]) for s in all_sequences]

    log.info(f"Sampled {len(dna_sequences)} DNA sequences from policy.")
    return dna_sequences


def create_dataloaders(
    landscape,
    sequences: list[str],
    fitness_values: list[float],
    mode: Literal["train", "train_and_val", "test"],
    batch_size: int = 64,
    val_split: float = 0.2,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Create train/val/test dataloaders from sequences and fitness values.

    Args:
        landscape: Landscape (used to determine sequence type and length).
        sequences: List of sequences (peptides or DNA).
        fitness_values: Fitness values for each sequence.
        mode: One of 'train', 'train_and_val', or 'test'.
        batch_size: Batch size for dataloaders.
        val_split: Validation split fraction (only used in 'train_and_val' mode).

    Returns:
        Tuple of (train_loader, val_loader). val_loader is None except in 'train_and_val' mode.
    """
    str_length = landscape.str_length
    if landscape.is_peptide:
        DatasetClass = PeptideDataset
        str_length = str_length // 3
    else:
        DatasetClass = DNADataset

    if mode == "test":
        test_dataset = DatasetClass(sequences, fitness_values, str_length)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        log.info(f"Created test dataloader with {len(test_dataset)} samples.")
        return test_loader, None

    elif mode == "train":
        train_dataset = DatasetClass(sequences, fitness_values, str_length)
        # Drop last batch only if it would have size 1 (BatchNorm fails with batch size 1)
        drop_last = len(train_dataset) % batch_size == 1
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=drop_last,
            generator=torch.Generator().manual_seed(42),
        )
        log.info(f"Created train dataloader with {len(train_dataset)} samples.")
        return train_loader, None

    elif mode == "train_and_val":
        train_seqs, val_seqs, train_targets, val_targets = train_test_split(
            sequences, fitness_values, test_size=val_split, random_state=42
        )
        assert val_split > 0
        train_dataset = DatasetClass(train_seqs, train_targets, str_length)
        val_dataset = DatasetClass(val_seqs, val_targets, str_length)
        # Drop last batch only if it would have size 1 (BatchNorm fails with batch size 1)
        train_drop_last = len(train_dataset) % batch_size == 1
        val_drop_last = len(val_dataset) % batch_size == 1
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=train_drop_last,
            generator=torch.Generator().manual_seed(42),
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, drop_last=val_drop_last
        )
        log.info(
            f"Created train/val dataloaders with {len(train_dataset)}/{len(val_dataset)} samples."
        )
        return train_loader, val_loader

    else:
        raise ValueError(
            f"Invalid mode '{mode}'. Must be one of: 'train', 'train_and_val', 'test'"
        )


def generate_for_random_comparison(
    landscape, seq_per_iter: int, num_iters: int, threshold: float
) -> tuple[list[int], list[int]]:
    """
    Simulates random sampling baseline for comparison with BO.

    Takes a landscape, threshold and num_iters and simulates the result of random
    sampling for each BO iteration.

    Returns the number of all hits and unique hits per round.
    """
    raise NotImplementedError(
        "generate_for_random_comparison needs to be updated for multi-objective support"
    )
