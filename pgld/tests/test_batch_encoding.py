"""Tests for batch one-hot encoding functions."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from constants import (
    AA_TO_IDX_NO_STOP,
    DNA_ALPHABET,
    one_hot_encode_peptide,
    one_hot_encode_dna,
    batch_one_hot_encode_peptide,
    batch_one_hot_encode_dna,
)


class TestBatchOneHotEncodePeptide:

    def test_matches_single_encoding(self) -> None:
        sequences = ["ACDE", "FGHI", "KLMN"]
        length = 4

        batch_result = batch_one_hot_encode_peptide(sequences, length)
        single_results = torch.stack(
            [one_hot_encode_peptide(seq, length) for seq in sequences]
        )

        assert torch.allclose(batch_result, single_results)

    def test_output_shape(self) -> None:
        sequences = ["ACE", "FGH", "KLM", "PQR"]
        length = 5
        num_classes = len(AA_TO_IDX_NO_STOP)

        result = batch_one_hot_encode_peptide(sequences, length)

        assert result.shape == (4, 5, num_classes)

    def test_padding_matches_single(self) -> None:
        sequences = ["AC", "FGH"]
        length = 5

        batch_result = batch_one_hot_encode_peptide(sequences, length)
        single_results = torch.stack(
            [one_hot_encode_peptide(seq, length) for seq in sequences]
        )

        assert torch.allclose(batch_result, single_results)

    def test_single_sequence(self) -> None:
        sequences = ["ACDE"]
        length = 4

        batch_result = batch_one_hot_encode_peptide(sequences, length)
        single_result = one_hot_encode_peptide(sequences[0], length).unsqueeze(0)

        assert torch.allclose(batch_result, single_result)

    def test_lowercase_uppercase_equivalent(self) -> None:
        upper = ["ACDE"]
        lower = ["acde"]
        length = 4

        result_upper = batch_one_hot_encode_peptide(upper, length)
        result_lower = batch_one_hot_encode_peptide(lower, length)

        assert torch.allclose(result_upper, result_lower)

    def test_device_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        sequences = ["ACDE", "FGHI"]
        length = 4

        result = batch_one_hot_encode_peptide(sequences, length, device="cuda")

        assert result.device.type == "cuda"

    def test_one_hot_sums_to_one_for_valid_positions(self) -> None:
        sequences = ["ACDE"]
        length = 4

        result = batch_one_hot_encode_peptide(sequences, length)

        assert torch.allclose(result.sum(dim=-1), torch.ones(1, 4))


class TestBatchOneHotEncodeDNA:

    def test_matches_single_encoding(self) -> None:
        sequences = ["ACGT", "TGCA", "AAAA"]
        length = 4

        batch_result = batch_one_hot_encode_dna(sequences, length)
        single_results = torch.stack(
            [one_hot_encode_dna(seq, length) for seq in sequences]
        )

        assert torch.allclose(batch_result, single_results)

    def test_output_shape(self) -> None:
        sequences = ["ACG", "TGC", "AAA", "TTT"]
        length = 5
        num_classes = len(DNA_ALPHABET)

        result = batch_one_hot_encode_dna(sequences, length)

        assert result.shape == (4, 5, num_classes)

    def test_padding_matches_single(self) -> None:
        sequences = ["AC", "TGC"]
        length = 5

        batch_result = batch_one_hot_encode_dna(sequences, length)
        single_results = torch.stack(
            [one_hot_encode_dna(seq, length) for seq in sequences]
        )

        assert torch.allclose(batch_result, single_results)

    def test_single_sequence(self) -> None:
        sequences = ["ACGT"]
        length = 4

        batch_result = batch_one_hot_encode_dna(sequences, length)
        single_result = one_hot_encode_dna(sequences[0], length).unsqueeze(0)

        assert torch.allclose(batch_result, single_result)

    def test_lowercase_uppercase_equivalent(self) -> None:
        upper = ["ACGT"]
        lower = ["acgt"]
        length = 4

        result_upper = batch_one_hot_encode_dna(upper, length)
        result_lower = batch_one_hot_encode_dna(lower, length)

        assert torch.allclose(result_upper, result_lower)

    def test_device_cuda(self) -> None:
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        sequences = ["ACGT", "TGCA"]
        length = 4

        result = batch_one_hot_encode_dna(sequences, length, device="cuda")

        assert result.device.type == "cuda"

    def test_one_hot_sums_to_one_for_valid_positions(self) -> None:
        sequences = ["ACGT"]
        length = 4

        result = batch_one_hot_encode_dna(sequences, length)

        assert torch.allclose(result.sum(dim=-1), torch.ones(1, 4))
