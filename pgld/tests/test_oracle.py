"""Tests for oracle normalization functionality."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle import OracleCNNBig


class TestOracleCNNBigNormalization:

    @pytest.fixture
    def model(self) -> OracleCNNBig:
        return OracleCNNBig(input_dim=9, input_channels=20)

    def test_fit_target_normalization(self, model: OracleCNNBig) -> None:
        targets = np.array([-3.5, -2.0, -4.0, -1.5, -5.0])
        model.fit_target_normalization(targets)

        assert abs(model.target_mean - np.mean(targets)) < 1e-6
        assert abs(model.target_std - np.std(targets)) < 1e-6

    def test_normalize_denormalize_roundtrip(self, model: OracleCNNBig) -> None:
        targets = np.array([-3.5, -2.0, -4.0, -1.5, -5.0])
        model.fit_target_normalization(targets)

        targets_tensor = torch.tensor(targets, dtype=torch.float32)
        normalized = model.normalize_targets(targets_tensor)
        denormalized = model.denormalize_predictions(normalized)

        assert torch.allclose(denormalized, targets_tensor)

    def test_normalize_without_fitting_raises_error(self, model: OracleCNNBig) -> None:
        targets_tensor = torch.tensor([-3.5, -2.0, -4.0])

        with pytest.raises(ValueError, match="Target normalization not fitted"):
            model.normalize_targets(targets_tensor)

    def test_predict_returns_denormalized(self, model: OracleCNNBig) -> None:
        targets = np.array([-3.5, -2.0, -4.0, -1.5, -5.0])
        model.fit_target_normalization(targets)

        x = torch.randn(4, 9, 20)
        model.eval()
        with torch.no_grad():
            output_predict = model.predict(x)
            output_forward_denorm = model(x, normalize_output=False)

        assert torch.allclose(output_predict, output_forward_denorm)
