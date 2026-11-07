"""
oracle.py

Defines oracles used for predicting fitness values from amino-acid sequences.
"""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class OracleCNNSmall(nn.Module):
    def __init__(
        self,
        input_dim=9,
        num_filters=64,
        kernel_size=3,
        hidden_dim=128,
        output_dim=1,
        dropout=0.3,
        input_channels=20,
    ):
        super(OracleCNNSmall, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=kernel_size,
            padding=1,
        )
        self.pool = nn.MaxPool1d(kernel_size=2)

        conv_output_size = num_filters * (input_dim // 2)

        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, input_dim, input_channels)
        input_dim = typically lenght of the sequence
        input_channels = typically 4 for DNA sequences and 20 for peptides
        """
        x = x.permute(0, 2, 1)  # (batch_size, input_channels, input_dim)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        out = self.fc2(x)
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with two convolutional layers, batch normalization, and skip connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Skip connection: if dimensions change, use 1x1 conv, otherwise identity
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.skip_bn = (
            nn.BatchNorm1d(out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.skip_bn(self.skip(x))

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        out = out + residual  # Add skip connection
        out = F.relu(out)

        return out


class OracleCNNBig(nn.Module):
    """
    Deep CNN with residual blocks for predicting peptide binding affinity.

    Architecture:
    - Input: (batch_size, seq_length, input_channels) one-hot encoded sequences
    - 3 Residual blocks with increasing filters
    - Global average pooling
    - 2 fully connected layers with dropout
    - Output: single value (log10 affinity)
    """

    def __init__(
        self,
        input_dim=9,
        num_filters=64,
        kernel_size=3,
        hidden_dim=128,
        output_dim=1,
        dropout=0.3,
        input_channels=20,
    ):
        super(OracleCNNBig, self).__init__()

        self.input_dim = input_dim

        # Initial convolution to transform from 20 or 4 channels
        self.conv_input = nn.Conv1d(input_channels, num_filters, kernel_size=1)
        self.bn_input = nn.BatchNorm1d(num_filters)

        # Residual blocks with increasing number of filters
        self.res_block1 = ResidualBlock(
            num_filters, num_filters, kernel_size, dropout=dropout * 0.5
        )
        self.res_block2 = ResidualBlock(
            num_filters, num_filters * 2, kernel_size, dropout=dropout * 0.5
        )
        self.res_block3 = ResidualBlock(
            num_filters * 2, num_filters * 4, kernel_size, dropout=dropout * 0.5
        )

        # Pooling layers
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Global average pooling to make network size-agnostic
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(num_filters * 4, hidden_dim)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._initialize_weights()

        # Target normalization parameters
        self.target_mean: Optional[float] = None
        self.target_std: Optional[float] = None

    def _initialize_weights(self):
        """Initialize network weights using He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def fit_target_normalization(self, targets: np.ndarray) -> None:
        """
        Compute normalization parameters from training targets.

        Args:
            targets: Training target values as numpy array.
        """
        self.target_mean = float(np.mean(targets))
        self.target_std = float(np.std(targets))
        log.info(
            f"Fitted target normalization - mean: {self.target_mean:.4f}, "
            f"std: {self.target_std:.4f}"
        )

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """
        Normalize targets to zero mean and unit variance.

        Args:
            targets: Raw target values.

        Returns:
            Normalized targets.
        """
        if self.target_mean is None or self.target_std is None:
            raise ValueError(
                "Target normalization not fitted. Call fit_target_normalization() first."
            )
        return (targets - self.target_mean) / self.target_std

    def denormalize_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Convert normalized predictions back to original scale.

        Args:
            predictions: Normalized predictions.

        Returns:
            Predictions on original scale.
        """
        if self.target_mean is None or self.target_std is None:
            raise ValueError(
                "Target normalization not fitted. Call fit_target_normalization() first."
            )
        return predictions * self.target_std + self.target_mean

    def forward(self, x: torch.Tensor, normalize_output: bool = True) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, seq_length, input_channels) - one-hot encoded.
            normalize_output: If True, return normalized predictions (for training).
                If False, return denormalized predictions (for deployment).

        Returns:
            output: tensor of shape (batch_size, 1)
        """
        # Permute to (batch_size, input_channels, seq_length) for Conv1d
        x = x.permute(0, 2, 1)

        # Initial convolution
        x = F.relu(self.bn_input(self.conv_input(x)))

        # Residual blocks with pooling
        x = self.res_block1(x)
        x = self.pool(x)

        x = self.res_block2(x)
        x = self.pool(x)

        x = self.res_block3(x)

        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = self.dropout(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn_fc2(self.fc2(x))))
        out = self.fc3(x)

        if not normalize_output and self.target_mean is not None:
            out = self.denormalize_predictions(out)

        return out

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate predictions on the original (unnormalized) scale.

        Args:
            x: Input tensor of shape (batch_size, seq_length, input_channels).

        Returns:
            Predictions on original scale, shape (batch_size, 1).
        """
        return self.forward(x, normalize_output=False)

    def get_num_parameters(self):
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
