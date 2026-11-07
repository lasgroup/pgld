import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict
from scipy.stats import spearmanr

from generate_training_data import create_dataloaders

log = logging.getLogger(__name__)


def train_oracle(
    model,
    train_loader,
    val_loader,
    params_dir,
    device="cpu",
    n_epochs=1000,
    lr=1e-3,
    patience=50,
) -> Dict[str, Dict]:
    """
    Train the oracle CNN model.

    Args:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        params_dir: Where to save the parameters of the oracle with the best validation loss
        device: Device to train on ('cpu' or 'cuda')
        n_epochs: Maximum number of epochs
        lr: Learning rate
        patience: Early stopping patience (epochs without improvement)

    Returns:
        training history dictionary
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0

    # Track losses for plotting
    history = {"train_loss": [], "val_loss": [], "epochs": []}
    validation = {"outputs": [], "targets": []}

    log.info(
        f"Started training with {len(train_loader.dataset)} training examples and {len(val_loader.dataset)} validation examples."
    )

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        for seqs, targets in train_loader:
            seqs, targets = seqs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(seqs).squeeze(1)
            loss = criterion(outputs, model.normalize_targets(targets))
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * seqs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for seqs, targets in val_loader:
                seqs, targets = seqs.to(device), targets.to(device)
                outputs = model(seqs).squeeze(1)
                loss = criterion(outputs, model.normalize_targets(targets))

                val_loss += loss.item() * seqs.size(0)

        val_loss /= len(val_loader.dataset)

        # Store losses
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["epochs"].append(epoch + 1)

        # Print progress
        if epoch % 100 == 0:
            log.info(
                f"Epoch [{epoch+1}/{n_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
            )

        # Save best model and check early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "target_mean": model.target_mean,
                    "target_std": model.target_std,
                },
                params_dir,
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            log.info(f"Early stopping at epoch {epoch+1}")
            break

    # Load best model before computing validation metrics
    checkpoint = torch.load(params_dir)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
        model.target_mean = checkpoint.get("target_mean")
        model.target_std = checkpoint.get("target_std")
    else:
        # Checkpoint saved as a bare state_dict
        model.load_state_dict(checkpoint)
    log.info(f"Training completed. Best validation loss: {best_val_loss:.4f}")

    # Get validation predictions from best model
    model.eval()
    validation["outputs"] = []
    validation["targets"] = []
    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            outputs = model.predict(seqs).squeeze(1)
            validation["outputs"].append(outputs.cpu())
            validation["targets"].append(targets.cpu())
    validation["outputs"] = torch.cat(validation["outputs"])
    validation["targets"] = torch.cat(validation["targets"])

    training_result = {"history": history, "validation": validation}

    return training_result


def evaluate_ensemble_on_held_out(
    ensemble: nn.ModuleList,
    val_loader,
    device: str = "cpu",
) -> Dict[str, float]:
    """
    Evaluate an ensemble on a held-out validation set.

    Uses the first model in the ensemble (consistent with training metrics).
    """
    model = ensemble[0]
    model.eval()

    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for seqs, targets in val_loader:
            seqs, targets = seqs.to(device), targets.to(device)
            outputs = model.predict(seqs).squeeze(1)
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())

    outputs = torch.cat(all_outputs).numpy()
    targets = torch.cat(all_targets).numpy()

    mse = float(np.mean((outputs - targets) ** 2))
    ss_res = np.sum((targets - outputs) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    spearman_corr, _ = spearmanr(targets, outputs)

    return {"mse": mse, "r2": r2, "spearman": float(spearman_corr)}


class BBOAlgorithm:
    def __init__(
        self,
        landscape,
        ensembles: Dict[str, List[nn.Module]],
        objectives: List[str] = None,
    ):
        self.active_objectives = objectives or list(ensembles.keys())
        self.observed_data: Dict[str, Dict[str, float]] = {}
        self.landscape = landscape
        self.ensembles = ensembles

    def add_observations(
        self, new_sequences: List[str], new_fitness_values: Dict[str, List[float]]
    ) -> None:
        """Add new observations. Duplicates are overwritten with new values."""
        for i, seq in enumerate(new_sequences):
            self.observed_data[seq] = {
                obj: new_fitness_values[obj][i] for obj in self.active_objectives
            }

    def _get_sequences_and_fitness(
        self, objective: str
    ) -> tuple[List[str], List[float]]:
        """Extract sequences and fitness values for a specific objective."""
        sequences = list(self.observed_data.keys())
        fitness_values = [self.observed_data[seq][objective] for seq in sequences]
        return sequences, fitness_values

    def train_ensemble_for_objective(
        self,
        objective: str,
        params_dir: str,
        subsample: float = 0.8,
        batch_size: int = 64,
        val_split: float = 0.2,
        device: str = "cpu",
        n_epochs: int = 1000,
        lr: float = 1e-3,
        patience: int = 50,
    ) -> Dict:
        """
        Train all models in the ensemble for a specific objective.

        Returns training result for the first model only (for visualization).
        """
        ensemble = self.ensembles[objective]
        sequences, fitness_values = self._get_sequences_and_fitness(objective)
        first_model_result = None

        # Fit normalization on full data (before subsampling)
        fitness_array = np.array(fitness_values)
        for model in ensemble:
            model.fit_target_normalization(fitness_array)

        for i, model in enumerate(ensemble):
            # Subsampling (skip if ensemble size is 1)
            if len(ensemble) == 1:
                sample_seqs = sequences
                sample_fitness = fitness_values
            else:
                n = len(sequences)
                indices = np.arange(n)
                np.random.shuffle(indices)
                split_size = int(n * subsample)
                indices = list(indices[:split_size])
                sample_seqs = [sequences[idx] for idx in indices]
                sample_fitness = [fitness_values[idx] for idx in indices]

            train_loader, val_loader = create_dataloaders(
                self.landscape,
                sample_seqs,
                sample_fitness,
                "train_and_val",
                batch_size=batch_size,
                val_split=val_split,
            )

            training_result = train_oracle(
                model,
                train_loader,
                val_loader,
                params_dir=os.path.join(params_dir, f"oracle_{i}.pth"),
                device=device,
                n_epochs=n_epochs,
                lr=lr,
                patience=patience,
            )

            if i == 0:
                first_model_result = training_result

        return first_model_result

    def train_all_ensembles(
        self,
        params_dir: str,
        subsample: float = 0.8,
        batch_size: int = 64,
        val_split: float = 0.2,
        device: str = "cpu",
        n_epochs: int = 1000,
        lr: float = 1e-3,
        patience: int = 50,
    ) -> Dict[str, Dict]:
        """
        Train ensembles for all objectives.

        Returns dict mapping objective name to training result (first model only).
        """
        results = {}
        for objective in self.active_objectives:
            obj_params_dir = os.path.join(params_dir, objective)
            os.makedirs(obj_params_dir, exist_ok=True)
            results[objective] = self.train_ensemble_for_objective(
                objective,
                obj_params_dir,
                subsample=subsample,
                batch_size=batch_size,
                val_split=val_split,
                device=device,
                n_epochs=n_epochs,
                lr=lr,
                patience=patience,
            )
        return results
