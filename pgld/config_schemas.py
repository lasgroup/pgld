"""
config_schemas.py

Hydra structured configs for type-safe configuration management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class OptimizationConfig:
    """Gradient descent optimization parameters."""

    iters: int
    lr: float
    ema_alpha: float = 0.9


@dataclass
class SamplingConfig:
    """Batch sampling parameters."""

    batch_size: int
    mc_samples: int = 1


@dataclass
class EntropyConfig:
    """Entropy regularization settings."""

    lambda_: float = 0.0
    type: str = (
        "exact"  # Options: "exact" | "gradient_estimate" | "monte_carlo" | "equal_share" | "marginal_gain"
    )
    sequence_type: str = "dna"  # Options: "dna" | "peptide"
    baseline_per_template: bool = False
    n_templates: Optional[int] = None  # Lowest n templates used for entropy


@dataclass
class PoolingConfig:
    """Landscape query optimization for pooled training."""

    enabled: bool = False
    gpu_accelerated: bool = False
    queries_per_batch: Optional[int] = None
    uniqueness: str = "dna"  # Options: "dna" | "peptide"


@dataclass
class EvaluationConfig:
    """Evaluation and logging settings."""

    extra_eval: bool = False
    eval_only_at_end: bool = False
    eval_threshold: Optional[float] = None  # Set at runtime for single-objective
    eval_thresholds: Optional[dict] = (
        None  # Set at runtime for multi-objective {obj_name: threshold}
    )
    eval_batch_size: int = 100000
    checkpointing: bool = True
    verbose: bool = True
    timing: bool = False
    steps_til_vis: Optional[int] = None  # null = only at end, int = every N steps


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    optimization: OptimizationConfig
    sampling: SamplingConfig
    entropy: EntropyConfig
    pooling: PoolingConfig
    evaluation: EvaluationConfig
    reward_type: str = (
        "hits"  # Options: "hits" | "average" | "unique_hits" | "unique_mean"
    )
    reward_aggregation_unit: str = "batch"  # Options: "batch" | "template"
    per_sequence_rewards: bool = False
    unique_hits_approx: str = "exp"  # Options: "exp" | "pade"


def build_training_config(cfg_dict) -> TrainingConfig:
    """
    Build TrainingConfig from a config dict (handles both OmegaConf and plain dict).

    Args:
        cfg_dict: Configuration dict from Hydra (cfg.training)

    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(
        optimization=OptimizationConfig(
            iters=cfg_dict.optimization.iters,
            lr=cfg_dict.optimization.lr,
            ema_alpha=cfg_dict.optimization.get("ema_alpha", 0.9),
        ),
        sampling=SamplingConfig(
            batch_size=cfg_dict.sampling.batch_size,
            mc_samples=cfg_dict.sampling.get("mc_samples", 1),
        ),
        entropy=EntropyConfig(
            lambda_=cfg_dict.entropy.get("lambda_", 0.0),
            type=cfg_dict.entropy.get("type", "exact"),
            sequence_type=cfg_dict.entropy.get("sequence_type", "dna"),
            baseline_per_template=cfg_dict.entropy.get("baseline_per_template", False),
            n_templates=cfg_dict.entropy.get("n_templates"),
        ),
        pooling=PoolingConfig(
            enabled=cfg_dict.pooling.get("enabled", False),
            gpu_accelerated=cfg_dict.pooling.get("gpu_accelerated", False),
            queries_per_batch=cfg_dict.pooling.get("queries_per_batch"),
            uniqueness=cfg_dict.pooling.get("uniqueness", "dna"),
        ),
        evaluation=EvaluationConfig(
            extra_eval=cfg_dict.evaluation.get("extra_eval", False),
            eval_only_at_end=cfg_dict.evaluation.get("eval_only_at_end", False),
            eval_threshold=cfg_dict.evaluation.get("eval_threshold"),
            eval_thresholds=None,  # Set at runtime in train.py
            eval_batch_size=cfg_dict.evaluation.get("eval_batch_size", 100000),
            checkpointing=cfg_dict.evaluation.get("checkpointing", True),
            verbose=cfg_dict.evaluation.get("verbose", True),
            timing=cfg_dict.evaluation.get("timing", False),
            steps_til_vis=cfg_dict.evaluation.get("steps_til_vis"),
        ),
        reward_type=cfg_dict.get("reward_type", "hits"),
        reward_aggregation_unit=cfg_dict.get("reward_aggregation_unit", "batch"),
        per_sequence_rewards=cfg_dict.get("per_sequence_rewards", False),
        unique_hits_approx=cfg_dict.get("unique_hits_approx", "exp"),
    )


@dataclass
class OracleTrainingConfig:
    """Oracle training configuration for Bayesian Optimization."""

    n_epochs: int = 200
    lr: float = 1e-3
    batch_size: int = 64
    patience: int = 50
    n_ensembles: int = 5


@dataclass
class BOConfig:
    """Bayesian Optimization configuration."""

    bo_iters: int = 5
    generate_per_round: int = (
        10000  # how many new data points to generate from our policy per round
    )
    with_replacement: bool = (
        True  # when we calculate the reward do we filter out sequences from the previous round, False if yes, True if no
    )
    oracle_training: OracleTrainingConfig = field(default_factory=OracleTrainingConfig)
    baseline: Optional[str] = (
        None  # Options: null | "random" | "rational" | "rational_simplified" | "trimer" | "all"
    )
    baselines_only: bool = False
    landscape_batch_size: int = 1024  # batch size for BoLandscape oracle predictions
    adaptive_threshold: bool = (
        False  # update hit threshold based on oracle predictions each round
    )
    save_samples: bool = (
        False  # save sampled sequences and fitness values per round for qualitative analysis
    )


def build_oracle_training_config(cfg_dict) -> OracleTrainingConfig:
    """
    Build OracleTrainingConfig from a config dict.

    Args:
        cfg_dict: Configuration dict from Hydra (cfg.bo.oracle_training)

    Returns:
        OracleTrainingConfig instance
    """
    return OracleTrainingConfig(
        n_epochs=cfg_dict.get("n_epochs", 200),
        lr=cfg_dict.get("lr", 1e-3),
        batch_size=cfg_dict.get("batch_size", 64),
        patience=cfg_dict.get("patience", 50),
        n_ensembles=cfg_dict.get("n_ensembles", 5),
    )


def build_bo_config(cfg_dict) -> BOConfig:
    """
    Build BOConfig from a config dict.

    Args:
        cfg_dict: Configuration dict from Hydra (cfg.bo)

    Returns:
        BOConfig instance
    """
    oracle_training = build_oracle_training_config(cfg_dict.oracle_training)
    return BOConfig(
        bo_iters=cfg_dict.get("bo_iters", 5),
        generate_per_round=cfg_dict.get("generate_per_round", 10000),
        with_replacement=cfg_dict.get("with_replacement", True),
        oracle_training=oracle_training,
        baseline=cfg_dict.get("baseline", None),
        baselines_only=cfg_dict.get("baselines_only", False),
        landscape_batch_size=cfg_dict.get("landscape_batch_size", 1024),
        adaptive_threshold=cfg_dict.get("adaptive_threshold", False),
        save_samples=cfg_dict.get("save_samples", False),
    )


def register_configs() -> None:
    """Register structured configs with Hydra ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(group="training", name="base_training_schema", node=TrainingConfig)
    cs.store(
        group="bo",
        name="base_bo_schema",
        node=BOConfig,
    )
