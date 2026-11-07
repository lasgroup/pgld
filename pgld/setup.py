"""
setup.py

Factory functions for setting up experiments.
Provides reusable setup code for train.py and train_bo.py.
"""

import logging
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from antibody_config import load_antibody_config
from config_schemas import build_training_config, TrainingConfig
from landscapes import (
    DHFRFitnessLandscape,
    GB1FitnessLandscape,
    MHCflurryLandscape,
    PLMLogLikelihoodLandscape,
    PhoQFitnessLandscape,
    RiboswitchLandscape,
    TFBindingLandscape,
    TrpBFitnessLandscape,
    enumerate_sequences_and_score,
    get_top_proportion_threshold,
    is_landscape_enumerable,
)
from policy_init import init_from_fasta_sequences, init_from_iupac_sequence
from policies import (
    ArbitraryBaseOptimizer,
    FiniteBaseOptimizer,
    IupacBaseOptimizer,
    IupacPlusOptimizer,
    SlotConstrainedIupacOptimizer,
)

log = logging.getLogger(__name__)


def setup_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")

    return device


def setup_seeds(seed: int) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and random.

    Args:
        seed: The random seed to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_landscape(
    cfg,
    objective_names: List[str],
    device: torch.device,
    invalid_scores: Dict[str, float] = None,
):
    """
    Build a landscape from the config.

    Args:
        cfg: Hydra config (needs cfg.landscape).
        objective_names: List of objective names to use.
        device: The device for landscapes that need it (e.g., riboswitch, PLM).
        invalid_scores: Dict mapping objective names to invalid sequence scores.

    Returns:
        The constructed landscape instance.

    Raises:
        ValueError: If landscape name is unknown.
    """
    if cfg.landscape.name == "tf_binding":
        landscape = TFBindingLandscape(
            str_length=cfg.landscape.str_length, objectives=objective_names
        )
    elif cfg.landscape.name == "mhc_flurry":
        landscape = MHCflurryLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            min_peptide_len=cfg.landscape.get("min_peptide_len"),
            max_peptide_len=cfg.landscape.get("max_peptide_len"),
            invalid_scores=invalid_scores,
        )
    elif cfg.landscape.name == "riboswitch":
        landscape = RiboswitchLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            device=device,
            oracle_batch_size=cfg.landscape.oracle_batch_size,
            oracle_dir=cfg.landscape.oracle_dir,
        )
    elif cfg.landscape.name.startswith("plm_"):  # plm_linear, plm_esm2, plm_iglm
        # Load antibody config and inject context into model configs
        ab_cfg = load_antibody_config(cfg.landscape.antibody)
        log.info(
            f"Loaded antibody config: {ab_cfg.name} (CDR-H3 length: {ab_cfg.cdrh3_length})"
        )

        # Derive values from antibody config
        str_length = ab_cfg.cdrh3_length * 3
        peptide_len = ab_cfg.cdrh3_length
        wild_type = ab_cfg.wild_type_cdrh3

        # Inject left/right context into each model config
        models_cfg = OmegaConf.to_container(cfg.landscape.models, resolve=True)
        for model_name in models_cfg:
            if "left_context" not in models_cfg[model_name]:
                models_cfg[model_name]["left_context"] = ab_cfg.left_context
            if "right_context" not in models_cfg[model_name]:
                if model_name == "ESM2":
                    if not ab_cfg.right_context_esm:
                        raise ValueError(
                            f"Antibody '{ab_cfg.name}' has no right_context_esm configured, "
                            f"but ESM2 requires extended right context"
                        )
                    models_cfg[model_name]["right_context"] = ab_cfg.right_context_esm
                else:
                    models_cfg[model_name]["right_context"] = ab_cfg.right_context
        models_cfg = OmegaConf.create(models_cfg)

        landscape = PLMLogLikelihoodLandscape(
            str_length=str_length,
            objectives=objective_names,
            device=device,
            min_peptide_len=peptide_len,
            max_peptide_len=peptide_len,
            models=models_cfg,
            oracle_batch_size_per_gpu=cfg.landscape.get(
                "oracle_batch_size_per_gpu", 1024
            ),
            wild_type=wild_type,
            max_edit_distance=cfg.landscape.get("max_edit_distance"),
        )
    elif cfg.landscape.name == "gb1":
        landscape = GB1FitnessLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            data_path=cfg.landscape.data_path,
            min_peptide_len=cfg.landscape.get("min_peptide_len", 4),
            max_peptide_len=cfg.landscape.get("max_peptide_len", 4),
            wild_type=cfg.landscape.get("wild_type"),
            invalid_scores=invalid_scores,
        )
    elif cfg.landscape.name == "phoq":
        landscape = PhoQFitnessLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            data_path=cfg.landscape.data_path,
            min_peptide_len=cfg.landscape.get("min_peptide_len", 4),
            max_peptide_len=cfg.landscape.get("max_peptide_len", 4),
            wild_type=cfg.landscape.get("wild_type"),
            invalid_scores=invalid_scores,
        )
    elif cfg.landscape.name == "dhfr":
        landscape = DHFRFitnessLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            data_path=cfg.landscape.data_path,
            wild_type=cfg.landscape.get("wild_type"),
            invalid_scores=invalid_scores,
        )
    elif cfg.landscape.name == "trpb":
        landscape = TrpBFitnessLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            data_path=cfg.landscape.data_path,
            min_peptide_len=cfg.landscape.get("min_peptide_len", 4),
            max_peptide_len=cfg.landscape.get("max_peptide_len", 4),
            wild_type=cfg.landscape.get("wild_type"),
            invalid_scores=invalid_scores,
        )
    else:
        raise ValueError(f"Unknown landscape name: {cfg.landscape.name}")
    return landscape


def build_policy(cfg, landscape, alphabet_size, device: torch.device):
    """
    Build a policy from the config.

    Args:
        cfg: Hydra config (needs cfg.policy and cfg.training.sampling.batch_size).
        landscape: The landscape (used for str_length and alphabet_size).
        device: The device to place the policy on.

    Returns:
        The constructed policy instance.

    Raises:
        ValueError: If policy name is unknown.
    """
    if cfg.policy.name == "arbitrary":
        policy = ArbitraryBaseOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=cfg.training.sampling.batch_size,
            num_templates=cfg.policy.num_templates,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "finite":
        policy = FiniteBaseOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=cfg.training.sampling.batch_size,
            num_templates=cfg.policy.num_templates,
            num_bottles=cfg.policy.num_bottles,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "iupac":
        policy = IupacBaseOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=cfg.training.sampling.batch_size,
            num_templates=cfg.policy.num_templates,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "iupac_plus":
        num_custom_bottles = getattr(cfg.policy, "num_custom_bottles", 5)
        policy = IupacPlusOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=cfg.training.sampling.batch_size,
            num_templates=cfg.policy.num_templates,
            num_custom_bottles=num_custom_bottles,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "slot_constrained_iupac":
        max_non_agct_slots = getattr(cfg.policy, "max_non_agct_slots", 4)
        policy = SlotConstrainedIupacOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=cfg.training.sampling.batch_size,
            num_templates=cfg.policy.num_templates,
            max_non_agct_slots=max_non_agct_slots,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", True),
        )
    else:
        raise ValueError(f"Unknown policy name: {cfg.policy.name}")

    # Initialize policy epsilon-greedy around IUPAC sequence if both sequence and epsilon are specified
    if (
        hasattr(cfg.policy, "init_iupac_sequence")
        and cfg.policy.init_iupac_sequence is not None
        and hasattr(cfg.policy, "init_epsilon")
        and cfg.policy.init_epsilon is not None
    ):
        if hasattr(policy, "latent_phi") or hasattr(policy, "latent_w"):
            log.info(
                f"Initializing policy from IUPAC sequence: {cfg.policy.init_iupac_sequence}"
            )
            log.info(f"  Using epsilon={cfg.policy.init_epsilon} for noise")
            init_from_iupac_sequence(
                policy, cfg.policy.init_iupac_sequence, epsilon=cfg.policy.init_epsilon
            )
            log.info(f"  Successfully initialized {cfg.policy.name} policy")
        else:
            log.warning(
                f"Policy '{cfg.policy.name}' does not support epsilon-greedy IUPAC initialization (no latent_phi or latent_w)"
            )

    # Initialize from FASTA file (multi-template with random codon selection)
    elif (
        hasattr(cfg.policy, "init_fasta_path")
        and cfg.policy.init_fasta_path is not None
        and hasattr(cfg.policy, "init_epsilon")
        and cfg.policy.init_epsilon is not None
    ):
        if hasattr(policy, "latent_phi") or hasattr(policy, "latent_w"):
            seed = getattr(cfg.policy, "init_fasta_seed", None)
            log.info(f"Initializing policy from FASTA: {cfg.policy.init_fasta_path}")
            log.info(
                f"  Templates: {policy.num_templates}, epsilon: {cfg.policy.init_epsilon}, seed: {seed}"
            )

            selected_seqs = init_from_fasta_sequences(
                policy,
                cfg.policy.init_fasta_path,
                epsilon=cfg.policy.init_epsilon,
                seed=seed,
            )

            log.info(f"  Selected AA sequences for {len(selected_seqs)} templates:")
            for i, seq in enumerate(selected_seqs):
                log.info(f"    Template {i}: {seq}")
        else:
            log.warning(
                f"Policy '{cfg.policy.name}' does not support FASTA initialization (no latent_phi or latent_w)"
            )

    return policy


def build_invalid_scores(
    cfg, objective_names, for_bo: bool = False
) -> Dict[str, float]:
    """
    Extract invalid_score from each objective config.

    Args:
        cfg: Hydra config (needs cfg.landscape.objectives).
        objective_names: List of objective names.
        for_bo: If True, use bo_invalid_score (falls back to invalid_score, then -inf).

    Returns:
        Dictionary mapping objective names to their invalid_score.
        Defaults to -inf if not specified (filter out behavior).
    """
    invalid_scores = {}
    for objective in objective_names:
        obj_cfg = cfg.landscape.objectives[objective]
        if for_bo:
            # For BO landscape: try bo_invalid_score, fall back to invalid_score, then -inf
            invalid_score = getattr(obj_cfg, "bo_invalid_score", None)
            if invalid_score is None:
                invalid_score = getattr(obj_cfg, "invalid_score", float("-inf"))
        else:
            invalid_score = getattr(obj_cfg, "invalid_score", float("-inf"))
        invalid_scores[objective] = invalid_score
    return invalid_scores


def build_thresholds(cfg, objective_names, landscape) -> Dict[str, float]:
    """
    Compute thresholds for each objective from config.

    Handles both explicit thresholds and top_prop-based threshold computation.

    Args:
        cfg: Hydra config (needs cfg.landscape.objectives).
        landscape: The landscape to compute thresholds for.

    Returns:
        Dictionary mapping objective names to their thresholds.

    Raises:
        ValueError: If neither threshold nor top_prop is specified,
                    or if both are specified,
                    or if top_prop is used with non-enumerable landscape.
    """
    # Parse thresholds for each objective
    thresholds = {}
    for objective in objective_names:
        threshold = getattr(cfg.landscape.objectives[objective], "threshold", None)
        top_prop = getattr(cfg.landscape.objectives[objective], "top_prop", None)
        if threshold is not None and top_prop is not None:
            raise ValueError(
                f"Specify either 'threshold' or 'top_prop' for {objective}, not both."
            )
        if top_prop is not None:
            if is_landscape_enumerable(landscape):
                log.info(
                    f"Computing threshold for {objective} at top {top_prop} proportion..."
                )
                sequence_scores = enumerate_sequences_and_score(
                    landscape, objective=objective
                )
                threshold = get_top_proportion_threshold(sequence_scores, top_prop)
                log.info(f"Computed threshold for {objective}: {threshold:.4f}")
            else:
                raise ValueError(
                    f"Top proportion threshold not supported for large landscapes, set to none for {objective}."
                )
        elif threshold is None:
            raise ValueError(
                f"Either 'threshold' or 'top_prop' must be specified for {objective}."
            )
        thresholds[objective] = threshold

    return thresholds
