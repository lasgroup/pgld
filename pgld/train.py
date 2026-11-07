"""
train.py

Main script for running single-objective policy gradient optimization experiments.
Uses Hydra for configuration management.
Make sure that the config only includes a single objective.
"""

import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

from config_schemas import build_training_config
from entropy import calculate_entropy, get_entropy_function
from landscapes import (
    MHCflurryLandscape,
    PLMLogLikelihoodLandscape,
    TFBindingLandscape,
    RiboswitchLandscape,
    enumerate_sequences_and_score,
    get_top_proportion_threshold,
    is_landscape_enumerable,
)
from rewards import get_reward_function
from policy_init import init_from_iupac_sequence
from policy_training import (
    train_policy,
    pooled_train_policy,
)
from fast_plm_training import fast_train_plm
from policies import (
    ArbitraryBaseOptimizer,
    BottleBasedOptimizer,
    FiniteBaseOptimizer,
    IupacBaseOptimizer,
    IupacPlusOptimizer,
    SlotConstrainedIupacOptimizer,
    try_different_temperatures,
)
from vis import (
    plot_arbitrary_optimizer_results,
    plot_edit_distance_histogram,
    plot_entropy_objective_history,
    plot_finite_optimizer_results,
    plot_landscape_histograms,
    plot_peptide_length_histogram,
    plot_reward_history,
    plot_amino_acid_policy_results,
    plot_sequence_diversity,
    plot_dna_and_peptide_diversity,
    print_oligo_ordering_template,
    print_theoretical_diversity,
    plot_theoretical_diversity_histogram,
)
from setup import build_invalid_scores, build_landscape, build_policy, build_thresholds
from library_size_estimator import print_total_library_diversity
from typing import Callable, Dict, List, Optional, Union


def run_visualization(
    policy,
    landscape,
    cfg: DictConfig,
    objective_names: List[str],
    thresholds: Union[Dict[str, float], float],
    reward_history: List[float],
    entropy_objective_history: List[float],
    vis_dir: str,
    alphabet: List[str],
    reward_type: str,
    training_cfg,
    verbose: bool = True,
) -> None:
    """Generate and save all visualizations to vis_dir."""
    # Suppress logging during intermediate visualization calls
    vis_logger = logging.getLogger("vis")
    original_level = vis_logger.level
    if not verbose:
        vis_logger.setLevel(logging.WARNING)

    try:
        if not os.path.exists(vis_dir):
            os.makedirs(vis_dir)
        if verbose:
            log.info(f"Saving visualizations to: {vis_dir}")

        # Plot training progress
        save_path = os.path.join(vis_dir, "reward_history.png")
        if reward_type == "average":
            reward_name = "Average Fitness"
        else:
            reward_name = "Proportion of Unique Hits"
            if training_cfg.pooling.enabled:
                reward_name += " (IS estimate)"
        plot_reward_history(
            reward_history, policy.name, save_path=save_path, reward_name=reward_name
        )

        # Plot entropy objective history if entropy was used during training
        if training_cfg.entropy.lambda_ > 0:
            save_path = os.path.join(vis_dir, "entropy_objective_history.png")
            plot_entropy_objective_history(
                entropy_objective_history, policy.name, save_path=save_path
            )

        # Plot policy-specific parameter visualizations
        if policy.name == "Arbitrary":
            save_path = os.path.join(vis_dir, "arbitrary_policy_params.png")
            plot_arbitrary_optimizer_results(policy, alphabet, save_path)
        elif isinstance(policy, BottleBasedOptimizer):
            save_path = os.path.join(
                vis_dir, f"{policy.name.lower()}_policy_params.png"
            )
            plot_finite_optimizer_results(policy, alphabet, save_path)

        # Plot amino acid probabilities for DNA→peptide landscapes only
        if (
            cfg.landscape.seq_scored == "peptide"
            and len(alphabet) == 4
            and set(alphabet) == {"A", "C", "G", "T"}
            and landscape.str_length % 3 == 0
        ):
            if verbose:
                log.info("Generating amino acid probability visualization...")
            save_path = os.path.join(
                vis_dir, f"{policy.name.lower()}_amino_acid_probs.png"
            )
            try:
                plot_amino_acid_policy_results(policy, alphabet, save_path)
            except Exception as e:
                log.warning(f"Failed to generate amino acid visualization: {e}")

        # Plot landscape and sampled sequence distributions
        plot_landscape_histograms(
            landscape,
            objective_names,
            thresholds,
            vis_dir,
            policy,
            cfg,
            cfg.landscape.name,
        )

        # Plot peptide length and edit distance histograms for DNA->peptide landscapes
        if cfg.landscape.seq_scored == "peptide":
            if verbose:
                log.info("Generating peptide analysis histograms...")
            sample_result = policy.sample(10000, greedy=True)
            sampled_sequences_str = [
                "".join([alphabet[c] for c in s]) for s in sample_result.sequences
            ]
            save_path = os.path.join(vis_dir, "peptide_length_histogram.png")
            plot_peptide_length_histogram(sampled_sequences_str, save_path)

            if hasattr(landscape, "wild_type") and landscape.wild_type is not None:
                save_path = os.path.join(vis_dir, "edit_distance_histogram.png")
                plot_edit_distance_histogram(
                    sampled_sequences_str, landscape.wild_type, save_path
                )

        # Plot sequence diversity for all landscapes
        if verbose:
            log.info("Generating sequence diversity visualizations...")
        sample_result = policy.sample(10000, greedy=True)
        sampled_sequences_str = [
            "".join([alphabet[c] for c in s]) for s in sample_result.sequences
        ]

        if cfg.landscape.seq_scored == "peptide":
            save_path = os.path.join(vis_dir, "sequence_diversity.png")
            plot_dna_and_peptide_diversity(
                sampled_sequences_str,
                save_path,
                "Policy Sampled Sequence Diversity",
                verbose=verbose,
            )
        else:
            save_path = os.path.join(vis_dir, "sequence_diversity.png")
            sequence_type = "DNA" if len(alphabet) == 4 else "sequence"
            plot_sequence_diversity(
                sampled_sequences_str,
                save_path,
                "Policy Sampled Sequence Diversity",
                sequence_type,
                verbose=verbose,
            )

        # Print oligo ordering template for bottle-based policies
        oligo_template_path = os.path.join(vis_dir, "oligo_ordering_template.txt")
        print_oligo_ordering_template(
            policy, log_to_file=oligo_template_path, verbose=verbose
        )

        # Print theoretical diversity (per-template)
        diversity_path = os.path.join(vis_dir, "theoretical_diversity.txt")
        diversity_dict = print_theoretical_diversity(
            policy,
            landscape.alphabet,
            is_peptide_landscape=(cfg.landscape.seq_scored == "peptide"),
            log_to_file=diversity_path,
            verbose=verbose,
        )

        # Plot theoretical diversity histogram
        if diversity_dict:
            plot_theoretical_diversity_histogram(
                diversity_dict,
                save_path=os.path.join(vis_dir, "theoretical_diversity_histogram.png"),
            )

        # Print total library diversity (union of all templates)
        print_total_library_diversity(
            policy,
            landscape.alphabet,
            is_peptide_landscape=(cfg.landscape.seq_scored == "peptide"),
            log_to_file=diversity_path,
            verbose=verbose,
        )
    finally:
        # Restore original logging level
        if not verbose:
            vis_logger.setLevel(original_level)


def make_vis_callback(
    policy,
    landscape,
    cfg: DictConfig,
    objective_names: List[str],
    thresholds: Union[Dict[str, float], float],
    alphabet: List[str],
    base_output_dir: str,
    reward_type: str,
    training_cfg,
) -> Callable[[List[float], List[float], Union[int, None]], None]:
    """Create visualization callback that captures necessary context."""

    def vis_callback(
        reward_history: List[float],
        entropy_history: List[float],
        step: Union[int, None] = None,
    ) -> None:
        # Intermediate steps (step != None) run quietly; final vis (step=None) is verbose
        verbose = step is None
        if step is not None:
            vis_dir = os.path.join(base_output_dir, f"vis_step_{step}")
            os.makedirs(vis_dir, exist_ok=True)
            torch.save(policy.state_dict(), os.path.join(vis_dir, "parameters.pth"))
        else:
            vis_dir = os.path.join(base_output_dir, cfg.vis.vis_dir)
        run_visualization(
            policy,
            landscape,
            cfg,
            objective_names,
            thresholds,
            reward_history,
            entropy_history,
            vis_dir,
            alphabet,
            reward_type,
            training_cfg,
            verbose=verbose,
        )

    return vis_callback


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("--- Configuration ---")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("---------------------\n")

    log.info("--- Landscape Config ---")
    log.info(OmegaConf.to_yaml(cfg.landscape))
    log.info("------------------------\n")

    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        log.info("Using CPU")

    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    objective_names = [k for k, v in cfg.landscape.objectives.items() if v is not None]
    invalid_scores = build_invalid_scores(cfg, objective_names)

    landscape = build_landscape(cfg, objective_names, device, invalid_scores)

    alphabet = landscape.alphabet
    alphabet_size = len(alphabet)

    thresholds = build_thresholds(cfg, objective_names, landscape)

    # Check for multi-objective without proper reward setup
    if len(objective_names) > 1:
        log.info(
            f"Multi-objective landscape detected with {len(objective_names)} objectives: {objective_names}"
        )
    else:
        # Single objective - extract the single threshold value
        threshold = thresholds[objective_names[0]]

    # 2. Set up the Policy (Agent) based on the config
    policy = build_policy(cfg, landscape, alphabet_size, device)

    # 3. Set up the Reward based on the config
    reward_type = getattr(cfg.training, "reward_type", "hits")
    per_sequence_rewards = getattr(cfg.training, "per_sequence_rewards", False)
    use_fast_plm = getattr(cfg.training, "fast_mode", False)

    if reward_type == "average" and landscape.is_peptide:
        # PLM landscapes store invalid scores in models config (landscape.invalid_sequence_scores)
        # rather than objectives config (invalid_scores), so check the right one
        scores_to_check = getattr(landscape, "invalid_sequence_scores", invalid_scores)
        for obj, score in scores_to_check.items():
            if np.isinf(score) and score < 0:
                raise ValueError(
                    f"Objective '{obj}' has invalid_score=-inf but reward_type='average'. "
                    "Set a finite invalid_score in the landscape config."
                )

    # unique_hits/unique_mean only work in fast mode and compute rewards internally
    if reward_type in ("unique_hits", "unique_mean") and not use_fast_plm:
        raise ValueError(f"{reward_type} reward_type requires fast_mode=true")

    # Fast mode computes rewards internally, so skip reward function creation
    if use_fast_plm:
        reward_function = None
    else:
        reward_function = get_reward_function(
            landscape,
            objective_names,
            thresholds,
            threshold if len(objective_names) == 1 else None,
            cfg.training.sampling.mc_samples,
            cfg.training.sampling.batch_size,
            reward_type,
            cfg.training.reward_aggregation_unit,
            per_sequence_rewards,
            uniqueness_type=cfg.training.pooling.uniqueness,
        )

    # 4. Build structured training configuration
    base_output_dir = HydraConfig.get().runtime.output_dir

    # Convert config to structured dataclass
    training_cfg = build_training_config(cfg.training)

    # Set eval thresholds at runtime
    training_cfg.evaluation.eval_thresholds = thresholds
    training_cfg.evaluation.eval_threshold = (
        threshold if len(objective_names) == 1 else None
    )

    # Fast mode validation
    if use_fast_plm:
        if not isinstance(landscape, PLMLogLikelihoodLandscape):
            raise ValueError("fast_mode requires a PLM landscape")
        if cfg.training.reward_type not in (
            "average",
            "hits",
            "unique_hits",
            "unique_mean",
        ):
            raise ValueError(
                "fast_mode requires reward_type in (average, hits, unique_hits, unique_mean)"
            )
        # unique_hits/unique_mean have diversity built-in, entropy only applies to average/hits
        if (
            cfg.training.reward_type not in ("unique_hits", "unique_mean")
            and cfg.training.entropy.lambda_ > 0
        ):
            if cfg.training.entropy.type not in ("marginal_gain", "exact"):
                raise ValueError(
                    "fast_mode requires entropy.type in (marginal_gain, exact)"
                )

    # Set up entropy function (fast mode handles entropy internally, so skip)
    entropy_function = None
    if training_cfg.entropy.lambda_ > 0 and not use_fast_plm:
        entropy_function = get_entropy_function(
            training_cfg.entropy.type,
            training_cfg.entropy.sequence_type,
            policy,
            training_cfg.entropy.baseline_per_template,
        )

    # Validate template-level reward aggregation requires pooled training
    if (
        not training_cfg.pooling.enabled
        and training_cfg.reward_aggregation_unit == "template"
    ):
        raise ValueError(
            "Template-level reward aggregation requires pooled=True. "
            "Set training.pooling.enabled=true or use reward_aggregation_unit=batch."
        )

    # Create visualization callback
    vis_callback = make_vis_callback(
        policy,
        landscape,
        cfg,
        objective_names,
        thresholds if len(objective_names) > 1 else threshold,
        alphabet,
        base_output_dir,
        reward_type,
        training_cfg,
    )

    # 5. Run the training loop
    if use_fast_plm:
        log.info("Using FAST PLM training path (vectorized)")
        unique_hits_N = getattr(cfg.training, "unique_hits_batch_size", None)
        # Extract invalid_score for unique_mean reward type
        invalid_score = None
        if reward_type == "unique_mean":
            invalid_score = landscape.invalid_sequence_scores[objective_names[0]]
        reward_history, entropy_objective_history = fast_train_plm(
            policy,
            landscape,
            training_cfg,
            base_output_dir,
            device,
            threshold=threshold,
            vis_callback=vis_callback,
            unique_hits_batch_size=unique_hits_N,
            invalid_score=invalid_score,
        )
    elif training_cfg.pooling.enabled:
        log.info("Using pooled training with landscape query subsampling")
        reward_history, entropy_objective_history = pooled_train_policy(
            policy,
            landscape,
            training_cfg,
            base_output_dir,
            reward_function,
            entropy_function,
            device,
            vis_callback=vis_callback,
        )
    else:
        reward_history, entropy_objective_history = train_policy(
            policy,
            landscape,
            training_cfg,
            base_output_dir,
            reward_function,
            device,
            vis_callback=vis_callback,
        )

    params_dir = os.path.join(base_output_dir, "parameters.pth")
    log.info(f"Saving policy parameters to: {params_dir}")
    torch.save(policy.state_dict(), params_dir)

    # 6. Generate and save final visualizations
    log.info(f"\nBase output directory: {base_output_dir}")
    vis_callback(reward_history, entropy_objective_history, step=None)

    if len(objective_names) == 1 and cfg.training.evaluation.extra_eval:
        log.info("Evaluating on batch size 120000:")
        try_different_temperatures(policy, landscape, [1], threshold)
    if policy.num_templates == 1 and isinstance(policy, ArbitraryBaseOptimizer):
        with torch.no_grad():
            log.info(f"Calculating entropy: {calculate_entropy(policy)}")


if __name__ == "__main__":
    main()
