"""
policy_training.py

Implements the training loop for policy objects.
Includes a version with and without cache.
"""

import logging

import torch

from config_schemas import TrainingConfig
from entropy import GradientEstimateEntropy
from policies import BottleBasedOptimizer
from train_utils_policies import (
    compute_policy_gradient_loss,
    log_peptide_stats,
    run_extra_eval,
    save_checkpoint,
    PooledLandscapeCache,
    compute_reward_and_advantage,
)

log = logging.getLogger(__name__)


def train_policy(
    policy,
    landscape,
    training_cfg: TrainingConfig,
    output_dir: str,
    reward_function,
    device: str = "cpu",
    vis_callback=None,
):
    """
    A general training loop for any policy.

    For a more optimized version that queries the landscape on a subset of unique
    sequences, see pooled_train_policy.

    Args:
        policy: The policy to train.
        landscape: The environment.
        training_cfg: Structured training configuration.
        output_dir: Directory for saving checkpoints.
        reward_function: The reward to use for training.
        device: torch device.
        vis_callback: Optional callback for periodic visualization.

    Returns:
        Tuple of (reward_history, entropy_objective_history) lists.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=training_cfg.optimization.lr)
    reward_history = []
    entropy_objective_history = []
    alphabet = landscape.alphabet

    baseline_reward = 0.0
    baseline_entropy = 0.0
    alpha = training_cfg.optimization.ema_alpha
    entropy_lambda = training_cfg.entropy.lambda_
    entropy_type = training_cfg.entropy.type
    mc_samples = training_cfg.sampling.mc_samples
    batch_size = training_cfg.sampling.batch_size
    extra_eval = training_cfg.evaluation.extra_eval
    n_templates_considered = training_cfg.entropy.n_templates

    # train_policy only supports DNA-level uniqueness
    if training_cfg.pooling.uniqueness != "dna":
        raise NotImplementedError(
            "train_policy only supports 'dna' uniqueness. For peptide-level uniqueness, use pooled_train_policy."
        )

    if training_cfg.evaluation.verbose:
        log.info(f"Starting training for {policy.name} policy...")

    for i in range(training_cfg.optimization.iters):
        optimizer.zero_grad()
        total_loss = 0.0
        total_reward = 0.0
        total_entropy_reward = 0.0
        total_entropy_objective = 0.0

        if (
            entropy_lambda > 0
            and isinstance(policy, BottleBasedOptimizer)
            and entropy_type == "gradient_estimate"
        ):
            raise NotImplementedError(
                "train_policy does not support gradient_estimate entropy for BottleBasedOptimizer policies "
                "(e.g. IUPAC, Finite). Use entropy_type='marginal_gain', 'exact', or 'monte_carlo' instead."
            )
        # The first if clause isn't suitable for BottleBasedOptimizer because it will sample all mc_samples together
        # with one call to the policy.sample. This will make the bottle selection across mc samples dependent.
        if (
            mc_samples > 1
            and batch_size == 1
            and not (isinstance(policy, BottleBasedOptimizer))
        ):
            sample_result = policy.sample(mc_samples)
            loss, reward_val, entropy_loss, entropy_reward = (
                compute_policy_gradient_loss(
                    sample_result,
                    reward_function,
                    landscape,
                    baseline_reward,
                    policy,
                    training_cfg.entropy,
                    alphabet,
                    device,
                    mc_samples,
                    baseline_entropy,
                    n_templates_considered,
                )
            )
            total_loss = loss
            total_reward = reward_val
            total_entropy_reward = entropy_reward
            total_entropy_objective = -entropy_loss.item()
        else:
            for j in range(mc_samples):
                sample_result = policy.sample()
                loss, reward_val, entropy_loss, entropy_reward = (
                    compute_policy_gradient_loss(
                        sample_result,
                        reward_function,
                        landscape,
                        baseline_reward,
                        policy,
                        training_cfg.entropy,
                        alphabet,
                        device,
                        1,
                        baseline_entropy,
                        n_templates_considered,
                    )
                )
                total_loss += loss
                total_reward += reward_val
                total_entropy_reward += entropy_reward
                total_entropy_objective += -entropy_loss.item()

        total_loss /= mc_samples
        total_loss.backward()
        optimizer.step()

        total_reward /= mc_samples
        total_entropy_reward /= mc_samples
        baseline_reward = alpha * baseline_reward + (1 - alpha) * total_reward
        baseline_entropy = alpha * baseline_entropy + (1 - alpha) * total_entropy_reward
        reward_history.append(total_reward)
        entropy_objective_history.append(total_entropy_objective / mc_samples)

        if i % 200 == 0:
            if training_cfg.evaluation.checkpointing:
                save_checkpoint(policy, output_dir, i)

            log.info(
                f"Iter {i:04d} ({policy.name}): Loss={total_loss.item():.2f}, Reward={total_reward:.2f}, Baseline={baseline_reward:.2f}, Entropy={-entropy_loss.item():.2f} Entropy Baseline={baseline_entropy:.2f}"
            )

            if hasattr(landscape, "min_peptide_len"):
                log_peptide_stats(policy, landscape, alphabet)

        if (
            i % 1000 == 0
            and extra_eval
            and not training_cfg.evaluation.eval_only_at_end
        ):
            run_extra_eval(policy, landscape, training_cfg, output_dir, i)

        # Periodic visualization
        steps_til_vis = getattr(training_cfg.evaluation, "steps_til_vis", None)
        if vis_callback and steps_til_vis and (i + 1) % steps_til_vis == 0:
            log.info(f"Running visualization at step {i + 1}")
            vis_callback(reward_history, entropy_objective_history, step=i + 1)

    # Final eval after training completes
    if extra_eval:
        run_extra_eval(
            policy, landscape, training_cfg, output_dir, training_cfg.optimization.iters
        )

    if training_cfg.evaluation.verbose:
        log.info(f"Training finished for {policy.name} policy.")
    return reward_history, entropy_objective_history


def pooled_train_policy(
    policy,
    landscape,
    training_cfg: TrainingConfig,
    output_dir: str,
    reward_function,
    entropy_function,
    device: str = "cpu",
    vis_callback=None,
):
    """
    Training loop with pooled landscape queries across MC samples.

    Subsamples unique sequences to query the landscape less frequently.

    Args:
        policy: The policy to train.
        landscape: The environment.
        training_cfg: Structured training configuration.
        output_dir: Directory for saving checkpoints.
        reward_function: Reward function with compute_with_cache support.
        entropy_function: Entropy function (or None if entropy disabled).
        device: torch device.
        vis_callback: Optional callback for periodic visualization.

    Returns:
        Tuple of (reward_history, entropy_objective_history) lists.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=training_cfg.optimization.lr)
    reward_history = []
    entropy_objective_history = []
    alphabet = landscape.alphabet

    alpha = training_cfg.optimization.ema_alpha
    entropy_lambda = training_cfg.entropy.lambda_
    entropy_sequence_type = training_cfg.entropy.sequence_type
    entropy_baseline_per_template = (
        entropy_function is not None
        and training_cfg.entropy.baseline_per_template
        and hasattr(entropy_function, "baseline_per_template")
    )
    mc_samples = training_cfg.sampling.mc_samples
    batch_size = training_cfg.sampling.batch_size
    extra_eval = training_cfg.evaluation.extra_eval
    landscape_queries_per_batch = training_cfg.pooling.queries_per_batch
    uniqueness = training_cfg.pooling.uniqueness
    if entropy_lambda > 0 and uniqueness != entropy_sequence_type:
        raise ValueError(
            f"Cache ({uniqueness}) and Entropy ({entropy_sequence_type}) sequence type mismatch."
        )
    if (
        entropy_lambda > 0
        and isinstance(policy, BottleBasedOptimizer)
        and isinstance(entropy_function, GradientEstimateEntropy)
    ):
        raise NotImplementedError(
            "pooled_train_policy does not support GradientEstimateEntropy for BottleBasedOptimizer policies "
            "(e.g. IUPAC, Finite). Use entropy_type='marginal_gain', 'exact', or 'monte_carlo' instead."
        )
    use_gpu = training_cfg.pooling.gpu_accelerated
    reward_aggregation_unit = training_cfg.reward_aggregation_unit

    if training_cfg.evaluation.verbose:
        log.info(f"Starting pooled training for {policy.name} policy...")
        log.info(f"Landscape queries per batch: {landscape_queries_per_batch}")
        log.info(f"Uniqueness type: {uniqueness}")
        if use_gpu:
            log.info(f"Using GPU-accelerated cache building")

    # set up the baselines
    if reward_aggregation_unit == "batch":
        baseline_reward = torch.tensor(0.0, device=policy.device)
    elif reward_aggregation_unit == "template":
        baseline_reward = torch.zeros(policy.num_templates, device=device)
    else:
        raise ValueError(f"Unknown aggregation unit {reward_aggregation_unit}")

    if entropy_baseline_per_template:
        baseline_entropy = torch.zeros(policy.num_templates, device=device)
    else:
        baseline_entropy = torch.tensor(0.0, device=device)

    cache = None
    sample_results = None
    for i in range(training_cfg.optimization.iters):
        optimizer.zero_grad()

        # Must delete old cache and sample_results before constructing new ones.
        # Python evaluates the RHS before reassigning, so without explicit deletion,
        # both old and new objects exist simultaneously during construction.
        if cache is not None:
            del cache
        if sample_results is not None:
            del sample_results
        torch.cuda.empty_cache()

        sample_results = [policy.sample(batch_size) for _ in range(mc_samples)]

        cache = PooledLandscapeCache(
            sample_results,
            alphabet,
            policy,
            landscape,
            landscape_queries_per_batch,
            uniqueness,
            use_gpu,
            reward_aggregation_unit,
        )

        total_loss = 0.0
        total_reward = torch.zeros_like(baseline_reward, device=device)
        total_entropy_reward = torch.zeros_like(baseline_entropy, device=device)
        total_entropy_objective = 0.0

        for batch_idx, sample_result in enumerate(sample_results):
            batch_info = cache.get_batch_info(batch_idx)
            reward_unit = batch_info.get("reward_unit", None)
            unit_idx = batch_info.get("unit_idx", None)
            if hasattr(reward_function, "compute_with_cache"):
                reward = reward_function.compute_with_cache(batch_info)
            else:
                sequences_str = batch_info["all_sequences"]
                reward = reward_function(sequences_str, landscape)

            if baseline_reward.dim() == 0:
                baseline_reward_active = baseline_reward
            else:
                baseline_reward_active = baseline_reward[unit_idx]

            advantage, reward_val = compute_reward_and_advantage(
                reward, baseline_reward_active, device, reward_unit
            )

            log_probs = policy.get_log_probs(sample_result)
            loss = -torch.sum(log_probs * advantage)

            if hasattr(policy, "get_batch_log_probs"):
                batch_log_probs = policy.get_batch_log_probs(
                    sample_result, sum=(reward_aggregation_unit == "batch")
                )
                if reward_aggregation_unit == "batch":
                    advantage_per_unit = advantage
                else:
                    advantage_per_unit = reward_val - baseline_reward_active

                loss += -torch.sum(batch_log_probs * advantage_per_unit)

            if entropy_lambda > 0:
                if hasattr(entropy_function, "compute_with_cache"):
                    entropy, entropy_baseline_update = (
                        entropy_function.compute_with_cache(
                            policy, batch_info, baseline_entropy
                        )
                    )
                else:
                    entropy, entropy_baseline_update = entropy_function(
                        policy, landscape, sample_result, log_probs, baseline_entropy
                    )

                loss -= entropy * entropy_lambda
            else:
                entropy = torch.tensor(0.0, device=device)
                entropy_baseline_update = torch.zeros_like(
                    total_entropy_reward, device=device
                )

            total_loss += loss
            if isinstance(reward_val, torch.Tensor):
                total_reward[unit_idx] += reward_val  # (n_active_units,)
            else:
                total_reward += reward_val
            total_entropy_reward += entropy_baseline_update
            total_entropy_objective += entropy.item()

        total_loss /= mc_samples
        total_loss.backward()
        optimizer.step()

        total_reward /= mc_samples
        total_entropy_reward /= mc_samples
        baseline_reward = alpha * baseline_reward + (1 - alpha) * total_reward
        baseline_entropy = alpha * baseline_entropy + (1 - alpha) * total_entropy_reward
        reward_history.append(total_reward.sum().item())
        entropy_objective_history.append(total_entropy_objective / mc_samples)

        if i % 200 == 0:
            if training_cfg.evaluation.checkpointing:
                save_checkpoint(policy, output_dir, i)

            log.info(
                f"Iter {i:04d} ({policy.name}): Loss={total_loss.item():.2f}, "
                f"Reward={total_reward.sum().item():.2f}, Baseline={baseline_reward.sum().item():.2f}, "
                f"Entropy={entropy.item():.2f}, Entropy Baseline={baseline_entropy.mean().item():.2f}, Queries={cache.total_queries}"
            )

            if hasattr(landscape, "min_peptide_len"):
                log_peptide_stats(policy, landscape, alphabet)

        if (
            i % 1000 == 0
            and extra_eval
            and not training_cfg.evaluation.eval_only_at_end
        ):
            run_extra_eval(policy, landscape, training_cfg, output_dir, i)

        # Periodic visualization
        steps_til_vis = getattr(training_cfg.evaluation, "steps_til_vis", None)
        if vis_callback and steps_til_vis and (i + 1) % steps_til_vis == 0:
            log.info(f"Running visualization at step {i + 1}")
            vis_callback(reward_history, entropy_objective_history, step=i + 1)

    # Final eval after training completes
    if extra_eval:
        run_extra_eval(
            policy, landscape, training_cfg, output_dir, training_cfg.optimization.iters
        )

    if training_cfg.evaluation.verbose:
        log.info(f"Pooled training finished for {policy.name} policy.")

    return reward_history, entropy_objective_history
