"""Fast training path for PLM landscapes with mean reward and marginal entropy.

This module provides a vectorized, high-performance training loop specifically
optimized for PLM landscape evaluation with the following constraints:
- mc_samples=1
- batch_size % num_templates == 0 (stratified sampling)
- uniform_template_mix=True

"""

import logging
import time
from typing import Callable, Optional

import torch

from constants import AMINO_ACIDS, CODON_TABLE, DNA_BASES
from policies import SampleResult
from tensor_translation_gpu import (
    build_codon_lookup_gpu,
    translate_dna_to_aa_indices_gpu,
)

log = logging.getLogger(__name__)


def scatter_add_native(
    src: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Native PyTorch scatter_add."""
    out = torch.zeros(dim_size, device=src.device, dtype=src.dtype)
    return out.scatter_add_(0, index, src)


def fast_train_plm(
    policy,
    landscape,
    training_cfg,
    output_dir: str,
    device: str = "cuda",
    threshold: Optional[float] = None,
    vis_callback: Optional[Callable] = None,
    unique_hits_batch_size: Optional[int] = None,
    invalid_score: Optional[float] = None,
) -> tuple[list[float], list[float]]:
    """
    Ultra-fast training loop for PLM landscape.

    Args:
        policy: Policy optimizer (IupacBaseOptimizer or ArbitraryBaseOptimizer)
        landscape: PLMLogLikelihoodLandscape
        training_cfg: Training configuration dataclass
        output_dir: Directory for saving outputs
        device: Device to run on
        threshold: Fitness threshold for hits/unique_hits reward types
        vis_callback: Optional callback for periodic visualization
        unique_hits_batch_size: Hypothetical batch size N for unique_hits/unique_mean
        invalid_score: Score for invalid sequences (required for unique_mean)

    Returns:
        Tuple of (reward_history, entropy_history)

    Enforced constraints:
        - mc_samples=1
        - batch_size % num_templates == 0 (stratified sampling)
        - uniform_template_mix=True
    """
    batch_size = training_cfg.sampling.batch_size
    num_templates = policy.num_templates

    assert (
        training_cfg.sampling.mc_samples == 1
    ), f"fast_mode requires mc_samples=1, got {training_cfg.sampling.mc_samples}"
    assert batch_size % num_templates == 0, (
        f"fast_mode requires batch_size % num_templates == 0, "
        f"got {batch_size} % {num_templates}"
    )
    assert policy.uniform_template_mix, "fast_mode requires uniform_template_mix=True"

    samples_per_template = batch_size // num_templates
    reward_type = (
        training_cfg.reward_type
    )  # "average", "hits", "unique_hits", or "unique_mean"
    approx_type = training_cfg.unique_hits_approx  # "exp" or "pade"

    # Validate threshold/invalid_score for reward types
    if reward_type in ("hits", "unique_hits") and threshold is None:
        raise ValueError(f"{reward_type} reward requires threshold parameter")
    if reward_type == "unique_mean" and invalid_score is None:
        raise ValueError("unique_mean reward requires invalid_score parameter")

    if reward_type in ("unique_hits", "unique_mean"):
        effective_N = unique_hits_batch_size or DEFAULT_UNIQUE_HITS_BATCH_SIZE
        log.info(
            f"Fast PLM training: {num_templates} templates, "
            f"{samples_per_template} samples each, reward_type={reward_type}, "
            f"approx={approx_type}, N={effective_N}"
        )
    else:
        log.info(
            f"Fast PLM training: {num_templates} templates, "
            f"{samples_per_template} samples each, reward_type={reward_type}"
        )

    optimizer = torch.optim.Adam(policy.parameters(), lr=training_cfg.optimization.lr)
    # unique_hits/unique_mean have diversity built-in, so no entropy regularization needed
    entropy_lambda = (
        0.0
        if reward_type in ("unique_hits", "unique_mean")
        else training_cfg.entropy.lambda_
    )
    alpha = training_cfg.optimization.ema_alpha
    codon_lookup = build_codon_lookup_gpu(device)

    baseline_reward = torch.zeros(num_templates, device=device)
    baseline_entropy = torch.zeros(num_templates, device=device)

    reward_history = []
    entropy_history = []

    total_iters = training_cfg.optimization.iters

    for iteration in range(total_iters):
        iter_start = time.perf_counter()
        optimizer.zero_grad()

        sample_result = stratified_sample(policy, samples_per_template, device)

        # Time scoring (build_fast_cache includes landscape.score_tensor_batch)
        torch.cuda.synchronize() if device == "cuda" else None
        t_score_start = time.perf_counter()
        cache = build_fast_cache(
            sample_result,
            landscape,
            codon_lookup,
            device,
            num_templates,
            objective=landscape.active_objs[0],
        )
        torch.cuda.synchronize() if device == "cuda" else None
        t_score = time.perf_counter() - t_score_start

        # Compute reward based on type
        if reward_type in ("unique_hits", "unique_mean"):
            target_N = unique_hits_batch_size or DEFAULT_UNIQUE_HITS_BATCH_SIZE
            score_type = "hits" if reward_type == "unique_hits" else "continuous"
            reward_per_template, summary_stat, debug_info = compute_fast_unique_reward(
                cache,
                policy,
                num_templates,
                samples_per_template,
                device,
                target_batch_size=target_N,
                approx_type=approx_type,
                score_type=score_type,
                threshold=threshold,
                invalid_score=invalid_score,
            )
            hit_rate = summary_stat if reward_type == "unique_hits" else None
            mean_fitness = summary_stat if reward_type == "unique_mean" else None
        elif reward_type == "hits":
            reward_per_template, hit_rate = compute_fast_hits_reward(
                cache, threshold, num_templates
            )
            mean_fitness = None
        else:  # average
            reward_per_template = compute_fast_reward(cache, num_templates)
            hit_rate = None
            mean_fitness = None

        # Time entropy computation
        torch.cuda.synchronize() if device == "cuda" else None
        t_entropy_start = time.perf_counter()
        if entropy_lambda > 0:
            entropy_type = training_cfg.entropy.type
            if entropy_type == "exact":
                entropy_per_template, H_summary = compute_fast_exact_entropy(
                    cache, policy, num_templates, samples_per_template, device
                )
            else:  # marginal_gain
                entropy_per_template, H_summary = compute_fast_marginal_entropy(
                    cache, policy, num_templates, samples_per_template, device
                )
        else:
            entropy_per_template = torch.zeros(num_templates, device=device)
            H_summary = torch.tensor(0.0, device=device)
        torch.cuda.synchronize() if device == "cuda" else None
        t_entropy = time.perf_counter() - t_entropy_start

        # Policy gradient with baselines
        reward_advantage = reward_per_template - baseline_reward
        entropy_advantage = entropy_per_template - baseline_entropy

        log_probs = policy.get_log_probs(sample_result)
        template_idx = sample_result.mixture_indices

        # Reward REINFORCE on sequence log probs
        loss = -torch.sum(log_probs * reward_advantage[template_idx])

        # Batch log prob terms for bottle-based policies
        if (
            hasattr(policy, "get_batch_log_probs")
            and sample_result.bottle_indices is not None
        ):
            batch_log_probs = policy.get_batch_log_probs(sample_result, sum=False)
            # batch_log_probs is (num_templates,) for stratified sampling

            # Reward REINFORCE on batch log probs (same as original)
            loss += -torch.sum(batch_log_probs * reward_advantage)

            # Entropy REINFORCE on batch log probs + direct entropy gradient
            entropy_loss = torch.sum(
                batch_log_probs * entropy_advantage.detach() + entropy_advantage
            )
            loss -= entropy_lambda * entropy_loss
        else:
            # No batch log probs - just use direct entropy (cases like having one template)
            loss -= entropy_lambda * H_summary

        loss.backward()
        optimizer.step()

        # EMA baseline updates
        with torch.no_grad():
            baseline_reward = (
                alpha * baseline_reward + (1 - alpha) * reward_per_template.detach()
            )
            baseline_entropy = (
                alpha * baseline_entropy + (1 - alpha) * entropy_per_template.detach()
            )

        mean_reward = reward_per_template.mean().item()
        reward_history.append(mean_reward)
        entropy_history.append(H_summary.item())

        t_iter = time.perf_counter() - iter_start

        if iteration % 100 == 0:
            if reward_type == "unique_hits":
                log.info(
                    f"Iter {iteration:04d}/{total_iters}: "
                    f"Loss={loss.item():.2f}, "
                    f"Reward={mean_reward:.4f}, "
                    f"HitRate={hit_rate:.4f}, "
                    f"Hits={debug_info['n_hits']}, "
                    f"N*p_mix={debug_info['mean_Np_mix']:.2e}, "
                    f"Uniq={cache['n_unique']} | "
                    f"Time: {t_iter:.2f}s"
                )
            elif reward_type == "unique_mean":
                log.info(
                    f"Iter {iteration:04d}/{total_iters}: "
                    f"Loss={loss.item():.2f}, "
                    f"Reward={mean_reward:.4f}, "
                    f"MeanFitness={mean_fitness:.4f}, "
                    f"Valid={debug_info['n_valid']}, "
                    f"N*p_mix={debug_info['mean_Np_mix']:.2e}, "
                    f"Uniq={cache['n_unique']} | "
                    f"Time: {t_iter:.2f}s"
                )
            elif reward_type == "hits":
                log.info(
                    f"Iter {iteration:04d}/{total_iters}: "
                    f"Loss={loss.item():.2f}, "
                    f"HitRate={hit_rate:.4f}, "
                    f"Entropy={H_summary.item():.4f}, "
                    f"Uniq={cache['n_unique']} (valid={cache['n_valid']}/{cache['n_total']}) | "
                    f"Time: {t_iter:.2f}s (score={t_score:.2f}s, entropy={t_entropy:.2f}s)"
                )
            else:
                log.info(
                    f"Iter {iteration:04d}/{total_iters}: "
                    f"Loss={loss.item():.2f}, "
                    f"Reward={mean_reward:.4f}, "
                    f"Entropy={H_summary.item():.4f}, "
                    f"Uniq={cache['n_unique']} (valid={cache['n_valid']}/{cache['n_total']}) | "
                    f"Time: {t_iter:.2f}s (score={t_score:.2f}s, entropy={t_entropy:.2f}s)"
                )

        # Periodic visualization
        steps_til_vis = getattr(training_cfg.evaluation, "steps_til_vis", None)
        if vis_callback and steps_til_vis and (iteration + 1) % steps_til_vis == 0:
            log.info(f"Running visualization at step {iteration + 1}")
            vis_callback(reward_history, entropy_history, step=iteration + 1)

    return reward_history, entropy_history


def stratified_sample(policy, samples_per_template: int, device: str) -> SampleResult:
    """
    Sample exactly samples_per_template sequences from each template.

    Args:
        policy: Policy optimizer with get_probs() method
        samples_per_template: Number of samples per template
        device: Device to run on

    Returns:
        SampleResult with all templates represented equally
    """
    T = policy.num_templates

    is_bottle_based = hasattr(policy, "get_bottle_probs")

    if is_bottle_based:
        v_probs, phi_probs, _ = policy.get_probs()
        # Sample bottles once per template
        bottle_dist = torch.distributions.Categorical(probs=phi_probs)
        bottles_per_template = bottle_dist.sample()  # (T, L)

        # Get character probs from bottles: (T, L, alphabet_size)
        char_probs_per_template = v_probs[bottles_per_template]
    else:
        template_probs, _ = policy.get_probs()
        char_probs_per_template = template_probs  # (T, L, alphabet_size)
        bottles_per_template = None

    # Vectorized sampling for all templates at once
    # char_probs_per_template: (T, L, A)
    T, L, A = char_probs_per_template.shape

    # Expand for samples_per_template samples per template
    # Shape: (T, samples_per_template, L, A)
    expanded_probs = char_probs_per_template.unsqueeze(1).expand(
        T, samples_per_template, L, A
    )
    # Reshape to (T * samples_per_template, L, A)
    flat_probs = expanded_probs.reshape(T * samples_per_template, L, A)

    # Sample from each position
    char_dist = torch.distributions.Categorical(probs=flat_probs)
    sequences = char_dist.sample()  # (T * samples_per_template, L)

    # Create mixture indices: [0,0,...,0, 1,1,...,1, ..., T-1,T-1,...,T-1]
    mixture_indices = torch.repeat_interleave(
        torch.arange(T, device=device), samples_per_template
    )

    # Create bottle indices if applicable
    if bottles_per_template is not None:
        # Expand bottles to match samples
        bottle_indices = (
            bottles_per_template.unsqueeze(1)
            .expand(T, samples_per_template, L)
            .reshape(T * samples_per_template, L)
        )
    else:
        bottle_indices = None

    return SampleResult(
        sequences=sequences,
        mixture_indices=mixture_indices,
        bottle_indices=bottle_indices,
        selected_bottles=None,
    )


def build_fast_cache(
    sample_result: SampleResult,
    landscape,
    codon_lookup: torch.Tensor,
    device: str,
    num_templates: int,
    objective: str,
) -> dict:
    """Build tensor-based cache for stratified sampling."""
    sequences = sample_result.sequences  # (N, L)
    template_indices = sample_result.mixture_indices  # (N,)

    aa_indices, first_stop = translate_dna_to_aa_indices_gpu(sequences, codon_lookup)

    # Valid mask (no stop codons)
    valid_mask = first_stop == -1

    # Mask positions after stop codon
    num_codons = aa_indices.shape[1]
    positions = torch.arange(num_codons, device=device).unsqueeze(0)
    stops = first_stop.unsqueeze(1)
    pos_mask = (positions < stops) | (stops == -1)
    aa_indices_masked = torch.where(
        pos_mask, aa_indices, torch.tensor(-1, device=device)
    )

    # Deduplicate peptides
    unique_peptides, sample_to_peptide = torch.unique(
        aa_indices_masked, dim=0, return_inverse=True
    )
    n_total = sequences.shape[0]
    n_valid = valid_mask.sum().item()
    n_unique = unique_peptides.shape[0]
    log.debug(
        f"Sequences: {n_total} total, {n_valid} valid, {n_unique} unique peptides"
    )

    # Score unique peptides
    peptide_fitness = landscape.score_tensor_batch(unique_peptides, objective=objective)

    return {
        "sample_to_peptide": sample_to_peptide,  # (N,)
        "peptide_fitness": peptide_fitness,  # (U,)
        "template_indices": template_indices,  # (N,)
        "valid_mask": valid_mask,  # (N,)
        "aa_indices": aa_indices_masked,  # (N, num_codons)
        "sequences": sequences,  # (N, L)
        "bottle_indices": sample_result.bottle_indices,  # (N, L) or None
        "n_total": n_total,  # scalar
        "n_valid": n_valid,  # scalar
        "n_unique": n_unique,  # scalar
    }


def compute_fast_reward(cache: dict, num_templates: int) -> torch.Tensor:
    """Compute per-template reward using scatter operations."""
    sample_to_peptide = cache["sample_to_peptide"]
    template_indices = cache["template_indices"]
    peptide_fitness = cache["peptide_fitness"]

    # Get fitness for each sample
    sample_fitness = peptide_fitness[sample_to_peptide]

    # Aggregate by template - mean over ALL samples
    template_fitness_sum = scatter_add_native(
        sample_fitness, template_indices, num_templates
    )
    samples_per_template = scatter_add_native(
        torch.ones_like(sample_fitness), template_indices, num_templates
    )

    # Mean per template
    template_reward = template_fitness_sum / samples_per_template.clamp(min=1)

    return template_reward


def compute_fast_hits_reward(
    cache: dict,
    threshold: float,
    num_templates: int,
) -> tuple[torch.Tensor, float]:
    """
    Compute per-template hit rate (binary 0/1 reward for each sample).

    Args:
        cache: Sample cache from build_fast_cache
        threshold: Fitness threshold for counting as a hit
        num_templates: Number of templates T

    Returns:
        reward_per_template: (T,) mean hit rate per template
        overall_hit_rate: scalar, overall hit rate across all samples
    """
    sample_to_peptide = cache["sample_to_peptide"]  # (N,)
    template_indices = cache["template_indices"]  # (N,)
    peptide_fitness = cache["peptide_fitness"]  # (U,)
    device = template_indices.device

    # Get fitness for each sample
    sample_fitness = peptide_fitness[sample_to_peptide]  # (N,)

    # Binary hit reward: 1 if above threshold, 0 otherwise
    hit_reward = (sample_fitness > threshold).float()  # (N,)

    # Aggregate by template - mean hit rate
    template_hits_sum = scatter_add_native(hit_reward, template_indices, num_templates)
    samples_per_template = scatter_add_native(
        torch.ones_like(hit_reward), template_indices, num_templates
    )
    reward_per_template = template_hits_sum / samples_per_template.clamp(min=1)  # (T,)

    overall_hit_rate = hit_reward.mean().item()

    return reward_per_template, overall_hit_rate


# Default target batch size for unique_hits objective
DEFAULT_UNIQUE_HITS_BATCH_SIZE = 1_000_000_000


def compute_fast_unique_reward(
    cache: dict,
    policy,
    num_templates: int,
    samples_per_template: int,
    device: str,
    target_batch_size: int = DEFAULT_UNIQUE_HITS_BATCH_SIZE,
    approx_type: str = "exp",
    score_type: str = "hits",
    threshold: float = None,
    invalid_score: float = None,
) -> tuple[torch.Tensor, float, dict]:
    """
    Compute per-template reward with diversity-weighted scoring.

    Unified function for both unique_hits (binary) and unique_mean (continuous):
        R(s) = score(s) * (N/T) * approx(N * p(s|mixture))

    Two score_type options:
        - "hits": score=1 for hits (above threshold), 0 otherwise
        - "continuous": score=(fitness - invalid_score), uses all valid samples

    Two approx_type options for (1-p)^N:
        - "exp": exp(-Np) - standard exponential approximation
        - "pade": 1/(1+Np) - Padé approximant, bounded in (0,1], decays slower

    Args:
        cache: Sample cache from build_fast_cache
        policy: Policy with get_probs() method
        num_templates: Number of templates T
        samples_per_template: Samples per template K
        device: Device string
        target_batch_size: N, the hypothetical batch size
        approx_type: "exp" or "pade" approximation for uniqueness weighting
        score_type: "hits" (binary) or "continuous" (shifted fitness)
        threshold: Fitness threshold for hits (required if score_type="hits")
        invalid_score: Score for invalid sequences (required if score_type="continuous")

    Returns:
        reward_per_template: (T,) mean reward per template
        summary_stat: hit_rate for "hits", mean_fitness for "continuous"
        debug_info: dict with diagnostic values
    """
    T = num_templates
    N = target_batch_size
    template_indices = cache["template_indices"]  # (N_samples,)
    sample_to_peptide = cache["sample_to_peptide"]  # (N_samples,)
    peptide_fitness = cache["peptide_fitness"]  # (U,)
    valid_mask = cache["valid_mask"]  # (N_samples,)
    aa_indices = cache["aa_indices"]  # (N_samples, num_codons)

    sample_fitness = peptide_fitness[sample_to_peptide]  # (N_samples,)

    # Determine which samples to include and their scores
    if score_type == "hits":
        if threshold is None:
            raise ValueError("threshold required for score_type='hits'")
        include_mask = (sample_fitness > threshold) & valid_mask
        scores = torch.ones_like(sample_fitness)  # score=1 for hits
        summary_stat = include_mask.float().mean().item()  # hit_rate
        n_included = include_mask.sum().item()
        debug_key = "n_hits"
    else:  # continuous
        if invalid_score is None:
            raise ValueError("invalid_score required for score_type='continuous'")
        include_mask = valid_mask
        scores = sample_fitness - invalid_score  # shifted to be positive
        summary_stat = (
            sample_fitness[valid_mask].mean().item() if valid_mask.any() else 0.0
        )
        n_included = valid_mask.sum().item()
        debug_key = "n_valid"

    if not include_mask.any():
        debug_info = {debug_key: 0, "mean_p_mix": 0.0, "mean_Np_mix": 0.0}
        return torch.zeros(T, device=device), summary_stat, debug_info

    # Get included sequences
    include_indices = include_mask.nonzero(as_tuple=True)[0]
    included_aa_seqs = aa_indices[include_indices]  # (S, num_codons)
    included_templates = template_indices[include_indices]  # (S,)
    included_scores = scores[include_indices]  # (S,)

    # Get template AA probs for computing p(s|mixture)
    template_dna_probs = _get_template_dna_probs(
        cache, policy, T, samples_per_template, device
    )  # (T, L, 4)
    log_template_aa_probs = dna_probs_to_aa_probs_batch(
        template_dna_probs
    )  # (T, num_codons, 20)

    # Compute log p(seq | template) for all T x S pairs
    log_probs_all = compute_log_probs_all_pairs(
        log_template_aa_probs, included_aa_seqs
    )  # (T, S)

    # Compute p(seq | mixture) = (1/T) * Σ_t p(seq | t)
    log_mix_weight = torch.log(
        torch.tensor(1.0 / T, device=device, dtype=log_probs_all.dtype)
    )
    log_joint = log_probs_all + log_mix_weight  # (T, S)
    log_p_mixture = torch.logsumexp(log_joint, dim=0)  # (S,)
    p_mixture = torch.exp(log_p_mixture)  # (S,)

    mean_p_mix = p_mixture.mean().item()
    mean_Np_mix = (N * p_mixture).mean().item()

    # Uniqueness weight: (N/T) * approx(-N * p_mixture)
    N_over_T = N / T
    if approx_type == "pade":
        uniqueness_weight = N_over_T / (1 + N * p_mixture)  # (S,)
    else:
        log_N_over_T = torch.log(
            torch.tensor(N_over_T, device=device, dtype=p_mixture.dtype)
        )
        log_weight = log_N_over_T - N * p_mixture
        uniqueness_weight = torch.exp(log_weight.clamp(max=50))  # (S,)

    # Reward: score * uniqueness_weight
    reward_per_sample = included_scores * uniqueness_weight  # (S,)

    # Aggregate by template
    reward_sum = scatter_add_native(reward_per_sample, included_templates, T)  # (T,)
    reward_per_template = reward_sum / samples_per_template  # (T,)

    debug_info = {
        debug_key: n_included,
        "mean_p_mix": mean_p_mix,
        "mean_Np_mix": mean_Np_mix,
    }
    if score_type == "continuous":
        debug_info["mean_shifted_score"] = included_scores.mean().item()

    return reward_per_template, summary_stat, debug_info


def _get_template_dna_probs(
    cache: dict, policy, num_templates: int, samples_per_template: int, device: str
) -> torch.Tensor:
    """Extract DNA probs for each template using sampled bottles."""
    if hasattr(policy, "get_bottle_probs"):
        v_probs, _, _ = policy.get_probs()
        bottle_indices = cache["bottle_indices"]  # (N, L)
        template_indices = cache["template_indices"]  # (N,)

        # Stratified sampling: samples ordered [T0, T0, ..., T1, T1, ...]
        first_idx = torch.arange(num_templates, device=device) * samples_per_template

        # Sanity check: verify stratified ordering
        expected_templates = torch.arange(num_templates, device=device)
        assert (template_indices[first_idx] == expected_templates).all(), (
            f"Stratified sampling order violated: expected {expected_templates}, "
            f"got {template_indices[first_idx]}"
        )

        template_bottles = bottle_indices[first_idx]  # (T, L)
        return v_probs[template_bottles]  # (T, L, 4)
    else:
        template_dna_probs, _ = policy.get_probs()
        return template_dna_probs  # (T, L, 4)


def compute_fast_marginal_entropy(
    cache: dict,
    policy,
    num_templates: int,
    samples_per_template: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized marginal entropy gain computation.

    With stratified sampling, all templates are active and each has exactly
    samples_per_template samples. This simplifies the computation of
    marginal contributions.

    Returns:
        marginal_gain: (T,) per-template entropy marginal contribution
        H_full: scalar, full mixture entropy
    """
    valid_mask = cache["valid_mask"]  # (N,)
    template_indices = cache["template_indices"]  # (N,)
    aa_sequences = cache["aa_indices"]  # (N, num_codons)

    # Filter to valid samples only
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    valid_aa_sequences = aa_sequences[valid_indices]  # (N_valid, num_codons)
    valid_template_indices = template_indices[valid_indices]  # (N_valid,)

    T = num_templates
    N = len(valid_indices)

    if N == 0:
        return torch.zeros(T, device=device), torch.tensor(0.0, device=device)

    template_dna_probs = _get_template_dna_probs(
        cache, policy, num_templates, samples_per_template, device
    )  # (T, L, 4)
    log_template_aa_probs = dna_probs_to_aa_probs_batch(
        template_dna_probs
    )  # (T, num_codons, 20)

    # Compute log p(seq | template) for all T x N pairs
    log_probs_per_template = compute_log_probs_all_pairs(
        log_template_aa_probs, valid_aa_sequences
    )  # (T, N)

    # Add mixture weights (uniform, so log(1/T) for all)
    log_mix = torch.log(
        torch.tensor(1.0 / T, device=device, dtype=log_probs_per_template.dtype)
    )
    log_joint = log_probs_per_template + log_mix  # (T, N)

    # Full entropy: H = -E[log p(x)]
    log_marginal = torch.logsumexp(log_joint, dim=0)  # (N,)
    H_full = -log_marginal.mean()

    # Vectorized leave-one-out
    # For each template t, compute H without template t

    # Create (T, T, N) tensor where row t has template t masked out
    eye_mask = torch.eye(T, device=device, dtype=torch.bool).unsqueeze(-1)  # (T, T, 1)
    log_joint_exp = log_joint.unsqueeze(0).expand(T, -1, -1)  # (T, T, N)
    log_joint_masked = log_joint_exp.masked_fill(eye_mask, float("-inf"))  # (T, T, N)

    # Renormalize: log(1 - 1/T) = log((T-1)/T)
    log_renorm = torch.log(
        torch.tensor((T - 1) / T, device=device, dtype=log_probs_per_template.dtype)
    )
    log_marginal_without = (
        torch.logsumexp(log_joint_masked, dim=1) - log_renorm
    )  # (T, N)

    # For template t, average over samples NOT from template t
    # template_mask[t, i] = True if sample i is NOT from template t
    template_mask = valid_template_indices.unsqueeze(0) != torch.arange(
        T, device=device
    ).unsqueeze(
        1
    )  # (T, N)

    # Count valid samples per leave-one-out
    counts = template_mask.sum(dim=1).float()  # (T,)

    # Masked sum for H_without_t
    log_marginal_without_masked = log_marginal_without.masked_fill(~template_mask, 0.0)
    H_without_t = -log_marginal_without_masked.sum(dim=1) / counts.clamp(min=1)  # (T,)

    marginal_gain = H_full - H_without_t  # (T,)

    return marginal_gain, H_full


def compute_fast_exact_entropy(
    cache: dict,
    policy,
    num_templates: int,
    samples_per_template: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute exact entropy of each template's AA distribution in closed form.

    H(template_t) = -Σ_{pos} Σ_{aa} p(aa|t,pos) * log p(aa|t,pos)

    Returns:
        entropy_per_template: (T,) entropy of each template
        mean_entropy: scalar - mean over templates
    """
    template_dna_probs = _get_template_dna_probs(
        cache, policy, num_templates, samples_per_template, device
    )  # (T, L, 4)
    log_template_aa_probs = dna_probs_to_aa_probs_batch(
        template_dna_probs
    )  # (T, num_codons, 20)
    template_aa_probs = torch.exp(log_template_aa_probs)  # (T, num_codons, 20)

    # Entropy at each position: -Σ_aa p * log p
    pos_entropy = -(template_aa_probs * log_template_aa_probs).sum(
        dim=-1
    )  # (T, num_codons)

    # Total entropy per template
    entropy_per_template = pos_entropy.sum(dim=-1)  # (T,)
    mean_entropy = entropy_per_template.mean()

    return entropy_per_template, mean_entropy


def dna_probs_to_aa_probs_batch(dna_probs: torch.Tensor) -> torch.Tensor:
    """
    Batched conversion of DNA probs to AA probs.

    Args:
        dna_probs: (T, L, 4) DNA base probabilities

    Returns:
        (T, num_codons, 20) log amino acid probabilities
    """
    T, L, _ = dna_probs.shape
    num_codons = L // 3
    device = dna_probs.device
    dtype = dna_probs.dtype

    # Reshape to (T, num_codons, 3, 4)
    dna_probs = dna_probs[:, : num_codons * 3, :].reshape(T, num_codons, 3, 4)

    # Compute all 64 codon probabilities via einsum
    pos1 = dna_probs[:, :, 0, :]  # (T, num_codons, 4)
    pos2 = dna_probs[:, :, 1, :]
    pos3 = dna_probs[:, :, 2, :]

    codon_probs = torch.einsum(
        "tci,tcj,tck->tcijk", pos1, pos2, pos3
    )  # (T, num_codons, 4, 4, 4)
    codon_probs = codon_probs.reshape(T, num_codons, 64)  # (T, num_codons, 64)

    # Build codon to AA mapping (cached)
    codon_to_aa = get_codon_to_aa_mapping(device)  # (64,) indices 0-20

    # Scatter into amino acid indices
    aa_probs = torch.zeros(T, num_codons, 21, device=device, dtype=dtype)
    codon_to_aa_exp = codon_to_aa.view(1, 1, 64).expand(T, num_codons, -1)
    aa_probs.scatter_add_(2, codon_to_aa_exp, codon_probs)

    # Exclude stop codons (index 20) and take log
    aa_probs = aa_probs[:, :, :20]
    aa_probs = aa_probs / aa_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    return torch.log(aa_probs).clamp(min=-100)


_codon_to_aa_cache: Optional[torch.Tensor] = None


def get_codon_to_aa_mapping(device: torch.device) -> torch.Tensor:
    """Get cached codon to AA index mapping."""
    global _codon_to_aa_cache
    if _codon_to_aa_cache is None:
        aa_to_idx = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
        mapping = torch.zeros(64, dtype=torch.long)
        for i, b1 in enumerate(DNA_BASES):
            for j, b2 in enumerate(DNA_BASES):
                for k, b3 in enumerate(DNA_BASES):
                    codon = b1 + b2 + b3
                    codon_idx = i * 16 + j * 4 + k
                    mapping[codon_idx] = aa_to_idx[CODON_TABLE[codon]]
        _codon_to_aa_cache = mapping

    return _codon_to_aa_cache.to(device)


def compute_log_probs_all_pairs(
    log_template_probs: torch.Tensor, aa_sequences: torch.Tensor
) -> torch.Tensor:
    """
    Compute log p(seq | template) for all template-sequence pairs.

    Args:
        log_template_probs: (T, num_codons, 20) log AA probs per template
        aa_sequences: (N, num_codons) AA indices per sequence

    Returns:
        (T, N) log probabilities
    """
    T, num_codons, _ = log_template_probs.shape
    N = aa_sequences.shape[0]

    # Ensure indices are long type for gather
    aa_sequences = aa_sequences.long()

    # Expand for broadcasting
    log_probs_exp = log_template_probs.unsqueeze(1).expand(
        -1, N, -1, -1
    )  # (T, N, num_codons, 20)
    seqs_exp = (
        aa_sequences.unsqueeze(0).unsqueeze(-1).expand(T, -1, -1, 1)
    )  # (T, N, num_codons, 1)

    log_gathered = log_probs_exp.gather(3, seqs_exp.clamp(min=0)).squeeze(
        -1
    )  # (T, N, num_codons)

    # Mask invalid positions (where aa_sequences == -1)
    valid_pos = (aa_sequences >= 0).unsqueeze(0)  # (1, N, num_codons)
    log_gathered = log_gathered * valid_pos

    return log_gathered.sum(dim=-1)  # (T, N)
