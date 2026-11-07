"""
regenerate_vis.py

Script to regenerate visualizations for a saved training run.
Loads the saved parameters and configuration, then creates all visualizations.
"""

import argparse
import logging
import os
import sys
import torch
import numpy as np
import editdistance
from omegaconf import OmegaConf

logging.basicConfig(level=logging.INFO, format="%(message)s")

from landscapes import (
    dna_to_protein,
    enumerate_sequences_and_score,
    generate_uniform_random_sequences,
    get_top_proportion_threshold,
    is_landscape_enumerable,
)
from setup import build_landscape
from policies import (
    ArbitraryBaseOptimizer,
    BottleBasedOptimizer,
    FiniteBaseOptimizer,
    IupacBaseOptimizer,
    SlotConstrainedIupacOptimizer,
)
from vis import (
    plot_arbitrary_optimizer_results,
    plot_early_stop_comparison,
    plot_finite_optimizer_results,
    plot_landscape_histogram,
    plot_landscape_histograms,
    plot_likelihood_ecdf_comparison,
    plot_mmd_comparison,
    plot_witness_inlier_fraction,
    plot_bear_bayes_factor,
    plot_reward_history,
    plot_sampled_scores_histogram,
    plot_score_histogram,
    plot_peptide_length_histogram,
    plot_edit_distance_histogram,
    plot_edit_distance_overlay,
    plot_amino_acid_policy_results,
    plot_sequence_diversity,
    plot_dna_and_peptide_diversity,
    plot_tvd_per_position,
    plot_umap_sequence_space_grid,
    print_oligo_ordering_template,
    print_theoretical_diversity,
    plot_theoretical_diversity_histogram,
)
from mmd import (
    encode_sequences,
    compute_mmd_v_statistic,
    witness_inlier_fraction,
    WITNESS_ALPHA,
)
from bear import bear_two_sample_bayes_factor
from plm_utils import load_fasta
from constants import AMINO_ACID_ALPHABET
from library_size_estimator import print_total_library_diversity
from plot_config import apply_paper_style, PAPER_FIGSIZE


def load_config(run_dir: str):
    """Load the configuration from a training run."""
    cfg_path = os.path.join(run_dir, ".hydra", "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config file not found at {cfg_path}")
    return OmegaConf.load(cfg_path)


def get_batch_size(cfg):
    """Get batch_size from config, supporting both flat and nested config structures."""
    # Prefer top-level training.batch_size if it exists (even if falsy), otherwise fall back
    training_cfg = cfg.training
    if hasattr(training_cfg, "batch_size"):
        return training_cfg.batch_size
    if hasattr(training_cfg, "sampling") and hasattr(
        training_cfg.sampling, "batch_size"
    ):
        return training_cfg.sampling.batch_size
    raise AttributeError(
        "Could not determine batch_size from cfg.training or cfg.training.sampling"
    )


def load_policy(cfg, landscape, device):
    """Rebuild and load the policy from saved parameters."""
    alphabet_size = len(landscape.alphabet)
    batch_size = get_batch_size(cfg)

    if cfg.policy.name == "arbitrary":
        policy = ArbitraryBaseOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=batch_size,
            num_templates=cfg.policy.num_templates,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "finite":
        policy = FiniteBaseOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=batch_size,
            num_templates=cfg.policy.num_templates,
            num_bottles=cfg.policy.num_bottles,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "iupac":
        policy = IupacBaseOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=batch_size,
            num_templates=cfg.policy.num_templates,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", False),
        )
    elif cfg.policy.name == "slot_constrained_iupac":
        max_non_agct_slots = getattr(cfg.policy, "max_non_agct_slots", 4)
        policy = SlotConstrainedIupacOptimizer(
            str_length=landscape.str_length,
            alphabet_size=alphabet_size,
            batch_size=batch_size,
            num_templates=cfg.policy.num_templates,
            max_non_agct_slots=max_non_agct_slots,
            device=device,
            uniform_template_mix=getattr(cfg.policy, "uniform_template_mix", True),
        )
    else:
        raise ValueError(f"Unknown policy: {cfg.policy.name}")

    return policy


def _policy_peptides(policy, alphabet: list[str], n: int, greedy: bool = True) -> list[str]:
    """Sample n sequences from the policy and convert them to peptides."""
    res = policy.sample(n, greedy=greedy)
    dna = ["".join(alphabet[c] for c in s) for s in res.sequences]
    return [dna_to_protein(seq) for seq in dna]


def _collect_policy_pool(
    policy, alphabet: list[str], is_valid, n: int, max_draws: int
) -> list[str]:
    """Sample the policy until the pool holds at least n valid peptides; return the pool.

    'all' is the pool's first n and 'valid_only' its first n valid, so both derive from
    the same draw -- a fully-valid policy then gets identical all/valid populations
    instead of spurious sampling noise between them.
    """
    pool: list[str] = []
    n_valid = 0
    while n_valid < n and len(pool) < max_draws:
        batch = _policy_peptides(policy, alphabet, n)
        pool.extend(batch)
        n_valid += sum(1 for p in batch if is_valid(p))
    return pool


def regenerate_visualizations(
    run_dir: str,
    output_dir: str = None,
    checkpoint_dir: str = None,
    reference_fasta: str = None,
    max_vis_samples: int = 10000,
    var_synth_fasta: str = None,
    only_compare: bool = False,
    mmd_seq_count: int = 10000,
    paper_ready: bool = False,
    skip_mmd_witness: bool = False,
):
    """
    Regenerate all visualizations for a saved training run.

    Args:
        run_dir: Path to the training run directory (contains .hydra/config.yaml)
        output_dir: Custom output directory for visualizations (defaults to run_dir/vis_regenerated)
        checkpoint_dir: Directory containing parameters.pth (defaults to run_dir). Use for vis_step_N checkpoints.
        max_vis_samples: Max samples for policy sampling and FASTA subsampling (default: 10000).
        var_synth_fasta: Path to var_synth AA samples.fasta for likelihood ECDF comparison.
        only_compare: If True, skip standard visualizations and only run ECDF/UMAP comparisons.
        mmd_seq_count: Max sequences per set for MMD computation (default: 10000).
        paper_ready: If True, use paper-ready style and figure sizes for publication.
    """
    # Apply paper style if requested
    if paper_ready:
        apply_paper_style()
        print("Using paper-ready style")

    # Set figsize for plots (None uses function defaults)
    figsize = PAPER_FIGSIZE if paper_ready else None
    plot_ext = ".pdf" if paper_ready else ".png"

    print(f"Loading run from: {run_dir}")

    # Set device
    if torch.cuda.is_available():
        device = "cuda"
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Load configuration
    cfg = load_config(run_dir)
    print("Configuration loaded successfully")

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Rebuild landscape
    print(f"Rebuilding landscape: {cfg.landscape.name}")
    objective_names = list(cfg.landscape.objectives.keys())
    landscape = build_landscape(cfg, objective_names, device)
    alphabet = landscape.alphabet

    # Rebuild policy
    print(f"Rebuilding policy: {cfg.policy.name}")
    policy = load_policy(cfg, landscape, device)

    # Load saved parameters
    params_dir = checkpoint_dir or run_dir
    param_path = os.path.join(params_dir, "parameters.pth")
    if not os.path.exists(param_path):
        raise FileNotFoundError(f"Parameters file not found at {param_path}")

    print(f"Loading parameters from: {param_path}")
    policy.load_state_dict(torch.load(param_path, map_location=device))
    print("Parameters loaded successfully")

    # Set up output directory
    if output_dir is None:
        output_dir = os.path.join(run_dir, "vis_regenerated")
    else:
        # Remove trailing slash to prevent double slashes
        output_dir = output_dir.rstrip("/")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving visualizations to: {output_dir}")

    # Parse thresholds for each objective (matching train.py logic)
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
                print(
                    f"Computing threshold for {objective} at top {top_prop} proportion..."
                )
                sequence_scores = enumerate_sequences_and_score(
                    landscape, objective=objective
                )
                threshold = get_top_proportion_threshold(sequence_scores, top_prop)
                print(f"Computed threshold for {objective}: {threshold:.4f}")
            else:
                raise ValueError(
                    f"Top proportion threshold not supported for large landscapes, set to none for {objective}."
                )
        elif threshold is None:
            raise ValueError(
                f"Either 'threshold' or 'top_prop' must be specified for {objective}."
            )
        thresholds[objective] = threshold

    # Determine seq_scored (needed for multiple sections)
    seq_scored = cfg.landscape.get("seq_scored")
    if seq_scored is None:
        seq_scored = (
            "peptide"
            if cfg.landscape.name in ["mhc_flurry"]
            else "dna"
        )

    # Use max_vis_samples for visualizations, mmd_seq_count only for MMD computation
    vis_sample_count = max_vis_samples

    # Generate visualizations
    print("\nGenerating visualizations...")

    if not only_compare:
        # 1. Policy-specific parameter visualizations
        print("  - Policy parameter visualization...")
        if policy.name == "Arbitrary":
            save_path = os.path.join(output_dir, "arbitrary_policy_params.png")
            plot_arbitrary_optimizer_results(policy, alphabet, save_path)
            print(f"    Saved to: {save_path}")
        elif isinstance(policy, BottleBasedOptimizer):
            save_path = os.path.join(
                output_dir, f"{policy.name.lower()}_policy_params.png"
            )
            plot_finite_optimizer_results(policy, alphabet, save_path)
            print(f"    Saved to: {save_path}")

        # Print oligo ordering template and theoretical diversity
        print("  - Generating oligo ordering template and theoretical diversity...")
        oligo_template_path = os.path.join(output_dir, "oligo_ordering_template.txt")
        print_oligo_ordering_template(policy, log_to_file=oligo_template_path)
        diversity_path = os.path.join(output_dir, "theoretical_diversity.txt")
        diversity_dict = print_theoretical_diversity(
            policy,
            alphabet,
            is_peptide_landscape=(seq_scored == "peptide"),
            log_to_file=diversity_path,
        )

        # Plot theoretical diversity histogram
        if diversity_dict:
            plot_theoretical_diversity_histogram(
                diversity_dict,
                save_path=os.path.join(
                    output_dir, f"theoretical_diversity_histogram{plot_ext}"
                ),
                figsize=figsize,
                paper_ready=paper_ready,
            )

        # Print total library diversity (union of all templates)
        print_total_library_diversity(
            policy,
            alphabet,
            is_peptide_landscape=(seq_scored == "peptide"),
            log_to_file=diversity_path,
        )

        # Plot amino acid probabilities for DNA→peptide landscapes only (mhc)
        if (
            seq_scored == "peptide"
            and len(alphabet) == 4
            and set(alphabet) == {"A", "C", "G", "T"}
            and landscape.str_length % 3 == 0
        ):
            print("  - Generating amino acid probability visualization...")
            save_path = os.path.join(
                output_dir, f"{policy.name.lower()}_amino_acid_probs.png"
            )
            try:
                plot_amino_acid_policy_results(policy, alphabet, save_path)
                print(f"    Amino acid visualization saved to: {save_path}")
            except Exception as e:
                print(f"    Warning: Failed to generate amino acid visualization: {e}")

        # 2. Use unified histogram generation from train_utils (matching train.py exactly)
        print("  - Landscape and sampled sequence histograms...")
        plot_landscape_histograms(
            landscape,
            objective_names,
            thresholds if len(objective_names) > 1 else thresholds[objective_names[0]],
            output_dir,
            policy,
            cfg,
            cfg.landscape.name,
        )
        print("    Histograms generated successfully")

        # 3. Plot peptide length and edit distance histograms for DNA->peptide landscapes (matching train.py)
        if seq_scored == "peptide":
            print("  - Generating peptide analysis histograms...")
            # Sample a large batch for histogram generation
            sample_result = policy.sample(vis_sample_count, greedy=True)
            sampled_sequences_str = [
                "".join([alphabet[c] for c in s]) for s in sample_result.sequences
            ]

            # Plot peptide length distribution
            save_path = os.path.join(output_dir, "peptide_length_histogram.png")
            plot_peptide_length_histogram(sampled_sequences_str, save_path)
            print(f"    Peptide length histogram saved to: {save_path}")

            # Plot edit distance distribution if wild-type is specified
            if hasattr(landscape, "wild_type") and landscape.wild_type is not None:
                save_path = os.path.join(
                    output_dir, f"edit_distance_histogram{plot_ext}"
                )
                plot_edit_distance_histogram(
                    sampled_sequences_str,
                    landscape.wild_type,
                    save_path,
                    figsize=figsize,
                    paper_ready=paper_ready,
                )
                print(f"    Edit distance histogram saved to: {save_path}")

        # 4. Plot sequence diversity for all landscapes
        print("  - Generating sequence diversity visualizations...")
        sample_result = policy.sample(vis_sample_count, greedy=True)
        sampled_sequences_str = [
            "".join([alphabet[c] for c in s]) for s in sample_result.sequences
        ]

        if seq_scored == "peptide":
            # For DNA->peptide landscapes, show both DNA and peptide diversity
            save_path = os.path.join(output_dir, "sequence_diversity.png")
            plot_dna_and_peptide_diversity(
                sampled_sequences_str, save_path, "Policy Sampled Sequence Diversity"
            )
        else:
            # For other landscapes, show just DNA/sequence diversity
            save_path = os.path.join(output_dir, "sequence_diversity.png")
            sequence_type = "DNA" if len(alphabet) == 4 else "sequence"
            plot_sequence_diversity(
                sampled_sequences_str,
                save_path,
                "Policy Sampled Sequence Diversity",
                sequence_type,
            )
        print("    Sequence diversity visualizations generated successfully")

        # 5. TVD per-position plot (PLM only, requires reference FASTA)
        if reference_fasta and seq_scored == "peptide":
            print("  - Generating TVD per-position plot...")

            # Sample sequences and convert to peptides
            tvd_sample_result = policy.sample(vis_sample_count, greedy=True)
            tvd_dna_seqs = [
                "".join([alphabet[c] for c in s]) for s in tvd_sample_result.sequences
            ]
            tvd_peptide_seqs = [dna_to_protein(seq) for seq in tvd_dna_seqs]
            # Filter out empty peptides (from early stops)
            tvd_peptide_seqs = [p for p in tvd_peptide_seqs if p]

            # Load var_synth sequences if provided
            var_synth_seqs_for_tvd = None
            if var_synth_fasta:
                var_synth_seqs_for_tvd = load_fasta(var_synth_fasta)
                if len(var_synth_seqs_for_tvd) > max_vis_samples:
                    var_synth_seqs_for_tvd = var_synth_seqs_for_tvd[:max_vis_samples]

            save_path = os.path.join(output_dir, f"tvd_per_position{plot_ext}")
            try:
                plot_tvd_per_position(
                    tvd_peptide_seqs,
                    reference_fasta,
                    save_path,
                    title="Per-Position Error",
                    max_samples=max_vis_samples,
                    var_synth_seqs=var_synth_seqs_for_tvd,
                    figsize=figsize,
                    paper_ready=paper_ready,
                )
                print(f"    TVD per-position plot saved to: {save_path}")
            except Exception as e:
                print(f"    Warning: Failed to generate TVD plot: {e}")
        elif reference_fasta and seq_scored != "peptide":
            print(
                f"  - Skipping TVD plot (only supported for peptide landscapes, got seq_scored={seq_scored})"
            )

    # 6. Likelihood ECDF comparison (PLM only, requires var_synth FASTA)
    if var_synth_fasta and seq_scored == "peptide":
        print("  - Generating likelihood ECDF comparison...")

        # Sample from policy and convert to peptides
        ecdf_sample_result = policy.sample(vis_sample_count, greedy=True)
        ecdf_dna_seqs = [
            "".join([alphabet[c] for c in s]) for s in ecdf_sample_result.sequences
        ]
        ecdf_peptide_seqs = [dna_to_protein(seq) for seq in ecdf_dna_seqs]

        # Load var_synth sequences (subsample if needed)
        var_synth_sequences = load_fasta(var_synth_fasta)
        if len(var_synth_sequences) > max_vis_samples:
            var_synth_sequences = var_synth_sequences[:max_vis_samples]

        # Get expected peptide length from policy
        num_codons = landscape.str_length // 3

        # Score sequences once, then derive three plots via masking
        objective = landscape.active_objs[0]
        orig_max_edit_dist = getattr(landscape, "max_edit_distance", None)
        invalid_score = landscape.invalid_sequence_scores[objective]

        # Disable the edit-distance constraint for scoring; it is applied manually for plot 3
        if hasattr(landscape, "max_edit_distance"):
            landscape.max_edit_distance = None

        # All peptides (no filtering)
        policy_peptides_all = ecdf_peptide_seqs[:max_vis_samples]
        var_synth_peptides_all = var_synth_sequences[:max_vis_samples]

        # Build validity masks (correct length, no stop codons)
        # Note: use bool() to avoid numpy string array issue when empty strings are present
        policy_valid_mask = np.array(
            [
                bool(p) and len(p) == num_codons and "*" not in p
                for p in policy_peptides_all
            ],
            dtype=bool,
        )
        var_synth_valid_mask = np.array(
            [
                bool(s) and len(s) == num_codons and "*" not in s
                for s in var_synth_peptides_all
            ],
            dtype=bool,
        )

        # Initialize all scores to invalid_score
        policy_scores_all = np.full(
            len(policy_peptides_all), invalid_score, dtype=np.float32
        )
        var_synth_scores_all = np.full(
            len(var_synth_peptides_all), invalid_score, dtype=np.float32
        )

        # Score valid sequences using score_tensor_batch (expects AA indices)
        policy_peptides_valid = [
            p for p, m in zip(policy_peptides_all, policy_valid_mask) if m
        ]
        if policy_peptides_valid:
            policy_indices = torch.tensor(
                [
                    [AMINO_ACID_ALPHABET[aa] for aa in seq]
                    for seq in policy_peptides_valid
                ],
                dtype=torch.long,
                device=device,
            )
            valid_scores = (
                landscape.score_tensor_batch(policy_indices, objective=objective)
                .cpu()
                .numpy()
            )
            policy_scores_all[policy_valid_mask] = valid_scores

        var_synth_peptides_valid = [
            p for p, m in zip(var_synth_peptides_all, var_synth_valid_mask) if m
        ]
        if var_synth_peptides_valid:
            var_synth_indices = torch.tensor(
                [
                    [AMINO_ACID_ALPHABET[aa] for aa in seq]
                    for seq in var_synth_peptides_valid
                ],
                dtype=torch.long,
                device=device,
            )
            valid_scores = (
                landscape.score_tensor_batch(var_synth_indices, objective=objective)
                .cpu()
                .numpy()
            )
            var_synth_scores_all[var_synth_valid_mask] = valid_scores

        # Restore max_edit_distance
        if hasattr(landscape, "max_edit_distance"):
            landscape.max_edit_distance = orig_max_edit_dist

        # Get wild-type score if available
        wt_score = None
        wt = None
        if hasattr(landscape, "wild_type") and landscape.wild_type:
            wt = landscape.wild_type
            wt_indices = torch.tensor(
                [[AMINO_ACID_ALPHABET[aa] for aa in wt]],
                dtype=torch.long,
                device=device,
            )
            wt_score = landscape.score_tensor_batch(
                wt_indices, objective=objective
            ).item()

        # Score PLM samples (reference FASTA) if provided - these are already valid peptides
        plm_sample_scores = None
        ref_peptides = None
        if reference_fasta:
            ref_sequences = load_fasta(reference_fasta)
            ref_peptides = [
                s for s in ref_sequences if len(s) == num_codons and "*" not in s
            ][:max_vis_samples]
            if ref_peptides:
                ref_indices = torch.tensor(
                    [[AMINO_ACID_ALPHABET[aa] for aa in seq] for seq in ref_peptides],
                    dtype=torch.long,
                    device=device,
                )
                plm_sample_scores = (
                    landscape.score_tensor_batch(ref_indices, objective=objective)
                    .cpu()
                    .numpy()
                )

        if not policy_valid_mask.any() or not var_synth_valid_mask.any():
            print(
                f"    Warning: Not enough valid sequences "
                f"(policy={policy_valid_mask.sum()}, var_synth={var_synth_valid_mask.sum()})"
            )
        else:
            # ===== First ECDF: valid sequences only (correct length, no stops) =====
            policy_scores = policy_scores_all[policy_valid_mask]
            var_synth_scores = var_synth_scores_all[var_synth_valid_mask]

            save_path = os.path.join(
                output_dir, f"likelihood_ecdf_comparison{plot_ext}"
            )
            plot_likelihood_ecdf_comparison(
                policy_scores=policy_scores,
                var_synth_scores=var_synth_scores,
                save_path=save_path,
                title="Log-Likelihood CDF",
                wild_type_score=wt_score,
                plm_sample_scores=plm_sample_scores,
                figsize=figsize,
                paper_ready=paper_ready,
            )
            print(f"    ECDF comparison saved to: {save_path}")
            n_plm = len(plm_sample_scores) if plm_sample_scores is not None else 0
            print(
                f"    (PGLD: {policy_valid_mask.sum()}, "
                f"Var-Synth: {var_synth_valid_mask.sum()}, "
                f"PLM Samples: {n_plm})"
            )

            # ===== Second ECDF: valid + within edit distance constraint =====
            if orig_max_edit_dist is not None and wt is not None:
                # Further filter by edit distance (policy_peptides_valid, var_synth_peptides_valid already computed above)
                policy_edit_mask = np.array(
                    [
                        editdistance.eval(p, wt) <= orig_max_edit_dist
                        for p in policy_peptides_valid
                    ]
                )
                var_synth_edit_mask = np.array(
                    [
                        editdistance.eval(p, wt) <= orig_max_edit_dist
                        for p in var_synth_peptides_valid
                    ]
                )

                policy_scores_valid = policy_scores[policy_edit_mask]
                var_synth_scores_valid = var_synth_scores[var_synth_edit_mask]

                # Filter PLM samples by edit distance if present
                plm_scores_valid = None
                if plm_sample_scores is not None and ref_peptides:
                    plm_edit_mask = np.array(
                        [
                            editdistance.eval(p, wt) <= orig_max_edit_dist
                            for p in ref_peptides
                        ]
                    )
                    plm_scores_valid = plm_sample_scores[plm_edit_mask]

                save_path_valid = os.path.join(
                    output_dir, f"likelihood_ecdf_comparison_valid_only{plot_ext}"
                )
                plot_likelihood_ecdf_comparison(
                    policy_scores=policy_scores_valid,
                    var_synth_scores=var_synth_scores_valid,
                    save_path=save_path_valid,
                    title="Log-Likelihood CDF",
                    wild_type_score=wt_score,
                    plm_sample_scores=plm_scores_valid,
                    figsize=figsize,
                    paper_ready=paper_ready,
                )
                print(f"    ECDF (valid edit dist only) saved to: {save_path_valid}")
                n_plm_valid = (
                    len(plm_scores_valid) if plm_scores_valid is not None else 0
                )
                print(
                    f"    (PGLD: {policy_edit_mask.sum()}, "
                    f"Var-Synth: {var_synth_edit_mask.sum()}, "
                    f"PLM Samples: {n_plm_valid})"
                )

            # ===== Third ECDF: ALL sequences (early stoppers, edit distance violators) =====
            if plm_sample_scores is not None:
                # Copy scores and apply invalid_score to edit distance violators
                policy_scores_all_adj = policy_scores_all.copy()
                var_synth_scores_all_adj = var_synth_scores_all.copy()

                if orig_max_edit_dist is not None and wt is not None:
                    for i, p in enumerate(policy_peptides_all):
                        if (
                            p
                            and len(p) == len(wt)
                            and editdistance.eval(p, wt) > orig_max_edit_dist
                        ):
                            policy_scores_all_adj[i] = invalid_score
                    for i, p in enumerate(var_synth_peptides_all):
                        if (
                            p
                            and len(p) == len(wt)
                            and editdistance.eval(p, wt) > orig_max_edit_dist
                        ):
                            var_synth_scores_all_adj[i] = invalid_score

                # Set xlim based on reference scores to keep plot readable
                ref_min, ref_max = plm_sample_scores.min(), plm_sample_scores.max()
                ref_range = ref_max - ref_min
                xlim_all = (ref_min - 0.05 * ref_range, ref_max + 0.05 * ref_range)

                save_path_all = os.path.join(
                    output_dir, f"likelihood_ecdf_comparison_all_seqs{plot_ext}"
                )
                plot_likelihood_ecdf_comparison(
                    policy_scores=policy_scores_all_adj,
                    var_synth_scores=var_synth_scores_all_adj,
                    save_path=save_path_all,
                    title="Log-Likelihood CDF",
                    wild_type_score=wt_score,
                    plm_sample_scores=plm_sample_scores,
                    figsize=figsize,
                    paper_ready=paper_ready,
                    xlim=xlim_all,
                )
                print(f"    ECDF (all seqs incl. invalid) saved to: {save_path_all}")
                print(
                    f"    (PGLD: {len(policy_peptides_all)}, "
                    f"Var-Synth: {len(var_synth_peptides_all)}, "
                    f"PLM Samples: {len(plm_sample_scores)})"
                )

            # Plot overlayed edit distance histogram (policy vs var-synth vs reference)
            if wt is not None:
                overlay_path = os.path.join(
                    output_dir, f"edit_distance_overlay{plot_ext}"
                )
                plot_edit_distance_overlay(
                    policy_sequences=ecdf_dna_seqs[:max_vis_samples],
                    var_synth_sequences=var_synth_peptides_valid,
                    wild_type=wt,
                    save_path=overlay_path,
                    reference_sequences=ref_peptides if reference_fasta else None,
                    figsize=figsize,
                    paper_ready=paper_ready,
                )
                print(f"    Edit distance overlay saved to: {overlay_path}")

    elif var_synth_fasta and seq_scored != "peptide":
        print("  - Skipping ECDF comparison (only supported for peptide landscapes)")

    # 7. UMAP sequence space visualization (PLM only, requires reference_fasta)
    if reference_fasta and seq_scored == "peptide":
        print("  - Generating UMAP sequence space plots...")
        MAX_UMAP_REF_SAMPLES = 100000

        # Load raw sequences - filtering handled by plot function via expected_length
        # Subsample var_synth and policy to max_vis_samples to avoid wasting compute on transform
        umap_ref_seqs = load_fasta(reference_fasta)[:MAX_UMAP_REF_SAMPLES]
        umap_var_seqs = (
            load_fasta(var_synth_fasta)[:max_vis_samples] if var_synth_fasta else None
        )

        # Policy: sample and convert DNA->peptide
        umap_sample = policy.sample(vis_sample_count, greedy=True)
        umap_pol_dna = [
            "".join([alphabet[c] for c in s]) for s in umap_sample.sequences
        ]
        umap_pol_seqs = [dna_to_protein(seq) for seq in umap_pol_dna][:max_vis_samples]

        try:
            plot_umap_sequence_space_grid(
                reference_seqs=umap_ref_seqs,
                var_synth_seqs=umap_var_seqs,
                policy_seqs=umap_pol_seqs,
                output_dir=output_dir,
                max_plot_samples=max_vis_samples,
                expected_length=num_codons,  # filter to full-length, no stop codons
                figsize=figsize,
                plot_ext=plot_ext,
                paper_ready=paper_ready,
            )
        except Exception as e:
            # UMAP is an optional viz (needs the umap package); don't let it block the
            # MMD/witness/BEAR metrics that follow.
            print(f"    Warning: skipping UMAP plots ({e})")

    # 8. MMD + witness-function comparison (all sequences and valid-only)
    should_compute_mmd = (
        var_synth_fasta
        and reference_fasta
        and seq_scored == "peptide"
        and mmd_seq_count > 0
    )
    if should_compute_mmd:
        num_codons = landscape.str_length // 3
        wt = getattr(landscape, "wild_type", None)
        max_ed = getattr(landscape, "max_edit_distance", None)

        def is_valid(pep: str) -> bool:
            """Full length, no stop codon, and (if constrained) within the WT ball."""
            if not pep or len(pep) != num_codons or "*" in pep:
                return False
            if max_ed is not None and wt:
                return editdistance.eval(pep, wt) <= max_ed
            return True

        ref_full = load_fasta(reference_fasta)
        var_full = load_fasta(var_synth_fasta)
        max_draws = 50 * mmd_seq_count  # policy resampling cap
        # One policy draw feeds both populations so a fully-valid policy gets identical
        # all/valid sets rather than sampling noise between them.
        pol_pool = _collect_policy_pool(policy, alphabet, is_valid, mmd_seq_count, max_draws)

        # "all": first mmd_seq_count raw sequences (invalids padded in the kernel).
        # "valid_only": first mmd_seq_count that pass is_valid.
        populations = {
            "all": (
                ref_full[:mmd_seq_count],
                var_full[:mmd_seq_count],
                pol_pool[:mmd_seq_count],
            ),
            "valid_only": (
                [s for s in ref_full if is_valid(s)][:mmd_seq_count],
                [s for s in var_full if is_valid(s)][:mmd_seq_count],
                [p for p in pol_pool if is_valid(p)][:mmd_seq_count],
            ),
        }

        for pop_label, (refs, vars_, pols) in populations.items():
            suffix = "" if pop_label == "all" else "_valid_only"
            print(
                f"  - MMD + witness ({pop_label}): "
                f"{len(refs)} ref, {len(vars_)} var-synth, {len(pols)} policy"
            )
            if min(len(refs), len(vars_), len(pols)) < 100:
                print(f"    Warning: too few sequences for '{pop_label}', skipping")
                continue
            if (
                pop_label == "valid_only"
                and min(len(refs), len(vars_), len(pols)) < mmd_seq_count
            ):
                print(
                    f"    Warning: fewer than {mmd_seq_count} valid sequences "
                    f"(ref={len(refs)}, var-synth={len(vars_)}, policy={len(pols)})"
                )

            # MMD and the witness function are the O(n^2) cost; skip them (keep the
            # cheap BEAR test below) when skip_mmd_witness is set.
            if not skip_mmd_witness:
                # Pad all three sets to a shared width so the Hamming kernel can broadcast.
                width = max(len(s) for s in refs + vars_ + pols)
                ref_enc = encode_sequences(refs, max_len=width)
                var_enc = encode_sequences(vars_, max_len=width)
                pol_enc = encode_sequences(pols, max_len=width)

                print(f"    Computing MMD² ({pop_label})...")
                mmd_policy_val = compute_mmd_v_statistic(pol_enc, ref_enc)
                mmd_var_synth_val = compute_mmd_v_statistic(var_enc, ref_enc)
                plot_mmd_comparison(
                    mmd_policy_val,
                    mmd_var_synth_val,
                    os.path.join(output_dir, f"mmd_comparison{suffix}{plot_ext}"),
                    figsize=figsize,
                    paper_ready=paper_ready,
                )
                print(
                    f"    MMD²(PGLD,Ref)={mmd_policy_val:.6f}  "
                    f"MMD²(Var-Synth,Ref)={mmd_var_synth_val:.6f}"
                )

                print(f"    Computing witness inlier fraction ({pop_label})...")
                frac_policy, _ = witness_inlier_fraction(pol_enc, ref_enc)
                frac_var_synth, _ = witness_inlier_fraction(var_enc, ref_enc)
                plot_witness_inlier_fraction(
                    frac_policy,
                    frac_var_synth,
                    os.path.join(output_dir, f"witness_inlier_fraction{suffix}{plot_ext}"),
                    ideal=1 - WITNESS_ALPHA,
                    figsize=figsize,
                    paper_ready=paper_ready,
                )
                print(
                    f"    Clearly-realistic: PGLD={100*frac_policy:.1f}%  "
                    f"Var-Synth={100*frac_var_synth:.1f}%"
                )

            print(f"    Computing BEAR two-sample Bayes factor ({pop_label})...")
            # Cap the lag at the peptide length so variable-length invalid sequences in
            # the 'all' set can't blow up the k-mer context count (and RAM).
            bf_policy = bear_two_sample_bayes_factor(pols, refs, max_lag=num_codons)[
                "bayes_factor_per_sequence"
            ]
            bf_var_synth = bear_two_sample_bayes_factor(vars_, refs, max_lag=num_codons)[
                "bayes_factor_per_sequence"
            ]
            plot_bear_bayes_factor(
                bf_policy,
                bf_var_synth,
                os.path.join(output_dir, f"bear_bayes_factor{suffix}{plot_ext}"),
                figsize=figsize,
                paper_ready=paper_ready,
            )
            print(
                f"    BEAR BF/seq: PGLD={bf_policy:.3f}  Var-Synth={bf_var_synth:.3f}"
            )

    # 9. Early stop percentage comparison (requires var_synth)
    if var_synth_fasta and seq_scored == "peptide":
        print("  - Computing early stop percentages...")
        num_codons = landscape.str_length // 3

        # Load var_synth sequences
        var_seqs = load_fasta(var_synth_fasta)[:vis_sample_count]

        # Sample from policy and convert to peptides
        early_stop_sample = policy.sample(vis_sample_count, greedy=True)
        pol_dna = [
            "".join([alphabet[c] for c in s]) for s in early_stop_sample.sequences
        ]
        pol_seqs = [dna_to_protein(seq) for seq in pol_dna]

        # Early stop = has "*" or length < expected
        def is_early_stop(seq: str) -> bool:
            return "*" in seq or len(seq) < num_codons

        n_early_policy = sum(1 for s in pol_seqs if is_early_stop(s))
        n_early_var = sum(1 for s in var_seqs if is_early_stop(s))

        pct_policy = 100.0 * n_early_policy / len(pol_seqs)
        pct_var_synth = 100.0 * n_early_var / len(var_seqs)

        save_path = os.path.join(output_dir, f"early_stop_comparison{plot_ext}")
        plot_early_stop_comparison(
            pct_policy,
            pct_var_synth,
            save_path,
            figsize=figsize,
            paper_ready=paper_ready,
        )
        print(f"    PGLD: {pct_policy:.1f}% ({n_early_policy}/{len(pol_seqs)})")
        print(f"    Var-Synth: {pct_var_synth:.1f}% ({n_early_var}/{len(var_seqs)})")
        print(f"    Saved to: {save_path}")

    print(f"\n✓ All visualizations regenerated successfully!")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate visualizations for a saved training run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to the training run directory (e.g., outputs/2025-11-13/12-06-56_iupac)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional custom output directory for visualizations (defaults to run_dir/vis_regenerated)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory containing parameters.pth (defaults to run-dir). Use for vis_step_N checkpoints.",
    )
    parser.add_argument(
        "--reference-fasta",
        type=str,
        default=None,
        help="FASTA file for reference AA distribution (enables TVD per-position plot for PLM runs).",
    )
    parser.add_argument(
        "--max-vis-samples",
        type=int,
        default=10000,
        help="Max samples for policy sampling and FASTA subsampling (default: 10000).",
    )
    parser.add_argument(
        "--var-synth-fasta",
        type=str,
        default=None,
        help="Path to var_synth AA samples.fasta for likelihood ECDF comparison.",
    )
    parser.add_argument(
        "--only-compare",
        action="store_true",
        help="Skip standard visualizations; only run ECDF/UMAP comparisons using FASTA files.",
    )
    parser.add_argument(
        "--mmd-seq-count",
        type=int,
        default=10000,
        help="Max sequences per set for MMD computation (default: 10000).",
    )
    parser.add_argument(
        "--paper-ready",
        action="store_true",
        help="Use paper-ready style and figure sizes (3 per row tiling).",
    )
    parser.add_argument(
        "--skip-mmd-witness",
        action="store_true",
        help="Skip the O(n^2) MMD and witness computations; still run BEAR + everything else.",
    )

    args = parser.parse_args()

    try:
        regenerate_visualizations(
            args.run_dir,
            args.output_dir,
            args.checkpoint_dir,
            args.reference_fasta,
            args.max_vis_samples,
            args.var_synth_fasta,
            args.only_compare,
            args.mmd_seq_count,
            args.paper_ready,
            args.skip_mmd_witness,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
