"""
eval_fixed_library.py

Evaluate a fixed IUPAC library on a landscape without any training.
"""

import logging
import os

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

from landscapes import (
    MHCflurryLandscape,
    TFBindingLandscape,
    RiboswitchLandscape,
    enumerate_sequences_and_score,
    generate_uniform_random_sequences,
    get_top_proportion_threshold,
    is_landscape_enumerable,
    hit_above_threshold,
    hit_above_threshold_all,
)
from policies import FixedIupacPolicy, evaluate_on_batch
from vis import (
    plot_landscape_histogram,
    plot_sampled_scores_histogram,
    plot_score_histogram,
    plot_peptide_length_histogram,
    plot_edit_distance_histogram,
    plot_sequence_diversity,
    plot_dna_and_peptide_diversity,
    plot_amino_acid_policy_results,
    plot_arbitrary_optimizer_results,
    print_theoretical_diversity,
)
from library_size_estimator import print_total_library_diversity


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("--- Configuration ---")
    log.info(OmegaConf.to_yaml(cfg))
    log.info("---------------------\n")

    if torch.cuda.is_available():
        device = "cuda"
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        log.info("Using CPU")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    objective_names = list(cfg.landscape.objectives.keys())

    # Load landscape (same logic as train.py)
    if cfg.landscape.name == "tf_binding":
        landscape = TFBindingLandscape(
            str_length=cfg.landscape.str_length, objectives=objective_names
        )
    elif cfg.landscape.name == "mhc_flurry":
        landscape = MHCflurryLandscape(
            str_length=cfg.landscape.str_length, objectives=objective_names
        )
    elif cfg.landscape.name == "riboswitch":
        landscape = RiboswitchLandscape(
            str_length=cfg.landscape.str_length,
            objectives=objective_names,
            device=device,
            oracle_batch_size=cfg.landscape.oracle_batch_size,
            oracle_dir=cfg.landscape.oracle_dir,
        )
    else:
        raise ValueError(f"Unknown landscape name: {cfg.landscape.name}")

    alphabet = landscape.alphabet

    # Get IUPAC template from config
    if not hasattr(cfg, "iupac_template") or not cfg.iupac_template:
        raise ValueError(
            "Must specify 'iupac_template' in config (e.g., iupac_template='NNNNNN')"
        )

    iupac_template = cfg.iupac_template
    log.info(f"\nEvaluating fixed IUPAC library: {iupac_template}")
    log.info(f"Template length: {len(iupac_template)}")
    log.info(f"Landscape sequence length: {landscape.str_length}")

    if len(iupac_template) != landscape.str_length:
        raise ValueError(
            f"IUPAC template length ({len(iupac_template)}) must match "
            f"landscape sequence length ({landscape.str_length})"
        )

    # Create fixed policy
    policy = FixedIupacPolicy(iupac_template, alphabet)

    # Validate that each objective has a threshold
    for objective_name in objective_names:
        threshold = getattr(cfg.landscape.objectives[objective_name], "threshold", None)
        top_prop = getattr(cfg.landscape.objectives[objective_name], "top_prop", None)
        if threshold is not None and top_prop is not None:
            raise ValueError(
                f"Specify either 'threshold' or 'top_prop' for {objective_name}, not both."
            )
        if threshold is None and top_prop is None:
            raise ValueError(
                f"Either 'threshold' or 'top_prop' must be specified for {objective_name}."
            )
        if top_prop is not None:
            raise ValueError(
                "top_prop not yet supported for multi-objective evaluation. Use fixed thresholds."
            )

        log.info(f"{objective_name} threshold: {threshold}")

    # Evaluate for each objective
    eval_batch_size = cfg.eval_batch_size
    log.info(f"\nEvaluating on batch size {eval_batch_size}:")

    for objective_name in objective_names:
        obj_threshold = cfg.landscape.objectives[objective_name].threshold
        log.info(
            f"\n=== Evaluating {objective_name} objective (threshold: {obj_threshold}) ==="
        )

        sample_result = policy.sample(eval_batch_size, greedy=True)
        sequences = sample_result.sequences
        sequences_str = ["".join([landscape.alphabet[c] for c in s]) for s in sequences]

        unique_hits = hit_above_threshold(
            landscape, sequences_str, obj_threshold, objective=objective_name
        )
        all_hits = hit_above_threshold_all(
            landscape, sequences_str, obj_threshold, objective=objective_name
        )

        log.info(f"{objective_name} unique hits: {unique_hits}")
        log.info(f"{objective_name} all hits: {all_hits}")
        log.info(f"{objective_name} hit rate: {all_hits / eval_batch_size:.4f}")

    # Setup output directory
    hydra_cfg = HydraConfig.get()
    base_output_dir = hydra_cfg.runtime.output_dir
    vis_dir = os.path.join(base_output_dir, cfg.vis.vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    log.info(f"\nSaving visualizations to: {vis_dir}")

    # Generate visualizations for each objective
    vis_batch_size = cfg.vis_batch_size
    for objective_name in objective_names:
        obj_threshold = cfg.landscape.objectives[objective_name].threshold
        log.info(f"\nGenerating visualizations for {objective_name}...")

        if is_landscape_enumerable(landscape):
            save_path = os.path.join(
                vis_dir, f"landscape_histogram_{objective_name}.png"
            )
            plot_landscape_histogram(landscape, obj_threshold, save_path)
        else:
            log.info(
                f"Landscape too large to enumerate. Plotting MC histogram of uniform random sequences for {objective_name}."
            )
            random_seqs = generate_uniform_random_sequences(
                alphabet, landscape.str_length, vis_batch_size
            )
            random_scores = landscape.get_fitness_values(
                random_seqs, objective=objective_name
            )
            save_path = os.path.join(
                vis_dir, f"uniform_random_sequences_histogram_{objective_name}.png"
            )
            plot_score_histogram(
                random_scores,
                save_path,
                f"uniform random sequences histogram ({objective_name})",
            )

        save_path = os.path.join(
            vis_dir, f"sampled_scores_histogram_{objective_name}.png"
        )
        plot_sampled_scores_histogram(
            policy,
            landscape,
            num_sequences=vis_batch_size,
            save_path=save_path,
            objective=objective_name,
        )

    # Plot DNA policy visualization
    log.info("Generating DNA policy visualization...")
    save_path = os.path.join(vis_dir, "fixed_iupac_policy_params.png")
    plot_arbitrary_optimizer_results(policy, alphabet, save_path)

    # Print theoretical diversity (per-template)
    diversity_path = os.path.join(vis_dir, "theoretical_diversity.txt")
    print_theoretical_diversity(
        policy,
        alphabet,
        is_peptide_landscape=(cfg.landscape.seq_scored == "peptide"),
        log_to_file=diversity_path,
    )

    # Print total library diversity (union of all templates)
    print_total_library_diversity(
        policy,
        alphabet,
        is_peptide_landscape=(cfg.landscape.seq_scored == "peptide"),
        log_to_file=diversity_path,
    )

    # Plot peptide length and edit distance histograms for DNA->peptide landscapes
    if cfg.landscape.seq_scored == "peptide":
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

    # Plot amino acid probabilities for DNA→peptide landscapes only (mhc)
    if (
        cfg.landscape.seq_scored == "peptide"
        and len(alphabet) == 4
        and set(alphabet) == {"A", "C", "G", "T"}
        and landscape.str_length % 3 == 0
    ):
        log.info("Generating amino acid probability visualization...")
        save_path = os.path.join(vis_dir, "fixed_library_amino_acid_probs.png")
        try:
            plot_amino_acid_policy_results(policy, alphabet, save_path)
        except Exception as e:
            log.warning(f"Failed to generate amino acid visualization: {e}")

    # Plot sequence diversity
    log.info("Generating sequence diversity visualizations...")
    sample_result = policy.sample(10000, greedy=True)
    sampled_sequences_str = [
        "".join([alphabet[c] for c in s]) for s in sample_result.sequences
    ]

    if cfg.landscape.seq_scored == "peptide":
        save_path = os.path.join(vis_dir, "sequence_diversity.png")
        plot_dna_and_peptide_diversity(
            sampled_sequences_str, save_path, "Fixed Library Sequence Diversity"
        )
    else:
        save_path = os.path.join(vis_dir, "sequence_diversity.png")
        sequence_type = "DNA" if len(alphabet) == 4 else "sequence"
        plot_sequence_diversity(
            sampled_sequences_str,
            save_path,
            "Fixed Library Sequence Diversity",
            sequence_type,
        )

    # Save summary
    summary_path = os.path.join(base_output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Fixed IUPAC Library Evaluation\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"IUPAC Template: {iupac_template}\n")
        f.write(f"Landscape: {cfg.landscape.name}\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Evaluation batch size: {eval_batch_size}\n\n")
        f.write(f"Results:\n")
        f.write(f"  Unique hits: {unique_hits}\n")
        f.write(f"  All hits: {all_hits}\n")
        f.write(f"  Hit rate: {all_hits / eval_batch_size:.4f}\n")

    log.info(f"\nSummary saved to: {summary_path}")
    log.info("Evaluation complete!")


if __name__ == "__main__":
    main()
