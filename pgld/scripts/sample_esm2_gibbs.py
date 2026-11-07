#!/usr/bin/env python
"""Sample sequences from ESM2 using Gibbs sampling with streaming output.

Loads antibody context from pgld/conf/antibody/{name}.yaml.
Streams sequences to disk as they're generated (no large buffer in memory).

Usage:
    python pgld/scripts/sample_esm2_gibbs.py --antibody c05 --total-steps 10000
    python pgld/scripts/sample_esm2_gibbs.py --antibody trastuzumab --init-seq WGGDGFYAMDY
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add pgld to path for imports
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from antibody_config import AntibodyConfig, load_antibody_config
from constants import AMINO_ACIDS, AMINO_ACID_ALPHABET
from scripts.eval_aa_policy import build_plm_landscape, plot_score_histogram


def gibbs_sample_streaming(
    model,
    alphabet,
    left_context: str,
    right_context: str,
    init_seq: str,
    batch_size: int,
    total_steps: int,
    eval_every: int,
    save_every: int,
    burn_in: int,
    temperature: float,
    device: str,
    output_path: str,
    scoring_model: str = "ESM2",
    ab_cfg: AntibodyConfig | None = None,
    max_edit_distance: int | None = None,
    wild_type: str | None = None,
    verbose: bool = False,
) -> int:
    """
    Run Gibbs sampling with streaming output to disk.

    Args:
        model: ESM2 model for Gibbs sampling.
        alphabet: ESM2 alphabet.
        left_context: Fixed left flanking sequence.
        right_context: Fixed right flanking sequence.
        init_seq: Initial sequence for variable region.
        batch_size: Number of parallel Gibbs chains.
        total_steps: Total Gibbs steps to run (after burn-in).
        eval_every: Steps between evaluations.
        save_every: Steps between writing sequences to disk.
        burn_in: Number of burn-in steps before collecting samples.
        temperature: Sampling temperature.
        device: Torch device string.
        output_path: Path for output fasta file (sequences streamed here).
        scoring_model: Model to use for scoring.
        ab_cfg: Antibody config for building scoring landscape.
        max_edit_distance: Max allowed edit distance from wild_type (None = unconstrained).
        wild_type: Reference sequence for edit distance (required if max_edit_distance set).
        verbose: If True, print progress bars and eval stats.

    Returns:
        Total number of sequences written.
    """
    from esme.alphabet import tokenize  # Heavy import, keep local

    seq_len = len(init_seq)

    # Validate edit distance constraints
    if max_edit_distance is not None:
        if wild_type is None:
            raise ValueError("wild_type must be provided when max_edit_distance is set")
        if len(wild_type) != seq_len:
            raise ValueError(f"wild_type length {len(wild_type)} != seq_len {seq_len}")
        wt_list = list(wild_type)
        if verbose:
            print(
                f"Edit distance constraint: max {max_edit_distance} edits from {wild_type}"
            )

    # Build AA token IDs (for sampling only valid AAs)
    aa_token_ids = torch.tensor(
        [alphabet.token_to_idx[aa] for aa in AMINO_ACIDS[:20]],
        device=device,
    )

    # Initialize batch
    current_seqs = [list(init_seq) for _ in range(batch_size)]

    # Token positions: [CLS] + left_context + variable + right_context + [EOS]
    var_start_tok = 1 + len(left_context)

    # Build scoring landscape (smaller batch for ESM2 due to 24x pseudolikelihood expansion)
    eval_batch_size = 64 if scoring_model == "ESM2" else 1024
    if verbose:
        print(f"Building {scoring_model} landscape for scoring...")
    scoring_landscape = build_plm_landscape(
        seq_len,
        ab_cfg,
        model_name=scoring_model,
        device=device,
        oracle_batch_size_per_gpu=eval_batch_size,
    )
    scoring_objective = scoring_landscape.active_objs[0]

    # Score wild type
    wt_indices = torch.tensor(
        [[AMINO_ACID_ALPHABET[aa] for aa in init_seq]], dtype=torch.long, device=device
    )
    wt_score = scoring_landscape.score_tensor_batch(
        wt_indices, objective=scoring_objective
    ).item()
    if verbose:
        print(f"Wild type score: {wt_score:.2f}")

    # Output directory for histograms
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Per-chain tracking for mixing diagnostics (reset each eval)
    chain_sequences: list[set[str]] = [set() for _ in range(batch_size)]
    chain_mutation_sets: list[set[frozenset]] = [set() for _ in range(batch_size)]

    # Track sequences written
    total_written = 0

    def run_gibbs_step():
        """Run one Gibbs sampling step: mask random position, resample.

        With max_edit_distance: uniform position selection + rejection sampling.
        """
        # Save current state for rejection
        if max_edit_distance is not None:
            prev_seqs = [seq.copy() for seq in current_seqs]

        full_seqs = [
            left_context + "".join(seq) + right_context for seq in current_seqs
        ]
        tokens = tokenize(full_seqs).to(device)

        # Always uniform position selection
        mask_positions = torch.randint(0, seq_len, (batch_size,), device=device)
        mask_positions_tok = var_start_tok + mask_positions

        masked_tokens = tokens.clone()
        batch_idx = torch.arange(batch_size, device=device)
        masked_tokens[batch_idx, mask_positions_tok] = alphabet.mask_idx

        with torch.no_grad():
            logits = model(masked_tokens).to(torch.float32)

        masked_logits = logits[batch_idx, mask_positions_tok, :] / temperature
        aa_logits = masked_logits[:, aa_token_ids]
        probs = F.softmax(aa_logits, dim=-1)

        sampled_aa_idx = torch.multinomial(probs, num_samples=1).squeeze(-1)
        sampled_aas = [AMINO_ACIDS[idx] for idx in sampled_aa_idx.tolist()]

        for i, (pos, aa) in enumerate(zip(mask_positions.tolist(), sampled_aas)):
            current_seqs[i][pos] = aa

        # Rejection: revert sequences that violate constraint
        if max_edit_distance is not None:
            for i, seq in enumerate(current_seqs):
                edit_dist = sum(1 for j in range(seq_len) if seq[j] != wt_list[j])
                if edit_dist > max_edit_distance:
                    current_seqs[i] = prev_seqs[i]  # reject

        # Track per-chain mixing diagnostics and verify constraint
        if max_edit_distance is not None:
            for i, seq in enumerate(current_seqs):
                seq_str = "".join(seq)
                chain_sequences[i].add(seq_str)
                mut_positions = frozenset(
                    j for j in range(seq_len) if seq[j] != wt_list[j]
                )
                chain_mutation_sets[i].add(mut_positions)
                if len(mut_positions) > max_edit_distance:
                    raise RuntimeError(
                        f"Edit distance constraint violated: batch {i} has {len(mut_positions)} edits "
                        f"(max {max_edit_distance}). Seq: {seq_str}, WT: {wild_type}"
                    )

    def evaluate_batch(step: int):
        """Score current batch and print stats."""
        nonlocal chain_sequences, chain_mutation_sets

        # Score current batch
        batch_seqs = ["".join(seq) for seq in current_seqs]
        seq_indices = torch.tensor(
            [[AMINO_ACID_ALPHABET[aa] for aa in seq] for seq in batch_seqs],
            dtype=torch.long,
            device=device,
        )
        scores = scoring_landscape.score_tensor_batch(
            seq_indices, objective=scoring_objective
        )
        scores_np = scores.cpu().numpy()

        if verbose:
            print(
                f"\n[Step {step}] Batch scores: mean={scores_np.mean():.2f}, "
                f"std={scores_np.std():.2f}, best={scores_np.max():.2f}"
            )
            print(f"  Fraction better than WT: {(scores_np > wt_score).mean():.1%}")

            # Edit distance distribution in current batch
            if max_edit_distance is not None:
                edit_dists = [
                    sum(1 for j in range(seq_len) if seq[j] != wt_list[j])
                    for seq in current_seqs
                ]
                edit_hist = np.bincount(edit_dists, minlength=max_edit_distance + 1)
                print(f"  Edit dist: {dict(enumerate(edit_hist.tolist()))}")

                # Per-chain mixing since last eval
                seqs_per_chain = [len(s) for s in chain_sequences]
                mut_sets_per_chain = [len(s) for s in chain_mutation_sets]
                print(
                    f"  Mixing: {np.mean(seqs_per_chain):.0f} seqs/chain, "
                    f"{np.mean(mut_sets_per_chain):.0f} mut-sets/chain"
                )

        # Reset per-chain tracking for next interval
        chain_sequences[:] = [set() for _ in range(batch_size)]
        chain_mutation_sets[:] = [set() for _ in range(batch_size)]

        # Save histogram
        hist_path = os.path.join(output_dir, f"score_histogram_step{step:06d}.png")
        plot_score_histogram(
            scores_np,
            hist_path,
            title=f"Step {step} batch (mean={scores_np.mean():.1f})",
            wild_type_score=wt_score,
        )

    def write_batch(f, step: int):
        """Write current batch to fasta file."""
        nonlocal total_written
        for i, seq in enumerate(current_seqs):
            seq_str = "".join(seq)
            f.write(f">step{step}_chain{i}\n{seq_str}\n")
            total_written += 1

    # Burn-in phase
    if burn_in > 0:
        if verbose:
            print(f"Running {burn_in} burn-in steps...")
        for _ in tqdm(range(burn_in), desc="Burn-in", disable=not verbose):
            run_gibbs_step()
        # Reset per-chain tracking after burn-in
        chain_sequences[:] = [set() for _ in range(batch_size)]
        chain_mutation_sets[:] = [set() for _ in range(batch_size)]

    # Initial evaluation
    evaluate_batch(0)

    # Main Gibbs loop - stream to file
    with open(output_path, "w") as f:
        pbar = tqdm(
            range(total_steps),
            desc="Gibbs sampling",
            disable=not verbose,
            mininterval=30,
        )
        for step in pbar:
            run_gibbs_step()

            # Write to disk every save_every steps
            if (step + 1) % save_every == 0:
                write_batch(f, step + 1)
                f.flush()  # Ensure written to disk

            # Periodic evaluation
            if (step + 1) % eval_every == 0:
                evaluate_batch(step + 1)

            # Update progress bar
            if verbose and step % 1000 == 0:
                pbar.set_postfix({"written": total_written})

    # Final evaluation if not already done
    if total_steps % eval_every != 0:
        evaluate_batch(total_steps)

    return total_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample sequences from ESM2 using Gibbs sampling (streaming to disk)."
    )
    parser.add_argument(
        "--antibody",
        type=str,
        default="c05",
        help="Antibody config name (from conf/antibody/)",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000,
        help="Total Gibbs steps (after burn-in)",
    )
    parser.add_argument(
        "--burn-in",
        type=int,
        default=100,
        help="Burn-in steps before collecting samples",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=24,
        help="Steps between writing batch to disk",
    )
    parser.add_argument(
        "--eval-every", type=int, default=1000, help="Steps between evaluations"
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--init-seq",
        type=str,
        default=None,
        help="Initial sequence (default: wild_type_cdrh3 from antibody config)",
    )
    parser.add_argument(
        "--scoring-model", type=str, default="ESM2", choices=["ESM2", "IgLM", "IgLM-S"]
    )
    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=None,
        help="Max edit distance from wild type. None = unconstrained.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output (progress bars, eval stats). Default: silent.",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="8m",
        choices=["8m", "35m", "150m"],
        help="ESM2 model size: 8m, 35m, or 150m (default: 8m)",
    )
    return parser.parse_args()


def get_default_output_dir() -> str:
    """Generate default output directory with timestamp."""
    scratch = os.environ.get("PGLD_SCRATCH", os.path.expanduser("~/scratch"))
    output_dir = os.path.join(scratch, "esm2_gibbs_runs")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(output_dir, timestamp)


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load antibody config
    ab_cfg = load_antibody_config(args.antibody)
    if args.verbose:
        print(f"Antibody: {ab_cfg.name}")
        if ab_cfg.wild_type_cdrh3:
            print(f"  Wild-type CDRH3: {ab_cfg.wild_type_cdrh3}")

    # ESM2 requires extended right context
    if not ab_cfg.right_context_esm:
        raise ValueError(
            f"Antibody '{ab_cfg.name}' has no right_context_esm configured, "
            f"but ESM2 requires extended right context"
        )
    left_context = ab_cfg.left_context
    right_context = ab_cfg.right_context_esm

    # Get init sequence
    init_seq = args.init_seq or ab_cfg.wild_type_cdrh3
    if not init_seq:
        raise ValueError(
            f"No init_seq provided and antibody '{ab_cfg.name}' has no wild_type_cdrh3"
        )
    if len(init_seq) != ab_cfg.cdrh3_length:
        raise ValueError(
            f"init_seq must be {ab_cfg.cdrh3_length} AA (matching antibody config), got {len(init_seq)}"
        )

    output_dir = args.output_dir or get_default_output_dir()
    output_path = os.path.join(output_dir, "sequences.fasta")

    if args.verbose:
        print(f"Loading ESM2-{args.model_size.upper()}...")
    from esme import ESM2
    from esme.alphabet import Alphabet

    ckpt_dir = os.environ.get("PGLD_ESM2_CKPT_DIR")
    if not ckpt_dir:
        raise ValueError("PGLD_ESM2_CKPT_DIR environment variable not set")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_file = f"esm2_{args.model_size}.safetensors"
    model_path = os.path.join(ckpt_dir, model_file)
    model = ESM2.from_pretrained(model_path, device=device).eval()
    alphabet = Alphabet()

    # Always print model info to confirm correct model loaded
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded model: {model_path}")
    print(f"  Parameters: {num_params:.1f}M")

    if args.verbose:
        print(
            f"Running Gibbs sampling: {args.total_steps} steps, batch={args.batch_size}"
        )
        print(f"  Output: {output_path}")

    # Wild type for edit distance constraint
    wild_type = ab_cfg.wild_type_cdrh3 if ab_cfg.wild_type_cdrh3 else init_seq

    total_written = gibbs_sample_streaming(
        model=model,
        alphabet=alphabet,
        left_context=left_context,
        right_context=right_context,
        init_seq=init_seq,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        save_every=args.save_every,
        burn_in=args.burn_in,
        temperature=args.temperature,
        device=device,
        output_path=output_path,
        scoring_model=args.scoring_model,
        ab_cfg=ab_cfg,
        max_edit_distance=args.max_edit_distance,
        wild_type=wild_type,
        verbose=args.verbose,
    )

    print(f"Wrote {total_written} sequences to {output_path}")


if __name__ == "__main__":
    main()
