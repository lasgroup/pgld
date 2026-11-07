"""Run variational synthesis training, sampling, and display template weights."""

import argparse
import configparser
import pickle
from pathlib import Path

import numpy as np

from VariationalSynthesis.model import main as train_main
from VariationalSynthesis.sample import main as sample_main


def get_latest_run_dir(out_folder: str) -> Path:
    """Get the most recent timestamped run directory."""
    logs_dir = Path(out_folder) / "logs"
    run_dirs = sorted(logs_dir.iterdir(), key=lambda x: x.name)
    return run_dirs[-1]


def display_weights(params_file: str) -> None:
    """Load model and display template weights."""
    with open(params_file, "rb") as f:
        model = pickle.load(f)

    weights = model.params["w"].cpu().numpy()
    print("\n" + "=" * 60)
    print("TEMPLATE WEIGHTS")
    print("=" * 60)
    print(f"Number of templates: {len(weights)}")
    print(f"Sum of weights: {weights.sum():.6f}")
    print(f"Min weight: {weights.min():.6f}")
    print(f"Max weight: {weights.max():.6f}")
    print(f"Std of weights: {weights.std():.6f}")
    print("\nTop 10 templates by weight:")
    top_idx = np.argsort(weights)[::-1][:10]
    for i, idx in enumerate(top_idx):
        print(f"  {i+1}. Template {idx}: {weights[idx]:.6f}")

    print("\nAll weights:")
    print(weights)
    print("=" * 60 + "\n")


def train_with_seed(config_path: str, seed: int, out_folder: str) -> tuple[Path, float]:
    """Train with a specific seed and return (run_dir, perplexity)."""
    config = configparser.ConfigParser()
    config.read(config_path)
    config["general"]["rng_seed"] = str(seed)

    print(f"\n--- Training with seed {seed} ---")
    config, _ = train_main(config)

    run_dir = get_latest_run_dir(out_folder)
    run_config = configparser.ConfigParser()
    run_config.read(run_dir / "config.cfg")
    perplexity = float(run_config["results"]["perpl_per_res"])

    print(f"Seed {seed}: perplexity = {perplexity:.4f}")
    return run_dir, perplexity


def main() -> None:
    """Train var-synth with multiple seeds, select best, and sample."""
    parser = argparse.ArgumentParser(description="Run var-synth training and sampling.")
    parser.add_argument("config_path", help="Path to config file")
    parser.add_argument(
        "--nsamples", type=int, default=1000, help="Number of samples to draw"
    )
    parser.add_argument(
        "--skip-train", action="store_true", help="Skip training, use latest run"
    )
    parser.add_argument(
        "--nseeds", type=int, default=3, help="Number of random seeds to try"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1,
        help="Base seed (will use base, base+1, ...)",
    )
    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_path)

    out_folder = config["general"]["out_folder"]
    logs_dir = Path(out_folder) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        print("=" * 60)
        print(f"TRAINING ({args.nseeds} seeds, selecting best perplexity)")
        print("=" * 60)

        results = []
        for i in range(args.nseeds):
            seed = args.base_seed + i
            run_dir, perplexity = train_with_seed(args.config_path, seed, out_folder)
            results.append((run_dir, perplexity, seed))

        # Select best (lowest perplexity)
        results.sort(key=lambda x: x[1])
        best_run_dir, best_perplexity, best_seed = results[0]

        print("\n" + "=" * 60)
        print("SEED COMPARISON")
        print("=" * 60)
        for run_dir, perplexity, seed in results:
            marker = " <-- BEST" if seed == best_seed else ""
            print(f"  Seed {seed}: perplexity = {perplexity:.4f}{marker}")

        run_dir = best_run_dir
    else:
        run_dir = get_latest_run_dir(out_folder)

    print(f"\nUsing run directory: {run_dir}")

    run_config = configparser.ConfigParser()
    run_config.read(run_dir / "config.cfg")

    params_file = run_config["results"]["params_file"]
    display_weights(params_file)

    print("=" * 60)
    print("SAMPLING")
    print("=" * 60)

    sample_args = argparse.Namespace(
        nsamples=args.nsamples,
        truncate_stop=True,
        compute_logp=True,
        rng_seed=42,
        out_folder=out_folder,
    )

    sample_main(run_config, sample_args, output_dir=str(run_dir))

    print(f"\nSamples saved to: {run_dir / 'samples.fasta'}")


if __name__ == "__main__":
    main()
