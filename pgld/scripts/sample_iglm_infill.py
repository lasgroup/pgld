#!/usr/bin/env python
"""Sample sequences from IgLM using infilling mode.

Loads antibody context from pgld/conf/antibody/{name}.yaml.
Fixes the broken HuggingFace tokenizer by replacing it with one using IGLM_VOCAB.
Outputs sequences in FASTA format to $PGLD_SCRATCH/iglm_samples/.

Usage:
    # Natural-length sampling with trastuzumab (default antibody)
    python pgld/scripts/sample_iglm_infill.py --model IgLM-S --num-samples 100 --natural-length

    # Use C05 antibody instead
    python pgld/scripts/sample_iglm_infill.py --antibody c05 --model IgLM-S --num-samples 100

    # Fixed-length (forces exact CDRH3 length by masking non-AA tokens)
    python pgld/scripts/sample_iglm_infill.py --model IgLM-S --num-samples 100

    # Variable-length (IgLM's native behavior, varying lengths)
    python pgld/scripts/sample_iglm_infill.py --model IgLM-S --num-samples 100 --variable-length
"""

import argparse
import os
import sys
from collections import Counter
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add pgld to path for imports
PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)

from antibody_config import load_antibody_config
from constants import IGLM_VOCAB


class IgLMTokenizer:
    """Working tokenizer for IgLM using IGLM_VOCAB from constants.py."""

    def __init__(self):
        self.vocab = IGLM_VOCAB
        self.id_to_token = {v: k for k, v in IGLM_VOCAB.items()}
        self.unk_token = "[UNK]"
        self.unk_token_id = IGLM_VOCAB["[UNK]"]
        self.pad_token = "[PAD]"
        self.pad_token_id = IGLM_VOCAB["[PAD]"]
        self.mask_token = "[MASK]"
        self.mask_token_id = IGLM_VOCAB["[MASK]"]
        self.sep_token = "[SEP]"
        self.sep_token_id = IGLM_VOCAB["[SEP]"]
        self.cls_token = "[CLS]"
        self.cls_token_id = IGLM_VOCAB["[CLS]"]
        self.eos_token_id = IGLM_VOCAB["[SEP]"]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode text to token IDs."""
        tokens = []
        i = 0
        while i < len(text):
            if text[i] == "[":
                for special in [
                    "[HEAVY]",
                    "[LIGHT]",
                    "[HUMAN]",
                    "[MOUSE]",
                    "[CAMEL]",
                    "[RABBIT]",
                    "[RAT]",
                    "[RHESUS]",
                    "[MASK]",
                    "[SEP]",
                    "[PAD]",
                    "[UNK]",
                    "[CLS]",
                ]:
                    if text[i:].startswith(special):
                        tokens.append(self.vocab[special])
                        i += len(special)
                        break
                else:
                    tokens.append(self.unk_token_id)
                    i += 1
            elif text[i] in self.vocab:
                tokens.append(self.vocab[text[i]])
                i += 1
            else:
                tokens.append(self.unk_token_id)
                i += 1
        return tokens

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text."""
        result = []
        for tid in token_ids:
            tok = self.id_to_token.get(tid, self.unk_token)
            if skip_special_tokens and tok.startswith("["):
                continue
            result.append(tok)
        return "".join(result)

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        """Convert token strings to IDs."""
        return [self.vocab.get(t, self.unk_token_id) for t in tokens]

    def __len__(self) -> int:
        return len(self.vocab)


def infill_fixed_length(
    model,
    tokenizer: IgLMTokenizer,
    left_context: str,
    right_context: str,
    infill_length: int,
    num_to_generate: int,
    chain: str,
    species: str,
    temperature: float,
    top_p: float,
    deduplicate: bool = False,
) -> list[str]:
    """
    Generate sequences with exactly `infill_length` AA in the variable region.

    Uses IgLM's infilling format: [chain, species, left_ctx, MASK, right_ctx, SEP]
    Forces exactly infill_length AAs by masking out non-AA tokens.
    """
    device = model.model.device

    start_tokens = (
        [chain, species]
        + list(left_context)
        + ["[MASK]"]
        + list(right_context)
        + ["[SEP]"]
    )
    prompt_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(start_tokens)], dtype=torch.long, device=device
    )
    prompt_len = len(start_tokens)

    # Only allow AA tokens (IDs 5-24)
    non_aa_token_ids = list(range(0, 5)) + list(range(25, 33))

    generated_seqs = []
    seen = set() if deduplicate else None
    pbar = tqdm(total=num_to_generate)

    while len(generated_seqs) < num_to_generate:
        current = prompt_ids.clone()

        for _ in range(infill_length):
            with torch.no_grad():
                logits = model.model(current).logits[:, -1, :] / temperature
                logits[:, non_aa_token_ids] = float("-inf")

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    logits[mask.scatter(1, sorted_idx, mask)] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                current = torch.cat([current, next_token], dim=1)

        generated_ids = current[0, prompt_len:].tolist()
        cdrh3 = tokenizer.decode(generated_ids, skip_special_tokens=True)

        if deduplicate:
            if cdrh3 not in seen:
                seen.add(cdrh3)
                generated_seqs.append(cdrh3)
                pbar.update(1)
        else:
            generated_seqs.append(cdrh3)
            pbar.update(1)

    pbar.close()
    return generated_seqs


def infill_natural_length(
    model,
    tokenizer: IgLMTokenizer,
    left_context: str,
    right_context: str,
    target_length: int,
    num_to_generate: int,
    chain: str,
    species: str,
    temperature: float,
    top_p: float,
    max_length: int = 50,
    deduplicate: bool = False,
) -> list[str]:
    """
    Generate sequences by letting IgLM naturally decide when to stop.

    Only keeps sequences that are exactly `target_length` AAs. This ensures
    the kept sequences are ones the model "wanted" to generate at that length,
    rather than being forced to continue.

    Args:
        max_length: Maximum tokens to generate before giving up on a sequence.
    """
    device = model.model.device

    start_tokens = (
        [chain, species]
        + list(left_context)
        + ["[MASK]"]
        + list(right_context)
        + ["[SEP]"]
    )
    prompt_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(start_tokens)], dtype=torch.long, device=device
    )
    prompt_len = len(start_tokens)

    cls_token_id = IGLM_VOCAB["[CLS]"]
    sep_token_id = IGLM_VOCAB["[SEP]"]
    aa_token_ids = set(range(5, 25))

    generated_seqs = []
    seen = set() if deduplicate else None
    total_attempts = 0
    too_short = 0
    too_long = 0
    pbar = tqdm(total=num_to_generate, desc=f"Generating (target_len={target_length})")

    while len(generated_seqs) < num_to_generate:
        current = prompt_ids.clone()
        aa_tokens = []
        total_attempts += 1

        for _ in range(max_length):
            with torch.no_grad():
                logits = model.model(current).logits[:, -1, :] / temperature

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    mask = cumprobs > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    logits[mask.scatter(1, sorted_idx, mask)] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                token_id = next_token[0, 0].item()

                if token_id == cls_token_id or token_id == sep_token_id:
                    break

                if token_id in aa_token_ids:
                    aa_tokens.append(token_id)
                    current = torch.cat([current, next_token], dim=1)
                else:
                    break

        if len(aa_tokens) == target_length:
            cdrh3 = "".join([tokenizer.id_to_token.get(t, "?") for t in aa_tokens])
            if deduplicate:
                if cdrh3 not in seen:
                    seen.add(cdrh3)
                    generated_seqs.append(cdrh3)
                    pbar.update(1)
            else:
                generated_seqs.append(cdrh3)
                pbar.update(1)
        elif len(aa_tokens) < target_length:
            too_short += 1
        else:
            too_long += 1

        if total_attempts % 100 == 0:
            accept_rate = len(generated_seqs) / total_attempts * 100
            pbar.set_description(f"Generating (accept={accept_rate:.1f}%)")

    pbar.close()
    accept_rate = len(generated_seqs) / total_attempts * 100
    print(f"\nSampling stats:")
    print(f"  Total attempts: {total_attempts}")
    print(
        f"  Accepted (len={target_length}): {len(generated_seqs)} ({accept_rate:.1f}%)"
    )
    print(f"  Rejected too short: {too_short} ({too_short/total_attempts*100:.1f}%)")
    print(f"  Rejected too long: {too_long} ({too_long/total_attempts*100:.1f}%)")
    return generated_seqs


def write_fasta(sequences: list[str], headers: list[str], path: str) -> None:
    """Write sequences to a FASTA file."""
    with open(path, "w") as f:
        for header, seq in zip(headers, sequences):
            f.write(f">{header}\n{seq}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample sequences from IgLM using infilling mode."
    )
    parser.add_argument(
        "--antibody",
        type=str,
        default="trastuzumab",
        help="Antibody config name (from conf/antibody/)",
    )
    parser.add_argument("--model", type=str, required=True, choices=["IgLM", "IgLM-S"])
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--variable-length",
        action="store_true",
        help="Use variable-length infill (IgLM's native behavior).",
    )
    parser.add_argument(
        "--natural-length",
        action="store_true",
        help="Let model naturally decide length, only keep sequences of target length. "
        "Much slower but sequences score higher under IgLM.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Only keep unique sequences (slower, keeps sampling until num_samples unique).",
    )
    return parser.parse_args()


def get_default_output_path(
    antibody_name: str, model_name: str, num_samples: int
) -> str:
    """Generate default output path with timestamp."""
    scratch = os.environ.get("PGLD_SCRATCH", os.path.expanduser("~/scratch"))
    output_dir = os.path.join(scratch, "iglm_samples")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return os.path.join(
        output_dir, f"{timestamp}_{antibody_name}_{model_name}_{num_samples}.fasta"
    )


def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Set random seed: {args.seed}")

    # Load antibody config
    ab_cfg = load_antibody_config(args.antibody)
    print(f"Antibody: {ab_cfg.name}")
    if ab_cfg.wild_type_cdrh3:
        print(f"  Wild-type CDRH3: {ab_cfg.wild_type_cdrh3}")

    output_path = args.output or get_default_output_path(
        ab_cfg.name, args.model, args.num_samples
    )

    print(f"Loading {args.model}...")
    import iglm

    model = iglm.IgLM(model_name=args.model)

    tokenizer = IgLMTokenizer()
    model.tokenizer = tokenizer
    print(f"  Replaced tokenizer (vocab size: {len(tokenizer)})")

    print(f"Generating {args.num_samples} sequences...")
    print(
        f"  Context: {len(ab_cfg.left_context)} AA left + {ab_cfg.cdrh3_length} AA variable + {len(ab_cfg.right_context)} AA right"
    )

    if args.natural_length:
        print(
            f"  Mode: natural-length (reject unless exactly {ab_cfg.cdrh3_length} AA)"
        )
        print(f"  Deduplicate: {args.deduplicate}")
        generated_seqs = infill_natural_length(
            model=model,
            tokenizer=tokenizer,
            left_context=ab_cfg.left_context,
            right_context=ab_cfg.right_context,
            target_length=ab_cfg.cdrh3_length,
            num_to_generate=args.num_samples,
            chain=ab_cfg.chain,
            species=ab_cfg.species,
            temperature=args.temperature,
            top_p=args.top_p,
            deduplicate=args.deduplicate,
        )
    elif args.variable_length:
        if args.deduplicate:
            raise ValueError(
                "--deduplicate not supported with --variable-length (uses IgLM's native infill)"
            )
        print(f"  Mode: variable-length (IgLM native)")
        # Use IgLM's native infill (variable length)
        placeholder = "A" * ab_cfg.cdrh3_length
        parent_sequence = ab_cfg.left_context + placeholder + ab_cfg.right_context
        infill_start = len(ab_cfg.left_context)
        infill_end = infill_start + ab_cfg.cdrh3_length

        full_seqs = model.infill(
            parent_sequence,
            ab_cfg.chain,
            ab_cfg.species,
            infill_range=(infill_start, infill_end),
            num_to_generate=args.num_samples,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        # Extract just the CDRH3 region from full sequences
        generated_seqs = [
            seq[len(ab_cfg.left_context) : -len(ab_cfg.right_context)]
            for seq in full_seqs
        ]
    else:
        print(f"  Mode: fixed-length (force {ab_cfg.cdrh3_length} AA)")
        print(f"  Deduplicate: {args.deduplicate}")
        generated_seqs = infill_fixed_length(
            model=model,
            tokenizer=tokenizer,
            left_context=ab_cfg.left_context,
            right_context=ab_cfg.right_context,
            infill_length=ab_cfg.cdrh3_length,
            num_to_generate=args.num_samples,
            chain=ab_cfg.chain,
            species=ab_cfg.species,
            temperature=args.temperature,
            top_p=args.top_p,
            deduplicate=args.deduplicate,
        )

    headers = [
        f"iglm_sample_{i}|antibody={ab_cfg.name}|model={args.model}|chain=HEAVY|species=HUMAN"
        for i in range(len(generated_seqs))
    ]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    write_fasta(generated_seqs, headers, output_path)
    print(f"Wrote {len(generated_seqs)} sequences to {output_path}")

    # Generate duplicate frequency histogram
    seq_counts = Counter(generated_seqs)
    count_of_counts = Counter(seq_counts.values())

    print(f"\nDuplicate statistics:")
    print(f"  Total sequences: {len(generated_seqs)}")
    print(f"  Unique sequences: {len(seq_counts)}")
    print(f"  Duplicate rate: {100 * (1 - len(seq_counts) / len(generated_seqs)):.2f}%")

    # Save histogram
    hist_path = output_path.replace(".fasta", "_duplicate_hist.png")
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = sorted(count_of_counts.keys())
    frequencies = [count_of_counts[c] for c in counts]
    x_positions = np.arange(len(counts))
    ax.bar(x_positions, frequencies, edgecolor="black", alpha=0.7)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(c) for c in counts])
    ax.set_xlabel("Times sequence appeared")
    ax.set_ylabel("Number of unique sequences")
    ax.set_title(
        f"Sequence Duplication Frequency (n={len(generated_seqs)}, unique={len(seq_counts)})"
    )
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved duplicate histogram to {hist_path}")


if __name__ == "__main__":
    main()
