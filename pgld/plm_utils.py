"""General utility functions."""

import os

from omegaconf import OmegaConf

from antibody_config import AntibodyConfig
from landscapes import PLMLogLikelihoodLandscape


def load_fasta(path: str) -> list[str]:
    """Load sequences from a FASTA file."""
    sequences = []
    current_seq = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_seq:
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line)
        if current_seq:
            sequences.append("".join(current_seq))

    return sequences


def build_plm_landscape(
    peptide_len: int,
    ab_cfg: AntibodyConfig,
    model_name: str = "IgLM",
    device: str = "cuda",
    use_infill: bool = True,
    oracle_batch_size_per_gpu: int = 1024,
) -> PLMLogLikelihoodLandscape:
    """
    Build a minimal PLM landscape for scoring AA sequences.

    Args:
        peptide_len: Length of peptides to score
        ab_cfg: Antibody config with left/right context
        model_name: PLM model ("IgLM", "IgLM-S", "ESM2", "Linear")
        device: Device for model
        use_infill: For IgLM, use infill scoring P(var | left, right). Default True.
        oracle_batch_size_per_gpu: Batch size per GPU. ESM2 needs ~64-128 due to 24x expansion.
    """
    if model_name in ("IgLM", "IgLM-S"):
        models_config = {
            model_name: {
                "right_context": ab_cfg.right_context,
                "left_context": ab_cfg.left_context,
                "invalid_sequence_score": -200,
                "args": {"device": device, "use_infill": use_infill},
            }
        }
        if model_name == "IgLM-S":
            models_config[model_name]["args"]["model_name"] = "IgLM-S"
    elif model_name == "ESM2":
        esm2_path = os.environ.get("PGLD_ESM2_CKPT_DIR")
        if not esm2_path:
            raise ValueError("PGLD_ESM2_CKPT_DIR environment variable not set")
        if not ab_cfg.right_context_esm:
            raise ValueError(
                f"Antibody '{ab_cfg.name}' has no right_context_esm configured"
            )
        models_config = {
            "ESM2": {
                "right_context": ab_cfg.right_context_esm,
                "left_context": ab_cfg.left_context,
                "invalid_sequence_score": -100,
                "args": {
                    "default_token_logprob": -20.0,
                    "device": device,
                    "model_name": "esm2_8m",
                    "path": esm2_path,
                },
            }
        }
    elif model_name == "Linear":
        models_config = {
            "Linear": {
                "right_context": "",
                "left_context": "",
                "invalid_sequence_score": -100,
                "args": {"default_token_logprob": -20.0, "device": device, "seed": 42},
            }
        }
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    models_config = OmegaConf.create(models_config)

    landscape = PLMLogLikelihoodLandscape(
        str_length=peptide_len * 3,
        objectives=[model_name],
        device=device,
        min_peptide_len=peptide_len,
        max_peptide_len=peptide_len,
        models=models_config,
        oracle_batch_size_per_gpu=oracle_batch_size_per_gpu,
    )
    return landscape
