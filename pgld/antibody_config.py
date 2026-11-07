"""Antibody context configuration loading."""

from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

ANTIBODY_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "conf", "antibody")


@dataclass
class AntibodyConfig:
    """Antibody context configuration for PLM sampling/scoring."""

    name: str
    left_context: str
    right_context: str
    cdrh3_length: int
    chain: str
    species: str
    wild_type_cdrh3: str | None = None
    right_context_esm: str | None = None  # Extended right context for ESM2


def load_antibody_config(name: str) -> AntibodyConfig:
    """Load antibody config from YAML file in conf/antibody/."""
    config_path = os.path.join(ANTIBODY_CONFIG_DIR, f"{name}.yaml")
    if not os.path.exists(config_path):
        available = [
            f.replace(".yaml", "")
            for f in os.listdir(ANTIBODY_CONFIG_DIR)
            if f.endswith(".yaml")
        ]
        raise FileNotFoundError(
            f"Antibody config '{name}' not found. Available: {available}"
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return AntibodyConfig(
        name=data["name"],
        left_context=data["left_context"],
        right_context=data["right_context"],
        cdrh3_length=data["cdrh3_length"],
        chain=data["chain"],
        species=data["species"],
        wild_type_cdrh3=data.get("wild_type_cdrh3"),
        right_context_esm=data.get("right_context_esm"),
    )
