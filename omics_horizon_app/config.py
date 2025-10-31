"""Configuration helpers for the Omics Horizon Streamlit app."""

from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path

from biomni.config import BiomniConfig, default_config

LLM_MODEL = "gemini-2.5-pro"
CURRENT_ABS_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_config_paths() -> tuple[str, str]:
    """Load project-level config.yaml settings with repo-root fallbacks."""
    try:
        repo_root = Path(__file__).resolve().parents[1]
    except Exception:  # pragma: no cover
        repo_root = Path.cwd()

    config_path = repo_root / "config.yaml"
    cfg: dict = {}

    if config_path.is_file():
        try:
            import yaml

            with config_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

    biomni_data_default = str(repo_root / "biomni_data")
    workspace_default = str(repo_root / "workspace")
    return (
        cfg.get("BIOMNI_DATA_PATH", biomni_data_default),
        cfg.get("WORKSPACE_PATH", workspace_default),
    )


BIOMNI_DATA_PATH, WORKSPACE_PATH = _load_config_paths()

LOGO_COLOR_PATH = str(Path(CURRENT_ABS_DIR) / "logo" / "OMICS-HORIZON_Logo_Color.svg")
LOGO_MONO_PATH = str(Path(CURRENT_ABS_DIR) / "logo" / "OMICS-HORIZON_Logo_Mono.svg")


def create_agent_config() -> BiomniConfig:
    """Return an isolated BiomniConfig instance for the Streamlit session."""
    config_values = asdict(default_config)
    config_values.update(
        {
            "llm": LLM_MODEL,
            "commercial_mode": True,
            "path": BIOMNI_DATA_PATH,
        }
    )
    return BiomniConfig(**config_values)
