import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from biomni.config import BiomniConfig

REPO_ROOT = Path(__file__).parents[1]
CUSTOM_ENV_NAMES = (
    "BIOMNI_CUSTOM_BASE_URL",
    "CUSTOM_MODEL_BASE_URL",
    "BIOMNI_CUSTOM_API_KEY",
    "CUSTOM_MODEL_API_KEY",
    "BIOMNI_SOURCE",
    "LLM_SOURCE",
)


@pytest.fixture(autouse=True)
def clear_custom_provider_environment(monkeypatch):
    for name in CUSTOM_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_canonical_custom_provider_environment(monkeypatch):
    monkeypatch.setenv("BIOMNI_CUSTOM_BASE_URL", "https://canonical.example/v1")
    monkeypatch.setenv("BIOMNI_CUSTOM_API_KEY", "canonical-key")
    monkeypatch.setenv("BIOMNI_SOURCE", "Custom")

    config = BiomniConfig()

    assert config.base_url == "https://canonical.example/v1"
    assert config.api_key == "canonical-key"
    assert config.source == "Custom"


def test_legacy_documented_names_remain_compatible(monkeypatch):
    monkeypatch.setenv("CUSTOM_MODEL_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("CUSTOM_MODEL_API_KEY", "legacy-key")
    monkeypatch.setenv("LLM_SOURCE", "Custom")

    config = BiomniConfig()

    assert config.base_url == "https://legacy.example/v1"
    assert config.api_key == "legacy-key"
    assert config.source == "Custom"


def test_canonical_names_take_precedence_over_aliases(monkeypatch):
    monkeypatch.setenv("BIOMNI_CUSTOM_BASE_URL", "https://canonical.example/v1")
    monkeypatch.setenv("CUSTOM_MODEL_BASE_URL", "https://legacy.example/v1")
    monkeypatch.setenv("BIOMNI_CUSTOM_API_KEY", "canonical-key")
    monkeypatch.setenv("CUSTOM_MODEL_API_KEY", "legacy-key")
    monkeypatch.setenv("BIOMNI_SOURCE", "Custom")
    monkeypatch.setenv("LLM_SOURCE", "Groq")

    config = BiomniConfig()

    assert config.base_url == "https://canonical.example/v1"
    assert config.api_key == "canonical-key"
    assert config.source == "Custom"


def test_fresh_process_default_config_loads_legacy_aliases():
    environment = os.environ.copy()
    for name in CUSTOM_ENV_NAMES:
        environment.pop(name, None)
    environment.update(
        {
            "CUSTOM_MODEL_BASE_URL": "https://legacy.example/v1",
            "CUSTOM_MODEL_API_KEY": "legacy-key",
            "LLM_SOURCE": "Custom",
        }
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json; from biomni.config import default_config; print(json.dumps(default_config.to_dict()))",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    config = json.loads(completed.stdout)

    assert config["base_url"] == "https://legacy.example/v1"
    assert config["api_key"] == "legacy-key"
    assert config["source"] == "Custom"


def test_example_files_use_names_read_by_config():
    assignment_pattern = re.compile(r"^\s*(?:#\s*)?(?:export\s+)?([A-Z][A-Z0-9_]+)=", re.MULTILINE)
    env_example_names = set(assignment_pattern.findall((REPO_ROOT / ".env.example").read_text()))
    readme_names = set(assignment_pattern.findall((REPO_ROOT / "README.md").read_text()))
    canonical_names = {
        "BIOMNI_SOURCE",
        "BIOMNI_LLM",
        "BIOMNI_CUSTOM_BASE_URL",
        "BIOMNI_CUSTOM_API_KEY",
    }

    assert canonical_names <= env_example_names
    assert canonical_names <= readme_names
