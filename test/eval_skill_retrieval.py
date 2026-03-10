#!/usr/bin/env python3
"""
Simple Stage-1 skill retrieval evaluation script.

Goal:
- Evaluate whether skill retrieval hits expected skills for a small fixed test set.
- Avoid full A1 initialization and large datalake checks/downloads.
"""

from __future__ import annotations

import argparse
# Updated by Kyle
import sys
from pathlib import Path

# Updated by Kyle
# Ensure project root is importable when running from ./test directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomni.config import default_config
from biomni.llm import get_llm
from biomni.model.retriever import ToolRetriever


TEST_CASES: list[tuple[str, list[str]]] = [
    ("Query PubMed for papers about CRISPR and return 5 results", ["literature"]),
    (
        "Plan a CRISPR screen to identify genes that regulate T cell exhaustion",
        ["molecular_biology", "immunology"],
    ),
    ("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", ["pharmacology"]),
    ("Perform scRNA-seq annotation and generate meaningful hypothesis", ["genomics", "cell_biology"]),
    ("Analyze protein structure with AlphaFold and predict binding sites", ["biochemistry", "biophysics", "database"]),
    ("Design sgRNA library for genome-wide knockout screen", ["molecular_biology", "genomics"]),
    ("Analyze 16S microbiome diversity from gut samples", ["microbiology"]),
    ("Model metabolic flux and identify bottlenecks in E. coli", ["systems_biology", "bioengineering"]),
    ("Annotate genetic variants from a GWAS study", ["genetics", "genomics"]),
    ("Analyze H&E stained pathology slides for tumor classification", ["pathology", "bioimaging"]),
]


def load_skills() -> list[dict]:
    """Load skill name/description entries from biomni/skills/*/SKILL.md files."""
    skills_root = Path(__file__).resolve().parents[1] / "biomni" / "skills"
    if not skills_root.exists():
        return []

    parsed_skills: list[dict] = []
    for skill_md in sorted(skills_root.glob("*/SKILL.md")):
        module_name = f"biomni.tool.{skill_md.parent.name}"
        try:
            content = skill_md.read_text(encoding="utf-8")
        except OSError:
            continue

        skill_name = None
        skill_desc = ""
        in_front_matter = False
        for raw in content.splitlines():
            line = raw.strip()
            if line == "---":
                in_front_matter = not in_front_matter
                continue
            if not in_front_matter:
                continue
            if line.lower().startswith("name:"):
                skill_name = line.split(":", 1)[1].strip()
            elif line.lower().startswith("description:"):
                skill_desc = line.split(":", 1)[1].strip()

        if not skill_name:
            skill_name = skill_md.parent.name

        parsed_skills.append(
            {
                "name": skill_name,
                "description": skill_desc,
                "module": module_name,
            }
        )

    return parsed_skills


def run_eval(model: str) -> None:
    skills = load_skills()
    if not skills:
        raise RuntimeError("No skills found under biomni/skills.")

    retriever = ToolRetriever()
    llm = get_llm(model=model, temperature=0.0, config=default_config)

    hits = 0
    total = len(TEST_CASES)
    print(f"\nRunning skill retrieval eval with model: {model}")
    print(f"Loaded skills: {len(skills)}")
    print("-" * 72)

    for idx, (query, expected) in enumerate(TEST_CASES, start=1):
        resources = {"tools": skills, "data_lake": [], "libraries": [], "know_how": []}
        selected = retriever.prompt_based_retrieval(query, resources, llm=llm)

        selected_names = {str(s.get("name", "")).strip() for s in selected.get("tools", []) if isinstance(s, dict)}
        selected_names_lower = {x.lower() for x in selected_names}
        expected_lower = {x.lower() for x in expected}
        hit = any(e in selected_names_lower for e in expected_lower)
        hits += int(hit)

        status = "OK" if hit else "MISS"
        print(f"[{idx:02d}] {status} {query[:72]}")
        print(f"     expected: {expected}")
        print(f"     selected: {sorted(selected_names)}")

    hit_rate = hits / total if total else 0.0
    print("-" * 72)
    print(f"Hit rate: {hits}/{total} = {hit_rate:.0%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 skill retrieval hit rate.")
    parser.add_argument(
        "--model",
        default=getattr(default_config, "retrieval_llm", None) or default_config.llm,
        help="Model to use for retrieval eval (default: config retrieval_llm or llm).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(args.model)
