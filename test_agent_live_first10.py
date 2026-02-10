#!/usr/bin/env python3
"""
Live (non-mocked) agent test for first 10 PDF questions.

Usage:
  LIVE=1 python3 test_agent_live_first10.py

Optional:
  STRICT=1  -> enable stricter assertions on expected outputs.
"""
from __future__ import annotations

import os

from biomni.agent import A1

CSV_PATH = "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"

QUESTIONS = [
    "Is YES1 amplified as ecDNA or BFB in CCLE? In which samples?",
    "Summarize the size distribution of all BFB amplifications (Captured interval length) at CCLE data",
    "List all BFB amplifications in HARA (feature IDs, loci, genes, copy number) at CCLE data",
    "Which genes are most frequently amplified as BFB? (top 25) at CCLE data",
    "Do Complexity scores differ between ecDNA vs BFB vs Linear vs Complex-non-cyclic? at CCLE data",
    "What are the highest copy-number amplifications in CCLE? (top 5) at CCLE data",
    "Which oncogenes appear in both ecDNA and BFB across CCLE? at CCLE data",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_one(agent: A1, prompt: str, strict: bool) -> None:
    log, final_content = agent.go(prompt)
    combined = "\n".join(log)

    # Basic checks (should always pass if agent/tool ran)
    _assert(len(final_content.strip()) > 0, "Empty final response")
    _assert("<observation>" in combined, "No tool execution observed")

    if strict:
        # Light, stable checks for first 10 questions
        if "YES1" in prompt:
            _assert("KYSE70_OESOPHAGUS" in combined, "Expected YES1 sample not found")
        if "BFB" in prompt and "size distribution" in prompt:
            _assert("BFB" in combined, "Expected BFB in output")
        if "HARA" in prompt:
            _assert("HARA_LUNG" in combined, "Expected HARA_LUNG in output")
        if "highest copy-number" in prompt:
            _assert("NCIH524_LUNG" in combined, "Expected top copy-number sample not found")


if __name__ == "__main__":
    if os.getenv("LIVE") != "1":
        raise SystemExit("LIVE=1 is required to run live agent tests.")

    strict = os.getenv("STRICT") == "1"

    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CCLE.csv not found at {CSV_PATH}")

    # Help the tool find CCLE.csv via environment path
    os.environ["BIOMNI_DATA_PATH"] = os.path.dirname(CSV_PATH)

    llm_model = os.getenv("LLM_MODEL")
    llm_source = os.getenv("LLM_SOURCE")

    agent = A1(llm=llm_model, source=llm_source, expected_data_lake_files=[], use_tool_retriever=False)

    target_q = os.getenv("Q")
    if target_q:
        try:
            target_idx = int(target_q)
        except ValueError:
            raise SystemExit("Q must be an integer from 1 to 10")
        if not 1 <= target_idx <= len(QUESTIONS):
            raise SystemExit("Q must be between 1 and 10")
        indices = [target_idx]
    else:
        indices = list(range(1, len(QUESTIONS) + 1))

    for idx in indices:
        q = QUESTIONS[idx - 1]
        print(f"\n=== LIVE Q{idx} ===")
        prompt = (
            f"{q} Use amplicon_table and csv_path='{CSV_PATH}'. "
            "Be concise. Return only <execute> and <solution> tags."
        )
        _run_one(agent, prompt, strict)

    print("\n✓ Live agent tests (Q1–Q10) completed")
