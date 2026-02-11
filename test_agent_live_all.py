#!/usr/bin/env python3
"""
Live (non-mocked) agent test for all questions in Tests.pdf.

Usage:
    LIVE=1 python3 test_agent_live_first10.py

Optional:
    STRICT=1  -> enable stricter assertions on expected outputs.
"""

from __future__ import annotations

import os

from biomni.agent import A1

CSV_PATH = "/home/oem/Desktop/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"

QUESTIONS = [
    "Is YES1 amplified as ecDNA or BFB in CCLE? In which samples?",
    "Summarize the size distribution of all BFB amplifications (Captured interval length)",
    "List all BFB amplifications in HARA (feature IDs, loci, genes, copy number)",
    "Which genes are most frequently amplified as BFB? (top 25)",
    "Show the distribution of amplification classes across cancer types (Tissue of origin)",
    "For each tissue, what fraction of samples have any ecDNA?",
    "Do Complexity scores differ between ecDNA vs BFB vs Linear vs Complex-non-cyclic?",
    "For ecDNA vs BFB, how does size relate to max copy number? (scatter + correlation)",
    "What are the highest copy-number amplifications in CCLE? (top 5)",
    "Which oncogenes appear in both ecDNA and BFB across CCLE?",
    "Which samples contain both ecDNA and BFB amplifications?",
    "Are ecDNA amplifications larger than BFB on average?",
    "Which samples have high-copy (>20 CN) ecDNA?",
    "What is the largest amplicon (by size) per class?",
    "Which tissues show highest amplification complexity (median)?",
    "For a given gene, list all co-amplified genes (same feature)",
    "Which genes are exclusive to ecDNA (never BFB or Linear)?",
    "Which chromosomes are most frequently amplified?",
    "Which oncogenes are most recurrent across unique samples (any class)?",
    "For each tissue, what are the top 5 amplified oncogenes (any class, by #unique samples)?",
    "Which tissues have the highest fraction of BFB features?",
    "Among samples with ecDNA, what are the most common co-amplified oncogene pairs?",
    "Which samples have amplifications on multiple chromosomes? (by feature loci)",
    "For each class, what is the distribution of number of oncogenes per feature?",
]

CASE_IDS = [
    "Q1",
    "Q2",
    "Q3",
    "Q4",
    "Q5",
    "Q6",
    "Q7",
    "Q8",
    "Q9",
    "Q10",
    "Q12",
    "Q13",
    "Q14",
    "Q15",
    "Q16",
    "Q17",
    "Q18",
    "Q19",
    "Q20",
    "Q21",
    "Q22",
    "Q23",
    "Q24",
    "Q25",
]


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _run_strict_case(case_id: str) -> None:
    import importlib

    integration = importlib.import_module("test_agent_integration")
    code = integration._execute_code(case_id)
    local_vars: dict[str, object] = {}
    exec(code, local_vars, local_vars)
    ok = local_vars.get("ok")
    _assert(ok is True, f"{case_id} strict check failed")


def _run_one(agent: A1, prompt: str, strict: bool, case_id: str) -> None:
    log, final_content = agent.go(prompt)
    combined = "\n".join(log)

    # Basic checks (should always pass if agent/tool ran)
    _assert(len(final_content.strip()) > 0, "Empty final response")
    _assert("<observation>" in combined, "No tool execution observed")

    if strict:
        _run_strict_case(case_id)


if __name__ == "__main__":
    if os.getenv("LIVE") != "1":
        raise SystemExit("LIVE=1 is required to run live agent tests.")

    strict = os.getenv("STRICT") == "1"

    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"CCLE.csv not found at {CSV_PATH}")

    # Help the tool find CCLE.csv via environment path
    os.environ["BIOMNI_DATA_PATH"] = os.path.dirname(CSV_PATH)

    # llm_model = os.getenv("LLM_MODEL")
    llm_model = "gpt-5-mini"
    # llm_source = os.getenv("LLM_SOURCE")
    llm_source = "OpenAI"
    agent = A1(llm=llm_model, source=llm_source, expected_data_lake_files=[], use_tool_retriever=False)

    target_q = os.getenv("Q")
    if target_q:
        try:
            target_idx = int(target_q)
        except ValueError:
            raise SystemExit(f"Q must be an integer from 1 to {len(QUESTIONS)}")
        if not 1 <= target_idx <= len(QUESTIONS):
            raise SystemExit(f"Q must be between 1 and {len(QUESTIONS)}")
        indices = [target_idx]
    else:
        indices = list(range(1, len(QUESTIONS) + 1))

    for idx in indices:
        q = QUESTIONS[idx - 1]
        case_id = CASE_IDS[idx - 1]
        print(f"\n=== LIVE Q{idx} ===")
        prompt = (
            f"{q} Use amplicon_table and csv_path='{CSV_PATH}'. "
            "Note: query_amplicons returns a dict with a 'rows' key; use result['rows'] as the table. "
            "Be concise. Return only <execute> and <solution> tags."
        )
        _run_one(agent, prompt, strict, case_id)

    print(f"\n✓ Live agent tests (Q1–Q{len(QUESTIONS)}) completed")
