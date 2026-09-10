"""Continuation benchmark runner.

Validates that the memory system supports cross-session task continuation, judged
on *retrieval* (not answer quality):

  * is the previous session's memory recalled?
  * are the previous session's key facts present in the retrieved context?
  * is the retrieval free of cross-user leakage?

Flow (per case): the shared corpus (which already contains every session_a memory,
plus the other users' memories for isolation testing) is ingested once; then
``session_b``'s query is issued and the returned ``MemoryContext`` is checked
against the case's ``success_criteria`` (translated into objective checks from the
case's ``expected_memory`` structure).

Run once per version (baseline vs improved) by pointing ``PYTHONPATH`` at the
source tree to measure::

    PYTHONPATH=/home/ytz/Biomni1         venv/bin/python3 run_continuation.py \
        --out results/continuation/improved.json
    PYTHONPATH=/tmp/membench/baseline    venv/bin/python3 run_continuation.py \
        --out results/continuation/baseline.json

This does NOT modify production code and uses a throwaway SQLite + Chroma store.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.dirname(_HERE)
for _p in (_BENCH, os.path.dirname(_BENCH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from _common import (  # noqa: E402
    CONTINUATION_CASES,
    Env,
    build_corpus_content,
    build_embedding,
    detect_version,
)

TOP_K = 10  # enough headroom for multi-memory continuation (>=2 previous_tasks)


def ingest_corpus(env: Env, content: dict) -> dict[str, str]:
    """Store every memory + facts; return ``{memory_id_str: memory_key}``."""
    id2key: dict[str, str] = {}
    for key, mem in content.items():
        mid = env.create_memory(mem["user_id"], mem["summary"])
        env.store_summary(str(mid), mem["summary"], mem["user_id"])
        for (e, r, v) in mem["facts"]:
            env.save_fact(mid, e, r, v, confidence=0.9)
        id2key[str(mid)] = key
    return id2key


def evaluate_case(env: Env, case: dict, content: dict) -> dict:
    sa = case["session_a"]
    sb = case["session_b"]
    category = case["category"]
    keys = sa["memory_keys"]

    sa_summaries = [content[k]["summary"] for k in keys]
    must_facts = [
        (f["entity"], f["relation"], f["value"])
        for f in case["expected_memory"].get("facts_must_contain", [])
    ]

    ctx = env.retrieve(sb["query"], sb["user_id"])
    prev_text = " ".join(ctx.previous_tasks)
    fact_triples = {(f.entity, f.relation, f.value) for f in ctx.facts}

    # 1. previous memory recalled (>=2 summaries for cross-memory continuation)
    recalled = sum(1 for s in sa_summaries if s and s in prev_text)

    # 2. key facts retained
    retained = [f for f in must_facts if f in fact_triples]
    key_facts_ok = len(retained) == len(must_facts)

    # 3. no cross-user leak (foreign facts + foreign summaries)
    foreign_facts = set()
    foreign_summaries = []
    for k, mem in content.items():
        if mem["user_id"] != sb["user_id"]:
            foreign_facts.update(mem["facts"])
            if mem["summary"]:
                foreign_summaries.append(mem["summary"])
    leaked_facts = fact_triples & foreign_facts
    leaked_summaries = [s for s in foreign_summaries if s and s in prev_text]
    no_leak = not leaked_facts and not leaked_summaries

    need = 2 if category == "cross_memory_continuation" else 1
    recalled_ok = recalled >= need

    if category == "user_isolation":
        # negative case: the previous memory MUST NOT be recalled, and no foreign
        # fact may leak into session_b's context.
        passed = recalled == 0 and no_leak
    else:
        passed = recalled_ok and key_facts_ok and no_leak

    failure_reasons = []
    if category == "user_isolation":
        if recalled != 0:
            failure_reasons.append(f"foreign memory leaked into context (recalled={recalled})")
        if not no_leak:
            failure_reasons.append(f"cross-user facts leaked ({len(leaked_facts)})")
    else:
        if not recalled_ok:
            failure_reasons.append(f"previous memory not recalled ({recalled}/{need})")
        if not key_facts_ok:
            failure_reasons.append(
                f"key facts missing ({len(retained)}/{len(must_facts)}: "
                + ", ".join(f"{e}/{v}" for e, r, v in must_facts if (e, r, v) not in fact_triples)
                + ")"
            )
        if not no_leak:
            failure_reasons.append(f"cross-user leak ({len(leaked_facts)} facts, {len(leaked_summaries)} summaries)")

    return {
        "scenario_id": case["scenario_id"],
        "name": case["name"],
        "category": category,
        "passed": passed,
        "recalled_count": recalled,
        "facts_retained": f"{len(retained)}/{len(must_facts)}",
        "leaked_facts": len(leaked_facts),
        "leaked_summaries": len(leaked_summaries),
        "failure_reasons": failure_reasons,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write JSON result to this path")
    ap.add_argument("--embedding", default="sentence_transformer",
                    choices=["sentence_transformer", "hash"])
    ap.add_argument("--top-k", type=int, default=TOP_K)
    args = ap.parse_args()

    cases = json.load(open(CONTINUATION_CASES, encoding="utf-8"))["cases"]
    content = build_corpus_content()
    env = Env(build_embedding(args.embedding), top_k=args.top_k)
    ingest_corpus(env, content)

    results = [evaluate_case(env, case, content) for case in cases]

    by_cat: dict[str, dict] = defaultdict(lambda: {"n": 0, "passed": 0})
    for r in results:
        by_cat[r["category"]]["n"] += 1
        by_cat[r["category"]]["passed"] += int(r["passed"])

    n_passed = sum(1 for r in results if r["passed"])
    report = {
        "version": detect_version(),
        "n_cases": len(results),
        "passed": n_passed,
        "failed": len(results) - n_passed,
        "pass_rate": round(n_passed / len(results), 4) if results else 0.0,
        "category_results": {
            cat: {"n": v["n"], "passed": v["passed"], "pass_rate": round(v["passed"] / v["n"], 4)}
            for cat, v in sorted(by_cat.items())
        },
        "cases": results,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
