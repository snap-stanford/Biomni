"""Retrieval benchmark runner (memory-level + fact-level + category breakdown).

Runs the 20-case retrieval suite against the 20-memory shared corpus with a real
semantic embedding (``sentence_transformer`` / ``all-MiniLM-L6-v2``). Ground truth
references stable ``memory_key`` ids; the runner maps each version's ``memory_id``
back to ``memory_key`` at run time.

Run once per version (baseline vs improved) by pointing ``PYTHONPATH`` at the
source tree to measure::

    PYTHONPATH=<repo-root>         venv/bin/python3 run_retrieval.py \
        --out results/retrieval_final/improved.json
    PYTHONPATH=<baseline-checkout> venv/bin/python3 run_retrieval.py \
        --out results/retrieval_final/baseline.json

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
_EVAL = os.path.dirname(_BENCH)
for _p in (_BENCH, _EVAL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import metrics as M  # noqa: E402

from _common import (  # noqa: E402
    RETRIEVAL_CASES,
    Env,
    build_corpus_content,
    build_embedding,
    detect_version,
)

TOP_K = 5


def ingest_corpus(env: Env, content: dict) -> dict[str, str]:
    id2key: dict[str, str] = {}
    for key, mem in content.items():
        mid = env.create_memory(mem["user_id"], mem["summary"])
        env.store_summary(str(mid), mem["summary"], mem["user_id"])
        for (e, r, v) in mem["facts"]:
            env.save_fact(mid, e, r, v, confidence=0.9)
        id2key[str(mid)] = key
    return id2key


def run(env: Env, cases: list[dict], content: dict, id2key: dict[str, str]) -> dict:
    per_cat = defaultdict(lambda: {"n": 0, "p1": [], "p3": [], "p5": [], "r5": [], "mrr": [], "hit": [], "leak": []})
    fact_precisions: list[float] = []
    n_fact_queries = 0

    for q in cases:
        k = q.get("k", TOP_K)
        hits = env.search_memories(q["query"], q["user_id"], k=k)
        ranked = [id2key[str(h.id)] for h in hits if str(h.id) in id2key]
        expected = q["expected_memory_keys"]
        cat = q["category"]

        p1 = M.precision_at_k(ranked, expected, 1)
        p3 = M.precision_at_k(ranked, expected, 3)
        p5 = M.precision_at_k(ranked, expected, 5)
        r5 = M.recall_at_k(ranked, expected, 5)
        rr = M.mrr(ranked, expected)

        row = per_cat[cat]
        row["n"] += 1
        row["p1"].append(p1)
        row["p3"].append(p3)
        row["p5"].append(p5)
        row["r5"].append(r5)
        row["mrr"].append(rr)

        if cat == "low_lexical_overlap":
            row["hit"].append(M.hit_rate(ranked, expected))

        if cat == "user_isolation":
            excluded = set(q.get("excluded_memory_keys", []))
            row["leak"].append(1.0 if excluded & set(ranked) else 0.0)

        if cat == "specific_fact":
            # expected facts = the facts of the expected memory (human-authored).
            expected_facts = set()
            for mk in expected:
                expected_facts.update(content.get(mk, {}).get("facts", []))
            returned = env.facts_for_query(q["query"], q["user_id"])
            n_fact_queries += 1
            fact_precisions.append(M.fact_precision(returned, expected_facts))

    category_results = {}
    for cat, row in sorted(per_cat.items()):
        entry = {
            "n": row["n"],
            "precision_at_1": round(M.mean(row["p1"]), 4),
            "precision_at_3": round(M.mean(row["p3"]), 4),
            "precision_at_5": round(M.mean(row["p5"]), 4),
            "recall_at_5": round(M.mean(row["r5"]), 4),
            "mrr": round(M.mean(row["mrr"]), 4),
        }
        if cat == "low_lexical_overlap":
            entry["hit_rate"] = round(M.mean(row["hit"]), 4)
        if cat == "user_isolation":
            entry["leak_rate"] = round(M.mean(row["leak"]), 4)
        category_results[cat] = entry

    all_p1 = [x for r in per_cat.values() for x in r["p1"]]
    all_p3 = [x for r in per_cat.values() for x in r["p3"]]
    all_p5 = [x for r in per_cat.values() for x in r["p5"]]
    all_r5 = [x for r in per_cat.values() for x in r["r5"]]
    all_rr = [x for r in per_cat.values() for x in r["mrr"]]

    return {
        "version": detect_version(),
        "corpus": {"n_memories": len(content), "n_queries": len(cases)},
        "memory_level": {
            "precision_at_1": round(M.mean(all_p1), 4),
            "precision_at_3": round(M.mean(all_p3), 4),
            "precision_at_5": round(M.mean(all_p5), 4),
            "recall_at_5": round(M.mean(all_r5), 4),
            "mrr": round(M.mean(all_rr), 4),
        },
        "fact_level": {
            "fact_precision": round(M.mean(fact_precisions), 4),
            "n_fact_queries": n_fact_queries,
        },
        "category_results": category_results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write JSON result to this path")
    ap.add_argument("--embedding", default="sentence_transformer",
                    choices=["sentence_transformer", "hash"])
    args = ap.parse_args()

    cases = json.load(open(RETRIEVAL_CASES, encoding="utf-8"))["cases"]
    content = build_corpus_content()
    env = Env(build_embedding(args.embedding), top_k=TOP_K)
    id2key = ingest_corpus(env, content)
    report = run(env, cases, content, id2key)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
