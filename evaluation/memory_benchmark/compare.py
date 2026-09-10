"""Flatten two run_benchmark.py JSON outputs into a |metric|baseline|improved|delta| table."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# direction for each metric: whether a higher value is "better" (+1) or "worse" (-1).
# Delta is reported as improved - baseline; a positive delta is good for "+" metrics.
DIRECTION = {
    # conflict: exposure/duplication are bad; accuracy is good
    "conflict.stale_fact_exposure_rate": -1,
    "conflict.active_fact_accuracy": +1,
    "conflict.duplicate_fact_exposure_rate": -1,
    # lifecycle: exposure of retracted/expired facts is bad
    "lifecycle.retract_exposure_rate": -1,
    "lifecycle.expire_exposure_rate": -1,
    # isolation: leakage is bad
    "isolation.vector_leakage_rate": -1,
    "isolation.sql_leakage_rate": -1,
    # retrieval: recall/mrr/precision are good
    "retrieval.recall_at_5": +1,
    "retrieval.mrr": +1,
    "retrieval.fact_precision": +1,
    # continuation: hit/retention are good
    "continuation.required_fact_hit_rate": +1,
    "continuation.specific_value_retention_rate": +1,
}

GROUP_LABEL = {
    "conflict": "Conflict Resolution",
    "lifecycle": "Lifecycle (retract / TTL)",
    "isolation": "User Isolation",
    "retrieval": "Retrieval Quality",
    "continuation": "Continuation",
}


def flatten(data: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for group, metrics in data.items():
        if group == "version":
            continue
        for name, value in metrics.items():
            out[f"{group}.{name}"] = value
    return out


def fmt(v: float) -> str:
    return f"{v:.4f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("improved", type=Path)
    args = ap.parse_args()

    base = flatten(json.loads(args.baseline.read_text()))
    impr = flatten(json.loads(args.improved.read_text()))

    print("| metric | baseline | improved | delta |")
    print("|--------|---------:|---------:|------:|")

    current_group = None
    for key in DIRECTION:
        group = key.split(".", 1)[0]
        if group != current_group:
            current_group = group
            print(f"| **{GROUP_LABEL[group]}** | | | |")
        b = base[key]
        i = impr[key]
        d = i - b
        arrow = ""
        if abs(d) > 1e-9:
            good = (d > 0) == (DIRECTION[key] > 0)
            arrow = " ✅" if good else " ❌"
        print(f"| {key} | {fmt(b)} | {fmt(i)} | {fmt(d)}{arrow} |")


if __name__ == "__main__":
    main()
