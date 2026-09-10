"""Shared evaluation metrics for the memory benchmarks.

All functions are pure: they take ranked/returned ``memory_key`` lists (or fact
tuples) plus ground truth and return floats in ``[0, 1]`` (higher = better,
except where noted). They are deliberately version-agnostic and corpus-agnostic
so both the retrieval and the isolation runners can reuse them.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence


def _key_set(keys: Iterable[str]) -> set[str]:
    return {str(k) for k in keys}


def precision_at_k(ranked_keys: Sequence[str], expected_keys: Sequence[str], k: int) -> float:
    """Fraction of the top-``k`` returned memories that are relevant."""
    top = list(ranked_keys)[:k]
    if not top:
        return 0.0
    return len(_key_set(top) & _key_set(expected_keys)) / len(top)


def recall_at_k(ranked_keys: Sequence[str], expected_keys: Sequence[str], k: int) -> float:
    """Fraction of relevant memories retrieved within the top-``k``."""
    rel = _key_set(expected_keys)
    if not rel:
        return 0.0
    return len(_key_set(list(ranked_keys)[:k]) & rel) / len(rel)


def mrr(ranked_keys: Sequence[str], expected_keys: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant memory (0 if none)."""
    rel = _key_set(expected_keys)
    for rank, key in enumerate(ranked_keys, start=1):
        if str(key) in rel:
            return 1.0 / rank
    return 0.0


def fact_precision(
    returned_facts: Sequence[tuple], expected_facts: Sequence[tuple]
) -> float:
    """Fraction of returned ``(entity, relation, value)`` facts that are expected.

    ``returned_facts`` / ``expected_facts`` are sequences of hashable tuples
    (typically ``(entity, relation, value)``); duplicates in the returned list
    are not double-counted.
    """
    returned = set(returned_facts)
    if not returned:
        return 0.0
    return len(returned & set(expected_facts)) / len(returned)


def hit_rate(returned_keys: Sequence[str], expected_keys: Sequence[str]) -> float:
    """Binary-per-query success: 1 if *any* relevant memory is returned, else 0."""
    return 1.0 if (_key_set(returned_keys) & _key_set(expected_keys)) else 0.0


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a (possibly empty) sequence, returning 0.0 when empty."""
    if not values:
        return 0.0
    return sum(values) / len(values)
