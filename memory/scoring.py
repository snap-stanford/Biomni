"""Fact scoring: importance, recency, and usage signals.

Pure functions so the math is unit-testable independent of the database and the
LLM. ``importance_score`` is the composite used for ranking mature facts; the
other two are factored out for clarity and direct testing.

Conceptual separation (see the memory design notes):
  * vector similarity  -> how relevant an *episode* is to the current query
  * ``importance_score`` -> how worth keeping / prioritizing an individual *fact* is
  * ``recency_score``    -> one input to importance, NOT the lifecycle TTL rule
"""

from __future__ import annotations

import math
from datetime import datetime, timezone


def _as_utc(dt: datetime) -> datetime:
    """Normalize a possibly-naive datetime to timezone-aware UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def recency_score(created_at: datetime, now: datetime, recency_lambda: float) -> float:
    """Smooth exponential decay of recency with age (in days), in [0, 1].

    ``recency_score = exp(-recency_lambda * age_days)``. Newer facts score closer
    to 1; there is no hard threshold. ``recency_lambda`` controls the decay rate.
    """
    age_days = max(0.0, (_as_utc(now) - _as_utc(created_at)).total_seconds() / 86400.0)
    return math.exp(-recency_lambda * age_days)


def usage_score(access_count: int, usage_saturation: float) -> float:
    """Log-normalized usage in [0, 1]; saturates at ``usage_saturation`` accesses.

    ``log1p`` keeps a single high-use fact from blowing up the scale: going from
    0 to 1 access matters much more than going from 100 to 101.
    """
    if access_count <= 0:
        return 0.0
    return min(1.0, math.log1p(access_count) / math.log1p(usage_saturation))


def feedback_score(positive: int, negative: int) -> float:
    """Smoothed explicit-feedback signal in [0, 1]; 0.5 means "no feedback yet".

    Laplace-smoothed fraction of positive feedback, ``(positive + 1) /
    (positive + negative + 2)``. With zero votes this is exactly 0.5 (neutral),
    so an un-reviewed fact is NOT judged low-quality. It drifts toward 1
    (helpful) or 0 (wrong/useless) as feedback accumulates, and a single negative
    vote does not collapse it to 0.
    """
    p = max(0, int(positive))
    n = max(0, int(negative))
    return (p + 1) / (p + n + 2)


def importance_score(
    confidence: float,
    access_count: int,
    created_at: datetime,
    now: datetime,
    *,
    confidence_weight: float,
    usage_weight: float,
    recency_weight: float,
    feedback_weight: float = 0.0,
    positive_feedback: int = 0,
    negative_feedback: int = 0,
    usage_saturation: float,
    recency_lambda: float,
) -> float:
    """Weighted composite of confidence, usage, recency, and user feedback.

    The four weights must sum to 1.0 so the result stays bounded to [0, 1].
    ``feedback_weight`` defaults to 0.0 for backward compatibility: callers that
    pass only the original three weights get the original behavior. ``confidence``
    is clamped defensively even though the schema already bounds it.
    """
    conf = max(0.0, min(1.0, float(confidence)))
    return (
        confidence_weight * conf
        + usage_weight * usage_score(int(access_count), usage_saturation)
        + recency_weight * recency_score(created_at, now, recency_lambda)
        + feedback_weight * feedback_score(positive_feedback, negative_feedback)
    )
