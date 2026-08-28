"""Helpers for detecting repeated, identical execution failures."""

import hashlib
from typing import TypedDict

MAX_IDENTICAL_FAILURES = 2


class ExecutionFailureState(TypedDict):
    """Serializable state for consecutive identical execution failures."""

    code_signature: str
    result_signature: str
    count: int


def _normalize_for_signature(value: str) -> str:
    """Normalize platform line endings and trailing whitespace."""
    normalized = value.replace("\r\n", "\n").replace("\r", "\n").strip()
    return "\n".join(line.rstrip() for line in normalized.split("\n"))


def _signature(value: str) -> str:
    normalized = _normalize_for_signature(value)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def is_execution_failure(result: str) -> bool:
    """Return whether a runner result uses Biomni's standard error format."""
    normalized = result.lstrip().casefold()
    return normalized.startswith(("error", "timeout", "traceback"))


def record_execution_result(
    previous: ExecutionFailureState | None,
    code: str,
    result: str,
) -> ExecutionFailureState | None:
    """Record a failure, resetting the tracker after success or a changed failure."""
    if not is_execution_failure(result):
        return None

    code_signature = _signature(code)
    result_signature = _signature(result)
    count = 1

    if (
        previous is not None
        and previous["code_signature"] == code_signature
        and previous["result_signature"] == result_signature
    ):
        count = previous["count"] + 1

    return {
        "code_signature": code_signature,
        "result_signature": result_signature,
        "count": count,
    }


def should_block_execution(
    previous: ExecutionFailureState | None,
    code: str,
    max_identical_failures: int = MAX_IDENTICAL_FAILURES,
) -> bool:
    """Return whether code already produced the same failure too many times."""
    if previous is None or previous["count"] < max_identical_failures:
        return False

    return previous["code_signature"] == _signature(code)


def repeated_failure_guidance(count: int) -> str:
    """Build feedback that asks the model to change strategy."""
    return (
        f"Loop guard: The same code produced the same execution failure {count} consecutive times. "
        "Do not repeat the same code. Diagnose the failure and use a different approach."
    )


def blocked_execution_message() -> str:
    """Build the observation returned when a repeated action is skipped."""
    return (
        "Loop guard: Execution skipped because this code already produced the same failure "
        f"{MAX_IDENTICAL_FAILURES} consecutive times. Change the code or choose a different strategy."
    )
