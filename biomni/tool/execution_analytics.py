"""
Tool Execution Analytics and Adaptive Retry System

This module provides comprehensive analytics tracking, performance monitoring,
and intelligent retry mechanisms for tool execution in Biomni agents.

Features:
- Execution performance tracking (success rate, latency, error patterns)
- Adaptive retry strategies based on error classification
- Result caching to avoid redundant tool calls
- Analytics reporting for tool optimization
"""

import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd


class ErrorType(Enum):
    """Classification of error types for adaptive retry strategies."""

    TIMEOUT = "timeout"
    NETWORK = "network"
    VALIDATION = "validation"
    RESOURCE = "resource"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    RATE_LIMIT = "rate_limit"
    UNKNOWN = "unknown"


class RetryStrategy(Enum):
    """Retry strategies based on error type."""

    IMMEDIATE = "immediate"  # Retry immediately (for transient errors)
    EXPONENTIAL_BACKOFF = "exponential_backoff"  # Exponential backoff (for rate limits)
    LINEAR_BACKOFF = "linear_backoff"  # Linear backoff (for network issues)
    NO_RETRY = "no_retry"  # Don't retry (for permanent errors)


@dataclass
class ExecutionRecord:
    """Record of a single tool execution attempt."""

    tool_name: str
    timestamp: float
    execution_time: float
    success: bool
    error_type: Optional[ErrorType] = None
    error_message: Optional[str] = None
    parameters_hash: str = ""
    result_hash: Optional[str] = None
    retry_count: int = 0


@dataclass
class ToolAnalytics:
    """Analytics for a specific tool."""

    tool_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    error_counts: Dict[ErrorType, int] = field(default_factory=lambda: defaultdict(int))
    last_execution: Optional[float] = None
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate."""
        if self.total_executions == 0:
            return 0.0
        return self.failed_executions / self.total_executions


class ExecutionAnalytics:
    """Comprehensive analytics and retry system for tool execution."""

    def __init__(
        self,
        enable_caching: bool = True,
        cache_ttl: int = 3600,
        max_retries: int = 3,
        enable_analytics: bool = True,
    ):
        """Initialize the execution analytics system.

        Args:
            enable_caching: Whether to cache successful tool results
            cache_ttl: Time-to-live for cached results in seconds
            max_retries: Maximum number of retry attempts
            enable_analytics: Whether to track execution analytics
        """
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        self.max_retries = max_retries
        self.enable_analytics = enable_analytics

        # Analytics storage
        self.tool_analytics: Dict[str, ToolAnalytics] = {}
        self.execution_history: List[ExecutionRecord] = []

        # Result cache: {tool_name: {param_hash: (result, timestamp)}}
        self.result_cache: Dict[str, Dict[str, Tuple[Any, float]]] = defaultdict(dict)

        # Error classification patterns
        self.error_patterns = {
            ErrorType.TIMEOUT: ["timeout", "timed out", "exceeded", "deadline"],
            ErrorType.NETWORK: ["connection", "network", "dns", "unreachable", "refused"],
            ErrorType.VALIDATION: ["validation", "invalid", "malformed", "format", "type error"],
            ErrorType.RESOURCE: ["memory", "out of memory", "resource", "quota", "limit exceeded"],
            ErrorType.PERMISSION: ["permission", "forbidden", "unauthorized", "access denied"],
            ErrorType.NOT_FOUND: ["not found", "404", "does not exist", "missing"],
            ErrorType.RATE_LIMIT: ["rate limit", "429", "too many requests", "throttle"],
        }

        # Retry strategy mapping
        self.retry_strategies = {
            ErrorType.TIMEOUT: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorType.NETWORK: RetryStrategy.LINEAR_BACKOFF,
            ErrorType.RATE_LIMIT: RetryStrategy.EXPONENTIAL_BACKOFF,
            ErrorType.RESOURCE: RetryStrategy.NO_RETRY,
            ErrorType.PERMISSION: RetryStrategy.NO_RETRY,
            ErrorType.NOT_FOUND: RetryStrategy.NO_RETRY,
            ErrorType.VALIDATION: RetryStrategy.NO_RETRY,
            ErrorType.UNKNOWN: RetryStrategy.IMMEDIATE,
        }

    def _classify_error(self, error: Exception) -> ErrorType:
        """Classify an error into an ErrorType."""
        error_str = str(error).lower()
        error_type_str = type(error).__name__.lower()

        # Check error message patterns
        for error_type, patterns in self.error_patterns.items():
            for pattern in patterns:
                if pattern in error_str or pattern in error_type_str:
                    return error_type

        return ErrorType.UNKNOWN

    def _get_parameters_hash(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a hash for tool parameters."""
        param_str = json.dumps(
            {"tool": tool_name, "args": args, "kwargs": kwargs}, sort_keys=True, default=str
        )
        return hashlib.md5(param_str.encode()).hexdigest()

    def _get_result_hash(self, result: Any) -> str:
        """Generate a hash for tool result."""
        result_str = json.dumps(result, sort_keys=True, default=str)
        return hashlib.md5(result_str.encode()).hexdigest()

    def _get_cached_result(self, tool_name: str, param_hash: str) -> Optional[Any]:
        """Retrieve cached result if available and not expired."""
        if not self.enable_caching:
            return None

        if tool_name not in self.result_cache:
            return None

        if param_hash not in self.result_cache[tool_name]:
            return None

        result, timestamp = self.result_cache[tool_name][param_hash]
        current_time = time.time()

        # Check if cache entry is still valid
        if current_time - timestamp > self.cache_ttl:
            # Cache expired, remove it
            del self.result_cache[tool_name][param_hash]
            return None

        return result

    def _cache_result(self, tool_name: str, param_hash: str, result: Any):
        """Cache a successful tool result."""
        if not self.enable_caching:
            return

        current_time = time.time()
        self.result_cache[tool_name][param_hash] = (result, current_time)

    def _calculate_backoff_delay(
        self, retry_count: int, strategy: RetryStrategy, base_delay: float = 1.0
    ) -> float:
        """Calculate delay before retry based on strategy."""
        if strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        elif strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return base_delay * (2 ** retry_count)
        elif strategy == RetryStrategy.LINEAR_BACKOFF:
            return base_delay * (retry_count + 1)
        elif strategy == RetryStrategy.NO_RETRY:
            return float("inf")
        else:
            return 0.0

    def _update_analytics(
        self,
        tool_name: str,
        execution_time: float,
        success: bool,
        error_type: Optional[ErrorType] = None,
        from_cache: bool = False,
    ):
        """Update analytics for a tool execution."""
        if not self.enable_analytics:
            return

        if tool_name not in self.tool_analytics:
            self.tool_analytics[tool_name] = ToolAnalytics(tool_name=tool_name)

        analytics = self.tool_analytics[tool_name]
        analytics.total_executions += 1
        analytics.total_execution_time += execution_time
        analytics.last_execution = time.time()

        if from_cache:
            analytics.cache_hits += 1
        else:
            analytics.cache_misses += 1

        if success:
            analytics.successful_executions += 1
        else:
            analytics.failed_executions += 1
            if error_type:
                analytics.error_counts[error_type] += 1

        # Update average execution time
        if analytics.total_executions > 0:
            analytics.average_execution_time = (
                analytics.total_execution_time / analytics.total_executions
            )

    def execute_with_retry(
        self,
        tool_func: Callable,
        tool_name: str,
        args: tuple = (),
        kwargs: dict = None,
        retry_on_error: bool = True,
    ) -> Tuple[Any, ExecutionRecord]:
        """Execute a tool with retry logic and analytics tracking.

        Args:
            tool_func: The tool function to execute
            tool_name: Name of the tool for tracking
            args: Positional arguments for the tool
            kwargs: Keyword arguments for the tool
            retry_on_error: Whether to retry on errors

        Returns:
            Tuple of (result, execution_record)
        """
        if kwargs is None:
            kwargs = {}

        param_hash = self._get_parameters_hash(tool_name, args, kwargs)

        # Check cache first
        cached_result = self._get_cached_result(tool_name, param_hash)
        if cached_result is not None:
            record = ExecutionRecord(
                tool_name=tool_name,
                timestamp=time.time(),
                execution_time=0.0,
                success=True,
                parameters_hash=param_hash,
                result_hash=self._get_result_hash(cached_result),
            )
            self._update_analytics(tool_name, 0.0, True, from_cache=True)
            return cached_result, record

        # Execute with retry logic
        last_error = None
        last_error_type = None
        retry_count = 0

        while retry_count <= self.max_retries:
            start_time = time.time()
            try:
                result = tool_func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Cache successful result
                self._cache_result(tool_name, param_hash, result)

                # Record successful execution
                record = ExecutionRecord(
                    tool_name=tool_name,
                    timestamp=start_time,
                    execution_time=execution_time,
                    success=True,
                    parameters_hash=param_hash,
                    result_hash=self._get_result_hash(result),
                    retry_count=retry_count,
                )

                self.execution_history.append(record)
                self._update_analytics(tool_name, execution_time, True)

                return result, record

            except Exception as e:
                execution_time = time.time() - start_time
                error_type = self._classify_error(e)
                last_error = e
                last_error_type = error_type

                # Record failed execution
                record = ExecutionRecord(
                    tool_name=tool_name,
                    timestamp=start_time,
                    execution_time=execution_time,
                    success=False,
                    error_type=error_type,
                    error_message=str(e),
                    parameters_hash=param_hash,
                    retry_count=retry_count,
                )

                self.execution_history.append(record)
                self._update_analytics(tool_name, execution_time, False, error_type)

                # Determine retry strategy
                if not retry_on_error or retry_count >= self.max_retries:
                    break

                strategy = self.retry_strategies.get(error_type, RetryStrategy.NO_RETRY)
                if strategy == RetryStrategy.NO_RETRY:
                    break

                retry_count += 1
                delay = self._calculate_backoff_delay(retry_count, strategy)
                if delay > 0:
                    time.sleep(min(delay, 60.0))  # Cap delay at 60 seconds

        # All retries exhausted or no retry strategy
        raise last_error

    def get_tool_analytics(self, tool_name: Optional[str] = None) -> Dict[str, ToolAnalytics]:
        """Get analytics for a specific tool or all tools.

        Args:
            tool_name: Name of tool to get analytics for, or None for all tools

        Returns:
            Dictionary mapping tool names to their analytics
        """
        if tool_name:
            return {tool_name: self.tool_analytics.get(tool_name)} if tool_name in self.tool_analytics else {}
        return dict(self.tool_analytics)

    def get_analytics_summary(self) -> pd.DataFrame:
        """Get a summary of all tool analytics as a DataFrame.

        Returns:
            DataFrame with columns: tool_name, total_executions, success_rate,
            failure_rate, avg_execution_time, cache_hit_rate, most_common_error
        """
        rows = []
        for tool_name, analytics in self.tool_analytics.items():
            most_common_error = (
                max(analytics.error_counts.items(), key=lambda x: x[1])[0].value
                if analytics.error_counts
                else None
            )

            cache_total = analytics.cache_hits + analytics.cache_misses
            cache_hit_rate = (
                analytics.cache_hits / cache_total if cache_total > 0 else 0.0
            )

            rows.append(
                {
                    "tool_name": tool_name,
                    "total_executions": analytics.total_executions,
                    "success_rate": analytics.success_rate,
                    "failure_rate": analytics.failure_rate,
                    "avg_execution_time": analytics.average_execution_time,
                    "cache_hit_rate": cache_hit_rate,
                    "most_common_error": most_common_error,
                    "last_execution": (
                        datetime.fromtimestamp(analytics.last_execution).isoformat()
                        if analytics.last_execution
                        else None
                    ),
                }
            )

        return pd.DataFrame(rows)

    def get_error_analysis(self) -> pd.DataFrame:
        """Get error analysis across all tools.

        Returns:
            DataFrame with error statistics by tool and error type
        """
        rows = []
        for tool_name, analytics in self.tool_analytics.items():
            for error_type, count in analytics.error_counts.items():
                rows.append(
                    {
                        "tool_name": tool_name,
                        "error_type": error_type.value,
                        "error_count": count,
                        "error_rate": count / analytics.total_executions if analytics.total_executions > 0 else 0.0,
                    }
                )

        return pd.DataFrame(rows)

    def clear_cache(self, tool_name: Optional[str] = None):
        """Clear cached results for a specific tool or all tools.

        Args:
            tool_name: Name of tool to clear cache for, or None for all tools
        """
        if tool_name:
            if tool_name in self.result_cache:
                self.result_cache[tool_name].clear()
        else:
            self.result_cache.clear()

    def reset_analytics(self):
        """Reset all analytics data."""
        self.tool_analytics.clear()
        self.execution_history.clear()
        self.result_cache.clear()

