"""Pydantic models for the BIOMNI memory system.

These models define the data contracts that cross module boundaries:
the normalized conversation trace, the structured output of the memory
extractor, the working-memory state, and the assembled retrieval context.
"""
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class TraceMessage(BaseModel):
    """A single normalized entry in an agent conversation trace.

    The memory extractor consumes a list of these, so the trace is decoupled
    from LangChain's message types (which may change across versions).
    """

    type: str = Field(
        default="",
        description="One of: human, ai, tool, observation, system",
    )
    content: str = Field(default="", description="Text content of the message")
    tool_result: str | None = Field(
        default=None, description="Result of a tool call, if this entry is a tool"
    )


FactStatus = Literal["active", "superseded", "retracted", "expired"]


class MemoryFact(BaseModel):
    """A single durable fact about the world or the user (subject-predicate-object).

    Lifecycle and scoring fields (``created_at``, ``updated_at``, ``status``,
    ``importance_score``) are assigned by the memory system — never by the LLM.
    They carry defaults so the extractor's structured output is not asked to
    invent them; persistence ignores any LLM-supplied values and lets the
    database generate the authoritative timestamps.
    """

    entity: str = Field(..., description="Subject, e.g. 'BRCA1'")
    relation: str = Field(..., description="Predicate, e.g. 'has_mutation'")
    value: str = Field(..., description="Object, e.g. '185delAG'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extractor confidence 0..1")
    source: str = Field(
        default="",
        description="Origin: 'tool_result', 'user', or 'llm' (a guess, filterable)",
    )
    created_at: datetime | None = Field(
        default=None, description="First written to memory (set by Python/DB, not the LLM)"
    )
    updated_at: datetime | None = Field(
        default=None, description="Last content modification (set by Python/DB, not the LLM)"
    )
    status: FactStatus = Field(default="active", description="Lifecycle state of the fact")
    importance_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Composite retention score, filled at retrieval"
    )


class MemoryExtraction(BaseModel):
    """Structured output of the memory extractor over one trace."""

    summary: str = Field(
        ..., description="Concise, self-contained summary of what was accomplished"
    )
    facts: list[MemoryFact] = Field(
        default_factory=list, description="Durable facts extracted from the trace"
    )


class WorkingMemoryState(BaseModel):
    """Current task state kept in working memory.

    ``task_id`` is the immutable identity of a task's working-memory entry: it is
    assigned at creation and never changed by an ordinary state update. ``user_id``
    scopes the entry to an owner, so ``(user_id, task_id)`` forms the isolation
    boundary — one user's task can never read or overwrite another user's task of
    the same id.

    Lifecycle fields (``created_at``, ``updated_at``, ``expires_at``) are assigned
    by the store, never by the caller. ``version`` backs optimistic concurrency
    control on the SQL store: a save whose version no longer matches the stored row
    fails rather than silently overwriting a concurrent update.
    """

    task_id: str
    user_id: str = "default"
    current_step: str = Field(default="", description="Human-readable current step")
    variables: dict[str, Any] = Field(default_factory=dict)
    next_action: str = Field(default="", description="Planned next action")
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)
    expires_at: datetime | None = Field(
        default=None, description="Working-memory TTL expiry (set by the store)"
    )
    version: int = Field(default=0, ge=0, description="Optimistic-lock version")


class MemoryContext(BaseModel):
    """Retrieved memories assembled for injection into the agent prompt."""

    previous_tasks: list[str] = Field(default_factory=list)
    facts: list[MemoryFact] = Field(default_factory=list)

    def to_prompt(self) -> str:
        """Render the context as a prompt fragment for injection."""
        parts: list[str] = []
        if self.previous_tasks:
            parts.append(
                "Previous Task(s):\n"
                + "\n".join(f"- {s}" for s in self.previous_tasks)
            )
        if self.facts:
            parts.append(
                "Important Facts:\n"
                + "\n".join(
                    f"- {f.entity} {f.relation} {f.value}" for f in self.facts
                )
            )
        return "\n\n".join(parts)

    def is_empty(self) -> bool:
        return not self.previous_tasks and not self.facts


class MemoryConfig(BaseModel):
    """Runtime configuration for the memory subsystem.

    All fields have defaults so the system is usable out of the box in a dev
    environment (SQLite + sentence-transformer embeddings + Chroma). The
    deterministic hashing embedding stays available for tests/CI/offline by
    setting ``embedding_provider="hash"``.
    """

    enabled: bool = True
    database_url: str = "sqlite:///./biomni_data/memory.db"
    vector_db: str = "chroma"  # Supported: "chroma". Planned: "faiss", "milvus".
    persist_dir: str = "./biomni_data/memory/chroma"
    collection_name: str = "episodic_memory"
    embedding_provider: str = "sentence_transformer"  # "sentence_transformer" | "hash" | "openai"
    embedding_model_name: str = "all-MiniLM-L6-v2"  # light-weight semantic model
    min_confidence: float = 0.6
    top_k: int = 5
    async_extraction: bool = True

    # ---- fact lifecycle & scoring (see memory.scoring) ----
    fact_ttl_days: int = 365  # facts older than this (by created_at) expire
    recency_lambda: float = 0.01  # exponential decay rate per day for recency_score
    confidence_weight: float = 0.4
    usage_weight: float = 0.2
    recency_weight: float = 0.2
    feedback_weight: float = 0.2  # weight of explicit user feedback in importance_score
    usage_saturation: float = 100.0  # access_count at which usage_score saturates to 1.0
    scoring_threshold: int = 3  # access_count cutoff between cold-start and mature ranking
    max_facts: int = 20  # cap on facts injected into a single MemoryContext
    # Retained for backward compatibility: the query-aware fact ranking no longer
    # hard-filters facts below a similarity threshold (similarity only affects rank).
    fact_similarity_threshold: float = 0.45
    # Negative votes before a fact is retracted (never auto-deleted). A single
    # negative is recorded but does not retract; only crossing this threshold does.
    feedback_retract_threshold: int = 3

    # ---- working memory lifecycle ----
    working_memory_ttl_days: int = 7  # working-memory entries older than this are cleaned up

    @model_validator(mode="after")
    def _validate_scoring_weights(self) -> "MemoryConfig":
        total = (
            self.confidence_weight
            + self.usage_weight
            + self.recency_weight
            + self.feedback_weight
        )
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                "confidence_weight + usage_weight + recency_weight + feedback_weight "
                f"must sum to 1.0 (got {total}) so importance_score stays within [0, 1]"
            )
        return self
