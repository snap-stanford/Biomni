"""Pydantic models for the BIOMNI memory system.

These models define the data contracts that cross module boundaries:
the normalized conversation trace, the structured output of the memory
extractor, the working-memory state, and the assembled retrieval context.
"""
from typing import Any

from pydantic import BaseModel, Field


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


class MemoryFact(BaseModel):
    """A single durable fact about the world or the user (subject-predicate-object)."""

    entity: str = Field(..., description="Subject, e.g. 'BRCA1'")
    relation: str = Field(..., description="Predicate, e.g. 'has_mutation'")
    value: str = Field(..., description="Object, e.g. '185delAG'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Extractor confidence 0..1")
    source: str = Field(
        default="",
        description="Origin: 'tool_result', 'user', or 'llm' (a guess, filterable)",
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
    """Current task state kept in working memory."""

    task_id: str
    current_step: str = Field(default="", description="Human-readable current step")
    variables: dict[str, Any] = Field(default_factory=dict)
    next_action: str = Field(default="", description="Planned next action")


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
    environment (SQLite + deterministic hashing embedding), while remaining
    swappable for production (PostgreSQL + OpenAI embeddings + Chroma/Milvus).
    """

    enabled: bool = True
    database_url: str = "sqlite:///./biomni_data/memory.db"
    vector_db: str = "chroma"  # "chroma" | "faiss" | "milvus"
    persist_dir: str = "./biomni_data/memory/chroma"
    collection_name: str = "episodic_memory"
    embedding_provider: str = "hash"  # "hash" | "openai" | "langchain"
    embedding_model: str = "text-embedding-3-small"
    min_confidence: float = 0.6
    top_k: int = 5
    async_extraction: bool = True
