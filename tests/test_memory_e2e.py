"""End-to-end test of the memory pipeline: ingest a trace, then retrieve it.

Runs against a throwaway SQLite DB + Chroma directory with the deterministic
hashing embedding, so it needs no network access and no real LLM. This locks the
full ``ingest -> retrieve`` round-trip and the user-isolation boundary.
"""

from __future__ import annotations

import pytest
from memory.models import MemoryConfig, MemoryExtraction, MemoryFact, TraceMessage
from memory.system import MemorySystem


class _FakeStructuredLLM:
    """Returns a canned extraction when the extractor calls ``.invoke``."""

    def __init__(self, extraction: MemoryExtraction) -> None:
        self._extraction = extraction

    def invoke(self, prompt: str) -> MemoryExtraction:
        return self._extraction


class _FakeLLM:
    """Minimal LLM stand-in: only ``with_structured_output`` is exercised."""

    def __init__(self, extraction: MemoryExtraction) -> None:
        self._extraction = extraction

    def with_structured_output(self, schema):
        return _FakeStructuredLLM(self._extraction)


@pytest.fixture
def memory(tmp_path):
    extraction = MemoryExtraction(
        summary="Analyzed BRCA1 mutations and concluded 185delAG is pathogenic.",
        facts=[
            MemoryFact(
                entity="BRCA1",
                relation="pathogenic_mutation",
                value="185delAG",
                confidence=0.9,
                source="tool_result",
            ),
            MemoryFact(
                entity="BRCA1",
                relation="located_on",
                value="chr17",
                confidence=0.8,
                source="tool_result",
            ),
        ],
    )
    config = MemoryConfig(
        database_url=f"sqlite:///{tmp_path}/memory.db",
        persist_dir=str(tmp_path / "chroma"),
        embedding_provider="hash",
        vector_db="chroma",
    )
    return MemorySystem(config=config, llm=_FakeLLM(extraction))


def test_ingest_then_retrieve_roundtrip(memory):
    memory.ingest_sync(
        [TraceMessage(type="human", content="Analyze BRCA1 mutations")],
        user_id="alice",
    )

    fragment = memory.retrieve("BRCA1 mutations", user_id="alice")

    assert "BRCA1" in fragment
    assert "185delAG" in fragment


def test_retrieve_is_user_isolated(memory):
    memory.ingest_sync([TraceMessage(type="human", content="Analyze BRCA1")], user_id="alice")

    # bob has no episodes, so the scoped vector search must return nothing.
    assert memory.retrieve("BRCA1 mutations", user_id="bob") == ""


def test_retrieve_empty_when_store_empty(memory):
    assert memory.retrieve("anything at all", user_id="alice") == ""
