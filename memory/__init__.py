"""BIOMNI production memory system.

Pipeline: Agent Trace -> MemoryExtractor -> (FactValidator) -> EpisodicMemoryStore
+ SemanticMemoryStore -> MemoryRetriever -> context injection.

The :class:`MemorySystem` facade wires everything together for the agent.
"""
from .episodic import EpisodicMemoryStore
from .extractor import MemoryExtractor, trace_from_log, trace_from_messages
from .models import (
    MemoryConfig,
    MemoryContext,
    MemoryExtraction,
    MemoryFact,
    TraceMessage,
    WorkingMemoryState,
)
from .retriever import MemoryRetriever
from .semantic import SemanticMemoryStore
from .system import MemorySystem
from .validator import FactValidator
from .vector import (
    ChromaVectorStore,
    EmbeddingProvider,
    HashingEmbedding,
    LangChainEmbeddingProvider,
    SearchResult,
    VectorStore,
)
from .working import WorkingMemoryManager, WorkingMemoryStore

__all__ = [
    "MemoryConfig",
    "MemoryContext",
    "MemoryExtraction",
    "MemoryFact",
    "TraceMessage",
    "WorkingMemoryState",
    "MemoryExtractor",
    "trace_from_log",
    "trace_from_messages",
    "FactValidator",
    "EpisodicMemoryStore",
    "SemanticMemoryStore",
    "MemoryRetriever",
    "MemorySystem",
    "WorkingMemoryManager",
    "WorkingMemoryStore",
    "VectorStore",
    "EmbeddingProvider",
    "SearchResult",
    "HashingEmbedding",
    "LangChainEmbeddingProvider",
    "ChromaVectorStore",
]
