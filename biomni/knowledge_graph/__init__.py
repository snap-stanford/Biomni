"""Research Knowledge Graph for Biomni.

This module provides a knowledge graph for accumulating biomedical research knowledge
across agent sessions. It captures entities (genes, proteins, drugs, diseases, etc.)
and their relationships from tool outputs, persists them in SQLite, and surfaces
relevant prior knowledge in future agent invocations.

Main classes:
    - KnowledgeGraph: Core graph management with persistence
    - Entity: Biomedical entity (gene, protein, drug, disease, etc.)
    - Relation: Directed relationship between two entities
    - Storage: SQLite-based persistence backend

Quick start:
    >>> from biomni.knowledge_graph import KnowledgeGraph, Entity, Relation
    >>> kg = KnowledgeGraph("my_research.db")
    >>> tp53 = Entity(name="TP53", type="Gene", properties={"species": "human"})
    >>> bax = Entity(name="BAX", type="Gene")
    >>> kg.add_entity(tp53)
    >>> kg.add_entity(bax)
    >>> kg.add_relation(Relation(tp53, bax, type="REGULATES"))
    >>> kg.get_stats()
    {'entity_count': 2, 'relation_count': 1, 'entity_types': {'Gene': 2}, 'relation_types': {'REGULATES': 1}}
"""

from .entity import (
    ANATOMY,
    CELL_TYPE,
    CHEMICAL,
    DISEASE,
    DRUG,
    GENE,
    GENE_ONTOLOGY,
    ORGANISM,
    PAPER,
    PATHWAY,
    PROTEIN,
    VARIANT,
    Entity,
)
from .graph import KnowledgeGraph
from .relation import (
    ACTIVATES,
    ASSOCIATED_WITH,
    BINDS,
    CAUSES,
    CONVERTS_TO,
    ENCODES,
    EXPRESSED_IN,
    HAS_VARIANT,
    INHIBITS,
    INTERACTS_WITH,
    MUTATED_IN,
    PARTICIPATES_IN,
    REGULATES,
    RELATION_TYPES,
    TARGETS,
    TREATS,
    Relation,
)
from .storage import Storage

__all__ = [
    # Core classes
    "KnowledgeGraph",
    "Entity",
    "Relation",
    "Storage",
    # Entity type constants
    "GENE",
    "PROTEIN",
    "DISEASE",
    "DRUG",
    "PATHWAY",
    "VARIANT",
    "CELL_TYPE",
    "PAPER",
    "CHEMICAL",
    "ORGANISM",
    "GENE_ONTOLOGY",
    "ANATOMY",
    # Relation type constants
    "REGULATES",
    "TARGETS",
    "ASSOCIATED_WITH",
    "ENCODES",
    "BINDS",
    "MUTATED_IN",
    "CAUSES",
    "TREATS",
    "INHIBITS",
    "ACTIVATES",
    "PARTICIPATES_IN",
    "INTERACTS_WITH",
    "CONVERTS_TO",
    "EXPRESSED_IN",
    "HAS_VARIANT",
    "RELATION_TYPES",
]
