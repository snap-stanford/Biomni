"""Relation definitions for the Research Knowledge Graph.

This module defines the Relation class used to represent relationships between
entities in the knowledge graph. Relations are directed edges connecting two entities.

Relations are schema-less by design — any relation type can be added without predefined
constraints. Common types include REGULATES, TARGETS, ASSOCIATED_WITH, etc.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from .entity import Entity

# Common biomedical relation types (not enforced — any string is valid)
REGULATES = "REGULATES"
TARGETS = "TARGETS"
ASSOCIATED_WITH = "ASSOCIATED_WITH"
ENCODES = "ENCODES"
BINDS = "BINDS"
MUTATED_IN = "MUTATED_IN"
CAUSES = "CAUSES"
TREATS = "TREATS"
INHIBITS = "INHIBITS"
ACTIVATES = "ACTIVATES"
PARTICIPATES_IN = "PARTICIPATES_IN"
INTERACTS_WITH = "INTERACTS_WITH"
CONVERTS_TO = "CONVERTS_TO"
EXPRESSED_IN = "EXPRESSED_IN"
HAS_VARIANT = "HAS_VARIANT"

RELATION_TYPES = [
    REGULATES,
    TARGETS,
    ASSOCIATED_WITH,
    ENCODES,
    BINDS,
    MUTATED_IN,
    CAUSES,
    TREATS,
    INHIBITS,
    ACTIVATES,
    PARTICIPATES_IN,
    INTERACTS_WITH,
    CONVERTS_TO,
    EXPRESSED_IN,
    HAS_VARIANT,
]


@dataclass
class Relation:
    """A directed relationship between two entities in the knowledge graph.

    Attributes:
        source_entity: The source entity of the relation.
        target_entity: The target entity of the relation.
        type: The relation type (e.g., "REGULATES", "TARGETS").
        properties: Optional metadata (e.g., {"confidence": 0.95, "evidence": "STRING DB"}).
        source: Where this relation was first observed (e.g., "query_stringdb", "user_input").
        id: Unique identifier. Auto-generated if not provided.

    Examples:
        >>> tp53 = Entity(name="TP53", type="Gene")
        >>> bax = Entity(name="BAX", type="Gene")
        >>> rel = Relation(tp53, bax, type="REGULATES", properties={"evidence": "literature"})
        >>> rel.id
        'Gene:TP53--REGULATES-->Gene:BAX'
    """

    source_entity: Entity
    target_entity: Entity
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    id: str = ""

    def __post_init__(self):
        """Generate ID from entities and type if not provided."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a deterministic ID from source, target, and relation type."""
        return f"{self.source_entity.id}--{self.type}-->{self.target_entity.id}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize relation to dictionary."""
        return {
            "id": self.id,
            "source_entity": self.source_entity.to_dict(),
            "target_entity": self.target_entity.to_dict(),
            "type": self.type,
            "properties": self.properties,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relation:
        """Deserialize relation from dictionary."""
        return cls(
            source_entity=Entity.from_dict(data["source_entity"]),
            target_entity=Entity.from_dict(data["target_entity"]),
            type=data["type"],
            properties=data.get("properties", {}),
            source=data.get("source", ""),
            id=data.get("id", ""),
        )

    def fingerprint(self) -> str:
        """Generate a stable fingerprint for deduplication.

        Two relations with the same source, target, and type are considered
        the same relation (properties are merged on conflict).
        """
        key = f"{self.source_entity.fingerprint()}:{self.type}:{self.target_entity.fingerprint()}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def __hash__(self):
        """Hash based on fingerprint for use in sets/dicts."""
        return hash(self.fingerprint())

    def __eq__(self, other):
        """Equality based on fingerprint (source + type + target)."""
        if not isinstance(other, Relation):
            return False
        return self.fingerprint() == other.fingerprint()

    def merge_properties(self, other: Relation) -> Relation:
        """Merge properties from another relation with the same source+target+type.

        The other relation's properties take precedence for conflicting keys.
        Returns a new Relation with merged properties.
        """
        merged_props = {**self.properties, **other.properties}
        merged_source = self.source
        if other.source and other.source not in self.source:
            merged_source = f"{self.source},{other.source}" if self.source else other.source
        return Relation(
            source_entity=self.source_entity,
            target_entity=self.target_entity,
            type=self.type,
            properties=merged_props,
            source=merged_source,
            id=self.id,
        )

    def reverse(self) -> Relation:
        """Create a reversed relation (swap source and target entities)."""
        return Relation(
            source_entity=self.target_entity,
            target_entity=self.source_entity,
            type=self.type,
            properties=self.properties.copy(),
            source=self.source,
        )

    def __repr__(self):
        props_str = f", properties={self.properties}" if self.properties else ""
        source_str = f", source='{self.source}'" if self.source else ""
        return (
            f"Relation({self.source_entity.name!r} --[{self.type}]--> "
            f"{self.target_entity.name!r}{props_str}{source_str})"
        )
