"""Entity definitions for the Research Knowledge Graph.

This module defines the Entity class and entity type constants used to represent
biomedical entities (genes, proteins, drugs, diseases, etc.) in the knowledge graph.

Entities are schema-less by design — any entity type can be added without predefined
ontology constraints. Type alignment with BioPortal/OBO ontologies is deferred to
Phase 4.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any

# Common biomedical entity types (not enforced — any string is valid)
GENE = "Gene"
PROTEIN = "Protein"
DISEASE = "Disease"
DRUG = "Drug"
PATHWAY = "Pathway"
VARIANT = "Variant"
CELL_TYPE = "CellType"
PAPER = "Paper"
CHEMICAL = "Chemical"
ORGANISM = "Organism"
GENE_ONTOLOGY = "GeneOntology"
ANATOMY = "Anatomy"


@dataclass
class Entity:
    """A biomedical entity in the knowledge graph.

    Attributes:
        name: The canonical name of the entity (e.g., "TP53", "BRAF V600E").
        type: The entity type (e.g., "Gene", "Drug", "Disease").
        properties: Optional metadata dict (e.g., {"species": "human", "uniprot_id": "P04637"}).
        source: Where this entity was first observed (e.g., "query_uniprot", "user_input").
        id: Unique identifier. Auto-generated from name+type if not provided.

    Examples:
        >>> gene = Entity(name="TP53", type="Gene", properties={"species": "human"})
        >>> drug = Entity(name="Vemurafenib", type="Drug", source="query_chembl")
        >>> gene.id
        'Gene:TP53'
    """

    name: str
    type: str
    properties: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    id: str = ""

    def __post_init__(self):
        """Generate ID from name and type if not provided."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a deterministic ID from name and type."""
        return f"{self.type}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize entity to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        """Deserialize entity from dictionary."""
        return cls(**data)

    def fingerprint(self) -> str:
        """Generate a stable fingerprint for deduplication.

        Two entities with the same name and type but different properties
        are considered the same entity (properties are merged on conflict).
        """
        return hashlib.sha256(f"{self.type}:{self.name}".encode()).hexdigest()[:16]

    def __hash__(self):
        """Hash based on fingerprint for use in sets/dicts."""
        return hash(self.fingerprint())

    def __eq__(self, other):
        """Equality based on fingerprint (name + type)."""
        if not isinstance(other, Entity):
            return False
        return self.fingerprint() == other.fingerprint()

    def merge_properties(self, other: Entity) -> Entity:
        """Merge properties from another entity with the same name+type.

        The other entity's properties take precedence for conflicting keys.
        Returns a new Entity with merged properties.
        """
        merged_props = {**self.properties, **other.properties}
        merged_source = self.source
        if other.source and other.source not in self.source:
            merged_source = f"{self.source},{other.source}" if self.source else other.source
        return Entity(
            name=self.name,
            type=self.type,
            properties=merged_props,
            source=merged_source,
            id=self.id,
        )

    def __repr__(self):
        props_str = f", properties={self.properties}" if self.properties else ""
        source_str = f", source='{self.source}'" if self.source else ""
        return f"Entity(name='{self.name}', type='{self.type}'{props_str}{source_str})"
