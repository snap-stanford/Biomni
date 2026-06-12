"""Core Knowledge Graph for biomedical research.

This module provides the KnowledgeGraph class, which manages entities and relations
for accumulating biomedical knowledge across agent sessions. It serves as the
foundation for the Research Knowledge Graph feature in Biomni.

The KnowledgeGraph provides:
- Entity and relation management (add, query, merge)
- Persistence via SQLite (zero external dependencies)
- Query interface for retrieving relevant context
- Statistics and introspection

Examples:
    >>> from biomni.knowledge_graph import KnowledgeGraph, Entity, Relation
    >>> kg = KnowledgeGraph("my_research.db")
    >>> tp53 = Entity(name="TP53", type="Gene", properties={"species": "human"})
    >>> bax = Entity(name="BAX", type="Gene")
    >>> kg.add_entity(tp53)
    >>> kg.add_entity(bax)
    >>> kg.add_relation(Relation(tp53, bax, type="REGULATES"))
    >>> kg.query("TP53")
    [Entity(name='TP53', type='Gene', properties={'species': 'human'})]
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .storage import Storage

if TYPE_CHECKING:
    from pathlib import Path

    from .entity import Entity
    from .relation import Relation


class KnowledgeGraph:
    """A knowledge graph for accumulating biomedical research knowledge.

    The KnowledgeGraph stores entities (genes, proteins, drugs, diseases, etc.)
    and their relationships. It persists to SQLite for cross-session memory.

    Attributes:
        storage: The underlying SQLite storage backend.

    Examples:
        >>> kg = KnowledgeGraph()
        >>> gene = Entity(name="BRAF", type="Gene")
        >>> drug = Entity(name="Vemurafenib", type="Drug")
        >>> kg.add_entity(gene)
        >>> kg.add_entity(drug)
        >>> kg.add_relation(Relation(drug, gene, type="TARGETS"))
        >>> kg.get_stats()
        {'entity_count': 2, 'relation_count': 1, ...}
    """

    def __init__(self, db_path: str | Path = "biomni_kg.db"):
        """Initialize the knowledge graph.

        Args:
            db_path: Path to the SQLite database file. If the file doesn't exist,
                     it will be created. Defaults to "biomni_kg.db" in the current
                     directory.
        """
        self.storage = Storage(db_path)

    # -------------------------------------------------------------------------
    # Entity operations
    # -------------------------------------------------------------------------

    def add_entity(self, entity: Entity) -> Entity:
        """Add an entity to the knowledge graph.

        If an entity with the same name and type already exists, properties are
        merged (existing properties take precedence for conflicts).

        Args:
            entity: The entity to add.

        Returns:
            The entity that was added (or merged with existing).
        """
        self.storage.save_entity(entity)
        return entity

    def get_entity(self, entity_id: str) -> Entity | None:
        """Get an entity by its ID.

        Args:
            entity_id: The entity ID (e.g., "Gene:TP53").

        Returns:
            The entity if found, None otherwise.
        """
        return self.storage.load_entity(entity_id)

    def query_entities(self, name: str | None = None, entity_type: str | None = None) -> list[Entity]:
        """Query entities by name pattern and/or type.

        Args:
            name: Optional name pattern (case-insensitive partial match).
            entity_type: Optional entity type filter (e.g., "Gene", "Drug").

        Returns:
            List of entities matching the query.
        """
        if name and entity_type:
            # Filter by both name and type
            entities = self.storage.query_entities_by_name(name)
            return [e for e in entities if e.type == entity_type]
        elif name:
            return self.storage.query_entities_by_name(name)
        elif entity_type:
            return self.storage.query_entities_by_type(entity_type)
        else:
            return self.storage.load_all_entities()

    def get_all_entities(self) -> list[Entity]:
        """Get all entities in the knowledge graph.

        Returns:
            List of all entities.
        """
        return self.storage.load_all_entities()

    # -------------------------------------------------------------------------
    # Relation operations
    # -------------------------------------------------------------------------

    def add_relation(self, relation: Relation) -> Relation:
        """Add a relation to the knowledge graph.

        Source and target entities are automatically added if not already present.
        If a relation with the same source, target, and type already exists,
        properties are merged.

        Args:
            relation: The relation to add.

        Returns:
            The relation that was added (or merged with existing).
        """
        self.storage.save_relation(relation)
        return relation

    def get_relation(self, relation_id: str) -> Relation | None:
        """Get a relation by its ID.

        Args:
            relation_id: The relation ID.

        Returns:
            The relation if found, None otherwise.
        """
        return self.storage.load_relation(relation_id)

    def query_relations(
        self,
        source_entity_id: str | None = None,
        target_entity_id: str | None = None,
        relation_type: str | None = None,
    ) -> list[Relation]:
        """Query relations by source entity, target entity, and/or type.

        Args:
            source_entity_id: Optional source entity ID filter.
            target_entity_id: Optional target entity ID filter.
            relation_type: Optional relation type filter (e.g., "REGULATES").

        Returns:
            List of relations matching the query.
        """
        if source_entity_id and target_entity_id and relation_type:
            # Filter by all three
            relations = self.storage.query_relations_by_source(source_entity_id)
            return [r for r in relations if r.target_entity.id == target_entity_id and r.type == relation_type]
        elif source_entity_id and target_entity_id:
            relations = self.storage.query_relations_by_source(source_entity_id)
            return [r for r in relations if r.target_entity.id == target_entity_id]
        elif source_entity_id and relation_type:
            relations = self.storage.query_relations_by_source(source_entity_id)
            return [r for r in relations if r.type == relation_type]
        elif target_entity_id and relation_type:
            relations = self.storage.query_relations_by_target(target_entity_id)
            return [r for r in relations if r.type == relation_type]
        elif source_entity_id:
            return self.storage.query_relations_by_source(source_entity_id)
        elif target_entity_id:
            return self.storage.query_relations_by_target(target_entity_id)
        elif relation_type:
            return self.storage.query_relations_by_type(relation_type)
        else:
            return self.storage.load_all_relations()

    def get_outgoing_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations originating from an entity.

        Args:
            entity_id: The source entity ID.

        Returns:
            List of outgoing relations.
        """
        return self.storage.query_relations_by_source(entity_id)

    def get_incoming_relations(self, entity_id: str) -> list[Relation]:
        """Get all relations pointing to an entity.

        Args:
            entity_id: The target entity ID.

        Returns:
            List of incoming relations.
        """
        return self.storage.query_relations_by_target(entity_id)

    def get_all_relations(self) -> list[Relation]:
        """Get all relations in the knowledge graph.

        Returns:
            List of all relations.
        """
        return self.storage.load_all_relations()

    # -------------------------------------------------------------------------
    # Graph traversal and context retrieval
    # -------------------------------------------------------------------------

    def get_neighbors(self, entity_id: str, depth: int = 1) -> dict[str, Any]:
        """Get neighboring entities and relations up to a given depth.

        This is useful for retrieving context around a specific entity.

        Args:
            entity_id: The starting entity ID.
            depth: How many hops to traverse (default: 1).

        Returns:
            Dictionary with 'entities' and 'relations' lists.
        """
        visited_entities: set[str] = set()
        visited_relations: set[str] = set()
        entities: list[Entity] = []
        relations: list[Relation] = []

        # BFS traversal
        queue = [(entity_id, 0)]

        while queue:
            current_id, current_depth = queue.pop(0)

            if current_id in visited_entities:
                continue
            visited_entities.add(current_id)

            # Get entity
            entity = self.get_entity(current_id)
            if entity:
                entities.append(entity)

            if current_depth >= depth:
                continue

            # Get outgoing relations
            for rel in self.get_outgoing_relations(current_id):
                if rel.id not in visited_relations:
                    visited_relations.add(rel.id)
                    relations.append(rel)
                    queue.append((rel.target_entity.id, current_depth + 1))

            # Get incoming relations
            for rel in self.get_incoming_relations(current_id):
                if rel.id not in visited_relations:
                    visited_relations.add(rel.id)
                    relations.append(rel)
                    queue.append((rel.source_entity.id, current_depth + 1))

        return {"entities": entities, "relations": relations}

    def get_context_for_query(self, query: str, max_entities: int = 10, max_relations: int = 20) -> str:
        """Generate context string for injection into agent system prompt.

        This method extracts relevant entities and relations from the knowledge graph
        that might be useful for answering the given query. Currently uses simple
        name matching; future versions will use semantic similarity.

        Args:
            query: The user's query text.
            max_entities: Maximum number of entities to include.
            max_relations: Maximum number of relations to include.

        Returns:
            Formatted string with relevant knowledge graph context.
        """
        # Simple keyword-based matching (Phase 1)
        # Future: semantic similarity with embeddings
        query_lower = query.lower()
        words = query_lower.split()

        # Find entities whose names appear in the query
        relevant_entities: list[Entity] = []
        for word in words:
            if len(word) < 3:  # Skip short words
                continue
            matches = self.query_entities(name=word)
            relevant_entities.extend(matches)

        # Deduplicate and limit
        seen_ids: set[str] = set()
        unique_entities: list[Entity] = []
        for entity in relevant_entities:
            if entity.id not in seen_ids and len(unique_entities) < max_entities:
                seen_ids.add(entity.id)
                unique_entities.append(entity)

        # Get relations for these entities
        relevant_relations: list[Relation] = []
        for entity in unique_entities:
            if len(relevant_relations) >= max_relations:
                break
            outgoing = self.get_outgoing_relations(entity.id)
            incoming = self.get_incoming_relations(entity.id)
            relevant_relations.extend(outgoing)
            relevant_relations.extend(incoming)

        # Deduplicate relations
        seen_rel_ids: set[str] = set()
        unique_relations: list[Relation] = []
        for rel in relevant_relations:
            if rel.id not in seen_rel_ids and len(unique_relations) < max_relations:
                seen_rel_ids.add(rel.id)
                unique_relations.append(rel)

        # Format context string
        if not unique_entities and not unique_relations:
            return ""

        context_parts = []

        if unique_entities:
            context_parts.append("Relevant entities from prior research:")
            for entity in unique_entities:
                props_str = ""
                if entity.properties:
                    props_items = [f"{k}={v}" for k, v in list(entity.properties.items())[:3]]
                    props_str = f" ({', '.join(props_items)})"
                context_parts.append(f"  - {entity.type}: {entity.name}{props_str}")

        if unique_relations:
            context_parts.append("\nRelevant relationships:")
            for rel in unique_relations:
                context_parts.append(f"  - {rel.source_entity.name} --[{rel.type}]--> {rel.target_entity.name}")

        return "\n".join(context_parts)

    # -------------------------------------------------------------------------
    # Statistics and introspection
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph.

        Returns:
            Dictionary with entity count, relation count, and type breakdowns.
        """
        return self.storage.get_stats()

    def __repr__(self):
        stats = self.get_stats()
        return f"KnowledgeGraph(entities={stats['entity_count']}, relations={stats['relation_count']})"

    def close(self):
        """Close the underlying storage connection."""
        self.storage.close()

    def __del__(self):
        """Ensure storage is closed on garbage collection."""
        self.close()
