"""SQLite-based persistence for the Research Knowledge Graph.

This module provides storage and retrieval of entities and relations using SQLite.
SQLite was chosen for zero external dependencies and portability.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .entity import Entity
from .relation import Relation


class Storage:
    """SQLite-based storage for the knowledge graph.

    Attributes:
        db_path: Path to the SQLite database file.
        conn: Active database connection.
    """

    def __init__(self, db_path: str | Path = "biomni_kg.db"):
        """Initialize storage with the given database path.

        Args:
            db_path: Path to the SQLite database file. If the file doesn't exist,
                     it will be created.
        """
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create the entities and relations tables if they don't exist."""
        cursor = self.conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT '',
                fingerprint TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_fingerprint
            ON entities(fingerprint)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_type
            ON entities(type)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_entities_name
            ON entities(name)
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS relations (
                id TEXT PRIMARY KEY,
                source_entity_id TEXT NOT NULL,
                target_entity_id TEXT NOT NULL,
                type TEXT NOT NULL,
                properties TEXT NOT NULL DEFAULT '{}',
                source TEXT NOT NULL DEFAULT '',
                fingerprint TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_entity_id) REFERENCES entities(id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_fingerprint
            ON relations(fingerprint)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_source_entity
            ON relations(source_entity_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_target_entity
            ON relations(target_entity_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_relations_type
            ON relations(type)
        """)

        self.conn.commit()

    def save_entity(self, entity: Entity) -> None:
        """Save or update an entity in the database.

        If an entity with the same fingerprint exists, properties are merged.

        Args:
            entity: The entity to save.
        """
        cursor = self.conn.cursor()

        # Check if entity with same fingerprint exists
        cursor.execute(
            "SELECT id, properties, source FROM entities WHERE fingerprint = ?",
            (entity.fingerprint(),),
        )
        existing = cursor.fetchone()

        if existing:
            # Merge properties with existing entity
            existing_props = json.loads(existing["properties"])
            merged_props = {**existing_props, **entity.properties}
            merged_source = existing["source"]
            if entity.source and entity.source not in existing["source"]:
                merged_source = f"{existing['source']},{entity.source}" if existing["source"] else entity.source

            cursor.execute(
                """
                UPDATE entities
                SET properties = ?, source = ?, updated_at = CURRENT_TIMESTAMP
                WHERE fingerprint = ?
                """,
                (json.dumps(merged_props), merged_source, entity.fingerprint()),
            )
        else:
            # Insert new entity
            cursor.execute(
                """
                INSERT INTO entities (id, name, type, properties, source, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entity.id,
                    entity.name,
                    entity.type,
                    json.dumps(entity.properties),
                    entity.source,
                    entity.fingerprint(),
                ),
            )

        self.conn.commit()

    def save_relation(self, relation: Relation) -> None:
        """Save or update a relation in the database.

        If a relation with the same fingerprint exists, properties are merged.
        Source and target entities are saved automatically if not already present.

        Args:
            relation: The relation to save.
        """
        # Ensure source and target entities exist
        self.save_entity(relation.source_entity)
        self.save_entity(relation.target_entity)

        cursor = self.conn.cursor()

        # Check if relation with same fingerprint exists
        cursor.execute(
            "SELECT id, properties, source FROM relations WHERE fingerprint = ?",
            (relation.fingerprint(),),
        )
        existing = cursor.fetchone()

        if existing:
            # Merge properties with existing relation
            existing_props = json.loads(existing["properties"])
            merged_props = {**existing_props, **relation.properties}
            merged_source = existing["source"]
            if relation.source and relation.source not in existing["source"]:
                merged_source = f"{existing['source']},{relation.source}" if existing["source"] else relation.source

            cursor.execute(
                """
                UPDATE relations
                SET properties = ?, source = ?, updated_at = CURRENT_TIMESTAMP
                WHERE fingerprint = ?
                """,
                (json.dumps(merged_props), merged_source, relation.fingerprint()),
            )
        else:
            # Insert new relation
            cursor.execute(
                """
                INSERT INTO relations
                    (id, source_entity_id, target_entity_id, type, properties, source, fingerprint)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    relation.id,
                    relation.source_entity.id,
                    relation.target_entity.id,
                    relation.type,
                    json.dumps(relation.properties),
                    relation.source,
                    relation.fingerprint(),
                ),
            )

        self.conn.commit()

    def load_entity(self, entity_id: str) -> Entity | None:
        """Load an entity by its ID.

        Args:
            entity_id: The entity ID (e.g., "Gene:TP53").

        Returns:
            The entity if found, None otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities WHERE id = ?", (entity_id,))
        row = cursor.fetchone()

        if not row:
            return None

        return Entity(
            id=row["id"],
            name=row["name"],
            type=row["type"],
            properties=json.loads(row["properties"]),
            source=row["source"],
        )

    def load_relation(self, relation_id: str) -> Relation | None:
        """Load a relation by its ID.

        Args:
            relation_id: The relation ID.

        Returns:
            The relation if found, None otherwise.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM relations WHERE id = ?", (relation_id,))
        row = cursor.fetchone()

        if not row:
            return None

        source_entity = self.load_entity(row["source_entity_id"])
        target_entity = self.load_entity(row["target_entity_id"])

        if not source_entity or not target_entity:
            return None

        return Relation(
            id=row["id"],
            source_entity=source_entity,
            target_entity=target_entity,
            type=row["type"],
            properties=json.loads(row["properties"]),
            source=row["source"],
        )

    def load_all_entities(self) -> list[Entity]:
        """Load all entities from the database.

        Returns:
            List of all entities.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM entities")
        rows = cursor.fetchall()

        return [
            Entity(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                properties=json.loads(row["properties"]),
                source=row["source"],
            )
            for row in rows
        ]

    def load_all_relations(self) -> list[Relation]:
        """Load all relations from the database.

        Returns:
            List of all relations.
        """
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM relations")
        rows = cursor.fetchall()

        relations = []
        for row in rows:
            source_entity = self.load_entity(row["source_entity_id"])
            target_entity = self.load_entity(row["target_entity_id"])

            if source_entity and target_entity:
                relations.append(
                    Relation(
                        id=row["id"],
                        source_entity=source_entity,
                        target_entity=target_entity,
                        type=row["type"],
                        properties=json.loads(row["properties"]),
                        source=row["source"],
                    )
                )

        return relations

    def query_entities_by_type(self, entity_type: str) -> list[Entity]:
        """Query entities by type.

        Args:
            entity_type: The entity type to filter by (e.g., "Gene", "Drug").

        Returns:
            List of entities matching the type.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM entities WHERE type = ?",
            (entity_type,),
        )
        rows = cursor.fetchall()

        return [
            Entity(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                properties=json.loads(row["properties"]),
                source=row["source"],
            )
            for row in rows
        ]

    def query_entities_by_name(self, name: str) -> list[Entity]:
        """Query entities by name (case-insensitive partial match).

        Args:
            name: The name pattern to search for.

        Returns:
            List of entities matching the name pattern.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM entities WHERE name LIKE ?",
            (f"%{name}%",),
        )
        rows = cursor.fetchall()

        return [
            Entity(
                id=row["id"],
                name=row["name"],
                type=row["type"],
                properties=json.loads(row["properties"]),
                source=row["source"],
            )
            for row in rows
        ]

    def query_relations_by_source(self, entity_id: str) -> list[Relation]:
        """Query all relations where the given entity is the source.

        Args:
            entity_id: The source entity ID.

        Returns:
            List of relations originating from the entity.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM relations WHERE source_entity_id = ?",
            (entity_id,),
        )
        rows = cursor.fetchall()

        relations = []
        for row in rows:
            source_entity = self.load_entity(row["source_entity_id"])
            target_entity = self.load_entity(row["target_entity_id"])

            if source_entity and target_entity:
                relations.append(
                    Relation(
                        id=row["id"],
                        source_entity=source_entity,
                        target_entity=target_entity,
                        type=row["type"],
                        properties=json.loads(row["properties"]),
                        source=row["source"],
                    )
                )

        return relations

    def query_relations_by_target(self, entity_id: str) -> list[Relation]:
        """Query all relations where the given entity is the target.

        Args:
            entity_id: The target entity ID.

        Returns:
            List of relations pointing to the entity.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM relations WHERE target_entity_id = ?",
            (entity_id,),
        )
        rows = cursor.fetchall()

        relations = []
        for row in rows:
            source_entity = self.load_entity(row["source_entity_id"])
            target_entity = self.load_entity(row["target_entity_id"])

            if source_entity and target_entity:
                relations.append(
                    Relation(
                        id=row["id"],
                        source_entity=source_entity,
                        target_entity=target_entity,
                        type=row["type"],
                        properties=json.loads(row["properties"]),
                        source=row["source"],
                    )
                )

        return relations

    def query_relations_by_type(self, relation_type: str) -> list[Relation]:
        """Query all relations of a given type.

        Args:
            relation_type: The relation type to filter by (e.g., "REGULATES", "TARGETS").

        Returns:
            List of relations matching the type.
        """
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT * FROM relations WHERE type = ?",
            (relation_type,),
        )
        rows = cursor.fetchall()

        relations = []
        for row in rows:
            source_entity = self.load_entity(row["source_entity_id"])
            target_entity = self.load_entity(row["target_entity_id"])

            if source_entity and target_entity:
                relations.append(
                    Relation(
                        id=row["id"],
                        source_entity=source_entity,
                        target_entity=target_entity,
                        type=row["type"],
                        properties=json.loads(row["properties"]),
                        source=row["source"],
                    )
                )

        return relations

    def get_stats(self) -> dict[str, Any]:
        """Get statistics about the knowledge graph.

        Returns:
            Dictionary with entity count, relation count, and type breakdowns.
        """
        cursor = self.conn.cursor()

        cursor.execute("SELECT COUNT(*) as count FROM entities")
        entity_count = cursor.fetchone()["count"]

        cursor.execute("SELECT COUNT(*) as count FROM relations")
        relation_count = cursor.fetchone()["count"]

        cursor.execute("SELECT type, COUNT(*) as count FROM entities GROUP BY type ORDER BY count DESC")
        entity_types = {row["type"]: row["count"] for row in cursor.fetchall()}

        cursor.execute("SELECT type, COUNT(*) as count FROM relations GROUP BY type ORDER BY count DESC")
        relation_types = {row["type"]: row["count"] for row in cursor.fetchall()}

        return {
            "entity_count": entity_count,
            "relation_count": relation_count,
            "entity_types": entity_types,
            "relation_types": relation_types,
        }

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __del__(self):
        """Ensure connection is closed on garbage collection."""
        self.close()
