"""Tests for the Research Knowledge Graph module."""

import os
import tempfile
from pathlib import Path

import pytest
from biomni.knowledge_graph import (
    Entity,
    KnowledgeGraph,
    Relation,
    Storage,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def kg(temp_db):
    """Create a KnowledgeGraph with a temporary database."""
    graph = KnowledgeGraph(temp_db)
    yield graph
    graph.close()


class TestEntity:
    def test_entity_creation(self):
        entity = Entity(name="TP53", type="Gene")
        assert entity.name == "TP53"
        assert entity.type == "Gene"
        assert entity.id == "Gene:TP53"
        assert entity.properties == {}
        assert entity.source == ""

    def test_entity_with_properties(self):
        entity = Entity(
            name="TP53",
            type="Gene",
            properties={"species": "human", "uniprot_id": "P04637"},
            source="query_uniprot",
        )
        assert entity.properties["species"] == "human"
        assert entity.source == "query_uniprot"

    def test_entity_equality(self):
        e1 = Entity(name="TP53", type="Gene")
        e2 = Entity(name="TP53", type="Gene", properties={"species": "human"})
        assert e1 == e2  # Same name+type = equal

    def test_entity_inequality(self):
        e1 = Entity(name="TP53", type="Gene")
        e2 = Entity(name="BAX", type="Gene")
        assert e1 != e2

    def test_entity_hash(self):
        e1 = Entity(name="TP53", type="Gene")
        e2 = Entity(name="TP53", type="Gene", properties={"species": "human"})
        assert hash(e1) == hash(e2)

    def test_entity_merge_properties(self):
        e1 = Entity(name="TP53", type="Gene", properties={"species": "human"})
        e2 = Entity(name="TP53", type="Gene", properties={"uniprot_id": "P04637"})
        merged = e1.merge_properties(e2)
        assert merged.properties["species"] == "human"
        assert merged.properties["uniprot_id"] == "P04637"

    def test_entity_to_dict(self):
        entity = Entity(name="TP53", type="Gene", properties={"species": "human"})
        d = entity.to_dict()
        assert d["name"] == "TP53"
        assert d["type"] == "Gene"
        assert d["properties"]["species"] == "human"

    def test_entity_from_dict(self):
        d = {"name": "TP53", "type": "Gene", "properties": {"species": "human"}, "source": "", "id": "Gene:TP53"}
        entity = Entity.from_dict(d)
        assert entity.name == "TP53"
        assert entity.type == "Gene"


class TestRelation:
    def test_relation_creation(self):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        rel = Relation(tp53, bax, type="REGULATES")
        assert rel.source_entity == tp53
        assert rel.target_entity == bax
        assert rel.type == "REGULATES"
        assert rel.id == "Gene:TP53--REGULATES-->Gene:BAX"

    def test_relation_with_properties(self):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        rel = Relation(tp53, bax, type="REGULATES", properties={"confidence": 0.95})
        assert rel.properties["confidence"] == 0.95

    def test_relation_equality(self):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        r1 = Relation(tp53, bax, type="REGULATES")
        r2 = Relation(tp53, bax, type="REGULATES", properties={"confidence": 0.95})
        assert r1 == r2  # Same source+target+type = equal

    def test_relation_inequality(self):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        r1 = Relation(tp53, bax, type="REGULATES")
        r2 = Relation(bax, tp53, type="REGULATES")  # Reversed
        assert r1 != r2

    def test_relation_reverse(self):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        rel = Relation(tp53, bax, type="REGULATES")
        reversed_rel = rel.reverse()
        assert reversed_rel.source_entity == bax
        assert reversed_rel.target_entity == tp53

    def test_relation_to_dict(self):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        rel = Relation(tp53, bax, type="REGULATES")
        d = rel.to_dict()
        assert d["type"] == "REGULATES"
        assert d["source_entity"]["name"] == "TP53"
        assert d["target_entity"]["name"] == "BAX"

    def test_relation_from_dict(self):
        d = {
            "id": "Gene:TP53--REGULATES-->Gene:BAX",
            "source_entity": {"name": "TP53", "type": "Gene", "properties": {}, "source": "", "id": "Gene:TP53"},
            "target_entity": {"name": "BAX", "type": "Gene", "properties": {}, "source": "", "id": "Gene:BAX"},
            "type": "REGULATES",
            "properties": {},
            "source": "",
        }
        rel = Relation.from_dict(d)
        assert rel.source_entity.name == "TP53"
        assert rel.target_entity.name == "BAX"
        assert rel.type == "REGULATES"


class TestStorage:
    def test_storage_creation(self, temp_db):
        storage = Storage(temp_db)
        assert storage.db_path == Path(temp_db)
        storage.close()

    def test_save_and_load_entity(self, temp_db):
        storage = Storage(temp_db)
        entity = Entity(name="TP53", type="Gene", properties={"species": "human"})
        storage.save_entity(entity)

        loaded = storage.load_entity("Gene:TP53")
        assert loaded is not None
        assert loaded.name == "TP53"
        assert loaded.properties["species"] == "human"
        storage.close()

    def test_save_and_load_relation(self, temp_db):
        storage = Storage(temp_db)
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        rel = Relation(tp53, bax, type="REGULATES")

        storage.save_relation(rel)

        loaded = storage.load_relation(rel.id)
        assert loaded is not None
        assert loaded.source_entity.name == "TP53"
        assert loaded.target_entity.name == "BAX"
        assert loaded.type == "REGULATES"
        storage.close()

    def test_query_entities_by_type(self, temp_db):
        storage = Storage(temp_db)
        storage.save_entity(Entity(name="TP53", type="Gene"))
        storage.save_entity(Entity(name="BAX", type="Gene"))
        storage.save_entity(Entity(name="Vemurafenib", type="Drug"))

        genes = storage.query_entities_by_type("Gene")
        assert len(genes) == 2
        storage.close()

    def test_query_entities_by_name(self, temp_db):
        storage = Storage(temp_db)
        storage.save_entity(Entity(name="TP53", type="Gene"))
        storage.save_entity(Entity(name="TP53-Mutant", type="Variant"))

        results = storage.query_entities_by_name("TP53")
        assert len(results) == 2
        storage.close()

    def test_get_stats(self, temp_db):
        storage = Storage(temp_db)
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        storage.save_entity(tp53)
        storage.save_entity(bax)
        storage.save_relation(Relation(tp53, bax, type="REGULATES"))

        stats = storage.get_stats()
        assert stats["entity_count"] == 2
        assert stats["relation_count"] == 1
        assert stats["entity_types"]["Gene"] == 2
        assert stats["relation_types"]["REGULATES"] == 1
        storage.close()


class TestKnowledgeGraph:
    def test_kg_creation(self, kg):
        assert kg is not None
        stats = kg.get_stats()
        assert stats["entity_count"] == 0
        assert stats["relation_count"] == 0

    def test_add_entity(self, kg):
        entity = Entity(name="TP53", type="Gene")
        kg.add_entity(entity)
        stats = kg.get_stats()
        assert stats["entity_count"] == 1

    def test_get_entity(self, kg):
        kg.add_entity(Entity(name="TP53", type="Gene"))
        entity = kg.get_entity("Gene:TP53")
        assert entity is not None
        assert entity.name == "TP53"

    def test_query_entities(self, kg):
        kg.add_entity(Entity(name="TP53", type="Gene"))
        kg.add_entity(Entity(name="BAX", type="Gene"))
        kg.add_entity(Entity(name="Vemurafenib", type="Drug"))

        genes = kg.query_entities(entity_type="Gene")
        assert len(genes) == 2

        tp53 = kg.query_entities(name="TP53")
        assert len(tp53) == 1
        assert tp53[0].name == "TP53"

    def test_add_relation(self, kg):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        rel = Relation(tp53, bax, type="REGULATES")
        kg.add_relation(rel)

        stats = kg.get_stats()
        assert stats["entity_count"] == 2  # Auto-added
        assert stats["relation_count"] == 1

    def test_get_outgoing_relations(self, kg):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        kg.add_relation(Relation(tp53, bax, type="REGULATES"))

        outgoing = kg.get_outgoing_relations("Gene:TP53")
        assert len(outgoing) == 1
        assert outgoing[0].target_entity.name == "BAX"

    def test_get_incoming_relations(self, kg):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        kg.add_relation(Relation(tp53, bax, type="REGULATES"))

        incoming = kg.get_incoming_relations("Gene:BAX")
        assert len(incoming) == 1
        assert incoming[0].source_entity.name == "TP53"

    def test_get_neighbors(self, kg):
        tp53 = Entity(name="TP53", type="Gene")
        bax = Entity(name="BAX", type="Gene")
        mdm2 = Entity(name="MDM2", type="Gene")
        kg.add_relation(Relation(tp53, bax, type="REGULATES"))
        kg.add_relation(Relation(mdm2, tp53, type="REGULATES"))

        neighbors = kg.get_neighbors("Gene:TP53", depth=1)
        assert len(neighbors["entities"]) == 3  # TP53, BAX, MDM2
        assert len(neighbors["relations"]) == 2

    def test_get_context_for_query(self, kg):
        tp53 = Entity(name="TP53", type="Gene", properties={"species": "human"})
        bax = Entity(name="BAX", type="Gene")
        kg.add_relation(Relation(tp53, bax, type="REGULATES"))

        context = kg.get_context_for_query("What does TP53 regulate?")
        assert "TP53" in context
        assert "BAX" in context
        assert "REGULATES" in context

    def test_persistence(self, temp_db):
        kg1 = KnowledgeGraph(temp_db)
        kg1.add_entity(Entity(name="TP53", type="Gene"))
        kg1.close()

        kg2 = KnowledgeGraph(temp_db)
        entity = kg2.get_entity("Gene:TP53")
        assert entity is not None
        assert entity.name == "TP53"
        kg2.close()

    def test_repr(self, kg):
        kg.add_entity(Entity(name="TP53", type="Gene"))
        repr_str = repr(kg)
        assert "KnowledgeGraph" in repr_str
        assert "entities=1" in repr_str
