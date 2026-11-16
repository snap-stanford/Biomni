"""Offline tests for biomni.tool.literature.query_pubmed.

PubMed is replaced by a fake, so these tests make no network calls and do not need pymed installed.
"""

import sys
import types

import pytest
from biomni.tool import literature


class FakePaper:
    def __init__(self, title):
        self.title = title
        self.abstract = f"Abstract of {title}"
        self.journal = "Test Journal"


@pytest.fixture
def pubmed(monkeypatch):
    """Install a fake pymed.PubMed that returns papers only for the queries listed in ``results``."""
    state = {"results": {}, "calls": []}

    class FakePubMed:
        def __init__(self, tool, email):
            pass

        def query(self, query, max_results):
            state["calls"].append(query)
            return iter(state["results"].get(query, [])[:max_results])

    monkeypatch.setitem(sys.modules, "pymed", types.SimpleNamespace(PubMed=FakePubMed))
    monkeypatch.setattr(literature.time, "sleep", lambda seconds: None)
    return state


def test_exact_match_is_returned_without_a_note(pubmed):
    pubmed["results"]["noonan syndrome case report"] = [FakePaper("Paper A")]

    out = literature.query_pubmed("noonan syndrome case report")

    assert out.startswith("Title: Paper A")
    assert "Note:" not in out
    assert pubmed["calls"] == ["noonan syndrome case report"]


def test_results_from_a_broadened_query_are_labelled(pubmed):
    pubmed["results"]["noonan syndrome"] = [FakePaper("Paper B")]

    out = literature.query_pubmed("noonan syndrome case report")

    assert pubmed["calls"] == ["noonan syndrome case report", "noonan syndrome case", "noonan syndrome"]
    first_line = out.splitlines()[0]
    assert first_line.startswith("Note:")
    assert "'noonan syndrome case report'" in first_line
    assert "'noonan syndrome'" in first_line
    assert "Title: Paper B" in out


def test_max_retries_zero_searches_the_exact_query_only(pubmed):
    pubmed["results"]["noonan syndrome"] = [FakePaper("Paper B")]

    out = literature.query_pubmed("noonan syndrome case report", max_retries=0)

    assert pubmed["calls"] == ["noonan syndrome case report"]
    assert "Paper B" not in out
    assert out.startswith("No papers found")


def test_identical_query_is_not_resent_when_no_words_are_left_to_drop(pubmed):
    out = literature.query_pubmed("zzzunknownterm")

    assert pubmed["calls"] == ["zzzunknownterm"]
    assert out.startswith("No papers found")


def test_no_results_message_lists_every_query_tried(pubmed):
    out = literature.query_pubmed("alpha beta gamma")

    assert pubmed["calls"] == ["alpha beta gamma", "alpha beta", "alpha"]
    assert out.startswith("No papers found")
    for query in pubmed["calls"]:
        assert f"'{query}'" in out


def test_max_papers_is_passed_through(pubmed):
    pubmed["results"]["tp53"] = [FakePaper(f"Paper {i}") for i in range(5)]

    out = literature.query_pubmed("tp53", max_papers=2)

    assert out.count("Title: ") == 2
