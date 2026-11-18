import sys
import types

import pytest


@pytest.fixture
def retriever_module(monkeypatch):
    monkeypatch.delitem(sys.modules, "biomni.model.retriever", raising=False)
    messages = types.ModuleType("langchain_core.messages")
    messages.HumanMessage = lambda content: content
    monkeypatch.setitem(sys.modules, "langchain_core", types.ModuleType("langchain_core"))
    monkeypatch.setitem(sys.modules, "langchain_core.messages", messages)
    openai = types.ModuleType("langchain_openai")
    openai.ChatOpenAI = object
    monkeypatch.setitem(sys.modules, "langchain_openai", openai)
    from biomni.model.retriever import ToolRetriever

    return ToolRetriever


def test_parser_keeps_valid_indices_adjacent_to_malformed_token(retriever_module):
    response = "TOOLS: [0, not-an-index, 2]\nDATA_LAKE: [1, nope]\nLIBRARIES: [2, bad]\nKNOW_HOW: [3, invalid]"

    assert retriever_module()._parse_llm_response(response) == {
        "tools": [0, 2],
        "data_lake": [1],
        "libraries": [2],
        "know_how": [3],
    }


class _Response:
    content = "TOOLS: [-1, 1]\nDATA_LAKE: []\nLIBRARIES: []"


class _LLM:
    def invoke(self, _messages):
        return _Response()


def test_retrieval_ignores_negative_and_out_of_range_indices(retriever_module):
    _Response.content = "TOOLS: [-1, 1, 9]\nDATA_LAKE: [-1, 0, 9]\nLIBRARIES: [-2, 1, 9]\nKNOW_HOW: [-3, 0, 9]"
    resources = {
        "tools": ["first", "second"],
        "data_lake": ["lake"],
        "libraries": ["lib0", "lib1"],
        "know_how": ["guide"],
    }

    assert retriever_module().prompt_based_retrieval("test", resources, llm=_LLM()) == {
        "tools": ["second"],
        "data_lake": ["lake"],
        "libraries": ["lib1"],
        "know_how": ["guide"],
    }
