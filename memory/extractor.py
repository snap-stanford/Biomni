"""Memory extractor: turns an agent trace into structured memory.

The extractor is LLM-agnostic: it accepts any LangChain `BaseChatModel` and
falls back to JSON parsing if `with_structured_output` is unavailable.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from .models import MemoryExtraction, TraceMessage
from .prompts import MEMORY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def trace_from_messages(messages: Sequence[BaseMessage]) -> list[TraceMessage]:
    """Convert LangChain messages into a normalized trace."""
    trace: list[TraceMessage] = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            trace.append(TraceMessage(type="human", content=str(msg.content)))
        elif isinstance(msg, ToolMessage):
            trace.append(
                TraceMessage(type="tool", content=str(msg.content), tool_result=str(msg.content))
            )
        elif isinstance(msg, AIMessage):
            content = str(msg.content)
            if "<observation>" in content:
                trace.append(TraceMessage(type="observation", content=content))
            elif "<execute>" in content:
                trace.append(TraceMessage(type="tool", content=content))
            else:
                trace.append(TraceMessage(type="ai", content=content))
        else:
            trace.append(TraceMessage(type="system", content=str(msg.content)))
    return trace


def trace_from_log(log: list) -> list[TraceMessage]:
    """Convert the A1 agent's `self.log` list into a normalized trace."""
    trace: list[TraceMessage] = []
    for entry in log:
        text = str(entry)
        if "Human Message" in text:
            trace.append(TraceMessage(type="human", content=text))
        elif "<observation>" in text:
            trace.append(TraceMessage(type="observation", content=text))
        elif "Ai Message" in text:
            trace.append(TraceMessage(type="ai", content=text))
        else:
            trace.append(TraceMessage(type="system", content=text))
    return trace


def _format_trace(trace: Sequence[TraceMessage]) -> str:
    lines: list[str] = []
    for i, m in enumerate(trace):
        head = f"[{i}] {m.type}"
        if m.tool_result:
            lines.append(f"{head}\n  content: {m.content}\n  result: {m.tool_result}")
        else:
            lines.append(f"{head}\n  {m.content}")
    return "\n".join(lines)


class MemoryExtractor:
    """Extract summary + facts from a conversation trace using a structured LLM call."""

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        self.llm = llm
        self._structured_llm = None
        if llm is not None:
            try:
                self._structured_llm = llm.with_structured_output(MemoryExtraction)
            except (NotImplementedError, AttributeError):
                self._structured_llm = None

    def _invoke_structured(self, prompt: str) -> MemoryExtraction:
        if self._structured_llm is not None:
            return self._structured_llm.invoke(prompt)
        return self._invoke_json(prompt)

    def _invoke_json(self, prompt: str) -> MemoryExtraction:
        """Fallback: ask for JSON and parse it ourselves."""
        json_prompt = (
            prompt
            + "\n\nRespond with ONLY a JSON object with keys `summary` (string) and "
            "`facts` (array of {entity, relation, value, confidence, source})."
        )
        raw = self.llm.invoke(json_prompt)
        text = raw.content if hasattr(raw, "content") else str(raw)
        if isinstance(text, list):
            text = "".join(
                block.get("text") or "" for block in text if isinstance(block, dict)
            )
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return MemoryExtraction.model_validate_json(match.group(0))
        raise ValueError("Could not parse structured memory extraction from LLM output")

    def extract(self, trace: Sequence[TraceMessage]) -> MemoryExtraction:
        """Synchronous extraction (blocks on the LLM call)."""
        if self.llm is None:
            raise RuntimeError("MemoryExtractor has no LLM configured")
        prompt = MEMORY_EXTRACTION_PROMPT.format(trace=_format_trace(trace))
        return self._invoke_structured(prompt)

    async def extract_async(self, trace: Sequence[TraceMessage]) -> MemoryExtraction:
        """Async extraction that does not block the event loop."""
        return await asyncio.to_thread(self.extract, trace)
