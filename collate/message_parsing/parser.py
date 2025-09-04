from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from typing import Generator, Iterable, Iterator, List, Literal, Optional, Tuple


BlockType = Literal["execute", "solution", "observation", "think"]


@dataclass
class Block:
    type: BlockType
    text: str


@dataclass
class BlockEvent:
    event: Literal["block_start", "delta", "block_end"]
    block_id: str
    block_type: Optional[BlockType] = None
    text: Optional[str] = None


_TAG_TO_TYPE = {
    "execute": "execute",
    "solution": "solution",
    "observation": "observation",
    "think": "think",
}


_OPEN_TAG_RE = re.compile(r"<(execute|solution|observation|think)>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</(execute|solution|observation|think)>", re.IGNORECASE)


def parse_blocks(text: str) -> List[Block]:
    """Parse complete text and extract ordered non-overlapping blocks.

    Only returns well-formed tag pairs; content outside tags is ignored.
    Tags are case-insensitive; block types are normalized to lowercase.
    """
    if not text:
        return []

    blocks: List[Block] = []
    # Scan for matching pairs without nesting support
    idx = 0
    n = len(text)
    while idx < n:
        open_match = _OPEN_TAG_RE.search(text, idx)
        if not open_match:
            break
        tag = open_match.group(1).lower()
        start = open_match.end()
        close_match = _CLOSE_TAG_RE.search(text, start)
        if not close_match:
            # No closing tag; stop parsing further to avoid partial capture
            break
        close_tag = close_match.group(1).lower()
        if close_tag != tag:
            # Mismatched tags: advance past the open and continue
            idx = start
            continue
        inner = text[start:close_match.start()]
        blocks.append(Block(type=_TAG_TO_TYPE[tag], text=inner))
        idx = close_match.end()
    return blocks


class StreamingBlockParser:
    """Incremental parser for <execute>, <solution>, <observation>, <think> blocks.

    Feed text deltas via .feed() and receive events: block_start, delta, block_end.
    Maintains minimal buffer to handle tag boundaries across chunks.
    Does not support nested blocks.
    """

    def __init__(self) -> None:
        self._buffer: str = ""
        self._current_block_id: Optional[str] = None
        self._current_block_type: Optional[BlockType] = None

    def _emit_start(self, block_type: BlockType) -> BlockEvent:
        self._current_block_id = str(uuid.uuid4())
        self._current_block_type = block_type
        return BlockEvent(event="block_start", block_id=self._current_block_id, block_type=block_type)

    def _emit_delta(self, text: str) -> BlockEvent:
        assert self._current_block_id is not None
        return BlockEvent(event="delta", block_id=self._current_block_id, text=text)

    def _emit_end(self) -> BlockEvent:
        assert self._current_block_id is not None
        evt = BlockEvent(event="block_end", block_id=self._current_block_id)
        self._current_block_id = None
        self._current_block_type = None
        return evt

    def feed(self, delta: str) -> Iterator[BlockEvent]:
        """Feed a text delta and yield block events incrementally."""
        if not delta:
            return
        self._buffer += delta

        while True:
            if self._current_block_type is None:
                # Look for an opening tag
                open_match = _OPEN_TAG_RE.search(self._buffer)
                if not open_match:
                    # Keep buffer small to avoid OOM on untagged streams; trim older content
                    if len(self._buffer) > 4096:
                        self._buffer = self._buffer[-1024:]
                    break
                # Discard anything before opening tag
                self._buffer = self._buffer[open_match.start():]
                tag = open_match.group(1).lower()
                # Remove the open tag from buffer
                self._buffer = self._buffer[open_match.end() - open_match.start():]
                yield self._emit_start(_TAG_TO_TYPE[tag])
                continue

            # In a block: look for closing tag for the current type
            assert self._current_block_type is not None
            close_match = re.search(rf"</{self._current_block_type}>", self._buffer, re.IGNORECASE)
            if not close_match:
                # No close yet: emit all buffered text (if any) as delta and clear
                if self._buffer:
                    yield self._emit_delta(self._buffer)
                    self._buffer = ""
                break

            # We found a closing tag: emit text up to it, end the block, and remove consumed part
            text_before_close = self._buffer[: close_match.start()]
            if text_before_close:
                yield self._emit_delta(text_before_close)
            # Remove text up to and including the closing tag
            self._buffer = self._buffer[close_match.end():]
            yield self._emit_end()
            # Continue to look for next opening tag in the remaining buffer


