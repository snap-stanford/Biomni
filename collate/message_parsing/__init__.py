"""Utilities for parsing structured agent messages into typed blocks.

Exposes:
- parse_blocks(text): Extract ordered blocks for tags: execute, solution, observation, think
- StreamingBlockParser: Incremental parser that emits block start/delta/end events
"""

from .parser import Block, BlockEvent, parse_blocks, StreamingBlockParser

__all__ = [
    "Block",
    "BlockEvent",
    "parse_blocks",
    "StreamingBlockParser",
]


