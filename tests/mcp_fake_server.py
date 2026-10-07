"""Minimal MCP server used by test_mcp_client.py (stdio by default, `--http PORT` for HTTP)."""

import asyncio
import os
import sys

from mcp.server.fastmcp import FastMCP

HTTP = len(sys.argv) == 3 and sys.argv[1] == "--http"
mcp = FastMCP("fake", port=int(sys.argv[2]) if HTTP else 8000)


@mcp.tool()
def server_pid() -> str:
    """Return the server's process id."""
    return str(os.getpid())


@mcp.tool()
def echo(text: str, times: int = 1) -> str:
    """Repeat text."""
    return text * times


@mcp.tool()
def fail(message: str) -> str:
    """Always raise."""
    raise ValueError(message)


@mcp.tool()
async def slow(seconds: float) -> str:
    """Sleep without blocking the server, then answer."""
    await asyncio.sleep(seconds)
    return "done"


@mcp.tool()
def crash() -> str:
    """Kill the server process without answering."""
    os._exit(1)


if __name__ == "__main__":
    mcp.run(transport="streamable-http" if HTTP else "stdio")
