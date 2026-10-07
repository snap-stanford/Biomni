"""Long-lived connections to MCP servers for `A1.add_mcp`.

Each configured server gets one MCP session that stays open for the life of the process,
owned by a private event loop on a background thread. Tool wrappers are plain synchronous
functions that submit calls to that loop, so they behave the same in scripts, in Jupyter
(where an event loop is already running) and in Biomni's code-execution threads.

Keeping the session open matters for servers that hold state: models loaded into GPU
memory, background jobs, caches. Starting a fresh server process for every call would
discard all of that.
"""

from __future__ import annotations

import asyncio
import atexit
import concurrent.futures
import json
import shlex
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

DEFAULT_TIMEOUT_S = 600.0


class MCPToolError(RuntimeError):
    """An MCP server could not be reached, or a tool call failed."""


@dataclass
class MCPServerConfig:
    """How to reach one MCP server: a local `command` (stdio) or a `url` (Streamable HTTP)."""

    name: str
    command: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = DEFAULT_TIMEOUT_S

    def __post_init__(self) -> None:
        if bool(self.command) == bool(self.url):
            raise ValueError(f"MCP server '{self.name}' needs exactly one of 'command' or 'url'")

    def describe(self) -> str:
        return self.url if self.url else shlex.join(self.command or [])


@asynccontextmanager
async def _open_transport(config: MCPServerConfig):
    """Yield (read_stream, write_stream) for the server, for both mcp 1.x and 2.x."""
    if config.command:
        cmd, *args = config.command
        params = StdioServerParameters(command=cmd, args=args, env=config.env or None)
        async with stdio_client(params) as (read, write):
            yield read, write
        return

    try:  # mcp 1.x
        from mcp.client.streamable_http import streamablehttp_client
    except ImportError:  # mcp 2.x renamed it and takes an httpx client instead of headers
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        async with create_mcp_http_client(headers=config.headers or None) as client:
            async with streamable_http_client(config.url, http_client=client) as streams:
                yield streams[0], streams[1]
        return
    async with streamablehttp_client(config.url, headers=config.headers or None) as streams:
        yield streams[0], streams[1]


def _root_cause(exc: BaseException) -> BaseException:
    """Unwrap exception groups (anyio task groups) down to the first real error."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def _error_message(exc: BaseException) -> str:
    cause = _root_cause(exc)
    return f"{type(cause).__name__}: {cause}" if str(cause) else type(cause).__name__


def _attr(obj: Any, camel: str, snake: str, default: Any = None) -> Any:
    """Read a protocol field under its mcp 1.x (camelCase) or 2.x (snake_case) name."""
    return getattr(obj, camel, getattr(obj, snake, default))


def result_to_text(result: Any) -> str:
    """Turn a CallToolResult into the text an agent should see; raise if the tool failed."""
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        if text is not None:
            parts.append(text)
        else:
            mime = _attr(item, "mimeType", "mime_type") or _attr(
                getattr(item, "resource", None), "mimeType", "mime_type"
            )
            parts.append(f"[{getattr(item, 'type', 'non-text')} content{f' ({mime})' if mime else ''} omitted]")
    structured = _attr(result, "structuredContent", "structured_content")
    if not parts and structured is not None:
        parts.append(json.dumps(structured))
    text = "\n".join(parts)
    if _attr(result, "isError", "is_error", False):
        raise MCPToolError(text or "the tool reported an error without a message")
    return text


class _Connection:
    """One attempt at holding a session open; replaced when the server goes away."""

    def __init__(self, loop: asyncio.AbstractEventLoop):
        self.ready: asyncio.Future = loop.create_future()
        self.closing = asyncio.Event()
        self.task: asyncio.Task | None = None


class MCPClientManager:
    """Owns the background event loop and one live session per configured server."""

    def __init__(self) -> None:
        self._configs: dict[str, MCPServerConfig] = {}
        self._connections: dict[str, _Connection] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.Lock()

    # --- public, synchronous API ---------------------------------------------------

    def register(self, config: MCPServerConfig) -> None:
        """Add or replace a server. A replaced server's session is closed."""
        old = self._configs.get(config.name)
        self._configs[config.name] = config
        if old is not None and old != config:
            self._submit(self._disconnect(config.name), timeout=30)

    def list_tools(self, server: str) -> list[dict[str, Any]]:
        return self._submit(self._list_tools(server), self._configs[server].timeout)

    def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> str:
        return self._submit(self._call_tool(server, tool, arguments), self._configs[server].timeout)

    def close(self) -> None:
        """Close every session (stopping stdio servers) and the background loop."""
        if self._loop is None:
            return
        try:
            self._submit(self._disconnect_all(), timeout=30)
        except MCPToolError:
            pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None

    # --- plumbing -----------------------------------------------------------------

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                threading.Thread(target=loop.run_forever, name="biomni-mcp", daemon=True).start()
                self._loop = loop
            return self._loop

    def _submit(self, coro, timeout: float | None):
        future = asyncio.run_coroutine_threadsafe(coro, self._ensure_loop())
        try:
            return future.result(timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise MCPToolError(f"MCP call timed out after {timeout:g}s") from None

    async def _hold(self, config: MCPServerConfig, conn: _Connection) -> None:
        """Open the transport and session, publish it, and keep it open until asked to close."""
        try:
            async with _open_transport(config) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                conn.ready.set_result(session)
                await conn.closing.wait()
        except BaseException as exc:  # reported to the caller waiting on `ready`
            if not conn.ready.done():
                conn.ready.set_exception(
                    MCPToolError(
                        f"could not connect to MCP server '{config.name}' ({config.describe()}): "
                        f"{_error_message(exc)}. Check that this command or URL works on its own."
                    )
                )
            if isinstance(exc, asyncio.CancelledError):
                raise

    async def _session(self, server: str) -> ClientSession:
        if server not in self._configs:
            raise MCPToolError(f"unknown MCP server '{server}'")
        conn = self._connections.get(server)
        if conn is None or conn.task is None or conn.task.done():
            conn = _Connection(asyncio.get_running_loop())
            conn.task = asyncio.create_task(self._hold(self._configs[server], conn))
            self._connections[server] = conn
        return await asyncio.shield(conn.ready)

    async def _disconnect(self, server: str, only: _Connection | None = None) -> None:
        """Close the server's session (or only `only`, if it is still the current one)."""
        if only is not None and self._connections.get(server) is not only:
            return
        conn = self._connections.pop(server, None)
        if conn is None or conn.task is None:
            return
        conn.closing.set()
        try:
            await asyncio.wait_for(conn.task, timeout=10)
        except (TimeoutError, MCPToolError):
            conn.task.cancel()

    async def _disconnect_all(self) -> None:
        await asyncio.gather(*(self._disconnect(name) for name in list(self._connections)))

    async def _list_tools(self, server: str) -> list[dict[str, Any]]:
        session = await self._session(server)
        tools, cursor = [], None
        while True:
            page = await session.list_tools(cursor) if cursor else await session.list_tools()
            tools += [
                {"name": t.name, "description": t.description, "inputSchema": _attr(t, "inputSchema", "input_schema")}
                for t in page.tools
            ]
            cursor = _attr(page, "nextCursor", "next_cursor")
            if not cursor:
                return tools

    async def _call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> str:
        session = await self._session(server)
        conn = self._connections.get(server)
        try:
            result = await session.call_tool(tool, arguments)
        except Exception as exc:
            # The server went away or the session broke: reconnect on the next call.
            await self._disconnect(server, only=conn)
            raise MCPToolError(f"MCP tool '{tool}' on server '{server}' failed: {_error_message(exc)}") from exc
        try:
            return result_to_text(result)
        except MCPToolError as exc:
            raise MCPToolError(f"MCP tool '{tool}' on server '{server}' returned an error: {exc}") from None


_manager: MCPClientManager | None = None
_manager_lock = threading.Lock()


def get_manager() -> MCPClientManager:
    """The process-wide manager; its sessions are closed when the interpreter exits."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = MCPClientManager()
            atexit.register(_manager.close)
        return _manager
