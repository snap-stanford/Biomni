"""Tests for biomni.mcp_client and A1.add_mcp, against a real MCP server subprocess."""

import asyncio
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from biomni.mcp_client import MCPClientManager, MCPServerConfig, MCPToolError, result_to_text

SERVER = str(Path(__file__).with_name("mcp_fake_server.py"))


@pytest.fixture
def manager():
    m = MCPClientManager()
    m.register(MCPServerConfig(name="fake", command=[sys.executable, SERVER]))
    yield m
    m.close()


def test_discovers_tools(manager):
    names = {tool["name"] for tool in manager.list_tools("fake")}
    assert {"server_pid", "echo", "fail", "slow", "crash"} <= names


def test_calls_reuse_one_server_process(manager):
    pids = {manager.call_tool("fake", "server_pid", {}) for _ in range(3)}
    assert len(pids) == 1


def test_returns_the_tool_text(manager):
    assert manager.call_tool("fake", "echo", {"text": "ab", "times": 2}) == "abab"


def test_tool_errors_raise(manager):
    with pytest.raises(MCPToolError, match="'fail' on server 'fake' returned an error: .*boom"):
        manager.call_tool("fake", "fail", {"message": "boom"})
    assert manager.call_tool("fake", "echo", {"text": "still ok"}) == "still ok"


def test_reconnects_after_the_server_dies(manager):
    before = manager.call_tool("fake", "server_pid", {})
    with pytest.raises(MCPToolError, match="'crash' on server 'fake' failed"):
        manager.call_tool("fake", "crash", {})
    after = manager.call_tool("fake", "server_pid", {})
    assert after != before


def test_timeout_keeps_the_session_usable():
    m = MCPClientManager()
    m.register(MCPServerConfig(name="fake", command=[sys.executable, SERVER], timeout=1))
    try:
        m.call_tool("fake", "echo", {"text": "warm"})  # start the server outside the timed call
        with pytest.raises(MCPToolError, match="timed out after 1s"):
            m.call_tool("fake", "slow", {"seconds": 5})
        assert m.call_tool("fake", "echo", {"text": "ok"}) == "ok"
    finally:
        m.close()


def test_unreachable_server_names_the_command():
    m = MCPClientManager()
    m.register(MCPServerConfig(name="broken", command=[sys.executable, "-c", "import sys; sys.exit(1)"]))
    try:
        with pytest.raises(MCPToolError, match="could not connect to MCP server 'broken'.*sys.exit"):
            m.list_tools("broken")
    finally:
        m.close()


def test_works_inside_a_running_event_loop(manager):
    async def from_notebook_cell():
        return manager.call_tool("fake", "echo", {"text": "hi"})

    assert asyncio.run(from_notebook_cell()) == "hi"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_streamable_http_server():
    port = _free_port()
    proc = subprocess.Popen([sys.executable, SERVER, "--http", str(port)], stderr=subprocess.DEVNULL)
    m = MCPClientManager()
    try:
        deadline = time.monotonic() + 20
        while socket.socket().connect_ex(("127.0.0.1", port)) != 0:
            assert time.monotonic() < deadline, "HTTP test server did not start"
            time.sleep(0.2)
        m.register(MCPServerConfig(name="remote", url=f"http://127.0.0.1:{port}/mcp"))
        assert m.call_tool("remote", "echo", {"text": "over http"}) == "over http"
    finally:
        m.close()
        proc.terminate()
        proc.wait(timeout=10)


def test_config_needs_exactly_one_transport():
    with pytest.raises(ValueError):
        MCPServerConfig(name="none")
    with pytest.raises(ValueError):
        MCPServerConfig(name="both", command=["x"], url="http://localhost/mcp")


def test_result_to_text_covers_non_text_and_structured_content():
    image = SimpleNamespace(type="image", mimeType="image/png")
    assert result_to_text(SimpleNamespace(content=[image], isError=False)) == "[image content (image/png) omitted]"
    structured = SimpleNamespace(content=[], structuredContent={"n": 1}, isError=False)
    assert result_to_text(structured) == '{"n": 1}'


def test_add_mcp_registers_working_tools(tmp_path):
    pytest.importorskip("langgraph")
    from biomni.agent.a1 import A1

    config = tmp_path / "mcp.yaml"
    config.write_text(f"mcp_servers:\n  fake_a1:\n    command: [{sys.executable!r}, {SERVER!r}]\n")
    registered = []
    agent = A1.__new__(A1)  # skip __init__: it downloads the data lake
    agent.tool_registry = SimpleNamespace(register_tool=registered.append)
    agent.module2api, agent.configure = {}, lambda: None

    agent.add_mcp(str(config))

    assert {"echo", "server_pid"} <= {schema["name"] for schema in registered}
    from mcp_servers.fake_a1 import echo, server_pid

    assert echo(text="x", times=3) == "xxx"
    assert server_pid() == server_pid()
