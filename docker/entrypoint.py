"""Entrypoint for Biomni Docker container.

Supports two modes controlled by the BIOMNI_MODE environment variable:
  - "gradio" (default): Launch the Gradio web UI on port 7860
  - "mcp": Launch the MCP server (SSE transport) on port 8000
  - "both": Launch both Gradio UI and MCP server
"""

import os
import threading


def run_gradio():
    from biomni.agent.a1 import A1

    data_path = os.environ.get("BIOMNI_DATA_PATH", "/app/data")
    skip_data_lake = os.environ.get("BIOMNI_SKIP_DATA_LAKE", "false").lower() == "true"
    require_verification = os.environ.get("BIOMNI_REQUIRE_VERIFICATION", "false").lower() == "true"

    expected_files = [] if skip_data_lake else None
    agent = A1(path=data_path, expected_data_lake_files=expected_files)
    agent.launch_gradio_demo(
        server_name="0.0.0.0",
        share=False,
        require_verification=require_verification,
    )


def run_mcp():
    from biomni.agent.a1 import A1

    data_path = os.environ.get("BIOMNI_DATA_PATH", "/app/data")
    skip_data_lake = os.environ.get("BIOMNI_SKIP_DATA_LAKE", "false").lower() == "true"
    tool_modules_str = os.environ.get("BIOMNI_MCP_TOOL_MODULES", "")

    expected_files = [] if skip_data_lake else None
    agent = A1(path=data_path, expected_data_lake_files=expected_files)

    tool_modules = [m.strip() for m in tool_modules_str.split(",") if m.strip()] or None
    mcp = agent.create_mcp_server(tool_modules=tool_modules)
    mcp.run(transport="sse", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    mode = os.environ.get("BIOMNI_MODE", "gradio").lower()

    if mode == "gradio":
        run_gradio()
    elif mode == "mcp":
        run_mcp()
    elif mode == "both":
        mcp_thread = threading.Thread(target=run_mcp, daemon=True)
        mcp_thread.start()
        run_gradio()
    else:
        raise ValueError(f"Unknown BIOMNI_MODE: {mode}. Use 'gradio', 'mcp', or 'both'.")
