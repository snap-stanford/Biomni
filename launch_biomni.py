"""Launch Biomni with MCP servers on Gradio."""

from biomni.agent import A1

agent = A1(path="./data")
agent.add_mcp(config_path="./mcp_config.yaml")
print("\n🚀 Launching Gradio UI...")
agent.launch_gradio_demo(server_name="0.0.0.0", share=True)
