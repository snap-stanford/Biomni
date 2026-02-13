"""Launch Biomni with OKN-WOBD MCP server on Gradio."""

from biomni.agent import A1

agent = A1(path="./data", llm="gpt-5", expected_data_lake_files=[])
agent.add_mcp(config_path="./mcp_config.yaml")
print("\n🚀 Launching Gradio UI...")
agent.launch_gradio_demo(share=False)
