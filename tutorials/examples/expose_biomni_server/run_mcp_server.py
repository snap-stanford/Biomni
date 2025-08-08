from biomni.agent.a1 import A1

# Create the agent
agent = A1()

# Create the MCP server
mcp = agent.create_mcp_server(tool_modules=["biomni.tool.database"])

if __name__ == "__main__":
    # Run the server
    print("Starting Biomni MCP server...")
    mcp.run(transport="stdio")
