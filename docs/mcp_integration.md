# MCP (Model Context Protocol) Integration in Biomni

Biomni provides comprehensive support for the Model Context Protocol (MCP), allowing you to both integrate external MCP servers and expose Biomni tools as MCP servers. This enables seamless interoperability with a wide ecosystem of AI tools and services.

## Overview

MCP (Model Context Protocol) is a standard protocol for AI applications to communicate with external tools and services. Biomni supports two main MCP integration patterns:

1. **Adding External MCP Servers**: Import tools from external MCP servers into Biomni
2. **Exposing Biomni as MCP Server**: Make Biomni tools available to other MCP clients

## Adding External MCP Servers to Biomni

### Configuration File Format

MCP servers are configured using a YAML file that defines server connections and their tools. The `command` field specifies how to start the MCP server, which varies depending on how the server is packaged and distributed.

```yaml
mcp_servers:
  server_name:
    enabled: true  # Optional, defaults to true
    command: ["docker", "run", "-i", "--rm", "-e", "ENV_VAR", "image:tag"]  # Docker-based server
    # OR
    command: ["npx", "-y", "@modelcontextprotocol/server-name"]  # NPM-based server
    # OR
    command: ["python", "-m", "my_mcp_server"]  # Python-based server
    # OR
    command: ["./my-mcp-server", "--config", "config.json"]  # Binary server
    env:
      API_KEY: "${OPENAI_API_KEY}"  # Environment variable substitution
      CUSTOM_VAR: "static_value"
```

**Note**: The exact command format depends on the MCP server. Check the server's documentation for the correct command to use.

### Example Configuration

```yaml
mcp_servers:
  # GitHub MCP Server (Docker approach - recommended)
  github:
    command: ["docker", "run", "-i", "--rm", "-e", "GITHUB_PERSONAL_ACCESS_TOKEN", "ghcr.io/github/github-mcp-server"]
    enabled: true
    description: "Official GitHub MCP server for repository and issue management"
    env:
      GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"

  # Custom Local MCP Server - Only add tools necessary
  pubmed:
    command: ["python", "-m", "biomni.tool.mcp_tools.pubmed_mcp"]
    enabled: true
    tools:
      - biomni_name: search_pubmed
        description: "Search PubMed"
        parameters:
          query: {type: str, required: true,  description: "PubMed search term"}
          max_results:  {type: int, required: false, default: 10, description: "Maximum number of hits"}
      - biomni_name: get_article_abstract
        description: "Fetch PubMed abstract"
        parameters:
          pmid: {type: str, required: true, description: "PubMed ID"}

```

You can also create custom MCP servers and define your own tools by implementing the MCP protocol and adding them to your configuration file.

### Using the `add_mcp` Method

```python
from biomni.agent import A1

# Initialize Biomni agent
agent = A1()

# Optional: Set email for PubMed
# import os
# os.environ.setdefault("NCBI_EMAIL", "Your email address")

# Add MCP servers from configuration file
agent.add_mcp(config_path="./mcp_config.yaml")

# Now you can use the MCP tools in your queries
result = agent.go("Please list all repositories in the github account of the user.")
```

### How It Works

1. **Tool Discovery**: Biomni automatically discovers available tools from each MCP server
2. **Async-to-Sync Wrapping**: MCP tools are wrapped to work with Biomni's synchronous execution model
3. **Integration**: Tools are registered in Biomni's tool registry and made available for retrieval
4. **Module Organization**: Each MCP server gets its own module namespace (e.g., `mcp_servers.github`)

### Tool Registration Process

When you call `add_mcp()`, Biomni:

1. Loads the configuration file
2. For each enabled server:
   - Establishes a connection to the MCP server
   - Discovers available tools
   - Creates synchronous wrapper functions
   - Registers tools in the tool registry
   - Adds tools to the module2api mapping
   - Stores tools in custom functions registry

### Environment Variable Substitution

The configuration supports environment variable substitution using `${VARIABLE_NAME}` syntax:

```yaml
mcp_servers:
  my_server:
    command: ["my-server"]
    env:
      API_KEY: "${MY_API_KEY}"  # Will be replaced with os.getenv("MY_API_KEY")
      DEBUG: "true"              # Static value
```

You will need to properly set your environment variables (e.g., `GITHUB_TOKEN`) using a `.env` file or shell exports before running the agent.

## Exposing Biomni Tools as MCP Server

### Using the `create_mcp_server` Method

Biomni can expose its internal tools as an MCP server, making them available to other MCP clients:

```python
from biomni.agent.a1 import A1

# Create the agent
agent = A1()

# Create the MCP server with specific modules
mcp = agent.create_mcp_server(tool_modules=["biomni.tool.database"])

if __name__ == "__main__":
    # Run the server using stdio transport
    print("Starting Biomni MCP server...")
    mcp.run(transport="stdio")
```

### Server Configuration

The MCP server can be configured with various options:

```python
# Create server with specific modules
mcp = agent.create_mcp_server(tool_modules=[
    "biomni.tool.genetics",
    "biomni.tool.database",
    "biomni.tool.cell_biology"
])

# The server will expose all tools from these modules
# Tools are automatically wrapped with proper parameter validation
```

## Best Practices

### Configuration Management

1. **Environment Variables**: Use environment variables for sensitive data like API keys
2. **Docker Images**: Use official Docker images for MCP servers when available
3. **Server Validation**: Test server connections before adding them to production
4. **Tool Naming**: Ensure unique tool names across different servers to avoid conflicts

### Performance Considerations

1. **Connection Management**: MCP servers are created on-demand for each tool call
2. **Tool Discovery**: Tool discovery happens once during `add_mcp()` call
3. **Error Handling**: Failed tool calls are properly handled and reported
4. **Docker Overhead**: Containerized servers may have additional startup time

### Security

1. **Environment Variables**: Never hardcode sensitive information in configuration files
2. **Docker Security**: Use trusted Docker images and consider security implications
3. **Server Permissions**: Ensure MCP servers have appropriate permissions for their operations
4. **Network Access**: Be aware of network access requirements for external MCP servers

## Troubleshooting

### Common Issues

1. **Docker Not Running**: Ensure Docker is running for containerized MCP servers
2. **Environment Variables**: Verify that required environment variables are set (e.g., `GITHUB_TOKEN`, `NCBI_EMAIL`)
3. **Tool Discovery Failures**: Check server logs for connection or authentication issues
4. **Naming Conflicts**: Tools with duplicate names will be skipped with a warning

### Debugging

```python
# Test MCP connections without adding tools
test_results = agent.test_mcp_connection("./mcp_config.yaml")
print(test_results)

# List all MCP servers and their tools
servers = agent.list_mcp_servers()
print(servers)

# Remove specific MCP tools
agent.remove_mcp_tools("server_name")
```

### Testing MCP Server

Use the provided test script to verify your MCP server is working:

```bash
cd tutorials/examples/expose_biomni_server
python test_mcp_server.py
```

The test script allows you to:
- Test individual tools with custom parameters
- View available tools and their schemas
- Debug connection issues
- Validate tool responses

## Example Use Cases

### Adding External MCP Servers

```python
from biomni.agent import A1

# Initialize agent and add MCP servers
agent = A1()
agent.add_mcp(config_path="./mcp_config.yaml")

# Use GitHub tools alongside Biomni tools
result = agent.go("""
Please list all repositories in the github account of the user.
Then use Biomni's genetics tools to analyze any bioinformatics repositories found.
""")
```

### Exposing Biomni as MCP Server

```python
from biomni.agent.a1 import A1

# Create agent and MCP server
agent = A1()
mcp = agent.create_mcp_server(tool_modules=["biomni.tool.database"])

# Run the server
if __name__ == "__main__":
    print("Starting Biomni MCP server...")
    mcp.run(transport="stdio")
```

## Integration with Other Systems

### Available Examples

The Biomni repository includes complete examples in the `tutorials/examples/` directory:

#### Adding MCP Servers
- **Location**: `tutorials/examples/add_mcp_server/`
- **Files**:
  - `mcp_config.yaml` - Example configuration file
  - `mcp_example.ipynb` - Jupyter notebook demonstrating usage

#### Exposing Biomni as MCP Server
- **Location**: `tutorials/examples/expose_biomni_server/`
- **Files**:
  - `run_mcp_server.py` - Script to run Biomni MCP server
  - `test_mcp_server.py` - Comprehensive test script for MCP tools

To run the examples:

```bash
# Add MCP servers
cd tutorials/examples/add_mcp_server
jupyter notebook mcp_example.ipynb

# Expose Biomni as MCP server
cd tutorials/examples/expose_biomni_server
python run_mcp_server.py

# Test the server
python test_mcp_server.py
```

### Testing MCP Server

You can test the Biomni MCP server using the provided test script:

```python
#!/usr/bin/env python3
"""
Simple test script for testing a single Biomni MCP tool.
"""

import asyncio
import json
import sys

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Configuration - Change these to test different tools
TOOL_TO_TEST = "query_uniprot"  # Change this to the tool you want to test
TEST_ARGS = {"prompt": "Find information about human insulin protein"}

async def test_single_tool():
    """Test a single tool in the Biomni MCP server."""

    # Set up the server parameters
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "run_mcp_server.py")
    server_params = StdioServerParameters(command="python", args=[server_script])

    try:
        print("🔌 Connecting to MCP server...")

        # Connect to the server
        async with stdio_client(server_params) as (reader, writer):
            async with ClientSession(reader, writer) as session:
                # Initialize the session
                await session.initialize()
                print("✅ Connected to MCP server")

                # List available tools
                response = await session.list_tools()
                tools = response.tools
                print(f"✅ Found {len(tools)} tools")

                # Test the tool
                result = await session.call_tool(TOOL_TO_TEST, TEST_ARGS)
                print("✅ Tool call successful!")
                print(f"📄 Result: {result.content[0].text}")

    except Exception as e:
        print(f"❌ Failed to connect to MCP server: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_single_tool())
    sys.exit(0 if success else 1)
```

## Conclusion

Biomni's MCP integration provides a powerful way to extend its capabilities with external tools and services, while also making Biomni tools available to the broader MCP ecosystem. This enables seamless interoperability and allows you to build sophisticated AI workflows that combine the best of multiple tool ecosystems.
