#!/usr/bin/env python3
"""
Test script for Biomni HTTP MCP servers.
Run this from the Biomni directory after starting the servers.
"""
import asyncio
import json
import sys
from mcp import ClientSession
from mcp.client.sse import sse_client


async def test_a1_server(host: str = "localhost", port: int = 8080):
    """Test the A1 HTTP MCP server."""
    print(f"🧪 Testing Biomni A1 HTTP MCP Server at {host}:{port}...")
    
    base_url = f"http://{host}:{port}"
    
    try:
        async with sse_client(base_url) as client:
            async with ClientSession(client[0], client[1]) as session:
                await session.initialize()
                
                # List available tools
                tools_response = await session.list_tools()
                print(f"✅ A1 server has {len(tools_response.tools)} tools available")
                
                if tools_response.tools:
                    # Test the first available tool
                    tool_name = tools_response.tools[0].name
                    test_result = await session.call_tool(
                        tool_name,
                        {"prompt": "Find information about human insulin protein"},
                    )
                    
                    result_data = json.loads(test_result.content[0].text)
                    print(f"✅ A1 test query successful: {result_data.get('success', False)}")
                    return True
                else:
                    print("❌ No tools found on A1 server")
                    return False
                
    except Exception as e:
        print(f"❌ A1 server test failed: {e}")
        return False


async def test_react_server(host: str = "localhost", port: int = 8081):
    """Test the ReAct HTTP MCP server."""
    print(f"🧪 Testing Biomni ReAct HTTP MCP Server at {host}:{port}...")
    
    base_url = f"http://{host}:{port}"
    
    try:
        async with sse_client(base_url) as client:
            async with ClientSession(client[0], client[1]) as session:
                await session.initialize()
                
                # List available tools
                tools_response = await session.list_tools()
                print(f"✅ ReAct server has {len(tools_response.tools)} tools available")
                
                # Test the analyze_biomedical_query tool
                test_result = await session.call_tool(
                    "analyze_biomedical_query",
                    {"query": "What are the main functions of the TP53 gene?"},
                )
                
                result_data = json.loads(test_result.content[0].text)
                print(f"✅ ReAct test query successful: {result_data.get('success', False)}")
                return True
                
    except Exception as e:
        print(f"❌ ReAct server test failed: {e}")
        return False


async def main():
    """Run HTTP MCP server tests."""
    print("🚀 Testing Biomni HTTP MCP Servers\n")
    
    # Test each server
    a1_success = await test_a1_server()
    print()
    
    react_success = await test_react_server()
    print()
    
    # Summary
    total_tests = 2
    passed_tests = sum([a1_success, react_success])
    
    print(f"📋 Test Summary: {passed_tests}/{total_tests} servers passed")
    
    if passed_tests == total_tests:
        print("🎉 All Biomni HTTP MCP servers are working correctly!")
        return True
    else:
        print("⚠️  Some servers failed - check that servers are running and accessible")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
