"""
Utility functions and helper tests for Biomni tool testing.
"""
import pytest
import json
import pickle
from pathlib import Path
from unittest.mock import Mock, patch


class TestSchemaValidation:
    """Test schema file validation and structure."""
    
    def test_all_schemas_exist(self):
        """Test that all expected schema files exist."""
        schema_dir = Path(__file__).parent.parent / "biomni" / "tool" / "schema_db"
        
        expected_schemas = [
            # Phase 1 schemas
            "pubchem.pkl", "chembl.pkl", "unichem.pkl", "drugcentral.pkl",
            "clinicaltrials.pkl", "dailymed.pkl",
            # Phase 2 schemas
            "ols.pkl", "quickgo.pkl", "encode.pkl", "cellxgene_census.pkl"
        ]
        
        for schema_file in expected_schemas:
            schema_path = schema_dir / schema_file
            assert schema_path.exists(), f"Schema file {schema_file} not found"
    
    def test_schema_structure(self):
        """Test that schemas have required structure."""
        schema_dir = Path(__file__).parent.parent / "biomni" / "tool" / "schema_db"
        
        # Test a few key schemas
        test_schemas = ["pubchem.pkl", "ols.pkl", "encode.pkl"]
        
        for schema_file in test_schemas:
            schema_path = schema_dir / schema_file
            
            with open(schema_path, 'rb') as f:
                schema = pickle.load(f)
            
            assert isinstance(schema, dict), f"Schema {schema_file} is not a dictionary"
            assert 'description' in schema, f"Schema {schema_file} missing description"
    
    def test_schema_generators_exist(self):
        """Test that schema generator files exist."""
        generators_dir = Path(__file__).parent.parent / "biomni" / "tool" / "schema_db" / "generators"
        
        expected_generators = [
            "create_chembl_schema.py", "create_unichem_schema.py", "create_drugcentral_schema.py",
            "create_clinicaltrials_schema.py", "create_dailymed_schema.py", "create_ols_schema.py",
            "create_quickgo_schema.py", "create_encode_schema.py", "create_cellxgene_census_schema.py"
        ]
        
        for generator_file in expected_generators:
            generator_path = generators_dir / generator_file
            assert generator_path.exists(), f"Generator file {generator_file} not found"


class TestAPIHelperFunctions:
    """Test helper functions used by the tools."""
    
    def test_format_query_results(self):
        """Test the _format_query_results helper function."""
        try:
            from biomni.tool.database import _format_query_results
            
            # Test with dictionary input
            test_data = {"key": "value", "nested": {"inner": "data"}}
            result = _format_query_results(test_data)
            
            assert isinstance(result, dict)
            
        except ImportError:
            pytest.skip("Could not import _format_query_results function")
    
    def test_query_rest_api_helper(self):
        """Test the _query_rest_api helper function."""
        try:
            from biomni.tool.database import _query_rest_api
            
            with patch('biomni.tool.database.requests') as mock_requests:
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"test": "data"}
                mock_requests.get.return_value = mock_response
                
                result = _query_rest_api("https://example.com/api", "GET", "Test query")
                
                assert isinstance(result, dict)
                assert "success" in result
                
        except ImportError:
            pytest.skip("Could not import _query_rest_api function")
    
    def test_query_llm_for_api_helper(self):
        """Test the _query_llm_for_api helper function."""
        try:
            from biomni.tool.database import _query_llm_for_api
            
            with patch('biomni.tool.database.anthropic') as mock_anthropic:
                mock_client = Mock()
                mock_response = Mock()
                mock_response.content = [Mock()]
                mock_response.content[0].text = '{"test": "response"}'
                mock_client.messages.create.return_value = mock_response
                mock_anthropic.Anthropic.return_value = mock_client
                
                result = _query_llm_for_api(
                    prompt="test prompt",
                    schema={"test": "schema"},
                    system_template="test template {schema}",
                    api_key="test-key"
                )
                
                assert isinstance(result, dict)
                assert "success" in result
                
        except ImportError:
            pytest.skip("Could not import _query_llm_for_api function")


class TestErrorHandling:
    """Test error handling across all tools."""
    
    def test_missing_prompt_and_endpoint(self, tool_functions):
        """Test that tools handle missing prompt and endpoint gracefully."""
        # Test tools that require either prompt or endpoint
        tools_to_test = [
            'query_pubchem', 'query_chembl', 'query_unichem',
            'query_clinicaltrials', 'query_dailymed', 'query_ols',
            'query_quickgo', 'query_encode'
        ]
        
        for tool_name in tools_to_test:
            if tool_name in tool_functions:
                tool_func = tool_functions[tool_name]
                result = tool_func()  # Call without parameters
                
                assert isinstance(result, dict)
                assert "error" in result
                assert "prompt" in result["error"] or "endpoint" in result["error"]
    
    def test_invalid_api_key_handling(self, tool_functions):
        """Test handling of invalid API keys."""
        with patch('biomni.tool.database.anthropic') as mock_anthropic:
            # Mock API key error
            mock_anthropic.Anthropic.side_effect = Exception("Invalid API key")
            
            for tool_name in ['query_pubchem', 'query_ols']:
                if tool_name in tool_functions:
                    tool_func = tool_functions[tool_name]
                    result = tool_func(prompt="test query")
                    
                    assert isinstance(result, dict)
                    # Should handle the error gracefully
    
    def test_network_error_handling(self, tool_functions):
        """Test handling of network errors."""
        with patch('biomni.tool.database.requests') as mock_requests:
            # Mock network error
            mock_requests.get.side_effect = Exception("Network error")
            mock_requests.post.side_effect = Exception("Network error")
            
            for tool_name in ['query_pubchem', 'query_chembl']:
                if tool_name in tool_functions:
                    tool_func = tool_functions[tool_name]
                    result = tool_func(endpoint="/test")
                    
                    assert isinstance(result, dict)
                    # Should handle the error gracefully


class TestResponseValidation:
    """Test response format validation."""
    
    def test_response_structure(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test that all tools return properly structured responses."""
        test_tools = [
            'query_pubchem', 'query_chembl', 'query_ols', 'query_encode'
        ]
        
        for tool_name in test_tools:
            if tool_name in tool_functions:
                tool_func = tool_functions[tool_name]
                
                # Test with prompt
                result = tool_func(prompt="test query")
                assert isinstance(result, dict)
                
                # Should have either success or error
                assert "success" in result or "error" in result
    
    def test_verbose_vs_concise_responses(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test verbose vs concise response modes."""
        test_tools = ['query_pubchem', 'query_ols']
        
        for tool_name in test_tools:
            if tool_name in tool_functions:
                tool_func = tool_functions[tool_name]
                
                verbose_result = tool_func(prompt="test", verbose=True)
                concise_result = tool_func(prompt="test", verbose=False)
                
                assert isinstance(verbose_result, dict)
                assert isinstance(concise_result, dict)
                
                # Verbose should typically have more fields
                # (though this depends on the specific implementation)


class TestToolDocumentation:
    """Test tool documentation and descriptions."""
    
    def test_tool_descriptions_exist(self):
        """Test that tool descriptions exist."""
        try:
            from biomni.tool.tool_description.database import tools
            
            expected_tools = [
                'query_pubchem', 'query_chembl', 'query_unichem', 'query_drugcentral',
                'query_clinicaltrials', 'query_dailymed', 'query_ols', 'query_quickgo',
                'query_encode', 'query_cellxgene_census'
            ]
            
            tool_names = [tool['name'] for tool in tools]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Tool {expected_tool} not found in descriptions"
                
        except ImportError:
            pytest.skip("Could not import tool descriptions")
    
    def test_tool_parameter_documentation(self):
        """Test that tools have proper parameter documentation."""
        try:
            from biomni.tool.tool_description.database import tools
            
            for tool in tools:
                if tool['name'].startswith('query_'):
                    # Should have required and optional parameters
                    assert 'required_parameters' in tool
                    assert 'optional_parameters' in tool
                    
                    # Should have description
                    assert 'description' in tool
                    assert len(tool['description']) > 10  # Non-trivial description
                    
        except ImportError:
            pytest.skip("Could not import tool descriptions")


def create_mock_response(data, status_code=200):
    """Helper function to create mock HTTP responses."""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.json.return_value = data
    mock_response.text = json.dumps(data)
    return mock_response


def validate_api_response(response):
    """Helper function to validate API response structure."""
    assert isinstance(response, dict)
    
    if "success" in response:
        assert isinstance(response["success"], bool)
    
    if "error" in response:
        assert isinstance(response["error"], str)
        assert len(response["error"]) > 0
    
    return True
