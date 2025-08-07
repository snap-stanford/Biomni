"""
Comprehensive tests for Phase 2 knowledge and ontology tools.
"""
import pytest
from unittest.mock import patch, Mock
import json


class TestOLSTool:
    """Test cases for query_ols function."""
    
    def test_ols_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test OLS query with natural language prompt."""
        query_ols = tool_functions['query_ols']
        
        result = query_ols(prompt="Find Gene Ontology terms for apoptosis")
        
        assert isinstance(result, dict)
        assert "success" in result or "error" in result
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_ols_direct_endpoint(self, tool_functions, mock_requests):
        """Test OLS query with direct endpoint."""
        query_ols = tool_functions['query_ols']
        
        result = query_ols(endpoint="/terms?q=apoptosis&ontology=go")
        
        assert isinstance(result, dict)
        mock_requests.get.assert_called_once()
    
    def test_ols_pagination(self, tool_functions, mock_requests):
        """Test OLS pagination handling."""
        query_ols = tool_functions['query_ols']
        
        result = query_ols(endpoint="/terms?q=cancer", max_results=50)
        
        assert isinstance(result, dict)
        call_args = mock_requests.get.call_args
        assert "rows=50" in call_args[0][0]
    
    @pytest.mark.integration
    def test_ols_real_api(self, tool_functions):
        """Integration test with real OLS API."""
        query_ols = tool_functions['query_ols']
        
        result = query_ols(endpoint="/terms?q=cell&rows=1")
        
        assert isinstance(result, dict)


class TestQuickGOTool:
    """Test cases for query_quickgo function."""
    
    def test_quickgo_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test QuickGO query with natural language prompt."""
        query_quickgo = tool_functions['query_quickgo']
        
        result = query_quickgo(prompt="Find GO annotations for p53")
        
        assert isinstance(result, dict)
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_quickgo_limit_validation(self, tool_functions, mock_requests):
        """Test QuickGO limit validation (max 100)."""
        query_quickgo = tool_functions['query_quickgo']
        
        result = query_quickgo(endpoint="/annotation/search", max_results=150)
        
        assert isinstance(result, dict)
        # Should be capped at 100
        call_args = mock_requests.get.call_args
        assert "limit=100" in call_args[0][0]
    
    def test_quickgo_ontology_search(self, tool_functions, mock_requests):
        """Test QuickGO ontology search."""
        query_quickgo = tool_functions['query_quickgo']
        
        result = query_quickgo(endpoint="/ontology/go/search?query=apoptosis")
        
        assert isinstance(result, dict)
        mock_requests.get.assert_called_once()
    
    @pytest.mark.integration
    def test_quickgo_real_api(self, tool_functions):
        """Integration test with real QuickGO API."""
        query_quickgo = tool_functions['query_quickgo']
        
        result = query_quickgo(endpoint="/ontology/go/terms/GO:0008150")
        
        assert isinstance(result, dict)


class TestENCODETool:
    """Test cases for query_encode function."""
    
    def test_encode_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test ENCODE query with natural language prompt."""
        query_encode = tool_functions['query_encode']
        
        result = query_encode(prompt="Find ChIP-seq experiments for CTCF")
        
        assert isinstance(result, dict)
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_encode_format_json(self, tool_functions, mock_requests):
        """Test ENCODE format=json parameter addition."""
        query_encode = tool_functions['query_encode']
        
        result = query_encode(endpoint="/search/?type=Experiment")
        
        assert isinstance(result, dict)
        call_args = mock_requests.get.call_args
        assert "format=json" in call_args[0][0]
    
    def test_encode_limit_handling(self, tool_functions, mock_requests):
        """Test ENCODE limit parameter handling."""
        query_encode = tool_functions['query_encode']
        
        # Test with specific limit
        result1 = query_encode(endpoint="/search/?type=Experiment", max_results=50)
        
        # Test with "all" limit
        result2 = query_encode(endpoint="/search/?type=Experiment", max_results="all")
        
        assert isinstance(result1, dict)
        assert isinstance(result2, dict)
        
        call_args_1 = mock_requests.get.call_args_list[0]
        call_args_2 = mock_requests.get.call_args_list[1]
        
        assert "limit=50" in call_args_1[0][0]
        assert "limit=all" in call_args_2[0][0]
    
    @pytest.mark.integration
    def test_encode_real_api(self, tool_functions):
        """Integration test with real ENCODE API."""
        query_encode = tool_functions['query_encode']
        
        result = query_encode(endpoint="/search/?type=Experiment&limit=1&format=json")
        
        assert isinstance(result, dict)


class TestCELLxGENECensusTool:
    """Test cases for query_cellxgene_census function."""
    
    def test_cellxgene_census_with_prompt(self, tool_functions, mock_anthropic_api):
        """Test CELLxGENE Census query with natural language prompt."""
        query_cellxgene_census = tool_functions['query_cellxgene_census']
        
        result = query_cellxgene_census(prompt="Get human T cells from lung tissue")
        
        assert isinstance(result, dict)
        assert "python_code" in result or "error" in result
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_cellxgene_census_code_only(self, tool_functions, mock_anthropic_api):
        """Test CELLxGENE Census code-only mode."""
        query_cellxgene_census = tool_functions['query_cellxgene_census']
        
        # Mock successful response
        mock_anthropic_api.invoke.return_value.content = json.dumps({
            "python_code": "import cellxgene_census\nprint('test')",
            "explanation": "Test code",
            "key_functions": ["cellxgene_census.open_soma"]
        })
        
        result = query_cellxgene_census(prompt="Get mouse brain data", code_only=True)
        
        assert isinstance(result, str)  # Should return just the code string
    
    def test_cellxgene_census_verbose_mode(self, tool_functions, mock_anthropic_api):
        """Test CELLxGENE Census verbose vs non-verbose mode."""
        query_cellxgene_census = tool_functions['query_cellxgene_census']
        
        # Mock successful response
        mock_anthropic_api.invoke.return_value.content = json.dumps({
            "python_code": "import cellxgene_census",
            "explanation": "Test explanation",
            "key_functions": ["cellxgene_census.get_anndata"]
        })
        
        result_verbose = query_cellxgene_census(prompt="Get data", verbose=True)
        result_concise = query_cellxgene_census(prompt="Get data", verbose=False)
        
        assert isinstance(result_verbose, dict)
        assert isinstance(result_concise, dict)
        
        # Verbose should have more fields
        assert len(result_verbose) > len(result_concise)
        assert "installation" in result_verbose
        assert "api_type" in result_verbose
    
    def test_cellxgene_census_error_handling(self, tool_functions):
        """Test CELLxGENE Census error handling."""
        query_cellxgene_census = tool_functions['query_cellxgene_census']
        
        result = query_cellxgene_census()  # No prompt
        
        assert "error" in result
        assert "A prompt is required" in result["error"]


class TestKnowledgeToolsIntegration:
    """Integration tests for knowledge and ontology tools."""
    
    def test_all_knowledge_tools_available(self, tool_functions):
        """Test that all expected knowledge tools are available."""
        expected_tools = [
            'query_ols', 'query_quickgo', 'query_encode', 'query_cellxgene_census'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tool_functions
            assert callable(tool_functions[tool_name])
    
    def test_ontology_tools_consistency(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test consistency between ontology tools (OLS and QuickGO)."""
        query_ols = tool_functions['query_ols']
        query_quickgo = tool_functions['query_quickgo']
        
        # Both should handle similar queries
        ols_result = query_ols(prompt="Find terms for apoptosis")
        quickgo_result = query_quickgo(prompt="Find terms for apoptosis")
        
        assert isinstance(ols_result, dict)
        assert isinstance(quickgo_result, dict)
    
    def test_genomics_tools_consistency(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test consistency between genomics tools (ENCODE and CELLxGENE)."""
        query_encode = tool_functions['query_encode']
        query_cellxgene_census = tool_functions['query_cellxgene_census']
        
        encode_result = query_encode(prompt="Find RNA-seq experiments")
        census_result = query_cellxgene_census(prompt="Get single-cell RNA data")
        
        assert isinstance(encode_result, dict)
        assert isinstance(census_result, dict)
    
    @pytest.mark.slow
    def test_knowledge_tools_performance(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test performance of knowledge tools."""
        import time
        
        knowledge_tools = ['query_ols', 'query_quickgo', 'query_encode']
        
        for tool_name in knowledge_tools:
            tool_func = tool_functions[tool_name]
            
            start_time = time.time()
            result = tool_func(prompt="test query")
            end_time = time.time()
            
            # Should complete within reasonable time
            assert (end_time - start_time) < 5.0
            assert isinstance(result, dict)
    
    def test_schema_loading(self, tool_functions):
        """Test that all tools can load their schemas."""
        import os
        from pathlib import Path
        
        schema_dir = Path(__file__).parent.parent / "biomni" / "tool" / "schema_db"
        
        expected_schemas = [
            "ols.pkl", "quickgo.pkl", "encode.pkl", "cellxgene_census.pkl"
        ]
        
        for schema_file in expected_schemas:
            schema_path = schema_dir / schema_file
            assert schema_path.exists(), f"Schema file {schema_file} not found"
    
    @pytest.mark.integration
    def test_real_api_responses(self, tool_functions):
        """Integration test with real APIs (requires network)."""
        # Test a simple query to each real API
        test_cases = [
            ('query_ols', "/terms?q=cell&rows=1"),
            ('query_quickgo', "/ontology/go/terms/GO:0008150"),
            ('query_encode', "/search/?type=Experiment&limit=1&format=json"),
        ]
        
        for tool_name, endpoint in test_cases:
            if tool_name in tool_functions:
                tool_func = tool_functions[tool_name]
                result = tool_func(endpoint=endpoint)
                
                assert isinstance(result, dict)
                # Should either succeed or fail gracefully
                assert "success" in result or "error" in result
