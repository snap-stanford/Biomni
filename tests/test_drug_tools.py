"""
Comprehensive tests for Phase 1 drug and clinical trials tools.
"""
import pytest
from unittest.mock import patch, Mock
import json


class TestPubChemTool:
    """Test cases for query_pubchem function."""
    
    def test_pubchem_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test PubChem query with natural language prompt."""
        query_pubchem = tool_functions['query_pubchem']
        
        result = query_pubchem(prompt="Find information about aspirin")
        
        assert isinstance(result, dict)
        assert "success" in result or "error" in result
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_pubchem_direct_endpoint(self, tool_functions, mock_requests):
        """Test PubChem query with direct endpoint."""
        query_pubchem = tool_functions['query_pubchem']
        
        result = query_pubchem(endpoint="/compound/name/aspirin/JSON")
        
        assert isinstance(result, dict)
        mock_requests.get.assert_called_once()
    
    def test_pubchem_error_handling(self, tool_functions):
        """Test PubChem error handling."""
        query_pubchem = tool_functions['query_pubchem']
        
        result = query_pubchem()  # No prompt or endpoint
        
        assert "error" in result
        assert "Either a prompt or an endpoint must be provided" in result["error"]
    
    @pytest.mark.integration
    def test_pubchem_real_api(self, tool_functions):
        """Integration test with real PubChem API."""
        query_pubchem = tool_functions['query_pubchem']
        
        result = query_pubchem(endpoint="/compound/name/water/JSON")
        
        # This test requires network access
        assert isinstance(result, dict)


class TestChEMBLTool:
    """Test cases for query_chembl function."""
    
    def test_chembl_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test ChEMBL query with natural language prompt."""
        query_chembl = tool_functions['query_chembl']

        result = query_chembl(prompt="Find bioactivity data for aspirin")

        assert isinstance(result, dict)
        assert "success" in result
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_chembl_pagination(self, tool_functions, mock_requests):
        """Test ChEMBL pagination handling."""
        query_chembl = tool_functions['query_chembl']
        
        result = query_chembl(endpoint="/molecule", max_results=50)
        
        assert isinstance(result, dict)
        # Check that limit parameter is added
        mock_requests.get.assert_called_once()
        call_args = mock_requests.get.call_args
        assert "limit=50" in call_args[0][0]
    
    def test_chembl_verbose_mode(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test ChEMBL verbose vs non-verbose mode."""
        query_chembl = tool_functions['query_chembl']
        
        # Test verbose mode
        result_verbose = query_chembl(prompt="Find molecules", verbose=True)
        result_concise = query_chembl(prompt="Find molecules", verbose=False)
        
        assert isinstance(result_verbose, dict)
        assert isinstance(result_concise, dict)


class TestUniChemTool:
    """Test cases for query_unichem function."""
    
    def test_unichem_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test UniChem query with natural language prompt."""
        query_unichem = tool_functions['query_unichem']
        
        result = query_unichem(prompt="Map compound identifiers for aspirin")
        
        assert isinstance(result, dict)
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_unichem_post_request(self, tool_functions, mock_requests):
        """Test UniChem POST request handling."""
        query_unichem = tool_functions['query_unichem']

        result = query_unichem(endpoint="/compound/inchikey")

        assert isinstance(result, dict)
        # UniChem uses POST requests for some endpoints
        # The function should handle this internally


class TestDrugCentralTool:
    """Test cases for query_drugcentral function."""
    
    def test_drugcentral_database_info(self, tool_functions, mock_anthropic_api):
        """Test DrugCentral database information."""
        query_drugcentral = tool_functions['query_drugcentral']
        
        result = query_drugcentral(prompt="How to access aspirin data?")
        
        assert isinstance(result, dict)
        assert "database_type" in result or "access_method" in result
    
    def test_drugcentral_guidance(self, tool_functions, mock_anthropic_api):
        """Test DrugCentral guidance for database access."""
        query_drugcentral = tool_functions['query_drugcentral']
        
        result = query_drugcentral(prompt="Find drug information")
        
        assert isinstance(result, dict)
        # Should provide guidance since it's a database, not REST API
        assert "smart_api" in result or "connection_info" in result


class TestClinicalTrialsTool:
    """Test cases for query_clinicaltrials function."""
    
    def test_clinicaltrials_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test ClinicalTrials.gov query with natural language prompt."""
        query_clinicaltrials = tool_functions['query_clinicaltrials']
        
        result = query_clinicaltrials(prompt="Find cancer clinical trials")
        
        assert isinstance(result, dict)
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_clinicaltrials_pagination(self, tool_functions, mock_requests):
        """Test ClinicalTrials.gov pagination."""
        query_clinicaltrials = tool_functions['query_clinicaltrials']
        
        result = query_clinicaltrials(endpoint="/studies", max_results=100)
        
        assert isinstance(result, dict)
        call_args = mock_requests.get.call_args
        assert "pageSize=100" in call_args[0][0]
    
    @pytest.mark.integration
    def test_clinicaltrials_real_api(self, tool_functions):
        """Integration test with real ClinicalTrials.gov API."""
        query_clinicaltrials = tool_functions['query_clinicaltrials']
        
        result = query_clinicaltrials(endpoint="/studies?pageSize=1")
        
        assert isinstance(result, dict)


class TestDailyMedTool:
    """Test cases for query_dailymed function."""
    
    def test_dailymed_with_prompt(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test DailyMed query with natural language prompt."""
        query_dailymed = tool_functions['query_dailymed']
        
        result = query_dailymed(prompt="Find drug labeling information")
        
        assert isinstance(result, dict)
        mock_anthropic_api.invoke.assert_called_once()
    
    def test_dailymed_format_parameter(self, tool_functions, mock_requests):
        """Test DailyMed format parameter handling."""
        query_dailymed = tool_functions['query_dailymed']
        
        # Test JSON format
        result_json = query_dailymed(endpoint="/drugnames", format="json")
        result_xml = query_dailymed(endpoint="/drugnames", format="xml")
        
        assert isinstance(result_json, dict)
        assert isinstance(result_xml, dict)
        
        # Check that format extension is added
        json_call = mock_requests.get.call_args_list[0]
        xml_call = mock_requests.get.call_args_list[1]
        
        assert ".json" in json_call[0][0]
        assert ".xml" in xml_call[0][0]
    
    def test_dailymed_error_handling(self, tool_functions):
        """Test DailyMed error handling."""
        query_dailymed = tool_functions['query_dailymed']
        
        result = query_dailymed()  # No prompt or endpoint
        
        assert "error" in result


class TestToolIntegration:
    """Integration tests for multiple tools."""
    
    def test_all_tools_have_required_functions(self, tool_functions):
        """Test that all expected tools are available."""
        expected_tools = [
            'query_pubchem', 'query_chembl', 'query_unichem',
            'query_drugcentral', 'query_clinicaltrials', 'query_dailymed'
        ]
        
        for tool_name in expected_tools:
            assert tool_name in tool_functions
            assert callable(tool_functions[tool_name])
    
    def test_consistent_error_handling(self, tool_functions):
        """Test that all tools handle missing parameters consistently."""
        for tool_name, tool_func in tool_functions.items():
            if tool_name.startswith('query_'):
                result = tool_func()  # Call without required parameters
                assert isinstance(result, dict)
                assert "error" in result
    
    @pytest.mark.slow
    def test_tool_performance(self, tool_functions, mock_anthropic_api, mock_requests):
        """Test tool performance with mock responses."""
        import time
        
        for tool_name, tool_func in tool_functions.items():
            if tool_name in ['query_pubchem', 'query_chembl']:
                start_time = time.time()
                result = tool_func(prompt="test query")
                end_time = time.time()
                
                # Should complete within reasonable time
                assert (end_time - start_time) < 5.0
                assert isinstance(result, dict)
