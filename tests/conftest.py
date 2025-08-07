"""
Pytest configuration and fixtures for Biomni tool testing.
"""
import os
import sys
import pytest
from unittest.mock import Mock, patch
from pathlib import Path

# Add the biomni package to the Python path
biomni_root = Path(__file__).parent.parent
sys.path.insert(0, str(biomni_root))

# Import biomni modules after adding to path
try:
    from biomni.tool import database
except ImportError as e:
    pytest.skip(f"Could not import biomni.tool.database: {e}", allow_module_level=True)


@pytest.fixture
def mock_anthropic_api():
    """Mock LLM API responses for testing."""
    with patch('biomni.tool.database.get_llm') as mock_get_llm:
        # Mock successful LLM response
        mock_llm = Mock()
        mock_response = Mock()
        mock_response.content = '{"full_url": "https://example.com/api/test", "description": "Test query"}'
        mock_llm.invoke.return_value = mock_response
        mock_get_llm.return_value = mock_llm
        yield mock_llm


@pytest.fixture
def mock_requests():
    """Mock requests for API calls."""
    with patch('biomni.tool.database.requests') as mock_requests:
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"test": "data", "success": True}
        mock_response.text = '{"test": "data", "success": true}'
        mock_requests.get.return_value = mock_response
        mock_requests.post.return_value = mock_response
        yield mock_requests


@pytest.fixture
def sample_api_response():
    """Sample API response for testing."""
    return {
        "success": True,
        "data": {"test": "data"},
        "status_code": 200,
        "url": "https://example.com/api/test"
    }


@pytest.fixture
def test_prompts():
    """Common test prompts for different tool categories."""
    return {
        "drug_discovery": [
            "Find information about aspirin",
            "Search for compounds similar to caffeine",
            "Get drug interactions for ibuprofen"
        ],
        "clinical_trials": [
            "Find cancer clinical trials",
            "Search for COVID-19 studies",
            "Get recruiting trials for diabetes"
        ],
        "ontology": [
            "Find Gene Ontology terms for apoptosis",
            "Search for cell type ontology terms",
            "Get disease ontology for cancer"
        ],
        "genomics": [
            "Find ChIP-seq experiments for CTCF",
            "Search for RNA-seq data in human brain",
            "Get ENCODE experiments for H3K4me3"
        ],
        "single_cell": [
            "Get human T cells from lung tissue",
            "Find mouse brain single-cell data",
            "Search for stem cell differentiation data"
        ]
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Set dummy API key for testing
    os.environ.setdefault('ANTHROPIC_API_KEY', 'test-key-12345')
    yield
    # Cleanup is automatic with setdefault


@pytest.fixture
def tool_functions():
    """Dictionary of all tool functions for testing."""
    return {
        # Phase 1: Drug & Clinical Trials Tools
        'query_pubchem': database.query_pubchem,
        'query_chembl': database.query_chembl,
        'query_unichem': database.query_unichem,
        'query_drugcentral': database.query_drugcentral,
        'query_clinicaltrials': database.query_clinicaltrials,
        'query_dailymed': database.query_dailymed,
        
        # Phase 2: Knowledge & Ontology Tools
        'query_ols': database.query_ols,
        'query_quickgo': database.query_quickgo,
        'query_encode': database.query_encode,
        'query_cellxgene_census': database.query_cellxgene_census,
    }


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test (requires network)"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test (no network required)"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add integration marker to tests that likely need network
        if "integration" in item.name or "api" in item.name:
            item.add_marker(pytest.mark.integration)
        # Add unit marker to mock tests
        elif "mock" in item.name or "unit" in item.name:
            item.add_marker(pytest.mark.unit)
