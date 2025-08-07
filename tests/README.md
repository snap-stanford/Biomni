# Biomni Tools Testing Suite

This directory contains comprehensive tests for all Biomni tools implemented in Phase 1 and Phase 2.

## Test Structure

```
tests/
├── README.md                 # This file
├── requirements.txt          # Test dependencies
├── conftest.py              # Pytest configuration and fixtures
├── run_tests.py             # Test runner script
├── test_drug_tools.py       # Tests for Phase 1 drug & clinical trials tools
├── test_knowledge_tools.py  # Tests for Phase 2 knowledge & ontology tools
└── test_utils.py            # Utility tests and helper functions
```

## Tested Tools

### Phase 1: Drug & Clinical Trials Tools
- `query_pubchem` - PubChem PUG-REST API
- `query_chembl` - ChEMBL REST API
- `query_unichem` - UniChem 2.0 API
- `query_drugcentral` - DrugCentral database
- `query_clinicaltrials` - ClinicalTrials.gov API v2
- `query_dailymed` - DailyMed RESTful API

### Phase 2: Knowledge & Ontology Tools
- `query_ols` - Ontology Lookup Service API
- `query_quickgo` - QuickGO Gene Ontology API
- `query_encode` - ENCODE Portal API
- `query_cellxgene_census` - CELLxGENE Census Python API

## Test Categories

### Unit Tests
- Mock API responses
- Parameter validation
- Error handling
- Response format validation
- Schema loading

### Integration Tests
- Real API calls (requires network)
- End-to-end functionality
- Performance testing

### Utility Tests
- Schema validation
- Helper function testing
- Documentation validation

## Running Tests

### Prerequisites

1. Install test dependencies:
```bash
pip install -r tests/requirements.txt
```

2. Install Biomni package:
```bash
pip install -e .
```

3. Set up environment variables (optional):
```bash
export ANTHROPIC_API_KEY="your-api-key"  # For LLM integration tests
```

### Quick Start

Run all unit tests (no network required):
```bash
python tests/run_tests.py
```

Run with verbose output:
```bash
python tests/run_tests.py --verbose
```

### Test Categories

Run only unit tests:
```bash
python tests/run_tests.py --type unit
```

Run only integration tests (requires network):
```bash
python tests/run_tests.py --type integration --integration
```

Run drug tools tests only:
```bash
python tests/run_tests.py --type drug
```

Run knowledge tools tests only:
```bash
python tests/run_tests.py --type knowledge
```

### Using pytest directly

Run all tests:
```bash
pytest tests/ -v
```

Run specific test file:
```bash
pytest tests/test_drug_tools.py -v
```

Run tests with coverage:
```bash
pytest tests/ --cov=biomni.tool.database --cov-report=html
```

Skip integration tests:
```bash
pytest tests/ -m "not integration"
```

## Test Configuration

### Markers
- `@pytest.mark.unit` - Unit tests (no network)
- `@pytest.mark.integration` - Integration tests (requires network)
- `@pytest.mark.slow` - Slow-running tests

### Fixtures
- `mock_anthropic_api` - Mock Anthropic API responses
- `mock_requests` - Mock HTTP requests
- `tool_functions` - Dictionary of all tool functions
- `test_prompts` - Common test prompts for different categories

## Test Coverage

The test suite covers:

1. **Functionality Testing**
   - Natural language prompt processing
   - Direct endpoint access
   - Parameter validation
   - Response formatting

2. **Error Handling**
   - Missing parameters
   - Invalid API keys
   - Network errors
   - Malformed responses

3. **Integration Testing**
   - Real API connectivity
   - End-to-end workflows
   - Performance validation

4. **Schema Validation**
   - Schema file existence
   - Schema structure validation
   - Generator script validation

## Continuous Integration

For CI/CD pipelines, use:

```bash
# Install dependencies
pip install -r tests/requirements.txt
pip install -e .

# Run unit tests only (no network required)
python tests/run_tests.py --type unit

# Run with coverage
pytest tests/ -m "not integration" --cov=biomni.tool.database --cov-report=xml
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure Biomni is installed: `pip install -e .`
   - Check Python path includes project root

2. **Missing Dependencies**
   - Install test requirements: `pip install -r tests/requirements.txt`
   - Check dependency versions

3. **API Key Issues**
   - Set `ANTHROPIC_API_KEY` environment variable
   - Use mock tests if no API key available

4. **Network Issues**
   - Skip integration tests: `pytest -m "not integration"`
   - Check firewall/proxy settings

### Debug Mode

Run tests with maximum verbosity:
```bash
pytest tests/ -vvv --tb=long --capture=no
```

## Contributing

When adding new tools:

1. Add tests to appropriate test file
2. Update this README
3. Add any new dependencies to requirements.txt
4. Ensure tests pass in both unit and integration modes

## Performance Benchmarks

Expected test performance:
- Unit tests: < 30 seconds
- Integration tests: < 2 minutes (network dependent)
- Full test suite: < 3 minutes

## Test Data

Tests use:
- Mock responses for unit tests
- Real API endpoints for integration tests
- Minimal data requests to avoid rate limiting
- Cached responses where possible
