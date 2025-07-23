# OpenFDA Integration Documentation

## Overview
`query_openfda` provides a unified interface to the OpenFDA API, supporting both natural language prompts and direct endpoint queries. It is implemented in `database.py` and follows the same modular pattern as other data source integrations (e.g., Monarch, GWAS Catalog).

## Usage

### Natural Language Prompt
```
from biomni.tool.database import query_openfda

result = query_openfda(prompt="Find adverse events for Lipitor", max_results=5)
print(result)
```

### Direct Endpoint
```
result = query_openfda(endpoint="https://api.fda.gov/drug/event.json?search=patient.drug.medicinalproduct:lipitor&limit=5")
print(result)
```

## Supported Endpoints
- Drug adverse events: `/drug/event.json`
- Drug labels: `/drug/label.json`
- Drug recalls: `/drug/enforcement.json`
- Device events: `/device/event.json`
- Food recalls: `/food/enforcement.json`

See the schema file (`schema_db/openfda.pkl`) for details and field examples.

## Testing
Run the OpenFDA integration tests with:
```
pytest tests/test_query_openfda.py
```

## Notes
- The function supports both prompt-based and direct endpoint queries.
- Prompts are translated to endpoints using Claude and the OpenFDA schema.
- Use the `max_results` parameter to limit results (default: 100).
- For best results, ensure your Anthropic API key is set in the environment if using prompt mode.

## Example Prompts
- "Find adverse events for Lipitor"
- "Get the drug label for Lipitor"
- "List recalls for Atorvastatin"

## Troubleshooting
- If you see errors about missing schema, ensure `schema_db/openfda.pkl` exists (regenerate with `generate_openfda_schema.py` if needed).
- For API errors, check your query and endpoint formatting.
