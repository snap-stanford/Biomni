## Monarch Initiative Integration

Biomni now supports the Monarch Initiative as a first-class biomedical data source.

### Monarch API Usage

You can query Monarch using either a natural language prompt or a direct API endpoint:

```python
from biomni.tool.database import query_monarch

# Example 1: Natural language prompt (requires Anthropic API key)
result = query_monarch(prompt="Find phenotypes associated with BRCA1")
print(result)

# Example 2: Direct endpoint
result = query_monarch(endpoint="/bioentity/gene/NCBIGene:672/phenotypes")
print(result)
```

- The Monarch API provides endpoints for genes, diseases, and phenotypes using standard IDs (e.g., NCBIGene:672, MONDO:0005148, HP:0001250).
- For more details, see the [Monarch API documentation](https://api.monarchinitiative.org/api).

### Testing

A test file is provided: `tests/test_query_monarch.py`.

Run it with:

```bash
pytest tests/test_query_monarch.py
```

---
