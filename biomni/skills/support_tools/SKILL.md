---
name: support_tools
description: General-purpose execution support including Python REPL and utility helpers.
---

## Tools
- **run_python_repl**: Executes the provided Python command in the notebook environment and returns the output.
- **read_function_source_code**: Read the source code of a function from any module path.
- **download_synapse_data**: Download data from Synapse using entity IDs. Requires SYNAPSE_AUTH_TOKEN environment variable for authentication. CRITICAL: Always specify entity_type parameter based on what you're downloading (file, dataset, folder, project). Check user hints like 'files' or search results to determine correct type. Multiple IDs only work with entity_type='file'. Recursive only works with entity_type='folder'.
