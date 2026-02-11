#!/usr/bin/env python
"""Quick test of gene filter functionality."""

import sys

sys.path.insert(0, "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai")

from biomni.tool.amplicon_table import query_amplicons

# Use your CCLE path
csv_path = "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"

try:
    print("Testing gene filter with single gene...")
    result = query_amplicons(gene="MYC", csv_path=csv_path, limit=5)
    print(f"Result: {result['summary']}")
    print(f"Rows returned: {result['row_count_returned']}")
    if result["rows"]:
        print(f"First row: {result['rows'][0]}")
except Exception as e:
    print(f"ERROR (single gene): {type(e).__name__}: {e}")

try:
    print("\nTesting gene filter with multiple genes...")
    result = query_amplicons(gene=["MYC", "EGFR"], csv_path=csv_path, limit=5)
    print(f"Result: {result['summary']}")
    print(f"Rows returned: {result['row_count_returned']}")
except Exception as e:
    print(f"ERROR (multiple genes): {type(e).__name__}: {e}")

try:
    print("\nTesting gene filter with gene_field='oncogenes'...")
    result = query_amplicons(gene="MYC", gene_field="oncogenes", csv_path=csv_path, limit=5)
    print(f"Result: {result['summary']}")
    print(f"Rows returned: {result['row_count_returned']}")
except Exception as e:
    print(f"ERROR (oncogenes field): {type(e).__name__}: {e}")
