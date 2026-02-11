#!/usr/bin/env python3
"""Test the fixed gene filtering."""

import sys

sys.path.insert(0, "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai")

from biomni.tool.amplicon_table import query_amplicons

csv_path = "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"

print("=" * 60)
print("Testing gene filter with quoted gene names")
print("=" * 60)

try:
    print("\n1. Testing search for 'YES1' gene...")
    result = query_amplicons(gene="YES1", gene_field="oncogenes", csv_path=csv_path, limit=3)
    print(f"   ✓ Summary: {result['summary']}")
    print(f"   ✓ Total matches: {result['row_count_total']}")
    if result["rows"]:
        print(f"   ✓ Sample row: {result['rows'][0]['Sample name']}")

except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

try:
    print("\n2. Testing search for 'TYMS' gene...")
    result = query_amplicons(gene="TYMS", gene_field="oncogenes", csv_path=csv_path, limit=3)
    print(f"   ✓ Summary: {result['summary']}")
    print(f"   ✓ Total matches: {result['row_count_total']}")
    if result["rows"]:
        print(f"   ✓ Sample row: {result['rows'][0]['Sample name']}")

except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

try:
    print("\n3. Testing multiple genes ['YES1', 'TYMS']...")
    result = query_amplicons(gene=["YES1", "TYMS"], gene_field="oncogenes", csv_path=csv_path, limit=5)
    print(f"   ✓ Summary: {result['summary']}")
    print(f"   ✓ Total matches: {result['row_count_total']}")

except Exception as e:
    print(f"   ✗ Error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
