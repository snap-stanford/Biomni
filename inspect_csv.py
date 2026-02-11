#!/usr/bin/env python3
"""Inspect the actual data in the Oncogenes column."""

import pandas as pd

csv_path = "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"

df = pd.read_csv(csv_path)

print("=" * 60)
print("Inspecting Oncogenes column")
print("=" * 60)
print(f"\nTotal rows: {len(df)}")
print(f"\nColumns: {list(df.columns)}")

if "Oncogenes" in df.columns:
    print("\n'Oncogenes' column found!")
    print(f"Type: {df['Oncogenes'].dtype}")
    print(f"Non-null count: {df['Oncogenes'].notna().sum()}")

    # Show first 10 unique values
    print("\nFirst 10 non-null Oncogenes values:")
    for i, val in enumerate(df["Oncogenes"].dropna().head(10)):
        print(f"  {i + 1}. {repr(val)}")

    # Check if YES1 or TYMS are present
    print("\n\nSearching for YES1 or TYMS in Oncogenes column:")
    for idx, val in enumerate(df["Oncogenes"].dropna()):
        val_str = str(val).upper()
        if "YES1" in val_str or "TYMS" in val_str:
            print(f"  Row {idx}: {repr(val)}")
            if "YES1" in val_str:
                print("    → Contains YES1!")
            if "TYMS" in val_str:
                print("    → Contains TYMS!")

else:
    print("✗ 'Oncogenes' column NOT found!")
    print("Available columns with 'gene' in name:")
    for col in df.columns:
        if "gene" in col.lower():
            print(f"  - {col}")
