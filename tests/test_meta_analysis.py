#!/usr/bin/env python
"""Test Biomni agent with a complex meta-analysis task.

This script tests the agent's ability to handle a multi-step scientific task.
Run with: python tests/test_meta_analysis.py
"""

import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

from biomni.config import default_config

print("=" * 70)
print("BIOMNI META-ANALYSIS TEST")
print("=" * 70)
print(f"LLM (reasoning): {default_config.llm}")
print(f"LLM Lite (simple): {default_config.llm_lite}")
print()

TASK = """You are tasked with conducting a meta-analysis of publicly available gene expression datasets to identify genes and regulatory mechanisms that are consistently dysregulated in kidney tissue of patients with diabetic nephropathy (DN).

Biological question: What genes are reproducibly differentially expressed in kidney samples from DN patients compared to healthy controls, and what pathways and transcription factors form the core regulatory architecture of those expression changes?

Complete the following steps:

STEP 1 - Data acquisition: Search the NCBI Gene Expression Omnibus (GEO) for microarray or RNA-seq datasets that profile gene expression in human kidney tissue from DN patients and non-diabetic controls. Find at least 4 independent datasets. Apply inclusion criteria: human samples, availability of expression data with case/control groups, at least 5 samples per group. Exclude animal models.

STEP 2 - Data retrieval: For each suitable dataset, download or access the processed expression data. Extract the sample metadata to identify DN vs control groups.

STEP 3 - Differential expression analysis: For each dataset, perform differential expression analysis comparing DN vs control samples. Identify significantly differentially expressed genes (DEGs) using appropriate statistical thresholds (e.g., adjusted p-value < 0.05, |log2FC| > 1).

STEP 4 - Meta-analysis: Identify genes that are consistently differentially expressed across multiple datasets. Create a list of "consensus DEGs" that appear in at least 2-3 datasets.

STEP 5 - Pathway analysis: Perform pathway enrichment analysis on the consensus DEGs to identify biological pathways dysregulated in DN.

Execute each step in order, showing your work and intermediate results. Provide a final summary of the key findings."""

print(f"Task (first 500 chars):\n{TASK[:500]}...")
print("\n" + "=" * 70)

try:
    from biomni.agent import A1

    print("Creating agent...")
    agent = A1(
        path='./data',
        llm=default_config.llm,  # Use config from .env
        expected_data_lake_files=[],
        use_tool_retriever=True
    )

    print("Configuring agent...")
    agent.configure()

    print("Running task...")
    print("=" * 70)

    result = agent.go(TASK)

    messages, final_answer = result

    print("\n" + "=" * 70)
    print("CONVERSATION LOG:")
    print("=" * 70)
    for i, msg in enumerate(messages[-10:]):  # Last 10 messages
        print(f"\n--- Message {i+1} ---")
        print(msg[:1000] if len(msg) > 1000 else msg)

    print("\n" + "=" * 70)
    print("FINAL ANSWER:")
    print("=" * 70)
    print(final_answer)

except KeyboardInterrupt:
    print("\n\nInterrupted by user")
except Exception as e:
    print(f"\nEXCEPTION: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
