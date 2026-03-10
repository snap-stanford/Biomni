import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from biomni.agent.a1 import A1

# agent = A1(path="./biomni_data", llm="claude-sonnet-4-5-20250929")

agent = A1(
    path='./data',
    expected_data_lake_files=[],
    source='Anthropic',
    llm='claude-sonnet-4-20250514'
)
# agent.go("Plan a CRISPR screen to identify genes that regulate T cell exhaustion, generate 32 genes that maximize the perturbation effect.")
# agent.go("Perform scRNA-seq annotation at [PATH] and generate meaningful hypothesis")
# agent.go("Predict ADMET properties for this compound: CC(C)CC1=CC=C(C=C1)C(C)C(=O)O")

# agent.go("""Plan a CRISPR screen to identify genes that regulate T cell exhaustion,
#         measured by the change in T cell receptor (TCR) signaling between acute
#         (interleukin-2 [IL-2] only) and chronic (anti-CD3 and IL-2) stimulation conditions.
#         Generate 32 genes that maximize the perturbation effect.""")

agent.go("Query PubMed for papers about CRISPR and return 5 results.")
