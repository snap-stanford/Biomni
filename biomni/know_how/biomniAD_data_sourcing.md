# BiomniAD Data Sourcing (Alzheimer's & Related Dementias)

Short, actionable guidance for when a task involves Alzheimer's disease (AD) or related dementias and may benefit from additional data.

---

## When to use
Use this guide if the user's query mentions any of: "Alzheimer", "AD", "dementia", "MCI", "amyloid", "tau", "neurodegeneration", "cognition", or datasets/cohorts common in AD research.

## What to do (agent instructions)
1) Inspect local BiomniAD catalogs first (no downloads yet):
   - Read all JSON files in this folder that match `resource/BiomniAD*.json`.
   - Summarize what's available (dataset name, modality, and recommended use).
2) Decide whether specific datasets could materially improve the task (e.g., retrieval, benchmarking, fine-grained analysis).
3) If beneficial and user permits, download only the needed subsets to the data lake (e.g., `./data/biomniad/`).
4) Cache paths and reference them in subsequent tool calls.

## Minimal code to scan catalogs
```python
import os, glob, json

root = os.path.dirname(__file__)  # biomni/know_how
catalog_paths = glob.glob(os.path.join(root, "resource", "BiomniAD*.json"))

catalogs = []
for p in catalog_paths:
	try:
		with open(p, "r") as f:
			catalogs.append(json.load(f))
	except Exception as e:
		print(f"Warning: could not read {p}: {e}")

# Example: list candidate downloads with brief rationale
candidates = []
for cat in catalogs:
	datasets = cat if isinstance(cat, list) else cat.get("datasets", [])
	for d in datasets:
		candidates.append({
			"name": d.get("name"),
			"modality": d.get("modality"),
			"approx_size": d.get("size"),
			"url": d.get("url"),
			"recommended_for": d.get("recommended_for"),
		})

print("BiomniAD candidates (summarized):")
for c in candidates[:10]:
	print(c)
```

## Decision rule
- Download if the dataset's modality and "recommended_for" align with the task (e.g., genetics, transcriptomics, imaging, clinical outcomes) and expected benefit > cost (size/time/license).
- Prefer smaller, directly relevant subsets first; escalate to larger downloads only if needed.
- Always note licensing/access constraints in your response.

## One-liner the agent should follow
"Because this is an AD/dementia task, I first scanned local BiomniAD catalogs (resource/BiomniAD*.json), identified datasets aligned to the task, and will download only the minimal, relevant portions after user confirmation."
