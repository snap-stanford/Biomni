# BiomniAD Data Sourcing (Alzheimer's & Related Dementias)

Short, actionable guidance for when a task involves Alzheimer's disease (AD) or related dementias and may benefit from additional data.

---

## When to use
Use this guide if the user's query mentions any of: "Alzheimer", "AD", "dementia", "MCI", "amyloid", "tau", "neurodegeneration", "cognition", or datasets/cohorts common in AD research.

## What to do (agent instructions)
1) Inspect local BiomniAD catalogs first (no downloads yet):
	- Read and print all JSON files in this folder that match `resource/BiomniAD*.json`.
	- Each entry typically includes: `id`, `title`, `dataset_url`, `open_access_portal_url`, optional `manifest_url`, and `files[]` with per-file `uri` download links.
	- Summarize the relevant datasets that are available (dataset name, modality, and recommended use).
2) Decide whether specific datasets could materially improve the task. If helpful, download only the needed subsets to the data lake (e.g., `./data/biomniad/`).
3) Cache paths and reference them in subsequent tool calls.

## Minimal code to scan catalogs
```python
import os, glob, json

root = os.getcwd()
catalog_paths = glob.glob(os.path.join(root, "biomni/know_how/resource", "BiomniAD*.json"))

catalogs = []
for p in catalog_paths:
	try:
		with open(p, "r") as f:
			catalogs.append(json.load(f))
	except Exception as e:
		print(f"Failed to load {p}: {e}")

# Flatten dataset entries and print titles
datasets = []
for c in catalogs:
	if isinstance(c, dict) and "datasets" in c:
		datasets.extend(c["datasets"])

print(f"Loaded {len(datasets)} datasets from {len(catalog_paths)} catalogs")
for d in datasets:
	print(f"- {d.get('title', d.get('id'))}")
```

## Lookup download URLs by dataset title
```python
from typing import Dict, List, Optional

def load_catalogs(resource_dir: str) -> List[Dict]:
	paths = glob.glob(os.path.join(resource_dir, "BiomniAD*.json"))
	out = []
	for p in paths:
		try:
			with open(p, "r") as f:
				out.append(json.load(f))
		except Exception as e:
			print(f"Failed to load {p}: {e}")
	return out

def index_by_title_and_id(catalogs: List[Dict]) -> Dict[str, Dict]:
	idx = {}
	for c in catalogs:
		for d in c.get("datasets", []):
			title = d.get("title")
			did = d.get("id")
			if title:
				idx[title] = d
			if did:
				idx[did] = d
	return idx

def get_download_links(entry: Dict) -> Dict[str, List[str]]:
	# Returns dataset-level URLs and file URIs when available
	links = {
		"dataset_url": [u for u in [entry.get("dataset_url"), entry.get("open_access_portal_url"), entry.get("manifest_url")] if u],
		"file_uris": []
	}
	for f in entry.get("files", []):
		uri = f.get("uri")
		if uri:
			links["file_uris"].append(uri)
	return links

# Example usage
resource_dir = os.path.join(root, "biomni/know_how/resource")
catalogs = load_catalogs(resource_dir)
idx = index_by_title_and_id(catalogs)

title = "NG00102 Genomic Atlas of the Proteome from Brain, CSF and Plasma"
entry = idx.get(title)
if entry:
	links = get_download_links(entry)
	print("Dataset-level URLs:", links["dataset_url"])  # landing pages / manifests
	print("File URIs:", links["file_uris"])             # direct downloadable files (when provided)
else:
	print("Title not found in local catalogs")
```

## Available AD datasets (titles)

Below are the dataset titles currently present in the local BiomniAD catalogs. Refer to the code above to map any title to its download URLs.

### NIAGADS (NG*)
- NG00049 CSF Summary Statistics Cruchaga et al. (2013)
- NG00052 CLU, a Potential Endophenotype for AD: Summary Statistics Deming et al. (2016)
- NG00067 ADSP Umbrella
- NG00075 IGAP Rare Variant Summary Statistics Kunkle et al. (2019)
- NG00100 Novel Alzheimer Disease Risk Loci and Pathways in African American Individuals Kunkle et al. (2021)
- NG00102 Genomic Atlas of the Proteome from Brain, CSF and Plasma
- NG00103 RBFOX1 and Brain Amyloidosis in Early and Preclinical AD
- NG00105 MiGA Microglia Genomic Atlas
- NG00118 AMP-AD Structural Variant WGS and SV-xQTL
- NG00126 Removing Variant-Level Artifacts in ADSP Sequencing Data Belloy et al. (2022)
- NG00133 Safety and Pharmacokinetics of a Highly Bioavailable Resveratrol Preparation (JOTROL™)
- NG00148 Rare Variant Aggregation Analyses in the ADSP Discovery Case–Control Sample
- NG00156 Genetic Variants and Functional Pathways Associated with Resilience to Alzheimer’s Disease
- NG00157 Sex-Specific Genetic Predictors of Alzheimer’s Disease Biomarkers Deming et al. (2018)
- NG00158 Sex Differences in the Genetic Predictors of Alzheimer’s Pathology Dumitrescu et al. (2019)
- NG00159 Longitudinal Change in Memory Performance as an Endophenotype for AD Archer et al. (2023)
- NG00160 Sex-Specific Genetic Architecture of Late-Life Memory Performance
- NG00161 Sex Differences in the Genetic Architecture of Cognitive Resilience to Alzheimer’s Disease Eissman et al. (2022)
- NG00165 CHARGE Association Results from the ADSP R1 WGS Data Set Wang et al. (2024)
- NG00166 Association Results from the ADSP R3 17k WGS Data Set Lee et al. (2023)
- NG00169 Progressive Supranuclear Palsy (PSP) Summary Statistics Farrell et al. (2024)
- NG00172 GWAS Summary Statistics of SNVs/INDELs/SVs for Progressive Supranuclear Palsy (PSP) Wang et al. (2024)
- NG00175 Targeted Proteomics of Core ATN Biomarkers in ADSP
- NG00177 Multi-Omic Endophenotype GWAS in ADSP
- NG00180 Four Large-Scale Plasma pQTL and mQTL Atlases
- NG00182 Genetic Determinants of CSF and Plasma ATN Biomarkers in Multi-Ancestry Cohorts

### Sinai/Other
- RADR – Repository for Rare Alzheimer’s Disease and Related Dementia Variants
- SingleBrain – Single-nucleus eQTL Meta-analysis across Human Brain Cohorts
- GCST90027158 – New insights into the genetic etiology of Alzheimer’s disease and related dementias (Bellenguez et al., 2022)
- isoMiGA – Isoform and Gene-level Counts and TPM in Short-read Human Microglia
- isoMiGA – Expression and Splicing QTL Summary Statistics in Human Microglia

## Decision rule
- Download using the `files[].uri` or `manifest_url`/`dataset_url` if the dataset's modality and "recommended_for" align with the task (e.g., genetics, transcriptomics, imaging, clinical outcomes) and expected benefit > cost (size/time/license).
- Prefer smaller, directly relevant subsets first; escalate to larger downloads only if needed.

## One-liner the agent should follow
"Because this is an AD/dementia task, I first scan local BiomniAD catalogs (*BiomniAD*.json), identified datasets aligned to the task and proceed accordingly."
