# Monarch Initiative API schema for Biomni
# This file is a Python pickle containing a dictionary describing Monarch API endpoints and ID examples.

monarch_schema = {
    "endpoints": [
        {
            "path": "/bioentity/gene/{id}/phenotypes",
            "description": "Get phenotypes associated with a gene (NCBIGene:ID)",
        },
        {"path": "/bioentity/disease/{id}/genes", "description": "Get genes associated with a disease (MONDO:ID)"},
        {
            "path": "/bioentity/phenotype/{id}/diseases",
            "description": "Get diseases associated with a phenotype (HP:ID)",
        },
    ],
    "id_examples": {"gene": "NCBIGene:672", "disease": "MONDO:0005148", "phenotype": "HP:0001250"},
    "notes": [
        "Use NCBIGene:ID for genes, MONDO:ID for diseases, HP:ID for phenotypes.",
        "Endpoints return JSON by default.",
        "See https://api.monarchinitiative.org/api for full docs.",
    ],
}

import pickle

with open("monarch.pkl", "wb") as f:
    pickle.dump(monarch_schema, f)
