# OpenFDA API schema for prompt-to-endpoint translation
# This is a Python dictionary describing the main endpoints, parameters, and ID conventions for OpenFDA.
# Save as a pickle file for use in query_openfda.

openfda_schema = {
    "base_url": "https://api.fda.gov",
    "endpoints": {
        "drug_event": {
            "path": "/drug/event.json",
            "description": "Adverse event reports for drugs",
            "search_param": "search",
            "limit_param": "limit",
            "example": "/drug/event.json?search=patient.drug.medicinalproduct:lipitor&limit=10"
        },
        "drug_label": {
            "path": "/drug/label.json",
            "description": "Drug labeling information",
            "search_param": "search",
            "limit_param": "limit",
            "example": "/drug/label.json?search=openfda.brand_name:lipitor&limit=5"
        },
        "drug_enforcement": {
            "path": "/drug/enforcement.json",
            "description": "Drug recalls and enforcement reports",
            "search_param": "search",
            "limit_param": "limit",
            "example": "/drug/enforcement.json?search=product_description:lipitor&limit=5"
        },
        "device_event": {
            "path": "/device/event.json",
            "description": "Adverse event reports for devices",
            "search_param": "search",
            "limit_param": "limit",
            "example": "/device/event.json?search=device.generic_name:pacemaker&limit=5"
        },
        "food_enforcement": {
            "path": "/food/enforcement.json",
            "description": "Food recalls and enforcement reports",
            "search_param": "search",
            "limit_param": "limit",
            "example": "/food/enforcement.json?search=product_description:peanut&limit=5"
        }
    },
    "id_examples": {
        "drug_name": "lipitor",
        "brand_name": "lipitor",
        "generic_name": "atorvastatin calcium",
        "recall_number": "D-1234-2020"
    },
    "notes": [
        "Use the 'search' parameter for queries, e.g., search=patient.drug.medicinalproduct:lipitor",
        "Use 'limit' to restrict the number of results",
        "All endpoints return JSON by default",
        "URL-encode all search terms",
        "For adverse events, use patient.drug.medicinalproduct or openfda.brand_name fields",
        "For recalls, use product_description or recall_number fields"
    ]
}

import pickle
with open("openfda.pkl", "wb") as f:
    pickle.dump(openfda_schema, f)
