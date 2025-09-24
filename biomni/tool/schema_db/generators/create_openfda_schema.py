#!/usr/bin/env python3
"""
Schema generator for OpenFDA API based on official documentation.
"""

import os
import pickle


def create_openfda_schema():
    """Create OpenFDA API schema based on official documentation."""

    schema = {
        "name": "OpenFDA API",
        "description": "OpenFDA provides access to FDA data on drugs, devices, and foods",
        "base_url": "https://api.fda.gov",
        "endpoints": {
            "drug_event": {
                "url": "/drug/event.json",
                "description": "Drug adverse events and medication errors",
                "search_fields": [
                    "patient.drug.medicinalproduct",
                    "patient.reaction.reactionmeddrapt",
                    "occurcountry",
                    "receivedate",
                    "serious",
                    "patient.patientsex",
                    "patient.patientagegroup",
                ],
                "sort_fields": ["receivedate", "serious", "occurcountry"],
                "count_fields": ["patient.reaction.reactionmeddrapt.exact", "occurcountry", "serious"],
            },
            "drug_label": {
                "url": "/drug/label.json",
                "description": "Drug labeling information",
                "search_fields": [
                    "openfda.brand_name",
                    "openfda.generic_name",
                    "openfda.substance_name",
                    "indications_and_usage",
                    "dosage_and_administration",
                    "warnings",
                    "adverse_reactions",
                ],
                "sort_fields": ["openfda.brand_name", "openfda.generic_name"],
                "count_fields": ["openfda.brand_name.exact", "openfda.generic_name.exact"],
            },
            "drug_enforcement": {
                "url": "/drug/enforcement.json",
                "description": "Drug recalls and enforcement actions",
                "search_fields": [
                    "recalling_firm",
                    "product_description",
                    "reason_for_recall",
                    "recall_initiation_date",
                    "classification",
                ],
                "sort_fields": ["recall_initiation_date", "classification"],
                "count_fields": ["classification", "recalling_firm.exact"],
            },
            "device_event": {
                "url": "/device/event.json",
                "description": "Medical device adverse events",
                "search_fields": [
                    "device.name",
                    "patient.patientsex",
                    "patient.patientagegroup",
                    "mdr_text.event_type",
                    "occurcountry",
                ],
                "sort_fields": ["date_received", "occurcountry"],
                "count_fields": ["device.name.exact", "occurcountry"],
            },
            "device_classification": {
                "url": "/device/classification.json",
                "description": "Medical device classifications",
                "search_fields": ["device_name", "device_class", "regulation_number"],
                "sort_fields": ["device_name", "device_class"],
                "count_fields": ["device_class", "regulation_number"],
            },
            "device_recall": {
                "url": "/device/recall.json",
                "description": "Medical device recalls",
                "search_fields": [
                    "recalling_firm",
                    "product_description",
                    "reason_for_recall",
                    "recall_initiation_date",
                ],
                "sort_fields": ["recall_initiation_date"],
                "count_fields": ["recalling_firm.exact"],
            },
            "food_event": {
                "url": "/food/event.json",
                "description": "Food adverse events",
                "search_fields": ["products_brand_name", "reactions", "occurcountry"],
                "sort_fields": ["date_created", "occurcountry"],
                "count_fields": ["products_brand_name.exact", "occurcountry"],
            },
            "food_enforcement": {
                "url": "/food/enforcement.json",
                "description": "Food recalls and enforcement actions",
                "search_fields": [
                    "recalling_firm",
                    "product_description",
                    "reason_for_recall",
                    "recall_initiation_date",
                ],
                "sort_fields": ["recall_initiation_date"],
                "count_fields": ["recalling_firm.exact"],
            },
        },
        "query_parameters": {
            "search": {
                "description": "Search within specific fields using syntax: field:term",
                "examples": [
                    "patient.drug.medicinalproduct:lipitor",
                    "patient.reaction.reactionmeddrapt:fatigue+AND+occurcountry:ca",
                    "openfda.brand_name:aspirin",
                ],
            },
            "sort": {
                "description": "Sort results by field in ascending or descending order",
                "examples": ["receivedate:desc", "openfda.brand_name:asc"],
            },
            "count": {
                "description": "Count unique values of a field",
                "examples": ["patient.reaction.reactionmeddrapt.exact", "occurcountry"],
            },
            "limit": {
                "description": "Maximum number of records to return (max 1000)",
                "examples": ["10", "100", "1000"],
            },
            "skip": {
                "description": "Number of records to skip for pagination (max 25000)",
                "examples": ["0", "1000", "5000"],
            },
        },
        "query_syntax": {
            "basic_search": "search=field:term",
            "and_search": "search=field1:term1+AND+field2:term2",
            "or_search": "search=field1:term1+field2:term2",
            "exact_match": 'search=field:"exact phrase"',
            "sorting": "sort=field:asc|desc",
            "counting": "count=field.exact",
            "pagination": "limit=100&skip=1000",
        },
        "examples": [
            "Find adverse events for Lipitor: /drug/event.json?search=patient.drug.medicinalproduct:lipitor&limit=10",
            "Count patient reactions: /drug/event.json?count=patient.reaction.reactionmeddrapt.exact",
            "Find drug labels for aspirin: /drug/label.json?search=openfda.brand_name:aspirin",
            "Get recent recalls: /drug/enforcement.json?sort=recall_initiation_date:desc&limit=20",
        ],
    }

    return schema


def main():
    """Generate and save the OpenFDA schema."""
    schema = create_openfda_schema()

    # Save to pickle file
    schema_dir = os.path.dirname(os.path.dirname(__file__))
    schema_path = os.path.join(schema_dir, "openfda.pkl")

    with open(schema_path, "wb") as f:
        pickle.dump(schema, f)

    print(f"OpenFDA schema saved to: {schema_path}")
    print(f"Schema contains {len(schema['endpoints'])} endpoints")


if __name__ == "__main__":
    main()
