import os
import pickle

# DailyMed RESTful API schema based on official documentation
dailymed_schema = {
    "base_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2",
    "description": "DailyMed RESTful API for accessing current SPL (Structured Product Labeling) information",
    "docs_url": "https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm",
    "license": "Public domain (US government)",
    "api_version": "v2",
    "method": "GET",
    "rate_limit": "No explicit rate limit mentioned",
    "https_required": True,
    "deprecation_notice": "HTTP access disabled since January 31, 2016. HTTPS required.",
    "api_characteristics": {
        "http_method": "GET only",
        "versioning": "Version number (v2) must be included in URL",
        "format_specification": "Append .xml or .json to specify response format",
        "https_required": True,
        "base_uri_pattern": "https://dailymed.nlm.nih.gov/dailymed/services/v2/{resource}.{format}",
        "purpose": "Retrieve data from DailyMed",
    },
    "resources": {
        "applicationnumbers": {
            "description": "Returns a list of all NDA numbers",
            "url": "/applicationnumbers",
            "parameters": {},
            "example": "/applicationnumbers.json",
        },
        "drugclasses": {
            "description": "Returns a list of all drug classes associated with at least one SPL in the Pharmacologic Class Indexing Files",
            "url": "/drugclasses",
            "parameters": {},
            "example": "/drugclasses.json",
        },
        "drugnames": {
            "description": "Returns a list of all drug names",
            "url": "/drugnames",
            "parameters": {},
            "example": "/drugnames.json",
        },
        "ndcs": {
            "description": "Returns a list of all NDC codes",
            "url": "/ndcs",
            "parameters": {},
            "example": "/ndcs.json",
        },
        "rxcuis": {
            "description": "Returns a list of all product-level RxCUIs",
            "url": "/rxcuis",
            "parameters": {},
            "example": "/rxcuis.json",
        },
        "spls": {
            "description": "Returns a list of all SPLs",
            "url": "/spls",
            "parameters": {},
            "example": "/spls.json",
        },
        "spls_setid": {
            "description": "Returns an SPL document for specific SET ID",
            "url": "/spls/{SETID}",
            "parameters": {"SETID": "The SET ID of the SPL document (UUID format)"},
            "example": "/spls/12345678-1234-1234-1234-123456789012.json",
        },
        "spls_setid_history": {
            "description": "Returns version history for specific SET ID",
            "url": "/spls/{SETID}/history",
            "parameters": {"SETID": "The SET ID of the SPL document (UUID format)"},
            "example": "/spls/12345678-1234-1234-1234-123456789012/history.json",
        },
        "spls_setid_media": {
            "description": "Returns links to all media for specific SET ID",
            "url": "/spls/{SETID}/media",
            "parameters": {"SETID": "The SET ID of the SPL document (UUID format)"},
            "example": "/spls/12345678-1234-1234-1234-123456789012/media.json",
        },
        "spls_setid_ndcs": {
            "description": "Returns all NDCs for specific SET ID",
            "url": "/spls/{SETID}/ndcs",
            "parameters": {"SETID": "The SET ID of the SPL document (UUID format)"},
            "example": "/spls/12345678-1234-1234-1234-123456789012/ndcs.json",
        },
        "spls_setid_packaging": {
            "description": "Returns all product packaging descriptions for specific SET ID",
            "url": "/spls/{SETID}/packaging",
            "parameters": {"SETID": "The SET ID of the SPL document (UUID format)"},
            "example": "/spls/12345678-1234-1234-1234-123456789012/packaging.json",
        },
        "uniis": {
            "description": "Returns a list of all UNIIs",
            "url": "/uniis",
            "parameters": {},
            "example": "/uniis.json",
        },
    },
    "formats": {
        "xml": {
            "description": "XML format (append .xml to URL)",
            "extension": ".xml",
            "content_type": "application/xml",
        },
        "json": {
            "description": "JSON format (append .json to URL)",
            "extension": ".json",
            "content_type": "application/json",
        },
    },
    "error_codes": {"404": "Not found", "415": "Unsupported Media Type", "5xx": "Server Error"},
    "examples": {
        "get_all_spls_json": {
            "endpoint": "/spls.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.json",
            "description": "Get all SPL documents in JSON format",
        },
        "get_all_spls_xml": {
            "endpoint": "/spls.xml",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls.xml",
            "description": "Get all SPL documents in XML format",
        },
        "get_drug_names_json": {
            "endpoint": "/drugnames.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.json",
            "description": "Get all drug names in JSON format",
        },
        "get_drug_names_xml": {
            "endpoint": "/drugnames.xml",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/drugnames.xml",
            "description": "Get all drug names in XML format",
        },
        "get_specific_spl": {
            "endpoint": "/spls/12345678-1234-1234-1234-123456789012.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/12345678-1234-1234-1234-123456789012.json",
            "description": "Get specific SPL by SET ID in JSON format",
        },
        "get_ndc_codes": {
            "endpoint": "/ndcs.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/ndcs.json",
            "description": "Get all NDC codes in JSON format",
        },
        "get_drug_classes": {
            "endpoint": "/drugclasses.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/drugclasses.json",
            "description": "Get all drug classes in JSON format",
        },
        "get_spl_history": {
            "endpoint": "/spls/12345678-1234-1234-1234-123456789012/history.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/12345678-1234-1234-1234-123456789012/history.json",
            "description": "Get version history for specific SPL",
        },
        "get_spl_media": {
            "endpoint": "/spls/12345678-1234-1234-1234-123456789012/media.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/12345678-1234-1234-1234-123456789012/media.json",
            "description": "Get media links for specific SPL",
        },
        "get_spl_packaging": {
            "endpoint": "/spls/12345678-1234-1234-1234-123456789012/packaging.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/spls/12345678-1234-1234-1234-123456789012/packaging.json",
            "description": "Get packaging information for specific SPL",
        },
        "get_application_numbers": {
            "endpoint": "/applicationnumbers.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/applicationnumbers.json",
            "description": "Get all NDA application numbers",
        },
        "get_rxcuis": {
            "endpoint": "/rxcuis.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/rxcuis.json",
            "description": "Get all product-level RxCUIs",
        },
        "get_uniis": {
            "endpoint": "/uniis.json",
            "full_url": "https://dailymed.nlm.nih.gov/dailymed/services/v2/uniis.json",
            "description": "Get all UNIIs",
        },
    },
    "common_identifiers": {
        "SET_ID": "Unique identifier for SPL document (UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx)",
        "NDC": "National Drug Code (11-digit format)",
        "NDA": "New Drug Application number",
        "RxCUI": "RxNorm Concept Unique Identifier",
        "UNII": "Unique Ingredient Identifier",
    },
    "usage_notes": {
        "https_required": "API requires HTTPS (HTTP access disabled since January 31, 2016)",
        "version_required": "Version number (v2) must be included in URL",
        "format_extension": "Append .xml or .json to specify response format",
        "get_only": "API only supports GET method for data retrieval",
        "no_authentication": "No API key or authentication required",
        "no_rate_limit": "No explicit rate limiting mentioned in documentation",
        "optional_parameters": "Each resource may have optional query parameters to filter or control output",
    },
    "data_types": {
        "SPL": "Structured Product Labeling - standardized format for drug labeling information",
        "NDA": "New Drug Application - FDA application number for new drugs",
        "NDC": "National Drug Code - unique identifier for drug products",
        "RxCUI": "RxNorm Concept Unique Identifier - standardized drug names",
        "UNII": "Unique Ingredient Identifier - unique identifier for drug ingredients",
    },
}

# Save the schema
schema_dir = os.path.dirname(os.path.abspath(__file__))
schema_path = os.path.join(schema_dir, "..", "dailymed.pkl")

with open(schema_path, "wb") as f:
    pickle.dump(dailymed_schema, f)

print(f"DailyMed RESTful API schema created successfully at {schema_path}")
print(f"Base URL: {dailymed_schema['base_url']}")
print(f"Available resources: {list(dailymed_schema['resources'].keys())}")
print(f"Formats: {list(dailymed_schema['formats'].keys())}")
print(f"API Method: {dailymed_schema['method']}")
