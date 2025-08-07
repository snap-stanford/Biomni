import pickle

# DailyMed API schema
dailymed_schema = {
    'base_url': 'https://dailymed.nlm.nih.gov/dailymed/services/v2',
    'description': 'DailyMed RESTful API for accessing current SPL (Structured Product Labeling) information',
    'docs_url': 'https://dailymed.nlm.nih.gov/dailymed/app-support-web-services.cfm',
    'license': 'Public domain (US government)',
    'api_version': 'v2',
    'method': 'GET',
    'rate_limit': 'No explicit rate limit mentioned',
    
    'resources': {
        'applicationnumbers': {
            'description': 'Returns a list of all NDA numbers',
            'url': '/applicationnumbers',
            'parameters': {}
        },
        'drugclasses': {
            'description': 'Returns a list of all drug classes associated with at least one SPL',
            'url': '/drugclasses',
            'parameters': {}
        },
        'drugnames': {
            'description': 'Returns a list of all drug names',
            'url': '/drugnames',
            'parameters': {}
        },
        'ndcs': {
            'description': 'Returns a list of all NDC codes',
            'url': '/ndcs',
            'parameters': {}
        },
        'rxcuis': {
            'description': 'Returns a list of all product-level RxCUIs',
            'url': '/rxcuis',
            'parameters': {}
        },
        'spls': {
            'description': 'Returns a list of all SPLs (Structured Product Labels)',
            'url': '/spls',
            'parameters': {}
        },
        'spls_by_setid': {
            'description': 'Returns an SPL document for specific SET ID',
            'url': '/spls/{SETID}',
            'parameters': {
                'SETID': 'The SET ID of the SPL document'
            }
        },
        'spls_history': {
            'description': 'Returns version history for specific SET ID',
            'url': '/spls/{SETID}/history',
            'parameters': {
                'SETID': 'The SET ID of the SPL document'
            }
        },
        'spls_media': {
            'description': 'Returns links to all media for specific SET ID',
            'url': '/spls/{SETID}/media',
            'parameters': {
                'SETID': 'The SET ID of the SPL document'
            }
        },
        'spls_ndcs': {
            'description': 'Returns all NDCs for specific SET ID',
            'url': '/spls/{SETID}/ndcs',
            'parameters': {
                'SETID': 'The SET ID of the SPL document'
            }
        },
        'spls_packaging': {
            'description': 'Returns all product packaging descriptions for specific SET ID',
            'url': '/spls/{SETID}/packaging',
            'parameters': {
                'SETID': 'The SET ID of the SPL document'
            }
        },
        'uniis': {
            'description': 'Returns a list of all UNIIs (Unique Ingredient Identifiers)',
            'url': '/uniis',
            'parameters': {}
        }
    },
    
    'formats': {
        'xml': 'XML format (append .xml to URL)',
        'json': 'JSON format (append .json to URL)'
    },
    
    'download_options': {
        'zip': {
            'description': 'Download latest version of SPL as ZIP file',
            'url': 'https://dailymed.nlm.nih.gov/dailymed/downloadzipfile.cfm?setId={setid}',
            'headers': ['Content-Disposition', 'X-DAILYMED-LABEL-LAST-UPDATED']
        },
        'pdf': {
            'description': 'Download latest version of SPL as PDF file',
            'url': 'https://dailymed.nlm.nih.gov/dailymed/downloadpdffile.cfm?setId={setid}',
            'headers': ['Content-Disposition', 'X-DAILYMED-LABEL-LAST-UPDATED']
        },
        'zip_version': {
            'description': 'Download specific version of SPL as ZIP file',
            'url': 'https://dailymed.nlm.nih.gov/dailymed/getFile.cfm?type=zip&setid={setid}&version={versionnumber}'
        }
    },
    
    'error_codes': {
        '404': 'Not found',
        '415': 'Unsupported Media Type',
        '5xx': 'Server Error'
    },
    
    'status_codes': {
        '200 ACTIVE': 'The label is currently active',
        '200 ARCHIVED': 'The label has been archived'
    },
    
    'examples': {
        'get_all_spls': {
            'endpoint': '/spls.json',
            'description': 'Get all SPL documents in JSON format'
        },
        'get_drug_names': {
            'endpoint': '/drugnames.xml',
            'description': 'Get all drug names in XML format'
        },
        'get_specific_spl': {
            'endpoint': '/spls/12345678-1234-1234-1234-123456789012.json',
            'description': 'Get specific SPL by SET ID in JSON format'
        },
        'get_ndc_codes': {
            'endpoint': '/ndcs.json',
            'description': 'Get all NDC codes in JSON format'
        },
        'get_drug_classes': {
            'endpoint': '/drugclasses.json',
            'description': 'Get all drug classes in JSON format'
        },
        'get_spl_history': {
            'endpoint': '/spls/12345678-1234-1234-1234-123456789012/history.json',
            'description': 'Get version history for specific SPL'
        }
    },
    
    'common_identifiers': {
        'SET_ID': 'Unique identifier for SPL document (UUID format)',
        'NDC': 'National Drug Code',
        'NDA': 'New Drug Application number',
        'RxCUI': 'RxNorm Concept Unique Identifier',
        'UNII': 'Unique Ingredient Identifier'
    },
    
    'notes': {
        'https_required': 'API requires HTTPS (HTTP access disabled since 2016)',
        'version_required': 'Version number (v2) must be included in URL',
        'format_extension': 'Append .xml or .json to specify response format',
        'get_only': 'API only supports GET method for data retrieval'
    }
}

with open('dailymed.pkl', 'wb') as f:
    pickle.dump(dailymed_schema, f)

print('DailyMed schema created successfully')
