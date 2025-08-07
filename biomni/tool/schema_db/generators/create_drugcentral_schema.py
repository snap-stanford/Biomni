import pickle

# DrugCentral schema - Note: This is a PostgreSQL database, not a REST API
drugcentral_schema = {
    'type': 'database',
    'database_type': 'postgresql',
    'description': 'DrugCentral PostgreSQL database for approved drugs and active pharmaceutical ingredients',
    'docs_url': 'https://drugcentral.org',
    'license': 'CC BY-SA 4.0',
    'access_method': 'database_connection',
    
    # Default connection parameters (may be overridden)
    'default_connection': {
        'dbhost': 'unmtid-dbs.net',
        'dbport': 5433,
        'dbname': 'drugcentral',
        'dbuser': 'drugman',
        'dbpw': 'dosage'
    },
    
    'operations': {
        'list_products': {
            'description': 'List all drug products',
            'returns': 'TSV format'
        },
        'list_structures': {
            'description': 'List all drug structures',
            'returns': 'TSV format'
        },
        'list_active_ingredients': {
            'description': 'List all active ingredients',
            'returns': 'TSV format'
        },
        'list_indications': {
            'description': 'List all indications',
            'returns': 'TSV format'
        },
        'get_structure': {
            'description': 'Get structure by struct_id',
            'parameters': ['struct_id'],
            'returns': 'TSV format'
        },
        'get_structure_by_synonym': {
            'description': 'Get structure by synonym',
            'parameters': ['synonym'],
            'returns': 'TSV format'
        },
        'get_structure_by_xref': {
            'description': 'Get structure by xref ID',
            'parameters': ['xref_id', 'xref_type'],
            'returns': 'TSV format'
        },
        'get_drugpage': {
            'description': 'Get drug (structure), with products, xrefs, etc.',
            'parameters': ['struct_id'],
            'returns': 'JSON format'
        },
        'search_indications': {
            'description': 'Search indications by regular expression',
            'parameters': ['regex_pattern'],
            'returns': 'TSV format'
        },
        'search_products': {
            'description': 'Search products by regular expression',
            'parameters': ['regex_pattern'],
            'returns': 'TSV format'
        }
    },
    
    'xref_types': [
        'CHEBI_ID',
        'CHEMBL_ID',
        'DRUGBANK_ID',
        'KEGG_DRUG_ID',
        'MESH_ID',
        'PUBCHEM_CID',
        'UNII'
    ],
    
    'note': 'DrugCentral requires database connection credentials. For REST API access, consider using alternative endpoints or the Smart API interface.',
    
    'alternative_access': {
        'smart_api': 'https://drugcentral.org/OpenAPI',
        'download': 'https://drugcentral.org/download',
        'docker': 'https://hub.docker.com/repository/docker/unmtransinfo/drugcentral_db'
    },
    
    'examples': {
        'search_aspirin': {
            'operation': 'get_structure_by_synonym',
            'parameters': {'synonym': 'aspirin'}
        },
        'search_by_chembl': {
            'operation': 'get_structure_by_xref',
            'parameters': {'xref_id': 'CHEMBL25', 'xref_type': 'CHEMBL_ID'}
        },
        'search_alzheimer': {
            'operation': 'search_indications',
            'parameters': {'regex_pattern': '^Alzheimer'}
        }
    }
}

with open('drugcentral.pkl', 'wb') as f:
    pickle.dump(drugcentral_schema, f)

print('DrugCentral schema created successfully')
