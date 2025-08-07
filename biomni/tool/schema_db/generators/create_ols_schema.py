import pickle

# OLS (Ontology Lookup Service) API schema
ols_schema = {
    'base_url': 'https://www.ebi.ac.uk/ols4/api',
    'description': 'Ontology Lookup Service (OLS) 4 REST API for accessing biomedical ontologies',
    'docs_url': 'https://www.ebi.ac.uk/ols4/help',
    'license': 'Apache License 2.0',
    'api_version': '4.0',
    'format': 'JSON (HAL+JSON)',
    'rate_limit': 'No explicit rate limit mentioned',
    
    'main_resources': {
        'ontologies': {
            'description': 'Browse and search ontologies',
            'url': '/ontologies',
            'parameters': {
                'page': 'Page number for pagination',
                'size': 'Number of results per page',
                'lang': 'Language filter (e.g., en, de, fr)',
                'classification': 'Ontology classification filter'
            }
        },
        'terms': {
            'description': 'Search and browse ontology terms',
            'url': '/terms',
            'parameters': {
                'q': 'Search query string',
                'ontology': 'Restrict search to specific ontology',
                'type': 'Term type filter',
                'slim': 'Restrict to slim terms',
                'fieldList': 'Comma-separated list of fields to return',
                'obsoletes': 'Include obsolete terms (true/false)',
                'local': 'Restrict to local ontology terms',
                'childrenOf': 'Find children of specific term IRI',
                'allChildrenOf': 'Find all descendants of specific term IRI',
                'rows': 'Number of results to return',
                'start': 'Starting position for results'
            }
        },
        'properties': {
            'description': 'Browse ontology properties',
            'url': '/properties',
            'parameters': {
                'q': 'Search query string',
                'ontology': 'Restrict search to specific ontology',
                'iri': 'Property IRI'
            }
        },
        'individuals': {
            'description': 'Browse ontology individuals',
            'url': '/individuals',
            'parameters': {
                'q': 'Search query string',
                'ontology': 'Restrict search to specific ontology',
                'iri': 'Individual IRI'
            }
        }
    },
    
    'ontology_specific_endpoints': {
        'ontology_details': '/ontologies/{ontology_id}',
        'ontology_terms': '/ontologies/{ontology_id}/terms',
        'ontology_properties': '/ontologies/{ontology_id}/properties',
        'ontology_individuals': '/ontologies/{ontology_id}/individuals',
        'specific_term': '/ontologies/{ontology_id}/terms/{encoded_iri}',
        'term_parents': '/ontologies/{ontology_id}/terms/{encoded_iri}/parents',
        'term_children': '/ontologies/{ontology_id}/terms/{encoded_iri}/children',
        'term_ancestors': '/ontologies/{ontology_id}/terms/{encoded_iri}/ancestors',
        'term_descendants': '/ontologies/{ontology_id}/terms/{encoded_iri}/descendants',
        'term_graph': '/ontologies/{ontology_id}/terms/{encoded_iri}/graph'
    },
    
    'search_parameters': {
        'q': 'Free text search query',
        'exact': 'Exact match search (true/false)',
        'groupField': 'Field to group results by',
        'queryFields': 'Specific fields to search in',
        'fieldList': 'Fields to return in response',
        'childrenOf': 'Find children of specified term',
        'allChildrenOf': 'Find all descendants of specified term',
        'rows': 'Number of results (max 1000)',
        'start': 'Starting position for pagination'
    },
    
    'common_ontologies': {
        'go': 'Gene Ontology',
        'chebi': 'Chemical Entities of Biological Interest',
        'hp': 'Human Phenotype Ontology',
        'mondo': 'Monarch Disease Ontology',
        'uberon': 'Uber-anatomy ontology',
        'cl': 'Cell Ontology',
        'doid': 'Disease Ontology',
        'efo': 'Experimental Factor Ontology',
        'obi': 'Ontology for Biomedical Investigations',
        'so': 'Sequence Ontology',
        'pato': 'Phenotype and Trait Ontology',
        'ncbitaxon': 'NCBI Taxonomy',
        'pr': 'Protein Ontology',
        'bto': 'BRENDA Tissue Ontology',
        'fma': 'Foundational Model of Anatomy'
    },
    
    'response_format': {
        'hal_json': 'Hypertext Application Language JSON format',
        'embedded': 'Results embedded in _embedded field',
        'links': 'Navigation links in _links field',
        'pagination': 'Pagination info in page field'
    },
    
    'examples': {
        'search_all_terms': {
            'endpoint': '/terms',
            'parameters': {'q': 'cancer', 'rows': 10}
        },
        'search_go_terms': {
            'endpoint': '/terms',
            'parameters': {'q': 'apoptosis', 'ontology': 'go', 'rows': 20}
        },
        'get_ontology_list': {
            'endpoint': '/ontologies',
            'parameters': {'size': 50}
        },
        'get_term_details': {
            'endpoint': '/ontologies/go/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FGO_0006915',
            'description': 'Get details for GO:0006915 (apoptotic process)'
        },
        'get_term_children': {
            'endpoint': '/ontologies/go/terms/http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FGO_0006915/children',
            'description': 'Get children of apoptotic process'
        },
        'search_chebi_compounds': {
            'endpoint': '/terms',
            'parameters': {'q': 'aspirin', 'ontology': 'chebi', 'exact': 'true'}
        }
    },
    
    'iri_encoding': {
        'description': 'IRIs must be double URL encoded for API calls',
        'example': 'http://purl.obolibrary.org/obo/GO_0006915 becomes http%253A%252F%252Fpurl.obolibrary.org%252Fobo%252FGO_0006915'
    },
    
    'pagination': {
        'default_size': 20,
        'max_size': 1000,
        'page_parameter': 'page',
        'size_parameter': 'size'
    },
    
    'field_options': {
        'common_fields': [
            'iri', 'label', 'description', 'synonyms', 'annotation',
            'ontology_name', 'ontology_prefix', 'is_obsolete', 'term_replaced_by',
            'has_children', 'is_root', 'short_form', 'obo_id'
        ],
        'annotation_fields': [
            'id', 'database_cross_reference', 'has_obo_namespace',
            'has_alternative_id', 'created_by', 'creation_date'
        ]
    }
}

with open('ols.pkl', 'wb') as f:
    pickle.dump(ols_schema, f)

print('OLS schema created successfully')
