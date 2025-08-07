import pickle

# QuickGO API schema
quickgo_schema = {
    'base_url': 'https://www.ebi.ac.uk/QuickGO/services',
    'description': 'QuickGO REST API for Gene Ontology terms, annotations, and gene products',
    'docs_url': 'https://www.ebi.ac.uk/QuickGO/api/index.html',
    'license': 'Apache License 2.0',
    'format': 'JSON',
    'rate_limit': 'No explicit rate limit mentioned',
    
    'main_services': {
        'ontology': {
            'description': 'Access to GO and ECO ontology terms',
            'base_path': '/ontology',
            'endpoints': {
                'go_terms': '/ontology/go/terms/{go_id}',
                'go_search': '/ontology/go/search',
                'go_children': '/ontology/go/terms/{go_id}/children',
                'go_descendants': '/ontology/go/terms/{go_id}/descendants',
                'go_ancestors': '/ontology/go/terms/{go_id}/ancestors',
                'go_paths': '/ontology/go/terms/{go_id}/paths',
                'go_chart': '/ontology/go/terms/{go_id}/chart',
                'eco_terms': '/ontology/eco/terms/{eco_id}',
                'eco_search': '/ontology/eco/search'
            }
        },
        'annotation': {
            'description': 'Access to Gene Ontology annotations from GOA database',
            'base_path': '/annotation',
            'endpoints': {
                'search': '/annotation/search',
                'downloadSearch': '/annotation/downloadSearch',
                'stats': '/annotation/stats'
            }
        },
        'geneproduct': {
            'description': 'Access to gene products (proteins, RNA, complexes)',
            'base_path': '/geneproduct',
            'endpoints': {
                'search': '/geneproduct/search'
            }
        }
    },
    
    'ontology_parameters': {
        'go_search': {
            'query': 'Search term for GO terms',
            'limit': 'Number of results to return (default 25, max 100)',
            'page': 'Page number for pagination',
            'usage': 'Filter by usage (Unrestricted, Restricted)',
            'obsolete': 'Include obsolete terms (true/false)'
        },
        'go_relations': {
            'relations': 'Comma-separated list of relations (is_a, part_of, occurs_in, regulates, etc.)'
        }
    },
    
    'annotation_parameters': {
        'search': {
            'geneProductId': 'Gene product identifier (e.g., UniProtKB:P12345)',
            'goId': 'GO term identifier (e.g., GO:0008150)',
            'goUsage': 'GO term usage (Unrestricted, Restricted)',
            'goEvidence': 'Evidence code (IEA, IDA, IPI, etc.)',
            'qualifier': 'Qualifier (enables, involved_in, is_active_in, etc.)',
            'taxonId': 'NCBI taxonomy identifier',
            'taxonUsage': 'Taxon usage (exact, descendants)',
            'assignedBy': 'Annotation provider (e.g., UniProt, MGI)',
            'extension': 'Annotation extension',
            'aspect': 'GO aspect (biological_process, molecular_function, cellular_component)',
            'geneProductType': 'Type of gene product (protein, miRNA, complex)',
            'geneProductSubset': 'Gene product subset (e.g., Swiss-Prot, TrEMBL)',
            'proteome': 'Proteome identifier',
            'limit': 'Number of results to return (default 25, max 100)',
            'page': 'Page number for pagination'
        }
    },
    
    'geneproduct_parameters': {
        'search': {
            'query': 'Search term for gene products',
            'limit': 'Number of results to return (default 25, max 100)',
            'page': 'Page number for pagination',
            'type': 'Gene product type (protein, miRNA, complex)',
            'taxonId': 'NCBI taxonomy identifier',
            'taxonUsage': 'Taxon usage (exact, descendants)',
            'proteome': 'Proteome identifier'
        }
    },
    
    'go_aspects': [
        'biological_process',
        'molecular_function', 
        'cellular_component'
    ],
    
    'evidence_codes': [
        'IEA',  # Inferred from Electronic Annotation
        'IDA',  # Inferred from Direct Assay
        'IPI',  # Inferred from Physical Interaction
        'IMP',  # Inferred from Mutant Phenotype
        'IGI',  # Inferred from Genetic Interaction
        'IEP',  # Inferred from Expression Pattern
        'ISS',  # Inferred from Sequence or structural Similarity
        'ISO',  # Inferred from Sequence Orthology
        'ISA',  # Inferred from Sequence Alignment
        'ISM',  # Inferred from Sequence Model
        'IGC',  # Inferred from Genomic Context
        'IBA',  # Inferred from Biological aspect of Ancestor
        'IBD',  # Inferred from Biological aspect of Descendant
        'IKR',  # Inferred from Key Residues
        'IRD',  # Inferred from Rapid Divergence
        'RCA',  # Inferred from Reviewed Computational Analysis
        'TAS',  # Traceable Author Statement
        'NAS',  # Non-traceable Author Statement
        'IC',   # Inferred by Curator
        'ND'    # No biological Data available
    ],
    
    'qualifiers': [
        'enables',
        'involved_in',
        'is_active_in',
        'acts_upstream_of',
        'acts_upstream_of_positive_effect',
        'acts_upstream_of_negative_effect',
        'acts_upstream_of_or_within',
        'acts_upstream_of_or_within_positive_effect',
        'acts_upstream_of_or_within_negative_effect',
        'located_in',
        'part_of',
        'contributes_to',
        'colocalizes_with'
    ],
    
    'examples': {
        'search_go_terms': {
            'endpoint': '/ontology/go/search',
            'parameters': {'query': 'apoptosis', 'limit': 10}
        },
        'get_go_term': {
            'endpoint': '/ontology/go/terms/GO:0006915',
            'description': 'Get details for apoptotic process'
        },
        'search_annotations': {
            'endpoint': '/annotation/search',
            'parameters': {'goId': 'GO:0006915', 'limit': 20}
        },
        'search_protein_annotations': {
            'endpoint': '/annotation/search',
            'parameters': {'geneProductId': 'UniProtKB:P04637', 'limit': 50}
        },
        'search_human_annotations': {
            'endpoint': '/annotation/search',
            'parameters': {'taxonId': '9606', 'aspect': 'biological_process', 'limit': 100}
        },
        'search_gene_products': {
            'endpoint': '/geneproduct/search',
            'parameters': {'query': 'p53', 'taxonId': '9606', 'limit': 10}
        }
    },
    
    'response_format': {
        'structure': {
            'numberOfHits': 'Total number of results',
            'results': 'Array of result objects',
            'pageInfo': 'Pagination information'
        },
        'pagination': {
            'resultsPerPage': 'Number of results per page',
            'current': 'Current page number',
            'total': 'Total number of results'
        }
    },
    
    'common_organisms': {
        '9606': 'Homo sapiens (Human)',
        '10090': 'Mus musculus (Mouse)',
        '10116': 'Rattus norvegicus (Rat)',
        '7227': 'Drosophila melanogaster (Fruit fly)',
        '6239': 'Caenorhabditis elegans (Nematode)',
        '7955': 'Danio rerio (Zebrafish)',
        '3702': 'Arabidopsis thaliana (Thale cress)',
        '559292': 'Saccharomyces cerevisiae (Baker\'s yeast)',
        '284812': 'Schizosaccharomyces pombe (Fission yeast)',
        '511145': 'Escherichia coli str. K-12 substr. MG1655'
    },
    
    'limits': {
        'default_limit': 25,
        'max_limit': 100,
        'max_download_limit': 1000000
    }
}

with open('quickgo.pkl', 'wb') as f:
    pickle.dump(quickgo_schema, f)

print('QuickGO schema created successfully')
