import os
import pickle

# QuickGO API schema based on official documentation
quickgo_schema = {
    "base_url": "https://www.ebi.ac.uk/QuickGO/services",
    "description": "QuickGO REST API for Gene Ontology terms, annotations, and gene products",
    "docs_url": "https://www.ebi.ac.uk/QuickGO/api/index.html",
    "license": "Apache License 2.0",
    "format": "JSON",
    "rate_limit": "No explicit rate limit mentioned",
    "main_services": {
        "ontology": {
            "description": "Access to GO and ECO ontology terms",
            "base_path": "/ontology",
            "endpoints": {
                "go_terms": "/ontology/go/terms/{go_id}",
                "go_search": "/ontology/go/search",
                "go_children": "/ontology/go/terms/{go_id}/children",
                "go_descendants": "/ontology/go/terms/{go_id}/descendants",
                "go_ancestors": "/ontology/go/terms/{go_id}/ancestors",
                "go_paths": "/ontology/go/terms/{go_id}/paths",
                "go_chart": "/ontology/go/terms/{go_id}/chart",
                "eco_terms": "/ontology/eco/terms/{eco_id}",
                "eco_search": "/ontology/eco/search",
            },
        },
        "annotation": {
            "description": "Access to Gene Ontology annotations from GOA database",
            "base_path": "/annotation",
            "endpoints": {
                "search": "/annotation/search",
                "downloadSearch": "/annotation/downloadSearch",
                "stats": "/annotation/stats",
            },
        },
        "geneproduct": {
            "description": "Access to gene products (proteins, RNA, complexes)",
            "base_path": "/geneproduct",
            "endpoints": {"search": "/geneproduct/search"},
        },
    },
    "ontology_parameters": {
        "go_search": {
            "query": "Search term for GO terms (string)",
            "limit": "Number of results to return (integer, default 25, max 100)",
            "page": "Page number for pagination (integer)",
            "usage": "Filter by usage (string: Unrestricted, Restricted)",
            "obsolete": "Include obsolete terms (boolean: true/false)",
        },
        "go_relations": {
            "relations": "Comma-separated list of relations (string: is_a, part_of, occurs_in, regulates, etc.)"
        },
    },
    "annotation_parameters": {
        "search": {
            "geneProductId": "Gene product identifier (string, e.g., UniProtKB:P12345)",
            "goId": "GO term identifier (string, e.g., GO:0008150)",
            "goUsage": "GO term usage (string: Unrestricted, Restricted)",
            "goEvidence": "Evidence code (string: IEA, IDA, IPI, etc.)",
            "qualifier": "Qualifier (string: enables, involved_in, is_active_in, etc.)",
            "taxonId": "NCBI taxonomy identifier (string, e.g., 9606)",
            "taxonUsage": "Taxon usage (string: exact, descendants)",
            "assignedBy": "Annotation provider (string, e.g., UniProt, MGI)",
            "extension": "Annotation extension (string)",
            "aspect": "GO aspect (string: biological_process, molecular_function, cellular_component)",
            "geneProductType": "Type of gene product (string: protein, miRNA, complex)",
            "geneProductSubset": "Gene product subset (string, e.g., Swiss-Prot, TrEMBL)",
            "proteome": "Proteome identifier (string)",
            "limit": "Number of results to return (integer, default 25, max 100)",
            "page": "Page number for pagination (integer)",
            "includeFields": "Fields to include in response (string, comma-separated)",
            "excludeFields": "Fields to exclude from response (string, comma-separated)",
        }
    },
    "geneproduct_parameters": {
        "search": {
            "query": "Search term for gene products (string)",
            "limit": "Number of results to return (integer, default 25, max 100)",
            "page": "Page number for pagination (integer)",
            "type": "Gene product type (string: protein, miRNA, complex)",
            "taxonId": "NCBI taxonomy identifier (string, e.g., 9606)",
            "taxonUsage": "Taxon usage (string: exact, descendants)",
            "proteome": "Proteome identifier (string)",
        }
    },
    "go_aspects": ["biological_process", "molecular_function", "cellular_component"],
    "evidence_codes": [
        "IEA",  # Inferred from Electronic Annotation
        "IDA",  # Inferred from Direct Assay
        "IPI",  # Inferred from Physical Interaction
        "IMP",  # Inferred from Mutant Phenotype
        "IGI",  # Inferred from Genetic Interaction
        "IEP",  # Inferred from Expression Pattern
        "ISS",  # Inferred from Sequence or structural Similarity
        "ISO",  # Inferred from Sequence Orthology
        "ISA",  # Inferred from Sequence Alignment
        "ISM",  # Inferred from Sequence Model
        "IGC",  # Inferred from Genomic Context
        "IBA",  # Inferred from Biological aspect of Ancestor
        "IBD",  # Inferred from Biological aspect of Descendant
        "IKR",  # Inferred from Key Residues
        "IRD",  # Inferred from Rapid Divergence
        "RCA",  # Inferred from Reviewed Computational Analysis
        "TAS",  # Traceable Author Statement
        "NAS",  # Non-traceable Author Statement
        "IC",  # Inferred by Curator
        "ND",  # No biological Data available
    ],
    "qualifiers": [
        "enables",
        "involved_in",
        "is_active_in",
        "acts_upstream_of",
        "acts_upstream_of_positive_effect",
        "acts_upstream_of_negative_effect",
        "acts_upstream_of_or_within",
        "acts_upstream_of_or_within_positive_effect",
        "acts_upstream_of_or_within_negative_effect",
        "located_in",
        "part_of",
        "contributes_to",
        "colocalizes_with",
    ],
    "examples": {
        "search_go_terms": {
            "endpoint": "/ontology/go/search",
            "parameters": {"query": "apoptosis", "limit": 10},
            "description": "Search for GO terms related to apoptosis",
        },
        "get_go_term": {
            "endpoint": "/ontology/go/terms/GO:0006915",
            "description": "Get details for apoptotic process term",
        },
        "get_go_children": {
            "endpoint": "/ontology/go/terms/GO:0006915/children",
            "description": "Get child terms of apoptotic process",
        },
        "search_annotations": {
            "endpoint": "/annotation/search",
            "parameters": {"goId": "GO:0006915", "limit": 20},
            "description": "Find annotations for apoptotic process",
        },
        "search_protein_annotations": {
            "endpoint": "/annotation/search",
            "parameters": {"geneProductId": "UniProtKB:P04637", "limit": 50},
            "description": "Find annotations for p53 protein",
        },
        "search_human_annotations": {
            "endpoint": "/annotation/search",
            "parameters": {"taxonId": "9606", "aspect": "biological_process", "limit": 100},
            "description": "Find human biological process annotations",
        },
        "search_gene_products": {
            "endpoint": "/geneproduct/search",
            "parameters": {"query": "p53", "taxonId": "9606", "limit": 10},
            "description": "Search for human p53 gene products",
        },
        "search_with_evidence": {
            "endpoint": "/annotation/search",
            "parameters": {"goId": "GO:0006915", "goEvidence": "IDA", "limit": 20},
            "description": "Find experimentally verified apoptotic process annotations",
        },
    },
    "response_format": {
        "structure": {
            "numberOfHits": "Total number of results (integer)",
            "results": "Array of result objects",
            "pageInfo": "Pagination information object",
        },
        "pagination": {
            "resultsPerPage": "Number of results per page (integer)",
            "current": "Current page number (integer)",
            "total": "Total number of results (integer)",
        },
    },
    "common_organisms": {
        "9606": "Homo sapiens (Human)",
        "10090": "Mus musculus (Mouse)",
        "10116": "Rattus norvegicus (Rat)",
        "7227": "Drosophila melanogaster (Fruit fly)",
        "6239": "Caenorhabditis elegans (Nematode)",
        "7955": "Danio rerio (Zebrafish)",
        "3702": "Arabidopsis thaliana (Thale cress)",
        "559292": "Saccharomyces cerevisiae (Baker's yeast)",
        "284812": "Schizosaccharomyces pombe (Fission yeast)",
        "511145": "Escherichia coli str. K-12 substr. MG1655",
    },
    "limits": {"default_limit": 25, "max_limit": 100, "max_download_limit": 1000000},
    "field_selection": {
        "annotation_fields": [
            "id",
            "geneProductId",
            "goId",
            "goEvidence",
            "qualifier",
            "goAspect",
            "assignedBy",
            "taxonId",
            "date",
            "reference",
            "annotationExtension",
        ],
        "ontology_fields": [
            "id",
            "name",
            "definition",
            "synonyms",
            "isObsolete",
            "aspect",
            "ancestors",
            "children",
            "descendants",
        ],
        "geneproduct_fields": ["id", "name", "synonyms", "taxonId", "type", "proteome"],
    },
}

# Save the schema to the correct location
schema_dir = os.path.dirname(os.path.abspath(__file__))
schema_path = os.path.join(schema_dir, "..", "quickgo.pkl")

with open(schema_path, "wb") as f:
    pickle.dump(quickgo_schema, f)

print(f"QuickGO schema created successfully at {schema_path}")
