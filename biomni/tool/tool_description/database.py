description = [
    {
        "description": "Query the UniProt REST API using either natural language or a direct endpoint.",
        "name": "query_uniprot",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full or partial UniProt API "
                "endpoint URL to query directly "
                "(e.g., "
                '"https://rest.uniprot.org/uniprotkb/P01308")',
                "name": "endpoint",
                "type": "str",
            },
            {"default": 5, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'Natural language query about proteins (e.g., "Find information about human insulin")',
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the AlphaFold Database API for protein structure predictions.",
        "name": "query_alphafold",
        "optional_parameters": [
            {
                "default": "prediction",
                "description": 'Specific AlphaFold API endpoint to query: "prediction", "summary", or "annotations"',
                "name": "endpoint",
                "type": "str",
            },
            {
                "default": None,
                "description": 'Specific residue range in format "start-end" (e.g., "1-100")',
                "name": "residue_range",
                "type": "str",
            },
            {
                "default": False,
                "description": "Whether to download structure files",
                "name": "download",
                "type": "bool",
            },
            {"default": None, "description": "Directory to save downloaded files", "name": "output_dir", "type": "str"},
            {
                "default": "pdb",
                "description": 'Format of the structure file to download - "pdb" or "cif"',
                "name": "file_format",
                "type": "str",
            },
            {
                "default": "v4",
                "description": 'AlphaFold model version - "v4" (latest) or "v3", "v2", "v1"',
                "name": "model_version",
                "type": "str",
            },
            {
                "default": 1,
                "description": "Model number (1-5, with 1 being the highest confidence model)",
                "name": "model_number",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'UniProt accession ID (e.g., "P12345")',
                "name": "uniprot_id",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the InterPro REST API using natural language or a direct endpoint.",
        "name": "query_interpro",
        "optional_parameters": [
            {
                "default": None,
                "description": "Direct endpoint path or full URL (e.g., '/entry/interpro/IPR023411')",
                "name": "endpoint",
                "type": "str",
            },
            {
                "default": 3,
                "description": "Maximum number of results to return per page",
                "name": "max_results",
                "type": "int",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about protein domains or families",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the RCSB PDB database using natural language or a direct structured query.",
        "name": "query_pdb",
        "optional_parameters": [
            {
                "default": None,
                "description": "Direct structured query in RCSB Search API format (overrides prompt)",
                "name": "query",
                "type": "dict",
            },
            {"default": 3, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about protein structures",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Retrieve detailed data and/or download files for PDB identifiers.",
        "name": "query_pdb_identifiers",
        "optional_parameters": [
            {
                "default": "entry",
                "description": "Type of results: 'entry', 'assembly', 'polymer_entity', etc.",
                "name": "return_type",
                "type": "str",
            },
            {
                "default": False,
                "description": "Whether to download PDB structure files",
                "name": "download",
                "type": "bool",
            },
            {
                "default": None,
                "description": "List of specific attributes to retrieve",
                "name": "attributes",
                "type": "List[str]",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "List of PDB identifiers to query",
                "name": "identifiers",
                "type": "List[str]",
            }
        ],
    },
    {
        "description": "Take a natural language prompt and convert it to a structured KEGG API query.",
        "name": "query_kegg",
        "optional_parameters": [
            {"default": None, "description": "Direct KEGG API endpoint to query", "name": "endpoint", "type": "str"},
            {
                "default": True,
                "description": "Whether to return detailed API response information",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about KEGG "
                'data (e.g., "Find human pathways '
                'related to glycolysis")',
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the STRING protein interaction database using natural language or direct endpoint.",
        "name": "query_stringdb",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full URL to query directly (overrides prompt)",
                "name": "endpoint",
                "type": "str",
            },
            {
                "default": False,
                "description": "Whether to download image results (for image endpoints)",
                "name": "download_image",
                "type": "bool",
            },
            {"default": None, "description": "Directory to save downloaded files", "name": "output_dir", "type": "str"},
            {
                "default": True,
                "description": "Whether to return detailed response information",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about protein interactions",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the IUCN Red List API using natural language or a direct endpoint.",
        "name": "query_iucn",
        "optional_parameters": [
            {
                "default": None,
                "description": "Natural language query about species conservation status",
                "name": "prompt",
                "type": "str",
            },
            {
                "default": None,
                "description": 'API endpoint name (e.g., "species/id/12392") or full URL',
                "name": "endpoint",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to return detailed query information or just formatted results",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {"default": "", "description": "IUCN API token - required for all queries", "name": "token", "type": "str"}
        ],
    },
    {
        "description": "Query the Paleobiology Database (PBDB) API using natural language or a direct endpoint.",
        "name": "query_paleobiology",
        "optional_parameters": [
            {"default": None, "description": "API endpoint name or full URL", "name": "endpoint", "type": "str"},
            {
                "default": True,
                "description": "Whether to return detailed query information",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about fossil records",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the JASPAR REST API using natural language or a direct "
        "endpoint to retrieve transcription factor binding profiles.",
        "name": "query_jaspar",
        "optional_parameters": [
            {
                "default": None,
                "description": "API endpoint path (e.g., '/matrix/MA0002.2/') or full URL",
                "name": "endpoint",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to return detailed query information",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about transcription factor binding profiles",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the World Register of Marine Species (WoRMS) REST API "
        "using natural language or a direct endpoint.",
        "name": "query_worms",
        "optional_parameters": [
            {"default": None, "description": "Full URL or endpoint specification", "name": "endpoint", "type": "str"},
            {
                "default": True,
                "description": "Whether to return full API response details",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about marine species",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the cBioPortal REST API using natural language or a "
        "direct endpoint to access cancer genomics data.",
        "name": "query_cbioportal",
        "optional_parameters": [
            {
                "default": None,
                "description": "API endpoint path (e.g., '/studies/brca_tcga/patients') or full URL",
                "name": "endpoint",
                "type": "str",
            },
            {
                "default": True,
                "description": "Whether to return detailed API response information",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about cancer genomics data",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Take a natural language prompt and convert it to a structured ClinVar query.",
        "name": "query_clinvar",
        "optional_parameters": [
            {
                "default": None,
                "description": "Direct search term to use with the ClinVar API",
                "name": "search_term",
                "type": "str",
            },
            {"default": 3, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": 'Natural language query about genetic variants (e.g., "Find pathogenic BRCA1 variants")',
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the NCBI Gene Expression Omnibus (GEO) using natural language or a direct search term.",
        "name": "query_geo",
        "optional_parameters": [
            {"default": None, "description": "Direct search term in GEO syntax", "name": "search_term", "type": "str"},
            {"default": 3, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
            {"default": None, "description": "Whether to return verbose results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about RNA-seq, microarray, or other expression data",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the NCBI dbSNP database using natural language or a direct search term.",
        "name": "query_dbsnp",
        "optional_parameters": [
            {
                "default": None,
                "description": "Direct search term in dbSNP syntax",
                "name": "search_term",
                "type": "str",
            },
            {"default": 3, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about genetic variants/SNPs",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the UCSC Genome Browser API using natural language or a direct endpoint.",
        "name": "query_ucsc",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full URL or endpoint specification with parameters",
                "name": "endpoint",
                "type": "str",
            },
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about genomic data",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the Ensembl REST API using natural language or a direct endpoint.",
        "name": "query_ensembl",
        "optional_parameters": [
            {
                "default": None,
                "description": 'Direct API endpoint to query (e.g., "lookup/symbol/human/BRCA2") or full URL',
                "name": "endpoint",
                "type": "str",
            },
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about genomic data",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the OpenTargets Platform API using natural language or a direct GraphQL query.",
        "name": "query_opentarget",
        "optional_parameters": [
            {"default": None, "description": "Direct GraphQL query string", "name": "query", "type": "str"},
            {"default": None, "description": "Variables for the GraphQL query", "name": "variables", "type": "dict"},
            {"default": False, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about drug targets, diseases, and mechanisms",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the GWAS Catalog API using natural language or a direct endpoint.",
        "name": "query_gwas_catalog",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full API endpoint to query (e.g., "
                '"https://www.ebi.ac.uk/gwas/rest/api/studies?diseaseTraitId=EFO_0001360")',
                "name": "endpoint",
                "type": "str",
            },
            {"default": 3, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
        ],
        "required_parameters": [
            {"default": None, "description": "Natural language query about GWAS data", "name": "prompt", "type": "str"}
        ],
    },
    {
        "description": "Query the Monarch Initiative API using natural language or a direct endpoint.",
        "name": "query_monarch",
        "optional_parameters": [
            {"default": None, "description": "Direct API endpoint to query", "name": "endpoint", "type": "str"},
            {"default": 2, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
            {"default": False, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about genetic variants",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query gnomAD for variants in a gene using natural language or direct gene symbol.",
        "name": "query_gnomad",
        "optional_parameters": [
            {"default": None, "description": 'Gene symbol (e.g., "BRCA1")', "name": "gene_symbol", "type": "str"},
            {
                "default": True,
                "description": "Whether to return detailed query results",
                "name": "verbose",
                "type": "bool",
            },
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about genetic variants",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Identifies a DNA sequence using NCBI BLAST with improved "
        "error handling, timeout management, and debugging",
        "name": "blast_sequence",
        "optional_parameters": [],
        "required_parameters": [
            {
                "default": None,
                "description": "The sequence to identify. If DNA, "
                "use database: core_nt, program: "
                "blastn; if protein, use database: "
                "nr, program: blastp",
                "name": "sequence",
                "type": "str",
            },
            {"default": None, "description": "The BLAST database to search against", "name": "database", "type": "str"},
            {"default": None, "description": "The BLAST program to use", "name": "program", "type": "str"},
        ],
    },
    {
        "description": "Query ClinicalTrials.gov for studies using natural language, direct endpoint, or structured parameters.",
        "name": "query_clinicaltrials",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct API path or full URL to ClinicalTrials.gov endpoint",
                "default": None,
            },
            {"name": "term", "type": "str", "description": "Free-text search term", "default": None},
            {
                "name": "status",
                "type": "str",
                "description": "Overall recruitment status filter, e.g., RECRUITING",
                "default": None,
            },
            {"name": "condition", "type": "str", "description": "Condition/disease filter", "default": None},
            {"name": "intervention", "type": "str", "description": "Intervention filter", "default": None},
            {
                "name": "location",
                "type": "str",
                "description": "Location filter (country, city, etc.)",
                "default": None,
            },
            {
                "name": "phase",
                "type": "str",
                "description": "Trial phase filter (e.g., PHASE1, PHASE2, PHASE3)",
                "default": None,
            },
            {"name": "page_size", "type": "int", "description": "Items per page (1-100)", "default": 10},
            {"name": "max_pages", "type": "int", "description": "Max pages to fetch for pagination", "default": 1},
            {"name": "page_token", "type": "str", "description": "Start page token for pagination", "default": None},
            {
                "name": "verbose",
                "type": "bool",
                "description": "Whether to return detailed response structure",
                "default": True,
            },
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about clinical trials (if no endpoint/params provided)",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Reactome database using natural language or a direct endpoint.",
        "name": "query_reactome",
        "optional_parameters": [
            {"default": None, "description": "Direct API endpoint or full URL", "name": "endpoint", "type": "str"},
            {
                "default": False,
                "description": "Whether to download pathway diagrams",
                "name": "download",
                "type": "bool",
            },
            {"default": None, "description": "Directory to save downloaded files", "name": "output_dir", "type": "str"},
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about biological pathways",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the RegulomeDB database using natural language or "
        "direct variant/coordinate specification.",
        "name": "query_regulomedb",
        "optional_parameters": [
            {"default": None, "description": "Direct API endpoint to query", "name": "endpoint", "type": "str"},
            {"default": False, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about regulatory elements",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the PRIDE (PRoteomics IDEntifications) database using "
        "natural language or a direct endpoint.",
        "name": "query_pride",
        "optional_parameters": [
            {
                "default": None,
                "description": "The full endpoint to query (e.g., "
                '"https://www.ebi.ac.uk/pride/ws/archive/v2/projects?keyword=breast%20cancer")',
                "name": "endpoint",
                "type": "str",
            },
            {"default": 3, "description": "Maximum number of results to return", "name": "max_results", "type": "int"},
            {"default": None, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about proteomics data",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the Guide to PHARMACOLOGY database (GtoPdb) using natural language or a direct endpoint.",
        "name": "query_gtopdb",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full API endpoint to query (e.g., "
                '"https://www.guidetopharmacology.org/services/targets?type=GPCR&name=beta-2")',
                "name": "endpoint",
                "type": "str",
            },
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about drug targets, ligands, and interactions",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Given genomic coordinates, retrieves information of "
        "intersecting candidate cis-regulatory elements (cCREs).",
        "name": "region_to_ccre_screen",
        "optional_parameters": [
            {
                "default": "GRCh38",
                "description": "Assembly of the genome, formatted like 'GRCh38'",
                "name": "assembly",
                "type": "str",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Chromosome of the genomic region, formatted like 'chr12'",
                "name": "coord_chrom",
                "type": "str",
            },
            {"default": None, "description": "Starting chromosome coordinate", "name": "coord_start", "type": "int"},
            {"default": None, "description": "Ending chromosome coordinate", "name": "coord_end", "type": "int"},
        ],
    },
    {
        "description": "Given a cCRE (Candidate cis-Regulatory Element), return the "
        "k nearest genes sorted by distance.",
        "name": "get_genes_near_ccre",
        "optional_parameters": [
            {
                "default": 10,
                "description": "Number of nearby genes to return, sorted by distance",
                "name": "k",
                "type": "int",
            }
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "ENCODE Accession ID of query cCRE, e.g., EH38E1516980",
                "name": "accession",
                "type": "str",
            },
            {"default": None, "description": "Assembly of the gene, e.g., 'GRCh38'", "name": "assembly", "type": "str"},
            {
                "default": None,
                "description": "Chromosome of the gene, e.g., 'chr12'",
                "name": "chromosome",
                "type": "str",
            },
        ],
    },
    {
        "description": "Query the ReMap database for regulatory elements and transcription factor binding sites.",
        "name": "query_remap",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full API endpoint to query (e.g., "
                '"https://remap.univ-amu.fr/api/v1/catalogue/tf?tf=CTCF")',
                "name": "endpoint",
                "type": "str",
            },
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about transcription factors and binding sites",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the Mouse Phenome Database (MPD) for mouse strain "
        "phenotype data using natural language or direct endpoint "
        "access.",
        "name": "query_mpd",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full API endpoint to query (e.g., 'https://phenomedoc.jax.org/MPD_API/strains')",
                "name": "endpoint",
                "type": "str",
            },
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about mouse phenotypes, strains, or measurements",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
    {
        "description": "Query the Electron Microscopy Data Bank (EMDB) for 3D macromolecular structures.",
        "name": "query_emdb",
        "optional_parameters": [
            {
                "default": None,
                "description": 'Full API endpoint to query (e.g., "https://www.ebi.ac.uk/emdb/api/search")',
                "name": "endpoint",
                "type": "str",
            },
            {"default": True, "description": "Whether to return detailed results", "name": "verbose", "type": "bool"},
        ],
        "required_parameters": [
            {
                "default": None,
                "description": "Natural language query about EM structures and associated data",
                "name": "prompt",
                "type": "str",
            }
        ],
    },
]
