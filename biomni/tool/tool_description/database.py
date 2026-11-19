description = [
    {
        "description": "Query the UniProt REST API using either natural language or a direct endpoint.",
        "name": "query_uniprot",
        "optional_parameters": [
            {
                "default": None,
                "description": "Full or partial UniProt API endpoint URL to query directly (e.g., 'https://rest.uniprot.org/uniprotkb/P01308')",
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
        "description": "Query the AlphaFold Database API for protein structure predictions or metadata; optionally download structures.",
        "name": "query_alphafold",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Endpoint: 'prediction', 'summary', or 'annotations'",
                "default": "prediction",
            },
            {"name": "residue_range", "type": "str", "description": "Residue range as 'start-end'", "default": None},
            {"name": "download", "type": "bool", "description": "Whether to download structure file", "default": False},
            {"name": "output_dir", "type": "str", "description": "Directory to save downloaded files", "default": None},
            {"name": "file_format", "type": "str", "description": "Download format 'pdb' or 'cif'", "default": "pdb"},
            {
                "name": "model_version",
                "type": "str",
                "description": "AlphaFold model version (e.g., v4)",
                "default": "v4",
            },
            {"name": "model_number", "type": "int", "description": "Model number (1-5)", "default": 1},
        ],
        "required_parameters": [
            {
                "name": "uniprot_id",
                "type": "str",
                "description": "UniProt accession ID (e.g., 'P12345')",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the InterPro REST API using natural language or a direct endpoint.",
        "name": "query_interpro",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Endpoint path or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results per page", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about protein domains/families",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the RCSB PDB database using natural language or a direct structured query.",
        "name": "query_pdb",
        "optional_parameters": [
            {"name": "query", "type": "dict", "description": "Direct RCSB Search API query JSON", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum results to return", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about protein structures",
                "default": None,
            }
        ],
    },
    {
        "description": "Retrieve detailed data and/or download files for PDB identifiers.",
        "name": "query_pdb_identifiers",
        "optional_parameters": [
            {
                "name": "return_type",
                "type": "str",
                "description": "'entry', 'assembly', 'polymer_entity', etc.",
                "default": "entry",
            },
            {"name": "download", "type": "bool", "description": "Download PDB structure files", "default": False},
            {
                "name": "attributes",
                "type": "List[str]",
                "description": "Specific attributes to retrieve",
                "default": None,
            },
        ],
        "required_parameters": [
            {"name": "identifiers", "type": "List[str]", "description": "List of PDB identifiers", "default": None}
        ],
    },
    {
        "description": "Take a natural language prompt and convert it to a structured KEGG API query.",
        "name": "query_kegg",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct KEGG endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {"name": "prompt", "type": "str", "description": "Natural language query about KEGG data", "default": None}
        ],
    },
    {
        "description": "Query the STRING protein interaction database using natural language or direct endpoint.",
        "name": "query_stringdb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full URL to query directly", "default": None},
            {
                "name": "download_image",
                "type": "bool",
                "description": "Download image results if endpoint is image",
                "default": False,
            },
            {"name": "output_dir", "type": "str", "description": "Directory to save downloaded files", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about protein interactions",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the IUCN Red List API using natural language or a direct endpoint.",
        "name": "query_iucn",
        "optional_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about species conservation status",
                "default": None,
            },
            {"name": "endpoint", "type": "str", "description": "Endpoint name or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [{"name": "token", "type": "str", "description": "IUCN API token", "default": ""}],
    },
    {
        "description": "Query the Paleobiology Database (PBDB) API using natural language or a direct endpoint.",
        "name": "query_paleobiology",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "API endpoint name or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about fossil records",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the JASPAR REST API for transcription factor binding profiles.",
        "name": "query_jaspar",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "API endpoint path or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about TF binding profiles",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the World Register of Marine Species (WoRMS) REST API using natural language or a direct endpoint.",
        "name": "query_worms",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full URL or endpoint specification", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about marine species",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the cBioPortal REST API using natural language or a direct endpoint.",
        "name": "query_cbioportal",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "API endpoint path or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about cancer genomics",
                "default": None,
            }
        ],
    },
    {
        "description": "Convert a natural language prompt into a structured ClinVar search query and run it.",
        "name": "query_clinvar",
        "optional_parameters": [
            {"name": "search_term", "type": "str", "description": "Direct ClinVar search term", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genetic variants",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the NCBI GEO database (GDS/GEOPROFILES) using natural language or direct search term.",
        "name": "query_geo",
        "optional_parameters": [
            {"name": "search_term", "type": "str", "description": "Direct GEO search term", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about expression data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the NCBI dbSNP database using natural language or direct search term.",
        "name": "query_dbsnp",
        "optional_parameters": [
            {"name": "search_term", "type": "str", "description": "Direct dbSNP search term", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about SNPs/variants",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the UCSC Genome Browser API using natural language or a direct endpoint.",
        "name": "query_ucsc",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full URL or endpoint spec", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genomic data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Ensembl REST API using natural language or a direct endpoint.",
        "name": "query_ensembl",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct Ensembl endpoint or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genomic data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the OpenTargets Platform API using natural language or a direct GraphQL query.",
        "name": "query_opentarget",
        "optional_parameters": [
            {"name": "query", "type": "str", "description": "Direct GraphQL query string", "default": None},
            {"name": "variables", "type": "dict", "description": "Variables for GraphQL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": False},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about targets/diseases",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Monarch Initiative API using natural language or a direct endpoint.",
        "name": "query_monarch",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct endpoint or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results (adds limit param)", "default": 2},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": False},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genes/diseases/phenotypes",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the OpenFDA API using natural language or direct parameters.",
        "name": "query_openfda",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct endpoint or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results (limit)", "default": 100},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
            {"name": "search_params", "type": "dict", "description": "Search parameters mapping", "default": None},
            {"name": "sort_params", "type": "dict", "description": "Sort parameters mapping", "default": None},
            {"name": "count_params", "type": "str", "description": "Field to count", "default": None},
            {"name": "skip_results", "type": "int", "description": "Skip for pagination", "default": 0},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about OpenFDA data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the GWAS Catalog API using natural language or a direct endpoint.",
        "name": "query_gwas_catalog",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Endpoint name (e.g., 'studies')", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results per page (size)", "default": 3},
        ],
        "required_parameters": [
            {"name": "prompt", "type": "str", "description": "Natural language query about GWAS data", "default": None}
        ],
    },
    {
        "description": "Query gnomAD for variants in a gene using natural language or direct gene symbol.",
        "name": "query_gnomad",
        "optional_parameters": [
            {"name": "gene_symbol", "type": "str", "description": "Gene symbol (e.g., 'BRCA1')", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about genetic variants",
                "default": None,
            }
        ],
    },
    {
        "description": "Identify a DNA or protein sequence using NCBI BLAST.",
        "name": "blast_sequence",
        "optional_parameters": [],
        "required_parameters": [
            {"name": "sequence", "type": "str", "description": "Query sequence", "default": None},
            {"name": "database", "type": "str", "description": "BLAST database (e.g., core_nt or nr)", "default": None},
            {"name": "program", "type": "str", "description": "BLAST program (blastn or blastp)", "default": None},
        ],
    },
    {
        "description": "Query the Reactome database using natural language or a direct endpoint; optionally download pathway diagrams.",
        "name": "query_reactome",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct endpoint or full URL", "default": None},
            {
                "name": "download",
                "type": "bool",
                "description": "Download pathway diagram if available",
                "default": False,
            },
            {"name": "output_dir", "type": "str", "description": "Directory to save downloads", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about biological pathways",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the RegulomeDB database using natural language or direct endpoint.",
        "name": "query_regulomedb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct RegulomeDB endpoint URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": False},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about regulatory elements",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the PRIDE proteomics database using natural language or a direct endpoint.",
        "name": "query_pride",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full endpoint to query", "default": None},
            {"name": "max_results", "type": "int", "description": "Maximum number of results", "default": 3},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about proteomics data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Guide to PHARMACOLOGY (GtoPdb) database using natural language or a direct endpoint.",
        "name": "query_gtopdb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about drug targets/ligands/interactions",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the ReMap database for regulatory elements and transcription factor binding.",
        "name": "query_remap",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about TF binding sites",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Mouse Phenome Database (MPD) using natural language or a direct endpoint.",
        "name": "query_mpd",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about mouse phenotypes/strains",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the Electron Microscopy Data Bank (EMDB) using natural language or a direct endpoint.",
        "name": "query_emdb",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Full API endpoint to query", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about EM structures",
                "default": None,
            }
        ],
    },
    {
        "description": "Query Synapse REST API for biomedical datasets/files using natural language or structured search parameters. Supports optional authentication via SYNAPSE_AUTH_TOKEN.",
        "name": "query_synapse",
        "optional_parameters": [
            {
                "name": "query_term",
                "type": "str|list[str]",
                "description": "Search term(s) (AND logic across list)",
                "default": None,
            },
            {
                "name": "return_fields",
                "type": "list[str]",
                "description": "Fields to return",
                "default": ["name", "node_type", "description"],
            },
            {"name": "max_results", "type": "int", "description": "Max results (20 typical, up to 50)", "default": 20},
            {
                "name": "query_type",
                "type": "str",
                "description": "'dataset', 'file', or 'folder'",
                "default": "dataset",
            },
            {
                "name": "verbose",
                "type": "bool",
                "description": "Return full API response or formatted",
                "default": True,
            },
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about biomedical data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the PubChem PUG-REST API using natural language or a direct endpoint.",
        "name": "query_pubchem",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct PubChem API endpoint or full URL",
                "default": None,
            },
            {"name": "max_results", "type": "int", "description": "Max results (rate-limited to 5 rps)", "default": 5},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about chemical compounds",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the ChEMBL REST API via natural language, direct endpoint, or identifiers (chembl_id, smiles, molecule_name).",
        "name": "query_chembl",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct ChEMBL API endpoint or full URL",
                "default": None,
            },
            {"name": "chembl_id", "type": "str", "description": "ChEMBL ID (e.g., 'CHEMBL25')", "default": None},
            {"name": "smiles", "type": "str", "description": "SMILES for similarity/substructure", "default": None},
            {"name": "molecule_name", "type": "str", "description": "Molecule name to search", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results", "default": 20},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about bioactivity data",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the UniChem 2.0 REST API using natural language or a direct endpoint.",
        "name": "query_unichem",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct UniChem endpoint or full URL", "default": None},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about chemical cross-references",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the ClinicalTrials.gov API v2 using natural language or a direct endpoint.",
        "name": "query_clinicaltrials",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct ClinicalTrials.gov endpoint or full URL",
                "default": None,
            },
            {"name": "max_results", "type": "int", "description": "Page size for results (pageSize)", "default": 10},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about clinical trials",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the DailyMed RESTful API using natural language or a direct endpoint.",
        "name": "query_dailymed",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct DailyMed endpoint or full URL", "default": None},
            {"name": "format", "type": "str", "description": "'json' or 'xml'", "default": "json"},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about drug labeling",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the QuickGO API using natural language or a direct endpoint.",
        "name": "query_quickgo",
        "optional_parameters": [
            {"name": "endpoint", "type": "str", "description": "Direct QuickGO endpoint or full URL", "default": None},
            {"name": "max_results", "type": "int", "description": "Max results (limit, up to 100)", "default": 25},
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about GO terms/annotations",
                "default": None,
            }
        ],
    },
    {
        "description": "Query the ENCODE Portal API to locate functional genomics data (experiments, files, biosamples, datasets).",
        "name": "query_encode",
        "optional_parameters": [
            {
                "name": "endpoint",
                "type": "str",
                "description": "Direct ENCODE Portal endpoint or full URL",
                "default": None,
            },
            {
                "name": "max_results",
                "type": "int|str",
                "description": "Limit for search endpoints (number or 'all')",
                "default": 25,
            },
            {"name": "verbose", "type": "bool", "description": "Return detailed results", "default": True},
        ],
        "required_parameters": [
            {
                "name": "prompt",
                "type": "str",
                "description": "Natural language query about functional genomics data",
                "default": None,
            }
        ],
    },
    {
        "description": "Given genomic coordinates, retrieve intersecting ENCODE SCREEN cCREs.",
        "name": "region_to_ccre_screen",
        "optional_parameters": [
            {"name": "assembly", "type": "str", "description": "Genome assembly (e.g., 'GRCh38')", "default": "GRCh38"}
        ],
        "required_parameters": [
            {"name": "coord_chrom", "type": "str", "description": "Chromosome (e.g., 'chr12')", "default": None},
            {"name": "coord_start", "type": "int", "description": "Start coordinate", "default": None},
            {"name": "coord_end", "type": "int", "description": "End coordinate", "default": None},
        ],
    },
    {
        "description": "Given a cCRE accession, return k nearest genes sorted by distance.",
        "name": "get_genes_near_ccre",
        "optional_parameters": [
            {"name": "k", "type": "int", "description": "Number of nearby genes to return", "default": 10}
        ],
        "required_parameters": [
            {
                "name": "accession",
                "type": "str",
                "description": "ENCODE cCRE accession ID (e.g., 'EH38E1516980')",
                "default": None,
            },
            {"name": "assembly", "type": "str", "description": "Genome assembly (e.g., 'GRCh38')", "default": None},
            {
                "name": "chromosome",
                "type": "str",
                "description": "Chromosome of the cCRE (e.g., 'chr12')",
                "default": None,
            },
        ],
    },
    {
        "description": (
            "Select or count variants carried in one or more GRCh38 regions of the 1000 Genomes "
            "Project cohort (3,202 whole-genome-sequenced individuals), at the level of individual "
            "genotypes. Cohort-wide when 'samples' is None, or restricted to variants carried by the "
            "named individuals when 'samples' is given. Set count_only=True first to size the result "
            "set cheaply, then call again with count_only=False to retrieve records. Coordinates are "
            "1-based inclusive GRCh38 - resolve a gene/feature to coordinates with an authoritative "
            "source (e.g. Ensembl) BEFORE calling; a misplaced region returns results for the wrong "
            "locus without error. Returned variants carry 22 fields including 1000 Genomes AF/AC/AN, "
            "sample counts with hom/het/missing genotypes (separate female-on-X and male-on-XY breakdowns), "
            "gnomAD v4.1 exomes and genomes AF, AlphaMissense score, and HGVSp amino-acid change. "
            "AF/am_score of 0.0 means absent/unannotated in that source. Annotation-filter vocabularies: "
            "see references/annotation_vocabularies.md. Requires the optional 'dnaerys' package."
        ),
        "name": "query_1000_genomes_variants",
        "optional_parameters": [
            {
                "name": "chrom",
                "type": "str",
                "description": "Chromosome for single-region mode, e.g. 'chr17','17','X','MT'. Requires start and end.",
                "default": None,
            },
            {
                "name": "start",
                "type": "int",
                "description": "1-based inclusive start (single-region mode, with chrom).",
                "default": None,
            },
            {
                "name": "end",
                "type": "int",
                "description": "1-based inclusive end, >= start (single-region mode, with chrom).",
                "default": None,
            },
            {
                "name": "ref",
                "type": "str",
                "description": "Narrow to one reference allele (single-region mode only).",
                "default": None,
            },
            {
                "name": "alt",
                "type": "str",
                "description": "Narrow to one alternate allele (single-region mode only).",
                "default": None,
            },
            {
                "name": "regions",
                "type": "List[str]",
                "description": "Multi-region mode: list of 'CHR:START-END' strings; mutually exclusive with chrom/start/end and with ref/alt.",
                "default": None,
            },
            {
                "name": "samples",
                "type": "List[str]",
                "description": "Restrict to variants carried by these case-sensitive individual IDs (e.g. ['NA19240']). None = whole cohort.",
                "default": None,
            },
            {
                "name": "het_only",
                "type": "bool",
                "description": "Heterozygous (0/1) carriage only. Mutually exclusive with hom_only; default includes both.",
                "default": False,
            },
            {
                "name": "hom_only",
                "type": "bool",
                "description": "Homozygous (1/1) carriage only. Mutually exclusive with het_only; default includes both.",
                "default": False,
            },
            {
                "name": "count_only",
                "type": "bool",
                "description": "If True return only the integer count (cheap sizing); if False return the matching variant records.",
                "default": False,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Hard cap on returned variants when selecting (ignored if count_only, or if page_size is set). Default 200.",
                "default": 200,
            },
            {
                "name": "page_size",
                "type": "int",
                "description": "If set, retrieve ALL matching variants in pages of this size (full walk); overrides limit.",
                "default": None,
            },
            {
                "name": "af_lt",
                "type": "float",
                "description": "Keep variants with 1000 Genomes dataset AF < this value.",
                "default": None,
            },
            {
                "name": "af_gt",
                "type": "float",
                "description": "Keep variants with 1000 Genomes dataset AF > this value.",
                "default": None,
            },
            {
                "name": "gnomad_exomes_af_lt",
                "type": "float",
                "description": "Keep variants with gnomAD v4.1 exomes AF < this value (includes AF=0/unannotated).",
                "default": None,
            },
            {
                "name": "gnomad_exomes_af_gt",
                "type": "float",
                "description": "Keep variants with gnomAD v4.1 exomes AF > this value (use >0 to require presence in gnomAD exomes).",
                "default": None,
            },
            {
                "name": "gnomad_genomes_af_lt",
                "type": "float",
                "description": "Keep variants with gnomAD v4.1 genomes AF < this value (includes AF=0/unannotated).",
                "default": None,
            },
            {
                "name": "gnomad_genomes_af_gt",
                "type": "float",
                "description": "Keep variants with gnomAD v4.1 genomes AF > this value (use >0 to require presence in gnomAD genomes).",
                "default": None,
            },
            {
                "name": "clin_significance",
                "type": "List[str]",
                "description": "ClinVar significance terms (OR within field); benign token is CLNSIG_BENIGN. See annotation_vocabularies.md.",
                "default": None,
            },
            {
                "name": "consequence",
                "type": "List[str]",
                "description": "Sequence Ontology consequence terms, e.g. ['MISSENSE_VARIANT','STOP_GAINED'] (OR within field).",
                "default": None,
            },
            {
                "name": "impact",
                "type": "List[str]",
                "description": "VEP impact terms from HIGH,MODERATE,LOW,MODIFIER (OR within field).",
                "default": None,
            },
            {
                "name": "variant_type",
                "type": "List[str]",
                "description": "SO variant-class terms, e.g. ['SNV','INSERTION'] (OR within field).",
                "default": None,
            },
            {
                "name": "feature_type",
                "type": "List[str]",
                "description": "VEP feature types from TRANSCRIPT,REGULATORYFEATURE,MOTIFFEATURE (OR within field).",
                "default": None,
            },
            {
                "name": "bio_type",
                "type": "List[str]",
                "description": "VEP biotype terms, e.g. ['PROTEIN_CODING'] (OR within field).",
                "default": None,
            },
            {
                "name": "alpha_missense_class",
                "type": "List[str]",
                "description": "AlphaMissense classes from AM_LIKELY_BENIGN,AM_LIKELY_PATHOGENIC,AM_AMBIGUOUS. Mutually exclusive with alpha_missense_score_lt/gt.",
                "default": None,
            },
            {
                "name": "alpha_missense_score_lt",
                "type": "float",
                "description": "Keep variants with AlphaMissense score < this value. Mutually exclusive with alpha_missense_class.",
                "default": None,
            },
            {
                "name": "alpha_missense_score_gt",
                "type": "float",
                "description": "Keep variants with AlphaMissense score > this value. Mutually exclusive with alpha_missense_class.",
                "default": None,
            },
            {
                "name": "biallelic_only",
                "type": "bool",
                "description": "Keep only biallelic sites. Mutually exclusive with multiallelic_only.",
                "default": False,
            },
            {
                "name": "multiallelic_only",
                "type": "bool",
                "description": "Keep only multiallelic sites. Mutually exclusive with biallelic_only.",
                "default": False,
            },
            {
                "name": "exclude_males",
                "type": "bool",
                "description": "Exclude male samples. Mutually exclusive with exclude_females.",
                "default": False,
            },
            {
                "name": "exclude_females",
                "type": "bool",
                "description": "Exclude female samples. Mutually exclusive with exclude_males.",
                "default": False,
            },
            {
                "name": "min_len_bp",
                "type": "int",
                "description": "Minimum alternate-allele length in bp.",
                "default": None,
            },
            {
                "name": "max_len_bp",
                "type": "int",
                "description": "Maximum alternate-allele length in bp.",
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Count or list the 1000 Genomes Project individuals (3,202-person cohort, GRCh38) who carry "
            "at least one variant matching the given region and annotation criteria. Set count_only=True "
            "first to size the set, then count_only=False to get the individual IDs (names only). To see "
            "which variants qualified a given set of individuals, feed the returned IDs into "
            "query_1000_genomes_variants(samples=...). Coordinates are 1-based inclusive GRCh38 - resolve "
            "gene/feature to coordinates with an authoritative source BEFORE calling. Annotation-filter "
            "vocabularies: references/annotation_vocabularies.md. Requires the optional 'dnaerys' package."
        ),
        "name": "query_1000_genomes_carriers",
        "optional_parameters": [
            {
                "name": "chrom",
                "type": "str",
                "description": "Chromosome for single-region mode, e.g. 'chr17','17','X','MT'. Requires start and end.",
                "default": None,
            },
            {
                "name": "start",
                "type": "int",
                "description": "1-based inclusive start (single-region mode, with chrom).",
                "default": None,
            },
            {
                "name": "end",
                "type": "int",
                "description": "1-based inclusive end, >= start (single-region mode, with chrom).",
                "default": None,
            },
            {
                "name": "ref",
                "type": "str",
                "description": "Narrow to one reference allele (single-region mode only).",
                "default": None,
            },
            {
                "name": "alt",
                "type": "str",
                "description": "Narrow to one alternate allele (single-region mode only).",
                "default": None,
            },
            {
                "name": "regions",
                "type": "List[str]",
                "description": "Multi-region mode: list of 'CHR:START-END' strings; mutually exclusive with chrom/start/end and with ref/alt.",
                "default": None,
            },
            {
                "name": "het_only",
                "type": "bool",
                "description": "Heterozygous (0/1) carriage only. Mutually exclusive with hom_only; default includes both.",
                "default": False,
            },
            {
                "name": "hom_only",
                "type": "bool",
                "description": "Homozygous (1/1) carriage only. Mutually exclusive with het_only; default includes both.",
                "default": False,
            },
            {
                "name": "count_only",
                "type": "bool",
                "description": "If True return only the count of carrying individuals; if False return their IDs.",
                "default": False,
            },
            {
                "name": "skip",
                "type": "int",
                "description": "Skip the first N individuals (select mode).",
                "default": None,
            },
            {
                "name": "limit",
                "type": "int",
                "description": "Return at most N individuals (select mode).",
                "default": None,
            },
            {
                "name": "af_lt",
                "type": "float",
                "description": "Keep variants with 1000 Genomes dataset AF < this value.",
                "default": None,
            },
            {
                "name": "af_gt",
                "type": "float",
                "description": "Keep variants with 1000 Genomes dataset AF > this value.",
                "default": None,
            },
            {
                "name": "gnomad_exomes_af_lt",
                "type": "float",
                "description": "gnomAD v4.1 exomes AF < this value (includes AF=0/unannotated).",
                "default": None,
            },
            {
                "name": "gnomad_exomes_af_gt",
                "type": "float",
                "description": "gnomAD v4.1 exomes AF > this value (use >0 to require presence).",
                "default": None,
            },
            {
                "name": "gnomad_genomes_af_lt",
                "type": "float",
                "description": "gnomAD v4.1 genomes AF < this value (includes AF=0/unannotated).",
                "default": None,
            },
            {
                "name": "gnomad_genomes_af_gt",
                "type": "float",
                "description": "gnomAD v4.1 genomes AF > this value (use >0 to require presence).",
                "default": None,
            },
            {
                "name": "clin_significance",
                "type": "List[str]",
                "description": "ClinVar significance terms (OR within field); benign token is CLNSIG_BENIGN.",
                "default": None,
            },
            {
                "name": "consequence",
                "type": "List[str]",
                "description": "SO consequence terms, e.g. ['MISSENSE_VARIANT','STOP_GAINED'] (OR within field).",
                "default": None,
            },
            {
                "name": "impact",
                "type": "List[str]",
                "description": "VEP impact terms from HIGH,MODERATE,LOW,MODIFIER (OR within field).",
                "default": None,
            },
            {
                "name": "variant_type",
                "type": "List[str]",
                "description": "SO variant-class terms (OR within field).",
                "default": None,
            },
            {
                "name": "feature_type",
                "type": "List[str]",
                "description": "VEP feature types (OR within field).",
                "default": None,
            },
            {
                "name": "bio_type",
                "type": "List[str]",
                "description": "VEP biotype terms (OR within field).",
                "default": None,
            },
            {
                "name": "alpha_missense_class",
                "type": "List[str]",
                "description": "AlphaMissense classes; mutually exclusive with alpha_missense_score_lt/gt.",
                "default": None,
            },
            {
                "name": "alpha_missense_score_lt",
                "type": "float",
                "description": "AlphaMissense score < this value. Mutually exclusive with alpha_missense_class.",
                "default": None,
            },
            {
                "name": "alpha_missense_score_gt",
                "type": "float",
                "description": "AlphaMissense score > this value. Mutually exclusive with alpha_missense_class.",
                "default": None,
            },
            {
                "name": "biallelic_only",
                "type": "bool",
                "description": "Keep only biallelic sites. Mutually exclusive with multiallelic_only.",
                "default": False,
            },
            {
                "name": "multiallelic_only",
                "type": "bool",
                "description": "Keep only multiallelic sites. Mutually exclusive with biallelic_only.",
                "default": False,
            },
            {
                "name": "exclude_males",
                "type": "bool",
                "description": "Exclude male samples. Mutually exclusive with exclude_females.",
                "default": False,
            },
            {
                "name": "exclude_females",
                "type": "bool",
                "description": "Exclude female samples. Mutually exclusive with exclude_males.",
                "default": False,
            },
            {
                "name": "min_len_bp",
                "type": "int",
                "description": "Minimum alternate-allele length in bp.",
                "default": None,
            },
            {
                "name": "max_len_bp",
                "type": "int",
                "description": "Maximum alternate-allele length in bp.",
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "At a single GRCh38 position in the 1000 Genomes Project cohort (3,202 individuals), count or "
            "list the individuals with a homozygous-reference (0/0) genotype. Position is 1-based; resolve "
            "coordinates with an authoritative source BEFORE calling. The 'count' is a sentinel: -1 = no "
            "variant exists at this position at all (variant_present=False); 0 = a variant exists but no "
            "individual is homozygous reference; >0 = the number of homozygous-reference individuals. "
            "count_only=True returns just the sentinel; count_only=False also returns the individual IDs. "
            "Requires the optional 'dnaerys' package."
        ),
        "name": "query_1000_genomes_homozygous_reference",
        "optional_parameters": [
            {
                "name": "count_only",
                "type": "bool",
                "description": "If True return only the sentinel count; if False also return the homozygous-reference individual IDs.",
                "default": False,
            },
        ],
        "required_parameters": [
            {"name": "chrom", "type": "str", "description": "Chromosome, e.g. 'chr17','17','X','MT'.", "default": None},
            {"name": "position", "type": "int", "description": "1-based position.", "default": None},
        ],
    },
    {
        "description": (
            "Pairwise relatedness between two named 1000 Genomes Project individuals: the relatedness "
            "degree (TWINS_MONOZYGOTIC / FIRST_DEGREE / SECOND_DEGREE / THIRD_DEGREE / UNRELATED) and the "
            "KING between-family robust kinship coefficient (phi_bwf; ~0.5 monozygotic, 0.25 first-degree, "
            "0.125 second-degree, 0.0625 third-degree). Sample IDs are case-sensitive (e.g. 'NA19238'). "
            "Requires the optional 'dnaerys' package."
        ),
        "name": "query_1000_genomes_kinship",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "sample1",
                "type": "str",
                "description": "First individual ID (case-sensitive), e.g. 'NA19238'.",
                "default": None,
            },
            {
                "name": "sample2",
                "type": "str",
                "description": "Second individual ID (case-sensitive), e.g. 'NA19240'.",
                "default": None,
            },
        ],
    },
    {
        "description": (
            "Dataset totals for the 1000 Genomes Project cohort served by OneKGPd: total individuals "
            "(3,202), female/male split, total variant count, genome assembly (GRCh38), and the per-cohort "
            "breakdown. Takes no parameters and doubles as a connectivity check for the live 1000 Genomes "
            "query endpoint. Requires the optional 'dnaerys' package."
        ),
        "name": "get_1000_genomes_dataset_info",
        "optional_parameters": [],
        "required_parameters": [],
    },
    {
        "description": (
            "Pedigree and population metadata for specific 1000 Genomes Project individuals, from bundled "
            "cohort data (offline, no network). For each given sample ID returns family ID, gender, "
            "paternal/maternal IDs, relationship (mother/father/child), children recorded in the cohort, "
            "population and superpopulation (code and full name), and phase-3 inclusion. Sample IDs are "
            "case-sensitive (e.g. 'NA19240'). These are the same IDs used by the live variant/kinship "
            "tools, so the two layers compose."
        ),
        "name": "get_1000_genomes_sample_metadata",
        "optional_parameters": [],
        "required_parameters": [
            {
                "name": "samples",
                "type": "List[str]",
                "description": "Case-sensitive individual IDs, e.g. ['NA19240','HG00096'].",
                "default": None,
            },
        ],
    },
    {
        "description": (
            "Enumerate the population structure of the 1000 Genomes Project cohort (offline). With "
            "level='population' lists all 26 populations, each with its superpopulation and sample count; "
            "with level='superpopulation' lists the 5 superpopulations (AFR, AMR, EAS, EUR, SAS), each with "
            "sample count and constituent populations. Use this to discover the valid population/"
            "superpopulation codes and full names accepted by the other metadata tools."
        ),
        "name": "list_1000_genomes_populations",
        "optional_parameters": [
            {
                "name": "level",
                "type": "str",
                "description": "'population' (26 populations) or 'superpopulation' (5 superpopulations).",
                "default": "population",
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "Demographic statistics for named 1000 Genomes populations and/or superpopulations (offline): "
            "sample count, male/female split, phase-3 count, and trio count (offspring with both parents in "
            "the dataset). Pass 'populations' (codes or full names) for a per-population breakdown, and/or "
            "'superpopulations' for per-superpopulation totals with a nested per-population breakdown. "
            "Values match case-insensitively by short code or full name; at least one of the two arguments "
            "is required. Use list_1000_genomes_populations to discover valid values."
        ),
        "name": "get_1000_genomes_population_stats",
        "optional_parameters": [
            {
                "name": "populations",
                "type": "List[str]",
                "description": "Population codes or full names for a per-population breakdown, e.g. ['YRI','CHS'].",
                "default": None,
            },
            {
                "name": "superpopulations",
                "type": "List[str]",
                "description": "Superpopulation codes or full names for per-superpopulation summaries with nested per-population breakdown, e.g. ['EAS','EUR'].",
                "default": None,
            },
        ],
        "required_parameters": [],
    },
    {
        "description": (
            "List the 1000 Genomes Project individual IDs in a given population and/or superpopulation "
            "(offline). Provide 'population' and/or 'superpopulation' as a code or full name "
            "(case-insensitive); when both are given the results are intersected. Supports skip/limit "
            "pagination (default skip 0, limit 50, max 3202). Returns sorted, paginated sample IDs suitable "
            "for feeding into query_1000_genomes_variants(samples=...) or get_1000_genomes_sample_metadata."
        ),
        "name": "select_1000_genomes_samples_by_population",
        "optional_parameters": [
            {
                "name": "population",
                "type": "str",
                "description": "Population code or full name (case-insensitive), e.g. 'YRI'.",
                "default": None,
            },
            {
                "name": "superpopulation",
                "type": "str",
                "description": "Superpopulation code or full name (case-insensitive), e.g. 'AFR'.",
                "default": None,
            },
            {"name": "skip", "type": "int", "description": "Number of results to skip (>= 0).", "default": 0},
            {"name": "limit", "type": "int", "description": "Max results to return (1..3202).", "default": 50},
        ],
        "required_parameters": [],
    },
]
