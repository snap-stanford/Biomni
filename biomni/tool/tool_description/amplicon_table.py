# biomni/tool_desc/amplicon_table.py

description = [
    {
        "name": "query_amplicons",
        "description": (
            "Query and filter amplicon records from a CSV file. This tool provides "
            "ground-truth amplicon data including counts, examples, tissue distribution, "
            "classification (ecDNA, BFB, Linear, Complex-non-cyclic), genomic location, "
            "gene annotations, copy-number features, and amplicon complexity. Supports "
            "flexible filtering by tissue, classification, genes, genomic location, "
            "copy number thresholds, and complexity scores. Use csv_path to specify "
            "any compatible amplicon CSV file. The agent should call this tool before "
            "answering any amplicon-related question."
        ),
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "tissue_of_origin",
                "type": "string",
                "description": "Cancer tissue of origin (e.g., breast, lung, ovary, urinary tract). Accepts a single tissue value. Case-insensitive exact match.",
            },
            {
                "name": "classification",
                "type": "string",
                "enum": ["ecDNA", "BFB", "Linear", "Complex-non-cyclic"],
                "description": "Amplicon classification. Accepts a single value from: ecDNA, BFB, Linear, or Complex-non-cyclic.",
            },
            {
                "name": "gene",
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Gene symbol(s) to search for as a list (e.g., ['ERBB2', 'EGFR', 'MYC'] or ['CASC15', 'CDKAL1', 'E2F3']). "
                    "The filter returns amplicons containing one or more of the specified genes (OR logic). "
                    "Useful for finding amplicons affected by a set of driver genes."
                ),
            },
            {
                "name": "gene_field",
                "type": "string",
                "enum": ["oncogenes", "all_genes", "either"],
                "description": "Specifies which gene column(s) to search when filtering by gene: 'oncogenes' (search only oncogenes column), 'all_genes' (search all genes column), or 'either' (search both columns, default).",
            },
            {
                "name": "ncbi_gene_id",
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "NCBI Gene ID(s) to filter amplicons by as a list (e.g., ['NR_015410', 'NM_017774', 'NM_001949']). "
                    "Similar to gene filtering but uses NCBI gene identifiers instead of gene symbols. "
                    "Returns amplicons containing one or more of the specified gene IDs (OR logic)."
                ),
            },
            {
                "name": "genomic_location",
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Genomic interval(s) to filter by as a list of chromosome coordinates in the format chrN:start-end "
                    "(e.g., ['chr3:9099507-9519221', 'chr3:9521655-10184493', 'chr3:10187258-14005720']). "
                    "Each entry specifies a chromosome and base-pair range (start and end positions are absolute coordinates). "
                    "Amplicons with intervals overlapping any of the specified regions are returned (OR logic)."
                ),
            },
            {
                "name": "complexity_score_min",
                "type": "number",
                "description": "Minimum amplicon structural complexity score. Measures how complex the structure of an amplicon is; higher values indicate more complex structures. Use to filter amplicons by complexity.",
            },
            {
                "name": "complexity_score_max",
                "type": "number",
                "description": "Maximum amplicon structural complexity score. Measures how complex the structure of an amplicon is; higher values indicate more complex structures. Use to filter amplicons by complexity.",
            },
            {
                "name": "captured_interval_length_min",
                "type": "number",
                "description": "Minimum captured interval length (bp). This value represents the sum of all genomic intervals listed in the genomic location columns for an amplicon (i.e., total captured interval length across all segments). Use to filter amplicons by total captured length.",
            },
            {
                "name": "captured_interval_length_max",
                "type": "number",
                "description": "Maximum captured interval length (bp). This value represents the sum of all genomic intervals listed in the genomic location columns for an amplicon (i.e., total captured interval length across all segments). Use to filter amplicons by total captured length.",
            },
            {
                "name": "feature_max_copy_number_min",
                "type": "number",
                "description": "Minimum threshold on the feature maximum copy number. This filters amplicons by the highest segment-level copy number observed within the amplicon (i.e., the maximum copy number among segments).",
            },
            {
                "name": "feature_median_copy_number_min",
                "type": "number",
                "description": "Minimum threshold on the feature median copy number. This filters amplicons by the median segment-level copy number across all segments in the amplicon.",
            },
            {
                "name": "reference_version",
                "type": "string",
                "description": "Reference genome version (e.g., hg19, hg38). Accepts a single value and is used to interpret genomic coordinates and determine genomic regions of genes in the specified reference.",
            },
            {
                "name": "select",
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Columns to return (default: key identifying and feature columns). "
                    "Useful for reducing output size."
                ),
            },
            {
                "name": "limit",
                "type": "integer",
                "description": "Maximum number of rows to return. If not specified, returns all matching rows.",
            },
            {
                "name": "offset",
                "type": "integer",
                "description": "Row offset for pagination (default 0).",
            },
            {
                "name": "csv_path",
                "type": "string",
                "description": "Path to the amplicon CSV file (e.g., CCLE.csv, aggregated_results.csv, or any compatible amplicon data file). If not specified, uses default config path.",
            },
                {
                    "name": "add_normalized_columns",
                    "type": "boolean",
                    "description": "If true (default), adds lower snake_case aliases for each returned column (e.g., 'Classification' -> 'classification') and includes schema_normalized and column_map in the response.",
                },
        ],
        "returns": {
            "type": "object",
            "fields": [
                "summary",
                "row_count_total",
                "row_count_returned",
                "filters_applied",
                "rows",
                "schema",
                    "schema_normalized",
                    "column_map",
            ],
        },
    }
]
