# biomni/tool_desc/amplicon_table.py

description = [
    {
        "name": "amplicon_table",
        "description": (
            "Retrieve authoritative amplicon records from the Amplicon Repository "
            "using a local aggregated CSV file. This tool provides the ground-truth "
            "data source for all questions about cancer amplicons, including counts, "
            "examples, tissue distribution, classification (ecDNA, BFB, Linear, "
            "Complex-non-cyclic), genomic location, gene annotations, copy-number "
            "features, and amplicon complexity. The agent must call this tool before "
            "answering any amplicon-related question."
        ),
        "required_parameters": [],
        "optional_parameters": [
            {
                "name": "tissue_of_origin",
                "type": "string",
                "description": "Cancer tissue of origin (e.g., breast, lung, ovary). Case-insensitive exact match.",
            },
            {
                "name": "classification",
                "type": "string",
                "enum": ["ecDNA", "BFB", "Linear", "Complex-non-cyclic"],
                "description": "Amplicon classification using canonical labels.",
            },
            {
                "name": "gene",
                "type": "string",
                "description": "Gene symbol to search for (e.g., ERBB2, EGFR, MYC).",
            },
            {
                "name": "gene_field",
                "type": "string",
                "enum": ["oncogenes", "all_genes", "either"],
                "description": "Which gene column(s) to search (default: either).",
            },
            {
                "name": "ncbi_gene_id",
                "type": "string",
                "description": "NCBI Gene ID to filter amplicons by (optional).",
            },
            {
                "name": "genomic_location",
                "type": "string",
                "description": (
                    "Genomic interval in chromosome coordinates using base pairs, "
                    "formatted as chrN:start-end (e.g., chr2:12112-123421312). "
                    "Coordinates are interpreted as absolute base-pair positions."
                ),
            },
            {
                "name": "complexity_score_min",
                "type": "number",
                "description": "Minimum amplicon complexity score.",
            },
            {
                "name": "complexity_score_max",
                "type": "number",
                "description": "Maximum amplicon complexity score.",
            },
            {
                "name": "captured_interval_length_min",
                "type": "number",
                "description": "Minimum captured interval length (bp).",
            },
            {
                "name": "captured_interval_length_max",
                "type": "number",
                "description": "Maximum captured interval length (bp).",
            },
            {
                "name": "feature_max_copy_number_min",
                "type": "number",
                "description": "Minimum feature maximum copy number.",
            },
            {
                "name": "feature_median_copy_number_min",
                "type": "number",
                "description": "Minimum feature median copy number.",
            },
            {
                "name": "reference_version",
                "type": "string",
                "description": "Reference genome version (e.g., hg19, hg38).",
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
                "description": "Maximum number of rows to return (default 50, maximum 500).",
            },
            {
                "name": "offset",
                "type": "integer",
                "description": "Row offset for pagination (default 0).",
            },
            {
                "name": "csv_path",
                "type": "string",
                "description": "Path to the aggregated_results.csv file (optional).",
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
            ],
        },
    }
]
