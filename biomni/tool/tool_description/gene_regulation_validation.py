description = [
    {
        "name": "validate_gene_regulatory_claims",
        "description": (
            "Validate computational gene regulatory predictions against "
            "experimental knockdown data from the Replogle Perturb-seq atlas. "
            "Takes a CSV of claims (upstream_gene regulates downstream_gene in "
            "a predicted direction) and checks each against ~11,000 gene "
            "knockdowns in K562 cells and ~2,000 in RPE1 cells. Each claim is "
            "graded as VALIDATED, PARTIALLY_SUPPORTED, WEAK, CONTRADICTED, or "
            "UNTESTABLE based on the knockdown effect size, direction match, "
            "and percentile rank versus random genes after the same knockdown. "
            "Use this when you have a gene regulatory network or transcription "
            "factor-target predictions and want to check which edges are "
            "supported by experimental perturbation evidence."
        ),
        "required_parameters": [
            {
                "name": "claims_input",
                "type": "str",
                "description": (
                    "Path to a CSV file with columns: upstream_gene, "
                    "downstream_gene, predicted_direction (UP or DOWN). "
                    "Each row is one regulatory claim to validate."
                ),
                "default": None,
            },
            {
                "name": "data_lake_path",
                "type": "str",
                "description": "Path to the Biomni datalake root directory.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "output_folder",
                "type": "str",
                "description": ("Directory for output files. Results CSV will be saved here."),
                "default": "./tmp/",
            },
            {
                "name": "cell_type",
                "type": "str",
                "description": (
                    "Which cell line to validate against: 'K562', 'RPE1', or 'all' (uses both). Default is 'all'."
                ),
                "default": "all",
            },
        ],
    },
    {
        "name": "test_regulatory_specificity",
        "description": (
            "Permutation-based specificity test for gene regulatory claims. "
            "Tests whether your chosen upstream regulators are special for "
            "the downstream targets, or whether random genes from the knockdown "
            "dataset would produce equally strong regulatory evidence. For each "
            "permutation, downstream targets stay fixed while upstream genes "
            "are replaced with random knockdowns. Reports p-values for three "
            "metrics: number of supported claims, mean effect percentile, and "
            "direction match count. This is the critical test that separates "
            "'these genes are individually important' from 'these genes "
            "specifically regulate those targets.' Optionally accepts a custom "
            "comparison pool (e.g., other disease-associated genes) for a "
            "harder, more biologically meaningful test."
        ),
        "required_parameters": [
            {
                "name": "claims_input",
                "type": "str",
                "description": (
                    "Path to a CSV file with columns: upstream_gene, downstream_gene, predicted_direction (UP or DOWN)."
                ),
                "default": None,
            },
            {
                "name": "data_lake_path",
                "type": "str",
                "description": "Path to the Biomni datalake root directory.",
                "default": None,
            },
        ],
        "optional_parameters": [
            {
                "name": "n_permutations",
                "type": "int",
                "description": (
                    "Number of permutations to run. Higher values give more "
                    "precise p-values but take longer. Default is 1000."
                ),
                "default": 1000,
            },
            {
                "name": "comparison_pool_file",
                "type": "str",
                "description": (
                    "Optional path to a text file with one gene symbol per "
                    "line. Restricts the random pool to these genes (e.g., "
                    "other NDD transcription factors). If not provided, all "
                    "knockdown genes are used."
                ),
                "default": None,
            },
            {
                "name": "output_folder",
                "type": "str",
                "description": "Directory for output files.",
                "default": "./tmp/",
            },
        ],
    },
]
