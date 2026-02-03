"""
Amplicon Table Tool for querying the Amplicon Repository.

This module provides a function to retrieve amplicon records from a local
aggregated CSV file, supporting various filtering options for tissue of origin,
classification, gene annotations, genomic location, and copy number features.
"""

import os
import re
from typing import Any

import pandas as pd

try:
    from biomni.config import default_config
except Exception:
    default_config = None


# Default columns to return when select is not specified
DEFAULT_COLUMNS = [
    "Sample name",
    "AA amplicon number",
    "Classification",
    "Tissue of origin",
    "Oncogenes",
    "All genes",
    "Feature maximum copy number",
    "Feature median copy number",
    "Captured interval length",
    "Complexity score",
    "Location",
    "Reference version",
]


def _parse_genomic_location(location: str) -> tuple[str, int, int] | None:
    """Parse a genomic location string in the format chrN:start-end.

    Args:
        location: Genomic interval string (e.g., 'chr2:12112-123421312')

    Returns:
        Tuple of (chromosome, start, end) or None if parsing fails
    """
    pattern = r"^(chr[0-9XYM]+):(\d+)-(\d+)$"
    match = re.match(pattern, location, re.IGNORECASE)
    if match:
        chrom = match.group(1).lower()
        start = int(match.group(2))
        end = int(match.group(3))
        return (chrom, start, end)
    return None


def _intervals_overlap(intervals_str: str, query_chrom: str, query_start: int, query_end: int) -> bool:
    """Check if any interval in the intervals string overlaps with the query region.

    Args:
        intervals_str: String containing genomic intervals (e.g., 'chr1:100-200,chr2:300-400')
        query_chrom: Query chromosome
        query_start: Query start position
        query_end: Query end position

    Returns:
        True if any interval overlaps with the query region
    """
    if pd.isna(intervals_str) or not intervals_str:
        return False

    # Try to parse intervals from the string
    # Handle various formats: 'chr1:100-200', 'chr1:100-200,chr2:300-400', etc.
    interval_pattern = r"(chr[0-9XYM]+):(\d+)-(\d+)"
    matches = re.findall(interval_pattern, str(intervals_str), re.IGNORECASE)

    for chrom, start, end in matches:
        chrom = chrom.lower()
        start = int(start)
        end = int(end)

        # Check if this interval overlaps with the query
        if chrom == query_chrom and start <= query_end and end >= query_start:
            return True

    return False


def _gene_in_field(gene: str, field_value: str) -> bool:
    """Check if a gene symbol is present in a field value.

    Args:
        gene: Gene symbol to search for
        field_value: Field value (comma-separated gene list or similar)

    Returns:
        True if the gene is found in the field
    """
    if pd.isna(field_value) or not field_value:
        return False

    # Normalize and search
    field_str = str(field_value).upper()
    gene_upper = gene.upper()

    # Check for exact match (word boundary)
    # Handle comma-separated, semicolon-separated, or space-separated lists
    genes_in_field = re.split(r"[,;\s]+", field_str)
    return gene_upper in [g.strip() for g in genes_in_field]


def amplicon_table(
    tissue_of_origin: str | None = None,
    classification: str | None = None,
    gene: str | None = None,
    gene_field: str | None = None,
    ncbi_gene_id: str | None = None,
    genomic_location: str | None = None,
    complexity_score_min: float | None = None,
    complexity_score_max: float | None = None,
    captured_interval_length_min: float | None = None,
    captured_interval_length_max: float | None = None,
    feature_max_copy_number_min: float | None = None,
    feature_median_copy_number_min: float | None = None,
    reference_version: str | None = None,
    select: list[str] | None = None,
    limit: int | None = None,
    offset: int | None = None,
    csv_path: str | None = None,
) -> dict[str, Any]:
    """Retrieve amplicon records from the Amplicon Repository CSV file.

    This function provides access to the ground-truth data source for all questions
    about cancer amplicons, including counts, examples, tissue distribution,
    classification (ecDNA, BFB, Linear, Complex-non-cyclic), genomic location,
    gene annotations, copy-number features, and amplicon complexity.

    Args:
        tissue_of_origin: Cancer tissue of origin (e.g., breast, lung, ovary).
            Case-insensitive exact match.
        classification: Amplicon classification. Must be one of:
            'ecDNA', 'BFB', 'Linear', 'Complex-non-cyclic'.
        gene: Gene symbol to search for (e.g., ERBB2, EGFR, MYC).
        gene_field: Which gene column(s) to search. One of:
            'oncogenes', 'all_genes', 'either' (default: 'either').
        ncbi_gene_id: NCBI Gene ID to filter amplicons by.
        genomic_location: Genomic interval in chromosome coordinates,
            formatted as chrN:start-end (e.g., chr2:12112-123421312).
        complexity_score_min: Minimum amplicon complexity score.
        complexity_score_max: Maximum amplicon complexity score.
        captured_interval_length_min: Minimum captured interval length (bp).
        captured_interval_length_max: Maximum captured interval length (bp).
        feature_max_copy_number_min: Minimum feature maximum copy number.
        feature_median_copy_number_min: Minimum feature median copy number.
        reference_version: Reference genome version (e.g., hg19, hg38).
        select: Columns to return. If not specified, returns default columns.
        limit: Maximum number of rows to return (default 50, maximum 500).
        offset: Row offset for pagination (default 0).
        csv_path: Path to the aggregated_results.csv file. If not specified,
            uses the default path from config.

    Returns:
        Dictionary containing:
            - summary: Brief description of the results
            - row_count_total: Total number of matching rows (before pagination)
            - row_count_returned: Number of rows returned (after pagination)
            - filters_applied: Dictionary of filters that were applied
            - rows: List of matching amplicon records
            - schema: List of column names in the returned data

    Raises:
        FileNotFoundError: If the CSV file is not found
        ValueError: If invalid parameter values are provided
    """
    # Set defaults
    if limit is None:
        limit = 50
    if offset is None:
        offset = 0
    if gene_field is None:
        gene_field = "either"

    # Validate parameters
    limit = min(max(1, limit), 500)  # Clamp between 1 and 500
    offset = max(0, offset)

    valid_classifications = ["ecDNA", "BFB", "Linear", "Complex-non-cyclic"]
    if classification is not None and classification not in valid_classifications:
        raise ValueError(f"Invalid classification '{classification}'. Must be one of: {valid_classifications}")

    valid_gene_fields = ["oncogenes", "all_genes", "either"]
    if gene_field not in valid_gene_fields:
        raise ValueError(f"Invalid gene_field '{gene_field}'. Must be one of: {valid_gene_fields}")

    # Determine CSV path
    if csv_path is None:
        # Try to get path from config
        data_path = "/home/yasaman/amplicon-repo-agentai/data/biomni_data/data_lake/"
        if default_config is not None:
            data_path = default_config.path
        # Also check environment variable
        data_path = os.getenv("BIOMNI_PATH") or os.getenv("BIOMNI_DATA_PATH") or data_path
        csv_path = os.path.join(data_path, "CCLE.csv")

    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Amplicon data file not found at '{csv_path}'. "
            "Please ensure the aggregated_results.csv file is available "
            "or specify a custom path using the csv_path parameter."
        )

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Track applied filters
    filters_applied = {}

    # Apply filters
    mask = pd.Series([True] * len(df))

    # Tissue of origin filter (case-insensitive exact match)
    if tissue_of_origin is not None:
        if "Tissue of origin" in df.columns:
            mask &= df["Tissue of origin"].str.lower() == tissue_of_origin.lower()
            filters_applied["tissue_of_origin"] = tissue_of_origin
        else:
            raise ValueError("Column 'Tissue of origin' not found in the data")

    # Classification filter
    if classification is not None:
        if "Classification" in df.columns:
            mask &= df["Classification"] == classification
            filters_applied["classification"] = classification
        else:
            raise ValueError("Column 'Classification' not found in the data")

    # Gene filter
    if gene is not None:
        if gene_field == "oncogenes":
            if "Oncogenes" not in df.columns:
                raise ValueError("Column 'Oncogenes' not found in the data")
            mask &= df["Oncogenes"].apply(lambda x: _gene_in_field(gene, x))
        elif gene_field == "all_genes":
            if "All genes" not in df.columns:
                raise ValueError("Column 'All genes' not found in the data")
            mask &= df["All genes"].apply(lambda x: _gene_in_field(gene, x))
        else:  # either
            oncogenes_col = "Oncogenes" in df.columns
            all_genes_col = "All genes" in df.columns
            if not oncogenes_col and not all_genes_col:
                raise ValueError("Neither 'Oncogenes' nor 'All genes' columns found in the data")

            gene_mask = pd.Series([False] * len(df))
            if oncogenes_col:
                gene_mask |= df["Oncogenes"].apply(lambda x: _gene_in_field(gene, x))
            if all_genes_col:
                gene_mask |= df["All genes"].apply(lambda x: _gene_in_field(gene, x))
            mask &= gene_mask

        filters_applied["gene"] = gene
        filters_applied["gene_field"] = gene_field

    # NCBI Gene ID filter
    if ncbi_gene_id is not None:
        if "NCBI Gene IDs" in df.columns:
            mask &= df["NCBI Gene IDs"].astype(str).str.contains(str(ncbi_gene_id), na=False)
            filters_applied["ncbi_gene_id"] = ncbi_gene_id
        else:
            # Search in columns that might contain NCBI gene IDs
            ncbi_cols = [col for col in df.columns if "ncbi" in col.lower() or "gene_id" in col.lower()]
            if ncbi_cols:
                ncbi_mask = pd.Series([False] * len(df))
                for col in ncbi_cols:
                    ncbi_mask |= df[col].astype(str).str.contains(str(ncbi_gene_id), na=False)
                mask &= ncbi_mask
                filters_applied["ncbi_gene_id"] = ncbi_gene_id
            else:
                filters_applied["ncbi_gene_id"] = ncbi_gene_id
                filters_applied["ncbi_gene_id_warning"] = "No NCBI gene ID column found; filter may not apply correctly"

    # Genomic location filter
    if genomic_location is not None:
        parsed = _parse_genomic_location(genomic_location)
        if parsed is None:
            raise ValueError(
                f"Invalid genomic_location format '{genomic_location}'. "
                "Expected format: chrN:start-end (e.g., chr2:12112-123421312)"
            )

        query_chrom, query_start, query_end = parsed

        # Use Location column for genomic location filtering
        if "Location" in df.columns:
            mask &= df["Location"].apply(
                lambda x: _intervals_overlap(x, query_chrom, query_start, query_end)
            )
            filters_applied["genomic_location"] = genomic_location
        else:
            # Fallback to any interval column
            interval_cols = [col for col in df.columns if "interval" in col.lower() or "location" in col.lower()]
            if interval_cols:
                mask &= df[interval_cols[0]].apply(
                    lambda x: _intervals_overlap(x, query_chrom, query_start, query_end)
                )
                filters_applied["genomic_location"] = genomic_location
            else:
                raise ValueError("No Location column found in the data for genomic location filtering")

    # Numeric range filters (mapping parameter names to actual column names)
    numeric_filters = [
        ("Complexity score", complexity_score_min, complexity_score_max, "complexity_score"),
        ("Captured interval length", captured_interval_length_min, captured_interval_length_max, "captured_interval_length"),
        ("Feature maximum copy number", feature_max_copy_number_min, None, "feature_max_copy_number"),
        ("Feature median copy number", feature_median_copy_number_min, None, "feature_median_copy_number"),
    ]

    for col_name, min_val, max_val, filter_key in numeric_filters:
        if min_val is not None or max_val is not None:
            if col_name in df.columns:
                if min_val is not None:
                    mask &= df[col_name] >= min_val
                    filters_applied[f"{filter_key}_min"] = min_val
                if max_val is not None:
                    mask &= df[col_name] <= max_val
                    filters_applied[f"{filter_key}_max"] = max_val
            else:
                raise ValueError(f"Column '{col_name}' not found in the data")

    # Reference version filter
    if reference_version is not None:
        if "Reference version" in df.columns:
            mask &= df["Reference version"].str.lower() == reference_version.lower()
            filters_applied["reference_version"] = reference_version
        else:
            raise ValueError("Column 'Reference version' not found in the data")

    # Apply the filter mask
    filtered_df = df[mask]

    # Get total count before pagination
    row_count_total = len(filtered_df)

    # Apply pagination
    filtered_df = filtered_df.iloc[offset : offset + limit]
    row_count_returned = len(filtered_df)

    # Select columns
    if select is not None:
        # Validate requested columns
        invalid_cols = [col for col in select if col not in df.columns]
        if invalid_cols:
            raise ValueError(f"Invalid columns requested: {invalid_cols}. Available columns: {list(df.columns)}")
        output_columns = select
    else:
        # Use default columns, filtering to only those that exist
        output_columns = [col for col in DEFAULT_COLUMNS if col in df.columns]
        # If no default columns exist, return all columns
        if not output_columns:
            output_columns = list(df.columns)

    # Prepare output
    result_df = filtered_df[output_columns]
    rows = result_df.to_dict(orient="records")

    # Generate summary
    filter_desc = []
    if tissue_of_origin:
        filter_desc.append(f"tissue={tissue_of_origin}")
    if classification:
        filter_desc.append(f"classification={classification}")
    if gene:
        filter_desc.append(f"gene={gene}")
    if genomic_location:
        filter_desc.append(f"location={genomic_location}")

    filter_str = ", ".join(filter_desc) if filter_desc else "no filters"
    summary = f"Found {row_count_total} amplicon records ({filter_str}). Returning rows {offset + 1}-{offset + row_count_returned}."

    return {
        "summary": summary,
        "row_count_total": row_count_total,
        "row_count_returned": row_count_returned,
        "filters_applied": filters_applied,
        "rows": rows,
        "schema": output_columns,
    }
