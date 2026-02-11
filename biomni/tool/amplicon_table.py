"""
Amplicon Query Tool for querying the Amplicon Repository.

This module provides a function to retrieve amplicon records from a local
aggregated CSV file, supporting various filtering options for tissue of origin,
classification, gene annotations, genomic location, and copy number features.
Gene-related and location filters support lists for SQL-like IN clause matching.
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


def _normalize_column_name(name: str) -> str:
    """Normalize column names to lower snake_case for easier programmatic access."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()


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
        field_value: Field value (JSON-formatted list or comma-separated)

    Returns:
        True if the gene is found in the field
    """
    import json
    
    if pd.isna(field_value) or not field_value:
        return False

    field_str = str(field_value).strip()
    gene_upper = gene.upper()

    # Try to parse as JSON list first
    genes_list = []
    if field_str.startswith('['):
        try:
            parsed = json.loads(field_str)
            if isinstance(parsed, list):
                genes_list = [str(g).strip().strip("'\"") for g in parsed if g]
        except (json.JSONDecodeError, ValueError):
            pass
    
    # If JSON parsing failed, try comma/semicolon splitting
    if not genes_list:
        genes_in_field = re.split(r"[,;\s]+", field_str.upper())
        genes_list = [g.strip().strip("'\"") for g in genes_in_field if g]
    
    genes_normalized = [g.upper() for g in genes_list]
    return gene_upper in genes_normalized


def query_amplicons(
    tissue_of_origin: str | list[str] | None = None,
    classification: str | list[str] | None = None,
    gene: str | list[str] | None = None,
    gene_field: str | None = None,
    ncbi_gene_id: str | list[str] | None = None,
    genomic_location: str | list[str] | None = None,
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
    add_normalized_columns: bool | None = None,
) -> dict[str, Any]:
    """Retrieve amplicon records from the Amplicon Repository CSV file.

    This function provides access to the ground-truth data source for all questions
    about cancer amplicons, including counts, examples, tissue distribution,
    classification (ecDNA, BFB, Linear, Complex-non-cyclic), genomic location,
    gene annotations, copy-number features, and amplicon complexity.

    Args:
        tissue_of_origin: Cancer tissue of origin (e.g., breast, lung, ovary).
            Can be a single string or list of strings for OR matching.
            Case-insensitive matching.
        classification: Amplicon classification. Can be a single string or list of
            strings for OR matching. Valid values include 'ecDNA', 'BFB', 'Linear',
            'Complex-non-cyclic'.
        gene: Gene symbol(s) to search for. Can be a single string or list
            of strings for OR matching (e.g., 'MYC' or ['MYC', 'EGFR']).
        gene_field: Which gene column(s) to search. One of:
            'oncogenes', 'all_genes', 'either' (default: 'either').
        ncbi_gene_id: NCBI Gene ID(s) to filter by. Can be a single string
            or list of strings for OR matching.
        genomic_location: Genomic interval(s) in chromosome coordinates,
            formatted as chrN:start-end (e.g., chr2:12112-123421312).
            Can be a single string or list of strings for OR matching.
        complexity_score_min: Minimum amplicon complexity score.
        complexity_score_max: Maximum amplicon complexity score.
        captured_interval_length_min: Minimum captured interval length (bp).
        captured_interval_length_max: Maximum captured interval length (bp).
        feature_max_copy_number_min: Minimum feature maximum copy number.
        feature_median_copy_number_min: Minimum feature median copy number.
        reference_version: Reference genome version (e.g., hg19, hg38).
        select: Columns to return. If not specified, returns default columns.
        limit: Maximum number of rows to return. If not specified, returns all matching rows.
        offset: Row offset for pagination (default 0).
        csv_path: Path to the aggregated_results.csv file. If not specified,
            uses the default path from config.
        add_normalized_columns: If True, adds lower snake_case aliases for
            returned columns in each row (e.g., 'Classification' -> 'classification').
            Defaults to True to make downstream access more consistent.

    Returns:
        Dictionary containing:
            - summary: Brief description of the results
            - row_count_total: Total number of matching rows (before pagination)
            - row_count_returned: Number of rows returned (after pagination)
            - filters_applied: Dictionary of filters that were applied
            - rows: List of matching amplicon records
            - schema: List of column names in the returned data
            - schema_normalized: List of normalized column names (if enabled)
            - column_map: Mapping of normalized -> original column names (if enabled)

    Raises:
        FileNotFoundError: If the CSV file is not found
        ValueError: If invalid parameter values are provided
    """
    # Set defaults
    if offset is None:
        offset = 0
    if add_normalized_columns is None:
        add_normalized_columns = True
    if gene_field is None:
        gene_field = "either"

    # Validate parameters
    if limit is not None:
        limit = max(1, limit)  # Ensure at least 1 if specified
    offset = max(0, offset)

    valid_classifications = ["ecDNA", "BFB", "Linear", "Complex-non-cyclic"]
    if classification is not None:
        # Allow single string or list of strings
        if isinstance(classification, str):
            cls_list = [classification]
        elif isinstance(classification, list):
            cls_list = classification
        else:
            raise ValueError("Invalid type for classification; must be string or list of strings")
        invalid = [c for c in cls_list if c not in valid_classifications]
        if invalid:
            raise ValueError(f"Invalid classification value(s) {invalid}. Must be one of: {valid_classifications}")

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

    # Tissue of origin filter (supports list for OR matching)
    if tissue_of_origin is not None:
        if "Tissue of origin" in df.columns:
            tissue_list = [tissue_of_origin] if isinstance(tissue_of_origin, str) else tissue_of_origin
            tissue_mask = pd.Series([False] * len(df))
            for tissue in tissue_list:
                tissue_mask |= df["Tissue of origin"].str.lower() == tissue.lower()
            mask &= tissue_mask
            filters_applied["tissue_of_origin"] = tissue_of_origin
        else:
            raise ValueError("Column 'Tissue of origin' not found in the data")

    # Classification filter (supports list for OR matching)
    if classification is not None:
        if "Classification" in df.columns:
            class_list = [classification] if isinstance(classification, str) else classification
            class_mask = pd.Series([False] * len(df))
            for c in class_list:
                class_mask |= df["Classification"] == c
            mask &= class_mask
            filters_applied["classification"] = classification
        else:
            raise ValueError("Column 'Classification' not found in the data")

    # Gene filter (supports list for OR matching)
    if gene is not None:
        gene_list = [gene] if isinstance(gene, str) else gene

        def _any_gene_in_field(genes: list[str], field_value: str) -> bool:
            return any(_gene_in_field(g, field_value) for g in genes)

        if gene_field == "oncogenes":
            if "Oncogenes" not in df.columns:
                raise ValueError("Column 'Oncogenes' not found in the data")
            mask &= df["Oncogenes"].apply(lambda x: _any_gene_in_field(gene_list, x))
        elif gene_field == "all_genes":
            if "All genes" not in df.columns:
                raise ValueError("Column 'All genes' not found in the data")
            mask &= df["All genes"].apply(lambda x: _any_gene_in_field(gene_list, x))
        else:  # either
            oncogenes_col = "Oncogenes" in df.columns
            all_genes_col = "All genes" in df.columns
            if not oncogenes_col and not all_genes_col:
                raise ValueError("Neither 'Oncogenes' nor 'All genes' columns found in the data")

            gene_mask = pd.Series([False] * len(df))
            if oncogenes_col:
                gene_mask |= df["Oncogenes"].apply(lambda x: _any_gene_in_field(gene_list, x))
            if all_genes_col:
                gene_mask |= df["All genes"].apply(lambda x: _any_gene_in_field(gene_list, x))
            mask &= gene_mask

        filters_applied["gene"] = gene
        filters_applied["gene_field"] = gene_field

    # NCBI Gene ID filter (supports list for OR matching)
    if ncbi_gene_id is not None:
        ncbi_id_list = [ncbi_gene_id] if isinstance(ncbi_gene_id, str) else ncbi_gene_id

        def _any_ncbi_id_match(ids: list[str], field_value: str) -> bool:
            if pd.isna(field_value) or not field_value:
                return False
            field_str = str(field_value)
            return any(str(id_) in field_str for id_ in ids)

        if "NCBI Gene IDs" in df.columns:
            mask &= df["NCBI Gene IDs"].apply(lambda x: _any_ncbi_id_match(ncbi_id_list, x))
            filters_applied["ncbi_gene_id"] = ncbi_gene_id
        else:
            # Search in columns that might contain NCBI gene IDs
            ncbi_cols = [col for col in df.columns if "ncbi" in col.lower() or "gene_id" in col.lower()]
            if ncbi_cols:
                ncbi_mask = pd.Series([False] * len(df))
                for col in ncbi_cols:
                    ncbi_mask |= df[col].apply(lambda x: _any_ncbi_id_match(ncbi_id_list, x))
                mask &= ncbi_mask
                filters_applied["ncbi_gene_id"] = ncbi_gene_id
            else:
                filters_applied["ncbi_gene_id"] = ncbi_gene_id
                filters_applied["ncbi_gene_id_warning"] = "No NCBI gene ID column found; filter may not apply correctly"

    # Genomic location filter (supports list for OR matching)
    if genomic_location is not None:
        location_list = [genomic_location] if isinstance(genomic_location, str) else genomic_location

        # Parse and validate all locations
        parsed_locations = []
        for loc in location_list:
            parsed = _parse_genomic_location(loc)
            if parsed is None:
                raise ValueError(
                    f"Invalid genomic_location format '{loc}'. "
                    "Expected format: chrN:start-end (e.g., chr2:12112-123421312)"
                )
            parsed_locations.append(parsed)

        def _any_location_overlaps(intervals_str: str) -> bool:
            return any(
                _intervals_overlap(intervals_str, chrom, start, end)
                for chrom, start, end in parsed_locations
            )

        # Use Location column for genomic location filtering
        if "Location" in df.columns:
            mask &= df["Location"].apply(_any_location_overlaps)
            filters_applied["genomic_location"] = genomic_location
        else:
            # Fallback to any interval column
            interval_cols = [col for col in df.columns if "interval" in col.lower() or "location" in col.lower()]
            if interval_cols:
                mask &= df[interval_cols[0]].apply(_any_location_overlaps)
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
    if limit is not None:
        filtered_df = filtered_df.iloc[offset : offset + limit]
    else:
        filtered_df = filtered_df.iloc[offset:]
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

    # Optionally add normalized column aliases for easier access
    schema_normalized: list[str] | None = None
    column_map: dict[str, str] | None = None
    if add_normalized_columns:
        column_map = {_normalize_column_name(col): col for col in output_columns}
        schema_normalized = list(column_map.keys())
        for row in rows:
            for normalized, original in column_map.items():
                if normalized not in row:
                    row[normalized] = row.get(original)

    # Generate summary
    filter_desc = []
    if tissue_of_origin:
        filter_desc.append(f"tissue={tissue_of_origin}")
    if classification:
        filter_desc.append(f"classification={classification}")
    if gene:
        gene_str = gene if isinstance(gene, str) else ",".join(gene)
        filter_desc.append(f"gene={gene_str}")
    if genomic_location:
        loc_str = genomic_location if isinstance(genomic_location, str) else ",".join(genomic_location)
        filter_desc.append(f"location={loc_str}")

    filter_str = ", ".join(filter_desc) if filter_desc else "no filters"
    summary = f"Found {row_count_total} amplicon records ({filter_str}). Returning rows {offset + 1}-{offset + row_count_returned}."

    result = {
        "summary": summary,
        "row_count_total": row_count_total,
        "row_count_returned": row_count_returned,
        "filters_applied": filters_applied,
        "rows": rows,
        "schema": output_columns,
    }

    if add_normalized_columns:
        result["schema_normalized"] = schema_normalized
        result["column_map"] = column_map

    return result
