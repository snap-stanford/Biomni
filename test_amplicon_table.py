#!/usr/bin/env python3
"""
Automated test suite for amplicon_table tool.
Based on test cases from Tests.pdf
"""

import sys

sys.path.insert(0, "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai")

import numpy as np
from biomni.tool.amplicon_table import query_amplicons

# CSV path
CSV_PATH = "/Users/siavashraeisidehkordi/amplicon-repo-agentai/amplicon-repo-agentai/biomni/data/biomni_data/data_lake/CCLE.csv"

# Test results
test_results = []


def test(name, fn):
    """Run a test and track results."""
    try:
        fn()
        test_results.append({"name": name, "status": "✓ PASS", "error": None})
        print(f"✓ {name}")
    except AssertionError as e:
        test_results.append({"name": name, "status": "✗ FAIL", "error": str(e)})
        print(f"✗ {name}: {e}")
    except Exception as e:
        test_results.append({"name": name, "status": "✗ ERROR", "error": str(e)})
        print(f"✗ {name} (ERROR): {e}")


# ============================================================================
# TEST 1: Is YES1 amplified as ecDNA or BFB in CCLE? In which samples?
# ============================================================================


def test_yes1_gene_query():
    """Test YES1 gene query - should find amplifications."""
    result = query_amplicons(gene="YES1", csv_path=CSV_PATH)

    assert result["row_count_total"] > 0, "Should find YES1 amplifications"
    assert any(row["Sample name"] for row in result["rows"]), "Should have sample names"

    # Check that we found BFB classification (from expected results)
    classifications = [row["Classification"] for row in result["rows"]]
    assert "BFB" in classifications, "Should find YES1 in BFB"


def test_yes1_by_classification():
    """Test YES1 query filtered by classification."""
    result = query_amplicons(gene="YES1", classification="BFB", csv_path=CSV_PATH)

    assert result["row_count_total"] > 0, "Should find YES1 in BFB"
    # Verify all results are BFB
    for row in result["rows"]:
        assert row["Classification"] == "BFB", f"Expected BFB but got {row['Classification']}"


# ============================================================================
# TEST 2: BFB amplification statistics
# ============================================================================


def test_bfb_count():
    """Test BFB count - should be 114 features."""
    result = query_amplicons(classification="BFB", csv_path=CSV_PATH)

    assert result["row_count_total"] == 114, f"Expected 114 BFB features, got {result['row_count_total']}"


def test_bfb_captured_interval_length_stats():
    """Test BFB captured interval length statistics."""
    result = query_amplicons(classification="BFB", csv_path=CSV_PATH, select=["Captured interval length"])

    assert result["row_count_total"] == 114, "Should have 114 BFB records"

    # Extract captured interval lengths
    lengths = [
        float(row["Captured interval length"])
        for row in result["rows"]
        if row.get("Captured interval length") is not None
    ]

    assert len(lengths) == 114, "All BFB records should have captured interval length"

    # Verify statistics match expected values (from Tests.pdf)
    mean_length = np.mean(lengths)
    median_length = np.median(lengths)
    min_length = np.min(lengths)
    max_length = np.max(lengths)

    print(
        f"  BFB size stats - mean: {mean_length:.2e}, median: {median_length:.2e}, min: {min_length:.2e}, max: {max_length:.2e}"
    )

    # Expected ranges from Tests.pdf
    assert 2.2e6 < mean_length < 2.3e6, f"Mean should be ~2.27e6, got {mean_length:.2e}"
    assert 1.2e6 < median_length < 1.3e6, f"Median should be ~1.28e6, got {median_length:.2e}"
    assert min_length < 1e5, f"Min should be ~59k, got {min_length:.2e}"


# ============================================================================
# TEST 3: Classification distribution
# ============================================================================


def test_classification_distribution():
    """Test that classification counts match expected distribution."""
    classifications = {}
    for cls in ["Linear", "ecDNA", "Complex-non-cyclic", "BFB"]:
        result = query_amplicons(classification=cls, csv_path=CSV_PATH)
        classifications[cls] = result["row_count_total"]

    sum(classifications.values())

    # Expected from Tests.pdf: Linear:550, ecDNA:297, Complex-non-cyclic:196, BFB:114
    expected = {"Linear": 550, "ecDNA": 297, "Complex-non-cyclic": 196, "BFB": 114}

    for cls, count in expected.items():
        assert classifications[cls] == count, f"Expected {count} {cls} but got {classifications[cls]}"

    print(f"  Distribution: {classifications}")


# ============================================================================
# TEST 4: Multi-filter queries
# ============================================================================


def test_tissue_filter():
    """Test tissue_of_origin filter."""
    result = query_amplicons(tissue_of_origin="breast", csv_path=CSV_PATH)

    assert result["row_count_total"] > 0, "Should find breast cancer amplicons"
    for row in result["rows"]:
        assert row["Tissue of origin"].lower() == "breast", f"Expected breast but got {row['Tissue of origin']}"


def test_multiple_tissues():
    """Test multiple tissue_of_origin values (OR matching)."""
    result = query_amplicons(tissue_of_origin=["breast", "lung"], csv_path=CSV_PATH)

    assert result["row_count_total"] > 0, "Should find breast or lung cancer samples"
    tissues = {row["Tissue of origin"].lower() for row in result["rows"]}
    assert tissues.issubset({"breast", "lung"}), f"Unexpected tissues: {tissues}"


def test_tissue_and_classification():
    """Test combined tissue and classification filters."""
    result = query_amplicons(tissue_of_origin="breast", classification="BFB", csv_path=CSV_PATH)

    if result["row_count_total"] > 0:
        for row in result["rows"]:
            assert row["Tissue of origin"].lower() == "breast"
            assert row["Classification"] == "BFB"


# ============================================================================
# TEST 5: Pagination
# ============================================================================


def test_limit_offset():
    """Test limit and offset parameters."""
    # Get first 10
    result1 = query_amplicons(classification="BFB", limit=10, offset=0, csv_path=CSV_PATH)
    assert len(result1["rows"]) == 10, "Should return exactly 10 rows"

    # Get next 10
    result2 = query_amplicons(classification="BFB", limit=10, offset=10, csv_path=CSV_PATH)
    assert len(result2["rows"]) == 10, "Should return exactly 10 rows"

    # Rows should be different (use Sample name and Location for comparison)
    rows1 = [(row["Sample name"], row["Location"]) for row in result1["rows"]]
    rows2 = [(row["Sample name"], row["Location"]) for row in result2["rows"]]

    assert set(rows1).isdisjoint(set(rows2)), "Offset results should be different from first batch"


# ============================================================================
# TEST 6: Column selection
# ============================================================================


def test_select_columns():
    """Test select parameter to return specific columns."""
    select_cols = ["Sample name", "Classification", "Captured interval length"]
    result = query_amplicons(classification="BFB", select=select_cols, limit=5, csv_path=CSV_PATH)

    assert result["schema"] == select_cols, "Schema should match selected columns"
    for row in result["rows"]:
        assert set(row.keys()) == set(select_cols), "Row keys should match selected columns"


# ============================================================================
# TEST 7: Edge cases
# ============================================================================


def test_no_results():
    """Test query that returns no results."""
    result = query_amplicons(gene="NONEXISTENT_GENE_XYZ", csv_path=CSV_PATH)

    assert result["row_count_total"] == 0, "Should return no results for non-existent gene"
    assert result["rows"] == [], "Rows should be empty"


def test_multiple_genes():
    """Test multiple gene query (OR matching)."""
    result = query_amplicons(gene=["YES1", "TYMS"], csv_path=CSV_PATH)

    assert result["row_count_total"] > 0, "Should find at least one gene"


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AMPLICON_TABLE TEST SUITE")
    print("=" * 70)
    print()

    # Test 1: YES1 query
    print("TEST 1: YES1 Gene Query")
    print("-" * 70)
    test("YES1 gene exists in data", test_yes1_gene_query)
    test("YES1 filtered by BFB classification", test_yes1_by_classification)
    print()

    # Test 2: BFB statistics
    print("TEST 2: BFB Amplification Statistics")
    print("-" * 70)
    test("BFB feature count is 114", test_bfb_count)
    test("BFB captured interval length stats", test_bfb_captured_interval_length_stats)
    print()

    # Test 3: Classification distribution
    print("TEST 3: Classification Distribution")
    print("-" * 70)
    test("Classification distribution matches expected", test_classification_distribution)
    print()

    # Test 4: Multi-filters
    print("TEST 4: Multi-Filter Queries")
    print("-" * 70)
    test("Tissue filter (single)", test_tissue_filter)
    test("Tissue filter (multiple)", test_multiple_tissues)
    test("Combined tissue and classification", test_tissue_and_classification)
    print()

    # Test 5: Pagination
    print("TEST 5: Pagination")
    print("-" * 70)
    test("Limit and offset", test_limit_offset)
    print()

    # Test 6: Column selection
    print("TEST 6: Column Selection")
    print("-" * 70)
    test("Select specific columns", test_select_columns)
    print()

    # Test 7: Edge cases
    print("TEST 7: Edge Cases")
    print("-" * 70)
    test("No results for non-existent gene", test_no_results)
    test("Multiple genes (OR matching)", test_multiple_genes)
    print()

    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in test_results if "PASS" in r["status"])
    failed = sum(1 for r in test_results if "FAIL" in r["status"])
    errors = sum(1 for r in test_results if "ERROR" in r["status"])
    total = len(test_results)

    print(f"Total: {total} | Passed: {passed} | Failed: {failed} | Errors: {errors}")
    print()

    if failed > 0 or errors > 0:
        print("FAILED TESTS:")
        for r in test_results:
            if "FAIL" in r["status"] or "ERROR" in r["status"]:
                print(f"  - {r['name']}: {r['error']}")
    else:
        print("✓ ALL TESTS PASSED!")

    print("=" * 70)
