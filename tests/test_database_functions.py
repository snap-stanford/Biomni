#!/usr/bin/env python3
"""
Comprehensive test script for database functions.
Tests each function using the agent and logs results.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from biomni.agent.a1 import A1
from dotenv import load_dotenv

# Load API variables from .env file
load_dotenv(dotenv_path="../.env")


# Set up logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"database_tests_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )

    return logging.getLogger(__name__)


def test_database_function(agent, function_name, test_query, description):
    """Test a specific database function using the agent."""
    logger = logging.getLogger(__name__)

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Testing: {function_name}")
    logger.info(f"Query: {test_query}")
    logger.info(f"Description: {description}")
    logger.info(f"{'=' * 60}")

    try:
        # Run the agent query
        logger.info("Executing agent query...")
        result = agent.go(test_query)

        # Log the result
        logger.info("Query completed successfully!")
        logger.info(f"Result type: {type(result)}")

        if isinstance(result, str):
            logger.info(f"Result length: {len(result)} characters")
            # Format the result for better readability
            formatted_result = result.replace("\\n", "\n").replace("\\t", "\t")
            logger.info(f"Result preview:\n{formatted_result}...")
        else:
            # Format complex results for better readability
            if isinstance(result, list | tuple):
                logger.info(f"Result (list/tuple with {len(result)} items):")
                for i, item in enumerate(result):
                    if isinstance(item, str):
                        formatted_item = item.replace("\\n", "\n").replace("\\t", "\t")
                        logger.info(f"  Item {i + 1}:\n{formatted_item}...")
                    else:
                        logger.info(f"  Item {i + 1}: {str(item)}...")
            else:
                logger.info(f"Result: {result}")

        return {
            "function": function_name,
            "query": test_query,
            "description": description,
            "status": "SUCCESS",
            "result_type": str(type(result)),
            "result_length": len(str(result)) if isinstance(result, str) else "N/A",
            "error": None,
        }

    except Exception as e:
        error_msg = f"Error testing {function_name}: {str(e)}"
        logger.error(error_msg)

        return {
            "function": function_name,
            "query": test_query,
            "description": description,
            "status": "FAILED",
            "result_type": "N/A",
            "result_length": "N/A",
            "error": str(e),
        }


def main():
    """Main test function."""
    logger = setup_logging()

    logger.info("Starting comprehensive database function tests")
    logger.info(f"Test started at: {datetime.now()}")

    # Initialize the agent
    try:
        logger.info("Initializing agent...")
        agent = A1(expected_data_lake_files=[])
        logger.info("Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        return

    # Define test cases for each database/function
    test_cases = [
        # ChEMBL Database tests
        {
            "function": "query_chembl",
            "query": "Find binding targets for aspirin using ChEMBL database",
            "description": "Test ChEMBL database query for aspirin binding targets",
        },
        {
            "function": "query_chembl",
            "query": "Search for ibuprofen molecule information in ChEMBL",
            "description": "Test ChEMBL database query for ibuprofen molecule data",
        },
        # ClinicalTrials.gov tests
        {
            "function": "query_clinicaltrials",
            "query": "Find clinical trials for diabetes treatment in phase 3 clinical trials",
            "description": "Test ClinicalTrials.gov query for diabetes trials",
        },
        # UniChem tests
        {
            "function": "query_unichem",
            "query": "Find cross-references for aspirin across different chemical databases",
            "description": "Test UniChem query for chemical cross-references",
        },
        # DailyMed tests
        {
            "function": "query_dailymed",
            "query": "Search for metformin drug label information",
            "description": "Test DailyMed query for metformin drug label",
        },
        # openFDA tests
        {
            "function": "query_openfda",
            "query": "Search for drug recalls related to aspirin",
            "description": "Test openFDA query for recent drug recalls",
        },
        # ENCODE tests
        {
            "function": "query_encode",
            "query": "Find ENCODE CHIP-Seq experiments where in human HEK293 cells",
            "description": "Test ENCODE query for transcription factor data",
        },
        {
            "function": "query_encode",
            "query": "Search for mouse brain cell biosamples in ENCODE",
            "description": "Test ENCODE query for chromatin accessibility data",
        },
        # QuickGO tests
        {
            "function": "query_quickgo",
            "query": "Find GO terms related to apoptosis and cell death",
            "description": "Test QuickGO query for Gene Ontology terms",
        },
    ]

    # Run all tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}/{len(test_cases)}")

        result = test_database_function(
            agent=agent,
            function_name=test_case["function"],
            test_query=test_case["query"],
            description=test_case["description"],
        )

        results.append(result)

        # Add a small delay between tests to avoid overwhelming the APIs
        import time

        time.sleep(2)

    # Generate summary report
    logger.info(f"\n{'=' * 60}")
    logger.info("TEST SUMMARY REPORT")
    logger.info(f"{'=' * 60}")

    successful_tests = [r for r in results if r["status"] == "SUCCESS"]
    failed_tests = [r for r in results if r["status"] == "FAILED"]

    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {len(successful_tests)}")
    logger.info(f"Failed: {len(failed_tests)}")
    logger.info(f"Success rate: {len(successful_tests) / len(results) * 100:.1f}%")

    # Log successful tests
    if successful_tests:
        logger.info(f"\nSuccessful tests ({len(successful_tests)}):")
        for result in successful_tests:
            logger.info(f"✓ {result['function']}: {result['description']}")

    # Log failed tests
    if failed_tests:
        logger.info(f"\nFailed tests ({len(failed_tests)}):")
        for result in failed_tests:
            logger.info(f"✗ {result['function']}: {result['description']}")
            logger.info(f"  Error: {result['error']}")

    # Save detailed results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("logs") / f"test_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(
            {
                "test_timestamp": datetime.now().isoformat(),
                "total_tests": len(results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(results) * 100,
                "results": results,
            },
            f,
            indent=2,
        )

    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info(f"Test completed at: {datetime.now()}")

    return results


if __name__ == "__main__":
    main()
