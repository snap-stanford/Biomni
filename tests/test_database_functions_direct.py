#!/usr/bin/env python3
"""
Direct test script for database functions.
Tests each function directly without using the agent for more reliable results.
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import database functions
from biomni.tool.database import (
    query_chembl, query_clinicaltrials, query_dailymed, 
    query_openfda, query_cellxgene_census, query_encode,
    query_unichem, query_quickgo
)



# Set up logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"database_tests_direct_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def test_function_directly(function, function_name, test_params, description):
    """Test a function directly with given parameters."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing: {function_name}")
    logger.info(f"Parameters: {test_params}")
    logger.info(f"Description: {description}")
    logger.info(f"{'='*60}")
    
    try:
        # Call the function directly
        logger.info("Executing function...")
        result = function(**test_params)
        
        # Log the result
        logger.info("Function executed successfully!")
        logger.info(f"Result type: {type(result)}")
        
        if isinstance(result, dict):
            success = result.get('success', True)  # Default to True if not specified
            logger.info(f"Success: {success}")
            if not success:
                error = result.get('error', 'Unknown error')
                logger.warning(f"Function returned error: {error}")
            else:
                # Check if we have actual data (not just success flag)
                has_data = any(key in result for key in ['result', 'molecules', 'activities', 'data'])
                if has_data:
                    logger.info("Function returned data successfully")
                else:
                    logger.warning("Function succeeded but returned no data")
        
        if isinstance(result, str):
            logger.info(f"Result length: {len(result)} characters")
            # Format the result for better readability
            formatted_result = result.replace('\\n', '\n').replace('\\t', '\t')
            logger.info(f"Result preview:\n{formatted_result}...")
        else:
            logger.info(f"Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            # Log a sample of the result for better understanding
            if isinstance(result, dict) and 'result' in result:
                sample_result = result['result']
                if isinstance(sample_result, str):
                    formatted_sample = sample_result.replace('\\n', '\n').replace('\\t', '\t')
                    logger.info(f"Sample result data:\n{formatted_sample[:800]}...")
                elif isinstance(sample_result, (list, dict)):
                    logger.info(f"Sample result data: {str(sample_result)[:800]}...")
        
        return {
            "function": function_name,
            "parameters": test_params,
            "description": description,
            "status": "SUCCESS",
            "result_type": str(type(result)),
            "success": result.get('success', True) if isinstance(result, dict) else True,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error testing {function_name}: {str(e)}"
        logger.error(error_msg)
        
        return {
            "function": function_name,
            "parameters": test_params,
            "description": description,
            "status": "FAILED",
            "result_type": "N/A",
            "success": False,
            "error": str(e)
        }

def main():
    """Main test function."""
    logger = setup_logging()
    
    # Set up environment for API keys
    import os
    from dotenv import load_dotenv
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables")
    else:
        logger.info("OpenAI API key found in environment")
    
    logger.info("Starting direct database function tests")
    logger.info(f"Test started at: {datetime.now()}")
    
    # Define test cases for each database/function
    test_cases = [
        # ChEMBL Database tests
        {
            "function": query_chembl,
            "function_name": "query_chembl",
            "test_params": {"molecule_name": "aspirin", "verbose": False},
            "description": "Test ChEMBL database query for aspirin molecule"
        },
        
        # ClinicalTrials.gov tests
        {
            "function": query_clinicaltrials,
            "function_name": "query_clinicaltrials",
            "test_params": {"prompt": "cancer phase 3", "verbose": False},
            "description": "Test ClinicalTrials.gov query for phase 3 cancer trials"
        },
        
        # DailyMed tests

        {
            "function": query_dailymed,
            "function_name": "query_dailymed",
            "test_params": {"prompt": "metformin", "verbose": False},
            "description": "Test DailyMed query for metformin drug label"
        },
        
        # openFDA tests
        {
            "function": query_openfda,
            "function_name": "query_openfda",
            "test_params": {"prompt": "ibuprofen adverse events", "verbose": False},
            "description": "Test openFDA query for ibuprofen adverse events"
        },
        
        # CELLxGENE tests
        {
            "function": query_cellxgene_census,
            "function_name": "query_cellxgene_census",
            "test_params": {"prompt": "brain cell data in human", "verbose": False},
            "description": "Test CELLxGENE Census query for brain single-cell data"
        },
        
        # ENCODE tests
        {
            "function": query_encode,
            "function_name": "query_encode",
            "test_params": {"prompt": "chromatin accessibility mouse biosamples", "verbose": False},
            "description": "Test ENCODE query for chromatin accessibility data"
        },
        
        # UniChem tests
        {
            "function": query_unichem,
            "function_name": "query_unichem",
            "test_params": {"prompt": "aspirin cross-references", "verbose": False},
            "description": "Test UniChem query for chemical cross-references"
        },
        
        # QuickGO tests
        {
            "function": query_quickgo,
            "function_name": "query_quickgo",
            "test_params": {"prompt": "apoptosis GO terms", "verbose": False},
            "description": "Test QuickGO query for Gene Ontology terms"
        },
        

    ]
    
    # Run all tests
    results = []
    for i, test_case in enumerate(test_cases, 1):
        logger.info(f"\nTest {i}/{len(test_cases)}")
        
        result = test_function_directly(
            function=test_case["function"],
            function_name=test_case["function_name"],
            test_params=test_case["test_params"],
            description=test_case["description"]
        )
        
        results.append(result)
        
        # Add a small delay between tests to avoid overwhelming the APIs
        import time
        time.sleep(1)
    
    # Generate summary report
    logger.info(f"\n{'='*60}")
    logger.info("TEST SUMMARY REPORT")
    logger.info(f"{'='*60}")
    
    successful_tests = [r for r in results if r["status"] == "SUCCESS" and r["success"]]
    failed_tests = [r for r in results if r["status"] == "FAILED" or not r["success"]]
    
    logger.info(f"Total tests: {len(results)}")
    logger.info(f"Successful: {len(successful_tests)}")
    logger.info(f"Failed: {len(failed_tests)}")
    logger.info(f"Success rate: {len(successful_tests)/len(results)*100:.1f}%")
    
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
            if result['error']:
                logger.info(f"  Error: {result['error']}")
            elif not result['success']:
                logger.info(f"  Function returned success=False")
    
    # Save detailed results to JSON file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path("logs") / f"test_results_direct_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "test_timestamp": datetime.now().isoformat(),
            "total_tests": len(results),
            "successful_tests": len(successful_tests),
            "failed_tests": len(failed_tests),
            "success_rate": len(successful_tests)/len(results)*100,
            "results": results
        }, f, indent=2)
    
    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info(f"Test completed at: {datetime.now()}")
    
    return results

if __name__ == "__main__":
    main()
