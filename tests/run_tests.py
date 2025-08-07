#!/usr/bin/env python3
"""
Test runner script for Biomni tools.

This script runs comprehensive tests for all implemented tools and provides
detailed reporting on test results.
"""
import sys
import os
import subprocess
import argparse
from pathlib import Path


def setup_environment():
    """Set up the test environment."""
    # Add biomni to Python path
    biomni_root = Path(__file__).parent.parent
    sys.path.insert(0, str(biomni_root))
    
    # Set test environment variables
    os.environ.setdefault('ANTHROPIC_API_KEY', 'test-key-12345')
    os.environ.setdefault('PYTEST_CURRENT_TEST', 'true')


def run_tests(test_type="all", verbose=False, integration=False):
    """Run the specified tests."""
    test_dir = Path(__file__).parent
    
    # Base pytest command
    cmd = ["python", "-m", "pytest", str(test_dir)]
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Add test type filters
    if test_type == "unit":
        cmd.extend(["-m", "unit"])
    elif test_type == "integration":
        cmd.extend(["-m", "integration"])
    elif test_type == "drug":
        cmd.append("test_drug_tools.py")
    elif test_type == "knowledge":
        cmd.append("test_knowledge_tools.py")
    elif test_type == "utils":
        cmd.append("test_utils.py")
    
    # Skip integration tests unless explicitly requested
    if not integration and test_type == "all":
        cmd.extend(["-m", "not integration"])
    
    # Add coverage if available
    try:
        import coverage
        cmd.extend(["--cov=biomni.tool.database", "--cov-report=term-missing"])
    except ImportError:
        pass
    
    # Add output formatting
    cmd.extend(["--tb=short", "--strict-markers"])
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    result = subprocess.run(cmd, cwd=test_dir.parent)
    return result.returncode


def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = [
        "pytest", "requests", "anthropic"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\nInstall with: pip install -r tests/requirements.txt")
        return False
    
    return True


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run Biomni tool tests")
    parser.add_argument(
        "--type", 
        choices=["all", "unit", "integration", "drug", "knowledge", "utils"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--integration", "-i",
        action="store_true",
        help="Include integration tests (requires network)"
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if args.check_deps or not check_dependencies():
        return 1 if not check_dependencies() else 0
    
    # Setup environment
    setup_environment()
    
    # Run tests
    print(f"Running {args.type} tests...")
    if args.integration:
        print("Including integration tests (requires network access)")
    
    return_code = run_tests(
        test_type=args.type,
        verbose=args.verbose,
        integration=args.integration
    )
    
    if return_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with return code {return_code}")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main())
