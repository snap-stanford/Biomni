#!/usr/bin/env python3
"""
Test script for the Integrated LIMS & Analysis Platform
Tests core functionality without requiring Streamlit
"""

import os
import sys
import importlib.util
import types
from pathlib import Path
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

streamlit_dir = os.path.join(current_dir, 'streamlit')

def ensure_streamlit_app_stub():
    """Provide a lightweight stub for streamlit_app import during tests."""
    if 'streamlit_app' not in sys.modules:
        stub = types.ModuleType('streamlit_app')

        def _not_implemented(*args, **kwargs):
            raise RuntimeError("streamlit_app stub invoked during tests")

        stub.run_omicshorizon_app = _not_implemented
        sys.modules['streamlit_app'] = stub

def load_main_app_module():
    """Dynamically load the relocated main_app module."""
    ensure_streamlit_app_stub()
    module_path = os.path.join(streamlit_dir, 'main_app.py')
    spec = importlib.util.spec_from_file_location("main_app", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_data_manager():
    """Test DataManager functionality"""
    print("🧪 Testing DataManager...")

    # Import DataManager from main_app
    try:
        main_app = load_main_app_module()
        dm = main_app.DataManager()
        files = dm.list_data_files()

        print(f"✅ Found {len(files)} files in data directory")

        if files:
            print("📁 Sample files:")
            for file_info in files[:3]:  # Show first 3
                size_mb = file_info['size'] / (1024 * 1024)
                print(f"  - {file_info['name']} ({size_mb:.1f} MB, {file_info['extension']})")

        # Test workspace copying
        if files:
            workspace_path = os.path.join(current_dir, 'test_workspace')
            os.makedirs(workspace_path, exist_ok=True)

            test_file = files[0]['path']
            copied = dm.copy_files_to_workspace([test_file], workspace_path)

            if copied:
                print(f"✅ Successfully copied file to workspace: {os.path.basename(copied[0])}")
                # Clean up
                os.remove(copied[0])
                os.rmdir(workspace_path)
            else:
                print("❌ File copying failed")

        return True

    except Exception as e:
        print(f"❌ DataManager test failed: {e}")
        return False

def test_app_registry():
    """Test analysis apps registry"""
    print("\n🧪 Testing App Registry...")

    try:
        main_app = load_main_app_module()
        analysis_apps = main_app.ANALYSIS_APPS

        print(f"✅ Found {len(analysis_apps)} analysis apps")

        for app_id, app_info in analysis_apps.items():
            status = "✅" if app_info['enabled'] else "❌"
            print(f"  {status} {app_info['name']} ({app_info['icon']})")
            print(f"    - Description: {app_info['description']}")
            print(f"    - Data types: {', '.join(app_info['data_types'])}")
            print(f"    - Category: {app_info['category']}")

        return True

    except Exception as e:
        print(f"❌ App Registry test failed: {e}")
        return False

def test_omics_horizon_import():
    """Test OmicsHorizon app import (without executing)"""
    print("\n🧪 Testing OmicsHorizon Import...")

    try:
        # Test basic import without streamlit
        import ast
        import inspect

        # Read the streamlit_app.py file
        streamlit_app_path = os.path.join('streamlit', 'streamlit_app.py')
        with open(streamlit_app_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if run_omicshorizon_app function exists
        if 'def run_omicshorizon_app(' in content:
            print("✅ run_omicshorizon_app function found")
        else:
            print("❌ run_omicshorizon_app function not found")
            return False

        # Check for required imports (without actually importing)
        required_imports = [
            'from biomni.agent import A1_HITS',
            'from biomni.config import default_config'
        ]

        for imp in required_imports:
            if imp in content:
                print(f"✅ Found import: {imp}")
            else:
                print(f"❌ Missing import: {imp}")

        return True

    except Exception as e:
        print(f"❌ OmicsHorizon import test failed: {e}")
        return False

def test_directory_structure():
    """Test required directory structure"""
    print("\n🧪 Testing Directory Structure...")

    required_dirs = ['data', 'workspace', 'biomni_data']
    required_files = [
        os.path.join('streamlit', 'main_app.py'),
        os.path.join('streamlit', 'streamlit_app.py'),
    ]

    all_good = True

    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print(f"✅ Directory exists: {dir_name}/")
        else:
            print(f"❌ Directory missing: {dir_name}/")
            all_good = False

    for file_name in required_files:
        if os.path.exists(file_name) and os.path.isfile(file_name):
            print(f"✅ File exists: {file_name}")
        else:
            print(f"❌ File missing: {file_name}")
            all_good = False

    return all_good

def run_full_test():
    """Run all tests"""
    print("🚀 Starting Integrated LIMS & Analysis Platform Test Suite")
    print("=" * 60)

    tests = [
        test_directory_structure,
        test_data_manager,
        test_app_registry,
        test_omics_horizon_import,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")

    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! The system is ready to run.")
        print("\nTo start the system:")
        print("  streamlit run streamlit/main_app.py")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = run_full_test()
