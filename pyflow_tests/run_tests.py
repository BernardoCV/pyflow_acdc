import sys
from pathlib import Path
import importlib.util
from io import StringIO
import contextlib
from typing import Dict, List
import warnings
import re

# Configuration
TEST_DIR = Path(__file__).parent

# List of test cases to run (in order)
ALL_CASES = [
    # Documentation tests
    'test_docs_basic_grid_creation.py',
    'test_docs_add_components.py',
    'test_docs_power_flow.py',
    'test_docs_opf_quick.py',
    'test_docs_opf_detailed.py',
    
    #Power Flow
    'grid_creation.py',
    'CigreB4_PF.py',

    #OPF
    'DC_OPF.py',
    'CigreB4_OPF.py',
    'case39ac_OPF.py',
    'case39acdc_OPF.py',
    'case24_3zones_acdc_OPF.py',
    #loading matlab files
    'matlab_loader.py',
    #folium
    'folium_test.py',
    
    #Transmission Expansion
    #make sure OPF still works
    'case24_OPF.py',
    #DC
    'case6_TEP_DC.py',
    #AC
    'case24_TEP.py',
    #REC
    'case24_REC.py',
    #CT
    'array_sizing.py',
    #time series and dash
    'ts_dash.py'
]

# Quick tests (basic functionality only)
QUICK_CASES = [
    'test_docs_basic_grid_creation.py',
    'test_docs_add_components.py',
    'test_docs_power_flow.py',
    'grid_creation.py',
    'CigreB4_PF.py',
    'matlab_loader.py',
]

def run_test_case(case: str, show_output: bool = False) -> tuple[bool, str, List[str]]:
    """Run a test case and return (success, error_message, warnings)."""
    if show_output:
        print(f"\nRunning test case: {case}")
        print("-" * 70)
    
    # Load the module
    module_path = TEST_DIR / case
    spec = importlib.util.spec_from_file_location(case[:-3], module_path)
    if spec is None or spec.loader is None:
        error_msg = f"Error: Could not load module {case}"
        print(error_msg)
        return False, error_msg, []
        
    module = importlib.util.module_from_spec(spec)
    
    # Capture warnings
    captured_warnings = []
    
    try:
        if show_output:
            # Run the module directly to see all output
            spec.loader.exec_module(module)
        else:
            # Capture stdout to check for warning messages
            stdout_capture = StringIO()
            with contextlib.redirect_stdout(stdout_capture):
                spec.loader.exec_module(module)
            
            # Check stdout for explicit warning messages
            for line in stdout_capture.getvalue().split('\n'):
                if 'Warning' in line or 'warning' in line: # or 'WARNING' in line:
                    captured_warnings.append(line.strip())
            
        return True, "", captured_warnings
    except Exception as e:
        error_msg = f"Error running {case}: {str(e)}"
        print(error_msg)
        return False, error_msg, captured_warnings
    finally:
        if show_output:
            print("-" * 70)

def main():
    # Check command line arguments
    show_output = len(sys.argv) > 1 and sys.argv[1] == "--show-output"
    quick_mode = len(sys.argv) > 1 and sys.argv[1] == "--quick"
    
    # Choose which tests to run
    if quick_mode:
        CASES = QUICK_CASES
        print("Running quick tests (basic functionality only)")
    else:
        CASES = ALL_CASES
        print("Running all tests")
    
    print(f"Running {len(CASES)} test cases")
    if show_output:
        print("Showing full output for each test case")
    print("-" * 70)
    
    results: Dict[str, tuple[bool, str, List[str]]] = {}
    
    for case in CASES:
        success, error_msg, warnings = run_test_case(case, show_output)
        results[case] = (success, error_msg, warnings)
        if not show_output:
            status = "✓ Passed" if success else "✗ Failed"
            print(f"{status} - {case}")
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  {warning}")
    
    print("-" * 70)
    
    # Print summary
    success_count = sum(1 for result in results.values() if result[0])
    print(f"Summary: {success_count}/{len(CASES)} tests passed")
    
    # Print detailed error report if any tests failed
    failed_tests = [(case, error, warnings) for case, (success, error, warnings) in results.items() if not success]
    if failed_tests:
        print("\nFailed Tests:")
        for case, error, warnings in failed_tests:
            print(f"\n{case}:")
            print(f"  {error}")
            if warnings:
                print("\nWarnings:")
                for warning in warnings:
                    print(f"  {warning}")

if __name__ == "__main__":
    main()
