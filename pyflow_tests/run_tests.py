import sys
from pathlib import Path
import importlib.util
from io import StringIO
import contextlib
from typing import Dict, List

# Configuration
TEST_DIR = Path(__file__).parent

# List of test cases to run (in order)
CASES = [
    #Power Flow
    'grid_creation.py',
    'CigreB4_PF.py',

    #OPF
    'CigreB4_OPF.py',
    'case39ac_OPF.py',
    'case39acdc_OPF.py',
    'case24_3zones_acdc_OPF.py',
    #loading matlab files
    'matlab_loader.py',
    #folium
    'folium_test.py',
    
    #Transmission Expansion
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

def run_test_case(case: str, show_output: bool = False) -> tuple[bool, str]:
    """Run a test case and return (success, error_message)."""
    if show_output:
        print(f"\nRunning test case: {case}")
        print("-" * 70)
    
    # Load the module
    module_path = TEST_DIR / case
    spec = importlib.util.spec_from_file_location(case[:-3], module_path)
    if spec is None or spec.loader is None:
        error_msg = f"Error: Could not load module {case}"
        print(error_msg)
        return False, error_msg
        
    module = importlib.util.module_from_spec(spec)
    
    try:
        if show_output:
            # Run the module directly to see all output
            spec.loader.exec_module(module)
        else:
            # Capture output but don't show it
            with contextlib.redirect_stdout(StringIO()):
                spec.loader.exec_module(module)
        return True, ""
    except Exception as e:
        error_msg = f"Error running {case}: {str(e)}"
        print(error_msg)
        return False, error_msg
    finally:
        if show_output:
            print("-" * 70)

def main():
    # Check command line arguments
    show_output = len(sys.argv) > 1 and sys.argv[1] == "--show-output"
    
    print(f"Running {len(CASES)} test cases")
    if show_output:
        print("Showing full output for each test case")
    print("-" * 70)
    
    results: Dict[str, tuple[bool, str]] = {}
    
    for case in CASES:
        success, error_msg = run_test_case(case, show_output)
        results[case] = (success, error_msg)
        if not show_output:
            status = "✓ Passed" if success else "✗ Failed"
            print(f"{status} - {case}")
    
    print("-" * 70)
    
    # Print summary
    success_count = sum(1 for success, _ in results.values() if success)
    print(f"Summary: {success_count}/{len(CASES)} tests passed")
    
    # Print detailed error report if any tests failed
    failed_tests = [(case, error) for case, (success, error) in results.items() if not success]
    if failed_tests:
        print("\nFailed Tests:")
        for case, error in failed_tests:
            print(f"\n{case}:")
            print(f"  {error}")

if __name__ == "__main__":
    main()
