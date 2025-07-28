# -*- coding: utf-8 -*-
"""
Test for quick OPF example from documentation.
Tests the "Quick Example" section in OPF part of usage.rst
"""

import pyflow_acdc as pyf

def test_docs_opf_quick():
    """Test the quick OPF example from documentation."""

    obj = {'Energy_cost': 1}

    [grid, res] = pyf.case39_acdc()

    model, timing_info, [model_res, solver_stats] = pyf.OPF_ACDC(grid, ObjRule={'obj': {'w': 1}})

    res.All()
    print('------')
    print("✓ Quick OPF example test passed")


def run_test():
    """Test quick OPF example from documentation."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    test_docs_opf_quick()

if __name__ == "__main__":
    run_test()