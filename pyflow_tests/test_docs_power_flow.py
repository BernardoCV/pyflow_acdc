# -*- coding: utf-8 -*-
"""
Test for power flow examples from documentation.
Tests the "Running a Power Flow" section in usage.rst
"""

import pyflow_acdc as pyf

def run_test():
    """Test power flow example from documentation."""
    # Test the PEI grid example from docs
    [grid, res] = pyf.PEI_grid()

    time,tol,ps_iterations = pyf.ACDC_sequential(grid, QLimit=False)

    res.All()
    print('------')
    print("✓ Power flow example test passed")

if __name__ == "__main__":
    run_test()


