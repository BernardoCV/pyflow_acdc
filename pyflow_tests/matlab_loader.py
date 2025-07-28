# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:53:11 2024

@author: BernardoCastro
"""

import pyflow_acdc as pyf
from pathlib import Path

def matlab_loader():

    current_file = Path(__file__).resolve()
    path = str(current_file.parent)

    data = f'{path}/case39_var.mat'

    [grid,res]=pyf.Create_grid_from_mat(data)

    pyf.save_grid_to_file(grid, 'case39',folder_name='example_grids')


    obj = {'Energy_cost'  : 1}
    nac=grid.nn_AC

    print(nac)

    JustOne = True
        
    model, model_res,timing_info, solver_stats = pyf.Optimal_PF(grid,ObjRule=obj)

    res.All()

    print(timing_info)
    print(model_res)
    model.obj.display()

def run_test():
    """Test MATLAB file loading functionality."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return

    matlab_loader()

if __name__ == "__main__":
    run_test()