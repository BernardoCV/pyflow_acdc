# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:53:11 2024

@author: BernardoCastro
"""

import pyflow_acdc as pyf
import time
import sys
import pyomo.environ as pyo
from pathlib import Path

current_file = Path(__file__).resolve()
path = str(current_file.parent)

data = f'{path}/case39_var.mat'

[grid,res]=pyf.Create_grid_from_mat(data)

pyf.save_grid_to_file(grid, 'case39',folder_name='example_grids')


obj = {'Energy_cost'  : 1}
nac=grid.nn_AC

print(nac)

JustOne = True
    
model, timing_info, model_res,solver_stats = pyf.Optimal_PF(grid,ObjRule=obj)

res.All()
    