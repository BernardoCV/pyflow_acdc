# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:55:43 2023

@author: BernardoCastro

This grid is based on the CIGRE B4 test system. DCDC converters have been simplified to a load and a gain in respective nodes.
"""

import time
import pandas as pd
import pyflow_acdc as pyf
import os

start_time = time.time()

S_base=100 #MVAres
 
beta= 0.0165 #percent

path = os.path.dirname(os.path.abspath(__file__)).replace('\\', '/')

# Using forward slashes in paths
AC_node_data   = pd.read_csv(f"{path}/CigreB4/CigreB4_AC_node_data.csv")
DC_node_data   = pd.read_csv(f"{path}/CigreB4/CigreB4_DC_node_data.csv")
AC_line_data   = pd.read_csv(f"{path}/CigreB4/CigreB4_AC_line_data.csv")
DC_line_data   = pd.read_csv(f"{path}/CigreB4/CigreB4_DC_line_data.csv")
Converter_ACDC_data = pd.read_csv(f"{path}/CigreB4/CigreB4_Converter_data.csv")

[grid,res]=pyf.Create_grid_from_data(S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_ACDC_data)
for conv in grid.Converters_ACDC:
    conv.a_conv=0
    conv.b_conv=0
    conv.c_inver=0
    conv.c_rect=0


pyf.add_extGrid(grid, 'BaA0')
pyf.add_extGrid(grid, 'BaB0')



[model, model_res , timing_info]=pyf.OPF_ACDC(grid)

[opt_res_P_conv_DC, opt_res_P_conv_AC, opt_res_Q_conv_AC, opt_P_load,opt_res_P_extGrid, opt_res_Q_extGrid, opt_res_curtailment,opt_res_Loading_conv] =pyf.OPF_conv_results(model,grid)


end_time = time.time()
elapsed_time = end_time - start_time

print ('------')
print(f'Time elapsed : {elapsed_time}')
