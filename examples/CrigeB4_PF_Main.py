# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 10:55:43 2023

@author: BernardoCastro
"""

import time
import pandas as pd
import pyflow_acdc as pyf


start_time = time.time()

S_base=100 #MVAres
 


AC_node_data   = pd.read_csv('CigreB4/CigreB4_AC_node_data.csv')
DC_node_data   = pd.read_csv('CigreB4/CigreB4_DC_node_data.csv')
AC_line_data   = pd.read_csv('CigreB4/CigreB4_AC_line_data.csv')
DC_line_data   = pd.read_csv('CigreB4/CigreB4_DC_line_data.csv')
Converter_ACDC_data = pd.read_csv('CigreB4/CigreB4_Converter_data.csv')
Converter_DCDC_data = pd.read_csv('CigreB4/CigreB4_DCDC_conv.csv')

[grid,res]=pyf.Create_grid_from_data(S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_ACDC_data,Converter_DCDC_data,data_in_pu=False)

for conv in grid.Converters_ACDC:
    conv.a_conv=0
    conv.b_conv=0
    conv.c_inver=0
    conv.c_rect=0

"""
Sequential algorithm 

"""


pyf.ACDC_sequential(grid,maxIter=20,tol_lim=1e-5)
end_time = time.time()
elapsed_time = end_time - start_time
res.All()
print ('------')
print(f'Time elapsed : {elapsed_time}')