# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 14:05:19 2024

@author: BernardoCastro
"""

import time
import pandas as pd
import pyflow_acdc as pyf

start_time = time.time()
S_base=100 #MVA


AC_node_data   = pd.read_csv('Stagg5MATACDC/MATACDC_AC_node_data.csv')
DC_node_data   = pd.read_csv('Stagg5MATACDC/MATACDC_DC_node_data.csv')
AC_line_data   = pd.read_csv('Stagg5MATACDC/MATACDC_AC_line_data.csv')
DC_line_data   = pd.read_csv('Stagg5MATACDC/MATACDC_DC_line_data.csv')
Converter_data = pd.read_csv('Stagg5MATACDC/MATACDC_Converter_data.csv')



[grid,res]=pyf.Create_grid_from_data(S_base, AC_node_data, AC_line_data,DC_node_data, DC_line_data, Converter_data)

"""
Sequential algorithm 

"""

pyf.ACDC_sequential(grid)

end_time = time.time()
elapsed_time = end_time - start_time
res.All()
print ('------')
print(f'Time elapsed : {elapsed_time}')