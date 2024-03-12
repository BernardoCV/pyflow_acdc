# -*- coding: utf-8 -*-

"""
Created on Wed Nov 15 17:15:05 2023

@author: BernardoCastro
"""

import time
import pandas as pd
import pyflow_acdc as pyf


start_time = time.time()


S_base=100 #MVA

AC_node_data   = pd.read_csv('Time_series/TS_AC_node_data.csv')
DC_node_data   = pd.read_csv('Time_series/TS_DC_node_data.csv')
AC_line_data   = pd.read_csv('Time_series/TS_AC_line_data.csv')
DC_line_data   = pd.read_csv('Time_series/TS_DC_line_data.csv')
Converter_data = pd.read_csv('Time_series/TS_Converter_data.csv')

weights_def = {
    'Ext_Gen'         : {'w': 1},
    'Market_sale'     : {'w': 0},
    'AC_losses'       : {'w': 1},
    'DC_losses'       : {'w': 1},
    'Converter_Losses': {'w': 1}
}

[grid,res]=pyf.Create_grid_from_data(S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_data)

Time_Series_data= pd.read_csv('Time_series/TS_data.csv')


pyf.Add_TimeSeries(grid,Time_Series_data)


start=4500
end=4800


"uncomment to curtail RE sources"
# curtail=.99
# grid.Curtail_RE(curtail)


pyf.TS_ACDC_PF(grid,start,end,OPF=True,OPF_w=weights_def,VarPrice=True)

end_time = time.time()



elapsed_time = end_time - start_time

res.Time_series_plots()

res.Time_series_prob('Ldc8')
res.Time_series_prob('Ldc1')
res.Time_series_prob('VSC6')
res.Time_series_prob('OFAC1')
res.Time_series_prob('G2')

# 
print ('------')
print ('-ExtGrid+ losses + market-')
print(f'Time elapsed : {elapsed_time}')


