import time
import pandas as pd
import pyflow_acdc as pyf

start_time = time.time()
S_base=100 #MVA


[grid,res]=pyf.PEI_grid()

pyf.plot_folium(grid)
"""
Sequential algorithm 

"""

pyf.ACDC_sequential(grid,QLimit=False)




end_time = time.time()
elapsed_time = end_time - start_time
res.All()
print ('------')
print(f'Time elapsed : {elapsed_time}')
