"""Docs: api\plotting.rst — Time series results"""
import pyflow_acdc as pyf
import pandas as pd
from pyflow_tests.test_constants import NS_MTDC_MARKET_PRICES_URL, NS_MTDC_WIND_LOAD_URL
[grid,results] = pyf.cases['NS_MTDC']()
start = 5750
end = 6000
obj = {'Energy_cost': 1}
TS_MK = pd.read_csv(NS_MTDC_MARKET_PRICES_URL)
pyf.add_TimeSeries(grid,TS_MK)
TS_wl = pd.read_csv(NS_MTDC_WIND_LOAD_URL)
pyf.add_TimeSeries(grid,TS_wl)
times=pyf.ts_acdc_opf(grid,start,end,ObjRule=obj)  
pyf.plot_TS_res(grid, start, end, save_format='svg', show=False)
