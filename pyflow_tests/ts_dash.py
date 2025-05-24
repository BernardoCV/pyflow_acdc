import pyflow_acdc as pyf
import pandas as pd
[grid,results] = pyf.NS_MTDC()

start = 5990
end = 6000
obj = {'Energy_cost': 1}

market_prices_url = "https://raw.githubusercontent.com/BernardoCV/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_marketPrices_data_sd2024.csv"
TS_MK = pd.read_csv(market_prices_url)
pyf.add_TimeSeries(grid,TS_MK)

wind_load_url = "https://raw.githubusercontent.com/BernardoCV/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_WL_data2024.csv"
TS_wl = pd.read_csv(wind_load_url)
pyf.add_TimeSeries(grid,TS_wl)

times=pyf.TS_ACDC_OPF(grid,start,end,ObjRule=obj)  

print('Time series OPF completed')

try:
    import dash
    pyf.run_dash(grid)
except:
    print("Error: Dash is not installed. Please install it using 'pip install dash'.")
    