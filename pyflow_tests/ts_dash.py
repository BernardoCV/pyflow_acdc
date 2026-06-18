import pyflow_acdc as pyf
import pandas as pd
from pyflow_acdc.Graph_Dash import create_dash_app
from pyflow_tests.test_constants import NS_MTDC_MARKET_PRICES_URL, NS_MTDC_WIND_LOAD_URL

def ts_dash():
    [grid,results] = pyf.cases['NS_MTDC']()

    start = 5990
    end = 6000
    obj = {'Energy_cost': 1}

    TS_MK = pd.read_csv(NS_MTDC_MARKET_PRICES_URL)
    pyf.add_TimeSeries(grid,TS_MK)

    TS_wl = pd.read_csv(NS_MTDC_WIND_LOAD_URL)
    pyf.add_TimeSeries(grid,TS_wl)

    pyf.ts_acdc_opf(grid,start,end,ObjRule=obj)

    app = create_dash_app(grid)
    assert app.layout is not None
    assert app.callback_map

    print('Time series OPF completed and Dash app created')

def run_test():
    """Test time series dash functionality."""
    try:
        import dash
        
    except:
        print("dash is not installed...")
        return
    
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    ts_dash()

if __name__ == "__main__":
    run_test()