"""Docs: api\\tep.rst — Running multiple scenario based transmission expansion planning"""
import pandas as pd
import pyflow_acdc as pyf
from pyflow_tests.test_constants import NS_MTDC_MARKET_PRICES_URL, NS_MTDC_WIND_LOAD_URL

if not pyf.is_pyomo_solver_available("bonmin"):
    print("Skipped: Bonmin solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["NS_MTDC"]()
TS_MK = pd.read_csv(NS_MTDC_MARKET_PRICES_URL)
pyf.add_TimeSeries(grid, TS_MK)
TS_wl = pd.read_csv(NS_MTDC_WIND_LOAD_URL)
pyf.add_TimeSeries(grid, TS_wl)
n_cluster = 6
clustering_options = {
            'n_clusters': n_cluster,
            'time_series': ['price','Load','WPP'],
            'central_market': [],
            'thresholds': [0,0.8],
            'print_details': True,
            'correlation_decisions': [True,3,True],
            'cluster_algorithm': 'kmedoids'
        }

model, results, timing, stats, ts_results = pyf.multi_scenario_TEP(grid,clustering_options=clustering_options)
