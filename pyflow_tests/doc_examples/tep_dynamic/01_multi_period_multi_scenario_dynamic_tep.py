"""Docs: api\\tep_dynamic.rst — Multi-period Multi-scenario Dynamic TEP"""
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
    "n_clusters": n_cluster,
    "time_series": ["price", "Load", "WPP"],
    "central_market": [],
    "thresholds": [0, 0.8],
    "print_details": True,
    "correlation_decisions": [True, 3, True],
    "cluster_algorithm": "kmedoids",
}
obj = {"Energy_cost": 1}
model, model_results, timing_info, solver_stats, ts_results = pyf.multi_period_MS_TEP(
    grid,
    ObjRule=obj,
    clustering_options=clustering_options,
    solver="bonmin",
)
