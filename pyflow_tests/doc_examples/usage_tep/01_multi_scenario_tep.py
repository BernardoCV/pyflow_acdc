"""Docs: usage_tep.rst — Multi-scenario TEP"""
import pandas as pd
import pyflow_acdc as pyf
from pyflow_tests.test_constants import NS_MTDC_MARKET_PRICES_URL, NS_MTDC_WIND_LOAD_URL

if not pyf.is_pyomo_solver_available("ipopt"):
    print("Skipped: Ipopt solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["NS_MTDC"]()
pyf.add_TimeSeries(grid, pd.read_csv(NS_MTDC_MARKET_PRICES_URL))
pyf.add_TimeSeries(grid, pd.read_csv(NS_MTDC_WIND_LOAD_URL))

clustering_options = {
    "n_clusters": 6,
    "time_series": ["price", "Load", "WPP"],
    "central_market": [],
    "thresholds": [0, 0.8],
    "print_details": False,
    "correlation_decisions": [True, 3, True],
    "cluster_algorithm": "kmeans_medoids",
}

model, model_results, timing_info, solver_stats, ts_results = pyf.multi_scenario_TEP(
    grid,
    ObjRule={"Energy_cost": 1},
    clustering_options=clustering_options,
    solver="ipopt",
    tee=False,
)
