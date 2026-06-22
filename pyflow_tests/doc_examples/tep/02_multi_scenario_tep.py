"""Docs: usage_tep.rst — Multi-scenario TEP"""
import pandas as pd
import pyflow_acdc as pyf

if not pyf.is_pyomo_solver_available("ipopt"):
    print("Skipped: Ipopt solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["NS_MTDC_2025"](years_data="24", expandable=True, online=True)

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
    ObjRule={"Price_Zones": 1},
    clustering_options=clustering_options,
    solver="ipopt",
    tee=False,
)
res.all()
