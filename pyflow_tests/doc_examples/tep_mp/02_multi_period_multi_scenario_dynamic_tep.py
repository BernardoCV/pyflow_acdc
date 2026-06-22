"""Docs: usage_mp_tep.rst, api/tep_dynamic.rst — Multi-period Multi-scenario Dynamic TEP"""
import pyflow_acdc as pyf

if not pyf.is_pyomo_solver_available("ipopt"):
    print("Skipped: Ipopt solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["NS_MTDC_2025"](years_data="24", expandable=True, online=True)
mp_load_series = list(grid.Price_Zones[0].investment_decisions["Load"])
clustering_options = {
    "n_clusters": 6,
    "time_series": ["price", "Load", "WPP"],
    "central_market": [],
    "thresholds": [0, 0.8],
    "print_details": False,
    "correlation_decisions": [True, 3, True],
    "cluster_algorithm": "kmedoids",
}

model, model_results, timing_info, solver_stats, ts_results = pyf.multi_period_MS_TEP(
    grid,
    inv_periods=mp_load_series,
    ObjRule={"Energy_cost": 0, "PZ_cost_of_generation": 1, "Renewable_profit": 0},
    clustering_options=clustering_options,
    solver="ipopt",
    tee=False,
)
res.all()
