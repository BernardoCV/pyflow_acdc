"""Docs: usage_mp_tep.rst — Multi-period TEP on case24_MP"""
import pyflow_acdc as pyf
from pyflow_tests.test_constants import (
    CASE24_MP_GEN_MIX_LIMITS_URL,
    CASE24_MP_INV_SERIES_URL,
)

if not pyf.is_pyomo_solver_available("ipopt"):
    print("Skipped: Ipopt solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["case24_MP"]()
pyf.add_inv_series(grid, CASE24_MP_INV_SERIES_URL)
pyf.add_gen_mix_limits(grid, CASE24_MP_GEN_MIX_LIMITS_URL)

model, model_results, timing_info, solver_stats = pyf.multi_period_transmission_expansion(
    grid,
    n_years=10,
    Hy=8760,
    discount_rate=0.02,
    ObjRule={"Energy_cost": 1},
    solver="ipopt",
    tee=False,
    obj_scaling=1e9,
)
