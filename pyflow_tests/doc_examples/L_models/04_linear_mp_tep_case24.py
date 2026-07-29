"""Docs: api/L_models.rst — Linear multi-period TEP on case24_MP"""
import pyflow_acdc as pyf
from pyflow_tests.test_constants import (
    CASE24_MP_GEN_MIX_LIMITS_URL,
    CASE24_MP_INV_SERIES_URL,
)

build_only = True

grid, res = pyf.cases["case24_MP"]()
pyf.add_inv_series(grid, CASE24_MP_INV_SERIES_URL)
pyf.add_gen_mix_limits(grid, CASE24_MP_GEN_MIX_LIMITS_URL)

model, model_results, timing_info, solver_stats = pyf.linear_multi_period_transmission_expansion(
    grid,
    n_years=10,
    Hy=8760,
    discount_rate=0.02,
    ObjRule={"Energy_cost": 1},
    solver="gurobi",
    tee=True,
    obj_scaling=1e9,
    build_only=build_only,
)
res.all()
