"""Docs: usage_tep.rst, api/tep.rst — Running one state transmission expansion planning"""
import pyflow_acdc as pyf

build_only = not pyf.is_pyomo_solver_available("ipopt")

grid, res = pyf.cases["case118_TEP"]()
obj = {'Energy_cost': 1}
model, results, timing, stats = pyf.transmission_expansion(
    grid,
    solver="ipopt",
    ObjRule=obj,
    build_only=build_only,
)

res.all()
