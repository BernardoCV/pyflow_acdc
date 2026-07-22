"""Docs: usage_tep.rst, api/tep.rst — Reconductoring (REC) transmission expansion"""
import pyflow_acdc as pyf

build_only = not pyf.is_pyomo_solver_available("bonmin")

grid, res = pyf.cases["case24_REC"]()
obj = {'Energy_cost': 1}
model, results, timing, stats = pyf.transmission_expansion(
    grid,
    solver="bonmin",
    ObjRule=obj,
    NPV=True,
    build_only=build_only,
)

res.tep_n()
res.tep_norm()
model.obj.display()
