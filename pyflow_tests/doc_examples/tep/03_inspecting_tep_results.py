"""Docs: usage_tep.rst, api/tep.rst — Inspecting static TEP results"""
import pyflow_acdc as pyf

build_only = not pyf.is_pyomo_solver_available("ipopt")

grid, res = pyf.cases["case24_TEP"]()
obj = {'Energy_cost': 1}
model, results, timing, stats = pyf.transmission_expansion(
    grid,
    solver="ipopt",
    ObjRule=obj,
    NPV=True,
    build_only=build_only,
)

res.tep_n()
res.tep_norm()
model.obj.display()
