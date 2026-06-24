"""Docs: usage_tep.rst, api/tep.rst — Running one state transmission expansion planning"""
import pyflow_acdc as pyf

if not pyf.is_pyomo_solver_available("ipopt"):
    print("Skipped: Ipopt solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["case118_TEP"]()
obj = {'Energy_cost': 1}
model, results, timing, stats = pyf.transmission_expansion(grid, solver="ipopt",ObjRule=obj)

res.all()