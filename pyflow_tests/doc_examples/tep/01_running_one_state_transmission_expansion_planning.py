"""Docs: api\\tep.rst — Running one state transmission expansion planning"""
import pyflow_acdc as pyf

if not pyf.is_pyomo_solver_available("bonmin"):
    print("Skipped: Bonmin solver not available")
    raise SystemExit(0)

grid, res = pyf.cases["case118_TEP"]()
model, results, timing, stats = pyf.transmission_expansion(grid)
