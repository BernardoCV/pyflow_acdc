"""Docs: api/L_models.rst — Linear (MILP) transmission expansion planning"""
import pyflow_acdc as pyf
from pyflow_acdc.constants import PYOMO_LINEAR_SOLVERS

solver = next((s for s in PYOMO_LINEAR_SOLVERS if pyf.is_pyomo_solver_available(s)), None)
build_only = solver is None

grid, res = pyf.cases["case118_TEP"]()
obj = {'Energy_cost': 1}
model, results, timing, stats = pyf.linear_transmission_expansion(
    grid,
    ObjRule=obj,
    solver=solver or "highs",
    build_only=build_only,
)

res.all()
