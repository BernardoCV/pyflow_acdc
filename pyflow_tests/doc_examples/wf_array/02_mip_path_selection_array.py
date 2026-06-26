"""Docs: api\\wf_array.rst — MIP Path Selection (Array)"""
import pyflow_acdc as pyf
from pyflow_acdc.constants import MIPBackend , PYOMO_LINEAR_SOLVERS

grid, res = pyf.cases["alpha_ventus"]()
solver = next(
    (name for name in PYOMO_LINEAR_SOLVERS if pyf.is_pyomo_solver_available(name)),
    None,
)
build_only = solver is None
flag, high_flow, model_MIP, feasible_solutions_MIP = pyf.MIP_path_graph(
    grid,
    max_flow=10,
    solver_name=solver,
    crossings=True,
    tee=False,
    callback=False,
    backend=MIPBackend.PYOMO.value,
    build_only=build_only,
)
