"""Docs: api\\wf_array.rst — MIP Path Selection (Array)"""
import pyflow_acdc as pyf

try:
    import gurobipy  # noqa: F401
except ImportError as exc:
    print(f"Skipped: {exc}")
    raise SystemExit(0)

grid, res = pyf.cases["alpha_ventus"]()
flag, high_flow, model_MIP, feasible_solutions_MIP = pyf.MIP_path_graph(
    grid,
    max_flow=10,
    solver_name="gurobi",
    crossings=True,
    tee=False,
    callback=False,
)
