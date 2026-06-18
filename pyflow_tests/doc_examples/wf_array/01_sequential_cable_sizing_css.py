"""Docs: api\\wf_array.rst — Sequential Cable Sizing (CSS)"""
import pyflow_acdc as pyf
import sys
try:
    import gurobipy  # noqa: F401
except ImportError as exc:
    print(f"Skipped: {exc}")
    raise SystemExit(0)

grid, res = pyf.cases["alpha_ventus"]()
print(grid.lines_AC_ct[0].cable_types)

pyf.sequential_CSS(
    grid,
    NPV=True,
    max_turbines_per_string=None,
    MIP_solver="gurobi",
    CSS_L_solver="gurobi",
    max_iter=None,
    time_limit=300,
    tee=False,
)
res.all()
