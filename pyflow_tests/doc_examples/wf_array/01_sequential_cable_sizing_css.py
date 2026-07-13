"""Docs: api\\wf_array.rst — Sequential Cable Sizing (CSS)"""
import pyflow_acdc as pyf
from pyflow_acdc.solver_utils import resolve_pyomo_linear_solver

grid, res = pyf.cases["alpha_ventus"]()
print(grid.lines_AC_ct[0].cable_types)

solver = resolve_pyomo_linear_solver()
if solver is None:
    print("Skipped: no Pyomo MIP/CSS-L solver available")
else:
    pyf.sequential_CSS(
        grid,
        NPV=True,
        max_turbines_per_string=None,
        MIP_solver=solver,
        CSS_L_solver=solver,
        max_iter=None,
        time_limit=300,
        tee=False,
    )
    res.all()
