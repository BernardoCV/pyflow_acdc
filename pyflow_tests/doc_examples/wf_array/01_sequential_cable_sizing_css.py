"""Docs: api\\wf_array.rst — Sequential Cable Sizing (CSS)"""
import pyflow_acdc as pyf
from pyflow_acdc.constants import PYOMO_LINEAR_SOLVERS

grid, res = pyf.cases["alpha_ventus"]()
print(grid.lines_AC_ct[0].cable_types)

solver = next(
    (name for name in PYOMO_LINEAR_SOLVERS if pyf.is_pyomo_solver_available(name)),
    None,
)
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
