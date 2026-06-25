"""Usage guide: Quick OPF example (``docs/usage.rst``)."""
import pyflow_acdc as pyf

build_only = not pyf.is_pyomo_solver_available("ipopt")

obj = {'Energy_cost': 1}

[grid, res] = pyf.cases['case39_acdc']()

model, timing_info, model_res, solver_stats = pyf.optimal_pf(
    grid,
    ObjRule=obj,
    build_only=build_only,
)

res.all()
print('------')
