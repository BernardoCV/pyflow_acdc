"""Usage guide: Quick OPF example (``docs/usage.rst``)."""
import pyflow_acdc as pyf

obj = {'Energy_cost': 1}

[grid, res] = pyf.cases['case39_acdc']()

model, timing_info, model_res, solver_stats = pyf.optimal_pf(grid, ObjRule=obj)

res.all()
print('------')
