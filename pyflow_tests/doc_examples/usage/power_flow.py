"""Usage guide: Running a Power Flow (``docs/usage.rst``)."""
import pyflow_acdc as pyf

[grid, res] = pyf.cases['PEI_grid']()

time, tol, ps_iterations = pyf.acdc_sequential(grid, QLimit=False)

res.all()
print('------')
