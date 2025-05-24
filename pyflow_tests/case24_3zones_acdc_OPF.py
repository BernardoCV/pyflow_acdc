import pyflow_acdc as pyf

grid,res = pyf.case24_3zones_acdc()

pyf.Optimal_PF(grid)

res.All()


