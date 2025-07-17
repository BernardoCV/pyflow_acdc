import pyflow_acdc as pyf

grid,res = pyf.case39()
obj = {'Energy_cost': 1}
model, model_res , timing_info, solver_stats= pyf.Optimal_L_PF(grid,ObjRule=obj,solver='gurobi')


res.All()
model.obj.display()
model.obj.pprint()

