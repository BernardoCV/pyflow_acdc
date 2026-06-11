import pyflow_acdc as pyf

def run_test():

    grid,res = pyf.cases['case39']()
    obj = {'Energy_cost': 1}
    model, model_res , timing_info, solver_stats= pyf.optimal_l_pf(grid,ObjRule=obj,solver='gurobi')


    res.all()
    model.obj.display()
    model.obj.pprint()

if __name__ == "__main__":
    run_test()