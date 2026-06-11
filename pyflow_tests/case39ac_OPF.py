import pyflow_acdc as pyf

def case39ac_OPF():

    grid,res = pyf.cases['case39']()
    obj = {'Energy_cost': 1}
    pyf.optimal_pf(grid,ObjRule=obj)

    res.all()


def run_test():
    """Test case39 AC optimal power flow."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    case39ac_OPF()
if __name__ == "__main__":
    run_test()

