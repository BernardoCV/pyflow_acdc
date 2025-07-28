import pyflow_acdc as pyf

def case39acdc_OPF():

    grid,res = pyf.case39_acdc()

    pyf.Optimal_PF(grid)

    res.All()

def run_test():
    """Test case39 AC/DC optimal power flow."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    case39acdc_OPF()

if __name__ == "__main__":
    run_test()
