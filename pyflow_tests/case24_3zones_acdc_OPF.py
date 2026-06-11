import pyflow_acdc as pyf

def case24_3zones_acdc_OPF():

    grid,res = pyf.cases['case24_3zones_acdc']()

    pyf.optimal_pf(grid)

    res.all()


def run_test():
    """Test case24 3-zones AC/DC optimal power flow."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    case24_3zones_acdc_OPF()

if __name__ == "__main__":
    run_test()    