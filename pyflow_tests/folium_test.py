import pyflow_acdc as pyf

def folium_test():

        grid,res = pyf.NS_MTDC()

        pyf.Optimal_PF(grid)

        pyf.plot_folium(grid)

        print('folium test completed')

def run_folium_test():
    try:
        import folium
    except ImportError:
        print("folium is not installed. To use these functions, please install it using 'pip install folium'.")
        return
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    folium_test()
if __name__ == "__main__":
    run_folium_test()
    