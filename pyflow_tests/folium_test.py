import pyflow_acdc as pyf

try:
    import folium

    grid,res = pyf.NS_MTDC()

    pyf.Optimal_PF(grid)

    pyf.plot_folium(grid)

    print('folium test completed')
except ImportError:
    print("folium is not installed. To use these functions, please install it using 'pip install folium'.")
    



