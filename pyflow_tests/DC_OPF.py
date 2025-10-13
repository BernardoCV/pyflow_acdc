import pyflow_acdc as pyf

def DC_OPF():


    pyf.initialize_pyflowacdc()

    S_base = 100
    obj = {'Energy_cost': 1}

    DC_node_1 = pyf.Node_DC(node_type='Slack', Voltage_0=1,kV_base=200,name='Node_1')
    DC_node_2 = pyf.Node_DC(node_type='P', Voltage_0=1,kV_base=200,name='Node_2',Power_load=4)
    DC_node_3 = pyf.Node_DC(node_type='P', Voltage_0=1,kV_base=200,name='Node_3',Power_load=2)

    DC_nodes = [DC_node_1, DC_node_2, DC_node_3]

    grid = pyf.Grid(S_base,nodes_DC=DC_nodes)
    res= pyf.Results(grid,decimals=5)

    r= 0.010575 #Ohm/dist
    d12 = 5
    d13 = 15
    d23 = 60


    pyf.add_line_DC(grid,DC_node_1, DC_node_2,R_Ohm_km=r,Length_km=d12,MW_rating=600,polarity='m',update_grid=False)
    pyf.add_line_DC(grid,DC_node_1, DC_node_3,R_Ohm_km=r,Length_km=d13,MW_rating=600,polarity='m',update_grid=False)
    pyf.add_line_DC(grid,DC_node_2, DC_node_3,R_Ohm_km=r,Length_km=d23,MW_rating=600,polarity='m',update_grid=False)

    grid.create_Ybus_DC()
    grid.Update_Graph_DC()

    pyf.add_gen_DC(grid,'Node_1',qf=0.002,lf=20,fc=100,MWmax=400,MWmin=100)
    pyf.add_gen_DC(grid,'Node_2',qf=0.005,lf=25,fc=50,MWmax=400,MWmin=100)

    model, model_res , timing_info, solver_stats =pyf.Optimal_PF(grid,ObjRule=obj)

    # model.pprint()
    res.All()




    print(model_res)
    print(timing_info)
    model.obj.display()

def run_test():
    """Test DC optimal power flow."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    DC_OPF()

if __name__ == "__main__":
    run_test()
       

