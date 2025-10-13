import pyflow_acdc as pyf
import pandas as pd

def case6_TEP_DC():

    S_base=100

    nodes_AC_data = [
            {'type': 'Slack', 'Voltage_0': 1.02, 'theta_0': 0, 'kV_base': 345, 'Power_Gained': 0.5,  'Power_load': 0.8,'Reactive_load': 0.16,'Umin':0.95,'Umax':1.05, 'Node_id': '1'},
            {'type': 'PQ', 'Voltage_0': 1, 'theta_0': -0.1, 'kV_base': 345, 'Power_Gained': 0,  'Power_load': 2.4,  'Reactive_load': 0.48,'Umin':0.95,'Umax':1.05, 'Node_id': '2'},
            {'type': 'PV', 'Voltage_0': 1.04, 'theta_0': 0.1, 'kV_base': 345, 'Power_Gained': 1.65,  'Power_load': 0.4, 'Reactive_load': 0.08,'Umin':0.95,'Umax':1.05, 'Node_id': '3'},
            {'type': 'PQ', 'Voltage_0': 1, 'theta_0': 0.1, 'kV_base': 345, 'Power_Gained': 0,  'Power_load': 1.6,   'Reactive_load': 0.32,'Umin':0.95,'Umax':1.05, 'Node_id': '4'},
            {'type': 'PQ', 'Voltage_0': 1, 'theta_0': 0.1, 'kV_base': 345, 'Power_Gained': 0,  'Power_load': 2.4, 'Reactive_load': 0.48,'Umin':0.95,'Umax':1.05, 'Node_id': '5'},
            {'type': 'PV', 'Voltage_0': 1.04, 'theta_0': 0, 'kV_base': 345, 'Power_Gained': 5.45,  'Power_load': 0, 'Reactive_load': 0,'Umin':0.95,'Umax':1.05, 'Node_id': '6'}    
            ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': '1', 'toNode': '2', 'r': 0.040, 'x': 0.400, 'MVA_rating': 100, 'Line_id': '1-2'},
        {'fromNode': '1', 'toNode': '4', 'r': 0.060, 'x': 0.600, 'MVA_rating': 80,  'Line_id': '1-4'},
        {'fromNode': '1', 'toNode': '5', 'r': 0.020, 'x': 0.200, 'MVA_rating': 100, 'Line_id': '1-5'},
        {'fromNode': '2', 'toNode': '3', 'r': 0.020, 'x': 0.200, 'MVA_rating': 100, 'Line_id': '2-3'},
        {'fromNode': '2', 'toNode': '4', 'r': 0.040, 'x': 0.400, 'MVA_rating': 100, 'Line_id': '2-4'},
        {'fromNode': '3', 'toNode': '5', 'r': 0.020, 'x': 0.200, 'MVA_rating': 100, 'Line_id': '3-5'},
        ]
    lines_AC = pd.DataFrame(lines_AC_data)

    nodes_DC_data =[
        {'type':'P','Voltage_0':1,'kV_base':345,'Node_id':'2'},
        {'type':'P','Voltage_0':1,'kV_base':345,'Node_id':'3'},
        {'type':'P','Voltage_0':1,'kV_base':345,'Node_id':'4'},
        {'type':'P','Voltage_0':1,'kV_base':345,'Node_id':'5'},
        {'type':'Slack','Voltage_0':1,'kV_base':345,'Node_id':'6'},
        ]
    nodes_DC = pd.DataFrame(nodes_DC_data)

    lines_DC_data=[
        {'fromNode': '2', 'toNode': '5', 'r': 0.01, 'MW_rating': 100, 'Line_id': '2-5'},
        {'fromNode': '2', 'toNode': '6', 'r': 0.01, 'MW_rating': 100, 'Line_id': '2-6'},
        {'fromNode': '3', 'toNode': '6', 'r': 0.01, 'MW_rating': 100, 'Line_id': '3-6'},
        {'fromNode': '4', 'toNode': '5', 'r': 0.01, 'MW_rating': 100, 'Line_id': '4-5'},
        {'fromNode': '4', 'toNode': '6', 'r': 0.01, 'MW_rating': 100, 'Line_id': '4-6'},
        {'fromNode': '5', 'toNode': '6', 'r': 0.01, 'MW_rating': 100, 'Line_id': '5-6'},
        
        ]
    lines_DC = pd.DataFrame(lines_DC_data)

    convdc_data = [
        {'DC_node':'2','AC_node':'2','T_r':0.01,'T_x':0.01,'Filter_b':0.01,'PR_r':0.01,'PR_x':0.01,'MVA_rating':700,'Ucmin':0.9,'Ucmax':1.1,'Conv_id':'cn2'},
        {'DC_node':'3','AC_node':'3','T_r':0.01,'T_x':0.01,'Filter_b':0.01,'PR_r':0.01,'PR_x':0.01,'MVA_rating':700,'Ucmin':0.9,'Ucmax':1.1,'Conv_id':'cn3'},  
        {'DC_node':'4','AC_node':'4','T_r':0.01,'T_x':0.01,'Filter_b':0.01,'PR_r':0.01,'PR_x':0.01,'MVA_rating':700,'Ucmin':0.9,'Ucmax':1.1,'Conv_id':'cn4'},
        {'DC_node':'5','AC_node':'5','T_r':0.01,'T_x':0.01,'Filter_b':0.01,'PR_r':0.01,'PR_x':0.01,'MVA_rating':700,'Ucmin':0.9,'Ucmax':1.1,'Conv_id':'cn5'},
        {'DC_node':'6','AC_node':'6','T_r':0.01,'T_x':0.01,'Filter_b':0.01,'PR_r':0.01,'PR_x':0.01,'MVA_rating':700,'Ucmin':0.9,'Ucmax':1.1,'Conv_id':'cn6'},
        ]
    convacdc = pd.DataFrame(convdc_data)


    expandable_data = [
        {'Expandable elements': '2-5' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':1},
        {'Expandable elements': '2-6' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':1},
        {'Expandable elements': '3-6' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':1},
        {'Expandable elements': '4-5' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':1},
        {'Expandable elements': '4-6' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':1},
        {'Expandable elements': '5-6' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':1},
        {'Expandable elements': 'cn2' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':3},
        {'Expandable elements': 'cn3' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':3},
        {'Expandable elements': 'cn4' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':3},
        {'Expandable elements': 'cn5' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':3},
        {'Expandable elements': 'cn6' ,'N_b': 0,'N_max': 3,'Life_time': 30, 'base_cost':3}
        ]
    expandable_data = pd.DataFrame(expandable_data)



    grid,res = pyf.Create_grid_from_data(S_base,nodes_AC,lines_AC,nodes_DC,lines_DC,convacdc,data_in='pu')




    pyf.add_gen(grid,'1',MWmax=150,MVArmin=-10,MVArmax=48,PsetMW=105)
    pyf.add_gen(grid,'3',MWmax=360,MVArmin=-10,MVArmax=101,PsetMW=245)
    pyf.add_gen(grid,'6',MWmax=600,MVArmin=-10,MVArmax=183,PsetMW=400)

    # pyf.AC_PowerFlow(grid)


    pyf.expand_elements_from_pd(grid,expandable_data)

    model, model_results , timing_info, solver_stats= pyf.transmission_expansion(grid,NPV=True)


    print(timing_info)

def run_test():
    """Test case6 DC transmission expansion planning."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    try:
        import pyomo.environ as pyo
        solver = pyo.SolverFactory('bonmin')
        if not solver.available():
            raise ImportError("Bonmin solver not available")
    except (ImportError, Exception):
        print("Bonmin solver is not available. To use TEP functions, please install Bonmin solver.")
        return
    
    case6_TEP_DC()

if __name__ == "__main__":
    run_test()