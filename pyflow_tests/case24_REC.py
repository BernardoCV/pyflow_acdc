# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:17:12 2025

@author: BernardoCastro
"""
import pandas as pd
import pyflow_acdc as pyf

def case24_REC():
    S_base=100

    nodes_AC_data = [
        {'type': 'Slack', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 3.24, 'Reactive_load': 0.66, 'Node_id': '1'},
        {'type': 'PV',    'Voltage_0': 1.035, 'theta_0': 0.1, 'kV_base': 138,  'Power_load': 2.91, 'Reactive_load': 0.60, 'Node_id': '2'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.01, 'kV_base': 138,  'Power_load': 5.40, 'Reactive_load': 1.11, 'Node_id': '3'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 2.22, 'Reactive_load': 0.45, 'Node_id': '4'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 2.13, 'Reactive_load': 0.42, 'Node_id': '5'},
        {'type': 'PV',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 4.08, 'Reactive_load': 0.84, 'Node_id': '6'},
        {'type': 'PV',    'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138,'Power_load': 3.75, 'Reactive_load': 0.75, 'Node_id': '7'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 5.13, 'Reactive_load': 1.05, 'Node_id': '8'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 5.25, 'Reactive_load': 1.08, 'Node_id': '9'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138,  'Power_load': 5.85, 'Reactive_load': 1.20, 'Node_id': '10'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230,  'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '11'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230,  'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '12'},
        {'type': 'PV',    'Voltage_0': 1.02, 'theta_0': 0.0, 'kV_base': 230, 'Power_load': 7.95, 'Reactive_load': 1.62, 'Node_id': '13'},
        {'type': 'PV',    'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 230, 'Power_load': 5.82, 'Reactive_load': 1.17, 'Node_id': '14'},
        {'type': 'PV',    'Voltage_0': 1.014, 'theta_0': 0.0, 'kV_base': 230,'Power_load': 9.51, 'Reactive_load': 1.92, 'Node_id': '15'},
        {'type': 'PV',    'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 230,'Power_load': 3.00, 'Reactive_load': 0.60, 'Node_id': '16'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230,  'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '17'},
        {'type': 'PV',    'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230, 'Power_load': 9.99, 'Reactive_load': 2.04, 'Node_id': '18'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230,  'Power_load': 5.43, 'Reactive_load': 1.11, 'Node_id': '19'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230,  'Power_load': 3.84, 'Reactive_load': 0.78, 'Node_id': '20'},
        {'type': 'PV',    'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230, 'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '21'},
        {'type': 'PV',    'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230, 'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '22'},
        {'type': 'PV',    'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230, 'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '23'},
        {'type': 'PQ',    'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230,  'Power_load': 0.00, 'Reactive_load': 0.00, 'Node_id': '24'}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)


    lines_AC_data = [
        {'fromNode': '1', 'toNode': '2',   'r': 0.0026, 'x': 0.0139, 'b': 0.4611, 'MVA_rating': 200.0, 'Line_id': '1-2'},
        {'fromNode': '1', 'toNode': '3',   'r': 0.0546, 'x': 0.2112, 'b': 0.0572, 'MVA_rating': 220.0, 'Line_id': '1-3'},
        {'fromNode': '1', 'toNode': '5',   'r': 0.0218, 'x': 0.0845, 'b': 0.0229, 'MVA_rating': 220.0, 'Line_id': '1-5'},
        {'fromNode': '2', 'toNode': '4',   'r': 0.0328, 'x': 0.1267, 'b': 0.0343, 'MVA_rating': 220.0, 'Line_id': '2-4'},
        {'fromNode': '2', 'toNode': '6',   'r': 0.0497, 'x': 0.192,  'b': 0.052,  'MVA_rating': 220.0, 'Line_id': '2-6'},
        {'fromNode': '3', 'toNode': '9',   'r': 0.0308, 'x': 0.119,  'b': 0.0322, 'MVA_rating': 220.0, 'Line_id': '3-9'},
        {'fromNode': '3', 'toNode': '24',  'r': 0.0023, 'x': 0.0839, 'b': 0.0,    'MVA_rating': 600.0, 'Line_id': '3-24'},
        {'fromNode': '4', 'toNode': '9',   'r': 0.0268, 'x': 0.1037, 'b': 0.0281, 'MVA_rating': 220.0, 'Line_id': '4-9'},
        {'fromNode': '5', 'toNode': '10',  'r': 0.0228, 'x': 0.0883, 'b': 0.0239, 'MVA_rating': 220.0, 'Line_id': '5-10'},
        {'fromNode': '6', 'toNode': '10',  'r': 0.0139, 'x': 0.0605, 'b': 2.459,  'MVA_rating': 200.0, 'Line_id': '6-10'},
        {'fromNode': '7', 'toNode': '8',   'r': 0.0159, 'x': 0.0614, 'b': 0.0166, 'MVA_rating': 220.0, 'Line_id': '7-8'},
        {'fromNode': '8', 'toNode': '9',   'r': 0.0427, 'x': 0.1651, 'b': 0.0447, 'MVA_rating': 220.0, 'Line_id': '8-9'},
        {'fromNode': '8', 'toNode': '10',  'r': 0.0427, 'x': 0.1651, 'b': 0.0447, 'MVA_rating': 220.0, 'Line_id': '8-10'},
        {'fromNode': '9', 'toNode': '11',  'r': 0.0023, 'x': 0.0839, 'b': 0.0,    'MVA_rating': 600.0, 'Line_id': '9-11'},
        {'fromNode': '9', 'toNode': '12',  'r': 0.0023, 'x': 0.0839, 'b': 0.0,    'MVA_rating': 600.0, 'Line_id': '9-12'},
        {'fromNode': '10', 'toNode': '11', 'r': 0.0023, 'x': 0.0839, 'b': 0.0,    'MVA_rating': 600.0, 'Line_id': '10-11'},
        {'fromNode': '10', 'toNode': '12', 'r': 0.0023, 'x': 0.0839, 'b': 0.0,    'MVA_rating': 600.0, 'Line_id': '10-12'},
        {'fromNode': '11', 'toNode': '13', 'r': 0.0061, 'x': 0.0476, 'b': 0.0999, 'MVA_rating': 625.0, 'Line_id': '11-13'},
        {'fromNode': '11', 'toNode': '14', 'r': 0.0054, 'x': 0.0418, 'b': 0.0879, 'MVA_rating': 625.0, 'Line_id': '11-14'},
        {'fromNode': '12', 'toNode': '13', 'r': 0.0061, 'x': 0.0476, 'b': 0.0999, 'MVA_rating': 625.0, 'Line_id': '12-13'},
        {'fromNode': '12', 'toNode': '23', 'r': 0.0124, 'x': 0.0966, 'b': 0.203,  'MVA_rating': 625.0, 'Line_id': '12-23'},
        {'fromNode': '13', 'toNode': '23', 'r': 0.0111, 'x': 0.0865, 'b': 0.1818, 'MVA_rating': 625.0, 'Line_id': '13-23'},
        {'fromNode': '14', 'toNode': '16', 'r': 0.005,  'x': 0.0389, 'b': 0.0818, 'MVA_rating': 625.0, 'Line_id': '14-16'},
        {'fromNode': '15', 'toNode': '16', 'r': 0.0022, 'x': 0.0173, 'b': 0.0364, 'MVA_rating': 625.0, 'Line_id': '15-16'},
        {'fromNode': '15', 'toNode': '21', 'r': 0.0063, 'x': 0.049,  'b': 0.103,  'MVA_rating': 625.0, 'Line_id': '15-21'},
        {'fromNode': '15', 'toNode': '24', 'r': 0.0067, 'x': 0.0519, 'b': 0.1091, 'MVA_rating': 625.0, 'Line_id': '15-24'},
        {'fromNode': '16', 'toNode': '17', 'r': 0.0033, 'x': 0.0259, 'b': 0.0545, 'MVA_rating': 625.0, 'Line_id': '16-17'},
        {'fromNode': '16', 'toNode': '19', 'r': 0.003,  'x': 0.0231, 'b': 0.0485, 'MVA_rating': 625.0, 'Line_id': '16-19'},
        {'fromNode': '17', 'toNode': '18', 'r': 0.0018, 'x': 0.0144, 'b': 0.0303, 'MVA_rating': 625.0, 'Line_id': '17-18'},
        {'fromNode': '17', 'toNode': '22', 'r': 0.0135, 'x': 0.1053, 'b': 0.2212, 'MVA_rating': 625.0, 'Line_id': '17-22'},
        {'fromNode': '18', 'toNode': '21', 'r': 0.0033, 'x': 0.0259, 'b': 0.0545, 'MVA_rating': 625.0, 'Line_id': '18-21'},
        {'fromNode': '19', 'toNode': '20', 'r': 0.0051, 'x': 0.0396, 'b': 0.0833, 'MVA_rating': 625.0, 'Line_id': '19-20'},
        {'fromNode': '20', 'toNode': '23', 'r': 0.0028, 'x': 0.0216, 'b': 0.0455, 'MVA_rating': 625.0, 'Line_id': '20-23'},
        {'fromNode': '21', 'toNode': '22', 'r': 0.0087, 'x': 0.0678, 'b': 0.1424, 'MVA_rating': 625.0, 'Line_id': '21-22'}
        ]

    lines_AC = pd.DataFrame(lines_AC_data)

    upgradable_data = [
        {'Line_id': '1-2', 'r_new': 0.001733333, 'x_new': 0.009266667, 'b_new': 0.3074, 'MVA_rating_new': 300.0, 'base_cost': 0.9},
        {'Line_id': '1-3', 'r_new': 0.0364, 'x_new': 0.1408, 'b_new': 0.038133333, 'MVA_rating_new': 330.0, 'base_cost': 16.5},
        {'Line_id': '1-5', 'r_new': 0.014533333, 'x_new': 0.056333333, 'b_new': 0.015266667, 'MVA_rating_new': 330.0, 'base_cost': 6.6},
        {'Line_id': '2-4', 'r_new': 0.021866667, 'x_new': 0.084466667, 'b_new': 0.022866667, 'MVA_rating_new': 330.0, 'base_cost': 9.9},
        {'Line_id': '2-6', 'r_new': 0.033133333, 'x_new': 0.128, 'b_new': 0.034666667, 'MVA_rating_new': 330.0, 'base_cost': 15.0},
        {'Line_id': '3-9', 'r_new': 0.020533333, 'x_new': 0.079333333, 'b_new': 0.021466667, 'MVA_rating_new': 330.0, 'base_cost': 9.3},
        {'Line_id': '3-24', 'r_new': 0.001533333, 'x_new': 0.055933333, 'b_new': 0.0, 'MVA_rating_new': 900.0, 'base_cost': 15.0},
        {'Line_id': '4-9', 'r_new': 0.017866667, 'x_new': 0.069133333, 'b_new': 0.018733333, 'MVA_rating_new': 330.0, 'base_cost': 8.1},
        {'Line_id': '5-10', 'r_new': 0.0152, 'x_new': 0.058866667, 'b_new': 0.015933333, 'MVA_rating_new': 330.0, 'base_cost': 6.9},
        {'Line_id': '6-10', 'r_new': 0.009266667, 'x_new': 0.040333333, 'b_new': 1.639333333, 'MVA_rating_new': 300.0, 'base_cost': 4.8},
        {'Line_id': '7-8', 'r_new': 0.0106, 'x_new': 0.040933333, 'b_new': 0.011066667, 'MVA_rating_new': 330.0, 'base_cost': 4.8},
        {'Line_id': '8-9', 'r_new': 0.028466667, 'x_new': 0.110066667, 'b_new': 0.0298, 'MVA_rating_new': 330.0, 'base_cost': 12.9},
        {'Line_id': '8-10', 'r_new': 0.028466667, 'x_new': 0.110066667, 'b_new': 0.0298, 'MVA_rating_new': 330.0, 'base_cost': 12.9},
        {'Line_id': '9-11', 'r_new': 0.001533333, 'x_new': 0.055933333, 'b_new': 0.0, 'MVA_rating_new': 900.0, 'base_cost': 15.0},
        {'Line_id': '9-12', 'r_new': 0.001533333, 'x_new': 0.055933333, 'b_new': 0.0, 'MVA_rating_new': 900.0, 'base_cost': 15.0},
        {'Line_id': '10-11', 'r_new': 0.001533333, 'x_new': 0.055933333, 'b_new': 0.0, 'MVA_rating_new': 900.0, 'base_cost': 15.0},
        {'Line_id': '10-12', 'r_new': 0.001533333, 'x_new': 0.055933333, 'b_new': 0.0, 'MVA_rating_new': 900.0, 'base_cost': 15.0},
        {'Line_id': '11-13', 'r_new': 0.004066667, 'x_new': 0.031733333, 'b_new': 0.0666, 'MVA_rating_new': 937.5, 'base_cost': 19.8},
        {'Line_id': '11-14', 'r_new': 0.0036, 'x_new': 0.027866667, 'b_new': 0.0586, 'MVA_rating_new': 937.5, 'base_cost': 17.4},
        {'Line_id': '12-13', 'r_new': 0.004066667, 'x_new': 0.031733333, 'b_new': 0.0666, 'MVA_rating_new': 937.5, 'base_cost': 19.8},
        {'Line_id': '12-23', 'r_new': 0.008266667, 'x_new': 0.0644, 'b_new': 0.135333333, 'MVA_rating_new': 937.5, 'base_cost': 40.2},
        {'Line_id': '13-23', 'r_new': 0.0074, 'x_new': 0.057666667, 'b_new': 0.1212, 'MVA_rating_new': 937.5, 'base_cost': 36.0},
        {'Line_id': '14-16', 'r_new': 0.003333333, 'x_new': 0.025933333, 'b_new': 0.054533333, 'MVA_rating_new': 937.5, 'base_cost': 16.2},
        {'Line_id': '15-16', 'r_new': 0.001466667, 'x_new': 0.011533333, 'b_new': 0.024266667, 'MVA_rating_new': 937.5, 'base_cost': 7.2},
        {'Line_id': '15-21', 'r_new': 0.0042, 'x_new': 0.032666667, 'b_new': 0.068666667, 'MVA_rating_new': 937.5, 'base_cost': 20.4},
        {'Line_id': '15-24', 'r_new': 0.004466667, 'x_new': 0.0346, 'b_new': 0.072733333, 'MVA_rating_new': 937.5, 'base_cost': 21.6},
        {'Line_id': '16-17', 'r_new': 0.0022, 'x_new': 0.017266667, 'b_new': 0.036333333, 'MVA_rating_new': 937.5, 'base_cost': 10.8},
        {'Line_id': '16-19', 'r_new': 0.002, 'x_new': 0.0154, 'b_new': 0.032333333, 'MVA_rating_new': 937.5, 'base_cost': 9.6},
        {'Line_id': '17-18', 'r_new': 0.0012, 'x_new': 0.0096, 'b_new': 0.0202, 'MVA_rating_new': 937.5, 'base_cost': 6.0},
        {'Line_id': '17-22', 'r_new': 0.009, 'x_new': 0.0702, 'b_new': 0.147466667, 'MVA_rating_new': 937.5, 'base_cost': 43.8},
        {'Line_id': '18-21', 'r_new': 0.0022, 'x_new': 0.017266667, 'b_new': 0.036333333, 'MVA_rating_new': 937.5, 'base_cost': 10.8},
        {'Line_id': '19-20', 'r_new': 0.0034, 'x_new': 0.0264, 'b_new': 0.055533333, 'MVA_rating_new': 937.5, 'base_cost': 16.5},
        {'Line_id': '20-23', 'r_new': 0.001866667, 'x_new': 0.0144, 'b_new': 0.030333333, 'MVA_rating_new': 937.5, 'base_cost': 9.0},
        {'Line_id': '21-22', 'r_new': 0.0058, 'x_new': 0.0452, 'b_new': 0.094933333, 'MVA_rating_new': 937.5, 'base_cost': 28.2}
    ]

        

    upgradable_data = pd.DataFrame(upgradable_data)



    grid,res = pyf.Create_grid_from_data(S_base,nodes_AC,lines_AC,data_in='pu')

    #pyf.add_gen(grid,'3', MWmax=0.0, MVArmin=-9999,MVArmax=9999,PsetMW=0.0)
    #pyf.add_gen(grid,'4', MWmax=0.0, MVArmin=-9999,MVArmax=9999,PsetMW=0.0)
    #pyf.add_gen(grid,'9', MWmax=0.0, MVArmin=-9999,MVArmax=9999,PsetMW=0.0)

    pyf.add_gen(grid,'1', MWmax=576, MVArmin=-150,MVArmax=240,PsetMW=105)
    pyf.add_gen(grid,'2', MWmax=576, MVArmin=-150,MVArmax=240,PsetMW=245)
    pyf.add_gen(grid,'6', MWmax=0.0, MVArmin=-300,MVArmax=0.0,PsetMW=0.0)
    pyf.add_gen(grid,'7', MWmax=900, MVArmin= 0.0,MVArmax=540,PsetMW=400)
    pyf.add_gen(grid,'13',MWmax=1773,MVArmin= 0.0,MVArmax=720,PsetMW=105)
    pyf.add_gen(grid,'14',MWmax=0.0, MVArmin=-150,MVArmax=600,PsetMW=0.0)
    pyf.add_gen(grid,'15',MWmax=645, MVArmin=-150,MVArmax=330,PsetMW=951)
    pyf.add_gen(grid,'16',MWmax=465, MVArmin=-150,MVArmax=330,PsetMW=300)
    pyf.add_gen(grid,'18',MWmax=1200,MVArmin=-150,MVArmax=240,PsetMW=105)
    pyf.add_gen(grid,'21',MWmax=1200,MVArmin=-150,MVArmax=600,PsetMW=245)
    pyf.add_gen(grid,'22',MWmax=900, MVArmin=-180,MVArmax=288,PsetMW=400)
    pyf.add_gen(grid,'23',MWmax=1980,MVArmin=-375,MVArmax=930,PsetMW=105)


    pyf.repurpose_element_from_pd(grid,upgradable_data)

    model, model_results , timing_info, solver_stats= pyf.transmission_expansion(grid,NPV=True)
    #res.All()
    res.TEP_N()
    res.TEP_norm()

    print(timing_info)
    model.obj.display()

def run_test():
    """Test case24 renewable energy curtailment."""
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
    case24_REC()

if __name__ == "__main__":
    run_test()