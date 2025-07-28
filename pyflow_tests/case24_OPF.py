# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 15:17:12 2025

@author: BernardoCastro
"""
import pandas as pd
import pyflow_acdc as pyf
import time
import sys

def case24_OPF():

    t1=time.time()

    S_base=100

    nodes_AC_data = [
        {'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.08, 'Reactive_load': 0.22, 'Node_id': '1'},
        {'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.97, 'Reactive_load': 0.2, 'Node_id': '2'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.8, 'Reactive_load': 0.37, 'Node_id': '3'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.74, 'Reactive_load': 0.15, 'Node_id': '4'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.71, 'Reactive_load': 0.14, 'Node_id': '5'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.36, 'Reactive_load': 0.28, 'Node_id': '6'},
        {'type': 'PV', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.25, 'Reactive_load': 0.25, 'Node_id': '7'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.71, 'Reactive_load': 0.35, 'Node_id': '8'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.75, 'Reactive_load': 0.36, 'Node_id': '9'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.95, 'Reactive_load': 0.4, 'Node_id': '10'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '11'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '12'},
        {'type': 'Slack', 'Voltage_0': 1.02, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 2.65, 'Reactive_load': 0.54, 'Node_id': '13'},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 1.94, 'Reactive_load': 0.39, 'Node_id': '14'},
        {'type': 'PV', 'Voltage_0': 1.014, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 3.17, 'Reactive_load': 0.64, 'Node_id': '15'},
        {'type': 'PV', 'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 1.0, 'Reactive_load': 0.2, 'Node_id': '16'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '17'},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 3.33, 'Reactive_load': 0.68, 'Node_id': '18'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 1.81, 'Reactive_load': 0.37, 'Node_id': '19'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 1.28, 'Reactive_load': 0.26, 'Node_id': '20'},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '21'},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '22'},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '23'},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '24'}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': '1', 'toNode': '2',   'r': 0.0026, 'x': 0.0139, 'b': 0.4611, 'MVA_rating': 200.0, 'Line_id': '1-2'},
        {'fromNode': '1', 'toNode': '3',   'r': 0.0546, 'x': 0.2112, 'b': 0.0572, 'MVA_rating': 220.0, 'Line_id': '1-3'},
        {'fromNode': '1', 'toNode': '5',   'r': 0.0218, 'x': 0.0845, 'b': 0.0229, 'MVA_rating': 220.0, 'Line_id': '1-5'},
        {'fromNode': '1', 'toNode': '8',   'r': 0.0328, 'x': 0.1344, 'b': 0.0000, 'MVA_rating': 220.0, 'Line_id': '1-8'},
        {'fromNode': '2', 'toNode': '4',   'r': 0.0328, 'x': 0.1267, 'b': 0.0343, 'MVA_rating': 220.0, 'Line_id': '2-4'},
        {'fromNode': '2', 'toNode': '6',   'r': 0.0497, 'x': 0.192,  'b': 0.052,  'MVA_rating': 220.0, 'Line_id': '2-6'},
        {'fromNode': '2', 'toNode': '8',   'r': 0.0328, 'x': 0.1267, 'b': 0.000,  'MVA_rating': 220.0, 'Line_id': '2-8'},
        {'fromNode': '3', 'toNode': '9',   'r': 0.0308, 'x': 0.119,  'b': 0.0322, 'MVA_rating': 220.0, 'Line_id': '3-9'},
        {'fromNode': '3', 'toNode': '24',  'r': 0.0023, 'x': 0.0839, 'b': 0.0,    'MVA_rating': 600.0, 'Line_id': '3-24'},
        {'fromNode': '4', 'toNode': '9',   'r': 0.0268, 'x': 0.1037, 'b': 0.0281, 'MVA_rating': 220.0, 'Line_id': '4-9'},
        {'fromNode': '5', 'toNode': '10',  'r': 0.0228, 'x': 0.0883, 'b': 0.0239, 'MVA_rating': 220.0, 'Line_id': '5-10'},
        {'fromNode': '6', 'toNode': '7',   'r': 0.0497, 'x': 0.1920, 'b': 0.000,  'MVA_rating': 220.0, 'Line_id': '6-7'},
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
        {'fromNode': '13', 'toNode': '14', 'r': 0.0057, 'x': 0.0447, 'b': 0.0000, 'MVA_rating': 625.0, 'Line_id': '13-14'},
        {'fromNode': '13', 'toNode': '23', 'r': 0.0111, 'x': 0.0865, 'b': 0.1818, 'MVA_rating': 625.0, 'Line_id': '13-23'},
        {'fromNode': '14', 'toNode': '16', 'r': 0.005,  'x': 0.0389, 'b': 0.0818, 'MVA_rating': 625.0, 'Line_id': '14-16'},
        {'fromNode': '14', 'toNode': '23', 'r': 0.0080, 'x': 0.0620, 'b': 0.0000, 'MVA_rating': 625.0, 'Line_id': '14-23'},
        {'fromNode': '15', 'toNode': '16', 'r': 0.0022, 'x': 0.0173, 'b': 0.0364, 'MVA_rating': 625.0, 'Line_id': '15-16'},
        {'fromNode': '15', 'toNode': '21', 'r': 0.0063, 'x': 0.049,  'b': 0.103,  'MVA_rating': 625.0, 'Line_id': '15-21'},
        {'fromNode': '15', 'toNode': '24', 'r': 0.0067, 'x': 0.0519, 'b': 0.1091, 'MVA_rating': 625.0, 'Line_id': '15-24'},
        {'fromNode': '16', 'toNode': '17', 'r': 0.0033, 'x': 0.0259, 'b': 0.0545, 'MVA_rating': 625.0, 'Line_id': '16-17'},
        {'fromNode': '16', 'toNode': '19', 'r': 0.003,  'x': 0.0231, 'b': 0.0485, 'MVA_rating': 625.0, 'Line_id': '16-19'},
        {'fromNode': '16', 'toNode': '23', 'r': 0.0105, 'x': 0.0822, 'b': 0.0000, 'MVA_rating': 625.0, 'Line_id': '16-23'},
        {'fromNode': '17', 'toNode': '18', 'r': 0.0018, 'x': 0.0144, 'b': 0.0303, 'MVA_rating': 625.0, 'Line_id': '17-18'},
        {'fromNode': '17', 'toNode': '22', 'r': 0.0135, 'x': 0.1053, 'b': 0.2212, 'MVA_rating': 625.0, 'Line_id': '17-22'},
        {'fromNode': '18', 'toNode': '21', 'r': 0.0033, 'x': 0.0259, 'b': 0.0545, 'MVA_rating': 625.0, 'Line_id': '18-21'},
        {'fromNode': '19', 'toNode': '20', 'r': 0.0051, 'x': 0.0396, 'b': 0.0833, 'MVA_rating': 625.0, 'Line_id': '19-20'},
        {'fromNode': '19', 'toNode': '23', 'r': 0.0078, 'x': 0.0606, 'b': 0.0000, 'MVA_rating': 625.0, 'Line_id': '19-23'},
        {'fromNode': '20', 'toNode': '23', 'r': 0.0028, 'x': 0.0216, 'b': 0.0455, 'MVA_rating': 625.0, 'Line_id': '20-23'},
        {'fromNode': '21', 'toNode': '22', 'r': 0.0087, 'x': 0.0678, 'b': 0.1424, 'MVA_rating': 625.0, 'Line_id': '21-22'}
        ]

    lines_AC = pd.DataFrame(lines_AC_data)

    expandable_data = [
        {'Expandable elements': '1-2',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost':  3*10**6},
        {'Expandable elements': '1-3',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 55*10**6},
        {'Expandable elements': '1-5',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 22*10**6},
        {'Expandable elements': '1-8',  'N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost': 35*10**6},
        {'Expandable elements': '2-4',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 33*10**6},
        {'Expandable elements': '2-6',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '2-8',  'N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost': 33*10**6},
        {'Expandable elements': '3-9',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 31*10**6},
        {'Expandable elements': '3-24', 'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '4-9',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 27*10**6},
        {'Expandable elements': '5-10', 'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 23*10**6},
        {'Expandable elements': '6-7',  'N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '6-10', 'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 16*10**6},
        {'Expandable elements': '7-8',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 16*10**6},
        {'Expandable elements': '8-9',  'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 43*10**6},
        {'Expandable elements': '8-10', 'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 43*10**6},
        {'Expandable elements': '9-11', 'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '9-12', 'N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '10-11','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '10-12','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 50*10**6},
        {'Expandable elements': '11-13','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 66*10**6},
        {'Expandable elements': '11-14','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 58*10**6},
        {'Expandable elements': '12-13','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 66*10**6},
        {'Expandable elements': '12-23','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost':134*10**6},
        {'Expandable elements': '13-14','N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost': 62*10**6},
        {'Expandable elements': '13-23','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost':120*10**6},
        {'Expandable elements': '14-16','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 54*10**6},
        {'Expandable elements': '14-23','N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost': 86*10**6},
        {'Expandable elements': '15-16','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 24*10**6},
        {'Expandable elements': '15-21','N_b': 2, 'N_max': 3, 'Life_time': 30, 'base_cost': 68*10**6},
        {'Expandable elements': '15-24','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 72*10**6},
        {'Expandable elements': '16-17','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 36*10**6},
        {'Expandable elements': '16-19','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 32*10**6},
        {'Expandable elements': '16-23','N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost':114*10**6},
        {'Expandable elements': '17-18','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 20*10**6},
        {'Expandable elements': '17-22','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost':146*10**6},
        {'Expandable elements': '18-21','N_b': 2, 'N_max': 3, 'Life_time': 30, 'base_cost': 36*10**6},
        {'Expandable elements': '19-20','N_b': 2, 'N_max': 3, 'Life_time': 30, 'base_cost': 55*10**6},
        {'Expandable elements': '19-23','N_b': 0, 'N_max': 3, 'Life_time': 30, 'base_cost': 84*10**6},
        {'Expandable elements': '20-23','N_b': 2, 'N_max': 3, 'Life_time': 30, 'base_cost': 30*10**6},
        {'Expandable elements': '21-22','N_b': 1, 'N_max': 3, 'Life_time': 30, 'base_cost': 94*10**6},
    ]

        

    expandable_data = pd.DataFrame(expandable_data)



    grid,res = pyf.Create_grid_from_data(S_base,nodes_AC,lines_AC,data_in='pu')


    gen_size =5

    pyf.add_gen(grid, '1', '1', np_gen=1, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '1', '2', np_gen=1, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '1', '3', np_gen=1, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    pyf.add_gen(grid, '1', '4', np_gen=1, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    pyf.add_gen(grid, '2', '5', np_gen=1, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '2', '6', np_gen=1, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '2', '7', np_gen=1, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    pyf.add_gen(grid, '2', '8', np_gen=1, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    pyf.add_gen(grid, '7', '9', np_gen=1, fc=781.521,lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=0.0)
    pyf.add_gen(grid, '7', '10', np_gen=1, fc=781.521,lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=0.0)
    pyf.add_gen(grid, '7', '11', np_gen=1, fc=781.521,lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=0.0)
    pyf.add_gen(grid, '13', '12', np_gen=1, fc=832.7575,lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=0.0)
    pyf.add_gen(grid, '13', '13', np_gen=1, fc=832.7575,lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=0.0)
    pyf.add_gen(grid, '13', '14', np_gen=1, fc=832.7575,lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=0.0)
    pyf.add_gen(grid, '14', '15', np_gen=1, fc=0.0,lf=0.0, qf=0.0, MWmax=0.0, MWmin=0.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=0.0, QsetMVA=35.3)
    pyf.add_gen(grid, '15', '16', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15', '17', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15', '18', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15', '19', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15', '20', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15', '21', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    pyf.add_gen(grid, '16', '22', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    pyf.add_gen(grid, '18', '23', np_gen=1, fc=395.3749,lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=0.0)
    pyf.add_gen(grid, '21', '24', np_gen=1, fc=395.3749,lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=0.0)
    pyf.add_gen(grid, '22', '25', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    pyf.add_gen(grid, '22', '26', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    pyf.add_gen(grid, '22', '27', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    pyf.add_gen(grid, '22', '28', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    pyf.add_gen(grid, '22', '29', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    pyf.add_gen(grid, '22', '30', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    pyf.add_gen(grid, '23', '31', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    pyf.add_gen(grid, '23', '32', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    pyf.add_gen(grid, '23', '33', np_gen=1, fc=665.1094,lf=11.8495, qf=0.004895, MWmax=350.0, MWmin=140.0, MVArmax=150.0, MVArmin=-25.0, PsetMW=350.0, QsetMVA=0.0)
        
        

    obj = {'Energy_cost': 1}


    pyf.expand_elements_from_pd(grid,expandable_data)

    model, model_res,timing_info,solver_stats= pyf.Optimal_PF(grid,ObjRule=obj)



    res.All()

    print(model_res)
    print(timing_info)
    model.obj.display()
    t2= time.time()
    print(f'Total time :{t2-t1}')


def run_test():
    """Test case24 optimal power flow."""
    try:
        import pyomo
    except ImportError:
        print("pyomo is not installed...")
        return  
    
    case24_OPF()

if __name__ == "__main__":
    run_test()

       