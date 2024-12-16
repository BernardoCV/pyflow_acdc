

import pyflow_acdc as pyf
import pandas as pd
"""
Converted to PyFlowACDC format from
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                                                                  %%%%%
%%%%    IEEE PES Power Grid Library - Optimal Power Flow - v21.07     %%%%%
%%%%          (https://github.com/power-grid-lib/pglib-opf)           %%%%%
%%%%               Benchmark Group - Typical Operations               %%%%%
%%%%                         29 - July - 2021                         %%%%%
%%%%                                                                  %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Power flow data for the IEEE RELIABILITY TEST SYSTEM 1979.
%
%   IEEE Reliability Test System Task Force of the Applications of
%   Probability Methods Subcommittee, "IEEE reliability test system,"
%   IEEE Transactions on Power Apparatus and Systems, Vol. 98, No. 6,
%   Nov./Dec. 1979, pp. 2047-2054.
%
%   Cost data is from Web site run by Georgia Tech Power Systems Control
%   and Automation Laboratory:
%       http://pscal.ece.gatech.edu/testsys/index.html
%
%   Matpower case file data provided by Bruce Wollenberg.
%
%   Copyright (c) 1979 The Institute of Electrical and Electronics Engineers (IEEE)
%   Licensed under the Creative Commons Attribution 4.0
%   International license, http://creativecommons.org/licenses/by/4.0/
%
%   Contact M.E. Brennan (me.brennan@ieee.org) for inquries on further reuse of
%   this dataset.
%

generator cost c(0) modified to 0

"""

def pglib_opf_case24_ieee_rts():    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.08, 'Reactive_load': 0.22, 'Node_id': '1.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.97, 'Reactive_load': 0.2, 'Node_id': '2.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.8, 'Reactive_load': 0.37, 'Node_id': '3.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.74, 'Reactive_load': 0.15, 'Node_id': '4.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.71, 'Reactive_load': 0.14, 'Node_id': '5.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.36, 'Reactive_load': 0.28, 'Node_id': '6.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': -1.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.25, 'Reactive_load': 0.25, 'Node_id': '7.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.71, 'Reactive_load': 0.35, 'Node_id': '8.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.75, 'Reactive_load': 0.36, 'Node_id': '9.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.95, 'Reactive_load': 0.4, 'Node_id': '10.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '11.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '12.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.65, 'Reactive_load': 0.54, 'Node_id': '13.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.94, 'Reactive_load': 0.39, 'Node_id': '14.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.17, 'Reactive_load': 0.64, 'Node_id': '15.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.0, 'Reactive_load': 0.2, 'Node_id': '16.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '17.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.33, 'Reactive_load': 0.68, 'Node_id': '18.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.81, 'Reactive_load': 0.37, 'Node_id': '19.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.28, 'Reactive_load': 0.26, 'Node_id': '20.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '21.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '22.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '23.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '24.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': '1.0', 'toNode': '2.0', 'Resistance': 0.0026, 'Reactance': 0.0139, 'Conductance': 0, 'Susceptance': 0.4611, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '1'},
        {'fromNode': '1.0', 'toNode': '3.0', 'Resistance': 0.0546, 'Reactance': 0.2112, 'Conductance': 0, 'Susceptance': 0.0572, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '2'},
        {'fromNode': '1.0', 'toNode': '5.0', 'Resistance': 0.0218, 'Reactance': 0.0845, 'Conductance': 0, 'Susceptance': 0.0229, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '3'},
        {'fromNode': '2.0', 'toNode': '4.0', 'Resistance': 0.0328, 'Reactance': 0.1267, 'Conductance': 0, 'Susceptance': 0.0343, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '4'},
        {'fromNode': '2.0', 'toNode': '6.0', 'Resistance': 0.0497, 'Reactance': 0.192, 'Conductance': 0, 'Susceptance': 0.052, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '5'},
        {'fromNode': '3.0', 'toNode': '9.0', 'Resistance': 0.0308, 'Reactance': 0.119, 'Conductance': 0, 'Susceptance': 0.0322, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '6'},
        {'fromNode': '3.0', 'toNode': '24.0', 'Resistance': 0.0023, 'Reactance': 0.0839, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '7'},
        {'fromNode': '4.0', 'toNode': '9.0', 'Resistance': 0.0268, 'Reactance': 0.1037, 'Conductance': 0, 'Susceptance': 0.0281, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '8'},
        {'fromNode': '5.0', 'toNode': '10.0', 'Resistance': 0.0228, 'Reactance': 0.0883, 'Conductance': 0, 'Susceptance': 0.0239, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '9'},
        {'fromNode': '6.0', 'toNode': '10.0', 'Resistance': 0.0139, 'Reactance': 0.0605, 'Conductance': 0, 'Susceptance': 2.459, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '10'},
        {'fromNode': '7.0', 'toNode': '8.0', 'Resistance': 0.0159, 'Reactance': 0.0614, 'Conductance': 0, 'Susceptance': 0.0166, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '11'},
        {'fromNode': '8.0', 'toNode': '9.0', 'Resistance': 0.0427, 'Reactance': 0.1651, 'Conductance': 0, 'Susceptance': 0.0447, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '12'},
        {'fromNode': '8.0', 'toNode': '10.0', 'Resistance': 0.0427, 'Reactance': 0.1651, 'Conductance': 0, 'Susceptance': 0.0447, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '13'},
        {'fromNode': '9.0', 'toNode': '11.0', 'Resistance': 0.0023, 'Reactance': 0.0839, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '14'},
        {'fromNode': '9.0', 'toNode': '12.0', 'Resistance': 0.0023, 'Reactance': 0.0839, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '15'},
        {'fromNode': '10.0', 'toNode': '11.0', 'Resistance': 0.0023, 'Reactance': 0.0839, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.02, 'shift': 0.0, 'Line_id': '16'},
        {'fromNode': '10.0', 'toNode': '12.0', 'Resistance': 0.0023, 'Reactance': 0.0839, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.02, 'shift': 0.0, 'Line_id': '17'},
        {'fromNode': '11.0', 'toNode': '13.0', 'Resistance': 0.0061, 'Reactance': 0.0476, 'Conductance': 0, 'Susceptance': 0.0999, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '18'},
        {'fromNode': '11.0', 'toNode': '14.0', 'Resistance': 0.0054, 'Reactance': 0.0418, 'Conductance': 0, 'Susceptance': 0.0879, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '19'},
        {'fromNode': '12.0', 'toNode': '13.0', 'Resistance': 0.0061, 'Reactance': 0.0476, 'Conductance': 0, 'Susceptance': 0.0999, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '20'},
        {'fromNode': '12.0', 'toNode': '23.0', 'Resistance': 0.0124, 'Reactance': 0.0966, 'Conductance': 0, 'Susceptance': 0.203, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '21'},
        {'fromNode': '13.0', 'toNode': '23.0', 'Resistance': 0.0111, 'Reactance': 0.0865, 'Conductance': 0, 'Susceptance': 0.1818, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '22'},
        {'fromNode': '14.0', 'toNode': '16.0', 'Resistance': 0.005, 'Reactance': 0.0389, 'Conductance': 0, 'Susceptance': 0.0818, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '23'},
        {'fromNode': '15.0', 'toNode': '16.0', 'Resistance': 0.0022, 'Reactance': 0.0173, 'Conductance': 0, 'Susceptance': 0.0364, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '24'},
        {'fromNode': '15.0', 'toNode': '21.0', 'Resistance': 0.0063, 'Reactance': 0.049, 'Conductance': 0, 'Susceptance': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '25'},
        {'fromNode': '15.0', 'toNode': '21.0', 'Resistance': 0.0063, 'Reactance': 0.049, 'Conductance': 0, 'Susceptance': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '26'},
        {'fromNode': '15.0', 'toNode': '24.0', 'Resistance': 0.0067, 'Reactance': 0.0519, 'Conductance': 0, 'Susceptance': 0.1091, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '27'},
        {'fromNode': '16.0', 'toNode': '17.0', 'Resistance': 0.0033, 'Reactance': 0.0259, 'Conductance': 0, 'Susceptance': 0.0545, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '28'},
        {'fromNode': '16.0', 'toNode': '19.0', 'Resistance': 0.003, 'Reactance': 0.0231, 'Conductance': 0, 'Susceptance': 0.0485, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '29'},
        {'fromNode': '17.0', 'toNode': '18.0', 'Resistance': 0.0018, 'Reactance': 0.0144, 'Conductance': 0, 'Susceptance': 0.0303, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '30'},
        {'fromNode': '17.0', 'toNode': '22.0', 'Resistance': 0.0135, 'Reactance': 0.1053, 'Conductance': 0, 'Susceptance': 0.2212, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '31'},
        {'fromNode': '18.0', 'toNode': '21.0', 'Resistance': 0.0033, 'Reactance': 0.0259, 'Conductance': 0, 'Susceptance': 0.0545, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '32'},
        {'fromNode': '18.0', 'toNode': '21.0', 'Resistance': 0.0033, 'Reactance': 0.0259, 'Conductance': 0, 'Susceptance': 0.0545, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '33'},
        {'fromNode': '19.0', 'toNode': '20.0', 'Resistance': 0.0051, 'Reactance': 0.0396, 'Conductance': 0, 'Susceptance': 0.0833, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '34'},
        {'fromNode': '19.0', 'toNode': '20.0', 'Resistance': 0.0051, 'Reactance': 0.0396, 'Conductance': 0, 'Susceptance': 0.0833, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '35'},
        {'fromNode': '20.0', 'toNode': '23.0', 'Resistance': 0.0028, 'Reactance': 0.0216, 'Conductance': 0, 'Susceptance': 0.0455, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '36'},
        {'fromNode': '20.0', 'toNode': '23.0', 'Resistance': 0.0028, 'Reactance': 0.0216, 'Conductance': 0, 'Susceptance': 0.0455, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '37'},
        {'fromNode': '21.0', 'toNode': '22.0', 'Resistance': 0.0087, 'Reactance': 0.0678, 'Conductance': 0, 'Susceptance': 0.1424, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '38'}
    ]
    lines_AC = pd.DataFrame(lines_AC_data)

    nodes_DC = None

    lines_DC = None

    Converters_ACDC = None

    
    # Create the grid
    [grid, res] = pyf.Create_grid_from_data(S_base, nodes_AC, lines_AC, nodes_DC, lines_DC, Converters_ACDC, data_in_pu=True)
    
    # Assign Price Zones to Nodes
    for index, row in nodes_AC.iterrows():
        node_name = nodes_AC.at[index, 'Node_id']
        price_zone = nodes_AC.at[index, 'PZ']
        ACDC = 'AC'
        if price_zone is not None:
            pyf.assign_nodeToPrice_Zone(grid, node_name, ACDC, price_zone)
    
    # Add Generators
    pyf.add_gen(grid, '1.0', '1', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=18.0, QsetMVA=5.0)
    pyf.add_gen(grid, '1.0', '2', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=18.0, QsetMVA=5.0)
    pyf.add_gen(grid, '1.0', '3', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=45.6, QsetMVA=2.5)
    pyf.add_gen(grid, '1.0', '4', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=45.6, QsetMVA=2.5)
    pyf.add_gen(grid, '2.0', '5', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=18.0, QsetMVA=5.0)
    pyf.add_gen(grid, '2.0', '6', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=18.0, QsetMVA=5.0)
    pyf.add_gen(grid, '2.0', '7', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=45.6, QsetMVA=2.5)
    pyf.add_gen(grid, '2.0', '8', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=45.6, QsetMVA=2.5)
    pyf.add_gen(grid, '7.0', '9', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=62.5, QsetMVA=30.0)
    pyf.add_gen(grid, '7.0', '10', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=62.5, QsetMVA=30.0)
    pyf.add_gen(grid, '7.0', '11', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=62.5, QsetMVA=30.0)
    pyf.add_gen(grid, '13.0', '12', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=133.0, QsetMVA=40.0)
    pyf.add_gen(grid, '13.0', '13', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=133.0, QsetMVA=40.0)
    pyf.add_gen(grid, '13.0', '14', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=133.0, QsetMVA=40.0)
    pyf.add_gen(grid, '14.0', '15', price_zone_link=False, lf=0.0, qf=0.0, MWmax=0.0, MWmin=0.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=0.0, QsetMVA=75.0)
    pyf.add_gen(grid, '15.0', '16', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=7.200000000000001, QsetMVA=3.0)
    pyf.add_gen(grid, '15.0', '17', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=7.200000000000001, QsetMVA=3.0)
    pyf.add_gen(grid, '15.0', '18', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=7.200000000000001, QsetMVA=3.0)
    pyf.add_gen(grid, '15.0', '19', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=7.200000000000001, QsetMVA=3.0)
    pyf.add_gen(grid, '15.0', '20', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=7.200000000000001, QsetMVA=3.0)
    pyf.add_gen(grid, '15.0', '21', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=104.65, QsetMVA=15.0)
    pyf.add_gen(grid, '16.0', '22', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=104.65, QsetMVA=15.0)
    pyf.add_gen(grid, '18.0', '23', price_zone_link=False, lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=250.0, QsetMVA=75.0)
    pyf.add_gen(grid, '21.0', '24', price_zone_link=False, lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=250.0, QsetMVA=75.0)
    pyf.add_gen(grid, '22.0', '25', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=30.0, QsetMVA=3.0)
    pyf.add_gen(grid, '22.0', '26', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=30.0, QsetMVA=3.0)
    pyf.add_gen(grid, '22.0', '27', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=30.0, QsetMVA=3.0)
    pyf.add_gen(grid, '22.0', '28', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=30.0, QsetMVA=3.0)
    pyf.add_gen(grid, '22.0', '29', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=30.0, QsetMVA=3.0)
    pyf.add_gen(grid, '22.0', '30', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=30.0, QsetMVA=3.0)
    pyf.add_gen(grid, '23.0', '31', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=104.65, QsetMVA=15.0)
    pyf.add_gen(grid, '23.0', '32', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=104.65, QsetMVA=15.0)
    pyf.add_gen(grid, '23.0', '33', price_zone_link=False, lf=11.8495, qf=0.004895, MWmax=350.0, MWmin=140.0, MVArmax=150.0, MVArmin=-25.0, PsetMW=245.00000000000003, QsetMVA=62.5)
    
    
    # Add Renewable Source Zones

    
    # Add Renewable Sources

    
    # Return the grid
    return grid,res
