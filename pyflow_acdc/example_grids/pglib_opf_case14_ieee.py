

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
%   Power flow data for IEEE 14 bus test case.
%   This data was converted from IEEE Common Data Format
%   (ieee14cdf.txt) on 20-Sep-2004 by cdf2matp, rev. 1.11
%
%   Converted from IEEE CDF file from:
%        http://www.ee.washington.edu/research/pstca/
%
%   Copyright (c) 1999 by Richard D. Christie, University of Washington
%   Electrical Engineering Licensed under the Creative Commons Attribution 4.0
%   International license, http://creativecommons.org/licenses/by/4.0/
%
%   CDF Header:
%   08/19/93 UW ARCHIVE           100.0  1962 W IEEE 14 Bus Test Case
%
"""

def pglib_opf_case14_ieee():    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '1.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.217, 'Reactive_load': 0.127, 'Node_id': '2.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.9420000000000001, 'Reactive_load': 0.19, 'Node_id': '3.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.478, 'Reactive_load': -0.039, 'Node_id': '4.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.076, 'Reactive_load': 0.016, 'Node_id': '5.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.11199999999999999, 'Reactive_load': 0.075, 'Node_id': '6.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '7.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '8.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.295, 'Reactive_load': 0.166, 'Node_id': '9.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.19, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.09, 'Reactive_load': 0.057999999999999996, 'Node_id': '10.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.035, 'Reactive_load': 0.018000000000000002, 'Node_id': '11.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.061, 'Reactive_load': 0.016, 'Node_id': '12.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.135, 'Reactive_load': 0.057999999999999996, 'Node_id': '13.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 1.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.149, 'Reactive_load': 0.05, 'Node_id': '14.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': '1.0', 'toNode': '2.0', 'Resistance': 0.01938, 'Reactance': 0.05917, 'Conductance': 0, 'Susceptance': 0.0528, 'MVA_rating': 472.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '1'},
        {'fromNode': '1.0', 'toNode': '5.0', 'Resistance': 0.05403, 'Reactance': 0.22304, 'Conductance': 0, 'Susceptance': 0.0492, 'MVA_rating': 128.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '2'},
        {'fromNode': '2.0', 'toNode': '3.0', 'Resistance': 0.04699, 'Reactance': 0.19797, 'Conductance': 0, 'Susceptance': 0.0438, 'MVA_rating': 145.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '3'},
        {'fromNode': '2.0', 'toNode': '4.0', 'Resistance': 0.05811, 'Reactance': 0.17632, 'Conductance': 0, 'Susceptance': 0.034, 'MVA_rating': 158.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '4'},
        {'fromNode': '2.0', 'toNode': '5.0', 'Resistance': 0.05695, 'Reactance': 0.17388, 'Conductance': 0, 'Susceptance': 0.0346, 'MVA_rating': 161.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '5'},
        {'fromNode': '3.0', 'toNode': '4.0', 'Resistance': 0.06701, 'Reactance': 0.17103, 'Conductance': 0, 'Susceptance': 0.0128, 'MVA_rating': 160.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '6'},
        {'fromNode': '4.0', 'toNode': '5.0', 'Resistance': 0.01335, 'Reactance': 0.04211, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 664.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '7'},
        {'fromNode': '4.0', 'toNode': '7.0', 'Resistance': 0.0, 'Reactance': 0.20912, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 141.0, 'kV_base': 1.0, 'm': 0.978, 'shift': 0.0, 'Line_id': '8'},
        {'fromNode': '4.0', 'toNode': '9.0', 'Resistance': 0.0, 'Reactance': 0.55618, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 53.0, 'kV_base': 1.0, 'm': 0.969, 'shift': 0.0, 'Line_id': '9'},
        {'fromNode': '5.0', 'toNode': '6.0', 'Resistance': 0.0, 'Reactance': 0.25202, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 117.0, 'kV_base': 1.0, 'm': 0.932, 'shift': 0.0, 'Line_id': '10'},
        {'fromNode': '6.0', 'toNode': '11.0', 'Resistance': 0.09498, 'Reactance': 0.1989, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 134.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '11'},
        {'fromNode': '6.0', 'toNode': '12.0', 'Resistance': 0.12291, 'Reactance': 0.25581, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 104.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '12'},
        {'fromNode': '6.0', 'toNode': '13.0', 'Resistance': 0.06615, 'Reactance': 0.13027, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 201.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '13'},
        {'fromNode': '7.0', 'toNode': '8.0', 'Resistance': 0.0, 'Reactance': 0.17615, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 167.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '14'},
        {'fromNode': '7.0', 'toNode': '9.0', 'Resistance': 0.0, 'Reactance': 0.11001, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 267.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '15'},
        {'fromNode': '9.0', 'toNode': '10.0', 'Resistance': 0.03181, 'Reactance': 0.0845, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 325.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '16'},
        {'fromNode': '9.0', 'toNode': '14.0', 'Resistance': 0.12711, 'Reactance': 0.27038, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 99.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '17'},
        {'fromNode': '10.0', 'toNode': '11.0', 'Resistance': 0.08205, 'Reactance': 0.19207, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 141.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '18'},
        {'fromNode': '12.0', 'toNode': '13.0', 'Resistance': 0.22092, 'Reactance': 0.19988, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 99.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '19'},
        {'fromNode': '13.0', 'toNode': '14.0', 'Resistance': 0.17093, 'Reactance': 0.34802, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 76.0, 'kV_base': 1.0, 'm': 1, 'shift': 0, 'Line_id': '20'}
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
    pyf.add_gen(grid, '1.0', '1', price_zone_link=False, lf=7.920951, qf=0.0, MWmax=340.0, MWmin=0.0, MVArmax=10.0, MVArmin=0.0, PsetMW=170.0, QsetMVA=5.0)
    pyf.add_gen(grid, '2.0', '2', price_zone_link=False, lf=23.269494, qf=0.0, MWmax=59.0, MWmin=0.0, MVArmax=30.0, MVArmin=-30.0, PsetMW=29.5, QsetMVA=0.0)
    pyf.add_gen(grid, '3.0', '3', price_zone_link=False, lf=0.0, qf=0.0, MWmax=0.0, MWmin=0.0, MVArmax=40.0, MVArmin=0.0, PsetMW=0.0, QsetMVA=20.0)
    pyf.add_gen(grid, '6.0', '4', price_zone_link=False, lf=0.0, qf=0.0, MWmax=0.0, MWmin=0.0, MVArmax=24.0, MVArmin=-6.0, PsetMW=0.0, QsetMVA=9.0)
    pyf.add_gen(grid, '8.0', '5', price_zone_link=False, lf=0.0, qf=0.0, MWmax=0.0, MWmin=0.0, MVArmax=24.0, MVArmin=-6.0, PsetMW=0.0, QsetMVA=9.0)
    
    
    # Add Renewable Source Zones

    
    # Add Renewable Sources

    
    # Return the grid
    return grid,res
