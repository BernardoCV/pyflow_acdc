

import pyflow_acdc as pyf
import pandas as pd

"""
Converted to PyFlowACDC format from
%CASE24_IEEE_RTS1996_3zones  Power flow data for system based on the
%IEEE RELIABILITY TEST SYSTEM.
%   Please see CASEFORMAT for details on the case file format.
%
%   This system data is based on the MATPOWER case file CASE24_IEEE_RTS
%   which is based on the IEEE RELIABILITY TEST SYSTEM
%
%   The data has been adopted to corresponding with the
%   IEEE Two Area RTS-96 data from...
%   IEEE Reliability Test System Task Force of Applications of
%   Probability Methods Subcommittee, "IEEE reliability test system-96,"
%   IEEE Transactions on Power Systems, Vol. 14, No. 3, Aug. 1999,
%   pp. 1010-1020.
%
%   The IEEE Two Area RTS-96 network has been extended and now includes 3
%   asynchronous zones (node numbers 1xx, 2xx and 3xx).
%   Data on zone 1 and 2 taken from the IEEE Two Area RTS-96 with following
%   adaptations:
%   - nodes renumbered according to IEEE Two Area RTS-96 data
%   - gen U100 at node 107 disabled (commented)
%   - gen U76 at node 201 disabled (commented)
%   - slack node zone 2: node 213
%   - lines 107-203, 113-215, 123-217 removed (commented)
%   Data on zone 3 added:
%   - nodes 301 and 302
%   - gen at node 302
%   - line 301-302
%
%   MATPOWER case file data provided by Bruce Wollenberg
%   (MATPOWER file case24_ieee_rts.m) and adapted for use with MatACDC
%   by Jef Beerten.

generator C(0) modified to 0

"""

def case24_3zones_acdc():    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.08, 'Reactive_load': 0.22, 'Node_id': '101.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.97, 'Reactive_load': 0.2, 'Node_id': '102.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.8, 'Reactive_load': 0.37, 'Node_id': '103.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.74, 'Reactive_load': 0.15, 'Node_id': '104.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.71, 'Reactive_load': 0.14, 'Node_id': '105.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.36, 'Reactive_load': 0.28, 'Node_id': '106.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.01, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.25, 'Reactive_load': 0.25, 'Node_id': '107.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.71, 'Reactive_load': 0.35, 'Node_id': '108.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.75, 'Reactive_load': 0.36, 'Node_id': '109.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.95, 'Reactive_load': 0.4, 'Node_id': '110.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '111.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '112.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 1.02, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.65, 'Reactive_load': 0.54, 'Node_id': '113.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.94, 'Reactive_load': 0.39, 'Node_id': '114.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.014, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.17, 'Reactive_load': 0.64, 'Node_id': '115.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.0, 'Reactive_load': 0.2, 'Node_id': '116.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '117.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.33, 'Reactive_load': 0.68, 'Node_id': '118.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.81, 'Reactive_load': 0.37, 'Node_id': '119.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.28, 'Reactive_load': 0.26, 'Node_id': '120.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '121.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '122.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '123.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '124.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.08, 'Reactive_load': 0.22, 'Node_id': '201.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.97, 'Reactive_load': 0.2, 'Node_id': '202.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.8, 'Reactive_load': 0.37, 'Node_id': '203.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.74, 'Reactive_load': 0.15, 'Node_id': '204.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.71, 'Reactive_load': 0.14, 'Node_id': '205.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.36, 'Reactive_load': 0.28, 'Node_id': '206.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.01, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.25, 'Reactive_load': 0.25, 'Node_id': '207.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.71, 'Reactive_load': 0.35, 'Node_id': '208.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.75, 'Reactive_load': 0.36, 'Node_id': '209.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.95, 'Reactive_load': 0.4, 'Node_id': '210.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '211.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '212.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 1.02, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.65, 'Reactive_load': 0.54, 'Node_id': '213.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.94, 'Reactive_load': 0.39, 'Node_id': '214.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.014, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.17, 'Reactive_load': 0.64, 'Node_id': '215.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.0, 'Reactive_load': 0.2, 'Node_id': '216.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '217.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.33, 'Reactive_load': 0.68, 'Node_id': '218.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.81, 'Reactive_load': 0.37, 'Node_id': '219.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.28, 'Reactive_load': 0.26, 'Node_id': '220.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '221.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '222.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '223.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '224.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '301.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '302.0', 'Umin': 0.95, 'Umax': 1.05, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': '101.0', 'toNode': '102.0', 'Resistance': 0.003, 'Reactance': 0.014, 'Conductance': 0, 'Susceptance': 0.461, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '1'},
        {'fromNode': '101.0', 'toNode': '103.0', 'Resistance': 0.055, 'Reactance': 0.211, 'Conductance': 0, 'Susceptance': 0.057, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '2'},
        {'fromNode': '101.0', 'toNode': '105.0', 'Resistance': 0.022, 'Reactance': 0.085, 'Conductance': 0, 'Susceptance': 0.023, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '3'},
        {'fromNode': '102.0', 'toNode': '104.0', 'Resistance': 0.033, 'Reactance': 0.127, 'Conductance': 0, 'Susceptance': 0.034, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '4'},
        {'fromNode': '102.0', 'toNode': '106.0', 'Resistance': 0.05, 'Reactance': 0.192, 'Conductance': 0, 'Susceptance': 0.052, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '5'},
        {'fromNode': '103.0', 'toNode': '109.0', 'Resistance': 0.031, 'Reactance': 0.119, 'Conductance': 0, 'Susceptance': 0.032, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '6'},
        {'fromNode': '103.0', 'toNode': '124.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015, 'shift': 0.0, 'Line_id': '7'},
        {'fromNode': '104.0', 'toNode': '109.0', 'Resistance': 0.027, 'Reactance': 0.104, 'Conductance': 0, 'Susceptance': 0.028, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '8'},
        {'fromNode': '105.0', 'toNode': '110.0', 'Resistance': 0.022, 'Reactance': 0.088, 'Conductance': 0, 'Susceptance': 0.024, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '9'},
        {'fromNode': '106.0', 'toNode': '110.0', 'Resistance': 0.014, 'Reactance': 0.061, 'Conductance': 0, 'Susceptance': 2.459, 'MVA_rating': 400.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '10'},
        {'fromNode': '107.0', 'toNode': '108.0', 'Resistance': 0.016, 'Reactance': 0.061, 'Conductance': 0, 'Susceptance': 0.017, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '11'},
        {'fromNode': '108.0', 'toNode': '109.0', 'Resistance': 0.043, 'Reactance': 0.165, 'Conductance': 0, 'Susceptance': 0.045, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '12'},
        {'fromNode': '108.0', 'toNode': '110.0', 'Resistance': 0.043, 'Reactance': 0.165, 'Conductance': 0, 'Susceptance': 0.045, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '13'},
        {'fromNode': '109.0', 'toNode': '111.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '14'},
        {'fromNode': '109.0', 'toNode': '112.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '15'},
        {'fromNode': '110.0', 'toNode': '111.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015, 'shift': 0.0, 'Line_id': '16'},
        {'fromNode': '110.0', 'toNode': '112.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015, 'shift': 0.0, 'Line_id': '17'},
        {'fromNode': '111.0', 'toNode': '113.0', 'Resistance': 0.006, 'Reactance': 0.048, 'Conductance': 0, 'Susceptance': 0.1, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '18'},
        {'fromNode': '111.0', 'toNode': '114.0', 'Resistance': 0.005, 'Reactance': 0.042, 'Conductance': 0, 'Susceptance': 0.088, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '19'},
        {'fromNode': '112.0', 'toNode': '113.0', 'Resistance': 0.006, 'Reactance': 0.048, 'Conductance': 0, 'Susceptance': 0.1, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '20'},
        {'fromNode': '112.0', 'toNode': '123.0', 'Resistance': 0.012, 'Reactance': 0.097, 'Conductance': 0, 'Susceptance': 0.203, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '21'},
        {'fromNode': '113.0', 'toNode': '123.0', 'Resistance': 0.011, 'Reactance': 0.087, 'Conductance': 0, 'Susceptance': 0.182, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '22'},
        {'fromNode': '114.0', 'toNode': '116.0', 'Resistance': 0.005, 'Reactance': 0.059, 'Conductance': 0, 'Susceptance': 0.082, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '23'},
        {'fromNode': '115.0', 'toNode': '116.0', 'Resistance': 0.002, 'Reactance': 0.017, 'Conductance': 0, 'Susceptance': 0.036, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '24'},
        {'fromNode': '115.0', 'toNode': '121.0', 'Resistance': 0.006, 'Reactance': 0.049, 'Conductance': 0, 'Susceptance': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '25'},
        {'fromNode': '115.0', 'toNode': '121.0', 'Resistance': 0.006, 'Reactance': 0.049, 'Conductance': 0, 'Susceptance': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '26'},
        {'fromNode': '115.0', 'toNode': '124.0', 'Resistance': 0.007, 'Reactance': 0.052, 'Conductance': 0, 'Susceptance': 0.109, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '27'},
        {'fromNode': '116.0', 'toNode': '117.0', 'Resistance': 0.003, 'Reactance': 0.026, 'Conductance': 0, 'Susceptance': 0.055, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '28'},
        {'fromNode': '116.0', 'toNode': '119.0', 'Resistance': 0.003, 'Reactance': 0.023, 'Conductance': 0, 'Susceptance': 0.049, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '29'},
        {'fromNode': '117.0', 'toNode': '118.0', 'Resistance': 0.002, 'Reactance': 0.014, 'Conductance': 0, 'Susceptance': 0.03, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '30'},
        {'fromNode': '117.0', 'toNode': '122.0', 'Resistance': 0.014, 'Reactance': 0.105, 'Conductance': 0, 'Susceptance': 0.221, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '31'},
        {'fromNode': '118.0', 'toNode': '121.0', 'Resistance': 0.003, 'Reactance': 0.026, 'Conductance': 0, 'Susceptance': 0.055, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '32'},
        {'fromNode': '118.0', 'toNode': '121.0', 'Resistance': 0.003, 'Reactance': 0.026, 'Conductance': 0, 'Susceptance': 0.055, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '33'},
        {'fromNode': '119.0', 'toNode': '120.0', 'Resistance': 0.005, 'Reactance': 0.04, 'Conductance': 0, 'Susceptance': 0.083, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '34'},
        {'fromNode': '119.0', 'toNode': '120.0', 'Resistance': 0.005, 'Reactance': 0.04, 'Conductance': 0, 'Susceptance': 0.083, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '35'},
        {'fromNode': '120.0', 'toNode': '123.0', 'Resistance': 0.003, 'Reactance': 0.022, 'Conductance': 0, 'Susceptance': 0.046, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '36'},
        {'fromNode': '120.0', 'toNode': '123.0', 'Resistance': 0.003, 'Reactance': 0.022, 'Conductance': 0, 'Susceptance': 0.046, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '37'},
        {'fromNode': '121.0', 'toNode': '122.0', 'Resistance': 0.009, 'Reactance': 0.068, 'Conductance': 0, 'Susceptance': 0.142, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '38'},
        {'fromNode': '201.0', 'toNode': '202.0', 'Resistance': 0.003, 'Reactance': 0.014, 'Conductance': 0, 'Susceptance': 0.461, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '39'},
        {'fromNode': '201.0', 'toNode': '203.0', 'Resistance': 0.055, 'Reactance': 0.211, 'Conductance': 0, 'Susceptance': 0.057, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '40'},
        {'fromNode': '201.0', 'toNode': '205.0', 'Resistance': 0.022, 'Reactance': 0.085, 'Conductance': 0, 'Susceptance': 0.023, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '41'},
        {'fromNode': '202.0', 'toNode': '204.0', 'Resistance': 0.033, 'Reactance': 0.127, 'Conductance': 0, 'Susceptance': 0.034, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '42'},
        {'fromNode': '202.0', 'toNode': '206.0', 'Resistance': 0.05, 'Reactance': 0.192, 'Conductance': 0, 'Susceptance': 0.052, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '43'},
        {'fromNode': '203.0', 'toNode': '209.0', 'Resistance': 0.031, 'Reactance': 0.119, 'Conductance': 0, 'Susceptance': 0.032, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '44'},
        {'fromNode': '203.0', 'toNode': '224.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015, 'shift': 0.0, 'Line_id': '45'},
        {'fromNode': '204.0', 'toNode': '209.0', 'Resistance': 0.027, 'Reactance': 0.104, 'Conductance': 0, 'Susceptance': 0.028, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '46'},
        {'fromNode': '205.0', 'toNode': '210.0', 'Resistance': 0.022, 'Reactance': 0.088, 'Conductance': 0, 'Susceptance': 0.024, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '47'},
        {'fromNode': '206.0', 'toNode': '210.0', 'Resistance': 0.014, 'Reactance': 0.061, 'Conductance': 0, 'Susceptance': 2.459, 'MVA_rating': 400.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '48'},
        {'fromNode': '207.0', 'toNode': '208.0', 'Resistance': 0.016, 'Reactance': 0.061, 'Conductance': 0, 'Susceptance': 0.017, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '49'},
        {'fromNode': '208.0', 'toNode': '209.0', 'Resistance': 0.043, 'Reactance': 0.165, 'Conductance': 0, 'Susceptance': 0.045, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '50'},
        {'fromNode': '208.0', 'toNode': '210.0', 'Resistance': 0.043, 'Reactance': 0.165, 'Conductance': 0, 'Susceptance': 0.045, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1, 'shift': 0, 'Line_id': '51'},
        {'fromNode': '209.0', 'toNode': '211.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '52'},
        {'fromNode': '209.0', 'toNode': '212.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03, 'shift': 0.0, 'Line_id': '53'},
        {'fromNode': '210.0', 'toNode': '211.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015, 'shift': 0.0, 'Line_id': '54'},
        {'fromNode': '210.0', 'toNode': '212.0', 'Resistance': 0.002, 'Reactance': 0.084, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015, 'shift': 0.0, 'Line_id': '55'},
        {'fromNode': '211.0', 'toNode': '213.0', 'Resistance': 0.006, 'Reactance': 0.048, 'Conductance': 0, 'Susceptance': 0.1, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '56'},
        {'fromNode': '211.0', 'toNode': '214.0', 'Resistance': 0.005, 'Reactance': 0.042, 'Conductance': 0, 'Susceptance': 0.088, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '57'},
        {'fromNode': '212.0', 'toNode': '213.0', 'Resistance': 0.006, 'Reactance': 0.048, 'Conductance': 0, 'Susceptance': 0.1, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '58'},
        {'fromNode': '212.0', 'toNode': '223.0', 'Resistance': 0.012, 'Reactance': 0.097, 'Conductance': 0, 'Susceptance': 0.203, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '59'},
        {'fromNode': '213.0', 'toNode': '223.0', 'Resistance': 0.011, 'Reactance': 0.087, 'Conductance': 0, 'Susceptance': 0.182, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '60'},
        {'fromNode': '214.0', 'toNode': '216.0', 'Resistance': 0.005, 'Reactance': 0.059, 'Conductance': 0, 'Susceptance': 0.082, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '61'},
        {'fromNode': '215.0', 'toNode': '216.0', 'Resistance': 0.002, 'Reactance': 0.017, 'Conductance': 0, 'Susceptance': 0.036, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '62'},
        {'fromNode': '215.0', 'toNode': '221.0', 'Resistance': 0.006, 'Reactance': 0.049, 'Conductance': 0, 'Susceptance': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '63'},
        {'fromNode': '215.0', 'toNode': '221.0', 'Resistance': 0.006, 'Reactance': 0.049, 'Conductance': 0, 'Susceptance': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '64'},
        {'fromNode': '215.0', 'toNode': '224.0', 'Resistance': 0.007, 'Reactance': 0.052, 'Conductance': 0, 'Susceptance': 0.109, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '65'},
        {'fromNode': '216.0', 'toNode': '217.0', 'Resistance': 0.003, 'Reactance': 0.026, 'Conductance': 0, 'Susceptance': 0.055, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '66'},
        {'fromNode': '216.0', 'toNode': '219.0', 'Resistance': 0.003, 'Reactance': 0.023, 'Conductance': 0, 'Susceptance': 0.049, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '67'},
        {'fromNode': '217.0', 'toNode': '218.0', 'Resistance': 0.002, 'Reactance': 0.014, 'Conductance': 0, 'Susceptance': 0.03, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '68'},
        {'fromNode': '217.0', 'toNode': '222.0', 'Resistance': 0.014, 'Reactance': 0.105, 'Conductance': 0, 'Susceptance': 0.221, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '69'},
        {'fromNode': '218.0', 'toNode': '221.0', 'Resistance': 0.003, 'Reactance': 0.026, 'Conductance': 0, 'Susceptance': 0.055, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '70'},
        {'fromNode': '218.0', 'toNode': '221.0', 'Resistance': 0.003, 'Reactance': 0.026, 'Conductance': 0, 'Susceptance': 0.055, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '71'},
        {'fromNode': '219.0', 'toNode': '220.0', 'Resistance': 0.005, 'Reactance': 0.04, 'Conductance': 0, 'Susceptance': 0.083, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '72'},
        {'fromNode': '219.0', 'toNode': '220.0', 'Resistance': 0.005, 'Reactance': 0.04, 'Conductance': 0, 'Susceptance': 0.083, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '73'},
        {'fromNode': '220.0', 'toNode': '223.0', 'Resistance': 0.003, 'Reactance': 0.022, 'Conductance': 0, 'Susceptance': 0.046, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '74'},
        {'fromNode': '220.0', 'toNode': '223.0', 'Resistance': 0.003, 'Reactance': 0.022, 'Conductance': 0, 'Susceptance': 0.046, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '75'},
        {'fromNode': '221.0', 'toNode': '222.0', 'Resistance': 0.009, 'Reactance': 0.068, 'Conductance': 0, 'Susceptance': 0.142, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '76'},
        {'fromNode': '301.0', 'toNode': '302.0', 'Resistance': 0.0, 'Reactance': 0.001, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1, 'shift': 0, 'Line_id': '77'}
    ]
    lines_AC = pd.DataFrame(lines_AC_data)

    nodes_DC_data = [
        {'type': 'Slack', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 150.0, 'Node_id': '1.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 150.0, 'Node_id': '2.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 150.0, 'Node_id': '3.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 300.0, 'Node_id': '4.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 300.0, 'Node_id': '5.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 300.0, 'Node_id': '6.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.0, 'Power_Gained': 0, 'Power_load': 0.0, 'kV_base': 300.0, 'Node_id': '7.0', 'Umin': 0.9, 'Umax': 1.1, 'x_coord': None, 'y_coord': None, 'PZ': None}
    ]
    nodes_DC = pd.DataFrame(nodes_DC_data)

    lines_DC_data = [
        {'fromNode': '1.0', 'toNode': '3.0', 'Resistance': 0.0352, 'MW_rating': 100.0, 'kV_base': 150.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '1'},
        {'fromNode': '2.0', 'toNode': '3.0', 'Resistance': 0.0352, 'MW_rating': 100.0, 'kV_base': 150.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '2'},
        {'fromNode': '4.0', 'toNode': '5.0', 'Resistance': 0.0828, 'MW_rating': 100.0, 'kV_base': 300.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '3'},
        {'fromNode': '4.0', 'toNode': '7.0', 'Resistance': 0.0704, 'MW_rating': 100.0, 'kV_base': 300.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '4'},
        {'fromNode': '4.0', 'toNode': '6.0', 'Resistance': 0.0718, 'MW_rating': 100.0, 'kV_base': 300.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '5'},
        {'fromNode': '5.0', 'toNode': '7.0', 'Resistance': 0.076, 'MW_rating': 100.0, 'kV_base': 300.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '6'},
        {'fromNode': '6.0', 'toNode': '7.0', 'Resistance': 0.0248, 'MW_rating': 100.0, 'kV_base': 300.0, 'Length_km': 1, 'Mono_Bi_polar': 'b', 'Line_id': '7'}
    ]
    lines_DC = pd.DataFrame(lines_DC_data)

    Converters_ACDC_data = [
        {'AC_type': 'PQ', 'DC_type': 'Slack', 'AC_node': '107.0', 'DC_node': '1.0', 'P_AC': 0.0, 'Q_AC': 0.5, 'P_DC': 0.0, 'T_R': 0.001, 'T_X': 0.1, 'PR_R': 0.0001, 'PR_X': 0.16, 'Filter': 0.09, 'Droop': 0.0, 'AC_kV_base': 138.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '1', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.9, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': '204.0', 'DC_node': '2.0', 'P_AC': 0.753, 'Q_AC': -0.5, 'P_DC': 0.0, 'T_R': 0.001, 'T_X': 0.1, 'PR_R': 0.0001, 'PR_X': 0.16, 'Filter': 0.09, 'Droop': 0.0, 'AC_kV_base': 138.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '2', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.9, 'Ucmax': 1.2},
        {'AC_type': 'PQ', 'DC_type': 'P', 'AC_node': '301.0', 'DC_node': '3.0', 'P_AC': -1.419, 'Q_AC': 1.3, 'P_DC': 0.0, 'T_R': 0.001, 'T_X': 0.05, 'PR_R': 0.0001, 'PR_X': 0.08, 'Filter': 0.045, 'Droop': 0.0, 'AC_kV_base': 138.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '3', 'lossa': 2.206, 'lossb': 0.887, 'losscrect': 1.442, 'losscinv': 2.185, 'Ucmin': 0.9, 'Ucmax': 1.2},
        {'AC_type': 'PQ', 'DC_type': 'Slack', 'AC_node': '113.0', 'DC_node': '4.0', 'P_AC': 1.315, 'Q_AC': 0.759, 'P_DC': 0.0, 'T_R': 0.0005, 'T_X': 0.05, 'PR_R': 0.0001, 'PR_X': 0.08, 'Filter': 0.0, 'Droop': 0.0, 'AC_kV_base': 345.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '4', 'lossa': 2.206, 'lossb': 1.8, 'losscrect': 5.94, 'losscinv': 9.0, 'Ucmin': 0.5, 'Ucmax': 1.2},
        {'AC_type': 'PQ', 'DC_type': 'P', 'AC_node': '123.0', 'DC_node': '5.0', 'P_AC': -0.617, 'Q_AC': 0.0, 'P_DC': 0.0, 'T_R': 0.001, 'T_X': 0.1, 'PR_R': 0.0001, 'PR_X': 0.16, 'Filter': 0.0, 'Droop': 0.0, 'AC_kV_base': 345.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '5', 'lossa': 1.103, 'lossb': 1.8, 'losscrect': 11.88, 'losscinv': 18.0, 'Ucmin': 0.5, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': '215.0', 'DC_node': '6.0', 'P_AC': -1.234, 'Q_AC': -0.1, 'P_DC': 0.0, 'T_R': 0.0005, 'T_X': 0.05, 'PR_R': 0.0001, 'PR_X': 0.08, 'Filter': 0.0, 'Droop': 0.0, 'AC_kV_base': 345.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '6', 'lossa': 2.206, 'lossb': 1.8, 'losscrect': 5.94, 'losscinv': 9.0, 'Ucmin': 0.5, 'Ucmax': 1.2},
        {'AC_type': 'PQ', 'DC_type': 'P', 'AC_node': '217.0', 'DC_node': '7.0', 'P_AC': 0.5, 'Q_AC': 0.2, 'P_DC': 0.0, 'T_R': 0.001, 'T_X': 0.1, 'PR_R': 0.0001, 'PR_X': 0.16, 'Filter': 0.0, 'Droop': 0.0, 'AC_kV_base': 345.0, 'MVA_rating': 200.0, 'Nconverter': 1, 'pol': 1, 'Conv_id': '7', 'lossa': 1.103, 'lossb': 1.8, 'losscrect': 11.88, 'losscinv': 18.0, 'Ucmin': 0.5, 'Ucmax': 1.2}
    ]
    Converters_ACDC = pd.DataFrame(Converters_ACDC_data)

    
    # Create the grid
    [grid, res] = pyf.Create_grid_from_data(S_base, nodes_AC, lines_AC, nodes_DC, lines_DC, Converters_ACDC, data_in_pu=True)
    
    # Assign Price Zones to Nodes
    for index, row in nodes_AC.iterrows():
        node_name = nodes_AC.at[index, 'Node_id']
        price_zone = nodes_AC.at[index, 'PZ']
        ACDC = 'AC'
        if price_zone is not None:
            pyf.assign_nodeToPrice_Zone(grid, node_name, ACDC, price_zone)
    
    for index, row in nodes_DC.iterrows():
        node_name = nodes_DC.at[index, 'Node_id']
        price_zone = nodes_DC.at[index, 'PZ']
        ACDC = 'DC'
        if price_zone is not None:
            pyf.assign_nodeToPrice_Zone(grid, node_name, ACDC, price_zone)
    
    # Add Generators
    pyf.add_gen(grid, '101.0', '1', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '101.0', '2', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '101.0', '3', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=14.099999999999998)
    pyf.add_gen(grid, '101.0', '4', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=14.099999999999998)
    pyf.add_gen(grid, '102.0', '5', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '102.0', '6', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '102.0', '7', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=7.000000000000001)
    pyf.add_gen(grid, '102.0', '8', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=7.000000000000001)
    pyf.add_gen(grid, '107.0', '9', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=17.2)
    pyf.add_gen(grid, '107.0', '10', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=17.2)
    pyf.add_gen(grid, '113.0', '11', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=40.7)
    pyf.add_gen(grid, '113.0', '12', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=40.7)
    pyf.add_gen(grid, '113.0', '13', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=40.7)
    pyf.add_gen(grid, '114.0', '14', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=0.0, MWmin=0.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=0.0, QsetMVA=13.699999999999998)
    pyf.add_gen(grid, '115.0', '15', price_zone_link=False, lf=0.0, qf=0.0, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '115.0', '16', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '115.0', '17', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '115.0', '18', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '115.0', '19', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '115.0', '20', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.05)
    pyf.add_gen(grid, '116.0', '21', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=25.22)
    pyf.add_gen(grid, '118.0', '22', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=137.4)
    pyf.add_gen(grid, '121.0', '23', price_zone_link=False, lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=108.2)
    pyf.add_gen(grid, '122.0', '24', price_zone_link=False, lf=4.4231, qf=0.000213, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '122.0', '25', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '122.0', '26', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '122.0', '27', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '122.0', '28', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '122.0', '29', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '123.0', '30', price_zone_link=False, lf=0.001, qf=0.0, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=31.790000000000003)
    pyf.add_gen(grid, '123.0', '31', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=31.790000000000003)
    pyf.add_gen(grid, '123.0', '32', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=350.0, MWmin=140.0, MVArmax=150.0, MVArmin=-25.0, PsetMW=350.0, QsetMVA=71.78)
    pyf.add_gen(grid, '201.0', '33', price_zone_link=False, lf=11.8495, qf=0.004895, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '201.0', '34', price_zone_link=False, lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '201.0', '35', price_zone_link=False, lf=130.0, qf=0.0, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=14.099999999999998)
    pyf.add_gen(grid, '202.0', '36', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '202.0', '37', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    pyf.add_gen(grid, '202.0', '38', price_zone_link=False, lf=130.0, qf=0.0, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=7.000000000000001)
    pyf.add_gen(grid, '202.0', '39', price_zone_link=False, lf=130.0, qf=0.0, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=7.000000000000001)
    pyf.add_gen(grid, '207.0', '40', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=17.2)
    pyf.add_gen(grid, '207.0', '41', price_zone_link=False, lf=16.0811, qf=0.014142, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=17.2)
    pyf.add_gen(grid, '207.0', '42', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=17.2)
    pyf.add_gen(grid, '213.0', '43', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=40.7)
    pyf.add_gen(grid, '213.0', '44', price_zone_link=False, lf=43.6615, qf=0.052672, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=40.7)
    pyf.add_gen(grid, '213.0', '45', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=40.7)
    pyf.add_gen(grid, '214.0', '46', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=0.0, MWmin=0.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=0.0, QsetMVA=13.68)
    pyf.add_gen(grid, '215.0', '47', price_zone_link=False, lf=48.5804, qf=0.00717, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '215.0', '48', price_zone_link=False, lf=0.0, qf=0.0, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '215.0', '49', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '215.0', '50', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '215.0', '51', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    pyf.add_gen(grid, '215.0', '52', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.048)
    pyf.add_gen(grid, '216.0', '53', price_zone_link=False, lf=56.564, qf=0.328412, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=25.22)
    pyf.add_gen(grid, '218.0', '54', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=137.4)
    pyf.add_gen(grid, '221.0', '55', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=108.2)
    pyf.add_gen(grid, '222.0', '56', price_zone_link=False, lf=4.4231, qf=0.000213, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '222.0', '57', price_zone_link=False, lf=4.4231, qf=0.000213, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '222.0', '58', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '222.0', '59', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '222.0', '60', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '222.0', '61', price_zone_link=False, lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=-4.96)
    pyf.add_gen(grid, '223.0', '62', price_zone_link=False, lf=0.001, qf=0.0, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=31.790000000000003)
    pyf.add_gen(grid, '223.0', '63', price_zone_link=False, lf=0.001, qf=0.0, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=31.790000000000003)
    pyf.add_gen(grid, '223.0', '64', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=350.0, MWmin=140.0, MVArmax=150.0, MVArmin=-25.0, PsetMW=350.0, QsetMVA=71.78)
    pyf.add_gen(grid, '302.0', '65', price_zone_link=False, lf=12.3883, qf=0.008342, MWmax=350.0, MWmin=140.0, MVArmax=150.0, MVArmin=-25.0, PsetMW=150.0, QsetMVA=10.0)
    
    
    # Add Renewable Source Zones

    
    # Add Renewable Sources

    
    # Return the grid
    return grid,res
