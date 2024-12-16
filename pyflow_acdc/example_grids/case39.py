

import pyflow_acdc as pyf
import pandas as pd
"""
Converted to PyFlowACDC format from

%CASE39 Power flow data for 39 bus New England system.
%   Please see CASEFORMAT for details on the case file format.
%
%   Data taken from [1] with the following modifications/additions:
%
%       - renumbered gen buses consecutively (as in [2] and [4])
%       - added Pmin = 0 for all gens
%       - added Qmin, Qmax for gens at 31 & 39 (copied from gen at 35)
%       - added Vg based on V in bus data (missing for bus 39)
%       - added Vg, Pg, Pd, Qd at bus 39 from [2] (same in [4])
%       - added Pmax at bus 39: Pmax = Pg + 100
%       - added line flow limits and area data from [4]
%       - added voltage limits, Vmax = 1.06, Vmin = 0.94
%       - added identical quadratic generator costs
%       - increased Pmax for gen at bus 34 from 308 to 508
%         (assumed typo in [1], makes initial solved case feasible)
%       - re-solved power flow
% 
%   Notes:
%       - Bus 39, its generator and 2 connecting lines were added
%         (by authors of [1]) to represent the interconnection with
%         the rest of the eastern interconnect, and did not include
%         Vg, Pg, Qg, Pd, Qd, Pmin, Pmax, Qmin or Qmax.
%       - As the swing bus, bus 31 did not include and Q limits.
%       - The voltages, etc in [1] appear to be quite close to the
%         power flow solution of the case before adding bus 39 with
%         it's generator and connecting branches, though the solution
%         is not exact.
%       - Explicit voltage setpoints for gen buses are not given, so
%         they are taken from the bus data, however this results in two
%         binding Q limits at buses 34 & 37, so the corresponding
%         voltages have probably deviated from their original setpoints.
%       - The generator locations and types are as follows:
%           1   30      hydro
%           2   31      nuke01
%           3   32      nuke02
%           4   33      fossil02
%           5   34      fossil01
%           6   35      nuke03
%           7   36      fossil04
%           8   37      nuke04
%           9   38      nuke05
%           10  39      interconnection to rest of US/Canada
%
%   This is a solved power flow case, but it includes the following
%   violations:
%       - Pmax violated at bus 31: Pg = 677.87, Pmax = 646
%       - Qmin violated at bus 37: Qg = -1.37,  Qmin = 0
%
%   References:
%   [1] G. W. Bills, et.al., "On-Line Stability Analysis Study"
%       RP90-1 Report for the Edison Electric Institute, October 12, 1970,
%       pp. 1-20 - 1-35.
%       prepared by E. M. Gulachenski - New England Electric System
%                   J. M. Undrill     - General Electric Co.
%       "generally representative of the New England 345 KV system, but is
%        not an exact or complete model of any past, present or projected
%        configuration of the actual New England 345 KV system.
%   [2] M. A. Pai, Energy Function Analysis for Power System Stability,
%       Kluwer Academic Publishers, Boston, 1989.
%       (references [3] as source of data)
%   [3] Athay, T.; Podmore, R.; Virmani, S., "A Practical Method for the
%       Direct Analysis of Transient Stability," IEEE Transactions on Power
%       Apparatus and Systems , vol.PAS-98, no.2, pp.573-584, March 1979.
%       URL: https://doi.org/10.1109/TPAS.1979.319407
%       (references [1] as source of data)
%   [4] Data included with TC Calculator at http://www.pserc.cornell.edu/tcc/
%       for 39-bus system.

%   MATPOWER

"""

def case39():    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'PQ', 'Voltage_0': 1.0393836, 'theta_0': -0.23625827443204947, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.976, 'Reactive_load': 0.442, 'Node_id': '1.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0484941, 'theta_0': -0.17078512035543097, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '2.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0307077, 'theta_0': -0.21426332103915155, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.22, 'Reactive_load': 0.024, 'Node_id': '3.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.00446, 'theta_0': -0.2203780820735137, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 5.0, 'Reactive_load': 1.84, 'Node_id': '4.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0060063, 'theta_0': -0.19534316654936965, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '5.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0082256, 'theta_0': -0.1816596281341014, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '6.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 0.99839728, 'theta_0': -0.2226276718529942, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.338, 'Reactive_load': 0.84, 'Node_id': '7.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 0.99787232, 'theta_0': -0.23275438633233067, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 5.22, 'Reactive_load': 1.766, 'Node_id': '8.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.038332, 'theta_0': -0.24746049570304987, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.065, 'Reactive_load': -0.6659999999999999, 'Node_id': '9.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0178431, 'theta_0': -0.14260867151889167, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '10.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0133858, 'theta_0': -0.1559794870747753, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '11.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.000815, 'theta_0': -0.15705910062616918, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.08529999999999999, 'Reactive_load': 0.88, 'Node_id': '12.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.014923, 'theta_0': -0.15585663160339816, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '13.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.012319, 'theta_0': -0.18701717807248577, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '14.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0161854, 'theta_0': -0.19801456750247215, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.2, 'Reactive_load': 1.53, 'Node_id': '15.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0325203, 'theta_0': -0.17511495759838802, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.29, 'Reactive_load': 0.32299999999999995, 'Node_id': '16.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0342365, 'theta_0': -0.19401840928722838, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '17.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0315726, 'theta_0': -0.20919809629718367, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.58, 'Reactive_load': 0.3, 'Node_id': '18.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0501068, 'theta_0': -0.09442358487791794, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '19.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 0.99101054, 'theta_0': -0.11905202020058953, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 6.8, 'Reactive_load': 1.03, 'Node_id': '20.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0323192, 'theta_0': -0.1331467372436766, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.74, 'Reactive_load': 1.15, 'Node_id': '21.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0501427, 'theta_0': -0.05555592274075265, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '22.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0451451, 'theta_0': -0.059014404354651544, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.475, 'Reactive_load': 0.846, 'Node_id': '23.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.038001, 'theta_0': -0.17302772707257427, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.0860000000000003, 'Reactive_load': -0.922, 'Node_id': '24.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0576827, 'theta_0': -0.14607071360446466, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.24, 'Reactive_load': 0.47200000000000003, 'Node_id': '25.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0525613, 'theta_0': -0.16473760685714817, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.39, 'Reactive_load': 0.17, 'Node_id': '26.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0383449, 'theta_0': -0.19830696251205876, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.81, 'Reactive_load': 0.755, 'Node_id': '27.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0503737, 'theta_0': -0.10346938728089702, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.06, 'Reactive_load': 0.276, 'Node_id': '28.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PQ', 'Voltage_0': 1.0501149, 'theta_0': -0.05532473991869198, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.835, 'Reactive_load': 0.26899999999999996, 'Node_id': '29.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0499, 'theta_0': -0.12863904920461205, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '30.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 0.982, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.092, 'Reactive_load': 0.046, 'Node_id': '31.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 0.9841, 'theta_0': -0.003288853063897563, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '32.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 0.9972, 'theta_0': -0.00337153018322916, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '33.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0123, 'theta_0': -0.028468397041837387, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '34.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0494, 'theta_0': 0.03100589458939765, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '35.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0636, 'theta_0': 0.07798894504925487, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '36.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0275, 'theta_0': -0.027626795785867218, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '37.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.0265, 'theta_0': 0.06794248604491286, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '38.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None},
        {'type': 'PV', 'Voltage_0': 1.03, 'theta_0': -0.2536880748202609, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 11.04, 'Reactive_load': 2.5, 'Node_id': '39.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': '1.0', 'toNode': '2.0', 'Resistance': 0.0035, 'Reactance': 0.0411, 'Conductance': 0, 'Susceptance': 0.6987, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '1'},
        {'fromNode': '1.0', 'toNode': '39.0', 'Resistance': 0.001, 'Reactance': 0.025, 'Conductance': 0, 'Susceptance': 0.75, 'MVA_rating': 1000.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '2'},
        {'fromNode': '2.0', 'toNode': '3.0', 'Resistance': 0.0013, 'Reactance': 0.0151, 'Conductance': 0, 'Susceptance': 0.2572, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '3'},
        {'fromNode': '2.0', 'toNode': '25.0', 'Resistance': 0.007, 'Reactance': 0.0086, 'Conductance': 0, 'Susceptance': 0.146, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '4'},
        {'fromNode': '2.0', 'toNode': '30.0', 'Resistance': 0.0, 'Reactance': 0.0181, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.025, 'shift': 0.0, 'Line_id': '5'},
        {'fromNode': '3.0', 'toNode': '4.0', 'Resistance': 0.0013, 'Reactance': 0.0213, 'Conductance': 0, 'Susceptance': 0.2214, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '6'},
        {'fromNode': '3.0', 'toNode': '18.0', 'Resistance': 0.0011, 'Reactance': 0.0133, 'Conductance': 0, 'Susceptance': 0.2138, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '7'},
        {'fromNode': '4.0', 'toNode': '5.0', 'Resistance': 0.0008, 'Reactance': 0.0128, 'Conductance': 0, 'Susceptance': 0.1342, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '8'},
        {'fromNode': '4.0', 'toNode': '14.0', 'Resistance': 0.0008, 'Reactance': 0.0129, 'Conductance': 0, 'Susceptance': 0.1382, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '9'},
        {'fromNode': '5.0', 'toNode': '6.0', 'Resistance': 0.0002, 'Reactance': 0.0026, 'Conductance': 0, 'Susceptance': 0.0434, 'MVA_rating': 1200.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '10'},
        {'fromNode': '5.0', 'toNode': '8.0', 'Resistance': 0.0008, 'Reactance': 0.0112, 'Conductance': 0, 'Susceptance': 0.1476, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '11'},
        {'fromNode': '6.0', 'toNode': '7.0', 'Resistance': 0.0006, 'Reactance': 0.0092, 'Conductance': 0, 'Susceptance': 0.113, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '12'},
        {'fromNode': '6.0', 'toNode': '11.0', 'Resistance': 0.0007, 'Reactance': 0.0082, 'Conductance': 0, 'Susceptance': 0.1389, 'MVA_rating': 480.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '13'},
        {'fromNode': '6.0', 'toNode': '31.0', 'Resistance': 0.0, 'Reactance': 0.025, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 1800.0, 'kV_base': 345.0, 'm': 1.07, 'shift': 0.0, 'Line_id': '14'},
        {'fromNode': '7.0', 'toNode': '8.0', 'Resistance': 0.0004, 'Reactance': 0.0046, 'Conductance': 0, 'Susceptance': 0.078, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '15'},
        {'fromNode': '8.0', 'toNode': '9.0', 'Resistance': 0.0023, 'Reactance': 0.0363, 'Conductance': 0, 'Susceptance': 0.3804, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '16'},
        {'fromNode': '9.0', 'toNode': '39.0', 'Resistance': 0.001, 'Reactance': 0.025, 'Conductance': 0, 'Susceptance': 1.2, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '17'},
        {'fromNode': '10.0', 'toNode': '11.0', 'Resistance': 0.0004, 'Reactance': 0.0043, 'Conductance': 0, 'Susceptance': 0.0729, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '18'},
        {'fromNode': '10.0', 'toNode': '13.0', 'Resistance': 0.0004, 'Reactance': 0.0043, 'Conductance': 0, 'Susceptance': 0.0729, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '19'},
        {'fromNode': '10.0', 'toNode': '32.0', 'Resistance': 0.0, 'Reactance': 0.02, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.07, 'shift': 0.0, 'Line_id': '20'},
        {'fromNode': '12.0', 'toNode': '11.0', 'Resistance': 0.0016, 'Reactance': 0.0435, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1.006, 'shift': 0.0, 'Line_id': '21'},
        {'fromNode': '12.0', 'toNode': '13.0', 'Resistance': 0.0016, 'Reactance': 0.0435, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 500.0, 'kV_base': 345.0, 'm': 1.006, 'shift': 0.0, 'Line_id': '22'},
        {'fromNode': '13.0', 'toNode': '14.0', 'Resistance': 0.0009, 'Reactance': 0.0101, 'Conductance': 0, 'Susceptance': 0.1723, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '23'},
        {'fromNode': '14.0', 'toNode': '15.0', 'Resistance': 0.0018, 'Reactance': 0.0217, 'Conductance': 0, 'Susceptance': 0.366, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '24'},
        {'fromNode': '15.0', 'toNode': '16.0', 'Resistance': 0.0009, 'Reactance': 0.0094, 'Conductance': 0, 'Susceptance': 0.171, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '25'},
        {'fromNode': '16.0', 'toNode': '17.0', 'Resistance': 0.0007, 'Reactance': 0.0089, 'Conductance': 0, 'Susceptance': 0.1342, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '26'},
        {'fromNode': '16.0', 'toNode': '19.0', 'Resistance': 0.0016, 'Reactance': 0.0195, 'Conductance': 0, 'Susceptance': 0.304, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '27'},
        {'fromNode': '16.0', 'toNode': '21.0', 'Resistance': 0.0008, 'Reactance': 0.0135, 'Conductance': 0, 'Susceptance': 0.2548, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '28'},
        {'fromNode': '16.0', 'toNode': '24.0', 'Resistance': 0.0003, 'Reactance': 0.0059, 'Conductance': 0, 'Susceptance': 0.068, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '29'},
        {'fromNode': '17.0', 'toNode': '18.0', 'Resistance': 0.0007, 'Reactance': 0.0082, 'Conductance': 0, 'Susceptance': 0.1319, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '30'},
        {'fromNode': '17.0', 'toNode': '27.0', 'Resistance': 0.0013, 'Reactance': 0.0173, 'Conductance': 0, 'Susceptance': 0.3216, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '31'},
        {'fromNode': '19.0', 'toNode': '20.0', 'Resistance': 0.0007, 'Reactance': 0.0138, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.06, 'shift': 0.0, 'Line_id': '32'},
        {'fromNode': '19.0', 'toNode': '33.0', 'Resistance': 0.0007, 'Reactance': 0.0142, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.07, 'shift': 0.0, 'Line_id': '33'},
        {'fromNode': '20.0', 'toNode': '34.0', 'Resistance': 0.0009, 'Reactance': 0.018, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.009, 'shift': 0.0, 'Line_id': '34'},
        {'fromNode': '21.0', 'toNode': '22.0', 'Resistance': 0.0008, 'Reactance': 0.014, 'Conductance': 0, 'Susceptance': 0.2565, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '35'},
        {'fromNode': '22.0', 'toNode': '23.0', 'Resistance': 0.0006, 'Reactance': 0.0096, 'Conductance': 0, 'Susceptance': 0.1846, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '36'},
        {'fromNode': '22.0', 'toNode': '35.0', 'Resistance': 0.0, 'Reactance': 0.0143, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.025, 'shift': 0.0, 'Line_id': '37'},
        {'fromNode': '23.0', 'toNode': '24.0', 'Resistance': 0.0022, 'Reactance': 0.035, 'Conductance': 0, 'Susceptance': 0.361, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '38'},
        {'fromNode': '23.0', 'toNode': '36.0', 'Resistance': 0.0005, 'Reactance': 0.0272, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.0, 'shift': 0.0, 'Line_id': '39'},
        {'fromNode': '25.0', 'toNode': '26.0', 'Resistance': 0.0032, 'Reactance': 0.0323, 'Conductance': 0, 'Susceptance': 0.531, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '40'},
        {'fromNode': '25.0', 'toNode': '37.0', 'Resistance': 0.0006, 'Reactance': 0.0232, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 900.0, 'kV_base': 345.0, 'm': 1.025, 'shift': 0.0, 'Line_id': '41'},
        {'fromNode': '26.0', 'toNode': '27.0', 'Resistance': 0.0014, 'Reactance': 0.0147, 'Conductance': 0, 'Susceptance': 0.2396, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '42'},
        {'fromNode': '26.0', 'toNode': '28.0', 'Resistance': 0.0043, 'Reactance': 0.0474, 'Conductance': 0, 'Susceptance': 0.7802, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '43'},
        {'fromNode': '26.0', 'toNode': '29.0', 'Resistance': 0.0057, 'Reactance': 0.0625, 'Conductance': 0, 'Susceptance': 1.029, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '44'},
        {'fromNode': '28.0', 'toNode': '29.0', 'Resistance': 0.0014, 'Reactance': 0.0151, 'Conductance': 0, 'Susceptance': 0.249, 'MVA_rating': 600.0, 'kV_base': 345.0, 'm': 1, 'shift': 0, 'Line_id': '45'},
        {'fromNode': '29.0', 'toNode': '38.0', 'Resistance': 0.0008, 'Reactance': 0.0156, 'Conductance': 0, 'Susceptance': 0.0, 'MVA_rating': 1200.0, 'kV_base': 345.0, 'm': 1.025, 'shift': 0.0, 'Line_id': '46'}
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
    pyf.add_gen(grid, '30.0', '1', price_zone_link=False, lf=0.3, qf=0.01, MWmax=1040.0, MWmin=0.0, MVArmax=400.0, MVArmin=140.0, PsetMW=250.0, QsetMVA=161.762)
    pyf.add_gen(grid, '31.0', '2', price_zone_link=False, lf=0.3, qf=0.01, MWmax=646.0, MWmin=0.0, MVArmax=300.0, MVArmin=-100.0, PsetMW=677.871, QsetMVA=221.574)
    pyf.add_gen(grid, '32.0', '3', price_zone_link=False, lf=0.3, qf=0.01, MWmax=725.0, MWmin=0.0, MVArmax=300.0, MVArmin=150.0, PsetMW=650.0, QsetMVA=206.96500000000003)
    pyf.add_gen(grid, '33.0', '4', price_zone_link=False, lf=0.3, qf=0.01, MWmax=652.0, MWmin=0.0, MVArmax=250.0, MVArmin=0.0, PsetMW=632.0, QsetMVA=108.29300000000002)
    pyf.add_gen(grid, '34.0', '5', price_zone_link=False, lf=0.3, qf=0.01, MWmax=508.0, MWmin=0.0, MVArmax=167.0, MVArmin=0.0, PsetMW=508.0, QsetMVA=166.688)
    pyf.add_gen(grid, '35.0', '6', price_zone_link=False, lf=0.3, qf=0.01, MWmax=687.0, MWmin=0.0, MVArmax=300.0, MVArmin=-100.0, PsetMW=650.0, QsetMVA=210.661)
    pyf.add_gen(grid, '36.0', '7', price_zone_link=False, lf=0.3, qf=0.01, MWmax=580.0, MWmin=0.0, MVArmax=240.0, MVArmin=0.0, PsetMW=560.0, QsetMVA=100.16500000000002)
    pyf.add_gen(grid, '37.0', '8', price_zone_link=False, lf=0.3, qf=0.01, MWmax=564.0, MWmin=0.0, MVArmax=250.0, MVArmin=0.0, PsetMW=540.0, QsetMVA=-1.36945)
    pyf.add_gen(grid, '38.0', '9', price_zone_link=False, lf=0.3, qf=0.01, MWmax=865.0, MWmin=0.0, MVArmax=300.0, MVArmin=-150.0, PsetMW=830.0000000000001, QsetMVA=21.7327)
    pyf.add_gen(grid, '39.0', '10', price_zone_link=False, lf=0.3, qf=0.01, MWmax=1100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-100.0, PsetMW=1000.0, QsetMVA=78.4674)
    
    
    # Add Renewable Source Zones

    
    # Add Renewable Sources

    
    # Return the grid
    return grid,res
