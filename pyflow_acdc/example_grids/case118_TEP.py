import pyflow_acdc as pyf
import pandas as pd


def case118_TEP(exp='All',N_b=1,N_i=1,N_max=5):    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'PV', 'Voltage_0': 0.955, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.51, 'Reactive_load': 0.27, 'Node_id': '1', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.2, 'Reactive_load': 0.09, 'Node_id': '2', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.39, 'Reactive_load': 0.1, 'Node_id': '3', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.998, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.39, 'Reactive_load': 0.12, 'Node_id': '4', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '5', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': -0.4},
        {'type': 'PV', 'Voltage_0': 0.99, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.52, 'Reactive_load': 0.22, 'Node_id': '6', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.19, 'Reactive_load': 0.02, 'Node_id': '7', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.015, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.28, 'Reactive_load': 0.0, 'Node_id': '8', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '9', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '10', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.7, 'Reactive_load': 0.23, 'Node_id': '11', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.99, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.47, 'Reactive_load': 0.1, 'Node_id': '12', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.34, 'Reactive_load': 0.16, 'Node_id': '13', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.14, 'Reactive_load': 0.01, 'Node_id': '14', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.97, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.9, 'Reactive_load': 0.3, 'Node_id': '15', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.25, 'Reactive_load': 0.1, 'Node_id': '16', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.11, 'Reactive_load': 0.03, 'Node_id': '17', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.973, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.6, 'Reactive_load': 0.34, 'Node_id': '18', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.962, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.45, 'Reactive_load': 0.25, 'Node_id': '19', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.18, 'Reactive_load': 0.03, 'Node_id': '20', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.14, 'Reactive_load': 0.08, 'Node_id': '21', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.1, 'Reactive_load': 0.05, 'Node_id': '22', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.07, 'Reactive_load': 0.03, 'Node_id': '23', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.992, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.13, 'Reactive_load': 0.0, 'Node_id': '24', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '25', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.015, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '26', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.968, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.71, 'Reactive_load': 0.13, 'Node_id': '27', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.17, 'Reactive_load': 0.07, 'Node_id': '28', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.24, 'Reactive_load': 0.04, 'Node_id': '29', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '30', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.967, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.43, 'Reactive_load': 0.27, 'Node_id': '31', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.963, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.59, 'Reactive_load': 0.23, 'Node_id': '32', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.23, 'Reactive_load': 0.09, 'Node_id': '33', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.984, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.59, 'Reactive_load': 0.26, 'Node_id': '34', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.14},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.33, 'Reactive_load': 0.09, 'Node_id': '35', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.31, 'Reactive_load': 0.17, 'Node_id': '36', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '37', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': -0.25},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '38', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.27, 'Reactive_load': 0.11, 'Node_id': '39', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.97, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.66, 'Reactive_load': 0.23, 'Node_id': '40', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.37, 'Reactive_load': 0.1, 'Node_id': '41', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.96, 'Reactive_load': 0.23, 'Node_id': '42', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.18, 'Reactive_load': 0.07, 'Node_id': '43', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.16, 'Reactive_load': 0.08, 'Node_id': '44', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.53, 'Reactive_load': 0.22, 'Node_id': '45', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.28, 'Reactive_load': 0.1, 'Node_id': '46', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.34, 'Reactive_load': 0.0, 'Node_id': '47', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.2, 'Reactive_load': 0.11, 'Node_id': '48', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.15},
        {'type': 'PV', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.87, 'Reactive_load': 0.3, 'Node_id': '49', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.17, 'Reactive_load': 0.04, 'Node_id': '50', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.17, 'Reactive_load': 0.08, 'Node_id': '51', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.18, 'Reactive_load': 0.05, 'Node_id': '52', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.23, 'Reactive_load': 0.11, 'Node_id': '53', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.955, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.13, 'Reactive_load': 0.32, 'Node_id': '54', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.952, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.63, 'Reactive_load': 0.22, 'Node_id': '55', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.954, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.84, 'Reactive_load': 0.18, 'Node_id': '56', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.12, 'Reactive_load': 0.03, 'Node_id': '57', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.12, 'Reactive_load': 0.03, 'Node_id': '58', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 2.77, 'Reactive_load': 1.13, 'Node_id': '59', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.78, 'Reactive_load': 0.03, 'Node_id': '60', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.995, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '61', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.998, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.77, 'Reactive_load': 0.14, 'Node_id': '62', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '63', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '64', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '65', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.39, 'Reactive_load': 0.18, 'Node_id': '66', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.28, 'Reactive_load': 0.07, 'Node_id': '67', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '68', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'Slack', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '69', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.984, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.66, 'Reactive_load': 0.2, 'Node_id': '70', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '71', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.12, 'Reactive_load': 0.0, 'Node_id': '72', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.991, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.06, 'Reactive_load': 0.0, 'Node_id': '73', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.958, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.68, 'Reactive_load': 0.27, 'Node_id': '74', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.12},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.47, 'Reactive_load': 0.11, 'Node_id': '75', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.943, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.68, 'Reactive_load': 0.36, 'Node_id': '76', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.006, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.61, 'Reactive_load': 0.28, 'Node_id': '77', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.71, 'Reactive_load': 0.26, 'Node_id': '78', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.39, 'Reactive_load': 0.32, 'Node_id': '79', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.2},
        {'type': 'PV', 'Voltage_0': 1.04, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.3, 'Reactive_load': 0.26, 'Node_id': '80', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '81', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.54, 'Reactive_load': 0.27, 'Node_id': '82', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.2},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.2, 'Reactive_load': 0.1, 'Node_id': '83', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.11, 'Reactive_load': 0.07, 'Node_id': '84', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.24, 'Reactive_load': 0.15, 'Node_id': '85', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.21, 'Reactive_load': 0.1, 'Node_id': '86', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.015, 'theta_0': 0.0, 'kV_base': 161.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '87', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.48, 'Reactive_load': 0.1, 'Node_id': '88', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '89', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.63, 'Reactive_load': 0.42, 'Node_id': '90', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.1, 'Reactive_load': 0.0, 'Node_id': '91', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.99, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.65, 'Reactive_load': 0.1, 'Node_id': '92', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.12, 'Reactive_load': 0.07, 'Node_id': '93', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.3, 'Reactive_load': 0.16, 'Node_id': '94', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.42, 'Reactive_load': 0.31, 'Node_id': '95', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.38, 'Reactive_load': 0.15, 'Node_id': '96', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.15, 'Reactive_load': 0.09, 'Node_id': '97', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.34, 'Reactive_load': 0.08, 'Node_id': '98', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.42, 'Reactive_load': 0.0, 'Node_id': '99', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.37, 'Reactive_load': 0.18, 'Node_id': '100', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.22, 'Reactive_load': 0.15, 'Node_id': '101', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.05, 'Reactive_load': 0.03, 'Node_id': '102', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.23, 'Reactive_load': 0.16, 'Node_id': '103', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.971, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.38, 'Reactive_load': 0.25, 'Node_id': '104', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.965, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.31, 'Reactive_load': 0.26, 'Node_id': '105', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.2},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.43, 'Reactive_load': 0.16, 'Node_id': '106', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.952, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.5, 'Reactive_load': 0.12, 'Node_id': '107', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.06},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.02, 'Reactive_load': 0.01, 'Node_id': '108', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.08, 'Reactive_load': 0.03, 'Node_id': '109', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.973, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.39, 'Reactive_load': 0.3, 'Node_id': '110', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.06},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '111', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.975, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.68, 'Reactive_load': 0.13, 'Node_id': '112', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 0.993, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.06, 'Reactive_load': 0.0, 'Node_id': '113', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.08, 'Reactive_load': 0.03, 'Node_id': '114', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.22, 'Reactive_load': 0.07, 'Node_id': '115', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 1.84, 'Reactive_load': 0.0, 'Node_id': '116', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.2, 'Reactive_load': 0.08, 'Node_id': '117', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0,  'Power_load': 0.33, 'Reactive_load': 0.15, 'Node_id': '118', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'Line_id': '1', 'fromNode': '1', 'toNode': '2', 'r': 0.0303, 'x': 0.1, 'b': 0.0254, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.83},
        {'Line_id': '2', 'fromNode': '1', 'toNode': '3', 'r': 0.0129, 'x': 0.04, 'b': 0.01082, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.18},
        {'Line_id': '3', 'fromNode': '4', 'toNode': '5', 'r': 0.00176, 'x': 0.01, 'b': 0.0021, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 0.86},
        {'Line_id': '4', 'fromNode': '3', 'toNode': '5', 'r': 0.0241, 'x': 0.11, 'b': 0.0284, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.56},
        {'Line_id': '5', 'fromNode': '5', 'toNode': '6', 'r': 0.0119, 'x': 0.05, 'b': 0.01426, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.8},
        {'Line_id': '6', 'fromNode': '6', 'toNode': '7', 'r': 0.00459, 'x': 0.02, 'b': 0.0055, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 2.23},
        {'Line_id': '7', 'fromNode': '8', 'toNode': '9', 'r': 0.00244, 'x': 0.03, 'b': 1.162, 'MVA_rating': 711.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.71},
        {'Line_id': '8', 'fromNode': '8', 'toNode': '5', 'r': 0.0, 'x': 0.03, 'b': 0.0, 'MVA_rating': 1099.0, 'm': 0.985, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '9', 'fromNode': '9', 'toNode': '10', 'r': 0.00258, 'x': 0.03, 'b': 1.23, 'MVA_rating': 710.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.35},
        {'Line_id': '10', 'fromNode': '4', 'toNode': '11', 'r': 0.0209, 'x': 0.07, 'b': 0.01748, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.77},
        {'Line_id': '11', 'fromNode': '5', 'toNode': '11', 'r': 0.0203, 'x': 0.07, 'b': 0.01738, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.73},
        {'Line_id': '12', 'fromNode': '11', 'toNode': '12', 'r': 0.00595, 'x': 0.02, 'b': 0.00502, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 1.93},
        {'Line_id': '13', 'fromNode': '2', 'toNode': '12', 'r': 0.0187, 'x': 0.06, 'b': 0.01572, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.06},
        {'Line_id': '14', 'fromNode': '3', 'toNode': '12', 'r': 0.0484, 'x': 0.16, 'b': 0.0406, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.73},
        {'Line_id': '15', 'fromNode': '7', 'toNode': '12', 'r': 0.00862, 'x': 0.03, 'b': 0.00874, 'MVA_rating': 164.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.49},
        {'Line_id': '16', 'fromNode': '11', 'toNode': '13', 'r': 0.02225, 'x': 0.07, 'b': 0.01876, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.2},
        {'Line_id': '17', 'fromNode': '12', 'toNode': '14', 'r': 0.0215, 'x': 0.07, 'b': 0.01816, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.96},
        {'Line_id': '18', 'fromNode': '13', 'toNode': '15', 'r': 0.0744, 'x': 0.24, 'b': 0.06268, 'MVA_rating': 115.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.33},
        {'Line_id': '19', 'fromNode': '14', 'toNode': '15', 'r': 0.0595, 'x': 0.2, 'b': 0.0502, 'MVA_rating': 144.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.32},
        {'Line_id': '20', 'fromNode': '12', 'toNode': '16', 'r': 0.0212, 'x': 0.08, 'b': 0.0214, 'MVA_rating': 164.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.58},
        {'Line_id': '21', 'fromNode': '15', 'toNode': '17', 'r': 0.0132, 'x': 0.04, 'b': 0.0444, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.3},
        {'Line_id': '22', 'fromNode': '16', 'toNode': '17', 'r': 0.0454, 'x': 0.18, 'b': 0.0466, 'MVA_rating': 158.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.81},
        {'Line_id': '23', 'fromNode': '17', 'toNode': '18', 'r': 0.0123, 'x': 0.05, 'b': 0.01298, 'MVA_rating': 167.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.24},
        {'Line_id': '24', 'fromNode': '18', 'toNode': '19', 'r': 0.01119, 'x': 0.05, 'b': 0.01142, 'MVA_rating': 173.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.23},
        {'Line_id': '25', 'fromNode': '19', 'toNode': '20', 'r': 0.0252, 'x': 0.12, 'b': 0.0298, 'MVA_rating': 178.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.66},
        {'Line_id': '26', 'fromNode': '15', 'toNode': '19', 'r': 0.012, 'x': 0.04, 'b': 0.0101, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.88},
        {'Line_id': '27', 'fromNode': '20', 'toNode': '21', 'r': 0.0183, 'x': 0.08, 'b': 0.0216, 'MVA_rating': 177.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.13},
        {'Line_id': '28', 'fromNode': '21', 'toNode': '22', 'r': 0.0209, 'x': 0.1, 'b': 0.0246, 'MVA_rating': 178.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.49},
        {'Line_id': '29', 'fromNode': '22', 'toNode': '23', 'r': 0.0342, 'x': 0.16, 'b': 0.0404, 'MVA_rating': 178.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.19},
        {'Line_id': '30', 'fromNode': '23', 'toNode': '24', 'r': 0.0135, 'x': 0.05, 'b': 0.0498, 'MVA_rating': 158.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.95},
        {'Line_id': '31', 'fromNode': '23', 'toNode': '25', 'r': 0.0156, 'x': 0.08, 'b': 0.0864, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.89},
        {'Line_id': '32', 'fromNode': '26', 'toNode': '25', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 768.0, 'm': 0.96, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '33', 'fromNode': '25', 'toNode': '27', 'r': 0.0318, 'x': 0.16, 'b': 0.1764, 'MVA_rating': 177.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.24},
        {'Line_id': '34', 'fromNode': '27', 'toNode': '28', 'r': 0.01913, 'x': 0.09, 'b': 0.0216, 'MVA_rating': 174.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.1},
        {'Line_id': '35', 'fromNode': '28', 'toNode': '29', 'r': 0.0237, 'x': 0.09, 'b': 0.0238, 'MVA_rating': 165.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.74},
        {'Line_id': '36', 'fromNode': '30', 'toNode': '17', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 756.0, 'm': 0.96, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '37', 'fromNode': '8', 'toNode': '30', 'r': 0.00431, 'x': 0.05, 'b': 0.514, 'MVA_rating': 580.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.87},
        {'Line_id': '38', 'fromNode': '26', 'toNode': '30', 'r': 0.00799, 'x': 0.09, 'b': 0.908, 'MVA_rating': 340.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.98},
        {'Line_id': '39', 'fromNode': '17', 'toNode': '31', 'r': 0.0474, 'x': 0.16, 'b': 0.0399, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.38},
        {'Line_id': '40', 'fromNode': '29', 'toNode': '31', 'r': 0.0108, 'x': 0.03, 'b': 0.0083, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.2},
        {'Line_id': '41', 'fromNode': '23', 'toNode': '32', 'r': 0.0317, 'x': 0.12, 'b': 0.1173, 'MVA_rating': 158.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.61},
        {'Line_id': '42', 'fromNode': '31', 'toNode': '32', 'r': 0.0298, 'x': 0.1, 'b': 0.0251, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.69},
        {'Line_id': '43', 'fromNode': '27', 'toNode': '32', 'r': 0.0229, 'x': 0.08, 'b': 0.01926, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.43},
        {'Line_id': '44', 'fromNode': '15', 'toNode': '33', 'r': 0.038, 'x': 0.12, 'b': 0.03194, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.18},
        {'Line_id': '45', 'fromNode': '19', 'toNode': '34', 'r': 0.0752, 'x': 0.25, 'b': 0.0632, 'MVA_rating': 114.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.37},
        {'Line_id': '46', 'fromNode': '35', 'toNode': '36', 'r': 0.00224, 'x': 0.01, 'b': 0.00268, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 1.09},
        {'Line_id': '47', 'fromNode': '35', 'toNode': '37', 'r': 0.011, 'x': 0.05, 'b': 0.01318, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.31},
        {'Line_id': '48', 'fromNode': '33', 'toNode': '37', 'r': 0.0415, 'x': 0.14, 'b': 0.0366, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': 14.13},
        {'Line_id': '49', 'fromNode': '34', 'toNode': '36', 'r': 0.00871, 'x': 0.03, 'b': 0.00568, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': 2.59},
        {'Line_id': '50', 'fromNode': '34', 'toNode': '37', 'r': 0.00256, 'x': 0.01, 'b': 0.00984, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': 0.95},
        {'Line_id': '51', 'fromNode': '38', 'toNode': '37', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 783.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': 14.68},
        {'Line_id': '52', 'fromNode': '37', 'toNode': '39', 'r': 0.0321, 'x': 0.11, 'b': 0.027, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.43},
        {'Line_id': '53', 'fromNode': '37', 'toNode': '40', 'r': 0.0593, 'x': 0.17, 'b': 0.042, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.91},
        {'Line_id': '54', 'fromNode': '30', 'toNode': '38', 'r': 0.00464, 'x': 0.05, 'b': 0.422, 'MVA_rating': 542.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.89},
        {'Line_id': '55', 'fromNode': '39', 'toNode': '40', 'r': 0.0184, 'x': 0.06, 'b': 0.01552, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.96},
        {'Line_id': '56', 'fromNode': '40', 'toNode': '41', 'r': 0.0145, 'x': 0.05, 'b': 0.01222, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.8},
        {'Line_id': '57', 'fromNode': '40', 'toNode': '42', 'r': 0.0555, 'x': 0.18, 'b': 0.0466, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.01},
        {'Line_id': '58', 'fromNode': '41', 'toNode': '42', 'r': 0.041, 'x': 0.14, 'b': 0.0344, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.29},
        {'Line_id': '59', 'fromNode': '43', 'toNode': '44', 'r': 0.0608, 'x': 0.25, 'b': 0.06068, 'MVA_rating': 117.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.91},
        {'Line_id': '60', 'fromNode': '34', 'toNode': '43', 'r': 0.0413, 'x': 0.17, 'b': 0.04226, 'MVA_rating': 167.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.48},
        {'Line_id': '61', 'fromNode': '44', 'toNode': '45', 'r': 0.0224, 'x': 0.09, 'b': 0.0224, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.34},
        {'Line_id': '62', 'fromNode': '45', 'toNode': '46', 'r': 0.04, 'x': 0.14, 'b': 0.0332, 'MVA_rating': 153.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.43},
        {'Line_id': '63', 'fromNode': '46', 'toNode': '47', 'r': 0.038, 'x': 0.13, 'b': 0.0316, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.54},
        {'Line_id': '64', 'fromNode': '46', 'toNode': '48', 'r': 0.0601, 'x': 0.19, 'b': 0.0472, 'MVA_rating': 148.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.43},
        {'Line_id': '65', 'fromNode': '47', 'toNode': '49', 'r': 0.0191, 'x': 0.06, 'b': 0.01604, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.12},
        {'Line_id': '66', 'fromNode': '42', 'toNode': '49', 'r': 0.0715, 'x': 0.32, 'b': 0.086, 'MVA_rating': 89.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.56},
        {'Line_id': '67', 'fromNode': '42', 'toNode': '49', 'r': 0.0715, 'x': 0.32, 'b': 0.086, 'MVA_rating': 89.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.56},
        {'Line_id': '68', 'fromNode': '45', 'toNode': '49', 'r': 0.0684, 'x': 0.19, 'b': 0.0444, 'MVA_rating': 138.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.55},
        {'Line_id': '69', 'fromNode': '48', 'toNode': '49', 'r': 0.0179, 'x': 0.05, 'b': 0.01258, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.79},
        {'Line_id': '70', 'fromNode': '49', 'toNode': '50', 'r': 0.0267, 'x': 0.08, 'b': 0.01874, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.13},
        {'Line_id': '71', 'fromNode': '49', 'toNode': '51', 'r': 0.0486, 'x': 0.14, 'b': 0.0342, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.99},
        {'Line_id': '72', 'fromNode': '51', 'toNode': '52', 'r': 0.0203, 'x': 0.06, 'b': 0.01396, 'MVA_rating': 142.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.62},
        {'Line_id': '73', 'fromNode': '52', 'toNode': '53', 'r': 0.0405, 'x': 0.16, 'b': 0.04058, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': 16.93},
        {'Line_id': '74', 'fromNode': '53', 'toNode': '54', 'r': 0.0263, 'x': 0.12, 'b': 0.031, 'MVA_rating': 177.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.12},
        {'Line_id': '75', 'fromNode': '49', 'toNode': '54', 'r': 0.073, 'x': 0.29, 'b': 0.0738, 'MVA_rating': 99.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.92},
        {'Line_id': '76', 'fromNode': '49', 'toNode': '54', 'r': 0.0869, 'x': 0.29, 'b': 0.073, 'MVA_rating': 97.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.33},
        {'Line_id': '77', 'fromNode': '54', 'toNode': '55', 'r': 0.0169, 'x': 0.07, 'b': 0.0202, 'MVA_rating': 169.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.4},
        {'Line_id': '78', 'fromNode': '54', 'toNode': '56', 'r': 0.00275, 'x': 0.01, 'b': 0.00732, 'MVA_rating': 155.0, 'm': 1, 'shift': 0, 'Cost MEUR': 0.95},
        {'Line_id': '79', 'fromNode': '55', 'toNode': '56', 'r': 0.00488, 'x': 0.02, 'b': 0.00374, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': 1.46},
        {'Line_id': '80', 'fromNode': '56', 'toNode': '57', 'r': 0.0343, 'x': 0.1, 'b': 0.0242, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.16},
        {'Line_id': '81', 'fromNode': '50', 'toNode': '57', 'r': 0.0474, 'x': 0.13, 'b': 0.0332, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.7},
        {'Line_id': '82', 'fromNode': '56', 'toNode': '58', 'r': 0.0343, 'x': 0.1, 'b': 0.0242, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.16},
        {'Line_id': '83', 'fromNode': '51', 'toNode': '58', 'r': 0.0255, 'x': 0.07, 'b': 0.01788, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.82},
        {'Line_id': '84', 'fromNode': '54', 'toNode': '59', 'r': 0.0503, 'x': 0.23, 'b': 0.0598, 'MVA_rating': 125.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.48},
        {'Line_id': '85', 'fromNode': '56', 'toNode': '59', 'r': 0.0825, 'x': 0.25, 'b': 0.0569, 'MVA_rating': 112.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.68},
        {'Line_id': '86', 'fromNode': '56', 'toNode': '59', 'r': 0.0803, 'x': 0.24, 'b': 0.0536, 'MVA_rating': 117.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.68},
        {'Line_id': '87', 'fromNode': '55', 'toNode': '59', 'r': 0.04739, 'x': 0.22, 'b': 0.05646, 'MVA_rating': 133.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.5},
        {'Line_id': '88', 'fromNode': '59', 'toNode': '60', 'r': 0.0317, 'x': 0.15, 'b': 0.0376, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.55},
        {'Line_id': '89', 'fromNode': '59', 'toNode': '61', 'r': 0.0328, 'x': 0.15, 'b': 0.0388, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 16.09},
        {'Line_id': '90', 'fromNode': '60', 'toNode': '61', 'r': 0.00264, 'x': 0.01, 'b': 0.01456, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 1.5},
        {'Line_id': '91', 'fromNode': '60', 'toNode': '62', 'r': 0.0123, 'x': 0.06, 'b': 0.01468, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.02},
        {'Line_id': '92', 'fromNode': '61', 'toNode': '62', 'r': 0.00824, 'x': 0.04, 'b': 0.0098, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.03},
        {'Line_id': '93', 'fromNode': '63', 'toNode': '59', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 760.0, 'm': 0.96, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '94', 'fromNode': '63', 'toNode': '64', 'r': 0.00172, 'x': 0.02, 'b': 0.216, 'MVA_rating': 687.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.46},
        {'Line_id': '95', 'fromNode': '64', 'toNode': '61', 'r': 0.0, 'x': 0.03, 'b': 0.0, 'MVA_rating': 1095.0, 'm': 0.985, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '96', 'fromNode': '38', 'toNode': '65', 'r': 0.00901, 'x': 0.1, 'b': 1.046, 'MVA_rating': 297.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.98},
        {'Line_id': '97', 'fromNode': '64', 'toNode': '65', 'r': 0.00269, 'x': 0.03, 'b': 0.38, 'MVA_rating': 675.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.1},
        {'Line_id': '98', 'fromNode': '49', 'toNode': '66', 'r': 0.018, 'x': 0.09, 'b': 0.0248, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.22},
        {'Line_id': '99', 'fromNode': '49', 'toNode': '66', 'r': 0.018, 'x': 0.09, 'b': 0.0248, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.22},
        {'Line_id': '100', 'fromNode': '62', 'toNode': '66', 'r': 0.0482, 'x': 0.22, 'b': 0.0578, 'MVA_rating': 132.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.57},
        {'Line_id': '101', 'fromNode': '62', 'toNode': '67', 'r': 0.0258, 'x': 0.12, 'b': 0.031, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.57},
        {'Line_id': '102', 'fromNode': '65', 'toNode': '66', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 793.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '103', 'fromNode': '66', 'toNode': '67', 'r': 0.0224, 'x': 0.1, 'b': 0.02682, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.9},
        {'Line_id': '104', 'fromNode': '65', 'toNode': '68', 'r': 0.00138, 'x': 0.02, 'b': 0.638, 'MVA_rating': 686.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.96},
        {'Line_id': '105', 'fromNode': '47', 'toNode': '69', 'r': 0.0844, 'x': 0.28, 'b': 0.07092, 'MVA_rating': 102.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.47},
        {'Line_id': '106', 'fromNode': '49', 'toNode': '69', 'r': 0.0985, 'x': 0.32, 'b': 0.0828, 'MVA_rating': 87.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.38},
        {'Line_id': '107', 'fromNode': '68', 'toNode': '69', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 793.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '108', 'fromNode': '69', 'toNode': '70', 'r': 0.03, 'x': 0.13, 'b': 0.122, 'MVA_rating': 170.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.35},
        {'Line_id': '109', 'fromNode': '24', 'toNode': '70', 'r': 0.00221, 'x': 0.41, 'b': 0.10198, 'MVA_rating': 72.0, 'm': 1, 'shift': 0, 'Cost MEUR': 14.89},
        {'Line_id': '110', 'fromNode': '70', 'toNode': '71', 'r': 0.00882, 'x': 0.04, 'b': 0.00878, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.68},
        {'Line_id': '111', 'fromNode': '24', 'toNode': '72', 'r': 0.0488, 'x': 0.2, 'b': 0.0488, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.87},
        {'Line_id': '112', 'fromNode': '71', 'toNode': '72', 'r': 0.0446, 'x': 0.18, 'b': 0.04444, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.86},
        {'Line_id': '113', 'fromNode': '71', 'toNode': '73', 'r': 0.00866, 'x': 0.05, 'b': 0.01178, 'MVA_rating': 188.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.08},
        {'Line_id': '114', 'fromNode': '70', 'toNode': '74', 'r': 0.0401, 'x': 0.13, 'b': 0.03368, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.02},
        {'Line_id': '115', 'fromNode': '70', 'toNode': '75', 'r': 0.0428, 'x': 0.14, 'b': 0.036, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.88},
        {'Line_id': '116', 'fromNode': '69', 'toNode': '75', 'r': 0.0405, 'x': 0.12, 'b': 0.124, 'MVA_rating': 145.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.78},
        {'Line_id': '117', 'fromNode': '74', 'toNode': '75', 'r': 0.0123, 'x': 0.04, 'b': 0.01034, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.99},
        {'Line_id': '118', 'fromNode': '76', 'toNode': '77', 'r': 0.0444, 'x': 0.15, 'b': 0.0368, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': 14.62},
        {'Line_id': '119', 'fromNode': '69', 'toNode': '77', 'r': 0.0309, 'x': 0.1, 'b': 0.1038, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.89},
        {'Line_id': '120', 'fromNode': '75', 'toNode': '77', 'r': 0.0601, 'x': 0.2, 'b': 0.04978, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.33},
        {'Line_id': '121', 'fromNode': '77', 'toNode': '78', 'r': 0.00376, 'x': 0.01, 'b': 0.01264, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 1.22},
        {'Line_id': '122', 'fromNode': '78', 'toNode': '79', 'r': 0.00546, 'x': 0.02, 'b': 0.00648, 'MVA_rating': 174.0, 'm': 1, 'shift': 0, 'Cost MEUR': 2.6},
        {'Line_id': '123', 'fromNode': '77', 'toNode': '80', 'r': 0.017, 'x': 0.05, 'b': 0.0472, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.62},
        {'Line_id': '124', 'fromNode': '77', 'toNode': '80', 'r': 0.0294, 'x': 0.11, 'b': 0.0228, 'MVA_rating': 157.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.55},
        {'Line_id': '125', 'fromNode': '79', 'toNode': '80', 'r': 0.0156, 'x': 0.07, 'b': 0.0187, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.53},
        {'Line_id': '126', 'fromNode': '68', 'toNode': '81', 'r': 0.00175, 'x': 0.02, 'b': 0.808, 'MVA_rating': 684.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.51},
        {'Line_id': '127', 'fromNode': '81', 'toNode': '80', 'r': 0.0, 'x': 0.04, 'b': 0.0, 'MVA_rating': 793.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': 14.67},
        {'Line_id': '128', 'fromNode': '77', 'toNode': '82', 'r': 0.0298, 'x': 0.09, 'b': 0.08174, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.11},
        {'Line_id': '129', 'fromNode': '82', 'toNode': '83', 'r': 0.0112, 'x': 0.04, 'b': 0.03796, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.59},
        {'Line_id': '130', 'fromNode': '83', 'toNode': '84', 'r': 0.0625, 'x': 0.13, 'b': 0.0258, 'MVA_rating': 122.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.86},
        {'Line_id': '131', 'fromNode': '83', 'toNode': '85', 'r': 0.043, 'x': 0.15, 'b': 0.0348, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': 14.71},
        {'Line_id': '132', 'fromNode': '84', 'toNode': '85', 'r': 0.0302, 'x': 0.06, 'b': 0.01234, 'MVA_rating': 122.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.75},
        {'Line_id': '133', 'fromNode': '85', 'toNode': '86', 'r': 0.035, 'x': 0.12, 'b': 0.0276, 'MVA_rating': 156.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.32},
        {'Line_id': '134', 'fromNode': '86', 'toNode': '87', 'r': 0.02828, 'x': 0.21, 'b': 0.0445, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': 16.62},
        {'Line_id': '135', 'fromNode': '85', 'toNode': '88', 'r': 0.02, 'x': 0.1, 'b': 0.0276, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.35},
        {'Line_id': '136', 'fromNode': '85', 'toNode': '89', 'r': 0.0239, 'x': 0.17, 'b': 0.047, 'MVA_rating': 168.0, 'm': 1, 'shift': 0, 'Cost MEUR': 16.54},
        {'Line_id': '137', 'fromNode': '88', 'toNode': '89', 'r': 0.0139, 'x': 0.07, 'b': 0.01934, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.91},
        {'Line_id': '138', 'fromNode': '89', 'toNode': '90', 'r': 0.0518, 'x': 0.19, 'b': 0.0528, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.1},
        {'Line_id': '139', 'fromNode': '89', 'toNode': '90', 'r': 0.0238, 'x': 0.1, 'b': 0.106, 'MVA_rating': 169.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.44},
        {'Line_id': '140', 'fromNode': '90', 'toNode': '91', 'r': 0.0254, 'x': 0.08, 'b': 0.0214, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.23},
        {'Line_id': '141', 'fromNode': '89', 'toNode': '92', 'r': 0.0099, 'x': 0.05, 'b': 0.0548, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.62},
        {'Line_id': '142', 'fromNode': '89', 'toNode': '92', 'r': 0.0393, 'x': 0.16, 'b': 0.0414, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': 16.38},
        {'Line_id': '143', 'fromNode': '91', 'toNode': '92', 'r': 0.0387, 'x': 0.13, 'b': 0.03268, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.53},
        {'Line_id': '144', 'fromNode': '92', 'toNode': '93', 'r': 0.0258, 'x': 0.08, 'b': 0.0218, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.35},
        {'Line_id': '145', 'fromNode': '92', 'toNode': '94', 'r': 0.0481, 'x': 0.16, 'b': 0.0406, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.56},
        {'Line_id': '146', 'fromNode': '93', 'toNode': '94', 'r': 0.0223, 'x': 0.07, 'b': 0.01876, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.21},
        {'Line_id': '147', 'fromNode': '94', 'toNode': '95', 'r': 0.0132, 'x': 0.04, 'b': 0.0111, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.27},
        {'Line_id': '148', 'fromNode': '80', 'toNode': '96', 'r': 0.0356, 'x': 0.18, 'b': 0.0494, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.3},
        {'Line_id': '149', 'fromNode': '82', 'toNode': '96', 'r': 0.0162, 'x': 0.05, 'b': 0.0544, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.19},
        {'Line_id': '150', 'fromNode': '94', 'toNode': '96', 'r': 0.0269, 'x': 0.09, 'b': 0.023, 'MVA_rating': 149.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.48},
        {'Line_id': '151', 'fromNode': '80', 'toNode': '97', 'r': 0.0183, 'x': 0.09, 'b': 0.0254, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 10.39},
        {'Line_id': '152', 'fromNode': '80', 'toNode': '98', 'r': 0.0238, 'x': 0.11, 'b': 0.0286, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 11.6},
        {'Line_id': '153', 'fromNode': '80', 'toNode': '99', 'r': 0.0454, 'x': 0.21, 'b': 0.0546, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.6},
        {'Line_id': '154', 'fromNode': '92', 'toNode': '100', 'r': 0.0648, 'x': 0.3, 'b': 0.0472, 'MVA_rating': 98.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.63},
        {'Line_id': '155', 'fromNode': '94', 'toNode': '100', 'r': 0.0178, 'x': 0.06, 'b': 0.0604, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.69},
        {'Line_id': '156', 'fromNode': '95', 'toNode': '96', 'r': 0.0171, 'x': 0.05, 'b': 0.01474, 'MVA_rating': 149.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.35},
        {'Line_id': '157', 'fromNode': '96', 'toNode': '97', 'r': 0.0173, 'x': 0.09, 'b': 0.024, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': 9.84},
        {'Line_id': '158', 'fromNode': '98', 'toNode': '100', 'r': 0.0397, 'x': 0.18, 'b': 0.0476, 'MVA_rating': 160.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.5},
        {'Line_id': '159', 'fromNode': '99', 'toNode': '100', 'r': 0.018, 'x': 0.08, 'b': 0.0216, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': 8.69},
        {'Line_id': '160', 'fromNode': '100', 'toNode': '101', 'r': 0.0277, 'x': 0.13, 'b': 0.0328, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 13.54},
        {'Line_id': '161', 'fromNode': '92', 'toNode': '102', 'r': 0.0123, 'x': 0.06, 'b': 0.01464, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.0},
        {'Line_id': '162', 'fromNode': '101', 'toNode': '102', 'r': 0.0246, 'x': 0.11, 'b': 0.0294, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 12.02},
        {'Line_id': '163', 'fromNode': '100', 'toNode': '103', 'r': 0.016, 'x': 0.05, 'b': 0.0536, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.17},
        {'Line_id': '164', 'fromNode': '100', 'toNode': '104', 'r': 0.0451, 'x': 0.2, 'b': 0.0541, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.56},
        {'Line_id': '165', 'fromNode': '103', 'toNode': '104', 'r': 0.0466, 'x': 0.16, 'b': 0.0407, 'MVA_rating': 153.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.68},
        {'Line_id': '166', 'fromNode': '103', 'toNode': '105', 'r': 0.0535, 'x': 0.16, 'b': 0.0408, 'MVA_rating': 145.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.66},
        {'Line_id': '167', 'fromNode': '100', 'toNode': '106', 'r': 0.0605, 'x': 0.23, 'b': 0.062, 'MVA_rating': 124.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.95},
        {'Line_id': '168', 'fromNode': '104', 'toNode': '105', 'r': 0.00994, 'x': 0.04, 'b': 0.00986, 'MVA_rating': 161.0, 'm': 1, 'shift': 0, 'Cost MEUR': 3.84},
        {'Line_id': '169', 'fromNode': '105', 'toNode': '106', 'r': 0.014, 'x': 0.05, 'b': 0.01434, 'MVA_rating': 164.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.63},
        {'Line_id': '170', 'fromNode': '105', 'toNode': '107', 'r': 0.053, 'x': 0.18, 'b': 0.0472, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.17},
        {'Line_id': '171', 'fromNode': '105', 'toNode': '108', 'r': 0.0261, 'x': 0.07, 'b': 0.01844, 'MVA_rating': 137.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.6},
        {'Line_id': '172', 'fromNode': '106', 'toNode': '107', 'r': 0.053, 'x': 0.18, 'b': 0.0472, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.17},
        {'Line_id': '173', 'fromNode': '108', 'toNode': '109', 'r': 0.0105, 'x': 0.03, 'b': 0.0076, 'MVA_rating': 138.0, 'm': 1, 'shift': 0, 'Cost MEUR': 2.71},
        {'Line_id': '174', 'fromNode': '103', 'toNode': '110', 'r': 0.03906, 'x': 0.18, 'b': 0.0461, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': 17.52},
        {'Line_id': '175', 'fromNode': '109', 'toNode': '110', 'r': 0.0278, 'x': 0.08, 'b': 0.0202, 'MVA_rating': 138.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.18},
        {'Line_id': '176', 'fromNode': '110', 'toNode': '111', 'r': 0.022, 'x': 0.08, 'b': 0.02, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.51},
        {'Line_id': '177', 'fromNode': '110', 'toNode': '112', 'r': 0.0247, 'x': 0.06, 'b': 0.062, 'MVA_rating': 135.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.99},
        {'Line_id': '178', 'fromNode': '17', 'toNode': '113', 'r': 0.00913, 'x': 0.03, 'b': 0.00768, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 2.96},
        {'Line_id': '179', 'fromNode': '32', 'toNode': '113', 'r': 0.0615, 'x': 0.2, 'b': 0.0518, 'MVA_rating': 139.0, 'm': 1, 'shift': 0, 'Cost MEUR': 18.38},
        {'Line_id': '180', 'fromNode': '32', 'toNode': '114', 'r': 0.0135, 'x': 0.06, 'b': 0.01628, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': 6.57},
        {'Line_id': '181', 'fromNode': '27', 'toNode': '115', 'r': 0.0164, 'x': 0.07, 'b': 0.01972, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': 7.92},
        {'Line_id': '182', 'fromNode': '114', 'toNode': '115', 'r': 0.0023, 'x': 0.01, 'b': 0.00276, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': 1.11},
        {'Line_id': '183', 'fromNode': '68', 'toNode': '116', 'r': 0.00034, 'x': 0.0, 'b': 0.164, 'MVA_rating': 7218.0, 'm': 1, 'shift': 0, 'Cost MEUR': 15.84},
        {'Line_id': '184', 'fromNode': '12', 'toNode': '117', 'r': 0.0329, 'x': 0.14, 'b': 0.0358, 'MVA_rating': 170.0, 'm': 1, 'shift': 0, 'Cost MEUR': 14.7},
        {'Line_id': '185', 'fromNode': '75', 'toNode': '118', 'r': 0.0145, 'x': 0.05, 'b': 0.01198, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 4.73},
        {'Line_id': '186', 'fromNode': '76', 'toNode': '118', 'r': 0.0164, 'x': 0.05, 'b': 0.01356, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': 5.35}
        ]
    lines_AC = pd.DataFrame(lines_AC_data)

    nodes_DC = None

    lines_DC = None

    Converters_ACDC = None

    # Create the grid
    [grid, res] = pyf.Create_grid_from_data(S_base, nodes_AC, lines_AC, nodes_DC, lines_DC, Converters_ACDC, data_in='pu')
    grid.name = 'case118_TEP'
    

    
    # Add Generators
    pyf.add_gen(grid, '1', '1', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=15.0, MVArmin=-5.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '4', '2', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '6', '3', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=50.0, MVArmin=-13.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '8', '4', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '10', '5', lf=20.0, qf=0.0222222222, MWmax=550.0, MWmin=0.0, MVArmax=200.0, MVArmin=-147.0, PsetMW=450.0, QsetMVA=0.0)
    pyf.add_gen(grid, '12', '6', lf=20.0, qf=0.117647059, MWmax=185.0, MWmin=0.0, MVArmax=120.0, MVArmin=-35.0, PsetMW=85.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15', '7', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=30.0, MVArmin=-10.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '18', '8', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=50.0, MVArmin=-16.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '19', '9', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=24.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '24', '10', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '25', '11', lf=20.0, qf=0.0454545455, MWmax=320.0, MWmin=0.0, MVArmax=140.0, MVArmin=-47.0, PsetMW=220.00000000000003, QsetMVA=0.0)
    pyf.add_gen(grid, '26', '12', lf=20.0, qf=0.0318471338, MWmax=413.99999999999994, MWmin=0.0, MVArmax=1000.0, MVArmin=-1000.0, PsetMW=314.0, QsetMVA=0.0)
    pyf.add_gen(grid, '27', '13', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '31', '14', lf=20.0, qf=1.42857143, MWmax=107.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=7.000000000000001, QsetMVA=0.0)
    pyf.add_gen(grid, '32', '15', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=42.0, MVArmin=-14.000000000000002, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '34', '16', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=24.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '36', '17', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=24.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '40', '18', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '42', '19', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '46', '20', lf=20.0, qf=0.526315789, MWmax=119.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=19.0, QsetMVA=0.0)
    pyf.add_gen(grid, '49', '21', lf=20.0, qf=0.0490196078, MWmax=304.0, MWmin=0.0, MVArmax=210.0, MVArmin=-85.0, PsetMW=204.0, QsetMVA=0.0)
    pyf.add_gen(grid, '54', '22', lf=20.0, qf=0.208333333, MWmax=148.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=48.0, QsetMVA=0.0)
    pyf.add_gen(grid, '55', '23', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '56', '24', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=15.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '59', '25', lf=20.0, qf=0.064516129, MWmax=254.99999999999997, MWmin=0.0, MVArmax=180.0, MVArmin=-60.0, PsetMW=155.0, QsetMVA=0.0)
    pyf.add_gen(grid, '61', '26', lf=20.0, qf=0.0625, MWmax=260.0, MWmin=0.0, MVArmax=300.0, MVArmin=-100.0, PsetMW=160.0, QsetMVA=0.0)
    pyf.add_gen(grid, '62', '27', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=20.0, MVArmin=-20.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '65', '28', lf=20.0, qf=0.0255754476, MWmax=491.0, MWmin=0.0, MVArmax=200.0, MVArmin=-67.0, PsetMW=391.0, QsetMVA=0.0)
    pyf.add_gen(grid, '66', '29', lf=20.0, qf=0.0255102041, MWmax=492.0, MWmin=0.0, MVArmax=200.0, MVArmin=-67.0, PsetMW=392.0, QsetMVA=0.0)
    pyf.add_gen(grid, '69', '30', lf=20.0, qf=0.0193648335, MWmax=805.1999999999999, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=516.4, QsetMVA=0.0)
    pyf.add_gen(grid, '70', '31', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=32.0, MVArmin=-10.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '72', '32', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '73', '33', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '74', '34', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=9.0, MVArmin=-6.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '76', '35', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '77', '36', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=70.0, MVArmin=-20.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '80', '37', lf=20.0, qf=0.0209643606, MWmax=577.0, MWmin=0.0, MVArmax=280.0, MVArmin=-165.0, PsetMW=476.99999999999994, QsetMVA=0.0)
    pyf.add_gen(grid, '85', '38', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '87', '39', lf=20.0, qf=2.5, MWmax=104.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-100.0, PsetMW=4.0, QsetMVA=0.0)
    pyf.add_gen(grid, '89', '40', lf=20.0, qf=0.0164744646, MWmax=707.0, MWmin=0.0, MVArmax=300.0, MVArmin=-210.0, PsetMW=607.0, QsetMVA=0.0)
    pyf.add_gen(grid, '90', '41', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '91', '42', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '92', '43', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=9.0, MVArmin=-3.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '99', '44', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '100', '45', lf=20.0, qf=0.0396825397, MWmax=352.0, MWmin=0.0, MVArmax=155.0, MVArmin=-50.0, PsetMW=252.0, QsetMVA=0.0)
    pyf.add_gen(grid, '103', '46', lf=20.0, qf=0.25, MWmax=140.0, MWmin=0.0, MVArmax=40.0, MVArmin=-15.0, PsetMW=40.0, QsetMVA=0.0)
    pyf.add_gen(grid, '104', '47', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '105', '48', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '107', '49', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=200.0, MVArmin=-200.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '110', '50', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '111', '51', lf=20.0, qf=0.277777778, MWmax=136.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-100.0, PsetMW=36.0, QsetMVA=0.0)
    pyf.add_gen(grid, '112', '52', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '113', '53', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=200.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '116', '54', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-1000.0, PsetMW=0.0, QsetMVA=0.0)
    
    
    # Expand Elements

    lines_AC.set_index('Line_id', inplace=True)
    if exp == 'All':
        for line in list(grid.lines_AC):  # Create a copy of the list
            name = line.name
            line_cost = lines_AC.loc[name,'Cost MEUR']*10**6
            pyf.Expand_element(grid,name,N_b=N_b,N_i=N_i,N_max=N_max,base_cost=line_cost)
    else:
        for line in list(grid.lines_AC):  
            name = line.name
            if name not in exp:
                continue
            line_cost = lines_AC.loc[name,'Cost MEUR']*10**6
            pyf.Expand_element(grid,name,N_b=N_b,N_i=N_i,N_max=N_max,base_cost=line_cost)


    # Return the grid
    return grid,res
