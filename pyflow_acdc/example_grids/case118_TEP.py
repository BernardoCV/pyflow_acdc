import pyflow_acdc as pyf
import pandas as pd


def case118_TEP(exp='All',N_b=1,N_i=1,N_max=5):    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'PV', 'Voltage_0': 0.955, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.51, 'Reactive_load': 0.27, 'Node_id': '1.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.2, 'Reactive_load': 0.09, 'Node_id': '2.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.39, 'Reactive_load': 0.1, 'Node_id': '3.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.998, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.39, 'Reactive_load': 0.12, 'Node_id': '4.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '5.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': -0.4, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.99, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.52, 'Reactive_load': 0.22, 'Node_id': '6.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.19, 'Reactive_load': 0.02, 'Node_id': '7.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.015, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.28, 'Reactive_load': 0.0, 'Node_id': '8.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '9.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '10.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.7, 'Reactive_load': 0.23, 'Node_id': '11.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.99, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.47, 'Reactive_load': 0.1, 'Node_id': '12.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.34, 'Reactive_load': 0.16, 'Node_id': '13.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.14, 'Reactive_load': 0.01, 'Node_id': '14.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.97, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.9, 'Reactive_load': 0.3, 'Node_id': '15.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.25, 'Reactive_load': 0.1, 'Node_id': '16.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.11, 'Reactive_load': 0.03, 'Node_id': '17.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.973, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.6, 'Reactive_load': 0.34, 'Node_id': '18.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.962, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.45, 'Reactive_load': 0.25, 'Node_id': '19.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.18, 'Reactive_load': 0.03, 'Node_id': '20.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.14, 'Reactive_load': 0.08, 'Node_id': '21.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.1, 'Reactive_load': 0.05, 'Node_id': '22.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.07, 'Reactive_load': 0.03, 'Node_id': '23.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.992, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.13, 'Reactive_load': 0.0, 'Node_id': '24.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '25.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.015, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '26.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.968, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.71, 'Reactive_load': 0.13, 'Node_id': '27.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.17, 'Reactive_load': 0.07, 'Node_id': '28.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.24, 'Reactive_load': 0.04, 'Node_id': '29.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '30.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.967, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.43, 'Reactive_load': 0.27, 'Node_id': '31.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.963, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.59, 'Reactive_load': 0.23, 'Node_id': '32.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.23, 'Reactive_load': 0.09, 'Node_id': '33.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.984, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.59, 'Reactive_load': 0.26, 'Node_id': '34.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.14, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.33, 'Reactive_load': 0.09, 'Node_id': '35.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.31, 'Reactive_load': 0.17, 'Node_id': '36.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '37.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': -0.25, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '38.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.27, 'Reactive_load': 0.11, 'Node_id': '39.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.97, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.66, 'Reactive_load': 0.23, 'Node_id': '40.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.37, 'Reactive_load': 0.1, 'Node_id': '41.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.96, 'Reactive_load': 0.23, 'Node_id': '42.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.18, 'Reactive_load': 0.07, 'Node_id': '43.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.16, 'Reactive_load': 0.08, 'Node_id': '44.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.53, 'Reactive_load': 0.22, 'Node_id': '45.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.28, 'Reactive_load': 0.1, 'Node_id': '46.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.34, 'Reactive_load': 0.0, 'Node_id': '47.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.2, 'Reactive_load': 0.11, 'Node_id': '48.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.15, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.87, 'Reactive_load': 0.3, 'Node_id': '49.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.17, 'Reactive_load': 0.04, 'Node_id': '50.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.17, 'Reactive_load': 0.08, 'Node_id': '51.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.18, 'Reactive_load': 0.05, 'Node_id': '52.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.23, 'Reactive_load': 0.11, 'Node_id': '53.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.955, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.13, 'Reactive_load': 0.32, 'Node_id': '54.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.952, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.63, 'Reactive_load': 0.22, 'Node_id': '55.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.954, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.84, 'Reactive_load': 0.18, 'Node_id': '56.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.12, 'Reactive_load': 0.03, 'Node_id': '57.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.12, 'Reactive_load': 0.03, 'Node_id': '58.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.77, 'Reactive_load': 1.13, 'Node_id': '59.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.78, 'Reactive_load': 0.03, 'Node_id': '60.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.995, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '61.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.998, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.77, 'Reactive_load': 0.14, 'Node_id': '62.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '63.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '64.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '65.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.39, 'Reactive_load': 0.18, 'Node_id': '66.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.28, 'Reactive_load': 0.07, 'Node_id': '67.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '68.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'Slack', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '69.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.984, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.66, 'Reactive_load': 0.2, 'Node_id': '70.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '71.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.12, 'Reactive_load': 0.0, 'Node_id': '72.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.991, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.06, 'Reactive_load': 0.0, 'Node_id': '73.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.958, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.68, 'Reactive_load': 0.27, 'Node_id': '74.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.12, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.47, 'Reactive_load': 0.11, 'Node_id': '75.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.943, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.68, 'Reactive_load': 0.36, 'Node_id': '76.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.006, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.61, 'Reactive_load': 0.28, 'Node_id': '77.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.71, 'Reactive_load': 0.26, 'Node_id': '78.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.39, 'Reactive_load': 0.32, 'Node_id': '79.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.2, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.04, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.3, 'Reactive_load': 0.26, 'Node_id': '80.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 345.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '81.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.54, 'Reactive_load': 0.27, 'Node_id': '82.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.2, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.2, 'Reactive_load': 0.1, 'Node_id': '83.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.1, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.11, 'Reactive_load': 0.07, 'Node_id': '84.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.24, 'Reactive_load': 0.15, 'Node_id': '85.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.21, 'Reactive_load': 0.1, 'Node_id': '86.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.015, 'theta_0': 0.0, 'kV_base': 161.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '87.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.48, 'Reactive_load': 0.1, 'Node_id': '88.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '89.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.985, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.63, 'Reactive_load': 0.42, 'Node_id': '90.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.1, 'Reactive_load': 0.0, 'Node_id': '91.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.99, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.65, 'Reactive_load': 0.1, 'Node_id': '92.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.12, 'Reactive_load': 0.07, 'Node_id': '93.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.3, 'Reactive_load': 0.16, 'Node_id': '94.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.42, 'Reactive_load': 0.31, 'Node_id': '95.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.38, 'Reactive_load': 0.15, 'Node_id': '96.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.15, 'Reactive_load': 0.09, 'Node_id': '97.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.34, 'Reactive_load': 0.08, 'Node_id': '98.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.42, 'Reactive_load': 0.0, 'Node_id': '99.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.37, 'Reactive_load': 0.18, 'Node_id': '100.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.22, 'Reactive_load': 0.15, 'Node_id': '101.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.05, 'Reactive_load': 0.03, 'Node_id': '102.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.23, 'Reactive_load': 0.16, 'Node_id': '103.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.971, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.38, 'Reactive_load': 0.25, 'Node_id': '104.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.965, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.31, 'Reactive_load': 0.26, 'Node_id': '105.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.2, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.43, 'Reactive_load': 0.16, 'Node_id': '106.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.952, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.5, 'Reactive_load': 0.12, 'Node_id': '107.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.06, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.02, 'Reactive_load': 0.01, 'Node_id': '108.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.08, 'Reactive_load': 0.03, 'Node_id': '109.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.973, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.39, 'Reactive_load': 0.3, 'Node_id': '110.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.06, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': '111.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.975, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.68, 'Reactive_load': 0.13, 'Node_id': '112.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 0.993, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.06, 'Reactive_load': 0.0, 'Node_id': '113.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.08, 'Reactive_load': 0.03, 'Node_id': '114.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.22, 'Reactive_load': 0.07, 'Node_id': '115.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PV', 'Voltage_0': 1.005, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 1.84, 'Reactive_load': 0.0, 'Node_id': '116.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.2, 'Reactive_load': 0.08, 'Node_id': '117.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None},
        {'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.33, 'Reactive_load': 0.15, 'Node_id': '118.0', 'Umin': 0.94, 'Umax': 1.06, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': None, 'y_coord': None, 'PZ': None, 'geometry': None}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'Line_id': '1', 'fromNode': '1', 'toNode': '2', 'r': 0.0303, 'x': 0.0999, 'b': 0.0254, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.51'},
        {'Line_id': '2', 'fromNode': '1', 'toNode': '3', 'r': 0.0129, 'x': 0.0424, 'b': 0.01082, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.77'},
        {'Line_id': '3', 'fromNode': '4', 'toNode': '5', 'r': 0.00176, 'x': 0.00798, 'b': 0.0021, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.49'},
        {'Line_id': '4', 'fromNode': '3', 'toNode': '5', 'r': 0.0241, 'x': 0.108, 'b': 0.0284, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.61'},
        {'Line_id': '5', 'fromNode': '5', 'toNode': '6', 'r': 0.0119, 'x': 0.054, 'b': 0.01426, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.3'},
        {'Line_id': '6', 'fromNode': '6', 'toNode': '7', 'r': 0.00459, 'x': 0.0208, 'b': 0.0055, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.27'},
        {'Line_id': '7', 'fromNode': '8', 'toNode': '9', 'r': 0.00244, 'x': 0.0305, 'b': 1.162, 'MVA_rating': 711.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.65'},
        {'Line_id': '8', 'fromNode': '8', 'toNode': '5', 'r': 0.0, 'x': 0.0267, 'b': 0.0, 'MVA_rating': 1099.0, 'm': 0.985, 'shift': 0, 'Cost MEUR': '1.34'},
        {'Line_id': '9', 'fromNode': '9', 'toNode': '10', 'r': 0.00258, 'x': 0.0322, 'b': 1.23, 'MVA_rating': 710.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.74'},
        {'Line_id': '10', 'fromNode': '4', 'toNode': '11', 'r': 0.0209, 'x': 0.0688, 'b': 0.01748, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.49'},
        {'Line_id': '11', 'fromNode': '5', 'toNode': '11', 'r': 0.0203, 'x': 0.0682, 'b': 0.01738, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.43'},
        {'Line_id': '12', 'fromNode': '11', 'toNode': '12', 'r': 0.00595, 'x': 0.0196, 'b': 0.00502, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.28'},
        {'Line_id': '13', 'fromNode': '2', 'toNode': '12', 'r': 0.0187, 'x': 0.0616, 'b': 0.01572, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.02'},
        {'Line_id': '14', 'fromNode': '3', 'toNode': '12', 'r': 0.0484, 'x': 0.16, 'b': 0.0406, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.42'},
        {'Line_id': '15', 'fromNode': '7', 'toNode': '12', 'r': 0.00862, 'x': 0.034, 'b': 0.00874, 'MVA_rating': 164.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.13'},
        {'Line_id': '16', 'fromNode': '11', 'toNode': '13', 'r': 0.02225, 'x': 0.0731, 'b': 0.01876, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.77'},
        {'Line_id': '17', 'fromNode': '12', 'toNode': '14', 'r': 0.0215, 'x': 0.0707, 'b': 0.01816, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.61'},
        {'Line_id': '18', 'fromNode': '13', 'toNode': '15', 'r': 0.0744, 'x': 0.2444, 'b': 0.06268, 'MVA_rating': 115.0, 'm': 1, 'shift': 0, 'Cost MEUR': '15.94'},
        {'Line_id': '19', 'fromNode': '14', 'toNode': '15', 'r': 0.0595, 'x': 0.195, 'b': 0.0502, 'MVA_rating': 144.0, 'm': 1, 'shift': 0, 'Cost MEUR': '12.73'},
        {'Line_id': '20', 'fromNode': '12', 'toNode': '16', 'r': 0.0212, 'x': 0.0834, 'b': 0.0214, 'MVA_rating': 164.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.23'},
        {'Line_id': '21', 'fromNode': '15', 'toNode': '17', 'r': 0.0132, 'x': 0.0437, 'b': 0.0444, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.85'},
        {'Line_id': '22', 'fromNode': '16', 'toNode': '17', 'r': 0.0454, 'x': 0.1801, 'b': 0.0466, 'MVA_rating': 158.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.28'},
        {'Line_id': '23', 'fromNode': '17', 'toNode': '18', 'r': 0.0123, 'x': 0.0505, 'b': 0.01298, 'MVA_rating': 167.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.14'},
        {'Line_id': '24', 'fromNode': '18', 'toNode': '19', 'r': 0.01119, 'x': 0.0493, 'b': 0.01142, 'MVA_rating': 173.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.03'},
        {'Line_id': '25', 'fromNode': '19', 'toNode': '20', 'r': 0.0252, 'x': 0.117, 'b': 0.0298, 'MVA_rating': 178.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.11'},
        {'Line_id': '26', 'fromNode': '15', 'toNode': '19', 'r': 0.012, 'x': 0.0394, 'b': 0.0101, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.57'},
        {'Line_id': '27', 'fromNode': '20', 'toNode': '21', 'r': 0.0183, 'x': 0.0849, 'b': 0.0216, 'MVA_rating': 177.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.16'},
        {'Line_id': '28', 'fromNode': '21', 'toNode': '22', 'r': 0.0209, 'x': 0.097, 'b': 0.0246, 'MVA_rating': 178.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.9'},
        {'Line_id': '29', 'fromNode': '22', 'toNode': '23', 'r': 0.0342, 'x': 0.159, 'b': 0.0404, 'MVA_rating': 178.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.66'},
        {'Line_id': '30', 'fromNode': '23', 'toNode': '24', 'r': 0.0135, 'x': 0.0492, 'b': 0.0498, 'MVA_rating': 158.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.14'},
        {'Line_id': '31', 'fromNode': '23', 'toNode': '25', 'r': 0.0156, 'x': 0.08, 'b': 0.0864, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.78'},
        {'Line_id': '32', 'fromNode': '26', 'toNode': '25', 'r': 0.0, 'x': 0.0382, 'b': 0.0, 'MVA_rating': 768.0, 'm': 0.96, 'shift': 0, 'Cost MEUR': '1.91'},
        {'Line_id': '33', 'fromNode': '25', 'toNode': '27', 'r': 0.0318, 'x': 0.163, 'b': 0.1764, 'MVA_rating': 177.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.74'},
        {'Line_id': '34', 'fromNode': '27', 'toNode': '28', 'r': 0.01913, 'x': 0.0855, 'b': 0.0216, 'MVA_rating': 174.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.23'},
        {'Line_id': '35', 'fromNode': '28', 'toNode': '29', 'r': 0.0237, 'x': 0.0943, 'b': 0.0238, 'MVA_rating': 165.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.9'},
        {'Line_id': '36', 'fromNode': '30', 'toNode': '17', 'r': 0.0, 'x': 0.0388, 'b': 0.0, 'MVA_rating': 756.0, 'm': 0.96, 'shift': 0, 'Cost MEUR': '1.94'},
        {'Line_id': '37', 'fromNode': '8', 'toNode': '30', 'r': 0.00431, 'x': 0.0504, 'b': 0.514, 'MVA_rating': 580.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.74'},
        {'Line_id': '38', 'fromNode': '26', 'toNode': '30', 'r': 0.00799, 'x': 0.086, 'b': 0.908, 'MVA_rating': 340.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.7'},
        {'Line_id': '39', 'fromNode': '17', 'toNode': '31', 'r': 0.0474, 'x': 0.1563, 'b': 0.0399, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.19'},
        {'Line_id': '40', 'fromNode': '29', 'toNode': '31', 'r': 0.0108, 'x': 0.0331, 'b': 0.0083, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.2'},
        {'Line_id': '41', 'fromNode': '23', 'toNode': '32', 'r': 0.0317, 'x': 0.1153, 'b': 0.1173, 'MVA_rating': 158.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.35'},
        {'Line_id': '42', 'fromNode': '31', 'toNode': '32', 'r': 0.0298, 'x': 0.0985, 'b': 0.0251, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.42'},
        {'Line_id': '43', 'fromNode': '27', 'toNode': '32', 'r': 0.0229, 'x': 0.0755, 'b': 0.01926, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.92'},
        {'Line_id': '44', 'fromNode': '15', 'toNode': '33', 'r': 0.038, 'x': 0.1244, 'b': 0.03194, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.12'},
        {'Line_id': '45', 'fromNode': '19', 'toNode': '34', 'r': 0.0752, 'x': 0.247, 'b': 0.0632, 'MVA_rating': 114.0, 'm': 1, 'shift': 0, 'Cost MEUR': '16.11'},
        {'Line_id': '46', 'fromNode': '35', 'toNode': '36', 'r': 0.00224, 'x': 0.0102, 'b': 0.00268, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.62'},
        {'Line_id': '47', 'fromNode': '35', 'toNode': '37', 'r': 0.011, 'x': 0.0497, 'b': 0.01318, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.04'},
        {'Line_id': '48', 'fromNode': '33', 'toNode': '37', 'r': 0.0415, 'x': 0.142, 'b': 0.0366, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.18'},
        {'Line_id': '49', 'fromNode': '34', 'toNode': '36', 'r': 0.00871, 'x': 0.0268, 'b': 0.00568, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.78'},
        {'Line_id': '50', 'fromNode': '34', 'toNode': '37', 'r': 0.00256, 'x': 0.0094, 'b': 0.00984, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.6'},
        {'Line_id': '51', 'fromNode': '38', 'toNode': '37', 'r': 0.0, 'x': 0.0375, 'b': 0.0, 'MVA_rating': 783.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': '1.88'},
        {'Line_id': '52', 'fromNode': '37', 'toNode': '39', 'r': 0.0321, 'x': 0.106, 'b': 0.027, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.91'},
        {'Line_id': '53', 'fromNode': '37', 'toNode': '40', 'r': 0.0593, 'x': 0.168, 'b': 0.042, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.37'},
        {'Line_id': '54', 'fromNode': '30', 'toNode': '38', 'r': 0.00464, 'x': 0.054, 'b': 0.422, 'MVA_rating': 542.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.93'},
        {'Line_id': '55', 'fromNode': '39', 'toNode': '40', 'r': 0.0184, 'x': 0.0605, 'b': 0.01552, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.95'},
        {'Line_id': '56', 'fromNode': '40', 'toNode': '41', 'r': 0.0145, 'x': 0.0487, 'b': 0.01222, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.16'},
        {'Line_id': '57', 'fromNode': '40', 'toNode': '42', 'r': 0.0555, 'x': 0.183, 'b': 0.0466, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.93'},
        {'Line_id': '58', 'fromNode': '41', 'toNode': '42', 'r': 0.041, 'x': 0.135, 'b': 0.0344, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.8'},
        {'Line_id': '59', 'fromNode': '43', 'toNode': '44', 'r': 0.0608, 'x': 0.2454, 'b': 0.06068, 'MVA_rating': 117.0, 'm': 1, 'shift': 0, 'Cost MEUR': '15.31'},
        {'Line_id': '60', 'fromNode': '34', 'toNode': '43', 'r': 0.0413, 'x': 0.1681, 'b': 0.04226, 'MVA_rating': 167.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.47'},
        {'Line_id': '61', 'fromNode': '44', 'toNode': '45', 'r': 0.0224, 'x': 0.0901, 'b': 0.0224, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.63'},
        {'Line_id': '62', 'fromNode': '45', 'toNode': '46', 'r': 0.04, 'x': 0.1356, 'b': 0.0332, 'MVA_rating': 153.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.78'},
        {'Line_id': '63', 'fromNode': '46', 'toNode': '47', 'r': 0.038, 'x': 0.127, 'b': 0.0316, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.25'},
        {'Line_id': '64', 'fromNode': '46', 'toNode': '48', 'r': 0.0601, 'x': 0.189, 'b': 0.0472, 'MVA_rating': 148.0, 'm': 1, 'shift': 0, 'Cost MEUR': '12.46'},
        {'Line_id': '65', 'fromNode': '47', 'toNode': '49', 'r': 0.0191, 'x': 0.0625, 'b': 0.01604, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.08'},
        {'Line_id': '66', 'fromNode': '42', 'toNode': '49', 'r': 0.0715, 'x': 0.323, 'b': 0.086, 'MVA_rating': 89.0, 'm': 1, 'shift': 0, 'Cost MEUR': '19.73'},
        {'Line_id': '67', 'fromNode': '42', 'toNode': '49', 'r': 0.0715, 'x': 0.323, 'b': 0.086, 'MVA_rating': 89.0, 'm': 1, 'shift': 0, 'Cost MEUR': '19.73'},
        {'Line_id': '68', 'fromNode': '45', 'toNode': '49', 'r': 0.0684, 'x': 0.186, 'b': 0.0444, 'MVA_rating': 138.0, 'm': 1, 'shift': 0, 'Cost MEUR': '12.72'},
        {'Line_id': '69', 'fromNode': '48', 'toNode': '49', 'r': 0.0179, 'x': 0.0505, 'b': 0.01258, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.42'},
        {'Line_id': '70', 'fromNode': '49', 'toNode': '50', 'r': 0.0267, 'x': 0.0752, 'b': 0.01874, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.1'},
        {'Line_id': '71', 'fromNode': '49', 'toNode': '51', 'r': 0.0486, 'x': 0.137, 'b': 0.0342, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.28'},
        {'Line_id': '72', 'fromNode': '51', 'toNode': '52', 'r': 0.0203, 'x': 0.0588, 'b': 0.01396, 'MVA_rating': 142.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.96'},
        {'Line_id': '73', 'fromNode': '52', 'toNode': '53', 'r': 0.0405, 'x': 0.1635, 'b': 0.04058, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.2'},
        {'Line_id': '74', 'fromNode': '53', 'toNode': '54', 'r': 0.0263, 'x': 0.122, 'b': 0.031, 'MVA_rating': 177.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.42'},
        {'Line_id': '75', 'fromNode': '49', 'toNode': '54', 'r': 0.073, 'x': 0.289, 'b': 0.0738, 'MVA_rating': 99.0, 'm': 1, 'shift': 0, 'Cost MEUR': '18.1'},
        {'Line_id': '76', 'fromNode': '49', 'toNode': '54', 'r': 0.0869, 'x': 0.291, 'b': 0.073, 'MVA_rating': 97.0, 'm': 1, 'shift': 0, 'Cost MEUR': '18.9'},
        {'Line_id': '77', 'fromNode': '54', 'toNode': '55', 'r': 0.0169, 'x': 0.0707, 'b': 0.0202, 'MVA_rating': 169.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.38'},
        {'Line_id': '78', 'fromNode': '54', 'toNode': '56', 'r': 0.00275, 'x': 0.00955, 'b': 0.00732, 'MVA_rating': 155.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.62'},
        {'Line_id': '79', 'fromNode': '55', 'toNode': '56', 'r': 0.00488, 'x': 0.0151, 'b': 0.00374, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1'},
        {'Line_id': '80', 'fromNode': '56', 'toNode': '57', 'r': 0.0343, 'x': 0.0966, 'b': 0.0242, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.55'},
        {'Line_id': '81', 'fromNode': '50', 'toNode': '57', 'r': 0.0474, 'x': 0.134, 'b': 0.0332, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.07'},
        {'Line_id': '82', 'fromNode': '56', 'toNode': '58', 'r': 0.0343, 'x': 0.0966, 'b': 0.0242, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.55'},
        {'Line_id': '83', 'fromNode': '51', 'toNode': '58', 'r': 0.0255, 'x': 0.0719, 'b': 0.01788, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.87'},
        {'Line_id': '84', 'fromNode': '54', 'toNode': '59', 'r': 0.0503, 'x': 0.2293, 'b': 0.0598, 'MVA_rating': 125.0, 'm': 1, 'shift': 0, 'Cost MEUR': '13.98'},
        {'Line_id': '85', 'fromNode': '56', 'toNode': '59', 'r': 0.0825, 'x': 0.251, 'b': 0.0569, 'MVA_rating': 112.0, 'm': 1, 'shift': 0, 'Cost MEUR': '16.68'},
        {'Line_id': '86', 'fromNode': '56', 'toNode': '59', 'r': 0.0803, 'x': 0.239, 'b': 0.0536, 'MVA_rating': 117.0, 'm': 1, 'shift': 0, 'Cost MEUR': '15.97'},
        {'Line_id': '87', 'fromNode': '55', 'toNode': '59', 'r': 0.04739, 'x': 0.2158, 'b': 0.05646, 'MVA_rating': 133.0, 'm': 1, 'shift': 0, 'Cost MEUR': '13.16'},
        {'Line_id': '88', 'fromNode': '59', 'toNode': '60', 'r': 0.0317, 'x': 0.145, 'b': 0.0376, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.84'},
        {'Line_id': '89', 'fromNode': '59', 'toNode': '61', 'r': 0.0328, 'x': 0.15, 'b': 0.0388, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.14'},
        {'Line_id': '90', 'fromNode': '60', 'toNode': '61', 'r': 0.00264, 'x': 0.0135, 'b': 0.01456, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.81'},
        {'Line_id': '91', 'fromNode': '60', 'toNode': '62', 'r': 0.0123, 'x': 0.0561, 'b': 0.01468, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.42'},
        {'Line_id': '92', 'fromNode': '61', 'toNode': '62', 'r': 0.00824, 'x': 0.0376, 'b': 0.0098, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.29'},
        {'Line_id': '93', 'fromNode': '63', 'toNode': '59', 'r': 0.0, 'x': 0.0386, 'b': 0.0, 'MVA_rating': 760.0, 'm': 0.96, 'shift': 0, 'Cost MEUR': '1.93'},
        {'Line_id': '94', 'fromNode': '63', 'toNode': '64', 'r': 0.00172, 'x': 0.02, 'b': 0.216, 'MVA_rating': 687.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.09'},
        {'Line_id': '95', 'fromNode': '64', 'toNode': '61', 'r': 0.0, 'x': 0.0268, 'b': 0.0, 'MVA_rating': 1095.0, 'm': 0.985, 'shift': 0, 'Cost MEUR': '1.34'},
        {'Line_id': '96', 'fromNode': '38', 'toNode': '65', 'r': 0.00901, 'x': 0.0986, 'b': 1.046, 'MVA_rating': 297.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.38'},
        {'Line_id': '97', 'fromNode': '64', 'toNode': '65', 'r': 0.00269, 'x': 0.0302, 'b': 0.38, 'MVA_rating': 675.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.65'},
        {'Line_id': '98', 'fromNode': '49', 'toNode': '66', 'r': 0.018, 'x': 0.0919, 'b': 0.0248, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.5'},
        {'Line_id': '99', 'fromNode': '49', 'toNode': '66', 'r': 0.018, 'x': 0.0919, 'b': 0.0248, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.5'},
        {'Line_id': '100', 'fromNode': '62', 'toNode': '66', 'r': 0.0482, 'x': 0.218, 'b': 0.0578, 'MVA_rating': 132.0, 'm': 1, 'shift': 0, 'Cost MEUR': '13.31'},
        {'Line_id': '101', 'fromNode': '62', 'toNode': '67', 'r': 0.0258, 'x': 0.117, 'b': 0.031, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.14'},
        {'Line_id': '102', 'fromNode': '65', 'toNode': '66', 'r': 0.0, 'x': 0.037, 'b': 0.0, 'MVA_rating': 793.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': '1.85'},
        {'Line_id': '103', 'fromNode': '66', 'toNode': '67', 'r': 0.0224, 'x': 0.1015, 'b': 0.02682, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.2'},
        {'Line_id': '104', 'fromNode': '65', 'toNode': '68', 'r': 0.00138, 'x': 0.016, 'b': 0.638, 'MVA_rating': 686.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.87'},
        {'Line_id': '105', 'fromNode': '47', 'toNode': '69', 'r': 0.0844, 'x': 0.2778, 'b': 0.07092, 'MVA_rating': 102.0, 'm': 1, 'shift': 0, 'Cost MEUR': '18.11'},
        {'Line_id': '106', 'fromNode': '49', 'toNode': '69', 'r': 0.0985, 'x': 0.324, 'b': 0.0828, 'MVA_rating': 87.0, 'm': 1, 'shift': 0, 'Cost MEUR': '21.13'},
        {'Line_id': '107', 'fromNode': '68', 'toNode': '69', 'r': 0.0, 'x': 0.037, 'b': 0.0, 'MVA_rating': 793.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': '1.85'},
        {'Line_id': '108', 'fromNode': '69', 'toNode': '70', 'r': 0.03, 'x': 0.127, 'b': 0.122, 'MVA_rating': 170.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.85'},
        {'Line_id': '109', 'fromNode': '24', 'toNode': '70', 'r': 0.00221, 'x': 0.4115, 'b': 0.10198, 'MVA_rating': 72.0, 'm': 1, 'shift': 0, 'Cost MEUR': '20.69'},
        {'Line_id': '110', 'fromNode': '70', 'toNode': '71', 'r': 0.00882, 'x': 0.0355, 'b': 0.00878, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.22'},
        {'Line_id': '111', 'fromNode': '24', 'toNode': '72', 'r': 0.0488, 'x': 0.196, 'b': 0.0488, 'MVA_rating': 146.0, 'm': 1, 'shift': 0, 'Cost MEUR': '12.24'},
        {'Line_id': '112', 'fromNode': '71', 'toNode': '72', 'r': 0.0446, 'x': 0.18, 'b': 0.04444, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.23'},
        {'Line_id': '113', 'fromNode': '71', 'toNode': '73', 'r': 0.00866, 'x': 0.0454, 'b': 0.01178, 'MVA_rating': 188.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.71'},
        {'Line_id': '114', 'fromNode': '70', 'toNode': '74', 'r': 0.0401, 'x': 0.1323, 'b': 0.03368, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.62'},
        {'Line_id': '115', 'fromNode': '70', 'toNode': '75', 'r': 0.0428, 'x': 0.141, 'b': 0.036, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.19'},
        {'Line_id': '116', 'fromNode': '69', 'toNode': '75', 'r': 0.0405, 'x': 0.122, 'b': 0.124, 'MVA_rating': 145.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.13'},
        {'Line_id': '117', 'fromNode': '74', 'toNode': '75', 'r': 0.0123, 'x': 0.0406, 'b': 0.01034, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.65'},
        {'Line_id': '118', 'fromNode': '76', 'toNode': '77', 'r': 0.0444, 'x': 0.148, 'b': 0.0368, 'MVA_rating': 152.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.62'},
        {'Line_id': '119', 'fromNode': '69', 'toNode': '77', 'r': 0.0309, 'x': 0.101, 'b': 0.1038, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.6'},
        {'Line_id': '120', 'fromNode': '75', 'toNode': '77', 'r': 0.0601, 'x': 0.1999, 'b': 0.04978, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': '13'},
        {'Line_id': '121', 'fromNode': '77', 'toNode': '78', 'r': 0.00376, 'x': 0.0124, 'b': 0.01264, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.81'},
        {'Line_id': '122', 'fromNode': '78', 'toNode': '79', 'r': 0.00546, 'x': 0.0244, 'b': 0.00648, 'MVA_rating': 174.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.5'},
        {'Line_id': '123', 'fromNode': '77', 'toNode': '80', 'r': 0.017, 'x': 0.0485, 'b': 0.0472, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.28'},
        {'Line_id': '124', 'fromNode': '77', 'toNode': '80', 'r': 0.0294, 'x': 0.105, 'b': 0.0228, 'MVA_rating': 157.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.72'},
        {'Line_id': '125', 'fromNode': '79', 'toNode': '80', 'r': 0.0156, 'x': 0.0704, 'b': 0.0187, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.3'},
        {'Line_id': '126', 'fromNode': '68', 'toNode': '81', 'r': 0.00175, 'x': 0.0202, 'b': 0.808, 'MVA_rating': 684.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.1'},
        {'Line_id': '127', 'fromNode': '81', 'toNode': '80', 'r': 0.0, 'x': 0.037, 'b': 0.0, 'MVA_rating': 793.0, 'm': 0.935, 'shift': 0, 'Cost MEUR': '1.85'},
        {'Line_id': '128', 'fromNode': '77', 'toNode': '82', 'r': 0.0298, 'x': 0.0853, 'b': 0.08174, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.76'},
        {'Line_id': '129', 'fromNode': '82', 'toNode': '83', 'r': 0.0112, 'x': 0.03665, 'b': 0.03796, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.4'},
        {'Line_id': '130', 'fromNode': '83', 'toNode': '84', 'r': 0.0625, 'x': 0.132, 'b': 0.0258, 'MVA_rating': 122.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.73'},
        {'Line_id': '131', 'fromNode': '83', 'toNode': '85', 'r': 0.043, 'x': 0.148, 'b': 0.0348, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.55'},
        {'Line_id': '132', 'fromNode': '84', 'toNode': '85', 'r': 0.0302, 'x': 0.0641, 'b': 0.01234, 'MVA_rating': 122.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.72'},
        {'Line_id': '133', 'fromNode': '85', 'toNode': '86', 'r': 0.035, 'x': 0.123, 'b': 0.0276, 'MVA_rating': 156.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.9'},
        {'Line_id': '134', 'fromNode': '86', 'toNode': '87', 'r': 0.02828, 'x': 0.2074, 'b': 0.0445, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.79'},
        {'Line_id': '135', 'fromNode': '85', 'toNode': '88', 'r': 0.02, 'x': 0.102, 'b': 0.0276, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.1'},
        {'Line_id': '136', 'fromNode': '85', 'toNode': '89', 'r': 0.0239, 'x': 0.173, 'b': 0.047, 'MVA_rating': 168.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.85'},
        {'Line_id': '137', 'fromNode': '88', 'toNode': '89', 'r': 0.0139, 'x': 0.0712, 'b': 0.01934, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.26'},
        {'Line_id': '138', 'fromNode': '89', 'toNode': '90', 'r': 0.0518, 'x': 0.188, 'b': 0.0528, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.99'},
        {'Line_id': '139', 'fromNode': '89', 'toNode': '90', 'r': 0.0238, 'x': 0.0997, 'b': 0.106, 'MVA_rating': 169.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.18'},
        {'Line_id': '140', 'fromNode': '90', 'toNode': '91', 'r': 0.0254, 'x': 0.0836, 'b': 0.0214, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.45'},
        {'Line_id': '141', 'fromNode': '89', 'toNode': '92', 'r': 0.0099, 'x': 0.0505, 'b': 0.0548, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.02'},
        {'Line_id': '142', 'fromNode': '89', 'toNode': '92', 'r': 0.0393, 'x': 0.1581, 'b': 0.0414, 'MVA_rating': 166.0, 'm': 1, 'shift': 0, 'Cost MEUR': '9.87'},
        {'Line_id': '143', 'fromNode': '91', 'toNode': '92', 'r': 0.0387, 'x': 0.1272, 'b': 0.03268, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.3'},
        {'Line_id': '144', 'fromNode': '92', 'toNode': '93', 'r': 0.0258, 'x': 0.0848, 'b': 0.0218, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.53'},
        {'Line_id': '145', 'fromNode': '92', 'toNode': '94', 'r': 0.0481, 'x': 0.158, 'b': 0.0406, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.31'},
        {'Line_id': '146', 'fromNode': '93', 'toNode': '94', 'r': 0.0223, 'x': 0.0732, 'b': 0.01876, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.78'},
        {'Line_id': '147', 'fromNode': '94', 'toNode': '95', 'r': 0.0132, 'x': 0.0434, 'b': 0.0111, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.83'},
        {'Line_id': '148', 'fromNode': '80', 'toNode': '96', 'r': 0.0356, 'x': 0.182, 'b': 0.0494, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.88'},
        {'Line_id': '149', 'fromNode': '82', 'toNode': '96', 'r': 0.0162, 'x': 0.053, 'b': 0.0544, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.46'},
        {'Line_id': '150', 'fromNode': '94', 'toNode': '96', 'r': 0.0269, 'x': 0.0869, 'b': 0.023, 'MVA_rating': 149.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.69'},
        {'Line_id': '151', 'fromNode': '80', 'toNode': '97', 'r': 0.0183, 'x': 0.0934, 'b': 0.0254, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.59'},
        {'Line_id': '152', 'fromNode': '80', 'toNode': '98', 'r': 0.0238, 'x': 0.108, 'b': 0.0286, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.59'},
        {'Line_id': '153', 'fromNode': '80', 'toNode': '99', 'r': 0.0454, 'x': 0.206, 'b': 0.0546, 'MVA_rating': 140.0, 'm': 1, 'shift': 0, 'Cost MEUR': '12.57'},
        {'Line_id': '154', 'fromNode': '92', 'toNode': '100', 'r': 0.0648, 'x': 0.295, 'b': 0.0472, 'MVA_rating': 98.0, 'm': 1, 'shift': 0, 'Cost MEUR': '17.99'},
        {'Line_id': '155', 'fromNode': '94', 'toNode': '100', 'r': 0.0178, 'x': 0.058, 'b': 0.0604, 'MVA_rating': 150.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.79'},
        {'Line_id': '156', 'fromNode': '95', 'toNode': '96', 'r': 0.0171, 'x': 0.0547, 'b': 0.01474, 'MVA_rating': 149.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.59'},
        {'Line_id': '157', 'fromNode': '96', 'toNode': '97', 'r': 0.0173, 'x': 0.0885, 'b': 0.024, 'MVA_rating': 186.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.29'},
        {'Line_id': '158', 'fromNode': '98', 'toNode': '100', 'r': 0.0397, 'x': 0.179, 'b': 0.0476, 'MVA_rating': 160.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.94'},
        {'Line_id': '159', 'fromNode': '99', 'toNode': '100', 'r': 0.018, 'x': 0.0813, 'b': 0.0216, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.97'},
        {'Line_id': '160', 'fromNode': '100', 'toNode': '101', 'r': 0.0277, 'x': 0.1262, 'b': 0.0328, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '7.7'},
        {'Line_id': '161', 'fromNode': '92', 'toNode': '102', 'r': 0.0123, 'x': 0.0559, 'b': 0.01464, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.41'},
        {'Line_id': '162', 'fromNode': '101', 'toNode': '102', 'r': 0.0246, 'x': 0.112, 'b': 0.0294, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '6.83'},
        {'Line_id': '163', 'fromNode': '100', 'toNode': '103', 'r': 0.016, 'x': 0.0525, 'b': 0.0536, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.43'},
        {'Line_id': '164', 'fromNode': '100', 'toNode': '104', 'r': 0.0451, 'x': 0.204, 'b': 0.0541, 'MVA_rating': 141.0, 'm': 1, 'shift': 0, 'Cost MEUR': '12.46'},
        {'Line_id': '165', 'fromNode': '103', 'toNode': '104', 'r': 0.0466, 'x': 0.1584, 'b': 0.0407, 'MVA_rating': 153.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.25'},
        {'Line_id': '166', 'fromNode': '103', 'toNode': '105', 'r': 0.0535, 'x': 0.1625, 'b': 0.0408, 'MVA_rating': 145.0, 'm': 1, 'shift': 0, 'Cost MEUR': '10.8'},
        {'Line_id': '167', 'fromNode': '100', 'toNode': '106', 'r': 0.0605, 'x': 0.229, 'b': 0.062, 'MVA_rating': 124.0, 'm': 1, 'shift': 0, 'Cost MEUR': '14.48'},
        {'Line_id': '168', 'fromNode': '104', 'toNode': '105', 'r': 0.00994, 'x': 0.0378, 'b': 0.00986, 'MVA_rating': 161.0, 'm': 1, 'shift': 0, 'Cost MEUR': '2.39'},
        {'Line_id': '169', 'fromNode': '105', 'toNode': '106', 'r': 0.014, 'x': 0.0547, 'b': 0.01434, 'MVA_rating': 164.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.44'},
        {'Line_id': '170', 'fromNode': '105', 'toNode': '107', 'r': 0.053, 'x': 0.183, 'b': 0.0472, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.8'},
        {'Line_id': '171', 'fromNode': '105', 'toNode': '108', 'r': 0.0261, 'x': 0.0703, 'b': 0.01844, 'MVA_rating': 137.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.82'},
        {'Line_id': '172', 'fromNode': '106', 'toNode': '107', 'r': 0.053, 'x': 0.183, 'b': 0.0472, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.8'},
        {'Line_id': '173', 'fromNode': '108', 'toNode': '109', 'r': 0.0105, 'x': 0.0288, 'b': 0.0076, 'MVA_rating': 138.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.97'},
        {'Line_id': '174', 'fromNode': '103', 'toNode': '110', 'r': 0.03906, 'x': 0.1813, 'b': 0.0461, 'MVA_rating': 159.0, 'm': 1, 'shift': 0, 'Cost MEUR': '11.02'},
        {'Line_id': '175', 'fromNode': '109', 'toNode': '110', 'r': 0.0278, 'x': 0.0762, 'b': 0.0202, 'MVA_rating': 138.0, 'm': 1, 'shift': 0, 'Cost MEUR': '5.2'},
        {'Line_id': '176', 'fromNode': '110', 'toNode': '111', 'r': 0.022, 'x': 0.0755, 'b': 0.02, 'MVA_rating': 154.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.88'},
        {'Line_id': '177', 'fromNode': '110', 'toNode': '112', 'r': 0.0247, 'x': 0.064, 'b': 0.062, 'MVA_rating': 135.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.44'},
        {'Line_id': '178', 'fromNode': '17', 'toNode': '113', 'r': 0.00913, 'x': 0.0301, 'b': 0.00768, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '1.96'},
        {'Line_id': '179', 'fromNode': '32', 'toNode': '113', 'r': 0.0615, 'x': 0.203, 'b': 0.0518, 'MVA_rating': 139.0, 'm': 1, 'shift': 0, 'Cost MEUR': '13.23'},
        {'Line_id': '180', 'fromNode': '32', 'toNode': '114', 'r': 0.0135, 'x': 0.0612, 'b': 0.01628, 'MVA_rating': 176.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.74'},
        {'Line_id': '181', 'fromNode': '27', 'toNode': '115', 'r': 0.0164, 'x': 0.0741, 'b': 0.01972, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': '4.53'},
        {'Line_id': '182', 'fromNode': '114', 'toNode': '115', 'r': 0.0023, 'x': 0.0104, 'b': 0.00276, 'MVA_rating': 175.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.64'},
        {'Line_id': '183', 'fromNode': '68', 'toNode': '116', 'r': 0.00034, 'x': 0.00405, 'b': 0.164, 'MVA_rating': 7218.0, 'm': 1, 'shift': 0, 'Cost MEUR': '0.22'},
        {'Line_id': '184', 'fromNode': '12', 'toNode': '117', 'r': 0.0329, 'x': 0.14, 'b': 0.0358, 'MVA_rating': 170.0, 'm': 1, 'shift': 0, 'Cost MEUR': '8.65'},
        {'Line_id': '185', 'fromNode': '75', 'toNode': '118', 'r': 0.0145, 'x': 0.0481, 'b': 0.01198, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.13'},
        {'Line_id': '186', 'fromNode': '76', 'toNode': '118', 'r': 0.0164, 'x': 0.0544, 'b': 0.01356, 'MVA_rating': 151.0, 'm': 1, 'shift': 0, 'Cost MEUR': '3.54'}
        ]
    lines_AC = pd.DataFrame(lines_AC_data)

    nodes_DC = None

    lines_DC = None

    Converters_ACDC = None

    # Create the grid
    [grid, res] = pyf.Create_grid_from_data(S_base, nodes_AC, lines_AC, nodes_DC, lines_DC, Converters_ACDC, data_in='pu')
    grid.name = 'case118_TEP'
    
    # Assign Price Zones to Nodes
    for index, row in nodes_AC.iterrows():
        node_name = nodes_AC.at[index, 'Node_id']
        price_zone = nodes_AC.at[index, 'PZ']
        ACDC = 'AC'
        if price_zone is not None:
            pyf.assign_nodeToPrice_Zone(grid, node_name, price_zone,ACDC)
    
    
    
    # Add Generators
    pyf.add_gen(grid, '1.0', '1', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=15.0, MVArmin=-5.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '4.0', '2', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '6.0', '3', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=50.0, MVArmin=-13.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '8.0', '4', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '10.0', '5', lf=20.0, qf=0.0222222222, MWmax=550.0, MWmin=0.0, MVArmax=200.0, MVArmin=-147.0, PsetMW=450.0, QsetMVA=0.0)
    pyf.add_gen(grid, '12.0', '6', lf=20.0, qf=0.117647059, MWmax=185.0, MWmin=0.0, MVArmax=120.0, MVArmin=-35.0, PsetMW=85.0, QsetMVA=0.0)
    pyf.add_gen(grid, '15.0', '7', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=30.0, MVArmin=-10.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '18.0', '8', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=50.0, MVArmin=-16.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '19.0', '9', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=24.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '24.0', '10', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '25.0', '11', lf=20.0, qf=0.0454545455, MWmax=320.0, MWmin=0.0, MVArmax=140.0, MVArmin=-47.0, PsetMW=220.00000000000003, QsetMVA=0.0)
    pyf.add_gen(grid, '26.0', '12', lf=20.0, qf=0.0318471338, MWmax=413.99999999999994, MWmin=0.0, MVArmax=1000.0, MVArmin=-1000.0, PsetMW=314.0, QsetMVA=0.0)
    pyf.add_gen(grid, '27.0', '13', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '31.0', '14', lf=20.0, qf=1.42857143, MWmax=107.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=7.000000000000001, QsetMVA=0.0)
    pyf.add_gen(grid, '32.0', '15', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=42.0, MVArmin=-14.000000000000002, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '34.0', '16', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=24.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '36.0', '17', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=24.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '40.0', '18', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '42.0', '19', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '46.0', '20', lf=20.0, qf=0.526315789, MWmax=119.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=19.0, QsetMVA=0.0)
    pyf.add_gen(grid, '49.0', '21', lf=20.0, qf=0.0490196078, MWmax=304.0, MWmin=0.0, MVArmax=210.0, MVArmin=-85.0, PsetMW=204.0, QsetMVA=0.0)
    pyf.add_gen(grid, '54.0', '22', lf=20.0, qf=0.208333333, MWmax=148.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=48.0, QsetMVA=0.0)
    pyf.add_gen(grid, '55.0', '23', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '56.0', '24', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=15.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '59.0', '25', lf=20.0, qf=0.064516129, MWmax=254.99999999999997, MWmin=0.0, MVArmax=180.0, MVArmin=-60.0, PsetMW=155.0, QsetMVA=0.0)
    pyf.add_gen(grid, '61.0', '26', lf=20.0, qf=0.0625, MWmax=260.0, MWmin=0.0, MVArmax=300.0, MVArmin=-100.0, PsetMW=160.0, QsetMVA=0.0)
    pyf.add_gen(grid, '62.0', '27', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=20.0, MVArmin=-20.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '65.0', '28', lf=20.0, qf=0.0255754476, MWmax=491.0, MWmin=0.0, MVArmax=200.0, MVArmin=-67.0, PsetMW=391.0, QsetMVA=0.0)
    pyf.add_gen(grid, '66.0', '29', lf=20.0, qf=0.0255102041, MWmax=492.0, MWmin=0.0, MVArmax=200.0, MVArmin=-67.0, PsetMW=392.0, QsetMVA=0.0)
    pyf.add_gen(grid, '69.0', '30', lf=20.0, qf=0.0193648335, MWmax=805.1999999999999, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=516.4, QsetMVA=0.0)
    pyf.add_gen(grid, '70.0', '31', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=32.0, MVArmin=-10.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '72.0', '32', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '73.0', '33', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '74.0', '34', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=9.0, MVArmin=-6.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '76.0', '35', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '77.0', '36', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=70.0, MVArmin=-20.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '80.0', '37', lf=20.0, qf=0.0209643606, MWmax=577.0, MWmin=0.0, MVArmax=280.0, MVArmin=-165.0, PsetMW=476.99999999999994, QsetMVA=0.0)
    pyf.add_gen(grid, '85.0', '38', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '87.0', '39', lf=20.0, qf=2.5, MWmax=104.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-100.0, PsetMW=4.0, QsetMVA=0.0)
    pyf.add_gen(grid, '89.0', '40', lf=20.0, qf=0.0164744646, MWmax=707.0, MWmin=0.0, MVArmax=300.0, MVArmin=-210.0, PsetMW=607.0, QsetMVA=0.0)
    pyf.add_gen(grid, '90.0', '41', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=300.0, MVArmin=-300.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '91.0', '42', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '92.0', '43', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=9.0, MVArmin=-3.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '99.0', '44', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=100.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '100.0', '45', lf=20.0, qf=0.0396825397, MWmax=352.0, MWmin=0.0, MVArmax=155.0, MVArmin=-50.0, PsetMW=252.0, QsetMVA=0.0)
    pyf.add_gen(grid, '103.0', '46', lf=20.0, qf=0.25, MWmax=140.0, MWmin=0.0, MVArmax=40.0, MVArmin=-15.0, PsetMW=40.0, QsetMVA=0.0)
    pyf.add_gen(grid, '104.0', '47', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '105.0', '48', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '107.0', '49', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=200.0, MVArmin=-200.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '110.0', '50', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=23.0, MVArmin=-8.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '111.0', '51', lf=20.0, qf=0.277777778, MWmax=136.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-100.0, PsetMW=36.0, QsetMVA=0.0)
    pyf.add_gen(grid, '112.0', '52', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '113.0', '53', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=200.0, MVArmin=-100.0, PsetMW=0.0, QsetMVA=0.0)
    pyf.add_gen(grid, '116.0', '54', lf=40.0, qf=0.01, MWmax=100.0, MWmin=0.0, MVArmax=1000.0, MVArmin=-1000.0, PsetMW=0.0, QsetMVA=0.0)
    
    
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
