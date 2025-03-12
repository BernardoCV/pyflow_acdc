

import pyflow_acdc as pyf
import pandas as pd


def NS_MTDC():    
    
    S_base=100
    
    # DataFrame Code:
    nodes_AC_data = [
        {'type': 'PV', 'Voltage_0': 1.095, 'theta_0': 0.045588, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'BE1', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 300.795, 'y_coord': 66.958, 'PZ': 'BE'},
        {'type': 'PV', 'Voltage_0': 1.1, 'theta_0': 0.058503437, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'BE2', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 262.294, 'y_coord': 87.0, 'PZ': 'oBE'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'BE4', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 316.51, 'y_coord': 28.859, 'PZ': 'BE'},
        {'type': 'PV', 'Voltage_0': 1.084, 'theta_0': -0.024521876, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'BE5', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 325.125, 'y_coord': 49.886, 'PZ': 'BE'},
        {'type': 'PQ', 'Voltage_0': 1.086, 'theta_0': -0.066497045, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'BE6', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 356.56, 'y_coord': 65.061, 'PZ': 'BE'},
        {'type': 'PQ', 'Voltage_0': 1.084, 'theta_0': -0.101438536, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 13.0, 'Reactive_load': 0.0, 'Node_id': 'BE7', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 359.083, 'y_coord': 35.584, 'PZ': 'BE'},
        {'type': 'PQ', 'Voltage_0': 1.087, 'theta_0': -0.065135688, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 5.5, 'Reactive_load': 0.0, 'Node_id': 'BE8', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 363.027, 'y_coord': 74.641, 'PZ': 'BE'},
        {'type': 'Slack', 'Voltage_0': 1.1, 'theta_0': 0.363028484, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 5.6, 'Reactive_load': 0.0, 'Node_id': 'DE1', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 532.889, 'y_coord': 249.256, 'PZ': 'DE'},
        {'type': 'PV', 'Voltage_0': 1.1, 'theta_0': 0.291347812, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DE2', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 568.121, 'y_coord': 273.69, 'PZ': 'DE'},
        {'type': 'PQ', 'Voltage_0': 1.099, 'theta_0': 0.243089458, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 5.15, 'Reactive_load': 0.0, 'Node_id': 'DE3', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 584.428, 'y_coord': 262.877, 'PZ': 'DE'},
        {'type': 'PQ', 'Voltage_0': 1.094, 'theta_0': 0.156433861, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 5.2, 'Reactive_load': 0.0, 'Node_id': 'DE4', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 640.293, 'y_coord': 281.663, 'PZ': 'DE'},
        {'type': 'PQ', 'Voltage_0': 1.092, 'theta_0': 0.151442219, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DE5', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 647.393, 'y_coord': 302.734, 'PZ': 'DE'},
        {'type': 'PQ', 'Voltage_0': 1.096, 'theta_0': 0.137078159, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 16.2, 'Reactive_load': 0.0, 'Node_id': 'DE6', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 670.575, 'y_coord': 308.377, 'PZ': 'DE'},
        {'type': 'PV', 'Voltage_0': 1.099, 'theta_0': 0.157882484, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DE7', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 638.666, 'y_coord': 332.342, 'PZ': 'DE'},
        {'type': 'PV', 'Voltage_0': 1.1, 'theta_0': 0.167045463, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DE8', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 629.164, 'y_coord': 339.096, 'PZ': 'DE'},
        {'type': 'PQ', 'Voltage_0': 1.1, 'theta_0': 0.134529979, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DE9', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 659.038, 'y_coord': 365.704, 'PZ': 'DE'},
        {'type': 'PQ', 'Voltage_0': 1.092, 'theta_0': 0.095172804, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DK1', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 633.028, 'y_coord': 439.304, 'PZ': 'DK1'},
        {'type': 'PV', 'Voltage_0': 1.096, 'theta_0': 0.108646746, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 1.2, 'Reactive_load': 0.0, 'Node_id': 'DK2', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 610.233, 'y_coord': 479.336, 'PZ': 'DK1'},
        {'type': 'PV', 'Voltage_0': 1.093, 'theta_0': 0.099012528, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DK3', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 622.068, 'y_coord': 481.488, 'PZ': 'DK1'},
        {'type': 'Slack', 'Voltage_0': 1.084, 'theta_0': 0.086323985, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 7.0, 'Reactive_load': 0.0, 'Node_id': 'DK4', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 644.237, 'y_coord': 488.879, 'PZ': 'DK1'},
        {'type': 'PV', 'Voltage_0': 1.095, 'theta_0': 0.106796697, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DK5', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 600.286, 'y_coord': 489.162, 'PZ': 'DK1'},
        {'type': 'PV', 'Voltage_0': 1.091, 'theta_0': 0.101647976, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'DK6', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 598.934, 'y_coord': 529.709, 'PZ': 'DK1'},
        {'type': 'PV', 'Voltage_0': 1.084, 'theta_0': 0.081384703, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 3.4, 'Reactive_load': 0.0, 'Node_id': 'DK7', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 648.299, 'y_coord': 576.164, 'PZ': 'DK1'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB1', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 30.661, 'y_coord': 443.574, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.957, 'theta_0': -0.094736472, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB3', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 89.354, 'y_coord': 326.547, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.962, 'theta_0': -0.114895025, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 13.0, 'Reactive_load': 0.0, 'Node_id': 'GB4', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 83.125, 'y_coord': 308.645, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.966, 'theta_0': -0.109920836, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB5', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 85.697, 'y_coord': 299.001, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.971, 'theta_0': -0.10273008, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB6', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 100.143, 'y_coord': 297.887, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.964, 'theta_0': -0.114266706, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB10', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 93.322, 'y_coord': 275.538, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.943, 'theta_0': -0.151145513, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 8.0, 'Reactive_load': 0.0, 'Node_id': 'GB11', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 69.822, 'y_coord': 230.216, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.953, 'theta_0': -0.128979832, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB12', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 121.429, 'y_coord': 240.591, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.942, 'theta_0': -0.14105751, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB13', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 146.841, 'y_coord': 211.941, 'PZ': 'GB'},
        {'type': 'PV', 'Voltage_0': 0.932, 'theta_0': -0.140010313, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB14', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 210.34, 'y_coord': 189.323, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.909, 'theta_0': -0.123063166, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB15', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 72.75, 'y_coord': 143.869, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.918, 'theta_0': -0.164863801, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB16', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 96.983, 'y_coord': 144.269, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.923, 'theta_0': -0.162228354, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB17', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 111.084, 'y_coord': 143.622, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.922, 'theta_0': -0.166085532, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB18', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 132.725, 'y_coord': 141.826, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.917, 'theta_0': -0.147305789, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB19', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 161.975, 'y_coord': 145.076, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.922, 'theta_0': -0.139015475, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB20', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 188.575, 'y_coord': 148.208, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.9, 'theta_0': -0.033684855, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB21', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 59.5, 'y_coord': 127.369, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.902, 'theta_0': -0.257558238, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 55.0, 'Reactive_load': 0.0, 'Node_id': 'GB22', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 116.366, 'y_coord': 111.017, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.9, 'theta_0': 0.036983527, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB23', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 63.0, 'y_coord': 89.869, 'PZ': 'GB'},
        {'type': 'PV', 'Voltage_0': 0.905, 'theta_0': -0.138806035, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB24', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 160.901, 'y_coord': 97.016, 'PZ': 'GB'},
        {'type': 'PQ', 'Voltage_0': 0.902, 'theta_0': 0.096237455, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB25', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 67.75, 'y_coord': 52.369, 'PZ': 'GB'},
        {'type': 'PV', 'Voltage_0': 0.906, 'theta_0': -0.07845255, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB26', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 208.446, 'y_coord': 79.596, 'PZ': 'GB'},
        {'type': 'PV', 'Voltage_0': 0.966, 'theta_0': -0.114388879, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'GB27', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 133.407, 'y_coord': 268.787, 'PZ': 'GB'},
        {'type': 'PV', 'Voltage_0': 1.098, 'theta_0': -0.010960668, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL1', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 349.334, 'y_coord': 137.085, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.099, 'theta_0': -0.003054326, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL2', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 375.583, 'y_coord': 150.687, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.098, 'theta_0': -0.006527531, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL3', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 380.288, 'y_coord': 139.414, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.096, 'theta_0': -0.019338248, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 7.1, 'Reactive_load': 0.0, 'Node_id': 'NL4', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 390.981, 'y_coord': 133.05, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.095, 'theta_0': -0.021781709, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL5', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 389.106, 'y_coord': 126.335, 'PZ': 'NL'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL6', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 391.355, 'y_coord': 115.472, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.1, 'theta_0': 0.021868976, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL7', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 382.913, 'y_coord': 194.46, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.098, 'theta_0': 0.026965337, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 9.3, 'Reactive_load': 0.0, 'Node_id': 'NL8', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 399.711, 'y_coord': 180.328, 'PZ': 'NL'},
        {'type': 'PV', 'Voltage_0': 1.1, 'theta_0': 0.033300882, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL9', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 394.03, 'y_coord': 226.572, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.1, 'theta_0': 0.166486957, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL10', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 463.987, 'y_coord': 194.457, 'PZ': 'NL'},
        {'type': 'PQ', 'Voltage_0': 1.099, 'theta_0': 0.316916886, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NL11', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 507.858, 'y_coord': 256.127, 'PZ': 'NL'},
        {'type': 'PV', 'Voltage_0': 1.1, 'theta_0': -0.02719223, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NO1', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 493.547, 'y_coord': 752.561, 'PZ': 'NO2'},
        {'type': 'PV', 'Voltage_0': 1.1, 'theta_0': -0.01844813, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NO2', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 562.581, 'y_coord': 738.332, 'PZ': 'NO2'},
        {'type': 'PV', 'Voltage_0': 1.098, 'theta_0': -0.029338985, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NO3', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 493.342, 'y_coord': 788.944, 'PZ': 'NO2'},
        {'type': 'PQ', 'Voltage_0': 1.097, 'theta_0': -0.016877334, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NO4', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 595.777, 'y_coord': 781.292, 'PZ': 'NO2'},
        {'type': 'PQ', 'Voltage_0': 1.081, 'theta_0': -0.099745567, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 5.0, 'Reactive_load': 0.0, 'Node_id': 'NO5', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 439.394, 'y_coord': 805.672, 'PZ': 'NO2'},
        {'type': 'PQ', 'Voltage_0': 1.085, 'theta_0': -0.039217548, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NO6', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 492.811, 'y_coord': 832.766, 'PZ': 'NO2'},
        {'type': 'PQ', 'Voltage_0': 1.08, 'theta_0': -0.003804818, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'NO7', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 523.862, 'y_coord': 861.59, 'PZ': 'NO2'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.5, 'Reactive_load': 0.0, 'Node_id': 'NO8', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 636.934, 'y_coord': 851.011, 'PZ': 'NO2'},
        {'type': 'PQ', 'Voltage_0': 1.07, 'theta_0': -0.000331613, 'kV_base': 380, 'Power_Gained': 0, 'Reactive_Gained': 0, 'Power_load': 2.5, 'Reactive_load': 0.0, 'Node_id': 'NO9', 'Umin': 0.9, 'Umax': 1.1, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 493.64, 'y_coord': 878.619, 'PZ': 'NO2'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'OFW_DK', 'Umin': 1.0, 'Umax': 1.0, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 517.924, 'y_coord': 582.0, 'PZ': 'oDK1'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'OFW_NL', 'Umin': 1.0, 'Umax': 1.0, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 345.995, 'y_coord': 300.0, 'PZ': 'oNL'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'OFW_DEC', 'Umin': 1.0, 'Umax': 1.0, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 463.058, 'y_coord': 439.0, 'PZ': 'oDE'},
        {'type': 'Slack', 'Voltage_0': 1.0, 'theta_0': 0.0, 'kV_base': 380, 'Power_Gained': 0.0, 'Reactive_Gained': 0.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'Node_id': 'OFW_NO', 'Umin': 1.0, 'Umax': 1.0, 'Gs': 0.0, 'Bs': 0.0, 'x_coord': 400.0, 'y_coord': 730.0, 'PZ': 'oNO2'}
    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)

    lines_AC_data = [
        {'fromNode': 'GB1', 'toNode': 'GB3', 'r': 0.0005817174515235457, 'x': 0.007796197215082969, 'g': 0.0, 'b': 4.115476323106137, 'MVA_rating': 9359.309743779184, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL2'},
        {'fromNode': 'GB3', 'toNode': 'GB4', 'r': 0.0001592797783933518, 'x': 0.002134673046987004, 'g': 0.0, 'b': 0.28171415306976527, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL3'},
        {'fromNode': 'GB3', 'toNode': 'GB6', 'r': 0.00034626038781163435, 'x': 0.004640593580406529, 'g': 0.0, 'b': 0.6124220718907941, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL4'},
        {'fromNode': 'GB4', 'toNode': 'GB5', 'r': 7.617728531855956e-05, 'x': 0.0010209305876894366, 'g': 0.0, 'b': 0.13473285581597472, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL5'},
        {'fromNode': 'GB5', 'toNode': 'GB6', 'r': 0.000110803324099723, 'x': 0.0014849899457300894, 'g': 0.0, 'b': 0.19597506300505413, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL6'},
        {'fromNode': 'GB6', 'toNode': 'GB10', 'r': 5.54016620498615e-05, 'x': 0.0007424949728650447, 'g': 0.0, 'b': 0.39195012601010826, 'MVA_rating': 9359.309743779184, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL10'},
        {'fromNode': 'GB10', 'toNode': 'GB11', 'r': 0.00045706371191135735, 'x': 0.00612558352613662, 'g': 0.0, 'b': 0.8083971348958483, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL11'},
        {'fromNode': 'GB11', 'toNode': 'GB15', 'r': 0.0006925207756232687, 'x': 0.009281187160813059, 'g': 0.0, 'b': 1.2248441437815882, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL12'},
        {'fromNode': 'GB10', 'toNode': 'GB16', 'r': 0.001066481994459834, 'x': 0.014293028227652112, 'g': 0.0, 'b': 1.886259981423646, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL13'},
        {'fromNode': 'GB10', 'toNode': 'GB17', 'r': 0.001066481994459834, 'x': 0.014293028227652112, 'g': 0.0, 'b': 1.886259981423646, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL14'},
        {'fromNode': 'GB10', 'toNode': 'GB12', 'r': 0.0003739612188365651, 'x': 0.005011841066839052, 'g': 0.0, 'b': 0.6614158376420577, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL15'},
        {'fromNode': 'GB12', 'toNode': 'GB13', 'r': 0.00030470914127423825, 'x': 0.004083722350757746, 'g': 0.0, 'b': 0.5389314232638989, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL16'},
        {'fromNode': 'GB13', 'toNode': 'GB18', 'r': 0.0006024930747922437, 'x': 0.008074632829907362, 'g': 0.0, 'b': 1.0656144050899818, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL17'},
        {'fromNode': 'GB13', 'toNode': 'GB14', 'r': 0.0005609418282548476, 'x': 0.007517761600258578, 'g': 0.0, 'b': 0.9921237564630865, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL18'},
        {'fromNode': 'GB14', 'toNode': 'GB20', 'r': 0.00036703601108033245, 'x': 0.004919029195230921, 'g': 0.0, 'b': 0.6491673962042418, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL19'},
        {'fromNode': 'GB15', 'toNode': 'GB16', 'r': 0.00019390581717451525, 'x': 0.0025987324050276567, 'g': 0.0, 'b': 0.34295636025884474, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL20'},
        {'fromNode': 'GB16', 'toNode': 'GB17', 'r': 0.000110803324099723, 'x': 0.0014849899457300894, 'g': 0.0, 'b': 0.19597506300505413, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL21'},
        {'fromNode': 'GB17', 'toNode': 'GB18', 'r': 0.00017313019390581717, 'x': 0.0023202967902032647, 'g': 0.0, 'b': 0.30621103594539706, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL22'},
        {'fromNode': 'GB18', 'toNode': 'GB20', 'r': 0.00046398891966759005, 'x': 0.006218395397744749, 'g': 0.0, 'b': 0.8206455763336642, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL23'},
        {'fromNode': 'GB18', 'toNode': 'GB19', 'r': 0.00047091412742382275, 'x': 0.0063112072693528795, 'g': 0.0, 'b': 0.20822350444287002, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL24'},
        {'fromNode': 'GB19', 'toNode': 'GB20', 'r': 0.0004293628808864266, 'x': 0.0057543360397040964, 'g': 0.0, 'b': 0.18985084228614618, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL25'},
        {'fromNode': 'GB15', 'toNode': 'GB21', 'r': 0.00017313019390581717, 'x': 0.0023202967902032647, 'g': 0.0, 'b': 0.30621103594539706, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL26'},
        {'fromNode': 'GB21', 'toNode': 'GB23', 'r': 0.0003116343490304709, 'x': 0.0041765342223658766, 'g': 0.0, 'b': 0.5511798647017148, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL27'},
        {'fromNode': 'GB15', 'toNode': 'GB22', 'r': 0.000533240997229917, 'x': 0.007146514113826056, 'g': 0.0, 'b': 0.943129990711823, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL28'},
        {'fromNode': 'GB16', 'toNode': 'GB22', 'r': 0.0003185595567867036, 'x': 0.004269346093974008, 'g': 0.0, 'b': 0.5634283061395305, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL29'},
        {'fromNode': 'GB18', 'toNode': 'GB22', 'r': 0.000554016620498615, 'x': 0.007424949728650447, 'g': 0.0, 'b': 0.24496882875631765, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL30'},
        {'fromNode': 'GB19', 'toNode': 'GB24', 'r': 0.0003947368421052632, 'x': 0.005290276681663444, 'g': 0.0, 'b': 0.6981611619555053, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL31'},
        {'fromNode': 'GB22', 'toNode': 'GB24', 'r': 0.0003739612188365651, 'x': 0.005011841066839052, 'g': 0.0, 'b': 0.6614158376420577, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL32'},
        {'fromNode': 'GB23', 'toNode': 'GB24', 'r': 0.0008310249307479224, 'x': 0.011137424592975672, 'g': 0.0, 'b': 1.4698129725379059, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL33'},
        {'fromNode': 'GB23', 'toNode': 'GB25', 'r': 0.0004155124653739612, 'x': 0.005568712296487836, 'g': 0.0, 'b': 0.7349064862689529, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL34'},
        {'fromNode': 'GB24', 'toNode': 'GB26', 'r': 0.00046398891966759005, 'x': 0.006218395397744749, 'g': 0.0, 'b': 0.8206455763336642, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL35'},
        {'fromNode': 'GB25', 'toNode': 'GB26', 'r': 0.0013157894736842107, 'x': 0.017634255605544814, 'g': 0.0, 'b': 2.3272038731850175, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL36'},
        {'fromNode': 'GB10', 'toNode': 'GB27', 'r': 0.0003808864265927978, 'x': 0.005104652938447182, 'g': 0.0, 'b': 0.6736642790798736, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'GBL37'},
        {'fromNode': 'BE1', 'toNode': 'BE2', 'r': 0.000171398891966759, 'x': 0.0006200511816295644, 'g': 0.0, 'b': 20.822350444287004, 'MVA_rating': 3633.1497739564766, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'BEL1'},
        {'fromNode': 'BE1', 'toNode': 'BE4', 'r': 0.00034626038781163435, 'x': 0.004640593580406529, 'g': 0.0, 'b': 0.6124220718907941, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'BEL2'},
        {'fromNode': 'BE4', 'toNode': 'BE5', 'r': 0.00018005540166204988, 'x': 0.0024131086618113954, 'g': 0.0, 'b': 0.31845947738321295, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'BEL4'},
        {'fromNode': 'BE5', 'toNode': 'BE6', 'r': 0.0003116343490304709, 'x': 0.0041765342223658766, 'g': 0.0, 'b': 0.5511798647017148, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'BEL5'},
        {'fromNode': 'BE6', 'toNode': 'BE7', 'r': 0.00023545706371191138, 'x': 0.0031556036346764398, 'g': 0.0, 'b': 0.41644700888574004, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'BEL6'},
        {'fromNode': 'BE6', 'toNode': 'BE8', 'r': 9.695290858725763e-05, 'x': 0.0012993662025138283, 'g': 0.0, 'b': 0.17147818012942237, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'BEL7'},
        {'fromNode': 'NL1', 'toNode': 'NL2', 'r': 0.0002493074792243767, 'x': 0.003341227377892701, 'g': 0.0, 'b': 0.4409438917613718, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL1'},
        {'fromNode': 'NL1', 'toNode': 'NL5', 'r': 0.00034626038781163435, 'x': 0.004640593580406529, 'g': 0.0, 'b': 0.6124220718907941, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL2'},
        {'fromNode': 'NL2', 'toNode': 'NL7', 'r': 0.00036703601108033245, 'x': 0.004919029195230921, 'g': 0.0, 'b': 0.6491673962042418, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL3'},
        {'fromNode': 'NL2', 'toNode': 'NL3', 'r': 9.695290858725763e-05, 'x': 0.0012993662025138283, 'g': 0.0, 'b': 0.17147818012942237, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL4'},
        {'fromNode': 'NL3', 'toNode': 'NL4', 'r': 0.0001038781163434903, 'x': 0.001392178074121959, 'g': 0.0, 'b': 0.18372662156723824, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL5'},
        {'fromNode': 'NL4', 'toNode': 'NL5', 'r': 5.54016620498615e-05, 'x': 0.0007424949728650447, 'g': 0.0, 'b': 0.09798753150252706, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL6'},
        {'fromNode': 'NL5', 'toNode': 'NL6', 'r': 9.002770083102494e-05, 'x': 0.0012065543309056977, 'g': 0.0, 'b': 0.15922973869160648, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL7'},
        {'fromNode': 'NL3', 'toNode': 'NL8', 'r': 0.0003808864265927978, 'x': 0.005104652938447182, 'g': 0.0, 'b': 0.6736642790798736, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL8'},
        {'fromNode': 'NL7', 'toNode': 'NL8', 'r': 0.00019390581717451525, 'x': 0.0025987324050276567, 'g': 0.0, 'b': 0.34295636025884474, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL9'},
        {'fromNode': 'NL7', 'toNode': 'NL9', 'r': 0.0002700831024930748, 'x': 0.003619662992717093, 'g': 0.0, 'b': 0.47768921607481946, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL10'},
        {'fromNode': 'NL8', 'toNode': 'NL10', 'r': 0.0006371191135734072, 'x': 0.008538692187948015, 'g': 0.0, 'b': 1.126856612279061, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL11'},
        {'fromNode': 'NL10', 'toNode': 'NL11', 'r': 0.0006786703601108033, 'x': 0.009095563417596798, 'g': 0.0, 'b': 1.2003472609059564, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NLL12'},
        {'fromNode': 'DE1', 'toNode': 'DE2', 'r': 0.00034626038781163435, 'x': 0.004640593580406529, 'g': 0.0, 'b': 0.6124220718907941, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL1'},
        {'fromNode': 'DE2', 'toNode': 'DE3', 'r': 0.00023545706371191138, 'x': 0.0031556036346764398, 'g': 0.0, 'b': 0.41644700888574004, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL2'},
        {'fromNode': 'DE3', 'toNode': 'DE4', 'r': 0.0005886426592797783, 'x': 0.007889009086691101, 'g': 0.0, 'b': 1.04111752221435, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL3'},
        {'fromNode': 'DE4', 'toNode': 'DE6', 'r': 0.0004293628808864266, 'x': 0.0057543360397040964, 'g': 0.0, 'b': 0.7594033691445847, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL4'},
        {'fromNode': 'DE4', 'toNode': 'DE5', 'r': 0.00011542012927054478, 'x': 0.0015468645268021765, 'g': 0.0, 'b': 0.4593165539180956, 'MVA_rating': 7019.482307834389, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL5'},
        {'fromNode': 'DE5', 'toNode': 'DE6', 'r': 0.00019390581717451525, 'x': 0.0025987324050276567, 'g': 0.0, 'b': 0.34295636025884474, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL6'},
        {'fromNode': 'DE6', 'toNode': 'DE8', 'r': 0.0004362880886426593, 'x': 0.0058471479113122275, 'g': 0.0, 'b': 0.7716518105824006, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL7'},
        {'fromNode': 'DE7', 'toNode': 'DE8', 'r': 9.002770083102494e-05, 'x': 0.0012065543309056977, 'g': 0.0, 'b': 0.15922973869160648, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL8'},
        {'fromNode': 'DE7', 'toNode': 'DE9', 'r': 0.0003116343490304709, 'x': 0.0041765342223658766, 'g': 0.0, 'b': 0.5511798647017148, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL9'},
        {'fromNode': 'DE6', 'toNode': 'DE9', 'r': 0.00046398891966759005, 'x': 0.006218395397744749, 'g': 0.0, 'b': 0.8206455763336642, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL10'},
        {'fromNode': 'DE5', 'toNode': 'DE7', 'r': 0.00024238227146814408, 'x': 0.003248415506284571, 'g': 0.0, 'b': 0.4286954503235559, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DEL11'},
        {'fromNode': 'DK1', 'toNode': 'DK3', 'r': 0.00034626038781163435, 'x': 0.004640593580406529, 'g': 0.0, 'b': 0.6124220718907941, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL1'},
        {'fromNode': 'DK1', 'toNode': 'DK4', 'r': 0.0008310249307479224, 'x': 0.011137424592975672, 'g': 0.0, 'b': 0.36745324313447647, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL2'},
        {'fromNode': 'DK2', 'toNode': 'DK5', 'r': 0.00011772853185595569, 'x': 0.0015778018173382199, 'g': 0.0, 'b': 0.20822350444287002, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL3'},
        {'fromNode': 'DK2', 'toNode': 'DK3', 'r': 9.695290858725763e-05, 'x': 0.0012993662025138283, 'g': 0.0, 'b': 0.17147818012942237, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL4'},
        {'fromNode': 'DK3', 'toNode': 'DK4', 'r': 0.00019390581717451525, 'x': 0.0025987324050276567, 'g': 0.0, 'b': 0.34295636025884474, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL5'},
        {'fromNode': 'DK5', 'toNode': 'DK6', 'r': 0.00033240997229916895, 'x': 0.004454969837190268, 'g': 0.0, 'b': 0.5879251890151623, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL6'},
        {'fromNode': 'DK3', 'toNode': 'DK7', 'r': 0.0007825484764542938, 'x': 0.010487741491718756, 'g': 0.0, 'b': 1.3840738824731946, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL7'},
        {'fromNode': 'DK6', 'toNode': 'DK7', 'r': 0.0013434903047091413, 'x': 0.018005503091977335, 'g': 0.0, 'b': 0.5940494097340703, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'DKL8'},
        {'fromNode': 'NO1', 'toNode': 'NO2', 'r': 0.00110803324099723, 'x': 0.014849899457300895, 'g': 0.0, 'b': 0.4899376575126353, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL1'},
        {'fromNode': 'NO1', 'toNode': 'NO3', 'r': 0.00029085872576177285, 'x': 0.0038980986075414846, 'g': 0.0, 'b': 0.5144345403882671, 'MVA_rating': 4679.654871889592, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL2'},
        {'fromNode': 'NO2', 'toNode': 'NO7', 'r': 0.002063711911357341, 'x': 0.027657937739222916, 'g': 0.0, 'b': 0.9125088871172833, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL3'},
        {'fromNode': 'NO2', 'toNode': 'NO4', 'r': 0.0008725761772853186, 'x': 0.011694295822624455, 'g': 0.0, 'b': 0.3858259052912003, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL4'},
        {'fromNode': 'NO3', 'toNode': 'NO6', 'r': 0.0007063711911357341, 'x': 0.009466810904029321, 'g': 0.0, 'b': 0.312335256664305, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL5'},
        {'fromNode': 'NO3', 'toNode': 'NO4', 'r': 0.0017313019390581717, 'x': 0.023202967902032648, 'g': 0.0, 'b': 0.7655275898634927, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL6'},
        {'fromNode': 'NO4', 'toNode': 'NO8', 'r': 0.0016481994459833795, 'x': 0.022089225442735082, 'g': 0.0, 'b': 0.728782265550045, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL7'},
        {'fromNode': 'NO5', 'toNode': 'NO6', 'r': 0.0009418282548476455, 'x': 0.012622414538705759, 'g': 0.0, 'b': 0.41644700888574004, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL8'},
        {'fromNode': 'NO6', 'toNode': 'NO9', 'r': 0.0007617728531855956, 'x': 0.010209305876894365, 'g': 0.0, 'b': 0.3368321395399368, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL9'},
        {'fromNode': 'NO9', 'toNode': 'NO7', 'r': 0.000554016620498615, 'x': 0.007424949728650447, 'g': 0.0, 'b': 0.24496882875631765, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL10'},
        {'fromNode': 'NO7', 'toNode': 'NO8', 'r': 0.0018698060941828255, 'x': 0.025059205334195263, 'g': 0.0, 'b': 0.826769797052572, 'MVA_rating': 2339.827435944796, 'kV_base': 380, 'm': 1, 'shift': 0, 'Line_id': 'NOL11'}
    ]
    lines_AC = pd.DataFrame(lines_AC_data)

    nodes_DC_data = [
        {'type': 'Slack', 'Voltage_0': 1.0, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC1', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 262.294, 'y_coord': 93.448, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC2', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 300.795, 'y_coord': 66.958, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC3', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 325.125, 'y_coord': 49.886, 'PZ': None},
        {'type': 'PAC', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC4', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 345.995, 'y_coord': 295.697, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC5', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 394.03, 'y_coord': 226.572, 'PZ': None},
        {'type': 'PAC', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC6', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 463.058, 'y_coord': 444.808, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC7', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 568.121, 'y_coord': 273.69, 'PZ': None},
        {'type': 'PAC', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC8', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 517.924, 'y_coord': 577.766, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC9', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 598.934, 'y_coord': 529.709, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC10', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 133.407, 'y_coord': 268.787, 'PZ': None},
        {'type': 'P', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'MTDC11', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 210.34, 'y_coord': 189.323, 'PZ': None},
        {'type': 'Slack', 'Voltage_0': 1.0, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'NODC1', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 493.342, 'y_coord': 788.944, 'PZ': None},
        {'type': 'PAC', 'Voltage_0': 1.01, 'Power_Gained': 0.0, 'Power_load': 0.0, 'kV_base': 525, 'Node_id': 'NODC2', 'Umin': 0.95, 'Umax': 1.05, 'x_coord': 400.0, 'y_coord': 730.0, 'PZ': None}
    ]
    nodes_DC = pd.DataFrame(nodes_DC_data)

    lines_DC_data = [
        {'fromNode': 'MTDC4', 'toNode': 'MTDC10', 'r': 0.0008961451247165532, 'MW_rating': 1800.75, 'kV_base': 525, 'Length_km': 260, 'Mono_Bi_polar': 'b', 'Line_id': 'LionLink'},
        {'fromNode': 'MTDC1', 'toNode': 'MTDC11', 'r': 0.0004825396825396826, 'MW_rating': 1399.99999965, 'kV_base': 525, 'Length_km': 140, 'Mono_Bi_polar': 'b', 'Line_id': 'Nautilus'},
        {'fromNode': 'MTDC1', 'toNode': 'MTDC8', 'r': 0.002068027210884354, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 600, 'Mono_Bi_polar': 'b', 'Line_id': 'TritonLink'},
        {'fromNode': 'MTDC1', 'toNode': 'MTDC4', 'r': 0.0008616780045351474, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 250, 'Mono_Bi_polar': 'b', 'Line_id': 'BE-NL Link'},
        {'fromNode': 'MTDC4', 'toNode': 'MTDC6', 'r': 0.0007410430839002268, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 215, 'Mono_Bi_polar': 'b', 'Line_id': 'DE-NL Link'},
        {'fromNode': 'MTDC6', 'toNode': 'MTDC8', 'r': 0.0005687074829931972, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 165, 'Mono_Bi_polar': 'b', 'Line_id': 'DK-DE Link'},
        {'fromNode': 'MTDC1', 'toNode': 'MTDC2', 'r': 0.00015510204081632654, 'MW_rating': 1399.99999965, 'kV_base': 525, 'Length_km': 45, 'Mono_Bi_polar': 'b', 'Line_id': 'BE Nautilus'},
        {'fromNode': 'MTDC1', 'toNode': 'MTDC3', 'r': 0.0002412698412698413, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 70, 'Mono_Bi_polar': 'b', 'Line_id': 'BE TritonLink'},
        {'fromNode': 'MTDC6', 'toNode': 'MTDC7', 'r': 0.00037913832199546485, 'MW_rating': 4000.0000005, 'kV_base': 525, 'Length_km': 220, 'Mono_Bi_polar': 'b', 'Line_id': 'DE HUB'},
        {'fromNode': 'MTDC8', 'toNode': 'MTDC9', 'r': 0.00022058956916099772, 'MW_rating': 4000.0000005, 'kV_base': 525, 'Length_km': 128, 'Mono_Bi_polar': 'b', 'Line_id': 'DK HUB'},
        {'fromNode': 'MTDC4', 'toNode': 'MTDC5', 'r': 0.0002757369614512472, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 80, 'Mono_Bi_polar': 'b', 'Line_id': 'NL HUB'},
        {'fromNode': 'NODC1', 'toNode': 'NODC2', 'r': 0.00034467120181405896, 'MW_rating': 2000.00000025, 'kV_base': 525, 'Length_km': 100, 'Mono_Bi_polar': 'b', 'Line_id': 'NO HUB'}
    ]
    lines_DC = pd.DataFrame(lines_DC_data)

    Converters_ACDC_data = [
        {'AC_type': 'PV', 'DC_type': 'Slack', 'AC_node': 'BE2', 'DC_node': 'MTDC1', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': -0.85, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.00125, 'PR_X': 0.015902690085320234, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 800.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_MTDC1', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'BE1', 'DC_node': 'MTDC2', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': -0.99, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.00125, 'PR_X': 0.015902690085320234, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 800.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_MTDC2', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'BE5', 'DC_node': 'MTDC3', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': -0.21, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1100.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_MTDC3', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'Slack', 'DC_type': 'PAC', 'AC_node': 'OFW_NL', 'DC_node': 'MTDC4', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 19.87, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_MTDC4', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'NL9', 'DC_node': 'MTDC5', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 0.0, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 2, 'pol': 2, 'Conv_id': 'Conv_MTDC5', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'Slack', 'DC_type': 'PAC', 'AC_node': 'OFW_DEC', 'DC_node': 'MTDC6', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 19.87, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 2, 'pol': 2, 'Conv_id': 'Conv_MTDC6', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'DE2', 'DC_node': 'MTDC7', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 0.0, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 2, 'pol': 2, 'Conv_id': 'Conv_MTDC7', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'Slack', 'DC_type': 'PAC', 'AC_node': 'OFW_DK', 'DC_node': 'MTDC8', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 29.63, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.00125, 'PR_X': 0.015902690085320234, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 800.0, 'Nconverter': 2, 'pol': 2, 'Conv_id': 'Conv_MTDC8', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'DK6', 'DC_node': 'MTDC9', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 0.0, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 2, 'pol': 2, 'Conv_id': 'Conv_MTDC9', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'GB27', 'DC_node': 'MTDC10', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 0.0, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.00125, 'PR_X': 0.015902690085320234, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 900.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_MTDC10', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'P', 'AC_node': 'GB14', 'DC_node': 'MTDC11', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 0.0, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1100.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_MTDC11', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'PV', 'DC_type': 'Slack', 'AC_node': 'NO3', 'DC_node': 'NODC1', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 0.0, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_NO1', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2},
        {'AC_type': 'Slack', 'DC_type': 'PAC', 'AC_node': 'OFW_NO', 'DC_node': 'NODC2', 'P_AC': 0.0, 'Q_AC': 0, 'P_DC': 19.87, 'T_R': 0.0, 'T_X': 0.0, 'PR_R': 0.0008326446280991736, 'PR_X': 0.010709974955419749, 'Filter': 0.0, 'Droop': 0, 'AC_kV_base': 220, 'MVA_rating': 1000.0, 'Nconverter': 1, 'pol': 2, 'Conv_id': 'Conv_NO2', 'lossa': 1.103, 'lossb': 0.887, 'losscrect': 2.885, 'losscinv': 4.371, 'Ucmin': 0.85, 'Ucmax': 1.2}
    ]
    Converters_ACDC = pd.DataFrame(Converters_ACDC_data)

    
    # Create the grid
    [grid, res] = pyf.Create_grid_from_data(S_base, nodes_AC, lines_AC, nodes_DC, lines_DC, Converters_ACDC, data_in = 'pu')
        
    
    # Add Price Zones:
    pyf.add_price_zone(grid,'BE',97.27,import_pu_L=1,export_pu_G=1,a=0,b=97.27,c=0,import_expand_pu=0)
    pyf.add_price_zone(grid,'DE',95.67,import_pu_L=1,export_pu_G=1,a=0,b=95.67,c=0,import_expand_pu=0)
    pyf.add_price_zone(grid,'DK1',81.26,import_pu_L=1,export_pu_G=1,a=0,b=81.26,c=0,import_expand_pu=0)
    pyf.add_price_zone(grid,'GB',108.23,import_pu_L=1,export_pu_G=1,a=0,b=108.23,c=0,import_expand_pu=0)
    pyf.add_price_zone(grid,'NL',95.82,import_pu_L=1,export_pu_G=1,a=0,b=95.82,c=0,import_expand_pu=0)
    pyf.add_price_zone(grid,'NO2',79.44,import_pu_L=1,export_pu_G=1,a=0,b=79.44,c=0,import_expand_pu=0)
    pyf.add_offshore_price_zone(grid,'BE','oBE')
    pyf.add_offshore_price_zone(grid,'DE','oDE')
    pyf.add_offshore_price_zone(grid,'DK1','oDK1')
    pyf.add_offshore_price_zone(grid,'NL','oNL')
    pyf.add_offshore_price_zone(grid,'NO2','oNO2')
    pyf.add_MTDC_price_zone(grid, 'MTDC',linked_price_zones=[],pricing_strategy='avg')

    

    
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
    pyf.add_gen(grid, 'BE4', 'BE4', price_zone_link=True, lf=97.27, qf=0, MWmax=4670.0, MWmin=0, MVArmax=524.0, MVArmin=-524.0, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'BE7', 'BE7', price_zone_link=True, lf=97.27, qf=0, MWmax=4670.0, MWmin=0, MVArmax=524.0, MVArmin=-524.0, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'DE1', 'DE1', price_zone_link=True, lf=95.67, qf=0, MWmax=4670.0, MWmin=0, MVArmax=1556.6666666666667, MVArmin=-1556.6666666666667, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'DE2', 'DE2', price_zone_link=True, lf=95.67, qf=0, MWmax=4670.0, MWmin=0, MVArmax=1556.6666666666667, MVArmin=-1556.6666666666667, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'DE3', 'DE3', price_zone_link=True, lf=95.67, qf=0, MWmax=4670.0, MWmin=0, MVArmax=1556.6666666666667, MVArmin=-1556.6666666666667, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'DE5', 'DE5', price_zone_link=True, lf=95.67, qf=0, MWmax=4670.0, MWmin=0, MVArmax=1556.6666666666667, MVArmin=-1556.6666666666667, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'DK4', 'DK4', price_zone_link=True, lf=81.26, qf=0, MWmax=1744.0000000000002, MWmin=0, MVArmax=581.3333333333334, MVArmin=-581.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'DK7', 'DK7', price_zone_link=True, lf=81.26, qf=0, MWmax=4670.0, MWmin=0, MVArmax=776.6666666666666, MVArmin=-776.6666666666666, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'GB1', 'GB1', price_zone_link=True, lf=108.23, qf=0, MWmax=4670.0, MWmin=0, MVArmax=596.3333333333334, MVArmin=-596.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'GB21', 'GB21', price_zone_link=True, lf=108.23, qf=0, MWmax=4670.0, MWmin=0, MVArmax=596.3333333333334, MVArmin=-596.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'GB23', 'GB23', price_zone_link=True, lf=108.23, qf=0, MWmax=4670.0, MWmin=0, MVArmax=596.3333333333334, MVArmin=-596.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'GB25', 'GB25', price_zone_link=True, lf=108.23, qf=0, MWmax=4670.0, MWmin=0, MVArmax=596.3333333333334, MVArmin=-596.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'NL6', 'NL6', price_zone_link=True, lf=95.82, qf=0, MWmax=4670.0, MWmin=0, MVArmax=1004.3333333333333, MVArmin=-1004.3333333333333, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'NL10', 'NL10', price_zone_link=True, lf=95.82, qf=0, MWmax=4670.0, MWmin=0, MVArmax=1004.3333333333333, MVArmin=-1004.3333333333333, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'NO8', 'NO8', price_zone_link=True, lf=79.44, qf=0, MWmax=2173.0, MWmin=0, MVArmax=724.3333333333334, MVArmin=-724.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'NO9', 'NO9', price_zone_link=True, lf=79.44, qf=0, MWmax=2173.0, MWmin=0, MVArmax=724.3333333333334, MVArmin=-724.3333333333334, PsetMW=0, QsetMVA=0)
    pyf.add_gen(grid, 'GB20', 'GB20', price_zone_link=True, lf=108.23, qf=0, MWmax=545.0, MWmin=490.5, MVArmax=545.0, MVArmin=-545.0, PsetMW=0.0, QsetMVA=0.0)
    
    
    # Add Renewable Source Zones
    pyf.add_RenSource_zone(grid,'BE')
    pyf.add_RenSource_zone(grid,'DE')
    pyf.add_RenSource_zone(grid,'DK')
    pyf.add_RenSource_zone(grid,'NL')
    pyf.add_RenSource_zone(grid,'NO')

    
    # Add Renewable Sources
    pyf.add_RenSource(grid, 'BE2', 3500.0, ren_source_name='BE2', available=1, zone='BE', price_zone='BE', Offshore=True, MTDC=None)
    pyf.add_RenSource(grid, 'OFW_DEC', 4500.0, ren_source_name='OFW_DEC', available=1, zone='DE', price_zone='DE', Offshore=True, MTDC=None)
    pyf.add_RenSource(grid, 'OFW_DK', 3500.0, ren_source_name='OFW_DK', available=1, zone='DK', price_zone='DK1', Offshore=True, MTDC=None)
    pyf.add_RenSource(grid, 'OFW_NL', 2000.0, ren_source_name='OFW_NL', available=1, zone='NL', price_zone='NL', Offshore=True, MTDC=None)
    pyf.add_RenSource(grid, 'OFW_NO', 2000.0, ren_source_name='OFW_NO', available=1, zone='NO', price_zone='NO2', Offshore=True, MTDC=None)

    
    # Return the grid
    return grid,res
