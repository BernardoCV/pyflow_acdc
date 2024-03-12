# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:38:12 2024

@author: BernardoCastro
"""

import numpy as np
import pandas as pd


from . import PyFlow_ACDC_PF 

try:
    import pyomo
    from . import PyFlow_ACDC_OPF 
    pyomo_imp= True
    
except ImportError:    
    pyomo_imp= False
weights_def = {
    'Ext_Gen'         : {'w': 1},
    'Market_sale'     : {'w': 0},
    'AC_losses'       : {'w': 1},
    'DC_losses'       : {'w': 1},
    'Converter_Losses': {'w': 1}
}
def Time_series_PF(grid):
    if grid.nodes_AC == None:
        print("only DC")
    elif grid.nodes_DC == None:
        print("only AC")
    else:
        print("Sequential")
        grid.TS_ACDC_PF(grid)

def TS_ACDC_PF(grid, start=1, end=99999, OPF=False,OPF_w=weights_def ,VarPrice=False):
    idx = start-1
    TS_len = len(grid.Time_series[0].TS)
    if TS_len < end:
        max_time = TS_len
    else:
        max_time = end
    grid.Time_series_res = []
    grid.Time_series_line_res = []
    grid.Time_series_grid_res = []
    grid.Time_series_input = []

    if OPF == True:
        grid.Time_series_Opt_res_P_conv_DC = []
        grid.Time_series_Opt_res_P_extGrid = []

    grid.Time_series_price = []
    grid.Time_series_money = []
    # saving droop configuration to reset each time, if not it takes power set from previous point.
    grid.Pconv_save = np.zeros(grid.nconv)
    for conv in grid.Converters_ACDC:
        grid.Pconv_save[conv.ConvNumber] = conv.P_DC

    cash = 0
    while idx < max_time:
        price_data = {'time': idx+1}
        for ts in grid.Time_series:
            typ = ts.type
            if typ == 'Load':
                ts.node.PLi = ts.TS[idx]
            elif typ == 'Slack':
                if VarPrice == True:
                    ts.node.price = ts.TS[idx]
                    price_data[f'{ts.name}_price'] = ts.TS[idx]
                    grid.VarPrice = True
                    OPF_w['Market_sale']['w']=1

            else:
                ts.node.PGi = ts.TS[idx]

            # ts.node.P_AC = ts.node.PGi-ts.node.PLi
            # ts.node.Q_AC = ts.node.QGi-ts.node.QLi

        if OPF == True and pyomo_imp==True:
            [model,results]=OPF_ACDC(grid,ObjRule=OPF_w)
            [opt_res_P_conv_DC,opt_res_P_conv_AC,opt_res_Q_conv_AC,opt_res_P_extGrid,opt_res_Q_extGrid] = OPF_conv_results(grid,model)
            [opt_res_P_conv_DC_dict, opt_res_P_conv_AC_dict ,opt_res_Q_conv_AC_dict, opt_res_P_extGrid_dict,opt_res_Q_extGrid_dict] = OPF_conv_results(grid,model,Dict=True)

            opt_res_P_conv_DC_dict['time'] = idx+1
            opt_res_P_extGrid_dict['time'] = idx+1
        for conv in grid.Converters_ACDC:

            
            if conv.type == 'Droop' or conv.type== 'P':
                if OPF == True and  pyomo_imp==True:
                    conv.P_DC = opt_res_P_conv_DC[conv.ConvNumber]
                    s = 1
                else:
                    conv.P_DC = grid.Pconv_save[conv.ConvNumber] #This resets the converters droop target
                    s = 1
            elif conv.type =='PAC' and conv.AC_type != 'Slack' :
              if OPF == True and  pyomo_imp==True:
                      conv.P_AC = opt_res_P_conv_AC[conv.ConvNumber]
                      
                      
                      s = 1
    
        s = 1
        cash_inst = 0
        if OPF==False:
            ACDC_sequential(grid,QLimit=False)

        money_data = {'time': idx+1}
        row_data = {'time': idx+1}
        in_data = {'time': idx+1}
        line_data = {'time': idx+1}
        grid_data = {'time': idx+1}

        for ts in grid.Time_series:
            if ts.type == 'Slack':
                node = ts.node
                PGi = (node.P_INJ-node.P_s+node.PLi).item()
                QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi
                col_namePg_ = f'Pg_{ts.name}'
                col_nameQg_ = f'Qg_{ts.name}'

                # Append new data to the existing DataFrame
                row_data[col_namePg_] = PGi
                row_data[col_nameQg_] = QGi
                money_data[f'{ts.name}_money'] = node.price*PGi
                cash_inst += node.price*PGi
            else:
                in_data[ts.name] = ts.TS[idx]
        cash += cash_inst
        money_data['Aggregative'] = cash
        money_data['Instant'] = cash_inst
        for conv in grid.Converters_ACDC:
            S_AC = np.sqrt(conv.P_AC**2+conv.Q_AC**2)
            P_DC = conv.P_DC
            col_name = f'{conv.name}_max'
            col_name2 = f'{conv.name}_P_DC'
            row_data[col_name] = np.maximum(S_AC, np.abs(P_DC))
            row_data[col_name2] = P_DC

        grid.Line_AC_calc()
        grid.Line_DC_calc()
        lossP_AC = np.zeros(grid.Num_Grids_AC)
        lossP_DC = np.zeros(grid.Num_Grids_AC)

        for line in grid.lines_AC:
            node = line.fromNode
            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber
            name = line.name
            G = grid.Graph_node_to_Grid_index_AC[node]
            Ploss = np.real(line.loss)

            Sfrom = abs(grid.Sij[i, j])
            Sto = abs(grid.Sij[j, i])

            load = max(Sfrom, Sto)

            lossP_AC[G] += Ploss

            col_name_load = f'Load_AC_{line.name}'
            col_name_loss = f'Loss_AC_{line.name}'

            line_data[col_name_load] = load
            line_data[col_name_loss] = Ploss

        for line in grid.lines_DC:

            node = line.fromNode
            G = grid.Graph_node_to_Grid_index_DC[node]

            Ploss = np.real(line.loss)

            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber

            p_to = grid.Pij_DC[j, i]
            p_from = grid.Pij_DC[i, j]

            load = max(p_to, p_from)

            lossP_DC[G] += Ploss

            col_name_load = f'Load_DC_{line.name}'
            col_name_loss = f'Loss_DC_{line.name}'

            line_data[col_name_load] = load
            line_data[col_name_loss] = Ploss

        tot = 0
        for g in range(grid.Num_Grids_AC):
            col_name_grid = f'Loss_AC_Grid_{g+1}'
            loss_g = lossP_AC[g]
            tot += loss_g
            grid_data[col_name_grid] = loss_g

        for g in range(grid.Num_Grids_DC):
            col_name_grid_DC = f'Loss_DC_Grid_{g+1}'
            loss_g = lossP_DC[g]
            tot += loss_g
            grid_data[col_name_grid_DC] = loss_g

        grid_data['Total'] = tot
        grid.Time_series_input.append(in_data)
        grid.Time_series_price.append(price_data)
        grid.Time_series_money.append(money_data)
        grid.Time_series_res.append(row_data)
        grid.Time_series_line_res.append(line_data)
        grid.Time_series_grid_res.append(grid_data)
        if OPF == True:
            grid.Time_series_Opt_res_P_conv_DC.append(opt_res_P_conv_DC_dict)
            grid.Time_series_Opt_res_P_extGrid.append(opt_res_P_extGrid_dict)
        # print(idx+1)
        idx += 1
    # Create the DataFrame from the list of rows
    if OPF == True:
        grid.Time_series_Opt_res_P_conv_DC = pd.DataFrame(grid.Time_series_Opt_res_P_conv_DC)
        grid.Time_series_Opt_res_P_extGrid = pd.DataFrame(grid.Time_series_Opt_res_P_extGrid)
        grid.Time_series_Opt_res_P_conv_DC.set_index('time', inplace=True)
        grid.Time_series_Opt_res_P_extGrid.set_index('time', inplace=True)
    grid.Time_series_input = pd.DataFrame(grid.Time_series_input)
    grid.Time_series_res = pd.DataFrame(grid.Time_series_res)
    grid.Time_series_line_res = pd.DataFrame(grid.Time_series_line_res)
    grid.Time_series_grid_res = pd.DataFrame(grid.Time_series_grid_res)
    grid.Time_series_price = pd.DataFrame(grid.Time_series_price)
    grid.Time_series_money = pd.DataFrame(grid.Time_series_money)

    # Set the 'time' column as the idx
    grid.Time_series_input.set_index('time', inplace=True)
    grid.Time_series_res.set_index('time', inplace=True)
    grid.Time_series_line_res.set_index('time', inplace=True)
    grid.Time_series_grid_res.set_index('time', inplace=True)
    grid.Time_series_price.set_index('time', inplace=True)
    grid.Time_series_money.set_index('time', inplace=True)

    grid.Time_series_ran = True

