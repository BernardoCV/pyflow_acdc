# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:38:12 2024

@author: BernardoCastro
"""

import numpy as np
import pandas as pd


from PyFlow_ACDC_PF import *

try:
    import pyomo
    from PyFlow_ACDC_OPF import *
    pyomo_imp= True
    
except ImportError:    
    pyomo_imp= False
weights_def = {
    'Ext_Gen'         : {'w': 1},
    'Energy_cost'     : {'w': 0},
    'AC_losses'       : {'w': 1},
    'DC_losses'       : {'w': 1},
    'Converter_Losses': {'w': 1},
    'Curtailment_Red' : {'w': 1}
}
def Time_series_PF(grid):
    if grid.nodes_AC == None:
        print("only DC")
    elif grid.nodes_DC == None:
        print("only AC")
    else:
        print("Sequential")
        grid.TS_ACDC_PF(grid)

def TS_ACDC_PF(grid, start=1, end=99999, OPF=False,OPF_w=weights_def ,VarPrice=False,OnlyGen=True):
    idx = start-1
    Time_series_step1=True
    TS_len = len(grid.Time_series[0].TS)
    if TS_len < end:
        max_time = TS_len
    else:
        max_time = end
    grid.Time_series_res = []
    grid.Time_series_line_res = []
    grid.Time_series_MTDC_line_res = []
    grid.Time_series_grid_loss = []
    grid.Time_series_grid_loading = []
    grid.Time_series_input = []
    grid.Time_series_WPP = []

    if OPF == True:
        grid.Time_series_Opt_res_P_conv_DC = []
        grid.Time_series_Opt_res_P_extGrid = []
        grid.Time_series_Opt_res_React_extGrid=[]
        grid.Time_series_Opt_curtailment =[]
    grid.Time_series_price = []

    # saving droop configuration to reset each time, if not it takes power set from previous point.
    grid.Pconv_save = np.zeros(grid.nconv)
    for conv in grid.Converters_ACDC:
        grid.Pconv_save[conv.ConvNumber] = conv.P_DC

    cash = 0
    while idx < max_time:
        price_data = {'time': idx+1}
        WPP_data   = {'time' :idx+1}
        for ts in grid.Time_series:
            typ = ts.type
            if typ == 'Load':
                ts.node.PLi = ts.TS[idx]
            elif typ == 'WPP' or typ =='OWPP':
                WPP_data[f'{ts.name}']= ts.TS[idx]
                ts.node.PGi=ts.TS[idx]
            elif typ == 'Slack' or typ == 'Generator' :
                price_data[f'{ts.node.name}'] = grid.Markets_dic[ts.market][idx]
               
            else:
                ts.node.PGi = ts.TS[idx]
                
        if VarPrice == True or OPF_w['Energy_cost']['w']!=0:
            if OPF_w['Energy_cost']['w']==0:
               OPF_w['Energy_cost']['w']=1
            grid.VarPrice = True
            for ts in grid.Time_series:
                ts.node.price = grid.Markets_dic[ts.market][idx]
                
                
          
        if OPF == True and pyomo_imp==True:
            # if Time_series_step1==True:
            [model, results] = OPF_ACDC(grid,ObjRule=OPF_w,OnlyGen=OnlyGen)
                # Time_series_step1=False
            # else:
            #     model= OPF_updateParam(grid,model)
            
            #     [model,results]= OPF_solve(model)
            #     ExportACDC_model_toPyflowACDC(grid,model)   
    
            # [opt_res_P_conv_DC,opt_res_P_conv_AC,opt_res_Q_conv_AC,opt_res_P_extGrid,opt_res_Q_extGrid,opt_res_curtailment] = OPF_conv_results(grid,model)
            [opt_res_P_conv_DC_dict, opt_res_P_conv_AC_dict ,opt_res_Q_conv_AC_dict, opt_res_P_extGrid_dict,opt_res_Q_extGrid_dict,opt_res_curtailment_dict,opt_res_React_extGrid_dict] = OPF_conv_results(grid,model,Dict=True)

            
            opt_res_curtailment_dict['time'] = idx+1
            opt_res_P_conv_DC_dict['time'] = idx+1
            opt_res_P_extGrid_dict['time'] = idx+1
            opt_res_React_extGrid_dict['time'] = idx+1
            
            
            # print(f'Time series step {idx+1}')
        for conv in grid.Converters_ACDC:

            
            if conv.type == 'Droop' or conv.type== 'P':
                if OPF == False:
                    conv.P_DC = grid.Pconv_save[conv.ConvNumber] #This resets the converters droop target
            
        cash_inst = 0
        if OPF==False:
            ACDC_sequential(grid,QLimit=False)

        
        row_data = {'time': idx+1}
        in_data = {'time': idx+1}
        line_data = {'time': idx+1}
        MTDC_line_data = {'time': idx+1}
        grid_data = {'time': idx+1}
        grid_data_loading = {'time': idx+1}

        for ts in grid.Time_series:
            if ts.type == 'Slack' or ts.type == 'Generator':
                node = ts.node
                PGi = (node.P_INJ-node.P_s+node.PLi).item()
                QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi
                col_namePg_ = f'Pg_{ts.name}'
                col_nameQg_ = f'Qg_{ts.name}'
                col_nameLoading = f'Loading_{ts.name}'
                loading= np.sqrt(PGi**2+QGi**2)*grid.S_base/ts.element_MW
                # Append new data to the existing DataFrame
                row_data[col_namePg_] = PGi
                row_data[col_nameQg_] = QGi
                row_data[col_nameLoading] = loading
            elif ts.type == 'Reactor':
                node = ts.node
                PGi = 0
                QGi = node.Q_INJ-node.Q_s-node.Q_s_fx+node.QLi
                
                col_nameQg_ = f'Qg_{ts.name}'
                col_nameLoading = f'Loading_{ts.name}'
                loading= np.sqrt(PGi**2+QGi**2)*grid.S_base/ts.element_MW
                # Append new data to the existing DataFrame
               
                row_data[col_nameQg_] = QGi
                row_data[col_nameLoading] = loading
            else:
                in_data[ts.name] = ts.TS[idx]
        
        for conv in grid.Converters_ACDC:
            S_AC = np.sqrt(conv.P_AC**2+conv.Q_AC**2)
            P_DC = conv.P_DC
            col_name = f'Loading_{conv.name}'
            col_name2 = f'{conv.name}_P_DC'
            row_data[col_name] = np.maximum(S_AC, np.abs(P_DC))*grid.S_base/conv.MVA_max
            row_data[col_name2] = P_DC

        grid.Line_AC_calc()
        grid.Line_DC_calc()
        lossP_AC = np.zeros(grid.Num_Grids_AC)
        lossP_DC = np.zeros(grid.Num_Grids_DC)
        loadS_AC = np.zeros(grid.Num_Grids_AC)
        loadP_DC = np.zeros(grid.Num_Grids_DC)

        for line in grid.lines_AC:
            node = line.fromNode
            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber
            name = line.name
            G = grid.Graph_line_to_Grid_index_AC[line]
            Ploss = np.real(line.loss)

            Sfrom = abs(grid.Sij[i, j])
            Sto = abs(grid.Sij[j, i])

            load = max(Sfrom, Sto)

            lossP_AC[G] += Ploss
            loadS_AC[G] += load

            col_name_load = f'Loading_Line_{line.name}'
            

            line_data[col_name_load] = load*grid.S_base/line.MVA_rating
            
        saving_lines={'LionLink','Nautilus','TritonLink','BE-NL Link','DE-NL Link','DK-DE Link'}    
        for line in grid.lines_DC:

            node = line.fromNode
            G = grid.Graph_line_to_Grid_index_DC[line]

    
            Ploss = np.real(line.loss)

            i = line.fromNode.nodeNumber
            j = line.toNode.nodeNumber

            p_to = grid.Pij_DC[j, i]
            p_from = grid.Pij_DC[i, j]

            load = max(p_to, p_from)
            
            col_name_load = f'Loading_Line_{line.name}'
            col_name_loss = f'Loss_DC_{line.name}'
            if line.name in saving_lines:
                opt_res_P_conv_DC_dict[line.name]=p_from
            if grid.Graph_number_lines_DC[G] >=2:
                i=grid.Graph_grid_to_MTDC[G]
                MTDC_line_data[line.name,i] = load*grid.S_base/line.MW_rating   
            else:
                line_data[col_name_load] = load*grid.S_base/line.MW_rating
                # line_data[col_name_loss] = Ploss
            
            # lossP_DC[G] += Ploss
            loadP_DC[G] += load

            

            

        tot = 0
        tot_loading =0
        for g in range(grid.Num_Grids_AC):
            col_name_grid = f'Loss_AC_Grid_{g+1}'
            loss_g = lossP_AC[g]
            tot += loss_g
            # grid_data[col_name_grid] = loss_g
           
            col_name_grid_load = f'Loading_Grid_AC_{g+1}'
            loading = loadS_AC[g]
            tot_loading += loading
            grid_data_loading[col_name_grid_load] =0
            if grid.rating_grid_AC[g] !=0:
                grid_data_loading[col_name_grid_load] = loading*grid.S_base/grid.rating_grid_AC[g]
            
        for g in range(grid.Num_Grids_DC):
            col_name_grid_DC = f'Loss_Grid_DC_{g+1}'
            loss_g = lossP_DC[g]
            tot += loss_g
            # grid_data[col_name_grid_DC] = loss_g

            col_name_grid_load = f'Loading_Grid_DC_{g+1}'
            loading = loadP_DC[g]
            tot_loading += loading
            grid_data_loading[col_name_grid_load] = loading*grid.S_base/grid.rating_grid_DC[g]
            s=1            
                
                
               
            
            
             
        grid_data['Total'] = tot
        grid.Time_series_input.append(in_data)
        grid.Time_series_price.append(price_data)
        grid.Time_series_WPP.append(WPP_data)
     
        grid.Time_series_res.append(row_data)
        grid.Time_series_line_res.append(line_data)
        grid.Time_series_MTDC_line_res.append(MTDC_line_data)
        grid.Time_series_grid_loss.append(grid_data)
        grid.Time_series_grid_loading.append(grid_data_loading)
        
        
        
        if OPF == True:
            grid.Time_series_Opt_res_P_conv_DC.append(opt_res_P_conv_DC_dict)
            grid.Time_series_Opt_res_P_extGrid.append(opt_res_P_extGrid_dict)
            grid.Time_series_Opt_res_React_extGrid.append(opt_res_React_extGrid_dict)
            
            
            
            grid.Time_series_Opt_curtailment.append(opt_res_curtailment_dict)
        print(idx+1)
        idx += 1
        
    # Create the DataFrame from the list of rows
    if OPF == True:
        grid.Time_series_Opt_res_P_conv_DC = pd.DataFrame(grid.Time_series_Opt_res_P_conv_DC)
        grid.Time_series_Opt_res_P_extGrid = pd.DataFrame(grid.Time_series_Opt_res_P_extGrid)
        grid.Time_series_Opt_res_React_extGrid = pd.DataFrame(grid.Time_series_Opt_res_React_extGrid)
        grid.Time_series_Opt_curtailment   = pd.DataFrame(grid.Time_series_Opt_curtailment)
        grid.Time_series_Opt_res_P_conv_DC.set_index('time', inplace=True)
        grid.Time_series_Opt_res_P_extGrid.set_index('time', inplace=True)
        grid.Time_series_Opt_res_React_extGrid.set_index('time', inplace=True)
        grid.Time_series_Opt_curtailment.set_index('time', inplace=True)
    grid.Time_series_input = pd.DataFrame(grid.Time_series_input)
    grid.Time_series_res = pd.DataFrame(grid.Time_series_res)
    grid.Time_series_line_res = pd.DataFrame(grid.Time_series_line_res)
    grid.Time_series_MTDC_line_res = pd.DataFrame(grid.Time_series_MTDC_line_res)
    grid.Time_series_grid_loss = pd.DataFrame(grid.Time_series_grid_loss)
    grid.Time_series_grid_loading = pd.DataFrame(grid.Time_series_grid_loading)
    
    grid.Time_series_price = pd.DataFrame(grid.Time_series_price)
    grid.Time_series_WPP   = pd.DataFrame(grid.Time_series_WPP)
   

    # Set the 'time' column as the idx
    grid.Time_series_input.set_index('time', inplace=True)
    grid.Time_series_res.set_index('time', inplace=True)
    grid.Time_series_line_res.set_index('time', inplace=True)
    grid.Time_series_MTDC_line_res.set_index('time', inplace=True)
    grid.Time_series_grid_loss.set_index('time', inplace=True)
    grid.Time_series_grid_loading.set_index('time', inplace=True)
    grid.Time_series_price.set_index('time', inplace=True)
    grid.Time_series_WPP.set_index('time', inplace=True)


    grid.Time_series_ran = True
  
def export_TS(grid,start,end,excel_file_path=None,grid_names=None,stats=None):
    
    money = -1*grid.Time_series_price*grid.Time_series_Opt_res_P_extGrid*grid.S_base
    grouped_columns = {}
    money_joined =pd.DataFrame()
    Ext_Gen_joined = pd.DataFrame()
    React=grid.Time_series_Opt_res_React_extGrid
    for col in money.columns:
        prefix = ''.join(filter(str.isalpha, col))
        if prefix not in grouped_columns:
            grouped_columns[prefix] = []
        grouped_columns[prefix].append(col)
    
    # Aggregate columns with the same prefix
    for prefix, cols in grouped_columns.items():
        money_joined[f'{prefix}'] = money[cols].sum(axis=1)    
        Ext_Gen_joined[f'{prefix}'] = grid.Time_series_Opt_res_P_extGrid[cols].sum(axis=1)    
    money_all=pd.DataFrame()
    money_all.index=money.index
    money_all['Instant']=money.sum(axis=1)  
    money_all['Aggregative'] = pd.Series(0, index=money_all.index)
    money_all.loc[start, 'Aggregative'] =money_all.loc[start, 'Instant']
    for index in money_all.index[1:]: 
        money_all.loc[index, 'Aggregative'] = money_all.loc[index - 1, 'Aggregative'] + money_all.loc[index, 'Instant']
    
    
    loading_MTDC= {}
    for mg in range(grid.num_MTDC):
        g=grid.MTDC[mg]
        loading_MTDC[mg] = pd.DataFrame()
        for line in grid.lines_DC:
            if grid.Graph_line_to_Grid_index_DC[line]==g:
                loading_MTDC[mg][line.name]= grid.Time_series_MTDC_line_res[line.name,mg]
      
    
    loading= grid.Time_series_grid_loading.copy()
    loading.columns = loading.columns.str.replace('Loading_', '')
    
    
    lines= grid.Time_series_line_res.copy()
    lines.columns = lines.columns.str.replace('Loading_Line_', '')
    
    
    
    for col in loading.columns:
        grid_number = col.split('_')[-1]
        g= int(grid_number)-1
        if 'AC_' in col:
            if  grid.rating_grid_AC[g] !=0:
                loading[col]= loading[col]
                s=1
            else:
                  loading.drop(columns=col,inplace=True)
        elif 'DC_' in col:
            if  grid.rating_grid_DC[g] !=0:
                loading[col]= loading[col]
    if grid_names is not None:
        loading =loading.rename(columns=grid_names)
    curt=grid.Time_series_Opt_curtailment.rename(columns=grid.OWPP_node_to_ts)
    WPP=grid.Time_series_WPP*(curt)
    
    
    React*=grid.S_base
    Ext_Gen_joined*=grid.S_base
    conv=grid.Time_series_Opt_res_P_conv_DC*grid.S_base
    WPP*=grid.S_base
    WPP_base=grid.Time_series_WPP*grid.S_base
    curt=1-curt
    curt*=100
    loading*=100
    lines*=100
    
    
    Cases_res={'TS curtailment' :curt,
              'TS Conv'        :grid.Time_series_Opt_res_P_conv_DC,
              'TS loading grid':loading,
              'TS ExtG'        :grid.Time_series_Opt_res_P_extGrid ,
              'TS Money Exchange':money,
              'TS WPP':grid.Time_series_WPP,
              'TS WPP curtailed':WPP,
              'TS Money joined':money_joined,
              'TS ExtG joined' :Ext_Gen_joined
              }
    for mg in range(grid.num_MTDC):
        Cases_res[f'MTDC_{mg}'] = loading_MTDC[mg]
    
    if excel_file_path is not None:
        # Create an ExcelWriter object
        with pd.ExcelWriter(excel_file_path) as writer:
            # Write each DataFrame to a separate sheet
            curt.to_excel(writer, sheet_name='TS curtailment', index=True)
            loading.to_excel(writer, sheet_name='TS loading grid', index=True)
            Ext_Gen_joined.to_excel(writer, sheet_name='TS ExtG MW', index=True)
            React.to_excel(writer, sheet_name='TS React MVAR', index=True)
            money_joined.to_excel(writer, sheet_name='TS Money Eu', index=True)
            money_all.to_excel(writer, sheet_name='TS Money all Eu', index=True)
            conv.to_excel(writer, sheet_name='TS Conv DC P MW', index=True)
            WPP_base.to_excel(writer, sheet_name='TS WPP MW', index=True)
            WPP.to_excel(writer, sheet_name='TS WPP curtailed MW', index=True)
            lines.to_excel(writer, sheet_name='TS loading lines', index=True)
            if stats is not None:
                stats.to_excel(writer, sheet_name='stats', index=True)
            for mg in range(grid.num_MTDC):
                LMTDC=loading_MTDC[mg]*100
                LMTDC.to_excel(writer, sheet_name=f'MTDC_{mg} Loading', index=True)
    return Cases_res    