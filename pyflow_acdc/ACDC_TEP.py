# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:58:18 2024

@author: BernardoCastro
"""

import numpy as np
import pyomo.environ as pyo
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor


from .ACDC_OPF_model import OPF_createModel_ACDC,analyse_OPF,ExportACDC_model_toPyflowACDC
from .ACDC_OPF import OPF_solve


__all__ = [
    'TEP_expansion_model',
    'update_grid_price_zone_data',
    'expand_elements_from_pd',
    'update_attributes',
    'Expand_element',
    'Translate_pd_TEP'
]

def pack_variables(*args):
    return args


def update_grid_price_zone_data(grid,ts,t,n_clusters,clustering):
    idx=t-1
    typ = ts.type
    
    if clustering:
        ts_data = ts.data_clustered[n_clusters]
    else:
        ts_data = ts.data
    
    if typ == 'a_CG':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.a = ts_data[idx]
                break
    elif typ == 'b_CG':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.b = ts_data[idx]
                break
    elif typ == 'c_CG':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.c = ts_data[idx]
                break
    elif typ == 'PGL_min':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.PGL_min= ts_data[idx]
                break
    elif typ == 'PGL_max':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.PGL_max= ts_data[idx]
                break
    if typ == 'price':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.price = ts_data[idx]
                break  # Stop after assigning to the correct price_zone
        for node in grid.nodes_AC:
            if ts.element_name == node.name:
                node.price = ts_data[idx]
                break  # Stop after assigning to the correct node    
    
    elif typ == 'Load':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.PLi_factor = ts_data[idx]
                break  # Stop after assigning to the correct price_zone
        for node in grid.nodes_AC:
            if ts.element_name == node.name:
                node.PLi_factor = ts_data[idx]
                break  # Stop after assigning to the correct node
    elif typ in ['WPP', 'OWPP','SF','REN']:
        for zone in grid.RenSource_zones:
            if ts.element_name == zone.name:
                zone.PRGi_available = ts_data[idx]
                # print(ts_data[idx])
                break  # Stop after assigning to the correct zone
        for rs in grid.RenSources:
            if ts.element_name == rs.name:
                rs.PRGi_available = ts_data[idx]
                break  # Stop after assigning to the correct node

def expand_elements_from_pd(grid,exp_elements):
    """
    This function iterates over exp_elements and applies Expand_element 
    with the corresponding columns (N_i, Life_time, and base_cost) if available.
    
    Parameters:
    exp_elements: DataFrame containing element data.
    grid: The grid object to be passed to Expand_element.
    """
    
    # Helper function to get the column value if it exists
    def get_column_value(row, col_name):
        return row[col_name] if col_name in row.index else None
    
    # Apply the Expand_element function for each element in exp_elements
    exp_elements.iloc[:, 0].apply(lambda name: Expand_element(
        grid,
        name,
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'N_i'),
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'N_max'),
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'Life_time'),
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'base_cost'),
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'per_unit_cost'),
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'phi'),
        get_column_value(exp_elements.loc[exp_elements[exp_elements.iloc[:, 0] == name].index[0], :], 'exp')
    ))

def update_attributes(element, N_i, N_max, Life_time, base_cost, per_unit_cost, phi, exp):
   """Updates the attributes of the given element if not None."""
   if N_i is not None:
       if hasattr(element, 'np_line'):
           element.np_line = N_i
       if hasattr(element, 'np_line_i'):
           element.np_line_i = N_i
       if hasattr(element, 'NumConvP'):
           element.NumConvP = N_i  
       if hasattr(element, 'NumConvP_i'):
           element.NumConvP_i = N_i      # Only set if it exists
   if N_max is not None:
       if hasattr(element, 'np_line_max'):
           element.np_line_max = N_max
       if hasattr(element, 'NumConvP_max'):
           element.NumConvP_max = N_max  
    
   if Life_time is not None:
       element.life_time = Life_time
   if base_cost is not None:
       element.base_cost = base_cost
   if per_unit_cost is not None:
       if hasattr(element, 'cost_perMWkm'):
           element.cost_perMWkm = per_unit_cost
       if hasattr(element, 'cost_perMVAkm'):
           element.cost_perMVAkm = per_unit_cost    
       if hasattr(element, 'cost_perMVA'):
           element.cost_perMVA = per_unit_cost
   if phi is not None:
       element.phi = phi
   else:
       phi_calculation(element)
   if exp is not None:
       element.exp = exp
        
def Expand_element(grid,name,N_i=None,N_max=None,Life_time=None,base_cost=None,per_unit_cost=None,phi=None, exp=None):
    
    if N_max is None:
        N_max= N_i+20
    
    for l in grid.lines_AC:
        if name == l.name:
            from .Class_editor import change_line_AC_to_expandable
            change_line_AC_to_expandable(grid, name)
    
    
    for l in grid.lines_AC_exp:
        if name == l.name:
            l.np_line_opf = True
            update_attributes(l, N_i, N_max,Life_time, base_cost, per_unit_cost, phi, exp)
            continue

    for l in grid.lines_DC:
        if name == l.name:
            l.np_line_opf = True
            update_attributes(l, N_i, N_max,Life_time, base_cost, per_unit_cost, phi, exp)
            continue
            
    for cn in grid.Converters_ACDC:
        if name == cn.name:
            cn.NUmConvP_opf = True
            update_attributes(cn, N_i, N_max, Life_time, base_cost, per_unit_cost, phi, exp)
            continue
            
def phi_calculation(element):
    from .Classes import Exp_Line_AC 
    if isinstance(element, Exp_Line_AC):
        if element.base_cost is not None:
            element.phi= element.base_cost/(element.life_time*8760*element.Length_km*element.MVA_rating)
            element.cost_perMVAkm = element.base_cost/(element.Length_km*element.MVA_rating)
        elif element.cost_perMVAkm is not None:
            element.phi= element.cost_perMVAkm/(element.life_time*8760)
    
    from .Classes import Line_DC 
    if isinstance(element, Line_DC):
        if element.base_cost is not None:
            element.phi= element.base_cost/(element.life_time*8760*element.Length_km*element.MW_rating)
            element.cost_perMVAkm = element.base_cost/(element.Length_km*element.MW_rating)
        elif element.cost_perMWkm is not None:
            element.phi= element.cost_perMWkm/(element.life_time*8760)
    from .Classes import AC_DC_converter
    if isinstance(element, AC_DC_converter):
        if element.base_cost is not None:
            element.phi= element.base_cost/(element.life_time*8760*element.MVA_max)
            element.cost_perMVA = element.base_cost/(element.MVA_max)
        elif element.cost_perMVA is not None:
            element.phi=element.cost_perMVA/(element.life_time*8760)

def Translate_pd_TEP(grid):
    """Translation of element wise to internal numbering"""
    # Price_Zones
    price_zone2node, price_zone_prices, price_zone_as, price_zone_bs, PGL_min, PGL_max, PL_price_zone = {}, {}, {}, {}, {}, {}, {}
    nn_M, node2price_zone, lista_M = 0, {}, []
    
    for m in grid.Price_Zones:
        price_zone2node[m.price_zone_num] = []
        nn_M += 1
        price_zone_prices[m.price_zone_num] = m.price
        price_zone_as[m.price_zone_num] = m.a
        price_zone_bs[m.price_zone_num] = m.b
        PGLmin = m.PGL_min
        PGLmax = m.PGL_max
        import_M = m.import_pu_L
        export_M = m.export_pu_G * (sum(node.PGi_ren + node.Max_pow_gen for node in m.nodes_AC))
        PL_price_zone[m.price_zone_num] = 0
        for n in m.nodes_AC:
            price_zone2node[m.price_zone_num].append(n.nodeNumber)
            node2price_zone[n.nodeNumber] = m.price_zone_num
            PL_price_zone[m.price_zone_num] += n.PLi
        PGL_min[m.price_zone_num] = max(PGLmin, -import_M * PL_price_zone[m.price_zone_num])
        PGL_max[m.price_zone_num] = min(PGLmax, export_M)
    lista_M = list(range(0, nn_M))

    Price_Zone_Lists = pack_variables(lista_M, node2price_zone, price_zone2node)
    Price_Zone_lim = pack_variables(price_zone_as, price_zone_bs, PGL_min, PGL_max)

   
    Price_Zone_info = pack_variables(Price_Zone_Lists, Price_Zone_lim)

    return Price_Zone_info

def TEP_expansion_model(grid,export=None,costmodel='Linear',n_clusters=None,cluster_algorithm='Kmeans',increase_Pmin=False,NPV=False,n_years=25,discount_rate=0.02):
    
    from .Time_series import cluster_TS, modify_parameters
    
    grid.TEP_n_years = n_years
    grid.TEP_discount_rate =discount_rate
    clustering = False
    if n_clusters is not None:
        n_clusters,_ = cluster_TS(grid,n_clusters,cluster_algorithm)
        clustering = True
    else:
        n_clusters = len(grid.Time_series[0].data)
        if n_clusters > 24: #limit to be revised 
            print(f"Warning: The number of clusters ({n_clusters}) exceeds 24. This may affect performance.")
    
    
    index = ["A", "B", "D", "E"]
    data = {
    "80kV": [-251790, 0.03198, 220000, 8.98],
    "150kV": [-100000, 0.0164, 220000, 8.98],
    "320kV": [286000, 0.00969, 220000, 8.98],
    "525kV": [745400, 0.0061, 220000, 8.98]
    }


    grid.DC_cables_cost_index= pd.DataFrame(data, index=index)
 
    
    NumConvP_i,NumConvP_max={},{}
    S_limit_conv={}
    P_lineDC_limit ={}
    NP_lineDC_i,NP_lineDC_max ={},{}
    NP_lineAC_i,NP_lineAC_max = {},{}
    Line_length ={}
    lista_lineas_DC = list(range(0, grid.nl_DC))
    lista_conv = list(range(0, grid.nconv))
    lista_AC   = list(range(0,grid.nle_AC))
    
    line_phi={}
    conv_phi={}
    
    for l in grid.lines_AC_exp:
        NP_lineAC_i[l.lineNumber]     = l.np_line+1 if l.np_line+1<=l.np_line_max else l.np_line_max
        NP_lineAC_max[l.lineNumber]   = l.np_line_max
        
    for conv in grid.Converters_ACDC:
        NumConvP_i [conv.ConvNumber]  = conv.NumConvP+1 if conv.NumConvP+1<=conv.NumConvP_max else conv.NumConvP_max
        NumConvP_max[conv.ConvNumber] = conv.NumConvP_max
        S_limit_conv[conv.ConvNumber] = conv.MVA_max/grid.S_base
    for l in grid.lines_DC:
        P_lineDC_limit[l.lineNumber]  = l.MW_rating/grid.S_base
        NP_lineDC_i[l.lineNumber]     = l.np_line+1 if l.np_line+1<=l.np_line_max else l.np_line_max
        NP_lineDC_max[l.lineNumber]   = l.np_line_max
        Line_length[l.lineNumber]     = l.Length_km
        
    
    
    conv_var=pack_variables(NumConvP_i,S_limit_conv,conv_phi)
    line_var=pack_variables(P_lineDC_limit,NP_lineDC_i,Line_length,line_phi)
    TEP=pack_variables(NP_lineDC_i,NumConvP_i)
    t1 = time.time()
    model = pyo.ConcreteModel()
    model.name        ="TEP MTDC AC/DC hybrid OPF"
    model.Time_frames = pyo.Set(initialize=range(1, n_clusters + 1))
    
    #print(list(model.Time_frames))
    model.submodel    = pyo.Block(model.Time_frames)
    model.conv        = pyo.Set(initialize=lista_conv)
    model.lines_DC    = pyo.Set(initialize=lista_lineas_DC)
    model.lines_AC_exp= pyo.Set(initialize=lista_AC)
    
    w={}
    coeff={}
    
    base_model = pyo.ConcreteModel()
    base_model = OPF_createModel_ACDC(base_model,grid,PV_set=False,Price_Zones=True,TEP=True)
    
    s=1
    
    for t in model.Time_frames:
        base_model_copy = base_model.clone()
        model.submodel[t].transfer_attributes_from(base_model_copy)
        
        for ts in grid.Time_series:
            update_grid_price_zone_data(grid,ts,t,n_clusters,clustering)
        if increase_Pmin: 
            for price_zone in grid.Price_Zones:
                 if price_zone.b > 0:
                     price_zone.PGL_min -= price_zone.ImportExpand
                     price_zone.a = -price_zone.b / (2 * price_zone.PGL_min * grid.S_base) 
        modify_parameters(grid,model.submodel[t],False,True)
        
        TEP_subObj(model.submodel[t])
        if clustering:
            w[t]= float(grid.Clusters[n_clusters][t-1])
            # num_time_frames = len(model.Time_frames)
            # w[t]=1/num_time_frames
    
        else:
            num_time_frames = len(model.Time_frames)
            w[t]=1/num_time_frames
    
    
    def NPline_bounds_AC(model, line):
        element=grid.lines_AC_exp[line]
        if element.np_line_opf:
            return (NP_lineAC_i[line]-1, NP_lineAC_max[line])
        else:
            return (NP_lineAC_i[line]-1, NP_lineAC_i[line]-1)
    
    model.NumLinesACP = pyo.Var(model.lines_AC_exp, bounds=NPline_bounds_AC,initialize=NP_lineAC_i)
    
    s=1
    
    
    def NPline_bounds(model, line):
        element=grid.lines_DC[line]
        if element.np_line_opf:
            return (NP_lineDC_i[line]-1, NP_lineDC_max[line])
        else:
            return (NP_lineDC_i[line]-1, NP_lineDC_i[line]-1)
    
    model.NumLinesDCP = pyo.Var(model.lines_DC, bounds=NPline_bounds,initialize=NP_lineDC_i)
    
    def NPconv_bounds(model, conv):
        element=grid.Converters_ACDC[conv]
        if element.NUmConvP_opf:
            return (NumConvP_i[conv]-1, NumConvP_max[conv])
        else:
            return (NumConvP_i[conv]-1, NumConvP_i[conv]-1)
    
    model.NumConvP = pyo.Var(model.conv, bounds=NPconv_bounds,initialize=NumConvP_i)
    
    def NP_ACline_link(model,line,t):
        element=grid.lines_AC_exp[line]
        if element.np_line_opf:
            return model.NumLinesACP[line] ==model.submodel[t].NumLinesACP[line]
        else:
            return pyo.Constraint.Skip
    
    
    def NP_line_link(model,line,t):
        element=grid.lines_DC[line]
        if element.np_line_opf:
            return model.NumLinesDCP[line] ==model.submodel[t].NumLinesDCP[line]
        else:
            return pyo.Constraint.Skip
    def NP_conv_link(model,conv,t):
        element=grid.Converters_ACDC[conv]
        if element.NUmConvP_opf:
            return model.NumConvP[conv] ==model.submodel[t].NumConvP[conv]
        else:
            return pyo.Constraint.Skip
    
    model.NP_ACline_link_constraint = pyo.Constraint(model.lines_AC_exp,model.Time_frames, rule=NP_ACline_link)
    model.NP_line_link_constraint = pyo.Constraint(model.lines_DC,model.Time_frames, rule=NP_line_link)
    model.NP_conv_link_constraint = pyo.Constraint(model.conv,model.Time_frames, rule=NP_conv_link)
    
    model.weights = pyo.Param(model.Time_frames, initialize=w)
    obj_rule= TEP_obj(model,grid,line_var,conv_var,costmodel,NPV,n_years,discount_rate)
    
    t2 = time.time()  
    t_modelcreate = t2-t1
    
    model_results,t_modelsolve= OPF_solve(model,grid)
    
    t1 = time.time()
    TEP_res = ExportACDC_TEP_toPyflowACDC(model,grid,n_clusters,clustering)   
    t2 = time.time()  
    t_modelexport = t2-t1
        
    # TEP_res ={}
    grid.OPF_run=True  
    grid.TEP_run=True
    
    timing_info = {
    "create": t_modelcreate,
    "solve": t_modelsolve,
    "export": t_modelexport,
    }
    
    return model, model_results ,TEP_res, timing_info

def TEP_subObj(submodel):
    # Define the social cost submodel function that can access the submodel's attributes
    def social_cost_submodel(model=None, index=None):
        return sum(submodel.SocialCost[price_zone] for price_zone in submodel.M)
    
    submodel.obj = pyo.Objective(rule=social_cost_submodel, sense=pyo.minimize)

def TEP_obj(model,grid,line_var,conv_var,costmodel,NPV,n_years,discount_rate):
    P_lineDC_limit,NP_lineDC_i,Line_length,line_phi = line_var
    NumConvP_i,S_limit_conv,conv_phi = conv_var
    
    OnlyAC,TEP_AC,TAP_tf=analyse_OPF(grid) 
    
    
    def AC_Line_investments():
        AC_Inv_lines=0
        for l in model.lines_AC_exp:
            line = grid.lines_AC_exp[l]
            if line.np_line_opf: 
               if NPV:
                   AC_Inv_lines+=model.NumLinesACP[l]*line.MVA_rating*line.Length_km*line.phi*line.life_time
               else: 
                   AC_Inv_lines+=model.NumLinesACP[l]*line.MVA_rating*line.Length_km*line.phi  
    
        return AC_Inv_lines
    def Cables_investments():
        Inv_lines=0
        for l in model.lines_DC:
           line= grid.lines_DC[l]
           if line.np_line_opf: 
             if NPV:
                 Inv_lines+=model.NumLinesDCP[l]*line.MW_rating*line.Length_km*line.phi*line.life_time
             else:
                 Inv_lines+=(model.NumLinesDCP[l]*line.MW_rating)*line.Length_km*line.phi
        return Inv_lines
            
    def Converter_investments():
        Inv_conv=0
        for cn in model.conv:
            conv= grid.Converters_ACDC[cn]
            if conv.NUmConvP_opf:
               if NPV: 
                 Inv_conv+=model.NumConvP[cn]*conv.MVA_max*conv.phi*conv.life_time
               else:
                 Inv_conv+=model.NumConvP[cn]*conv.MVA_max*conv.phi
        return Inv_conv
    
    
    # Calculate the weighted social cost for each submodel (subblock)
    weighted_social_cost = 0
    present_value =   (1 - (1 + discount_rate) ** -n_years) / discount_rate
    
        
    for t in model.Time_frames:
        submodel_sc = model.submodel[t].obj
        # Print types of the variables
        # print(f"t: {type(t)}")  # Check the type of t (likely int)
        # print(f"model.weights[t]: {type(model.weights[t])}")  # Check the type of the weight (likely float)
        # print(f"submodel_sc: {type(submodel_sc)}")  # Check the type of submodel_sc (should be a float or similar)
        
        # # Print values for debugging
        # print(f'{t}: {model.weights[t]}')
        
        # Calculate weighted social cost
        
        weighted_social_cost += model.weights[t] * submodel_sc
            
        model.submodel[t].obj.deactivate()
    if NPV:
        weighted_social_cost *=present_value
    
    if TEP_AC: 
        inv_line_AC = AC_Line_investments()
    else:
        inv_line_AC=0
    if not OnlyAC:
        inv_cable = Cables_investments()
        inv_conv = Converter_investments()
    else:
        inv_cable = 0
        inv_conv  = 0


    
    total_cost = weighted_social_cost + (inv_line_AC+inv_cable + inv_conv)
    
        
        
    model.obj = pyo.Objective(rule=total_cost, sense=pyo.minimize)
    
    return total_cost 


def get_price_zone_data(t, model, grid,n_clusters,clustering):
    row_data_price = {'Time_Frame': t}
    row_data_SC = {'Time_Frame': t}
    row_data_PN = {'Time_Frame': t}
    row_data_GEN = {'Time_Frame': t}
    # Collect price_zone data
    for m in grid.Price_Zones:
        nM = m.price_zone_num
        row_data_price[m.name] = np.round(np.float64(pyo.value(model.submodel[t].price_zone_price[nM])), decimals=2)
        
        from .Classes import Price_Zone
        from .Classes import MTDCPrice_Zone
        from .Classes import OffshorePrice_Zone
        gen=0
        for node in m.nodes_AC:
            nAC=node.nodeNumber
            PGi_ren = 0
            PGi_opt = sum(pyo.value(model.submodel[t].PGi_gen[gen.genNumber]) for gen in node.connected_gen)
            for rs in node.connected_RenSource:
                if rs.PGRi_linked:
                    rz = rs.Ren_source_zone
                    z  = grid.RenSource_zones[grid.RenSource_zones_dic[rz]]
                else:
                    z= rs
                try:    
                    if clustering:
                        factor = grid.Time_series[z.TS_dict['PRGi_available']].data_clustered[n_clusters][t-1]
                    else:
                        factor = grid.Time_series[z.TS_dict['PRGi_available']].data[t-1]
          
                    PGi_ren+=rs.PGi_ren_base*factor
                except KeyError:
                    PGi_ren+=rs.PGi_ren_base*rs.PRGi_available
                    print(f'Key {z} not found in Time series')   
                
                
            gen+=node.PGi +PGi_ren+PGi_opt
            
        row_data_GEN[m.name] = np.round(gen * grid.S_base, decimals=2)    

        if type(m) is Price_Zone:
            SC = np.float64(pyo.value(model.submodel[t].SocialCost[nM]))
            row_data_SC[m.name] = np.round(SC / 1000, decimals=2)

            PN = np.float64(pyo.value(model.submodel[t].PN[nM]))
            row_data_PN[m.name] = np.round(PN * grid.S_base, decimals=2)
            
            
            
            
    return row_data_price, row_data_SC, row_data_PN,row_data_GEN

def get_curtailment_data(t, model, grid,n_clusters,clustering):
    row_data_curt = {'Time_Frame': t}
    row_data_curt_per = {'Time_Frame': t}

    for rs in grid.RenSources:
        if rs.PGRi_linked:
            rz = rs.Ren_source_zone
            z  = grid.RenSource_zones[grid.RenSource_zones_dic[rz]]
        else:
            z= rs
        try:    
            if clustering:
                factor = grid.Time_series[z.TS_dict['PRGi_available']].data_clustered[n_clusters][t-1]
            else:
                factor = grid.Time_series[z.TS_dict['PRGi_available']].data[t-1]
  
            PGi_ren=rs.PGi_ren_base*factor
        except KeyError:
            PGi_ren=rs.PGi_ren_base*rs.PRGi_available
            print(f'Key {z} not found in Time series')    
         
        curt_value = np.round((1 - pyo.value(model.submodel[t].gamma[rs.rsNumber])) *PGi_ren* grid.S_base, decimals=2)
        row_data_curt[rs.name] = curt_value
        row_data_curt_per[rs.name] =  np.round(1 - pyo.value(model.submodel[t].gamma[rs.rsNumber]), decimals=2)*100

    return row_data_curt,row_data_curt_per

def get_line_data(t, model, grid):
    row_data_lines = {'Time_Frame': t}

    for l in grid.lines_DC:
        if l.np_line_opf:
            ln = l.lineNumber
            if l.np_line <= 0.00001:
                row_data_lines[l.name] = np.nan
            else:
                p_to = np.float64(pyo.value(model.submodel[t].PDC_to[ln])) * grid.S_base
                p_from = np.float64(pyo.value(model.submodel[t].PDC_from[ln])) * grid.S_base
                load = max(p_to, p_from) / l.MW_rating * 100
                row_data_lines[l.name] = np.round(load, decimals=0).astype(int)

    return row_data_lines

def get_converter_data(t, model, grid):
    row_data_conv = {'Time_Frame': t}

    for conv in grid.Converters_ACDC:
        if conv.NUmConvP_opf:
            cn = conv.ConvNumber
            if conv.NumConvP <= 0.00001:
                row_data_conv[conv.name] = np.nan
            else:
                P_DC = np.float64(pyo.value(model.submodel[t].P_conv_DC[conv.Node_DC.nodeNumber])) * grid.S_base
                P_s  = np.float64(pyo.value(model.submodel[t].P_conv_s_AC[cn])) * grid.S_base
                Q_s  = np.float64(pyo.value(model.submodel[t].Q_conv_s_AC[cn])) * grid.S_base
                S = np.sqrt(P_s**2 + Q_s**2)
                loading = max(S, abs(P_DC)) / (conv.MVA_max * conv.NumConvP) * 100
                row_data_conv[conv.name] = np.round(loading, decimals=0)
                

    return row_data_conv

def get_weight_data(model, t):
    return pyo.value(model.weights[t])


def ExportACDC_TEP_toPyflowACDC(model,grid,n_clusters,clustering):
    grid.V_AC =np.zeros(grid.nn_AC)
    grid.Theta_V_AC=np.zeros(grid.nn_AC)
    grid.V_DC=np.zeros(grid.nn_DC)
    
    SW= sum(pyo.value(model.weights[t]) for t in model.Time_frames)
    def process_ren_source(renSource):
        rs = renSource.rsNumber
        renSource.gamma =  np.float64(sum(pyo.value(model.submodel[t].gamma[rs]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
    
    def process_gen(gen):
        gn = gen.genNumber
        gen.PGen =  np.float64(sum(pyo.value(model.submodel[t].PGi_gen[gn]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        gen.QGen =  np.float64(sum(pyo.value(model.submodel[t].QGi_gen[gn]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
    
    
    def process_ac_node(node):
        nAC = node.nodeNumber
        node.V_AC = np.float64(sum(pyo.value(model.submodel[t].V_AC[nAC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        node.theta = np.float64(sum(pyo.value(model.submodel[t].thetha_AC[nAC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        node.P_s = np.float64(sum(pyo.value(model.submodel[t].P_conv_AC[nAC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        node.Q_s = np.float64(sum(pyo.value(model.submodel[t].Q_conv_AC[nAC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
    
        node.PGi_opt = np.float64(sum(pyo.value(model.submodel[t].PGi_opt[nAC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        node.QGi_opt = np.float64(sum(pyo.value(model.submodel[t].QGi_opt[nAC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
    
        grid.V_AC[nAC] = node.V_AC
        grid.Theta_V_AC[nAC] = node.theta
    
    # Helper function for DC nodes
    def process_dc_node(node):
        nDC = node.nodeNumber
        node.V = np.float64(sum(pyo.value(model.submodel[t].V_DC[nDC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        node.P = np.float64(sum(pyo.value(model.submodel[t].P_conv_DC[nDC]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        node.P_INJ = node.PGi - node.PLi + node.P
        grid.V_DC[nDC] = node.V
    
    # Helper function for converters
    def process_converter(conv):
        nconv = conv.ConvNumber
        nconvp=np.float64(pyo.value(model.NumConvP[nconv]))
        conv.P_DC  = np.float64(sum(pyo.value(model.submodel[t].P_conv_DC[conv.Node_DC.nodeNumber])   *nconvp* pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.P_AC  = np.float64(sum(pyo.value(model.submodel[t].P_conv_s_AC[nconv]) *nconvp* pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.Q_AC  = np.float64(sum(pyo.value(model.submodel[t].Q_conv_s_AC[nconv]) *nconvp* pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.Pc    = np.float64(sum(pyo.value(model.submodel[t].P_conv_c_AC[nconv]) *nconvp* pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.Qc    = np.float64(sum(pyo.value(model.submodel[t].Q_conv_c_AC[nconv]) *nconvp* pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.P_loss= np.float64(sum(pyo.value(model.submodel[t].P_conv_loss[nconv]) *nconvp* pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.P_loss_tf = abs(conv.P_AC - conv.Pc)
        conv.U_c   = np.float64(sum(pyo.value(model.submodel[t].Uc[nconv])   * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.U_f   = np.float64(sum(pyo.value(model.submodel[t].Uf[nconv])   * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.U_s   = np.float64(sum(pyo.value(model.submodel[t].V_AC[nconv]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.th_c  = np.float64(sum(pyo.value(model.submodel[t].th_c[nconv]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.th_f  = np.float64(sum(pyo.value(model.submodel[t].th_f[nconv]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.th_s  = np.float64(sum(pyo.value(model.submodel[t].thetha_AC[nconv]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        conv.NumConvP = nconvp
    # Helper function for price_zones
    def process_price_zone(m):
        nM = m.price_zone_num
        m.price = np.float64(sum(pyo.value(model.submodel[t].price_zone_price[nM]) * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        s=1
        from .Classes import Price_Zone
        if type(m) is Price_Zone:
       
            if clustering:
                m.a          = np.float64(sum(grid.Time_series[m.TS_dict['a_CG']].data_clustered[n_clusters][t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
                m.b          = np.float64(sum(grid.Time_series[m.TS_dict['b_CG']].data_clustered[n_clusters][t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
                m.PLi_factor = np.float64(sum(grid.Time_series[m.TS_dict['Load']].data_clustered[n_clusters][t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        
            else:
                m.a = np.float64(sum(grid.Time_series[m.TS_dict['a_CG']].data[t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
                m.b = np.float64(sum(grid.Time_series[m.TS_dict['b_CG']].data[t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
                m.PLi_factor = np.float64(sum(grid.Time_series[m.TS_dict['Load']].data[t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
        
    
    with ThreadPoolExecutor() as executor:
        futures_ac = [executor.submit(process_ac_node, node) for node in grid.nodes_AC]
        futures_dc = [executor.submit(process_dc_node, node) for node in grid.nodes_DC]
        futures_conv = [executor.submit(process_converter, conv) for conv in grid.Converters_ACDC]
        futures_price_zone = [executor.submit(process_price_zone, m) for m in grid.Price_Zones]
        futures_rs      = [executor.submit(process_ren_source, m) for m in grid.RenSources]
        futures_gen     = [executor.submit(process_gen, m) for m in grid.Generators]
       
    
        # Wait for all tasks to complete
        for future in futures_ac + futures_dc + futures_conv + futures_price_zone+futures_rs+futures_gen:
            future.result()
    
    Pf = np.zeros((grid.nn_AC, 1))
    Qf = np.zeros((grid.nn_AC, 1))

    G = np.real(grid.Ybus_AC)
    B = np.imag(grid.Ybus_AC)
    V = grid.V_AC
    Theta = grid.Theta_V_AC
    # Compute differences in voltage angles
    Theta_diff = Theta[:, None] - Theta
    
    # Calculate power flow
    Pf = (V[:, None] * V * (G * np.cos(Theta_diff) + B * np.sin(Theta_diff))).sum(axis=1)
    Qf = (V[:, None] * V * (G * np.sin(Theta_diff) - B * np.cos(Theta_diff))).sum(axis=1)
    

    for node in grid.nodes_AC:
        i = node.nodeNumber
        node.P_INJ = Pf[i]
        node.Q_INJ = Qf[i]

    NumLinesACP_values= {k: np.float64(pyo.value(v)) for k, v in model.NumLinesACP.items()}    
    NumLinesDCP_values= {k: np.float64(pyo.value(v)) for k, v in model.NumLinesDCP.items()}   

    for line in grid.lines_AC_exp:
        line.np_line=NumLinesACP_values[line.lineNumber] 
    # Parallelize DC line processing
    for line in grid.lines_DC:
        line.np_line = NumLinesDCP_values[line.lineNumber]

    for z in grid.RenSource_zones:
        if clustering:
            z.PRGi_available = np.float64(sum(grid.Time_series[z.TS_dict['PRGi_available']].data_clustered[n_clusters][t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)      
        else:
            z.PRGi_available = np.float64(sum(grid.Time_series[z.TS_dict['PRGi_available']].data[t-1] * pyo.value(model.weights[t]) for t in model.Time_frames) / SW)
           
    # Multithreading the time frame processing
    data_rows_PN = []
    data_rows_GEN= []
    data_rows_SC = []
    data_rows_curt = []
    data_rows_curt_per = []
    data_rows_lines = []
    data_rows_conv = []
    data_rows_price = []
    
    weights_row = []
    
    with ThreadPoolExecutor() as executor:
        futures = []
        
        for t in model.Time_frames:
            futures.append(executor.submit(get_price_zone_data, t, model, grid,n_clusters,clustering))
            futures.append(executor.submit(get_curtailment_data, t, model, grid,n_clusters,clustering))
            futures.append(executor.submit(get_line_data, t, model, grid))
            futures.append(executor.submit(get_converter_data, t, model, grid))
            futures.append(executor.submit(get_weight_data, model, t))
    
        # Retrieve results
        for i in range(0, len(futures), 5):
            price_data, SC_data, PN_data,GEN_data = futures[i].result()
            curt_data,curt_data_per = futures[i+1].result()
            lines_data = futures[i+2].result()
            conv_data = futures[i+3].result()
            weight_data = futures[i+4].result()
    
            data_rows_price.append(price_data)
            data_rows_SC.append(SC_data)
            data_rows_PN.append(PN_data)
            data_rows_GEN.append(GEN_data)
            data_rows_curt.append(curt_data)
            data_rows_curt_per.append(curt_data_per)
            data_rows_lines.append(lines_data)
            data_rows_conv.append(conv_data)
            weights_row.append(weight_data)
    
    # Convert to DataFrames
    data_PN = pd.DataFrame(data_rows_PN)
    data_GEN = pd.DataFrame(data_rows_GEN)
    data_SC = pd.DataFrame(data_rows_SC)
    data_curt = pd.DataFrame(data_rows_curt)
    data_curt_per = pd.DataFrame(data_rows_curt_per)
    data_lines = pd.DataFrame(data_rows_lines)
    data_conv = pd.DataFrame(data_rows_conv)
    data_price = pd.DataFrame(data_rows_price)
    
    # Transpose the DataFrame to flip rows and columns
    flipped_data_PN = data_PN.set_index('Time_Frame').T 
    flipped_data_GEN = data_GEN.set_index('Time_Frame').T 
    flipped_data_SC = data_SC.set_index('Time_Frame').T 
    flipped_data_curt = data_curt.set_index('Time_Frame').T 
    flipped_data_curt_per = data_curt_per.set_index('Time_Frame').T 
    flipped_data_lines = data_lines.set_index('Time_Frame').T 
    flipped_data_conv = data_conv.set_index('Time_Frame').T 
    flipped_data_price = data_price.set_index('Time_Frame').T 
    
    # Calculate Total SC
    total_sc = np.round(flipped_data_SC.sum(), decimals=2)
    
    # Calculate Weighted SC
    weighted_sc = np.round(total_sc * weights_row, decimals=2)
    
    # Create additional rows DataFrame
    additional_rows = pd.DataFrame({
        'Total SC': total_sc,
        '': [None] * len(total_sc),  # Blank row
        'Weight': weights_row,
        'Weighted SC': weighted_sc
    }).T
    
    # Combine original data with additional rows
    flipped_data_SC = pd.concat([flipped_data_SC, additional_rows])
    
    flipped_data_SC.loc['Weight'] = weights_row
    weighted_SC = flipped_data_SC.loc['Total SC'] * flipped_data_SC.loc['Weight']
    flipped_data_SC.loc['Weighted SC'] = np.round(weighted_SC, decimals=2)
    
    
    # Pack all variables into the final result
    TEP_res = pack_variables(clustering,n_clusters,
        flipped_data_PN,flipped_data_GEN ,flipped_data_SC, flipped_data_curt,flipped_data_curt_per, flipped_data_lines,
        flipped_data_conv, flipped_data_price
    )
    grid.TEP_res=TEP_res
    
      
    grid.Line_AC_calc()
    grid.Line_DC_calc()
    
    return TEP_res

def export_TEP_results_to_excel(grid,export):
    [clustering,n_clusters,flipped_data_PN,flipped_data_GEN ,flipped_data_SC, flipped_data_curt,flipped_data_curt_per, flipped_data_lines,
        flipped_data_conv, flipped_data_price] = grid.TEP_res
           # Define the column names for the DataFrame
    columns = ["Element", "Type", "Initial", "Optimized N", "Optimized Power Rating [MW]", "Expansion Cost [k€]","Unit cost [€/MVA]","Life time [years]", "phi [€/MVA-h]"]
    
    # Create an empty list to hold the data
    data = []
    
    tot = 0
    
    # Loop through DC lines and add data to the list
    for l in grid.lines_DC:
        if l.np_line_opf:
            element = l.name
            ini = l.np_line_i
            opt = l.np_line
            pr = opt * l.MW_rating
            cost = ((opt - ini) * l.MW_rating * l.Length_km * l.phi) * l.life_time * 8760 / 1000
            
            if l.cost_perMWkm is not None:
                unit_cost= l.Length_km*l.cost_perMWkm
            elif l.base_cost is not None:
                unit_cost= l.base_cost /l.MW_rating
            else:
                unit_cost = np.nan
                
            phi = l.phi
            
            tot += cost
            data.append([element, "DC Line", ini, np.round(opt, decimals=2), np.round(pr, decimals=0).astype(int), np.round(cost, decimals=2),unit_cost,l.life_time,phi])
    
    # Loop through ACDC converters and add data to the list
    for cn in grid.Converters_ACDC:
        if cn.NUmConvP_opf:
            element = cn.name
            ini = cn.NumConvP_i
            opt = cn.NumConvP
            pr = opt * cn.MVA_max
            cost = ((opt - ini) * cn.MVA_max * cn.phi) * cn.life_time * 8760 / 1000
            tot += cost
            
            if cn.cost_perMVA is not None:
                unit_cost= cn.cost_perMVA
            elif cn.base_cost is not None:
                unit_cost= cn.base_cost /cn.MVA_max
            else:
                unit_cost = np.nan
                
            phi = cn.phi
            
            data.append([element, "ACDC Conv", ini, np.round(opt, decimals=2), np.round(pr, decimals=0).astype(int), np.round(cost, decimals=2),unit_cost,cn.life_time,phi])
    
    # Create a pandas DataFrame with the collected data
    df = pd.DataFrame(data, columns=columns)    
    
    
    
    

    data = {}

    # Loop through RenSourceZones
    for z in grid.RenSource_zones:
        # Extract the zone name
        zone_name = z.name
        # Access the time series data for the specific 'PGRi' from the zone's TS_dict
        if clustering:
            time_series_data = grid.Time_series[z.TS_dict['PRGi_available']].data_clustered[n_clusters]
        else:
            time_series_data = grid.Time_series[z.TS_dict['PRGi_available']].data
        
        # Append the zone name and corresponding data as a row in the data list
        data[zone_name]= time_series_data
    
    # Create a DataFrame named Availability_factors from the collected data
    Availability_factors = pd.DataFrame(data)

    data_L = {}

    # Loop through 
    for z in grid.Price_Zones:
        
        # Extract the zone name
        zone_name = z.name
        # Access the time series data for the specific 'PGRi' from the zone's TS_dict
        if z.TS_dict is None or z.TS_dict.get('Load') is None:
            continue
        if clustering:
            time_series_data = grid.Time_series[z.TS_dict['Load']].data_clustered[n_clusters]
        else:
            time_series_data = grid.Time_series[z.TS_dict['Load']].data
        
        # Append the zone name and corresponding data as a row in the data list
        
        data_L[zone_name]= time_series_data 
    
    # Create a DataFrame named Availability_factors from the collected data
    Load_factors = pd.DataFrame(data)

    flipped_AV=Availability_factors.T
    flipped_LF = Load_factors.T
    
    


    with pd.ExcelWriter(f'{export}.xlsx') as writer:
        df.to_excel(writer, sheet_name='TEP solution', index=True)
        flipped_data_SC.to_excel(writer, sheet_name='Social Cost k€', index=True)
        flipped_data_PN.to_excel(writer, sheet_name='Net price_zone power MW', index=True)
        flipped_data_price.to_excel(writer, sheet_name='Price_Zone Price  € per MWh', index=True)
        flipped_data_GEN.to_excel(writer, sheet_name='Power Generation MW', index=True)
        flipped_data_curt.to_excel(writer, sheet_name='Curtailment MW', index=True)
        flipped_data_curt_per.to_excel(writer, sheet_name='Curtailment %', index=True)
        flipped_data_lines.to_excel(writer, sheet_name='Line loading %', index=True)
        flipped_data_conv.to_excel(writer, sheet_name='Converter loading %', index=True)
        flipped_AV.to_excel(writer, sheet_name='Availability Factors pu', index=True)
        flipped_LF.to_excel(writer, sheet_name='Load Factors  pu', index=True)
