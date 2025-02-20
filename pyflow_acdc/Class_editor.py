"""
Created on Fri Dec 15 15:24:42 2023

@author: BernardoCastro
"""

import pandas as pd
import numpy as np
import sys
import copy
import pandas as pd
from .Classes import*
from .Results import*

from shapely.geometry import Polygon, Point
from shapely.wkt import loads

import os
import importlib.util
from pathlib import Path    
    
"""
"""

__all__ = [
    # Add Grid Elements
    'add_AC_node',
    'add_DC_node',
    'add_line_AC',
    'add_line_DC',
    'add_ACDC_converter',
    'add_gen',
    'add_extGrid',
    'add_RenSource',
    'add_generators_fromcsv',
    
    # Add Zones
    'add_RenSource_zone',
    'add_price_zone',
    'add_MTDC_price_zone',
    'add_offshore_price_zone',
    
    # Add Time Series
    'add_TimeSeries',
    
    # Line Modifications
    'change_line_AC_to_expandable',
    'change_line_AC_to_tap_transformer',
    
    # Zone Assignments
    'assign_RenToZone',
    'assign_nodeToPrice_Zone',
    'assign_ConvToPrice_Zone',
    
    # Parameter Calculations
    'Cable_parameters',
    'Converter_parameters',
    
    # Utility Functions
    'pol2cart',
    'cart2pol',
    'pol2cartz',
    'cartz2pol',
]

def pol2cart(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return x, y


def pol2cartz(r, theta):
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = x+1j*y
    return z


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return rho, theta


def cartz2pol(z):
    r = np.abs(z)
    theta = np.angle(z)
    return r, theta



def Converter_parameters(S_base, kV_base, T_R_Ohm, T_X_mH, PR_R_Ohm, PR_X_mH, Filter_uF, f=50):

    Z_base = kV_base**2/S_base  # kv^2/MVA
    Y_base = 1/Z_base

    F = Filter_uF*10**(-6)
    PR_X_H = PR_X_mH/1000
    T_X_H = T_X_mH/1000

    B = f*F*np.pi
    T_X = f*T_X_H*np.pi
    PR_X = f*PR_X_H*np.pi

    T_R_pu = T_R_Ohm/Z_base
    T_X_pu = T_X/Z_base
    PR_R_pu = PR_R_Ohm/Z_base
    PR_X_pu = PR_X/Z_base
    Filter_pu = B/Y_base

    return [T_R_pu, T_X_pu, PR_R_pu, PR_X_pu, Filter_pu]


def Cable_parameters(S_base, R, L_mH, C_uF, G_uS, A_rating, kV_base, km, N_cables=1, f=50):

    Z_base = kV_base**2/S_base  # kv^2/MVA
    Y_base = 1/Z_base

    if L_mH == 0:
        MVA_rating = N_cables*A_rating*kV_base/(1000)
    else:
        MVA_rating = N_cables*A_rating*kV_base*np.sqrt(3)/(1000)

    C = C_uF*(10**(-6))
    L = L_mH/1000
    G = G_uS*(10**(-6))

    R_AC = R*km

    B = 2*f*C*np.pi*km
    X = 2*f*L*np.pi*km

    Z = R_AC+X*1j
    Y = G+B*1j

    # Zc=np.sqrt(Z/Y)
    # theta_Z=np.sqrt(Z*Y)

    Z_pi = Z
    Y_pi = Y

    # Z_pi=Zc*np.sinh(theta_Z)
    # Y_pi = 2*np.tanh(theta_Z/2)/Zc

    R_1 = np.real(Z_pi)
    X_1 = np.imag(Z_pi)
    G_1 = np.real(Y_pi)
    B_1 = np.imag(Y_pi)

    Req = R_1/N_cables
    Xeq = X_1/N_cables
    Geq = G_1*N_cables
    Beq = B_1*N_cables

    Rpu = Req/Z_base
    Xpu = Xeq/Z_base
    Gpu = Geq/Y_base
    Bpu = Beq/Y_base

    return [Rpu, Xpu, Gpu, Bpu, MVA_rating]

"Add main components"

def add_AC_node(grid, kV_base,node_type='PQ',Voltage_0=1.01, theta_0=0.01, Power_Gained=0, Reactive_Gained=0, Power_load=0, Reactive_load=0, name=None, Umin=0.9, Umax=1.1,Gs= 0,Bs=0,x_coord=None,y_coord=None,geometry=None):
    node = Node_AC( node_type, Voltage_0, theta_0,kV_base, Power_Gained, Reactive_Gained, Power_load, Reactive_load, name, Umin, Umax,Gs,Bs,x_coord,y_coord)
    if geometry is not None:
       if isinstance(geometry, str): 
            geometry = loads(geometry)  
       node.geometry = geometry
       node.x_coord = geometry.x
       node.y_coord = geometry.y
    
    grid.nodes_AC.append(node)
    
    return node

def add_DC_node(grid,kV_base,node_type='P', Voltage_0=1.01, Power_Gained=0, Power_load=0, name=None,Umin=0.95, Umax=1.05,x_coord=None,y_coord=None,geometry=None):
    node = Node_DC(node_type, Voltage_0, Power_Gained, Power_load,kV_base , name,Umin, Umax,x_coord,y_coord)
    grid.nodes_DC.append(node)
    if geometry is not None:
       if isinstance(geometry, str): 
            geometry = loads(geometry)  
       node.geometry = geometry
       node.x_coord = geometry.x
       node.y_coord = geometry.y
       
       
    return node
    
def add_line_AC(grid, fromNode, toNode,MVA_rating=None, r=0, x=0, b=0, g=0,R_Ohm_km=None,L_mH_km=None, C_uF_km=0, G_uS_km=0, A_rating=None ,m=1, shift=0, name=None,tap_changer=False,Expandable=False,N_cables=1,Length_km=1,geometry=None,data_in='pu',Cable_type:str ='Custom'):
    kV_base=toNode.kV_base
    if L_mH_km is not None:
        data_in = 'Real'
    if data_in == 'Ohm':
        Z_base = kV_base**2/grid.S_base
        
        Resistance_pu = r / Z_base if r!=0 else 0.00001
        Reactance_pu  = x  / Z_base if x!=0  else 0.00001
        Conductance_pu = b*Z_base
        Susceptance_pu = g*Z_base
    elif data_in== 'Real': 
       [Resistance_pu, Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating] = Cable_parameters(grid.S_base, R_Ohm_km, L_mH_km, C_uF_km, G_uS_km, A_rating, kV_base, Length_km,N_cables=N_cables)
    else:
        Resistance_pu = r if r!=0 else 0.00001
        Reactance_pu  = x if x!=0  else 0.00001
        Conductance_pu = b
        Susceptance_pu = g
    
    
    if tap_changer:
        line = TF_Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating, kV_base,m, shift, name)
        grid.lines_AC_tf.append(line)
        grid.Update_Graph_AC()
    elif Expandable:
        line = Exp_Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating, kV_base,Length_km,m, shift, name)
        grid.lines_AC_exp.append(line)
        grid.Update_Graph_AC()
        
    else:    
        line = Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating,Length_km,m, shift,N_cables, name,S_base=grid.S_base,Cable_type=Cable_type)
        
        grid.lines_AC.append(line)
        grid.create_Ybus_AC()
        grid.Update_Graph_AC()
        
    if geometry is not None:
       if isinstance(geometry, str): 
            geometry = loads(geometry)  
       line.geometry = geometry
    
    return line

def change_line_AC_to_expandable(grid, line_name):
    for line_to_process in grid.lines_AC:
        if line_name == line_to_process.name:
            l  = line_to_process
            break
    if l is not None:    
            grid.lines_AC.remove(l)
            l.remove()
            line_vars=l.get_relevant_attributes()
            expandable_line = Exp_Line_AC(**line_vars)
            grid.lines_AC_exp.append(expandable_line)
            grid.Update_Graph_AC()
            

    # Reassign line numbers to ensure continuity in grid.lines_AC
    for i, line in enumerate(grid.lines_AC):
        line.lineNumber = i 
    grid.create_Ybus_AC()
    for i, line in enumerate(grid.lines_AC_exp):
        line.lineNumber = i 
    s=1
        
def change_line_AC_to_tap_transformer(grid, line_name):
    l = None
    for line_to_process in grid.lines_AC:
        if line_name == line_to_process.name:
            l  = line_to_process
            break
    if l is not None:    
            grid.lines_AC.remove(l)
            l.remove()
            line_vars=l.get_relevant_attributes()
            trafo = TF_Line_AC(**line_vars)
            grid.lines_AC_tf.append(trafo)
    else:
        print(f"Line {line_name} not found.")
        return
    # Reassign line numbers to ensure continuity in grid.lines_AC
    for i, line in enumerate(grid.lines_AC):
        line.lineNumber = i 
    grid.create_Ybus_AC()
    s=1    

def add_line_DC(grid, fromNode, toNode, Resistance_pu=0.001, MW_rating=9999,Length_km=1, polarity='m', name=None,geometry=None,Cable_type:str ='Custom'):
    if isinstance(polarity, int):
        if polarity == 1:
            polarity = 'm'
        elif polarity == 2:
            polarity = 'b'
        else:
            print(f"Invalid polarity value: {polarity}")
            return
    line = Line_DC(fromNode, toNode, Resistance_pu, MW_rating,Length_km, polarity, name,Cable_type=Cable_type)
    grid.lines_DC.append(line)
    grid.create_Ybus_DC()
    if geometry is not None:
       if isinstance(geometry, str): 
            geometry = loads(geometry)  
       line.geometry = geometry
    grid.create_Ybus_DC()
    grid.Update_Graph_DC()
    return line

def add_ACDC_converter(grid,AC_node , DC_node , AC_type='PV', DC_type=None, P_AC_MW=0, Q_AC_MVA=0, P_DC_MW=0, Transformer_resistance=0, Transformer_reactance=0, Phase_Reactor_R=0, Phase_Reactor_X=0, Filter=0, Droop=0, kV_base=None, MVA_max= None,nConvP=1,polarity =1 ,lossa=1.103,lossb= 0.887,losscrect=2.885,losscinv=4.371,Ucmin= 0.85, Ucmax= 1.2, name=None,geometry=None):
    if MVA_max is None:
        MVA_max= grid.S_base*10
    if kV_base is None:
        kV_base = AC_node.kV_base
    if DC_type is None:
        DC_type = DC_node.type
        
    P_DC = P_DC_MW/grid.S_base
    P_AC = P_AC_MW/grid.S_base
    Q_AC = Q_AC_MVA/grid.S_base
    # if Filter !=0 and Phase_Reactor_R==0 and  Phase_Reactor_X!=0:
    #     print(f'Please fill out phase reactor values, converter {name} not added')
    #     return
    conv = AC_DC_converter(AC_type, DC_type, AC_node, DC_node, P_AC, Q_AC, P_DC, Transformer_resistance, Transformer_reactance, Phase_Reactor_R, Phase_Reactor_X, Filter, Droop, kV_base, MVA_max,nConvP,polarity ,lossa,lossb,losscrect,losscinv,Ucmin, Ucmax, name)
    if geometry is not None:
        if isinstance(geometry, str): 
             geometry = loads(geometry)  
        conv.geometry = geometry    
   
    conv.basekA  = grid.S_base/(np.sqrt(3)*conv.AC_kV_base)
    conv.a_conv  = conv.a_conv_og/grid.S_base
    conv.b_conv  = conv.b_conv_og*conv.basekA/grid.S_base
    conv.c_inver = conv.c_inver_og*conv.basekA**2/grid.S_base
    conv.c_rect  = conv.c_rect_og*conv.basekA**2/grid.S_base     

    grid.Converters_ACDC.append(conv)
    return conv

"Zones"


def add_RenSource_zone(Grid,name):
        
    RSZ = Ren_source_zone(name)
    Grid.RenSource_zones.append(RSZ)
    Grid.RenSource_zones_dic[name]=RSZ.ren_source_num
    
    return RSZ


def add_price_zone(Grid,name,price,import_pu_L=1,export_pu_G=1,a=0,b=1,c=0,import_expand_pu=0):

    if b==1:
        b= price
    
    M = Price_Zone(price,import_pu_L,export_pu_G,a,b,c,import_expand_pu,name)
    Grid.Price_Zones.append(M)
    Grid.Price_Zones_dic[name]=M.price_zone_num
    
    return M

def add_MTDC_price_zone(Grid, name,  linked_price_zones=None,pricing_strategy='avg'):
    # Initialize the MTDC price_zone and link it to the given price_zones
    mtdc_price_zone = MTDCPrice_Zone(name=name, linked_price_zones=linked_price_zones, pricing_strategy=pricing_strategy)
    Grid.Price_Zones.append(mtdc_price_zone)
    
    return mtdc_price_zone


def add_offshore_price_zone(Grid,main_price_zone,name):
    
    oprice_zone = OffshorePrice_Zone(name=name, price=main_price_zone.price, main_price_zone=main_price_zone)
    Grid.Price_Zones.append(oprice_zone)
    
    return oprice_zone

"Components for optimal power flow"

def add_generators_fromcsv(Grid,Gen_csv):
    if isinstance(Gen_csv, pd.DataFrame):
        Gen_data = Gen_csv
    else:
        Gen_data = pd.read_csv(Gen_csv)
   
    Gen_data = Gen_data.set_index('Gen')
    
    
    for index, row in Gen_data.iterrows():
        var_name = Gen_data.at[index, 'Gen_name'] if 'Gen_name' in Gen_data.columns else index
        node_name = str(Gen_data.at[index, 'Node'])
        
        MWmax = Gen_data.at[index, 'MWmax'] if 'MWmax' in Gen_data.columns else None
        MWmin = Gen_data.at[index, 'MWmin'] if 'MWmin' in Gen_data.columns else 0
        MVArmin = Gen_data.at[index, 'MVArmin'] if 'MVArmin' in Gen_data.columns else 0
        MVArmax = Gen_data.at[index, 'MVArmax'] if 'MVArmax' in Gen_data.columns else 99999
        
        PsetMW = Gen_data.at[index, 'PsetMW']  if 'PsetMW'  in Gen_data.columns else 0
        QsetMVA= Gen_data.at[index, 'QsetMVA'] if 'QsetMVA' in Gen_data.columns else 0
        lf = Gen_data.at[index, 'Linear factor']    if 'Linear factor' in Gen_data.columns else 0
        qf = Gen_data.at[index, 'Quadratic factor'] if 'Quadratic factor' in Gen_data.columns else 0
        geo  = Gen_data.at[index, 'geometry'] if 'geometry' in Gen_data.columns else None
        price_zone_link = False
        
        fuel_type = Gen_data.at[index, 'Fueltype']    if 'Fueltype' in Gen_data.columns else 'Other'
        if fuel_type.lower() in ["wind", "solar"]:
            add_RenSource(Grid,node_name, MWmax,ren_source_name=var_name ,geometry=geo,ren_type=fuel_type)
        else:
            add_gen(Grid, node_name,var_name, price_zone_link,lf,qf,MWmax,MWmin,MVArmin,MVArmax,PsetMW,QsetMVA,fuel_type=fuel_type,geometry=geo)  
        
def add_gen(Grid, node_name,gen_name=None, price_zone_link=False,lf=0,qf=0,MWmax=99999,MWmin=0,MVArmin=None,MVArmax=None,PsetMW=0,QsetMVA=0,Smax=None,fuel_type='Other',geometry= None):
    
    if MVArmin is None:
        MVArmin=-MWmax
    if MVArmax is None:
        MVArmax=MWmax
    if Smax is not None:
        Smax/=Grid.S_base
    Max_pow_gen=MWmax/Grid.S_base
 
    Max_pow_genR=MVArmax/Grid.S_base
    Min_pow_genR=MVArmin/Grid.S_base
    Min_pow_gen=MWmin/Grid.S_base
    Pset=PsetMW/Grid.S_base
    Qset=QsetMVA/Grid.S_base
    found=False    
    for node in Grid.nodes_AC:
   
        if node_name == node.name:
             gen = Gen_AC(gen_name, node,Max_pow_gen,Min_pow_gen,Max_pow_genR,Min_pow_genR,qf,lf,Pset,Qset,Smax)
             node.PGi = 0
             node.QGi = 0
             if fuel_type not in [
             "Nuclear", "Hard Coal", "Hydro", "Oil", "Lignite", "Natural Gas",
             "Solid Biomass",  "Other", "Waste", "Biogas", "Geothermal"
             ]:
                 fuel_type = 'Other'
             gen.gen_type = fuel_type
             if geometry is not None:
                 if isinstance(geometry, str): 
                      geometry = loads(geometry)  
                 gen.geometry= geometry
             found = True
             break

    if not found:
            print('Node does not exist')
            sys.exit()
    gen.price_zone_link=price_zone_link
    
    if price_zone_link:
        
        gen.qf= 0
        gen.lf= node.price
    Grid.Generators.append(gen)
    
   
            
            
def add_extGrid(Grid, node_name, gen_name=None,price_zone_link=False,lf=0,qf=0,MVAmax=99999,MVArmin=None,MVArmax=None,Allow_sell=True):
    
    
    if MVArmin is None:
        MVArmin=-MVAmax
    if MVArmax is None:
        MVArmax=MVAmax
    
    Max_pow_gen=MVAmax/Grid.S_base
 
    Max_pow_genR=MVArmax/Grid.S_base
    Min_pow_genR=MVArmin/Grid.S_base
    if Allow_sell:
        Min_pow_gen=-MVAmax/Grid.S_base
    else:
        Min_pow_gen=0
    found=False 
    for node in Grid.nodes_AC:
        if node_name == node.name:
             gen = Gen_AC(gen_name, node,Max_pow_gen,Min_pow_gen,Max_pow_genR,Min_pow_genR,qf,lf)
             node.PGi = 0
             node.QGi = 0
             found=True
             break
    if not found:
        print('Node {node_name} does not exist')
        sys.exit()
    gen.price_zone_link=price_zone_link
    if price_zone_link:
        gen.qf= 0
        gen.lf= node.price
    Grid.Generators.append(gen)

def add_RenSource(Grid,node_name, base,ren_source_name=None , available=1,zone=None,price_zone=None, Offshore=False,MTDC=None,geometry= None,ren_type='Wind'):
    if ren_source_name is None:
        ren_source_name= node_name
    found=False 
    for node in Grid.nodes_AC:
        if node_name == node.name:
            rensource= Ren_Source(ren_source_name,node,base/Grid.S_base)    
            rensource.PRGi_available=available
            rensource.connected= 'AC'
            ACDC='AC'
            rensource.rs_type= ren_type
            if geometry is not None:
                if isinstance(geometry, str): 
                     geometry = loads(geometry)  
                rensource.geometry= geometry
            Grid.rs2node['AC'][rensource.rsNumber]=node.nodeNumber
            found = True
            break
    for node in Grid.nodes_DC:
        if node_name == node.name:
            rensource= Ren_Source(ren_source_name,node,base/Grid.S_base)    
            rensource.PGi_available=available
            rensource.connected= 'DC'
            ACDC='DC'
            Grid.rs2node['DC'][rensource.rsNumber]=node.nodeNumber
            found = True
            break    

    if not found:
           print(f'Node {node_name} does not exist')
           sys.exit()
   
    Grid.RenSources.append(rensource)
    
    
    if zone is not None:
        rensource.zone=zone
        assign_RenToZone(Grid,ren_source_name,zone)
    
    if price_zone is not None:
        rensource.price_zone=price_zone
        if MTDC is not None:
            rensource.MTDC=MTDC
            main_price_zone = next((M for M in Grid.Price_Zones if price_zone == M.name), None)
            if main_price_zone is not None:
                # Find or create the MTDC price_zone
                MTDC_price_zone = next((mdc for mdc in Grid.Price_Zones if MTDC == mdc.name), None)

                if MTDC_price_zone is None:
                    # Create the offshore price_zone using the OffshorePrice_Zone class
                    MTDC_price_zone= add_MTDC_price_zone(Grid,MTDC)
            
            MTDC_price_zone.add_linked_price_zone(main_price_zone)
            main_price_zone.ImportExpand += base / Grid.S_base
            assign_nodeToPrice_Zone(Grid, node_name,ACDC, MTDC)
            # Additional logic for MTDC can be placed here
        elif Offshore:
            rensource.Offshore=True
            # Create an offshore price_zone by appending 'o' to the main price_zone's name
            oprice_zone_name = f'o{price_zone}'

            # Find the main price_zone
            main_price_zone = next((M for M in Grid.Price_Zones if price_zone == M.name), None)
            
            if main_price_zone is not None:
                # Find or create the offshore price_zone
                oprice_zone = next((m for m in Grid.Price_Zones if m.name == oprice_zone_name), None)

                if oprice_zone is None:
                    # Create the offshore price_zone using the OffshorePrice_Zone class
                    oprice_zone= add_offshore_price_zone(Grid,main_price_zone,oprice_zone_name)

                # Assign the node to the offshore price_zone
                assign_nodeToPrice_Zone(Grid, node_name,ACDC, oprice_zone_name)
                # Link the offshore price_zone to the main price_zone
                main_price_zone.link_price_zone(oprice_zone)
                # Expand the import capacity in the main price_zone
                main_price_zone.ImportExpand += base / Grid.S_base
        else:
            # Assign the node to the main price_zone
            assign_nodeToPrice_Zone(Grid, node_name,ACDC, price_zone)



"Time series data "


def time_series_dict(grid, ts):
    typ = ts.type
    
    if typ == 'a_CG':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == 'b_CG':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == 'c_CG':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == 'PGL_min':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == 'PGL_max':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
                
    if typ == 'price':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct price_zone
        for node in grid.nodes_AC + grid.nodes_DC:
            if ts.element_name == node.name:
                node.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct node    
    
    elif typ == 'Load':
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct price_zone
        for node in grid.nodes_AC + grid.nodes_DC:
            if ts.element_name == node.name:
                node.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct node
                
    elif typ in ['WPP', 'OWPP', 'SF', 'REN']:
        for zone in grid.RenSource_zones:
            if ts.element_name == zone.name:
                zone.TS_dict['PRGi_available'] = ts.TS_num
                break  # Stop after assigning to the correct zone
        for rs in grid.RenSources:
            if ts.element_name == rs.name:
                rs.TS_dict['PRGi_available'] = ts.TS_num
                break  # Stop after assigning to the correct node


def add_TimeSeries(Grid, Time_Series_data,associated=None,TS_type=None,ignore=None):
    TS = Time_Series_data
    Time_series = {}
    # check if there are nan values in Time series and change to 0
    TS.fillna(0, inplace=True)
    
    for col in TS.columns:
        if associated is not None and TS_type is not None:
            element_name = associated
            element_type = TS_type
            data = TS.loc[0:, col].astype(float).to_numpy()  
            name = f'{associated}_{TS_type}'
            
        
        elif associated is not None: 
            element_name = associated
            element_type = col
            data = TS.loc[0:, col].astype(float).to_numpy()  
            name = f'{associated}_{col}'
        
        elif TS_type is not None:
            element_name = col
            element_type = TS_type
            data = TS.loc[0:, col].astype(float).to_numpy()   
            name = f'{col}_{TS_type}'
        
        else: 
            element_name = TS.at[0, col]
            element_type = TS.at[1, col]
            data = TS.loc[2:, col].astype(float).to_numpy()   
            name = col
        if ignore and ignore in name:
            continue
    
        
        Time_serie = TimeSeries(element_type, element_name, data,name)                  
        Grid.Time_series.append(Time_serie)
        Grid.Time_series_dic[name]=Time_serie.TS_num
        time_series_dict(Grid, Time_serie)
        
        
        
    Grid.Time_series_ran = False
    s = 1


def assign_RenToZone(Grid,ren_source_name,new_zone_name):
    new_zone = None
    old_zone = None
    ren_source_to_reassign = None
    
    for RenZone in Grid.RenSource_zones:
        if RenZone.name == new_zone_name:
            new_zone = RenZone
            break
    if new_zone is None:
        raise ValueError(f"Zone {new_zone_name} not found.")
    
    # Remove node from its old price_zone
    for RenZone in Grid.RenSource_zones:
        for ren_source in RenZone.RenSources:
            if ren_source.name == ren_source_name:
                old_zone = RenZone
                ren_source_to_reassign = ren_source
                break
        if old_zone:
            break
        
    if old_zone is not None:
        RenZone.ren_source = [ren_source for ren_source in old_zone.RenSources 
                               if ren_source.name != ren_source_name]
    
    # If the node was not found in any Renewable zone, check Grid.nodes_AC
    if ren_source_to_reassign is None:
        for ren_source in Grid.RenSources:
            if ren_source.name == ren_source_name:
                ren_source_to_reassign = ren_source
                break
            
    if ren_source_to_reassign is None:
        raise ValueError(f"Renewable source {ren_source_name} not found.")
    ren_source_to_reassign.PGRi_linked = True
    ren_source_to_reassign.Ren_source_zone = new_zone.name
    # Add node to the new price_zone
    if ren_source_to_reassign not in new_zone.RenSources:
        new_zone.RenSources.append(ren_source_to_reassign)
 
"Assigning components to zones"
    
def assign_nodeToPrice_Zone(Grid,node_name,ACDC, new_price_zone_name):
        """ Assign node to a new price_zone and remove it from its previous price_zone """
        new_price_zone = None
        old_price_zone = None
        node_to_reassign = None
        
        nodes_attr = 'nodes_AC' if ACDC == 'AC' else 'nodes_DC'
        
        # Find the new price_zone
        for price_zone in Grid.Price_Zones:
            if price_zone.name == new_price_zone_name:
                new_price_zone = price_zone
                break

        if new_price_zone is None:
            raise ValueError(f"Price_Zone {new_price_zone_name} not found.")
        
        # Remove node from its old price_zone
        for price_zone in Grid.Price_Zones:
            nodes = getattr(price_zone, nodes_attr)
            for node in nodes:
                if node.name == node_name:
                    old_price_zone = price_zone
                    node_to_reassign = node
                    break
            if old_price_zone:
                break
            
        if old_price_zone is not None:
            setattr(old_price_zone, nodes_attr, [node for node in getattr(old_price_zone, nodes_attr) if node.name != node_name])

        # If the node was not found in any price_zone, check Grid.nodes_AC
        if node_to_reassign is None:
            nodes = getattr(Grid, nodes_attr)
            for node in nodes:
                if node.name == node_name:
                    node_to_reassign = node
                    break
                
        if node_to_reassign is None:
            raise ValueError(f"Node {node_name} not found.")
        
        # Add node to the new price_zone
        new_price_zone_nodes = getattr(new_price_zone, nodes_attr)
        if node_to_reassign not in new_price_zone_nodes:
            new_price_zone_nodes.append(node_to_reassign)
            node_to_reassign.PZ=new_price_zone.name
            node_to_reassign.price=new_price_zone.price

def assign_ConvToPrice_Zone(Grid, conv_name, new_price_zone_name):
        """ Assign node to a new price_zone and remove it from its previous price_zone """
        new_price_zone = None
        old_price_zone = None
        conv_to_reassign = None
        
        # Find the new price_zone
        for price_zone in Grid.Price_Zones:
            if price_zone.name == new_price_zone_name:
                new_price_zone = price_zone
                break

        if new_price_zone is None:
            raise ValueError(f"Price_Zone {new_price_zone_name} not found.")
        
        # Remove node from its old price_zone
        for price_zone in Grid.Price_Zones:
            for conv in price_zone.ConvACDC:
                if conv.name == conv_name:
                    old_price_zone = price_zone
                    conv_to_reassign = conv
                    break
            if old_price_zone:
                break
            
        if old_price_zone is not None:
            old_price_zone.ConvACDC = [conv for conv in old_price_zone.ConvACDC if conv.name != conv_name]
        
        # If the node was not found in any price_zone, check Grid.nodes_AC
        if conv_to_reassign is None:
            for conv in Grid.Converters_ACDC:
                if conv.name == conv_name:
                    conv_to_reassign = conv
                    break
                
        if conv_to_reassign is None:
            raise ValueError(f"Converter {conv_name} not found.")
        
        # Add node to the new price_zone
        if conv_to_reassign not in new_price_zone.ConvACDC:
            new_price_zone.ConvACDC.append(conv_to_reassign)            





    

