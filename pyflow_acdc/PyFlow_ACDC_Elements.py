"""
Created on Fri Dec 15 15:24:42 2023

@author: BernardoCastro
"""

from scipy.io import loadmat
import pandas as pd
import numpy as np
import sys

import pandas as pd
from .PyFlow_ACDC_Class import*
from .PyFlow_ACDC_Results import*


import os
import importlib.util
from pathlib import Path    
    
"""
"""





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

def reset_all_class():
    Node_AC.reset_class()
    Node_DC.reset_class()
    Line_AC.reset_class()
    Line_DC.reset_class()
    AC_DC_converter.reset_class()
    DC_DC_converter.reset_class()
    TimeSeries_AC.reset_class()
    
def Create_grid_from_data(S_base, AC_node_data=None, AC_line_data=None, DC_node_data=None, DC_line_data=None, Converter_data=None, DCDC_conv=None, data_in_pu=True):
    
    reset_all_class()
    
    if data_in_pu == True:
        [G, res] = Create_grid_from_data_pu(
            S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_data, DCDC_conv)
    else:
        [G, res] = Create_grid_from_data_calc(
            S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_data, DCDC_conv)

    return [G, res]


def Create_grid_from_data_calc(S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_data, DCDC_conv):
        
   

    if AC_node_data is None:
        AC_nodes_list = None
        AC_lines_list = None
    else:
        "AC nodes data sorting"
        AC_node_data = AC_node_data.set_index('Node_id')
        AC_nodes = {}
        for index, row in AC_node_data.iterrows():
            
            var_name = index
            element_type = AC_node_data.at[index, 'type']
            kV_base = AC_node_data.at[index, 'kV_base']
            
            Voltage_0 = AC_node_data.at[index, 'Voltage_0']             if 'Voltage_0'       in AC_node_data.columns else 1.01
            theta_0 = AC_node_data.at[index, 'theta_0']                 if 'theta_0'         in AC_node_data.columns else 0.01
            Power_Gained    = AC_node_data.at[index, 'Power_Gained']    if 'Power_Gained'    in AC_node_data.columns else 0
            Reactive_Gained = AC_node_data.at[index, 'Reactive_Gained'] if 'Reactive_Gained' in AC_node_data.columns else 0
            Power_load      = AC_node_data.at[index, 'Power_load']      if 'Power_load'      in AC_node_data.columns else 0
            Reactive_load   = AC_node_data.at[index, 'Reactive_load']   if 'Reactive_load'   in AC_node_data.columns else 0
            Umin            = AC_node_data.at[index, 'Umin']            if 'Umin'            in AC_node_data.columns else 0.9
            Umax            = AC_node_data.at[index, 'Umax']            if 'Umax'            in AC_node_data.columns else 1.1
            x_coord         = AC_node_data.at[index, 'x_coord']         if 'x_coord'         in AC_node_data.columns else None
            y_coord         = AC_node_data.at[index, 'y_coord']         if 'y_coord'         in AC_node_data.columns else None
            Bs              = AC_node_data.at[index, 'Bs']              if 'Bs'              in AC_node_data.columns else 0
            Gs              = AC_node_data.at[index, 'Gs']              if 'Gs'              in AC_node_data.columns else 0
            
            Bs/=S_base
            Gs/=S_base
            Power_Gained    /=S_base
            Reactive_Gained /=S_base
            Power_load      /=S_base
            Reactive_load   /=S_base

            AC_nodes[var_name] = Node_AC(element_type, Voltage_0, theta_0,kV_base, Power_Gained,
                                         Reactive_Gained, Power_load, Reactive_load, name=str(var_name),Umin=Umin,Umax=Umax,Gs=Gs,Bs=Bs,x_coord=x_coord,y_coord=y_coord)
        AC_nodes_list = list(AC_nodes.values())

        AC_line_data = AC_line_data.set_index('Line_id')
        AC_lines = {}
        for index, row in AC_line_data.iterrows():
            var_name = index
            
            
            fromNode = AC_line_data.at[index, 'fromNode']
            toNode = AC_line_data.at[index, 'toNode']
            R = AC_line_data.at[index, 'R_Ohm_km']
            L_mH = AC_line_data.at[index, 'L_mH_km']       
            C_uF = AC_line_data.at[index, 'C_uF_km']       if 'C_uF_km'    in AC_line_data.columns else 0
            G_uS = AC_line_data.at[index, 'G_uS_km']       if 'G_uS_km'    in AC_line_data.columns else 0
            A_rating = AC_line_data.at[index, 'A_rating']
            # kV_base = AC_line_data.at[index, 'kV_base']
            kV_base= AC_nodes[toNode].kV_base
            km = AC_line_data.at[index, 'Length_km']
            N_cables = AC_line_data.at[index, 'N_cables']  if 'N_cables'   in AC_line_data.columns else 1
            m    = AC_line_data.at[index, 'm']             if 'm'            in AC_line_data.columns else 1
            shift= AC_line_data.at[index, 'shift']         if 'shift'        in AC_line_data.columns else 0
                
            [Resistance, Reactance, Conductance, Susceptance, MVA_rating] = Cable_parameters(S_base, R, L_mH, C_uF, G_uS, A_rating, kV_base, km,N_cables=N_cables)
            
            
            
            AC_lines[var_name] = Line_AC(AC_nodes[fromNode], AC_nodes[toNode], Resistance,
                                         Reactance, Conductance, Susceptance, MVA_rating, kV_base,m,shift,name=str(var_name))
        AC_lines_list = list(AC_lines.values())

    if DC_node_data is None:

        DC_nodes_list = None
        DC_lines_list = None

    else:
        DC_node_data = DC_node_data.set_index('Node_id')

        "DC nodes data sorting"
        DC_nodes = {}
        for index, row in DC_node_data.iterrows():

            var_name = index
            node_type = DC_node_data.at[index, 'type']
            
            Voltage_0     = DC_node_data.at[index, 'Voltage_0']     if 'Power_Gained'  in DC_node_data.columns else 1.01
            Power_Gained  = DC_node_data.at[index, 'Power_Gained']  if 'Power_Gained'  in DC_node_data.columns else 0
            Power_load    = DC_node_data.at[index, 'Power_load']    if 'Power_load'    in DC_node_data.columns else 0
            kV_base       = DC_node_data.at[index, 'kV_base']  
            Umin          = DC_node_data.at[index, 'Umin']          if 'Umin'          in DC_node_data.columns else 0.95
            Umax          = DC_node_data.at[index, 'Umax']          if 'Umax'          in DC_node_data.columns else 1.05
            x_coord       = DC_node_data.at[index, 'x_coord']       if 'x_coord'       in DC_node_data.columns else None
            y_coord       = DC_node_data.at[index, 'y_coord']       if 'y_coord'       in DC_node_data.columns else None


            
            Power_Gained = Power_Gained/S_base
            Power_load = Power_load/S_base
             

            DC_nodes[var_name] = Node_DC(node_type, Voltage_0, Power_Gained, Power_load, kV_base ,name=str(var_name),Umin=Umin,Umax=Umax,x_coord=x_coord,y_coord=y_coord)
        DC_nodes_list = list(DC_nodes.values())

        DC_line_data = DC_line_data.set_index('Line_id')
        DC_lines = {}
        for index, row in DC_line_data.iterrows():
            var_name = index

            fromNode = DC_line_data.at[index, 'fromNode']
            toNode = DC_line_data.at[index, 'toNode']
            R = DC_line_data.at[index, 'R_Ohm_km']
            A_rating = DC_line_data.at[index, 'A_rating']
            kV_base = DC_nodes[toNode].kV_base 
            pol  = DC_line_data.at[index, 'Mono_Bi_polar']  if 'Mono_Bi_polar' in DC_line_data.columns else 'm'
            km = DC_line_data.at[index, 'Length_km']        if 'Length_km' in DC_line_data.columns else 1
            N_cables = DC_line_data.at[index, 'N_cables']   if 'N_cables' in DC_line_data.columns else 1
            L_mH = 0
            C_uF = 0
            G_uS = 0
            [Resistance, _, _, _, MW_rating] = Cable_parameters(S_base, R, L_mH, C_uF, G_uS, A_rating, kV_base, km, N_cables=N_cables)
            
            if pol == 'm':
                pol_val = 1
            elif pol == 'b' or pol == 'sm':
                pol_val = 2
            else:
                pol_val = 1
            MW_rating=MW_rating*pol_val
            
            DC_lines[var_name] = Line_DC(DC_nodes[fromNode], DC_nodes[toNode], Resistance, MW_rating, kV_base, km, pol,name=str(var_name))
        DC_lines_list = list(DC_lines.values())

    if Converter_data is None:
        Convertor_list = None
    else:
        Converter_data = Converter_data.set_index('Conv_id')
        "Convertor data sorting"
        Converters = {}
        for index, row in Converter_data.iterrows():
            var_name         = index
            AC_type          = Converter_data.at[index, 'AC_type']       if 'AC_type'        in Converter_data.columns else 'PV'
            DC_type          = Converter_data.at[index, 'DC_type']       
            AC_node          = Converter_data.at[index, 'AC_node']      
            DC_node          = Converter_data.at[index, 'DC_node']       
            P_AC             = Converter_data.at[index, 'P_MW_AC']       if 'P_MW_AC'        in Converter_data.columns else 0
            Q_AC             = Converter_data.at[index, 'Q_AC']          if 'Q_AC'           in Converter_data.columns else 0
            P_DC             = Converter_data.at[index, 'P_MW_DC']       if 'P_MW_DC'        in Converter_data.columns else 0
            Transformer_R    = Converter_data.at[index, 'T_R_Ohm']       if 'T_R_Ohm'        in Converter_data.columns else 0
            Transformer_X    = Converter_data.at[index, 'T_X_mH']        if 'T_X_mH'         in Converter_data.columns else 0
            Phase_Reactor_R  = Converter_data.at[index, 'PR_R_Ohm']      if 'PR_R_Ohm'       in Converter_data.columns else 0
            Phase_Reactor_X  = Converter_data.at[index, 'PR_X_mH']       if 'PR_X_mH'        in Converter_data.columns else 0
            Filter           = Converter_data.at[index, 'Filter_uF']     if 'Filter_uF'      in Converter_data.columns else 0
            Droop            = Converter_data.at[index, 'Droop']         if 'Droop'          in Converter_data.columns else 0
            kV_base          = Converter_data.at[index, 'AC_kV_base']    
            MVA_rating       = Converter_data.at[index, 'MVA_rating']    if 'MVA_rating'     in Converter_data.columns else S_base*1.05
            Ucmin           = Converter_data.at[index, 'Ucmin']          if 'Ucmin'          in Converter_data.columns else 0.85
            Ucmax           = Converter_data.at[index, 'Ucmax']          if 'Ucmax'          in Converter_data.columns else 1.2
            n               = Converter_data.at[index, 'Nconverter']     if 'Nconverter'     in Converter_data.columns else 1
            pol             = Converter_data.at[index, 'pol']            if 'pol'     in Converter_data.columns else 1

            [T_R_pu, T_X_pu, PR_R_pu, PR_X_pu, Filter_pu] = Converter_parameters(S_base, kV_base, Transformer_R, Transformer_X, Phase_Reactor_R, Phase_Reactor_X, Filter)


            MVA_max = MVA_rating
            P_AC = P_AC/S_base
            P_DC = P_DC/S_base
            
           
            Converters[var_name] = AC_DC_converter(AC_type, DC_type, AC_nodes[AC_node], DC_nodes[DC_node], P_AC, Q_AC,
                                                   P_DC, T_R_pu, T_X_pu, PR_R_pu, PR_X_pu, Filter_pu, Droop, kV_base, MVA_max=MVA_max,nConvP=n,polarity=pol,Ucmin=Ucmin,Ucmax=Ucmax ,name=str(var_name))
        Convertor_list = list(Converters.values())

    if DCDC_conv is None:
        Convertor_DC_list = None
    else:
        DCDC_conv = DCDC_conv.set_index('Conv_id')
        "Convertor data sorting"
        Converters_DC = {}
        for index, row in DCDC_conv.iterrows():
            var_name = index
            element_type = DCDC_conv.at[index, 'type']

            fromNode = DCDC_conv.at[index, 'fromNode']
            toNode = DCDC_conv.at[index, 'toNode']

            PowerTo = DCDC_conv.at[index, 'P_MW']
            R_Ohm = DCDC_conv.at[index, 'R_Ohm']
            kV_base = DCDC_conv.at[index, 'kV_nodefromBase']

            Z_base = kV_base**2/S_base

            R = R_Ohm/Z_base

            PowerTo = PowerTo/S_base

            Converters_DC[var_name] = DC_DC_converter(
                element_type, DC_nodes[fromNode], DC_nodes[toNode], PowerTo, R, name=str(var_name))
        Convertor_DC_list = list(Converters_DC.values())

    G = Grid(S_base, AC_nodes_list, AC_lines_list, nodes_DC=DC_nodes_list,
             lines_DC=DC_lines_list, Converters=Convertor_list, conv_DC=Convertor_DC_list)
    res = Results(G, decimals=3)

    return [G, res]


def Create_grid_from_data_pu(S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_data, DCDC_conv):

    if AC_node_data is None:
        AC_nodes_list = None
        AC_lines_list = None
    else:
        "AC nodes data sorting"
        AC_node_data = AC_node_data.set_index('Node_id')
        AC_nodes = {}
        for index, row in AC_node_data.iterrows():
            var_name = index
            element_type = AC_node_data.at[index, 'type']

            kV_base       = AC_node_data.at[index, 'kV_base']
            Voltage_0 = AC_node_data.at[index, 'Voltage_0']             if 'Voltage_0'       in AC_node_data.columns else 1.01
            theta_0 = AC_node_data.at[index, 'theta_0']                 if 'theta_0'         in AC_node_data.columns else 0.01
            Power_Gained    = AC_node_data.at[index, 'Power_Gained']    if 'Power_Gained'    in AC_node_data.columns else 0
            Reactive_Gained = AC_node_data.at[index, 'Reactive_Gained'] if 'Reactive_Gained' in AC_node_data.columns else 0
            Power_load      = AC_node_data.at[index, 'Power_load']      if 'Power_load'      in AC_node_data.columns else 0
            Reactive_load   = AC_node_data.at[index, 'Reactive_load']   if 'Reactive_load'   in AC_node_data.columns else 0
            Umin            = AC_node_data.at[index, 'Umin']            if 'Umin'            in AC_node_data.columns else 0.9
            Umax            = AC_node_data.at[index, 'Umax']            if 'Umax'            in AC_node_data.columns else 1.1
            x_coord         = AC_node_data.at[index, 'x_coord']         if 'x_coord'         in AC_node_data.columns else None
            y_coord         = AC_node_data.at[index, 'y_coord']         if 'y_coord'         in AC_node_data.columns else None
            Bs              = AC_node_data.at[index, 'Bs']              if 'Bs'              in AC_node_data.columns else 0
            Gs              = AC_node_data.at[index, 'Gs']              if 'Gs'              in AC_node_data.columns else 0

            AC_nodes[var_name] = Node_AC(element_type, Voltage_0, theta_0,kV_base, Power_Gained,
                                         Reactive_Gained, Power_load, Reactive_load, name=str(var_name),Umin=Umin,Umax=Umax,Gs=Gs,Bs=Bs,x_coord=x_coord,y_coord=y_coord)
        AC_nodes_list = list(AC_nodes.values())

        AC_line_data = AC_line_data.set_index('Line_id')
        AC_lines = {}
        for index, row in AC_line_data.iterrows():
            var_name = index

            fromNode     = AC_line_data.at[index, 'fromNode']
            toNode       = AC_line_data.at[index, 'toNode']
            Resistance   = AC_line_data.at[index, 'Resistance']
            Reactance    = AC_line_data.at[index, 'Reactance']    
            Conductance  = AC_line_data.at[index, 'Conductance']  if 'Conductance'  in AC_line_data.columns else 0
            Susceptance  = AC_line_data.at[index, 'Susceptance']  if 'Susceptance'  in AC_line_data.columns else 0
            MVA_rating   = AC_line_data.at[index, 'MVA_rating']   if 'MVA_rating'   in AC_line_data.columns else S_base*1.05
            kV_base      = AC_nodes[toNode].kV_base 
            m            = AC_line_data.at[index, 'm']            if 'm'            in AC_line_data.columns else 1
            shift        = AC_line_data.at[index, 'shift']        if 'shift'        in AC_line_data.columns else 0

            
            
            AC_lines[var_name] = Line_AC(AC_nodes[fromNode], AC_nodes[toNode], Resistance,
                                         Reactance, Conductance, Susceptance, MVA_rating, kV_base,m,shift ,name=str(var_name))
        AC_lines_list = list(AC_lines.values())

    if DC_node_data is None:

        DC_nodes_list = None
        DC_lines_list = None

    else:
        DC_node_data = DC_node_data.set_index('Node_id')

        "DC nodes data sorting"
        DC_nodes = {}
        for index, row in DC_node_data.iterrows():

            var_name = index
            node_type = DC_node_data.at[index, 'type']

            Voltage_0     = DC_node_data.at[index, 'Voltage_0']     if 'Voltage_0'     in DC_node_data.columns else 1.01
            Power_Gained  = DC_node_data.at[index, 'Power_Gained']  if 'Power_Gained'  in DC_node_data.columns else 0
            Power_load    = DC_node_data.at[index, 'Power_load']    if 'Power_load'    in DC_node_data.columns else 0
            kV_base       = DC_node_data.at[index, 'kV_base']  
            Umin          = DC_node_data.at[index, 'Umin']          if 'Umin'          in DC_node_data.columns else 0.95
            Umax          = DC_node_data.at[index, 'Umax']          if 'Umax'          in DC_node_data.columns else 1.05
            x_coord       = DC_node_data.at[index, 'x_coord']       if 'x_coord'       in DC_node_data.columns else None
            y_coord       = DC_node_data.at[index, 'y_coord']       if 'y_coord'       in DC_node_data.columns else None

                
            DC_nodes[var_name] = Node_DC(
                node_type, Voltage_0, Power_Gained, Power_load,kV_base , name=str(var_name),Umin=Umin,Umax=Umax,x_coord=x_coord,y_coord=y_coord)
        DC_nodes_list = list(DC_nodes.values())

        DC_line_data = DC_line_data.set_index('Line_id')
        DC_lines = {}
        for index, row in DC_line_data.iterrows():
            var_name = index

            fromNode      = DC_line_data.at[index, 'fromNode']
            toNode        = DC_line_data.at[index, 'toNode']
            Resistance    = DC_line_data.at[index, 'Resistance']
            MW_rating     = DC_line_data.at[index, 'MW_rating']      if 'MW_rating'     in DC_line_data.columns else S_base*1.05
            kV_base       = DC_nodes[toNode].kV_base 
            pol           = DC_line_data.at[index, 'Mono_Bi_polar']  if 'Mono_Bi_polar' in DC_line_data.columns else 'm'
            km            = DC_line_data.at[index, 'Length_km']        if 'Length_km' in DC_line_data.columns else 1
            N_cables      = DC_line_data.at[index, 'N_cables']   if 'N_cables' in DC_line_data.columns else 1
            
            DC_lines[var_name] = Line_DC(DC_nodes[fromNode], DC_nodes[toNode], Resistance, MW_rating, kV_base, km, pol,name=str(var_name))
           
        DC_lines_list = list(DC_lines.values())

    if Converter_data is None:
        Convertor_list = None
    else:
        Converter_data = Converter_data.set_index('Conv_id')
        "Convertor data sorting"
        Converters = {}
        for index, row in Converter_data.iterrows():
            var_name        = index
            AC_type         = Converter_data.at[index, 'AC_type']        if 'AC_type'        in Converter_data.columns else 'PV'
            DC_type         = Converter_data.at[index, 'DC_type']        
            AC_node         = Converter_data.at[index, 'AC_node']               
            DC_node         = Converter_data.at[index, 'DC_node']              
            P_AC            = Converter_data.at[index, 'P_AC']           if 'P_AC'           in Converter_data.columns else 0
            Q_AC            = Converter_data.at[index, 'Q_AC']           if 'Q_AC'           in Converter_data.columns else 0
            P_DC            = Converter_data.at[index, 'P_DC']           if 'P_DC'           in Converter_data.columns else 0
            Transformer_R   = Converter_data.at[index, 'T_R']            if 'T_R'            in Converter_data.columns else 0
            Transformer_X   = Converter_data.at[index, 'T_X']            if 'T_X'            in Converter_data.columns else 0
            Phase_Reactor_R = Converter_data.at[index, 'PR_R']           if 'PR_R'           in Converter_data.columns else 0
            Phase_Reactor_X = Converter_data.at[index, 'PR_X']           if 'PR_X'           in Converter_data.columns else 0   
            Filter          = Converter_data.at[index, 'Filter']         if 'Filter'         in Converter_data.columns else 0
            Droop           = Converter_data.at[index, 'Droop']          if 'Droop'          in Converter_data.columns else 0
            kV_base         = Converter_data.at[index, 'AC_kV_base']    
            MVA_max         = Converter_data.at[index, 'MVA_rating']     if 'MVA_rating'     in Converter_data.columns else S_base*1.05
            Ucmin           = Converter_data.at[index, 'Ucmin']          if 'Ucmin'          in Converter_data.columns else 0.85
            Ucmax           = Converter_data.at[index, 'Ucmax']          if 'Ucmax'          in Converter_data.columns else 1.2
            n               = Converter_data.at[index, 'Nconverter']     if 'Nconverter'     in Converter_data.columns else 1
            pol             = Converter_data.at[index, 'pol']            if 'pol'            in Converter_data.columns else 1

            Converters[var_name] = AC_DC_converter(AC_type, DC_type, AC_nodes[AC_node], DC_nodes[DC_node], P_AC, Q_AC, P_DC, Transformer_R, Transformer_X, Phase_Reactor_R, Phase_Reactor_X, Filter, Droop, kV_base, MVA_max=MVA_max,nConvP=n,polarity=pol,Ucmin=Ucmin,Ucmax=Ucmax, name=str(var_name))
        Convertor_list = list(Converters.values())

    if DCDC_conv is None:
        Convertor_DC_list = None
    else:
        DCDC_conv = Converter_data.set_index('Conv_id')
        "Convertor data sorting"
        Converters_DC = {}
        for index, row in Converter_data.iterrows():
            var_name = index
            element_type = DCDC_conv.at[index, 'type']

            AC_node = DCDC_conv.at[index, 'fromNode']
            DC_node = DCDC_conv.at[index, 'toNode']

            PowerTo = DCDC_conv.at[index, 'P']
            R = DCDC_conv.at[index, 'R']

            Converters_DC[var_name] = DC_DC_converter(
                element_type, DC_nodes[fromNode], DC_nodes[toNode], PowerTo, R, name=str(var_name))
        Convertor_DC_list = list(Converters.values())

    G = Grid(S_base, AC_nodes_list, AC_lines_list, nodes_DC=DC_nodes_list,
             lines_DC=DC_lines_list, Converters=Convertor_list, conv_DC=Convertor_DC_list)
    res = Results(G, decimals=3)

    s = 1
    return [G, res]

def Create_grid_from_mat(matfile):
    data = loadmat(matfile)

    bus_columns = ['bus_i', 'type', 'Pd', 'Qd', 'Gs', 'Bs', 'area', 'Vm', 'Va', 'baseKV', 'zone', 'Vmax', 'Vmin']
    branch_columns = ['fbus', 'tbus', 'r', 'x', 'b', 'rateA', 'rateB', 'rateC', 'ratio', 'angle', 'status', 'angmin', 'angmax']
    gen_columns = ['bus', 'Pg', 'Qg', 'Qmax', 'Qmin', 'Vg', 'mBase', 'status', 'Pmax', 'Pmin', 'Pc1', 'Pc2', 'Qc1min', 'Qc1max', 'Qc2min', 'Qc2max', 'ramp_agc', 'ramp_10', 'ramp_30', 'ramp_q', 'apf']

    gencost_columns = ['2', 'startup', 'shutdown', 'n', 'c(n-1)','c(n-2)' ,'c0']

    busdc_columns = ['busdc_i',  'grid', 'Pdc', 'Vdc', 'basekVdc', 'Vdcmax', 'Vdcmin', 'Cdc']
    converter_columns = ['busdc_i', 'busac_i', 'type_dc', 'type_ac', 'P_g', 'Q_g', 'islcc', 'Vtar', 'rtf', 'xtf', 'transformer', 'tm', 'bf', 'filter', 'rc', 'xc', 'reactor', 'basekVac', 'Vmmax', 'Vmmin', 'Imax', 'status', 'LossA', 'LossB', 'LossCrec', 'LossCinv', 'droop', 'Pdcset', 'Vdcset', 'dVdcset', 'Pacmax', 'Pacmin', 'Qacmax', 'Qacmin']
    branch_DC = ['fbusdc', 'tbusdc', 'r', 'l', 'c', 'rateA', 'rateB', 'rateC', 'status']
    



    S_base = data['baseMVA'][0, 0]
    
    dcpol = data['dcpol'][0, 0] if 'dcpol' in data else 2
    
    
    
    if 'bus' in data:
        num_data_columns = len(data['bus'][0])
        if num_data_columns > len(bus_columns):
            # Add extra column names if needed
            extra_columns = [f"extra_column_{i}" for i in range(num_data_columns - len(bus_columns))]
            bus_columns = bus_columns + extra_columns
        else:
            # Use only the required number of columns from bus_columns
            bus_columns = bus_columns[:num_data_columns]
        AC_node_data = pd.DataFrame(data['bus'], columns=bus_columns)  
    else:
        AC_node_data = None
    
    if 'branch' in data:
        num_data_columns = len(data['branch'][0])
        if num_data_columns > len(branch_columns):
            # Add extra column names if needed
            extra_columns = [f"extra_column_{i}" for i in range(num_data_columns - len(branch_columns))]
            branch_columns = branch_columns + extra_columns
        else:
            # Use only the required number of columns from bus_columns
            branch_columns = branch_columns[:num_data_columns]
        AC_line_data = pd.DataFrame(data['branch'], columns=branch_columns)  
    else:
        AC_line_data = None
    
   
    if 'gen' in data:
        num_data_columns = len(data['gen'][0])
        if num_data_columns > len(gen_columns):
            # Add extra column names if needed
            extra_columns = [f"extra_column_{i}" for i in range(num_data_columns - len(gen_columns))]
            gen_columns = gen_columns + extra_columns
        else:
            # Use only the required number of columns from gen_columns
            gen_columns = gen_columns[:num_data_columns]
        Gen_data = pd.DataFrame(data['gen'], columns=gen_columns)  
    else:
        Gen_data = None
    
    
    # Gen_data = pd.DataFrame(data['gen'], columns=gen_columns)             if 'gen' in data else None    
    Gen_data_cost = pd.DataFrame(data['gencost'], columns=gencost_columns) if 'gencost' in data else None

    if 'busdc' in data:
        num_data_columns = len(data['busdc'][0])
        if num_data_columns > len(busdc_columns):
            # Add extra column names if needed
            extra_columns = [f"extra_column_{i}" for i in range(num_data_columns - len(busdc_columns))]
            busdc_columns = busdc_columns + extra_columns
        else:
            # Use only the required number of columns from gen_columns
            busdc_columns = busdc_columns[:num_data_columns]
        DC_node_data = pd.DataFrame(data['busdc'], columns=busdc_columns)  
    else:
        DC_node_data = None


    DC_line_data=pd.DataFrame(data['branchdc'], columns=branch_DC) if 'branchdc' in data else None
    Converter_data=pd.DataFrame(data['convdc'], columns=converter_columns) if 'convdc' in data else None

    s=1


    if AC_node_data is None:
        AC_nodes_list = None
        AC_lines_list = None
    else:
        "AC nodes data sorting"
        AC_node_data = AC_node_data.set_index('bus_i')
        AC_nodes = {}
        for index, row in AC_node_data.iterrows():
            var_name = index
            
            mat_type=AC_node_data.at[index, 'type']
            if mat_type == 1:
                element_type = 'PQ'
            elif mat_type == 2:
                element_type = 'PV'
            elif mat_type == 3:
                element_type = 'Slack'
             
            Gs = AC_node_data.at[index, 'Gs']/S_base
            Bs = AC_node_data.at[index, 'Bs']/S_base
          
            kV_base         = AC_node_data.at[index, 'baseKV']
            Voltage_0       = AC_node_data.at[index, 'Vm']
            theta_0         = np.radians(AC_node_data.at[index, 'Va'])     
            
            
            Power_Gained = (Gen_data[Gen_data['bus'] == index]['Pg'].values[0] / S_base 
                if Gen_data is not None and not Gen_data[Gen_data['bus'] == index].empty 
                and Gen_data[Gen_data['bus'] == index]['status'].values[0] != 0 else 0)
            Reactive_Gained  = (Gen_data[Gen_data['bus'] == index]['Qg'].values[0] / S_base 
                if Gen_data is not None and not Gen_data[Gen_data['bus'] == index].empty 
                and Gen_data[Gen_data['bus'] == index]['status'].values[0] != 0 else 0)
            
            Power_load      = AC_node_data.at[index, 'Pd']/S_base   
            Reactive_load   = AC_node_data.at[index, 'Qd']/S_base
            Umin            = AC_node_data.at[index, 'Vmin']           
            Umax            = AC_node_data.at[index, 'Vmax']        
            x_coord         = AC_node_data.at[index, 'x_coord']         if 'x_coord'         in AC_node_data.columns else None
            y_coord         = AC_node_data.at[index, 'y_coord']         if 'y_coord'         in AC_node_data.columns else None
            

            AC_nodes[var_name] = Node_AC(element_type, Voltage_0, theta_0,kV_base, Power_Gained,
                                         Reactive_Gained, Power_load, Reactive_load, name=str(var_name),Umin=Umin,Umax=Umax,Gs=Gs,Bs=Bs,x_coord=x_coord,y_coord=y_coord)
        AC_nodes_list = list(AC_nodes.values())

        
        AC_lines = {}
        for index, row in AC_line_data.iterrows():
          if AC_line_data.at[index, 'status'] !=0:    
            var_name = index+1
            

            fromNode     = AC_line_data.at[index, 'fbus']
            toNode       = AC_line_data.at[index, 'tbus']
            Resistance   = AC_line_data.at[index, 'r']
            Reactance    = AC_line_data.at[index, 'x']    
            Conductance  = 0
            Susceptance  = AC_line_data.at[index, 'b']  
            
            
            
            kV_base      = AC_nodes[toNode].kV_base 
            if AC_line_data.at[index, 'rateA'] == 0:
                MVA_rating=9999
            else:
                MVA_rating   = AC_line_data.at[index, 'rateA']
            if AC_line_data.at[index, 'ratio']== 0:
                m=1
                shift=0
            else:
                m            = AC_line_data.at[index, 'ratio']  
                shift        = np.radians(AC_line_data.at[index, 'angle'])

            
            
            AC_lines[var_name] = Line_AC(AC_nodes[fromNode], AC_nodes[toNode], Resistance,
                                         Reactance, Conductance, Susceptance, MVA_rating, kV_base,m,shift ,name=str(var_name))
        AC_lines_list = list(AC_lines.values())

    if DC_node_data is None:

        DC_nodes_list = None
        DC_lines_list = None

    else:
        DC_node_data = DC_node_data.set_index('busdc_i')

        "DC nodes data sorting"
        DC_nodes = {} 
        for index, row in DC_node_data.iterrows():

            var_name = index
            node_type = 'P'

            Voltage_0     = DC_node_data.at[index, 'Vdc'] 
            Power_Gained  = 0
            Power_load    = DC_node_data.at[index, 'Pdc']/S_base   
            kV_base       = DC_node_data.at[index, 'basekVdc']  
            Umin          = DC_node_data.at[index, 'Vdcmin']         
            Umax          = DC_node_data.at[index, 'Vdcmax']       
            x_coord       = DC_node_data.at[index, 'x_coord']       if 'x_coord'       in DC_node_data.columns else None
            y_coord       = DC_node_data.at[index, 'y_coord']       if 'y_coord'       in DC_node_data.columns else None
            
            
                
            DC_nodes[var_name] = Node_DC(
                node_type, Voltage_0, Power_Gained, Power_load,kV_base ,name=str(var_name),Umin=Umin,Umax=Umax,x_coord=x_coord,y_coord=y_coord)
        DC_nodes_list = list(DC_nodes.values())

        # DC_line_data = DC_line_data.set_index('Line_id')
        DC_lines = {}
        for index, row in DC_line_data.iterrows():
           if DC_line_data.at[index, 'status'] !=0:    
            var_name = index+1

            fromNode      = DC_line_data.at[index, 'fbusdc']
            toNode        = DC_line_data.at[index, 'tbusdc']
            Resistance    = DC_line_data.at[index, 'r']
            MW_rating     = DC_line_data.at[index, 'rateA']    
            kV_base       = DC_nodes[toNode].kV_base 
            
            if dcpol == 2:
                pol = 'b'
            else:
                pol = 'sm'
            DC_lines[var_name] = Line_DC(DC_nodes[fromNode], DC_nodes[toNode], Resistance, MW_rating, kV_base, polarity=pol, name=str(var_name))
        DC_lines_list = list(DC_lines.values())

    if Converter_data is None:
        Convertor_list = None
    else:
        # Converter_data = Converter_data.set_index('Conv_id')
        "Convertor data sorting"
        Converters = {}
        for index, row in Converter_data.iterrows():
          if Converter_data.at[index, 'status'] !=0:   
            var_name  = index+1
            
            type_ac = Converter_data.at[index, 'type_ac']   
            if type_ac == 1:
                AC_type = 'PQ'
            elif type_ac == 2:
                AC_type = 'PV'
          
            type_dc= Converter_data.at[index, 'type_dc']     
            if type_dc == 1:
                 DC_type = 'P'
            elif type_dc == 2:
                DC_type = 'Slack'
            elif type_dc == 3:
                DC_type = 'Droop'
             
            
                       
            DC_node         = Converter_data.at[index, 'busdc_i']   
            AC_node         = Converter_data.at[index, 'busac_i']            
            P_AC            = Converter_data.at[index, 'P_g']/S_base      
            Q_AC            = Converter_data.at[index, 'Q_g']/S_base         
            P_DC            = Converter_data.at[index, 'Pdcset']/S_base         
            Transformer_R   = Converter_data.at[index, 'rtf']          
            Transformer_X   = Converter_data.at[index, 'xtf']           
            Phase_Reactor_R = Converter_data.at[index, 'rc']           
            Phase_Reactor_X = Converter_data.at[index, 'xc']      
            Filter          = Converter_data.at[index, 'bf']      
            Droop           = Converter_data.at[index, 'droop']        
            kV_base         = Converter_data.at[index, 'basekVac']    
            
            P_max  = Converter_data.at[index, 'Pacmax']
            P_min  = Converter_data.at[index, 'Pacmin']
            Q_max  = Converter_data.at[index, 'Qacmax']
            Q_min  = Converter_data.at[index, 'Qacmin']
            
            maxP = max(abs(P_max),abs(P_min))
            maxQ = max(abs(Q_max),abs(Q_min))
            
            MVA_max         = max(maxP,maxQ)
            Ucmin           = Converter_data.at[index, 'Vmmin']        
            Ucmax           = Converter_data.at[index, 'Vmmax']        
            n               = 1
            pol             = 1
            
            LossA           = Converter_data.at[index, 'LossA']
            LossB           = Converter_data.at[index, 'LossB']
            LossCrec        = Converter_data.at[index, 'LossCrec']
            LossCinv        = Converter_data.at[index, 'LossCinv']
            

            Converters[var_name] = AC_DC_converter(AC_type, DC_type, AC_nodes[AC_node], DC_nodes[DC_node], P_AC, Q_AC, P_DC, Transformer_R, Transformer_X, Phase_Reactor_R, Phase_Reactor_X, Filter, Droop, kV_base, MVA_max=MVA_max,nConvP=n,polarity=pol,Ucmin=Ucmin,Ucmax=Ucmax,lossa=LossA,lossb=LossB,losscrect=LossCrec ,losscinv=LossCinv ,name=str(var_name))
        Convertor_list = list(Converters.values())



    G = Grid(S_base, AC_nodes_list, AC_lines_list, nodes_DC=DC_nodes_list,
             lines_DC=DC_lines_list, Converters=Convertor_list, conv_DC=None)
    res = Results(G, decimals=3)
    
    if Gen_data is not None:        
        for index, row in Gen_data.iterrows():
          if Gen_data.at[index, 'status'] !=0:  
            var_name = index+1 
            node_name = str(Gen_data.at[index, 'bus'])
            
            MWmax  = Gen_data.at[index, 'Pmax']
            MWmin   = Gen_data.at[index, 'Pmin']
            MVArmin = Gen_data.at[index, 'Qmin']
            MVArmax = Gen_data.at[index, 'Qmax']
            
            
            
            
            PsetMW = Gen_data.at[index,'Pg']
            QsetMVA = Gen_data.at[index,'Qg']

            lf = Gen_data_cost.at[index, 'c(n-2)']   
            qf = Gen_data_cost.at[index, 'c(n-1)'] 
            
            price_zone_link = False
            
        

            add_gen(G, node_name,var_name, price_zone_link,lf,qf,MWmax,MWmin,MVArmin,MVArmax,PsetMW,QsetMVA) 
            
    
    return [G, res]



"Add main components"

def add_AC_node(grid, node_type, kV_base ,Voltage_0=1.01, theta_0=0.01, Power_Gained=0, Reactive_Gained=0, Power_load=0, Reactive_load=0, name=None, Umin=0.9, Umax=1.1,x_coord=None,y_coord=None):
    node = Node_AC(element_type, Voltage_0, theta_0, kV_base ,Power_Gained,Reactive_Gained, Power_load, Reactive_load, name,Umin,Umax,x_coord,y_coord)
    grid.nodes_AC.append(node)
    grid.nn_AC +=1
    return node

def add_DC_node(grid, node_type,kV_base , Voltage_0=1.01, Power_Gained=0, Power_load=0, name=None,Umin=0.95, Umax=1.05,x_coord=None,y_coord=None):
    node = Node_DC(node_type, Voltage_0, Power_Gained, Power_load,kV_base , name,Umin, Umax,x_coord,y_coord)
    grid.nodes_DC.append(node)
    grid.nn_DC +=1
    return node
    
def add_line_AC(grid, fromNode, toNode,MVA_rating, Resistance_pu, Reactance_pu=0, Conductance_pu=0, Susceptance_pu=0,m=1, shift=0, name=None,tap_changer=False,Expandable=False,Length_km=1):
    kV_base=toNode.kV_base
    if tap_changer:
        line = Tap_changer_transformer(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating, kV_base,m, shift, name)
        grid.lines_AC_tf.append(line)
        grid.nttf+=1
    elif Expandable:
        line = Exp_Line_AC(Length_km,fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating, kV_base,m, shift, name)
        grid.lines_AC_exp.append(line)
        grid.nle_AC+=1
    else:    
        line = Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating, kV_base,m, shift, name)
        grid.lines_AC.append(line)
        grid.nl_AC +=1
        grid.create_Ybus_AC()
    return line

def change_line_AC_to_expandable(grid, line_name, Length_km=1):
    for line_to_process in grid.lines_AC:
        if line_name == line_to_process.name:
            l  = line_to_process
            break
    if l is not None:    
            grid.lines_AC.remove(l)
            l.remove()
            grid.nl_AC -=1
            line_vars=l.get_relevant_attributes()
            expandable_line = Exp_Line_AC(Length_km=Length_km, **line_vars)
            grid.lines_AC_exp.append(expandable_line)
            grid.nle_AC+=1
            

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
            grid.nl_AC -=1
            line_vars=l.get_relevant_attributes()
            trafo = TF_Line_AC(**line_vars)
            grid.lines_AC_tf.append(trafo)
            grid.nttf+=1
    else:
        print(f"Line {line_name} not found.")
        return
    # Reassign line numbers to ensure continuity in grid.lines_AC
    for i, line in enumerate(grid.lines_AC):
        line.lineNumber = i 
    grid.create_Ybus_AC()
    s=1    

def add_line_DC(grid, fromNode, toNode, Resistance_pu, MW_rating,km=1, polarity='m', name=None):
    kV_base=toNode.kV_base
    line = Line_DC(fromNode, toNode, Resistance_pu, MW_rating, kV_base,km, polarity, name)
    grid.lines_DC.append(line)
    grid.nl_DC +=1
    grid.create_Ybus_DC()
    return line

def add_ACDC_converter(grid, DC_type, AC_node, DC_node,kV_base_AC,AC_type = 'PV' ,P_AC=0, Q_AC=0, P_DC=0, Transformer_resistance=0, Transformer_reactance=0, Phase_Reactor_R=0, Phase_Reactor_X=0, Filter=0, Droop=0, MVA_max= None ,nConvP=1,polarity=1 ,Ucmin= 0.85, Ucmax= 1.2, name=None):
    if MVA_max is None:
        MVA_max= grid.S_base*1.05
    if Filter !=0 and Phase_Reactor_R==0 and  Phase_Reactor_X!=0:
        print(f'Please fill out phase reactor values, converter {name} not added')
        return
    conv = AC_DC_converter(AC_type, DC_type, AC_node, DC_node, P_AC, Q_AC, P_DC, Transformer_resistance, Transformer_reactance, Phase_Reactor_R, Phase_Reactor_X, Filter, Droop, kV_base_AC,MVA_max,nConvP,polarity,Ucmin,Ucmax,name)
    grid.Converters_ACDC.append(conv)
    grid.nconv +=1
    return conv

"Zones"


def add_RenSource_zone(Grid,name):
        
    RSZ = Ren_source_zone(name)
    Grid.RenSource_zones.append(RSZ)
    
    return RSZ


def add_price_zone(Grid,name,price,import_pu_L=1,export_pu_G=1,a=0,b=1,c=0,import_expand_pu=0):

    if b==1:
        b= price
    
    M = Price_Zone(price,import_pu_L,export_pu_G,a,b,c,import_expand_pu,name)
    Grid.Price_Zones.append(M)
    
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
    Gen_data = pd.read_csv(Gen_csv)
    Gen_data = Gen_data.set_index('Gen')
    
    
    for index, row in Gen_data.iterrows():
        var_name = index
        node_name = str(Gen_data.at[index, 'Node'])
        
        MVAmax = Gen_data.at[index, 'MWmax'] if 'MWmax' in Gen_data.columns else None
        MWmin = Gen_data.at[index, 'MWmin'] if 'MWmin' in Gen_data.columns else None
        MVArmin = Gen_data.at[index, 'MVArmin'] if 'MVArmin' in Gen_data.columns else 0
        MVArmax = Gen_data.at[index, 'MVArmax'] if 'MVArmax' in Gen_data.columns else 99999
        
        PsetMW = Gen_data.at[index, 'PsetMW'] if 'MVArmax' in Gen_data.columns else 0
        
        lf = Gen_data.at[index, 'Linear factor']    if 'Linear factor' in Gen_data.columns else 0
        qf = Gen_data.at[index, 'Quadratic factor'] if 'Quadratic factor' in Gen_data.columns else 0
        
        price_zone_link = False
        
        

        add_gen(Grid, node_name,var_name, price_zone_link,lf,qf,MVAmax,MWmin,MVArmin,MVArmax,PsetMW)  
        
def add_gen(Grid, node_name,gen_name=None, price_zone_link=False,lf=0,qf=0,MWmax=99999,MWmin=0,MVArmin=None,MVArmax=None,PsetMW=0,QsetMVA=0,Smax=None):
    
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


            
def add_Reactor(Grid, node_name,react_name=None,MVArmin=-99999,MVArmax=99999):
    found=False
    Max_pow_genR= MVArmax/Grid.S_base
    Min_pow_genR=MVArmin/Grid.S_base
    for node in Grid.nodes_AC:
        if node_name == node.name:
            react = Reactive_AC(Grid,node,Min_pow_genR,Max_pow_genR,react_name)
            found=True
            node.QGi=0
            break
    if not found:
       print('Node {node_name} does not exist')
    Grid.Reactive_compensation.append(gen)        
   
            
            
def add_extGrid(Grid, node_name, gen_name=None,price_zone_link=False,lf=0,qf=0,MVAmax=99999,MVArmin=None,MVArmax=None,Allow_sell=False):
    
    
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

def add_RenSource(Grid,node_name, base,ren_source_name=None , available=1,zone=None,price_zone=None, Offshore=False,MTDC=None):
    if ren_source_name is None:
        ren_source_name= node_name
    found=False 
    for node in Grid.nodes_AC:
        if node_name == node.name:
            rensource= Ren_Source(ren_source_name,node,base/Grid.S_base)    
            rensource.PRGi_available=available
            rensource.connected= 'AC'
            ACDC='AC'
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
                    # If the offshore price_zone does not exist, create it as an OffshorePrice_Zone
                    from PyFlow_ACDC import add_MTDC_price_zone
                    # Create the offshore price_zone using the OffshorePrice_Zone class
                    MTDC_price_zone= add_MTDC_price_zone(Grid,MTDC)
            
            MTDC_price_zone.add_linked_price_zone(main_price_zone)
            main_price_zone.ImportExpand += base / Grid.S_base
            assign_nodeToPrice_Zone(Grid, node_namel,ACDC, MTDC)
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
                    # If the offshore price_zone does not exist, create it as an OffshorePrice_Zone
                    from PyFlow_ACDC import add_offshore_price_zone
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

def add_TimeSeries(Grid, Time_Series_data):
    TS = Time_Series_data
    Time_series = {}
    # check if there are nan values in Time series and change to 0
    TS.fillna(0, inplace=True)
    
    for col in TS.columns:
        element_name = TS.at[0, col]
        element_type = TS.at[1, col]
        data = TS.loc[2:, col].astype(float).to_numpy()
          
        name = col
        
        Time_serie = TimeSeries_AC(element_type, element_name, data,name)                  
        Grid.Time_series.append(Time_serie)
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
    
def assign_nodeToPrice_Zone(Grid, node_name,ACDC, new_price_zone_name):
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
            raise ValueError(f"Node {node_name} not found.")
        
        # Add node to the new price_zone
        if conv_to_reassign not in new_price_zone.ConvACDC:
            new_price_zone.ConvACDC.append(conv_to_reassign)            
            

"Iterating data for chosen cases for TEP"
    

def update_grid_price_zone_data(grid, hour, Price_Zone_files, Avalaility_factors, Load_factors,P_minmax=None,coeff=None):
    # Extract the price_zone keys
    price_zone_keys = list(Price_Zone_files.keys())

    # Create a list of DataFrames for each price_zone
    data_panda_list = []
   
    for key in price_zone_keys:
        # Append the data for each price_zone, assigning the price_zone name
        data_panda_list.append(pd.DataFrame([Price_Zone_files[key].loc[hour]]).assign(Price_Zone=key))

    # Concatenate the DataFrames and set 'Price_Zone' as the index
    Time_step_data = pd.concat(data_panda_list, ignore_index=True)
    Time_step_data.set_index('Price_Zone', inplace=True)

    # Initialize the dictionary to store min/max power limits for price_zones
    if P_minmax is None:
        P_minmax = {}
    if coeff is None:
        coeff = {}
    
    if hour not in coeff:
        coeff[hour] = {'a_CG': {}, 'b_CG': {}, 'c_CG': {}}
    
        
    # Update price_zone data in the grid
    for price_zone in grid.Price_Zones:
        
        if type(price_zone) is Price_Zone:
            # print(price_zone.name)
            # Update price_zone parameters based on the Time_step_data
            price_zone.price = Time_step_data['price'][price_zone.name]
            price_zone.a = Time_step_data['a_CG'][price_zone.name]
            price_zone.b = Time_step_data['b_CG'][price_zone.name]
            price_zone.c = Time_step_data['c_CG'][price_zone.name]
            price_zone.PGL_min = Time_step_data['PGL_min'][price_zone.name] / grid.S_base
            price_zone.PGL_max = Time_step_data['PGL_max'][price_zone.name] / grid.S_base

            # Store power min/max limits in P_minmax dictionary
            P_minmax[f'Pmin_{price_zone.name}'] = price_zone.PGL_min
            P_minmax[f'Pmax_{price_zone.name}'] = price_zone.PGL_max
            
            coeff[hour]['a_CG'][price_zone.name] = Time_step_data['a_CG'][price_zone.name]
            coeff[hour]['b_CG'][price_zone.name] = Time_step_data['b_CG'][price_zone.name]
            coeff[hour]['c_CG'][price_zone.name] = Time_step_data['c_CG'][price_zone.name]
            
            
            # Update load factors
            price_zone.PLi_factor = Load_factors.loc[hour, price_zone.name]

            # Adjust price_zone parameters if applicable
            if price_zone.b > 0:
                price_zone.PGL_min -= price_zone.ImportExpand
                price_zone.a = -price_zone.b / (2 * price_zone.PGL_min * grid.S_base)
                coeff[hour]['a_CG'][price_zone.name] = price_zone.a
            # Store expanded min limit after adjustment
            P_minmax[f'Pminexp_{price_zone.name}'] = price_zone.PGL_min

    # Update renewable source zones data
    for zone in grid.RenSource_zones:
        zone.PRGi_available = Avalaility_factors.loc[hour, zone.name]
    
    return P_minmax , coeff




# Dynamically load all .py files in the 'cases/' folder
case_folder = Path(__file__).parent / "example_grids"

# Namespace for all loaded cases
cases = {}

# Load each .py file in the cases folder
for case_file in case_folder.glob("*.py"):
    module_name = case_file.stem  # Get the file name without extension
    spec = importlib.util.spec_from_file_location(module_name, case_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Add all public functions from the module to the `cases` namespace
    cases.update({name: obj for name, obj in vars(module).items() if not name.startswith("_")})

# Optional: Add all cases to this module's global namespace
globals().update(cases)
