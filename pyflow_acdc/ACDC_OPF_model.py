# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:25:02 2024

@author: BernardoCastro
"""

import pyomo.environ as pyo
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def analyse_OPF(grid):
    OnlyAC = True
    if grid.nn_DC!=0:
        OnlyAC = False
    TAP_tf = False
    if grid.nttf !=0:
        TAP_tf = True
    
    
    return OnlyAC,TAP_tf
    

def OPF_createModel_ACDC(model,grid,PV_set,Price_Zones,MI=None):
    from PyFlow_ACDC_OPF import Translate_pyf_OPF 
    
    OnlyAC,TAP_tf = analyse_OPF(grid)
    
    
    if OnlyAC:
        [AC_info,Price_Zone_info]=Translate_pyf_OPF(grid,OnlyAC,Price_Zones=Price_Zones)
    else:
        [AC_info,DC_info,Conv_info,Price_Zone_info]=Translate_pyf_OPF(grid,OnlyAC,Price_Zones=Price_Zones)
   

    AC_Lists,AC_nodes_info,AC_lines_info,gen_info = AC_info
    lf,qf,P_renSource = gen_info
    
    lista_nodos_AC, lista_lineas_AC,lista_lineas_AC_exp,lista_lineas_AC_tf,lista_gen,lista_rs, AC_slack, AC_PV = AC_Lists
    u_min_ac,u_max_ac,V_ini_AC,Theta_ini, P_know,Q_know,price = AC_nodes_info
    S_lineAC_limit,S_lineACexp_limit,S_lineACtf_limit,m_tf_og,NP_lineAC = AC_lines_info
    
    if not OnlyAC:
        DC_Lists,DC_nodes_info,DC_lines_info = DC_info
        
        lista_nodos_DC, lista_lineas_DC,DC_slack ,DC_nodes_connected_conv   = DC_Lists
        u_min_dc, u_max_dc ,V_ini_DC,P_known_DC,price_dc  = DC_nodes_info
        P_lineDC_limit,NP_lineDC    = DC_lines_info
        
            
        Conv_Lists, Conv_Volt = Conv_info
        
        lista_conv,NumConvP_i = Conv_Lists
        u_c_min,u_c_max,S_limit_conv,P_conv_limit = Conv_Volt
        
    if MI is None: #Automatically fed from grid class
        Price_Zone_Lists,Price_Zone_lim = Price_Zone_info   
    else: #Manually fed
        Price_Zone_Lists,Price_Zone_lim = MI
    lista_M, node2price_zone ,price_zone2node=Price_Zone_Lists
    price_zone_as,price_zone_bs,PGL_min, PGL_max = Price_Zone_lim
    """
    MODEL INITIATION
    """


    
    "Model Sets"
    model.nodes_AC   = pyo.Set(initialize=lista_nodos_AC)
    model.lines_AC   = pyo.Set(initialize=lista_lineas_AC)
    
    model.lines_AC_exp = pyo.Set(initialize=lista_lineas_AC_exp)
    if TAP_tf:
        model.lines_AC_tf  = pyo.Set(initialize=lista_lineas_AC_tf) 
    
    
    model.gen_AC     = pyo.Set(initialize=lista_gen)
    
    model.ren_sources= pyo.Set(initialize=lista_rs)
    
    model.AC_slacks  = pyo.Set(initialize=AC_slack)
    model.AC_PVs     = pyo.Set(initialize=AC_PV)
    if not OnlyAC:
        model.nodes_DC   = pyo.Set(initialize=lista_nodos_DC)
        model.lines_DC   = pyo.Set(initialize=lista_lineas_DC)
        model.DC_slacks  = pyo.Set(initialize=DC_slack)
    
        model.conv       = pyo.Set(initialize=lista_conv)
     
    """Variables and limits
    """
    "Price_Zone Variables"
    def Price_Zone_P_bounds(model, price_zone):
        nM = grid.Price_Zones[price_zone]
        return (nM.PGL_min,nM.PGL_max)
    
    def lf_bounds(model, ngen):
        gen = grid.Generators[ngen]
        if gen.price_zone_link:
            return (None,None)
        else:
            return (lf[ngen],lf[ngen])
    
    if  Price_Zones:
        model.M = pyo.Set(initialize=lista_M)
        model.PN = pyo.Var(model.M,bounds=Price_Zone_P_bounds,initialize=0)
        # model.PN_load = pyo.Var(model.M)
        model.price = pyo.Var(model.nodes_AC,initialize=price)
        model.price_dc = pyo.Var(model.nodes_DC,initialize=price_dc)
        
        model.lf = pyo.Var (model.gen_AC,bounds=lf_bounds,  initialize=lf)
        model.price_zone_price = pyo.Var(model.M,initialize=0)
        model.price_zone_a = pyo.Param(model.M,initialize=price_zone_as,mutable=True)
        model.price_zone_b = pyo.Param(model.M,initialize=price_zone_bs,mutable=True)
        model.SocialCost = pyo.Var(model.M,initialize=0)
        
    else:
        model.price  = pyo.Param(model.nodes_AC, initialize=price,mutable=True)
        model.lf = pyo.Param (model.gen_AC, initialize=lf, mutable=True)
        if not OnlyAC:
            model.price_dc  = pyo.Param(model.nodes_DC, initialize=price_dc,mutable=True)
    
    "AC Variables"
    #AC nodes variables
    model.V_AC       = pyo.Var(model.nodes_AC, bounds=lambda model, node: (u_min_ac[node], u_max_ac[node]), initialize=V_ini_AC)
    model.thetha_AC  = pyo.Var(model.nodes_AC, bounds=(-1.6, 1.6), initialize=Theta_ini)

    model.P_known_AC = pyo.Param(model.nodes_AC, initialize=P_know,mutable=True)
    model.Q_known_AC = pyo.Param(model.nodes_AC, initialize=Q_know,mutable=True)
        
    def Pren_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_RenSource == []:
            return (0,0)
        else:
            return (None,None)
    
    def Qren_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_RenSource == []:
            return (0,0)
        else:
            return (None,None)
    
    def Pren_bounds_DC(model, node):
        nDC = grid.nodes_DC[node]
        if nDC.connected_RenSource == []:
            return (0,0)
        else:
            return (None,None)
    
    model.PGi_ren = pyo.Var(model.nodes_AC, bounds=Pren_bounds,initialize=0)
    model.QGi_ren = pyo.Var(model.nodes_AC, bounds=Qren_bounds,initialize=0)
    
    model.P_renSource = pyo.Param(model.ren_sources,initialize=P_renSource,mutable=True)
    
    def gamma_bounds(model,rs):
        ren_source= grid.RenSources[rs]
        if ren_source.curtailable:
            return (ren_source.min_gamma,1)
        else:
            return (1,1)
    model.gamma = pyo.Var(model.ren_sources, bounds=gamma_bounds, initialize=1)
    
    def Qren_bounds(model,rs):
        ren_source= grid.RenSources[rs]
        if ren_source.connected == 'AC':
            return (ren_source.Qmin,ren_source.Qmax)
        else:
            return (0,0)
    
    
    model.Q_renSource = pyo.Var(model.ren_sources,bounds=Qren_bounds, initialize=0)
    
    
    def P_Gen_bounds(model, ngen):
        gen = grid.Generators[ngen]
        return (gen.Min_pow_gen,gen.Max_pow_gen)
        
    def Q_Gen_bounds(model, ngen):
        gen = grid.Generators[ngen]
        return (gen.Min_pow_genR,gen.Max_pow_genR)
    
    def P_gen_ini(model,ngen):
        gen = grid.Generators[ngen]
        ini=gen.Pset
        if  gen.Min_pow_gen>ini:
            ini=gen.Min_pow_gen
        elif ini>gen.Max_pow_gen: 
             ini=gen.Max_pow_gen
        return (ini)
    
    def Q_gen_ini(model,ngen):
        gen = grid.Generators[ngen]
        ini=gen.Qset
        if gen.Min_pow_genR>ini:
            ini=gen.Min_pow_genR
        elif ini>gen.Max_pow_genR: 
             ini=gen.Max_pow_genR    
        return (ini)
    
   
    model.PGi_gen = pyo.Var(model.gen_AC,bounds=P_Gen_bounds, initialize=P_gen_ini)
    model.QGi_gen = pyo.Var(model.gen_AC,bounds=Q_Gen_bounds, initialize=Q_gen_ini) 
    
    
   
    def PGi_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_gen == []:
            return (0,0)
        else:
            return (None,None)
            
    def QGi_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_gen == []:
            return (0,0)
        else:
            return (None,None)
   
    
    model.PGi_opt = pyo.Var(model.nodes_AC,bounds=PGi_opt_bounds ,initialize=0)
    model.QGi_opt = pyo.Var(model.nodes_AC,bounds=QGi_opt_bounds, initialize=0)

    def toExp_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_toExpLine == []:
            return (0,0)
        else:
            return (None,None)
    
    def fromExp_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_fromExpLine == []:
            return (0,0)
        else:
            return (None,None)
        
    def toTF_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_toTFLine == []:
            return (0,0)
        else:
            return (None,None) 
        
    def fromTF_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_fromTFLine == []:
            return (0,0)
        else:
            return (None,None)     
    
    if TAP_tf:
        model.Pto_TF   = pyo.Var(model.nodes_AC,bounds=toTF_opt_bounds ,initialize=0)
        model.Pfrom_TF = pyo.Var(model.nodes_AC,bounds=fromTF_opt_bounds ,initialize=0)
        model.Qto_TF   = pyo.Var(model.nodes_AC,bounds=toTF_opt_bounds ,initialize=0)
        model.Qfrom_TF = pyo.Var(model.nodes_AC,bounds=fromTF_opt_bounds ,initialize=0)
   
    def AC_V_slack_rule(model, node):
        return model.V_AC[node] == V_ini_AC[node]

    def AC_theta_slack_rule(model, node):
        return model.thetha_AC[node] == Theta_ini[node]

    def AC_V_PV_rule(model, node):
        return model.V_AC[node] == V_ini_AC[node]

    
    model.AC_theta_slack_constraint = pyo.Constraint(model.AC_slacks, rule=AC_theta_slack_rule)
    if PV_set:
        model.AC_V_slack_constraint = pyo.Constraint(model.AC_slacks, rule=AC_V_slack_rule)
        model.AC_V_PV_constraint = pyo.Constraint(model.AC_PVs, rule=AC_V_PV_rule)
    
    #AC Lines variables
    def Sbounds_lines(model, line):
        return (-S_lineAC_limit[line], S_lineAC_limit[line])
   
    def Sbounds_lines_tf(model, line):
        return (-S_lineACtf_limit[line], S_lineACtf_limit[line])
    
    def bounds_tf_tap(model, tf):
        return (0.95*m_tf_og[tf], 1.05*m_tf_og[tf])
    
    model.PAC_to       = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.PAC_from     = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.QAC_to       = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.QAC_from     = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.PAC_line_loss= pyo.Var(model.lines_AC, initialize=0)
    
    if TAP_tf:
        model.tf_PAC_to       = pyo.Var(model.lines_AC_tf, bounds=Sbounds_lines_tf, initialize=0)
        model.tf_PAC_from     = pyo.Var(model.lines_AC_tf, bounds=Sbounds_lines_tf, initialize=0)
        model.tf_QAC_to       = pyo.Var(model.lines_AC_tf, bounds=Sbounds_lines_tf, initialize=0)
        model.tf_QAC_from     = pyo.Var(model.lines_AC_tf, bounds=Sbounds_lines_tf, initialize=0)
        model.tf_m            = pyo.Var(model.lines_AC_tf, bounds=bounds_tf_tap   , initialize=m_tf_og)
                  
        model.tf_PAC_line_loss= pyo.Var(model.lines_AC_tf, initialize=0)
    
    "DC variables"
    #DC nodes variables
    def DC_V_slack_rule(model, node):
        return model.V_DC[node] == V_ini_DC[node]
    def Pbounds_lines(model, line):
        return (-P_lineDC_limit[line], P_lineDC_limit[line])
    def P_conv_DC_node_bounds(model,node): #This limits the varable of those DC nodes that do not have a converter connected 
         if node in DC_nodes_connected_conv:
             return (None,None)
         else:
             return (0,0)
    
    if not OnlyAC:
        model.V_DC = pyo.Var(model.nodes_DC, bounds=lambda model, node: (u_min_dc[node], u_max_dc[node]), initialize=V_ini_DC)
        model.P_known_DC = pyo.Param(model.nodes_DC, initialize=P_known_DC,mutable=True)
        model.PGi_ren_DC = pyo.Var(model.nodes_DC, bounds=Pren_bounds_DC,initialize=0)

        model.DC_V_slack_constraint = pyo.Constraint(model.DC_slacks, rule=DC_V_slack_rule)
        
        #DC Lines variables
        
    
        model.PDC_to       = pyo.Var(model.lines_DC,bounds=Pbounds_lines ,  initialize=0)
        model.PDC_from     = pyo.Var(model.lines_DC,bounds=Pbounds_lines , initialize=0)
        model.PDC_line_loss= pyo.Var(model.lines_DC,bounds=Pbounds_lines , initialize=0)
        
        
        model.P_conv_DC = pyo.Var(model.nodes_DC, bounds=P_conv_DC_node_bounds,initialize=0)
   
       
    
    "Converter Variables"
    def conv_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if not nAC.connected_conv:
            return (0,0)
        else:
            return (None,None)
        
    if not OnlyAC:
        model.Uc   = pyo.Var(model.conv, bounds=lambda model, conv: (u_c_min[conv], u_c_max[conv]), initialize=1) 
        model.Uf   = pyo.Var(model.conv, bounds=lambda model, conv: (u_c_min[conv], u_c_max[conv]), initialize=1) 
        model.th_c   = pyo.Var(model.conv, bounds=(-1.6, 1.6), initialize=0) 
        model.th_f   = pyo.Var(model.conv, bounds=(-1.6, 1.6), initialize=0) 
        model.P_AC_loss_conv= pyo.Var(model.conv,within=pyo.NonNegativeReals)
        
           
        model.P_conv_loss = pyo.Var(model.conv, initialize=0)

        model.P_conv_AC = pyo.Var(model.nodes_AC,bounds=conv_opt_bounds, initialize=0)
        model.Q_conv_AC = pyo.Var(model.nodes_AC,bounds=conv_opt_bounds, initialize=0)
        
        model.P_conv_s_AC  = pyo.Var(model.conv, initialize=0)   
        model.Q_conv_s_AC = pyo.Var(model.conv, initialize=0)
    
        model.P_conv_c_AC  = pyo.Var(model.conv, initialize=0.0001)   
        model.Q_conv_c_AC = pyo.Var(model.conv, initialize=0.0001)
        
        model.P_conv_c_AC_sq = pyo.Var(model.conv, bounds=(1e-100,None), initialize=0.1)   
        model.Q_conv_c_AC_sq = pyo.Var(model.conv, bounds=(1e-100,None), initialize=0.1)
    
   
    
    if not OnlyAC:
        model.NumLinesDCP = pyo.Param(model.lines_DC,initialize=NP_lineDC)
        model.NumConvP = pyo.Param(model.conv,initialize=NumConvP_i)

    
    
    """EQUALITY CONSTRAINTS
    """
    "Price Zone equality constraints"
    
    def price_zone_price_formula(model,price_zone):
        from PyFlow_ACDC import Price_Zone
        if type(grid.Price_Zones[price_zone]) is Price_Zone:
            return model.price_zone_price[price_zone]==2*model.price_zone_a[price_zone]*model.PN[price_zone]*grid.S_base+model.price_zone_b[price_zone]
        else :
            return pyo.Constraint.Skip
    
    def node_price_set_AC(model,node):
        try: 
            price_zone=node2price_zone['AC'][node]
            return model.price_zone_price[price_zone]== model.price[node] 
        except:
            return model.price[node]==0
    def node_price_set_DC(model,node):
        try: 
            price_zone=node2price_zone['DC'][node]
            return model.price_zone_price[price_zone]== model.price_dc[node] 
        except:
            return model.price_dc[node]==0
    
    
    
    
    def P_price_zone(model,price_zone):
        Pm_AC=sum(model.P_known_AC[node]   + model.PGi_ren[node] + model.PGi_opt[node] for node in price_zone2node['AC'][price_zone])
        Pm_DC=sum(model.P_known_DC[node_DC]+ model.PGi_ren_DC[node_DC] for node_DC in price_zone2node['DC'][price_zone])
        
        return model.PN[price_zone] ==Pm_AC+Pm_DC

    def PZ_cost_of_generation(model,price_zone):
        from PyFlow_ACDC import Price_Zone
        if type(grid.Price_Zones[price_zone]) is Price_Zone:
            return model.SocialCost[price_zone]== model.price_zone_a[price_zone]*(model.PN[price_zone]*grid.S_base)**2+model.price_zone_b[price_zone]*(model.PN[price_zone]*grid.S_base)
        else:
            return model.SocialCost[price_zone]==0
        
        
    def Price_link(model,price_zone):
        from PyFlow_ACDC import Price_Zone
        if type(grid.Price_Zones[price_zone]) is Price_Zone:
            linked_price_zone=grid.Price_Zones[price_zone].linked_price_zone
            if linked_price_zone is not None:   
                return model.price_zone_price[price_zone] == model.price_zone_price[linked_price_zone.price_zone_num]
            else: 
                return pyo.Constraint.Skip
        else:
            return pyo.Constraint.Skip
    
    # def MTDC_price_link(model,price_zone):
    #     from PyFlow_ACDC import MTDCPrice_Zone
    #     if isinstance(grid.Price_Zones[price_zone], MTDCPrice_Zone): 
    #          pricing_strategy = grid.Price_Zones[price_zone].pricing_strategy
    #          linked_price_zones = [mkt.price_zone_num for mkt in grid.Price_Zones[price_zone].linked_price_zones]
    #          if pricing_strategy == 'min':
    #             return model.price_zone_price[price_zone] <= min(model.price_zone_price[mkt] for mkt in linked_price_zones)  # Set upper limit to the minimum of linked price_zones
    
    #          elif pricing_strategy == 'max':
    #             return model.price_zone_price[price_zone] >= max(model.price_zone_price[mkt] for mkt in linked_price_zones)  # Set lower limit to the maximum of linked price_zones
    
    #          elif pricing_strategy == 'avg':
    #             return model.price_zone_price[price_zone] == sum(model.price_zone_price[mkt] for mkt in linked_price_zones) / len(linked_price_zones)

    #          else:
    #             raise ValueError(f"Unsupported pricing strategy: {pricing_strategy}")
    #             return pyo.Constraint.Skip
    #     else:
    #         return pyo.Constraint.Skip
            
    if Price_Zones:
        grid.OPF_Price_Zones_constraints_used=True
        model.price_zone_gen_link = pyo.ConstraintList()
        for node in grid.nodes_AC:  # Loop through all nodes
            nAC = node.nodeNumber
            for g in node.connected_gen:  # Loop through all generators in the node
                if g.price_zone_link:
                    model.price_zone_gen_link.add(model.price[nAC] == model.lf[g.genNumber])
        
            
            
        model.price_zone_price_constraint = pyo.Constraint(model.M,rule=price_zone_price_formula)
        model.price_zone_price_link_ = pyo.Constraint(model.M,rule=Price_link)
        
        model.price_zone_MTDC_link = pyo.ConstraintList()
        
        from PyFlow_ACDC import MTDCPrice_Zone
        # Step 1: Define sets for the MTDC price_zones and linked price_zones
        model.MTDCPrice_Zones = pyo.Set(initialize=[m for m in model.M if isinstance(grid.Price_Zones[m], MTDCPrice_Zone)])
        
        for mtdc_price_zone in model.MTDCPrice_Zones:
            linked_price_zones = [mkt.price_zone_num for mkt in grid.Price_Zones[mtdc_price_zone].linked_price_zones]
            
            if not linked_price_zones:
                break
            # Define a set of linked price_zones for each MTDC price_zone
            # model.LinkedPrice_Zones = pyo.Set(initialize=linked_price_zones)
        
            pricing_strategy = grid.Price_Zones[mtdc_price_zone].pricing_strategy
        
            if pricing_strategy == 'min':
                grid.MixedBinCont=True
                # Step 1: Create distinct binary variables for each MTDC price_zone and its linked price_zones
                model.y_min = pyo.Var(linked_price_zones, model.MTDCPrice_Zones, domain=pyo.Binary, initialize=1)
    
                # Step 2: Ensure MTDC price_zone price is less than or equal to all linked price_zone prices
                for mkt in linked_price_zones:
                    model.price_zone_MTDC_link.add(model.price_zone_price[mtdc_price_zone] <= model.price_zone_price[mkt])
    
                # Step 3: Ensure that the MTDC price_zone price is equal to one of the linked price_zone prices
                model.price_zone_MTDC_link.add(sum(model.y_min[mkt, mtdc_price_zone] for mkt in linked_price_zones) == 1)
    
                # Step 4: Link the binary variable to the actual price_zone prices
                for mkt in linked_price_zones:
                    model.price_zone_MTDC_link.add(model.price_zone_price[mtdc_price_zone] == model.price_zone_price[mkt] * model.y_min[mkt, mtdc_price_zone])

            elif pricing_strategy == 'max':
                grid.MixedBinCont=True
                # Step 2: Create binary variables indexed by both the linked price_zones and the MTDC price_zone
                model.y_max = pyo.Var(linked_price_zones, model.MTDCPrice_Zones, domain=pyo.Binary, initialize=0)
        
                # Step 3: Ensure MTDC price_zone price is greater than or equal to all linked price_zone prices
                for mkt in linked_price_zones:
                    model.price_zone_MTDC_link.add(model.price_zone_price[mtdc_price_zone] >= model.price_zone_price[mkt])
        
                # Step 4: Ensure that the MTDC price_zone price is equal to one of the linked price_zone prices
                model.price_zone_MTDC_link.add(sum(model.y_max[mkt, mtdc_price_zone] for mkt in linked_price_zones) == 1)
        
                # Step 5: Link the binary variable to the actual price_zone prices
                for mkt in linked_price_zones:
                    model.price_zone_MTDC_link.add(model.price_zone_price[mtdc_price_zone] == model.price_zone_price[mkt] * model.y_max[mkt, mtdc_price_zone]/sum(model.y_max[mkt, mtdc_price_zone] for mkt in linked_price_zones))
        
            elif pricing_strategy == 'avg':
                # MTDC price_zone price equals the average of linked price_zone prices
                avg_expr = sum(model.price_zone_price[mkt] for mkt in linked_price_zones) / len(linked_price_zones)
                model.price_zone_MTDC_link.add(model.price_zone_price[mtdc_price_zone] == avg_expr)
        
            else:
                raise ValueError(f"Unsupported pricing strategy: {pricing_strategy}")
                    
                    
        
        model.price_constraint_AC = pyo.Constraint(model.nodes_AC,rule=node_price_set_AC)
        model.price_constraint_DC = pyo.Constraint(model.nodes_DC,rule=node_price_set_DC)
        
        model.PN_constraint = pyo.Constraint(model.M,rule=P_price_zone)
        # model.PNL_constraint = pyo.Constraint(model.M,rule=P_price_zone_load)
        model.CG_constraint = pyo.Constraint(model.M,rule=PZ_cost_of_generation)
        
    "AC equality constraints"
    # AC node constraints
    def P_AC_node_rule(model, node):
        P_sum = sum(
                model.V_AC[node] * model.V_AC[k] *
                (np.real(grid.Ybus_AC[node, k]) * pyo.cos(model.thetha_AC[node] - model.thetha_AC[k]) +
                 np.imag(grid.Ybus_AC[node, k]) * pyo.sin(model.thetha_AC[node] - model.thetha_AC[k]))
                for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0   )   
        P_var = model.P_known_AC[node] + model.PGi_ren[node] + model.PGi_opt[node]
        if not OnlyAC:
            P_var += model.P_conv_AC[node]
        if TAP_tf:
            P_var += model.Pto_TF[node]+model.Pfrom_TF[node]
        else:
            return P_sum == P_var

    def Q_AC_node_rule(model, node):

        Q_sum = sum(
            model.V_AC[node] * model.V_AC[k] *
            (np.real(grid.Ybus_AC[node, k]) * pyo.sin(model.thetha_AC[node] - model.thetha_AC[k]) -
             np.imag(grid.Ybus_AC[node, k]) * pyo.cos(model.thetha_AC[node] - model.thetha_AC[k]))
            for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0)
        Q_var = model.Q_known_AC[node] + model.QGi_ren[node] + model.QGi_opt[node]
        if not OnlyAC:
            Q_var += model.Q_conv_AC[node]
        if TAP_tf:
            Q_var += model.Qto_TF[node]+model.Qfrom_TF[node]
        else:
            return Q_sum == Q_var

    model.P_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=P_AC_node_rule)
    model.Q_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=Q_AC_node_rule)
    
    # Adds all generators in the AC nodes they are connected to
    def Gen_PAC_rule(model,node):
       nAC = grid.nodes_AC[node]
       P_gen = sum(model.PGi_gen[gen.genNumber] for gen in nAC.connected_gen)                  
       return  model.PGi_opt[node] ==   P_gen
           
    def Gen_Q_rule(model,node):
       nAC = grid.nodes_AC[node]
       Q_gen = sum(model.QGi_gen[gen.genNumber] for gen in nAC.connected_gen) 
       return  model.QGi_opt[node] ==   Q_gen
    
    model.Gen_PAC_constraint = pyo.Constraint(model.nodes_AC, rule=Gen_PAC_rule)
    model.Gen_QAC_constraint = pyo.Constraint(model.nodes_AC, rule=Gen_Q_rule)
    
    def Gen_PREN_rule(model,node):
       nAC = grid.nodes_AC[node]
       P_gen = sum(model.P_renSource[rs.rsNumber]*model.gamma[rs.rsNumber] for rs in nAC.connected_RenSource)                  
       return  model.PGi_ren[node] ==   P_gen
   
    def Gen_QREN_rule(model,node):
       nAC = grid.nodes_AC[node]
       Q_gen = sum(model.Q_renSource[rs.rsNumber] for rs in nAC.connected_RenSource)                  
       return  model.QGi_ren[node] ==   Q_gen
   
    model.Gen_PREN_constraint =pyo.Constraint(model.nodes_AC, rule=Gen_PREN_rule)
    model.Gen_QREN_constraint =pyo.Constraint(model.nodes_AC, rule=Gen_QREN_rule) 
    
    
    def toPtf_rule(model,node):
       nAC = grid.nodes_AC[node]
       toPtf = sum(model.tf_PAC_to[l.lineNumber] for l in nAC.connected_toTFLine)                  
       return  model.Pto_TF[node] ==   toPtf
    def fromPtf_rule(model,node):
       nAC = grid.nodes_AC[node]
       fromPtf = sum(model.tf_PAC_from[l.lineNumber] for l in nAC.connected_fromTFLine)                
       return  model.Pfrom_TF[node] ==   fromPtf
    
    def toQtf_rule(model,node):
       nAC = grid.nodes_AC[node]
       toQtf = sum(model.tf_QAC_to[l.lineNumber] for l in nAC.connected_toTFLine)                  
       return  model.Qto_TF[node] ==   toQtf
    def fromQtf_rule(model,node):
       nAC = grid.nodes_AC[node]
       fromQtf = sum(model.tf_QAC_from[l.lineNumber] for l in nAC.connected_fromTFLine)                  
       return  model.Qfrom_TF[node] ==  fromQtf   

    if TAP_tf:
        model.Pto_TF_constraint   = pyo.Constraint(model.nodes_AC, rule=toPtf_rule)
        model.Pfrom_TF_constraint = pyo.Constraint(model.nodes_AC, rule=fromPtf_rule)
        model.Qto_TF_constraint   = pyo.Constraint(model.nodes_AC, rule=toQtf_rule)
        model.Qfrom_TF_constraint = pyo.Constraint(model.nodes_AC, rule=fromQtf_rule)
    

    
    # AC line equality constraints
    
    def P_to_AC_line(model,line):   
       
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
        Gtt=np.real(l.Ybus_branch[1,1])
        Gtf=np.real(l.Ybus_branch[1,0])
        Btf=np.imag(l.Ybus_branch[1,0])
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Pto= Vt*Vt*Gtt + Vf*Vt*(Gtf*pyo.cos(tht - thf) + Btf*pyo.sin(tht - thf))
       
        
        return model.PAC_to[line] == Pto
    
    def P_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       f = l.fromNode.nodeNumber
       t = l.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
       Gff=np.real(l.Ybus_branch[0,0])
       Gft=np.real(l.Ybus_branch[0,1])
       Bft=np.imag(l.Ybus_branch[0,1])
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       
       Pfrom= Vf*Vf*Gff + Vf*Vt*(Gft*pyo.cos(thf - tht) + Bft*pyo.sin(thf - tht))

       return model.PAC_from[line] == Pfrom
    
    def Q_to_AC_line(model,line):   
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
       
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Btt=np.imag(l.Ybus_branch[1,1])
        Gtf=np.real(l.Ybus_branch[1,0])
        Btf=np.imag(l.Ybus_branch[1,0])
        
        Qto   = -Vt*Vt*Btt + Vf*Vt*(Gtf*pyo.sin(tht - thf) - Btf*pyo.cos(tht - thf))
         
        
        return model.QAC_to[line] == Qto
    
    def Q_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       f = l.fromNode.nodeNumber
       t = l.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
      
       Bff=np.imag(l.Ybus_branch[0,0])
       Gft=np.real(l.Ybus_branch[0,1])
       Bft=np.imag(l.Ybus_branch[0,1])
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       

       Qfrom = -Vf*Vf*Bff + Vf*Vt*(Gft*pyo.sin(thf - tht) - Bft*pyo.cos(thf - tht))
      

       return model.QAC_from[line] == Qfrom
    
    def P_loss_AC_rule(model,line):
        return model.PAC_line_loss[line]== model.PAC_to[line]+model.PAC_from[line]
    
    
    model.Pto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=P_to_AC_line)
    model.Pfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=P_from_AC_line)
    model.Qto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=Q_to_AC_line)
    model.Qfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=Q_from_AC_line)
    model.P_AC_loss_constraint     = pyo.Constraint(model.lines_AC, rule=P_loss_AC_rule)
    
    
    def P_to_AC_line_tf(model,trafo):   
        tf = grid.lines_AC_tf[trafo]
        f = tf.fromNode.nodeNumber
        t = tf.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
        
        Gtt=np.real(tf.Ybus_branch[1,1])
        Gtf=np.real(tf.Ybus_branch[1,0])
        Btf=np.imag(tf.Ybus_branch[1,0])
        
      
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        
        Pto= Vt*Vt*Gtt+ Vf/model.tf_m[tf]*Vt*(Gtf*pyo.cos(tht - thf) + Btf*pyo.sin(tht - thf))
       
        
        return model.tf_PAC_to[trafo] == Pto
    
    def P_from_AC_line_tf(model,trafo):       
       tf = grid.lines_AC_tf[trafo]
       f = tf.fromNode.nodeNumber
       t = tf.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
       Gff=np.real(tf.Ybus_branch[0,0])
       Gft=np.real(tf.Ybus_branch[0,1])
       Bft=np.imag(tf.Ybus_branch[0,1])
       
       
       
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       
       Pfrom= Vf*Vf*Gff/model.tf_m[tf]**2 + Vf/model.tf_m[tf]*Vt*(Gft*pyo.cos(thf - tht) + Bft*pyo.sin(thf - tht))

       return model.tf_PAC_from[trafo] == Pfrom
    
    def Q_to_AC_line_tf(model,trafo):   
        tf = grid.lines_AC_tf[trafo]
        f = tf.fromNode.nodeNumber
        t = tf.toNode.nodeNumber
        Vf=model.V_AC[f]
        Vt=model.V_AC[t]
       
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        
        Btt=np.imag(tf.Ybus_branch[1,1])
        Gtf=np.real(tf.Ybus_branch[1,0])
        Btf=np.imag(tf.Ybus_branch[1,0])
        
        Qto   = -Vt*Vt*Btt + Vf/model.tf_m[tf]*Vt*(Gtf*pyo.sin(tht - thf) - Btf*pyo.cos(tht - thf))
         
        
        return model.tf_QAC_to[trafo] == Qto
    
    def Q_from_AC_line_tf(model,trafo):       
       tf = grid.lines_AC_tf[trafo]
       f = tf.fromNode.nodeNumber
       t = tf.toNode.nodeNumber
       Vf=model.V_AC[f]
       Vt=model.V_AC[t]
      
       Bff=np.imag(tf.Ybus_branch[0,0])
       Gft=np.real(tf.Ybus_branch[0,1])
       Bft=np.imag(tf.Ybus_branch[0,1])
       
       thf=model.thetha_AC[f]
       tht=model.thetha_AC[t]
       

       Qfrom = -Vf*Vf*Bff/model.tf_m[tf]**2 + Vf/model.tf_m[tf]*Vt*(Gft*pyo.sin(thf - tht) - Bft*pyo.cos(thf - tht))
      

       return model.tf_QAC_from[trafo] == Qfrom
    
    def P_loss_AC_rule_tf(model,trafo):
        return model.tf_PAC_line_loss[trafo]== model.tf_PAC_to[trafo]+model.PAC_from[trafo]
    
    
    if TAP_tf:
        model.tf_Pto_AC_line_constraint   = pyo.Constraint(model.lines_AC_tf, rule=P_to_AC_line_tf)
        model.tf_Pfrom_AC_line_constraint = pyo.Constraint(model.lines_AC_tf, rule=P_from_AC_line_tf)
        model.tf_Qto_AC_line_constraint   = pyo.Constraint(model.lines_AC_tf, rule=Q_to_AC_line_tf)
        model.tf_Qfrom_AC_line_constraint = pyo.Constraint(model.lines_AC_tf, rule=Q_from_AC_line_tf)
        model.tf_P_AC_loss_constraint     = pyo.Constraint(model.lines_AC_tf, rule=P_loss_AC_rule_tf)
    
        

    "DC equality constraints"
    #DC node constraints
    
    def Gen_PREN_rule_DC(model,node):
       nDC = grid.nodes_DC[node]
       P_gen = sum(model.P_renSource[rs.rsNumber]*model.gamma[rs.rsNumber] for rs in nDC.connected_RenSource)                  
       return  model.PGi_ren_DC[node] ==   P_gen
   
    
    
    def P_DC_node_rule(model, node):
        i = node
        P_sum = 0
        for k in range(grid.nn_DC):
            Y = grid.Ybus_DC[i, k]

            if k != i:
                if Y != 0:
                    line = grid.get_lineDC_by_nodes(i, k)
                    pol = line.pol
                    
                    Y = -Y
                    P_sum += pol*model.V_DC[i] * (model.V_DC[i]-model.V_DC[k])*Y*model.NumLinesDCP[line.lineNumber]

        return P_sum == model.P_known_DC[node]+ model.PGi_ren_DC[node] + model.P_conv_DC[node]

    # def P_DC_noconv_rule(model, node):
    #     return model.P_conv_DC[node] == 0
    
    if not OnlyAC:
        model.Gen_PREN_constraint_DC =pyo.Constraint(model.nodes_DC, rule=Gen_PREN_rule_DC)
        model.P_DC_node_constraint = pyo.Constraint(model.nodes_DC, rule=P_DC_node_rule)
    
    #DC lines equality constraints
    
    def P_from_DC_line(model,line):       
        l = grid.lines_DC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        pol = l.pol
        
        Pfrom= (model.V_DC[t]-model.V_DC[f])*grid.Ybus_DC[f,t]*model.V_DC[f]*pol
        
        return model.PDC_from[line] == Pfrom
    
    def P_to_DC_line(model,line):   
        l = grid.lines_DC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        pol = l.pol

         
        Pto= (model.V_DC[f]-model.V_DC[t])*grid.Ybus_DC[t,f]*model.V_DC[t]*pol 
        
        
        return model.PDC_to[line] == Pto
    
    def P_loss_DC_line_rule(model,line):
        
        return model.PDC_line_loss[line]==(model.PDC_from[line]+ model.PDC_to[line])
    
    if not OnlyAC:
        model.Pfrom_DC_line_constraint   = pyo.Constraint(model.lines_DC, rule=P_from_DC_line)
        model.Pto_DC_line_constraint     = pyo.Constraint(model.lines_DC, rule=P_to_DC_line)
        model.Ploss_DC_line_constraint   = pyo.Constraint(model.lines_DC, rule=P_loss_DC_line_rule)    
        
    
    "Converter equality Constraints"
    
    def Conv_Ps_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        
       Gc  = element.Gc   
       Bc  = element.Bc   
       Gtf = element.Gtf  
       Btf = element.Btf  
       Bf  = element.Bf   
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           if Zeq == 0:
               return model.thetha_AC[nAC]==model.th_c[conv]
           Yeq = 1/Zeq
           
           Gc = np.real(Yeq)  
           Bc = np.imag(Yeq)  
           
           Ps = -model.V_AC[nAC]**2*Gc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
          

       elif element.Gtf == 0:
   
           Bcf = Bc+Bf

           Ps = -model.V_AC[nAC]**2*Gc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
          
           
       else:

           Ps = -model.V_AC[nAC]**2*Gtf+model.V_AC[nAC]*model.Uf[conv] * \
               (Gtf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv])+Btf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv]))
           
       return model.P_conv_s_AC[conv]-Ps==0
           
    def Conv_Qs_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
       
       Gc = element.Gc    
       Bc = element.Bc    
       Gtf = element.Gtf  
       Btf = element.Btf  
       Bf = element.Bf    
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           if Zeq == 0:
               return model.V_AC[nAC]==model.Uc[conv]
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  
           Bc = np.imag(Yeq)  
           
           Qs = model.V_AC[nAC]**2*Bc+model.V_AC[nAC]*model.Uc[conv]*(Gc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv])-Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))

       elif element.Gtf == 0:
  
           Bcf = Bc+Bf

           Qs = model.V_AC[nAC]**2*Bcf+model.V_AC[nAC]*model.Uc[conv] * \
                (Gc*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv])-Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))
         
       else:
                         
           Qs = model.V_AC[nAC]**2*Btf+model.V_AC[nAC]*model.Uf[conv] * \
               (Gtf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv])-Btf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv]))

       return model.Q_conv_s_AC[conv]-Qs==0
       
   
    

    def Conv_Pc_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
       
       Gc = element.Gc    
       Bc = element.Bc    
       Gtf = element.Gtf  
       Btf = element.Btf  
       Bf = element.Bf    
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           if Zeq == 0:
               return -model.P_conv_s_AC[conv]+model.P_conv_c_AC[conv]==0
           
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  
           Bc = np.imag(Yeq)  
           
           Pc = model.Uc[conv]**2*Gc-model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])-Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
          

       elif element.Gtf == 0:
                    
           Bcf = Bc+Bf
        
           Pc = model.Uc[conv]**2*Gc-model.V_AC[nAC]*model.Uc[conv]*(Gc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv])-Bc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv]))
           
           
       else:
           
           Pc = model.Uc[conv]**2*Gc-model.Uf[conv]*model.Uc[conv]*(Gc*pyo.cos(model.th_f[conv]-model.th_c[conv])-Bc*pyo.sin(model.th_f[conv]-model.th_c[conv]))
           
           
       return -Pc+model.P_conv_c_AC[conv]==0
           
    def Conv_Qc_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        
       Gc = element.Gc    
       Bc = element.Bc    
       Gtf = element.Gtf  
       Btf = element.Btf  
       Bf = element.Bf    
       
       if element.Bf == 0:
           Ztf = element.Ztf
           Zc = element.Zc
           Zeq = Ztf+Zc
           if Zeq == 0:
               return -model.Q_conv_s_AC[conv]+model.Q_conv_c_AC[conv]==0
           Yeq = 1/Zeq

           Gc = np.real(Yeq)  
           Bc = np.imag(Yeq)  
           
           
           Qc = -model.Uc[conv]**2*Bc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))

       elif element.Gtf == 0:
           
           Bcf = Bc+Bf

           Qc = -model.Uc[conv]*model.Uc[conv]*Bc+model.V_AC[nAC]*model.Uc[conv] * \
               (Gc*pyo.sin(model.thetha_AC[nAC]-model.th_c[conv])+Bc*pyo.cos(model.thetha_AC[nAC]-model.th_c[conv]))
     
       else:
           
           Qc = -model.Uc[conv]*model.Uc[conv]*Bc+model.Uf[conv]*model.Uc[conv] * \
               (Gc*pyo.sin(model.th_f[conv]-model.th_c[conv])+Bc*pyo.cos(model.th_f[conv]-model.th_c[conv]))

       return -Qc+model.Q_conv_c_AC[conv]==0




    def Conv_F1_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber

            
       if element.Bf == 0 or element.Gtf == 0:
        return pyo.Constraint.Skip
           
       else:
           Gc = element.Gc    
           Bc = element.Bc    
           Gtf = element.Gtf  
           Btf = element.Btf  
           Bf = element.Bf    
                
           Psf = model.Uf[conv]*model.Uf[conv]*Gtf-model.Uf[conv]*model.V_AC[nAC] * \
               (Gtf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv])-Btf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv]))
      
           Pcf = -model.Uf[conv]*model.Uf[conv]*Gc+model.Uf[conv]*model.Uc[conv] * \
               (Gc*pyo.cos(model.th_f[conv]-model.th_c[conv])+Bc*pyo.sin(model.th_f[conv]-model.th_c[conv]))
        

           F1 = Pcf-Psf
         
           
            
       return F1==0

    def Conv_F2_rule(model,conv):
       element=grid.Converters_ACDC[conv]
       nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
       nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
       constraints = pyo.ConstraintList()
       
       if element.Bf == 0 or element.Gtf == 0:
        return pyo.Constraint.Skip
           
       else:
           
           Gc = element.Gc    
           Bc = element.Bc    
           Gtf = element.Gtf  
           Btf = element.Btf  
           Bf = element.Bf    

         
           Qsf = -model.Uf[conv]**2*Btf+model.Uf[conv]*model.V_AC[nAC] * \
               (Gtf*pyo.sin(model.thetha_AC[nAC]-model.th_f[conv])+Btf*pyo.cos(model.thetha_AC[nAC]-model.th_f[conv]))

         
           Qcf = model.Uf[conv]**2*Bc+model.Uf[conv]*model.Uc[conv] * \
               (Gc*pyo.sin(model.th_f[conv]-model.th_c[conv])-Bc*pyo.cos(model.th_f[conv]-model.th_c[conv]))

           Qf = -model.Uf[conv]*model.Uf[conv]*Bf

           

           F2 = Qcf-Qsf-Qf
           
           
            
       return F2==0
    if not OnlyAC:
        model.Conv_Ps_constraint = pyo.Constraint(model.conv,rule=Conv_Ps_rule)
        model.Conv_Qs_constraint = pyo.Constraint(model.conv,rule=Conv_Qs_rule)
        model.Conv_Pc_constraint = pyo.Constraint(model.conv,rule=Conv_Pc_rule)
        model.Conv_Qc_constraint = pyo.Constraint(model.conv,rule=Conv_Qc_rule)
        model.Conv_F1_constraint = pyo.Constraint(model.conv,rule=Conv_F1_rule)
        model.Conv_F2_constraint = pyo.Constraint(model.conv,rule=Conv_F2_rule)
    
    # Adds all converters in the AC nodes they are connected to
    def Conv_PAC_rule(model,node):
       nAC = grid.nodes_AC[node]
       P_conv = sum(model.P_conv_s_AC[conv]*model.NumConvP[conv] for conv in nAC.connected_conv)
                  
       return  model.P_conv_AC[node] ==   P_conv
           
    def Conv_Q_rule(model,node):
       nAC = grid.nodes_AC[node]
       Q_conv = sum(model.Q_conv_s_AC[conv]*model.NumConvP[conv] for conv in nAC.connected_conv)
    
       return   model.Q_conv_AC[node] ==   Q_conv       
         

    # IGBTs losses
    def Conv_DC_rule(model, conv):
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber

        return model.P_conv_c_AC[conv]*model.NumConvP[conv]+model.P_conv_DC[nDC] + model.P_conv_loss[conv]*model.NumConvP[conv] == 0

    def Conv_loss_rule(model, conv):
        element=grid.Converters_ACDC[conv]
        nAC = grid.Converters_ACDC[conv].Node_AC.nodeNumber
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        a = grid.Converters_ACDC[conv].a_conv 
        b = grid.Converters_ACDC[conv].b_conv
      

        # current = pyo.sqrt(model.P_conv_c_AC_sq[conv]+model.Q_conv_c_AC_sq[conv])/(model.Uc[conv])
        currentsqr = (model.P_conv_c_AC[conv]**2+model.Q_conv_c_AC[conv]**2)/(model.Uc[conv]**2)

        

        # c_inver = (element.c_inver_og /model.NumConvP[conv])*element.basekA**2/grid.S_base
        # c_rect = (element.c_rect_og   /model.NumConvP[conv])*element.basekA**2/grid.S_base 
        
        c_inver=grid.Converters_ACDC[conv].c_inver 
        c_rect=grid.Converters_ACDC[conv].c_rect   
    
       
        P_loss = a  +c_rect * currentsqr
    
        
        return model.P_conv_loss[conv] == P_loss

    if not OnlyAC:
        model.Conv_DC_constraint = pyo.Constraint(model.conv, rule=Conv_DC_rule)
        model.Conv_PAC_constraint = pyo.Constraint(model.nodes_AC, rule=Conv_PAC_rule)
        model.Conv_QAC_constraint = pyo.Constraint(model.nodes_AC, rule=Conv_Q_rule)
        model.Conv_loss_constraint = pyo.Constraint(model.conv, rule=Conv_loss_rule)
   
    
    """
    INEQUALITY CONSTRAINTS
    """
    "Price_Zone inequality constraints"
    
    def import_rule(model,price_zone):
        return model.PN[price_zone] >= PGL_min[price_zone]
    def export_rule(model,price_zone):
        return model.PN[price_zone] <= PGL_max[price_zone]
    
    
    if Price_Zones:
        model.import_constraint = pyo.Constraint(model.M,rule=import_rule)
        model.export_constraint = pyo.Constraint(model.M,rule=export_rule)
    
    "AC inequality constraints"
    #AC gen inequality
    def S_gen_AC_limit_rule(model,ngen):
        gen = grid.Generators[ngen]
        if gen.Max_S is None:
            return pyo.Constraint.Skip
        else:    
            return model.PGi_gen[ngen]**2+model.QGi_gen[ngen]**2 <= gen.Max_S**2 
    
    model.S_gen_AC_limit_constraint   = pyo.Constraint(model.gen_AC, rule=S_gen_AC_limit_rule)
    #AC Ren sources inequality
    
    def S_renS_AC_limit_rule(model,rs):
        ren_source= grid.RenSources[rs]
        if ren_source.Max_S is None or ren_source.connected =='DC':
            return pyo.Constraint.Skip
        else:    
            return (model.P_renSource[rs]*model.gamma[rs])**2+model.Q_renSource[rs]**2 <= ren_source.Max_S**2 
    
    model.S_renS_AC_limit_constraint   = pyo.Constraint(model.ren_sources, rule=S_renS_AC_limit_rule)
    
    
   
    
    #AC lines inequality
    def S_to_AC_limit_rule(model,line):
        
        return model.PAC_to[line]**2+model.QAC_to[line]**2 <= S_lineAC_limit[line]**2
    def S_from_AC_limit_rule(model,line):
        
        return model.PAC_from[line]**2+model.QAC_from[line]**2 <= S_lineAC_limit[line]**2
    
    
    model.S_to_AC_limit_constraint   = pyo.Constraint(model.lines_AC, rule=S_to_AC_limit_rule)
    model.S_from_AC_limit_constraint = pyo.Constraint(model.lines_AC, rule=S_from_AC_limit_rule)
   
    
    def S_to_AC_limit_rule_tf(model,line):
        
        return model.tf_PAC_to[line]**2+model.tf_QAC_to[line]**2 <= S_lineACtf_limit[line]**2
    def S_from_AC_limit_rule_tf(model,line):
        
        return model.tf_PAC_from[line]**2+model.tf_QAC_from[line]**2 <= S_lineACtf_limit[line]**2
    
    if TAP_tf:
        model.tf_S_to_AC_limit_constraint   = pyo.Constraint(model.lines_AC_tf, rule=S_to_AC_limit_rule_tf)
        model.tf_S_from_AC_limit_constraint = pyo.Constraint(model.lines_AC_tf, rule=S_from_AC_limit_rule_tf)
    
    
    
    
    
    "DC inequality constraints"
    
    #they set in the variables themselves
    
    "Converters inequality constraints"
    
    def Conv_ACc_Limit_rule(model, conv):
        return (model.P_conv_c_AC[conv]**2+model.Q_conv_c_AC[conv]**2) <= (S_limit_conv[conv])**2 
    
    
    def Conv_ACs_Limit_rule(model, conv):
        return (model.P_conv_s_AC[conv]**2+model.Q_conv_s_AC[conv]**2) <= (S_limit_conv[conv])**2
    
    def Conv_DC_Limit_rule(model, conv):
        nDC = grid.Converters_ACDC[conv].Node_DC.nodeNumber
        return (model.P_conv_DC[nDC]**2) <= (P_conv_limit[nDC])**2
    
    
   #AC elements losses 
    def Conv_AC_loss1(model,conv):
        return  model.P_AC_loss_conv[conv] >= model.P_conv_c_AC[conv]-model.P_conv_s_AC[conv] 
    def Conv_AC_loss2(model,conv):
        return  model.P_AC_loss_conv[conv] >= model.P_conv_s_AC[conv]-model.P_conv_c_AC[conv]
    
    
    
    if not OnlyAC:
        model.Conv_ACc_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_ACc_Limit_rule)
        model.Conv_ACs_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_ACs_Limit_rule)
        model.Conv_DC_Limit_constraint = pyo.Constraint(model.conv, rule=Conv_DC_Limit_rule)
        model.Conv_AC_loss_constraint1= pyo.Constraint(model.conv,rule=Conv_AC_loss1)
        model.Conv_AC_loss_constraint2= pyo.Constraint(model.conv,rule=Conv_AC_loss2)
    
    
    
    return model




def ExportACDC_model_toPyflowACDC(model,grid,Price_Zones):
    if grid.nn_DC!=0:
        OnlyAC=False
    else:
        OnlyAC=True
    grid.V_AC = np.zeros(grid.nn_AC)
    grid.Theta_V_AC = np.zeros(grid.nn_AC)
    
    #AC bus
    V_AC_values     = {k: np.float64(pyo.value(v)) for k, v in model.V_AC.items()}
    theta_AC_values = {k: np.float64(pyo.value(v)) for k, v in model.thetha_AC.items()}
    
    PGi_opt_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_opt.items()}
    QGi_opt_values  = {k: np.float64(pyo.value(v)) for k, v in model.QGi_opt.items()}
    PGi_ren_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_ren.items()}
    QGi_ren_values  = {k: np.float64(pyo.value(v)) for k, v in model.QGi_ren.items()}
    
    if not OnlyAC:
        P_conv_AC_values= {k: np.float64(pyo.value(v)) for k, v in model.P_conv_AC.items()}
        Q_conv_AC_values= {k: np.float64(pyo.value(v)) for k, v in model.Q_conv_AC.items()}
    
    
    # Parallelize node processing
    def process_node_AC(node):
        nAC = node.nodeNumber
        node.V = V_AC_values[nAC]
        node.theta = theta_AC_values[nAC]
        
        node.PGi_opt = PGi_opt_values[nAC]
        node.QGi_opt = QGi_opt_values[nAC]
        node.PGi_ren = PGi_ren_values[nAC]
        node.QGi_ren = QGi_ren_values[nAC]
        
        if not OnlyAC:
            node.P_s = P_conv_AC_values[nAC]
            node.Q_s = Q_conv_AC_values[nAC]
        
        grid.V_AC[nAC] = node.V
        grid.Theta_V_AC[nAC] = node.theta
        
        
    with ThreadPoolExecutor() as executor:
        executor.map(process_node_AC, grid.nodes_AC)
    
    
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
        
    
      
    
    PGen_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_gen.items()}
    QGen_values  = {k: np.float64(pyo.value(v)) for k, v in model.QGi_gen.items()}
    gamma_values = {k: np.float64(pyo.value(v)) for k, v in model.gamma.items()}
    Qren_values  = {k: np.float64(pyo.value(v)) for k, v in model.Q_renSource.items()}
    
    def process_element(element):
        if hasattr(element, 'genNumber'):  # Generator
            element.PGen = PGen_values[element.genNumber]
            element.QGen = QGen_values[element.genNumber]
        elif hasattr(element, 'rsNumber'):  # Renewable Source
            element.gamma = gamma_values[element.rsNumber]
            element.Qren  = Qren_values[element.rsNumber]

    # Combine Generators and Renewable Sources into one iterable
    elements = grid.Generators + grid.RenSources
    
    # Parallelize processing
    with ThreadPoolExecutor() as executor:
        executor.map(process_element, elements)
        
    if Price_Zones:
        # Parallelize price zone processing
        
        def process_price_zone(m):
            nM = m.price_zone_num
            m.price = np.float64(pyo.value(model.price_zone_price[nM]))

        with ThreadPoolExecutor() as executor:
            executor.map(process_price_zone, grid.Price_Zones)

    grid.Line_AC_calc()    
    
    if OnlyAC:
         return    
    
    # Parallelize DC line processing
    def process_line_DC(line):
        line.np_line = np.float64(pyo.value(model.NumLinesDCP[line.lineNumber]))

    with ThreadPoolExecutor() as executor:
        executor.map(process_line_DC, grid.lines_DC)
    
    # DC nodes
    grid.V_DC = np.zeros(grid.nn_DC)  
    V_DC_values       = {k: np.float64(pyo.value(v)) for k, v in model.V_DC.items()}
    P_conv_DC_values  = {k: np.float64(pyo.value(v)) for k, v in model.P_conv_DC.items()}
    NumLinesDCP_values= {k: np.float64(pyo.value(v)) for k, v in model.NumLinesDCP.items()}
    
    # Parallelize DC node processing
    def process_node_DC(node):
        nDC = node.nodeNumber
        node.V        = V_DC_values[nDC]
        node.P        = P_conv_DC_values[nDC]
        node.P_INJ    = node.PGi - node.PLi + node.P
        grid.V_DC[nDC]= node.V

    with ThreadPoolExecutor() as executor:
        executor.map(process_node_DC, grid.nodes_DC)
    
    # converters

    P_conv_DC_conv_values= {k: np.float64(pyo.value(v)) for k, v in model.P_conv_DC.items()}
    P_conv_s_AC_values   = {k: np.float64(pyo.value(v)) for k, v in model.P_conv_s_AC.items()}
    Q_conv_s_AC_values   = {k: np.float64(pyo.value(v)) for k, v in model.Q_conv_s_AC.items()}
    P_conv_c_AC_values   = {k: np.float64(pyo.value(v)) for k, v in model.P_conv_c_AC.items()}
    Q_conv_c_AC_values   = {k: np.float64(pyo.value(v)) for k, v in model.Q_conv_c_AC.items()}
    P_conv_loss_values   = {k: np.float64(pyo.value(v)) for k, v in model.P_conv_loss.items()}
    Uc_values            = {k: np.float64(pyo.value(v)) for k, v in model.Uc.items()}
    Uf_values            = {k: np.float64(pyo.value(v)) for k, v in model.Uf.items()}
    
    NumConvP_values      = {k: np.float64(pyo.value(v)) for k, v in model.NumConvP.items()}
        
    
    # Parallelize converter processing
    def process_converter(conv):
        nconv = conv.ConvNumber
        conv.NumConvP  = NumConvP_values[nconv]
        conv.P_DC      = P_conv_DC_conv_values[conv.Node_DC.nodeNumber] * conv.NumConvP
        conv.P_AC      = P_conv_s_AC_values[nconv] * conv.NumConvP
        conv.Q_AC      = Q_conv_s_AC_values[nconv] * conv.NumConvP
        conv.Pc        = P_conv_c_AC_values[nconv] * conv.NumConvP
        conv.Qc        = Q_conv_c_AC_values[nconv] * conv.NumConvP
        conv.P_loss    = P_conv_loss_values[nconv] * conv.NumConvP
        conv.P_loss_tf = abs(conv.P_AC - conv.Pc)
        conv.U_c       = Uc_values[nconv]
        conv.U_f       = Uf_values[nconv]
        conv.U_s       = V_AC_values[conv.Node_AC.nodeNumber]
        conv.th_c      = Uc_values[nconv]
        conv.th_f      = Uf_values[nconv]
        conv.th_s      = theta_AC_values[conv.Node_AC.nodeNumber]
        

    with ThreadPoolExecutor() as executor:
        executor.map(process_converter, grid.Converters_ACDC)
        
    
    grid.Line_DC_calc()

