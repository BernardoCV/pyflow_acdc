# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:25:02 2024

@author: BernardoCastro
"""

import pyomo.environ as pyo
import numpy as np
from concurrent.futures import ThreadPoolExecutor

__all__ = [
    'OPF_create_LModel_ACDC',
    'ExportACDC_Lmodel_toPyflowACDC'
]


def OPF_create_LModel_ACDC(model,grid,PV_set,Price_Zones,TEP=False):
    from .ACDC_OPF import Translate_pyf_OPF 
    
    
    [AC_info,DC_info,Conv_info,Price_Zone_info,gen_info]=Translate_pyf_OPF(grid,Price_Zones=Price_Zones)
   
    Generation_variables(model,grid,gen_info,TEP)

    AC_variables(model,grid,AC_info,PV_set)


    if TEP:
        TEP_variables(model,grid)
    else:
        TEP_parameters(model,grid,AC_info,DC_info,Conv_info)


    AC_constraints(model,grid,AC_info)

    

def Generation_variables(model,grid,gen_info,TEP):
    gen_AC_info,gen_DC_info,P_renSource,lista_rs = gen_info
    lf,qf,fc,np_gen,lista_gen = gen_AC_info
    lf_DC,qf_DC,fc_DC,np_gen_DC,lista_gen_DC = gen_DC_info
    
    model.ren_sources= pyo.Set(initialize=lista_rs)
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
    
    
    grid.GPR = False
    
    if any(gen.np_gen_opf for gen in grid.Generators) and TEP:
        grid.GPR = True

    def P_Gen_bounds(model, g):
        gen = grid.Generators[g]
        return (gen.Min_pow_gen*gen.np_gen,gen.Max_pow_gen*gen.np_gen)
        
    def Q_Gen_bounds(model, g):
        gen = grid.Generators[g]
        return (gen.Min_pow_genR*gen.np_gen,gen.Max_pow_genR*gen.np_gen)
    
    def P_gen_ini(model,ngen):
        gen = grid.Generators[ngen]
        min_pow_gen = gen.Min_pow_gen * gen.np_gen
        ini=gen.Pset * gen.np_gen
        max_pow_gen = gen.Max_pow_gen * gen.np_gen
        if  min_pow_gen>ini:
            ini=min_pow_gen
        elif ini>max_pow_gen: 
            ini=max_pow_gen
        return (ini)
    
    def Q_gen_ini(model,ngen):
        gen = grid.Generators[ngen]
        min_pow_genR = gen.Min_pow_genR * gen.np_gen
        ini=gen.Qset * gen.np_gen
        max_pow_genR = gen.Max_pow_genR * gen.np_gen
        if min_pow_genR>ini:
            ini=min_pow_genR
        elif ini>max_pow_genR: 
            ini=max_pow_genR    
        return (ini)
    

    if grid.ACmode:
        model.gen_AC     = pyo.Set(initialize=lista_gen)
 
        if grid.GPR:
            model.PGi_gen = pyo.Var(model.gen_AC, initialize=P_gen_ini)
            model.QGi_gen = pyo.Var(model.gen_AC, initialize=Q_gen_ini) 
        else:
            model.PGi_gen = pyo.Var(model.gen_AC,bounds=P_Gen_bounds, initialize=P_gen_ini)
            model.QGi_gen = pyo.Var(model.gen_AC,bounds=Q_Gen_bounds, initialize=Q_gen_ini) 
    
    def P_Gen_bounds_DC(model, g):
        gen = grid.Generators_DC[g]
        return (gen.Min_pow_gen*gen.np_gen,gen.Max_pow_gen*gen.np_gen)
        
    def P_gen_ini_DC(model,ngen):
        gen = grid.Generators_DC[ngen]
        min_pow_gen = gen.Min_pow_gen * gen.np_gen
        ini=gen.Pset * gen.np_gen
        max_pow_gen = gen.Max_pow_gen * gen.np_gen
        if  min_pow_gen>ini:
            ini=min_pow_gen
        elif ini>max_pow_gen: 
            ini=max_pow_gen
        return (ini)
    
    if grid.DCmode:
        model.gen_DC     = pyo.Set(initialize=lista_gen_DC)    
    
        if grid.GPR:
            model.PGi_gen_DC = pyo.Var(model.gen_DC, initialize=P_gen_ini_DC)

        else:
            model.PGi_gen_DC = pyo.Var(model.gen_DC,bounds=P_Gen_bounds_DC, initialize=P_gen_ini_DC)
    
    model.lf = pyo.Param (model.gen_AC, initialize=lf, mutable=True)       
    s=1
def AC_variables(model,grid,AC_info,PV_set):

    AC_Lists,AC_nodes_info,AC_lines_info,EXP_info,REC_info,CT_info = AC_info
    
    
    lista_nodos_AC, lista_lineas_AC,lista_lineas_AC_tf,AC_slack, AC_PV = AC_Lists
    u_min_ac,u_max_ac,V_ini_AC,Theta_ini, P_know,Q_know,price = AC_nodes_info
    S_lineAC_limit,S_lineACtf_limit,m_tf_og = AC_lines_info

    lista_lineas_AC_exp,S_lineACexp_limit,NP_lineAC = EXP_info
    lista_lineas_AC_rec,S_lineACrec_lim,S_lineACrec_lim_new,grid.REC_AC_act = REC_info
    lista_lineas_AC_ct,S_lineACct_lim,cab_types_set,allowed_types = CT_info

    "Model Sets"
    model.nodes_AC   = pyo.Set(initialize=lista_nodos_AC)
    model.lines_AC   = pyo.Set(initialize=lista_lineas_AC)
    
    if grid.TEP_AC:
        model.lines_AC_exp = pyo.Set(initialize=lista_lineas_AC_exp)
    if grid.REC_AC:
        model.lines_AC_rec = pyo.Set(initialize=lista_lineas_AC_rec)
    if grid.CT_AC:
        model.lines_AC_ct = pyo.Set(initialize=lista_lineas_AC_ct)
    
    
    
    model.AC_slacks  = pyo.Set(initialize=AC_slack)
    
            
    "AC Variables"
    #AC nodes variables
    model.thetha_AC  = pyo.Var(model.nodes_AC, bounds=(-1.6, 1.6), initialize=Theta_ini)

    model.P_known_AC = pyo.Param(model.nodes_AC, initialize=P_know,mutable=True)
        
    def Pren_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_RenSource == []:
            return (0,0)
        else:
            return (None,None)
    
    
    model.PGi_ren = pyo.Var(model.nodes_AC, bounds=Pren_bounds,initialize=0)
    
    def PGi_opt_bounds(model, node):
        nAC = grid.nodes_AC[node]
        if nAC.connected_gen == []:
            return (0,0)
        else:
            return (None,None)
            
    model.PGi_opt = pyo.Var(model.nodes_AC,bounds=PGi_opt_bounds ,initialize=0)

    def make_opt_bounds(attribute_name):
        def bounds_func(model, node):
            nAC = grid.nodes_AC[node]
            connected_lines = getattr(nAC, attribute_name)
            return (0, 0) if not connected_lines else (None, None)
        return bounds_func

    # Create bounds functions dynamically
    toExp_opt_bounds    = make_opt_bounds('connected_toExpLine')
    fromExp_opt_bounds  = make_opt_bounds('connected_fromExpLine')
    toTF_opt_bounds     = make_opt_bounds('connected_toTFLine')
    fromTF_opt_bounds   = make_opt_bounds('connected_fromTFLine')
    toREC_opt_bounds    = make_opt_bounds('connected_toRepLine')
    fromREC_opt_bounds  = make_opt_bounds('connected_fromRepLine')
    toCT_opt_bounds     = make_opt_bounds('connected_toCTLine')
    fromCT_opt_bounds   = make_opt_bounds('connected_fromCTLine')

    if grid.TEP_AC:
        model.Pto_Exp  = pyo.Var(model.nodes_AC,bounds=toExp_opt_bounds ,initialize=0)
        model.Pfrom_Exp= pyo.Var(model.nodes_AC,bounds=fromExp_opt_bounds ,initialize=0)
    
    if grid.REC_AC:
        model.Pto_REP   = pyo.Var(model.nodes_AC,bounds=toREC_opt_bounds ,initialize=0)
        model.Pfrom_REP = pyo.Var(model.nodes_AC,bounds=fromREC_opt_bounds ,initialize=0)
    
    if grid.CT_AC:
        model.Pto_CT   = pyo.Var(model.nodes_AC,bounds=toCT_opt_bounds ,initialize=0)
        model.Pfrom_CT = pyo.Var(model.nodes_AC,bounds=fromCT_opt_bounds ,initialize=0)
            
    def AC_theta_slack_rule(model, node):
        return model.thetha_AC[node] == Theta_ini[node]

    model.AC_theta_slack_constraint = pyo.Constraint(model.AC_slacks, rule=AC_theta_slack_rule)
    
    #AC Lines variables
    def Sbounds_lines(model, line):
        return (-S_lineAC_limit[line], S_lineAC_limit[line])
    
    
    model.PAC_to       = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.PAC_from     = pyo.Var(model.lines_AC, bounds=Sbounds_lines, initialize=0)
    model.PAC_line_loss= pyo.Var(model.lines_AC, initialize=0)

    def Sbounds_lines_exp(model, line):
        return (-S_lineACexp_limit[line], S_lineACexp_limit[line])
   
    if grid.TEP_AC:
        model.exp_PAC_to       = pyo.Var(model.lines_AC_exp, bounds=Sbounds_lines_exp, initialize=0)
        model.exp_PAC_from     = pyo.Var(model.lines_AC_exp, bounds=Sbounds_lines_exp, initialize=0)    
        model.exp_PAC_line_loss= pyo.Var(model.lines_AC_exp, initialize=0)
    
    
    def state_based_bounds(model, line, state):
            max_min = max(S_lineACrec_lim[line], S_lineACrec_lim_new[line])
            return (-max_min, max_min)
            
    if grid.REC_AC:
        # Define a set for the branch states (0=old, 1=new)
        model.branch_states = pyo.Set(initialize=[0, 1])
        
        # Single variable for all power flows with two indices
        model.rec_PAC_to   = pyo.Var(model.lines_AC_rec,model.branch_states,bounds=state_based_bounds,initialize=0)
        model.rec_PAC_from = pyo.Var(model.lines_AC_rec,model.branch_states,bounds=state_based_bounds,initialize=0)
        model.rec_PAC_line_loss = pyo.Var(model.lines_AC_rec,initialize=0)
    
    def set_based_bounds(model, line, cab_type):
         max_min = max(S_lineACct_lim[line,ct] for ct in cab_types_set)  # Use cab_types_set instead of model.ct_set
         return (-max_min, max_min)
       
    
    if grid.CT_AC:

        model.ct_set = pyo.Set(initialize=cab_types_set)
        model.ct_PAC_to   = pyo.Var(model.lines_AC_ct,model.ct_set,bounds=set_based_bounds,initialize=0)
        model.ct_PAC_from = pyo.Var(model.lines_AC_ct,model.ct_set,bounds=set_based_bounds,initialize=0)
        
        model.z_to = pyo.Var(model.lines_AC_ct, model.ct_set, bounds=set_based_bounds)
        model.z_from = pyo.Var(model.lines_AC_ct, model.ct_set, bounds=set_based_bounds)   

def AC_constraints(model,grid,AC_info):
    
    
    AC_Lists,AC_nodes_info,AC_lines_info,EXP_info,REC_info,CT_info = AC_info
    S_lineAC_limit,S_lineACtf_limit,m_tf_og = AC_lines_info

    lista_lineas_AC_exp,S_lineACexp_limit,NP_lineAC = EXP_info
    lista_lineas_AC_rec,S_lineACrec_lim,S_lineACrec_lim_new,grid.REC_AC_act = REC_info
    lista_lineas_AC_ct,S_lineACct_lim,cab_types_set,allowed_types = CT_info

    "AC equality constraints"
    # AC node constraints
    def P_AC_node_rule(model, node):
        P_sum = sum(
            -np.imag(grid.Ybus_AC[node, k]) * (model.thetha_AC[node] - model.thetha_AC[k])
            for k in model.nodes_AC if grid.Ybus_AC[node, k] != 0
        )
        P_var = model.P_known_AC[node] + model.PGi_ren[node] + model.PGi_opt[node]

        if grid.TEP_AC:
            P_sum += model.Pto_Exp[node]+model.Pfrom_Exp[node]
        if grid.REC_AC:
            P_sum += model.Pto_REP[node]+model.Pfrom_REP[node]
        if grid.CT_AC:
            P_sum += model.Pto_CT[node]+model.Pfrom_CT[node]
        
        return P_sum == P_var


    model.P_AC_node_constraint = pyo.Constraint(model.nodes_AC, rule=P_AC_node_rule)
    
    # Adds all generators in the AC nodes they are connected to
    def Gen_PAC_rule(model,node):
       nAC = grid.nodes_AC[node]
       P_gen = sum(model.PGi_gen[gen.genNumber] for gen in nAC.connected_gen)                  
       return  model.PGi_opt[node] ==   P_gen
           
    model.Gen_PAC_constraint = pyo.Constraint(model.nodes_AC, rule=Gen_PAC_rule)
    
    def Gen_PREN_rule(model,node):
       nAC = grid.nodes_AC[node]
       P_gen = sum(model.P_renSource[rs.rsNumber]*model.gamma[rs.rsNumber] for rs in nAC.connected_RenSource)                  
       return  model.PGi_ren[node] ==   P_gen
   
    model.Gen_PREN_constraint =pyo.Constraint(model.nodes_AC, rule=Gen_PREN_rule)
    
    
    def toPexp_rule(model,node):
       nAC = grid.nodes_AC[node]
       toPexp = sum(model.exp_PAC_to[l.lineNumber]*model.NumLinesACP[l.lineNumber] for l in nAC.connected_toExpLine)                  
       return  model.Pto_Exp[node] ==  toPexp
    def fromPexp_rule(model,node):
       nAC = grid.nodes_AC[node]
       fromPexp = sum(model.exp_PAC_from[l.lineNumber]*model.NumLinesACP[l.lineNumber] for l in nAC.connected_fromExpLine)                
       return  model.Pfrom_Exp[node] ==   fromPexp
    
    
    if grid.TEP_AC:
        model.exp_Pto_constraint  = pyo.Constraint(model.nodes_AC, rule=toPexp_rule)
        model.exp_Pfrom_constraint= pyo.Constraint(model.nodes_AC, rule=fromPexp_rule)
       
   
    def toPre_rule(model,node):
       nAC = grid.nodes_AC[node]
       toPre = sum(model.rec_PAC_to[l.lineNumber,0]*(1-model.rec_branch[l.lineNumber])+model.rec_PAC_to[l.lineNumber,1]*model.rec_branch[l.lineNumber] for l in nAC.connected_toRepLine)                  
       return  model.Pto_REP[node] ==  toPre
    def fromPre_rule(model,node):
       nAC = grid.nodes_AC[node]
       fromPre = sum(model.rec_PAC_from[l.lineNumber,0]*(1-model.rec_branch[l.lineNumber])+model.rec_PAC_from[l.lineNumber,1]*model.rec_branch[l.lineNumber] for l in nAC.connected_fromRepLine)                
       return  model.Pfrom_REP[node] ==   fromPre
    
    
    if grid.REC_AC:
        model.rec_Pto_constraint  = pyo.Constraint(model.nodes_AC, rule=toPre_rule)
        model.rec_Pfrom_constraint= pyo.Constraint(model.nodes_AC, rule=fromPre_rule)
        
    # Fix the node constraints to use auxiliary variables:
    def toCT_rule_linear(model,node):
       nAC = grid.nodes_AC[node]
       toPre = 0
       for line in nAC.connected_toCTLine:
           for ct in model.ct_set:
               toPre += model.z_to[line.lineNumber,ct]  # ✅ Use z_to instead of bilinear term
       return model.Pto_CT[node] == toPre

    def fromCT_rule_linear(model,node):
       nAC = grid.nodes_AC[node]
       fromPre = 0
       for line in nAC.connected_fromCTLine:
           for ct in model.ct_set:
               fromPre += model.z_from[line.lineNumber,ct]  # ✅ Use z_from instead of bilinear term
       return model.Pfrom_CT[node] == fromPre


    def z_to_ub_rule(model, line, ct):
        return model.z_to[line, ct] <= model.ct_PAC_to[line, ct] + (1 - model.ct_branch[line, ct]) * (2*S_lineACct_lim[line,ct])

    def z_to_lb_rule(model, line, ct):
        return model.z_to[line, ct] >= model.ct_PAC_to[line, ct] - (1 - model.ct_branch[line, ct]) * (2*S_lineACct_lim[line,ct])

    def z_to_branch_ub_rule(model, line, ct):
        return model.z_to[line, ct] <= model.ct_branch[line, ct] * S_lineACct_lim[line,ct]

    def z_to_branch_lb_rule(model, line, ct):
        return model.z_to[line, ct] >= model.ct_branch[line, ct] * (-S_lineACct_lim[line,ct])

    
    def z_from_ub_rule(model, line, ct):
        return model.z_from[line, ct] <= model.ct_PAC_from[line, ct] + (1 - model.ct_branch[line, ct]) * (2*S_lineACct_lim[line,ct])

    def z_from_lb_rule(model, line, ct):
        return model.z_from[line, ct] >= model.ct_PAC_from[line, ct] - (1 - model.ct_branch[line, ct]) * (2*S_lineACct_lim[line,ct])

    def z_from_branch_ub_rule(model, line, ct):
        return model.z_from[line, ct] <= model.ct_branch[line, ct] * S_lineACct_lim[line,ct]

    def z_from_branch_lb_rule(model, line, ct):
        return model.z_from[line, ct] >= model.ct_branch[line, ct] * (-S_lineACct_lim[line,ct])

    
    if grid.CT_AC:
        model.ct_Pto_constraint = pyo.Constraint(model.nodes_AC, rule=toCT_rule_linear)  
        model.ct_Pfrom_constraint = pyo.Constraint(model.nodes_AC, rule=fromCT_rule_linear)  
        
        # McCormick envelopes for z_to
        model.z_to_ub_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_to_ub_rule)
        model.z_to_lb_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_to_lb_rule)
        model.z_to_branch_ub_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_to_branch_ub_rule)
        model.z_to_branch_lb_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_to_branch_lb_rule)
        
        # McCormick envelopes for z_from
        model.z_from_ub_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_from_ub_rule)
        model.z_from_lb_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_from_lb_rule)
        model.z_from_branch_ub_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_from_branch_ub_rule)
        model.z_from_branch_lb_con = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=z_from_branch_lb_rule)

    # AC line equality constraints
    def calculate_P(model, line, direction,idx=None):
        f = line.fromNode.nodeNumber
        t = line.toNode.nodeNumber
        
        if idx is None:
            Ybus = line.Ybus_branch
        elif idx == 'new':
            Ybus = line.Ybus_branch_new
        else:
            Ybus = line.Ybus_list[idx]
        thf=model.thetha_AC[f]
        tht=model.thetha_AC[t]
        if direction == 'to':
            B = np.imag(Ybus[1,0])  # Btf
            P = -B * (tht - thf)
        else:  # 'from'
            B = np.imag(Ybus[0,1])  # Bft
            P = -B * (thf - tht)
        return P


    def P_to_AC_line(model,line):   
        l = grid.lines_AC[line]
        Pto = calculate_P(model,l,'to')
        return model.PAC_to[line] == Pto
    
    def P_from_AC_line(model,line):       
       l = grid.lines_AC[line]
       Pfrom = calculate_P(model,l,'from')
       return model.PAC_from[line] == Pfrom
    
 
    model.Pto_AC_line_constraint   = pyo.Constraint(model.lines_AC, rule=P_to_AC_line)
    model.Pfrom_AC_line_constraint = pyo.Constraint(model.lines_AC, rule=P_from_AC_line)
 
    def P_to_AC_line_exp(model,line):   
        l = grid.lines_AC_exp[line]
        Pto = calculate_P(model,l,'to')
        return model.exp_PAC_to[line] == Pto
    
    def P_from_AC_line_exp(model,line):       
       l = grid.lines_AC_exp[line]
       Pfrom = calculate_P(model,l,'from')
       return model.exp_PAC_from[line] == Pfrom
    
 
    
    if grid.TEP_AC:
        model.exp_Pto_AC_line_constraint   = pyo.Constraint(model.lines_AC_exp, rule=P_to_AC_line_exp)
        model.exp_Pfrom_AC_line_constraint = pyo.Constraint(model.lines_AC_exp, rule=P_from_AC_line_exp)
 
    def P_to_AC_line_rec(model,line,state):   
        l = grid.lines_AC_rec[line]
        if state ==  0:
            Pto = calculate_P(model,l,'to')
        else:
            Pto = calculate_P(model,l,'to',idx='new')
        return model.rec_PAC_to[line,state] == Pto
    
    def P_from_AC_line_rec(model,line,state):       
       l = grid.lines_AC_rec[line]
       if state == 0:
           Pfrom = calculate_P(model,l,'from')
       else:
           Pfrom = calculate_P(model,l,'from',idx='new')
       return model.rec_PAC_from[line,state] == Pfrom
    

    
    if grid.REC_AC:
     
        model.rec_Pto_AC_line_constraint = pyo.Constraint( model.lines_AC_rec, model.branch_states, rule=P_to_AC_line_rec)
        model.rec_Pfrom_AC_line_constraint = pyo.Constraint( model.lines_AC_rec, model.branch_states, rule=P_from_AC_line_rec)
       
    def P_to_AC_line_ct(model,line,ct):   
        l = grid.lines_AC_ct[line]
        Pto = calculate_P(model,l,'to',idx=ct)
        return model.ct_PAC_to[line,ct] == Pto
    
    def P_from_AC_line_ct(model,line,ct):       
       l = grid.lines_AC_ct[line]
       Pfrom = calculate_P(model,l,'from',idx=ct)
       return model.ct_PAC_from[line,ct] == Pfrom
    
 

    if grid.CT_AC:   
        model.ct_Pto_AC_line_constraint = pyo.Constraint( model.lines_AC_ct, model.ct_set, rule=P_to_AC_line_ct)
        model.ct_Pfrom_AC_line_constraint = pyo.Constraint( model.lines_AC_ct, model.ct_set, rule=P_from_AC_line_ct)
      
    
    
    "AC inequality constraints"
    #AC gen inequality
   
    
    def calc_M_rec_linear(model, line):
        max_pow = max(S_lineACrec_lim[line], S_lineACrec_lim_new[line])
        return 1.1 * max_pow

    def S_to_AC_line_rule_rec_linear(model, line, state):
        M = calc_M_rec_linear(model, line)
        if state == 0:
            return model.rec_PAC_to[line, 0] <= S_lineACrec_lim[line] + M * model.rec_branch[line]
        else:
            return model.rec_PAC_to[line, 1] <= S_lineACrec_lim_new[line] + M * (1 - model.rec_branch[line])

    def S_to_AC_line_rule_rec_linear_neg(model, line, state):
        M = calc_M_rec_linear(model, line)
        if state == 0:
            return model.rec_PAC_to[line, 0] >= -S_lineACrec_lim[line] - M * model.rec_branch[line]
        else:
            return model.rec_PAC_to[line, 1] >= -S_lineACrec_lim_new[line] - M * (1 - model.rec_branch[line])

    def S_from_AC_limit_rule_rec_linear(model, line, state):
        M = calc_M_rec_linear(model, line)
        if state == 0:
            return model.rec_PAC_from[line, 0] <= S_lineACrec_lim[line] + M * model.rec_branch[line]
        else:
            return model.rec_PAC_from[line, 1] <= S_lineACrec_lim_new[line] + M * (1 - model.rec_branch[line])

    def S_from_AC_limit_rule_rec_linear_neg(model, line, state):
        M = calc_M_rec_linear(model, line)
        if state == 0:
            return model.rec_PAC_from[line, 0] >= -S_lineACrec_lim[line] - M * model.rec_branch[line]
        else:
            return model.rec_PAC_from[line, 1] >= -S_lineACrec_lim_new[line] - M * (1 - model.rec_branch[line])

    if grid.REC_AC:
        model.rec_S_to_AC_limit_constraint_upper = pyo.Constraint(model.lines_AC_rec, model.branch_states, rule=S_to_AC_line_rule_rec_linear)
        model.rec_S_to_AC_limit_constraint_lower = pyo.Constraint(model.lines_AC_rec, model.branch_states, rule=S_to_AC_line_rule_rec_linear_neg)
        model.rec_S_from_AC_limit_constraint_upper = pyo.Constraint(model.lines_AC_rec, model.branch_states, rule=S_from_AC_limit_rule_rec_linear)
        model.rec_S_from_AC_limit_constraint_lower = pyo.Constraint(model.lines_AC_rec, model.branch_states, rule=S_from_AC_limit_rule_rec_linear_neg)
    
    def calc_M_linear(model, line):
        max_pow = max(S_lineACct_lim[line,ct] for ct in model.ct_set)
        return 1.1 * max_pow

    def S_to_AC_line_rule_ct_linear(model, line, ct):
        M = calc_M_linear(model, line)
        return model.ct_PAC_to[line,ct] <= S_lineACct_lim[line,ct] * model.ct_branch[line,ct] + M * (1 - model.ct_branch[line,ct])

    def S_to_AC_line_rule_ct_linear_neg(model, line, ct):
        M = calc_M_linear(model, line)
        return model.ct_PAC_to[line,ct] >= -S_lineACct_lim[line,ct] * model.ct_branch[line,ct] - M * (1 - model.ct_branch[line,ct])

    def S_from_AC_limit_rule_ct_linear(model, line, ct):
        M = calc_M_linear(model, line)
        return model.ct_PAC_from[line,ct] <= S_lineACct_lim[line,ct] * model.ct_branch[line,ct] + M * (1 - model.ct_branch[line,ct])

    def S_from_AC_limit_rule_ct_linear_neg(model, line, ct):
        M = calc_M_linear(model, line)
        return model.ct_PAC_from[line,ct] >= -S_lineACct_lim[line,ct] * model.ct_branch[line,ct] - M * (1 - model.ct_branch[line,ct])

    if grid.CT_AC:
        model.ct_S_to_AC_limit_constraint_upper = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=S_to_AC_line_rule_ct_linear)
        model.ct_S_to_AC_limit_constraint_lower = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=S_to_AC_line_rule_ct_linear_neg)
        model.ct_S_from_AC_limit_constraint_upper = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=S_from_AC_limit_rule_ct_linear)
        model.ct_S_from_AC_limit_constraint_lower = pyo.Constraint(model.lines_AC_ct, model.ct_set, rule=S_from_AC_limit_rule_ct_linear_neg)

    s=1
    
def TEP_parameters(model,grid,AC_info,DC_info,Conv_info):
    
    
    
    from .ACDC_TEP import get_TEP_variables

    conv_var,DC_line_var,AC_line_var,gen_var = get_TEP_variables(grid)

    NumConvP,NumConvP_i,NumConvP_max,S_limit_conv = conv_var
    P_lineDC_limit,NP_lineDC,NP_lineDC_i,NP_lineDC_max,Line_length = DC_line_var
    NP_lineAC,NP_lineAC_i,NP_lineAC_max,Line_length,REC_branch,ct_ini = AC_line_var
    np_gen,np_gen_max,np_gen_DC,np_gen_max_DC = gen_var

    
    model.np_gen = pyo.Param(model.gen_AC,initialize=np_gen)
    if grid.TEP_AC:    
        model.NumLinesACP = pyo.Param(model.lines_AC_exp ,initialize=NP_lineAC)    

    if grid.REC_AC:
        model.rec_branch = pyo.Param(model.lines_AC_rec,initialize=REC_branch)
    
    if grid.CT_AC:
        if not grid.Array_opf:
            model.ct_branch = pyo.Param(model.lines_AC_ct,model.ct_set,initialize=ct_ini)
        else:
            model.ct_branch = pyo.Param(model.lines_AC_ct,model.ct_set,initialize=0)



def TEP_variables(model,grid):

    from .ACDC_TEP import get_TEP_variables

    conv_var,DC_line_var,AC_line_var,gen_var = get_TEP_variables(grid)

    NumConvP,NumConvP_i,NumConvP_max,S_limit_conv = conv_var
    P_lineDC_limit,NP_lineDC,NP_lineDC_i,NP_lineDC_max,Line_length = DC_line_var
    NP_lineAC,NP_lineAC_i,NP_lineAC_max,Line_length,REC_branch,ct_ini = AC_line_var
    np_gen,np_gen_max,np_gen_DC,np_gen_max_DC = gen_var    

    
    "TEP variables"
    
    def np_gen_bounds(model,gen):
        g = grid.Generators[gen]
        if g.np_gen_opf:
            return (np_gen[gen],np_gen_max[gen])
        else:
            return (np_gen[gen],np_gen[gen])

    if grid.GPR:

        def P_gen_lower_bound_rule(model, gen):
            g = grid.Generators[gen]
            return (g.Min_pow_gen * model.np_gen[gen] <= model.PGi_gen[gen])

        def P_gen_upper_bound_rule(model, gen):
            g = grid.Generators[gen]
            return (model.PGi_gen[gen] <= g.Max_pow_gen * model.np_gen[gen])

        

        model.np_gen = pyo.Var(model.gen_AC,within=pyo.NonNegativeIntegers,bounds=np_gen_bounds,initialize=np_gen)
        model.np_gen_base = pyo.Param(model.gen_AC,initialize=np_gen)  

        model.PGi_lower_bound = pyo.Constraint(model.gen_AC,rule=P_gen_lower_bound_rule)
        model.PGi_upper_bound = pyo.Constraint(model.gen_AC,rule=P_gen_upper_bound_rule)


    else:
        model.np_gen = pyo.Param(model.gen_AC,initialize=np_gen)
        

    if grid.TEP_AC:
        def NPline_bounds_AC(model, line):
            element=grid.lines_AC_exp[line]
            if not element.np_line_opf:
                return (NP_lineAC[line], NP_lineAC[line])
            else:
                return (NP_lineAC[line], NP_lineAC_max[line])
        
        model.NumLinesACP = pyo.Var(model.lines_AC_exp, within=pyo.NonNegativeIntegers,bounds=NPline_bounds_AC,initialize=NP_lineAC_i)
        model.NumLinesACP_base  =pyo.Param(model.lines_AC_exp,initialize=NP_lineAC)

    if grid.REC_AC:
        model.rec_branch = pyo.Var(model.lines_AC_rec,domain=pyo.Binary,initialize=REC_branch)

    if grid.CT_AC:
        model.ct_branch = pyo.Var(model.lines_AC_ct,model.ct_set,domain=pyo.Binary,initialize=ct_ini)
        model.ct_types = pyo.Var(model.ct_set,domain=pyo.Binary,initialize=0)


def ExportACDC_Lmodel_toPyflowACDC(model,grid,Price_Zones,TEP=False):

    
    grid.OPF_run=True

    #Generation 
    if grid.ACmode:
        PGen_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_gen.items()}
        QGen_values = {k: 0.0 for k in PGen_values.keys()}
    
    gamma_values = {k: np.float64(pyo.value(v)) for k, v in model.gamma.items()}
    Qren_values  = {k: 0.0 for k in gamma_values.keys()}
    
    def process_element(element):
        if hasattr(element, 'genNumber'):  # Generator
            element.PGen = PGen_values[element.genNumber]
            element.QGen = QGen_values[element.genNumber]
        elif hasattr(element, 'rsNumber'):  # Renewable Source
            element.gamma = gamma_values[element.rsNumber]
            element.QGi_ren  = Qren_values[element.rsNumber]

    # Combine Generators and Renewable Sources into one iterable
    elements = grid.Generators + grid.RenSources 

    # Parallelize processing
    with ThreadPoolExecutor() as executor:
        executor.map(process_element, elements)
   
   
    #AC bus
            
    grid.V_AC = np.ones(grid.nn_AC)
    grid.Theta_V_AC = np.zeros(grid.nn_AC)

    
    theta_AC_values = {k: np.float64(pyo.value(v)) for k, v in model.thetha_AC.items()}
    V_AC_values     = {k: 1.0 for k in theta_AC_values.keys()}
    PGi_opt_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_opt.items()}
    QGi_opt_values  = {k: 0.0 for k in PGi_opt_values.keys()}
    PGi_ren_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_ren.items()}
    QGi_ren_values  = {k: 0.0 for k in PGi_ren_values.keys()}
    
    # Parallelize node processing
    def process_node_AC(node):
        nAC = node.nodeNumber
        node.V = V_AC_values[nAC]
        node.theta = theta_AC_values[nAC]
        
        node.PGi_opt = PGi_opt_values[nAC]
        node.QGi_opt = QGi_opt_values[nAC]
        node.PGi_ren = PGi_ren_values[nAC]
        node.QGi_ren = QGi_ren_values[nAC]
        
        grid.Theta_V_AC[nAC] = node.theta
        
        
    with ThreadPoolExecutor() as executor:
        executor.map(process_node_AC, grid.nodes_AC)
    
    
    Pf = np.zeros((grid.nn_AC, 1))
    Qf = np.zeros((grid.nn_AC, 1))
    
    B = np.imag(grid.Ybus_AC)
    Theta = grid.Theta_V_AC

    Theta_diff = Theta[:, None] - Theta
    Pf_DC = (-B * Theta_diff).sum(axis=1)
    # Reactive power is not modeled in DC approximation
    Qf_DC = np.zeros_like(Pf_DC)  

    for node in grid.nodes_AC:
        i = node.nodeNumber
        node.P_INJ = Pf_DC[i]
        node.Q_INJ = Qf_DC[i]
        
    if grid.GPR:
        np_gen_values = {k: np.float64(pyo.value(v)) for k, v in model.np_gen.items()}
        for gen in grid.Generators:
            gen.np_gen = np_gen_values[gen.genNumber]
    
    if grid.TEP_AC:
        lines_AC_TEP = {k: np.float64(pyo.value(v)) for k, v in model.NumLinesACP.items()}
        lines_AC_TEP_fromP = {k: np.float64(pyo.value(v)) for k, v in model.exp_PAC_from.items()}
        lines_AC_TEP_toP = {k: np.float64(pyo.value(v)) for k, v in model.exp_PAC_to.items()}
        lines_AC_TEP_fromQ = {k: 0.0 for k in lines_AC_TEP_fromP.keys()}
        lines_AC_TEP_toQ = {k: 0.0 for k in lines_AC_TEP_toP.keys()}
        lines_AC_TEP_P_loss = {k: np.float64(pyo.value(v)) for k, v in model.exp_PAC_line_loss.items()}

        def process_line_AC_TEP(line):
            l = line.lineNumber
            line.np_line = lines_AC_TEP[l]
            line.P_loss = lines_AC_TEP_P_loss[l]*lines_AC_TEP[l]
            line.fromS = (lines_AC_TEP_fromP[l] + 1j*lines_AC_TEP_fromQ[l])*lines_AC_TEP[l]
            line.toS = (lines_AC_TEP_toP[l] + 1j*lines_AC_TEP_toQ[l])*lines_AC_TEP[l]
            line.loss = line.fromS + line.toS

        with ThreadPoolExecutor() as executor:
            executor.map(process_line_AC_TEP, grid.lines_AC_exp)

    if grid.REC_AC:
        lines_AC_REP = {k: np.float64(pyo.value(v)) for k, v in model.rec_branch.items()}
        lines_AC_REC_fromP = {k: {state: np.float64(pyo.value(model.rec_PAC_from[k, state])) for state in model.branch_states} for k in model.lines_AC_rec}
        lines_AC_REC_toP = {k: {state: np.float64(pyo.value(model.rec_PAC_to[k, state])) for state in model.branch_states} for k in model.lines_AC_rec}
        lines_AC_REC_fromQ = {k: {state: 0.0 for state in model.branch_states} for k in model.lines_AC_rec}
        lines_AC_REC_toQ = {k: {state: 0.0 for state in model.branch_states} for k in model.lines_AC_rec}
        lines_AC_REC_P_loss = {k: np.float64(pyo.value(v)) for k, v in model.rec_PAC_line_loss.items()}
        
        
        def process_line_AC_REP(line):
            l = line.lineNumber
            line.rec_branch = True if lines_AC_REP[l] >= 0.99999 else False
            line.P_loss = lines_AC_REC_P_loss[l]
            state = 1 if line.rec_branch else 0
            line.fromS = (lines_AC_REC_fromP[l][state] + 1j*lines_AC_REC_fromQ[l][state])
            line.toS = (lines_AC_REC_toP[l][state] + 1j*lines_AC_REC_toQ[l][state])
            line.loss = line.fromS + line.toS

        with ThreadPoolExecutor() as executor:
            executor.map(process_line_AC_REP, grid.lines_AC_rec)    

    if grid.CT_AC:   
        lines_AC_CT = {k: {ct: np.float64(pyo.value(model.ct_branch[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
        lines_AC_CT_fromP = {k: {ct: np.float64(pyo.value(model.ct_PAC_from[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
        lines_AC_CT_toP = {k: {ct: np.float64(pyo.value(model.ct_PAC_to[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
        lines_AC_CT_fromQ = {k: {ct: 0.0 for ct in model.ct_set} for k in model.lines_AC_ct}
        lines_AC_CT_toQ = {k: {ct: 0.0 for ct in model.ct_set} for k in model.lines_AC_ct}
       
        
        def process_line_AC_CT(line):
            l = line.lineNumber
            ct_selected = [lines_AC_CT[l][ct] >= 0.90  for ct in model.ct_set]
            if any(ct_selected):
                line.active_config = np.where(ct_selected)[0][0]
                ct = list(model.ct_set)[line.active_config]
                Pfrom = lines_AC_CT_fromP[l][ct]
                Pto   = lines_AC_CT_toP[l][ct]
                Qfrom = lines_AC_CT_fromQ[l][ct]
                Qto   = lines_AC_CT_toQ[l][ct]
            else:
                line.active_config = -1
                Pfrom = 0
                Pto   = 0
                Qfrom = 0
                Qto   = 0
            
            
            
            line.fromS = (Pfrom + 1j*Qfrom)
            line.toS = (Pto + 1j*Qto)
            line.loss = line.fromS + line.toS
            line.P_loss = 0

        with ThreadPoolExecutor() as executor:
            executor.map(process_line_AC_CT, grid.lines_AC_ct)
        #for line in grid.lines_AC_ct:
        #    process_line_AC_CT(line)
        
    # --- Step 1: Use voltage angles only ---
    Theta = grid.Theta_V_AC  # should be a 1D array with angle values in radians

    # --- Step 2: Iterate over lines and compute power flows ---
    for line in grid.lines_AC:
        i = line.fromNode.nodeNumber
        j = line.toNode.nodeNumber
        
        # Susceptance from Ybus (assuming purely imaginary admittance)
        B = -np.imag(line.Ybus_branch[0, 1])  # or [1,0] — symmetric for passive branches

        # Active power flow from i to j (DC approximation)
        P_ij = B * (Theta[i] - Theta[j])
        P_ji = B * (Theta[j] - Theta[i])

        # Store active powers
        line.fromP = P_ij
        line.toP = P_ji
        line.toS = P_ji + 1j*0
        line.fromS = P_ij + 1j*0
        # Loss is zero in DC model
        line.P_loss = 0
        line.loss = 0

        # Approximate current magnitude (linearized)
        line.i_from = abs(P_ij)  # or just set = P_ij if signed current
        line.i_to = abs(P_ji)
    