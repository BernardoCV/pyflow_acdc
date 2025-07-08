"""
Created on Thu Feb 15 13:24:05 2024

@author: BernardoCastro
"""
import numpy as np
import pyomo.environ as pyo

import time
from concurrent.futures import ThreadPoolExecutor
import re

from  .ACDC_OPF_model import *

import cProfile
import pstats
from io import StringIO



import logging
from pyomo.util.infeasible import log_infeasible_constraints

__all__ = [
    'Translate_pyf_OPF',
    'Optimal_PF',
    'TS_parallel_OPF',
    'OPF_solve',
    'OPF_updateParam',
    'OPF_obj',
    'OPF_line_res',
    'OPF_price_priceZone',
    'OPF_conv_results',
    'fx_conv'
]

def pack_variables(*args):
    return args
           
            

def obj_w_rule(grid,ObjRule,OnlyGen):
    weights_def = {
       'Ext_Gen': {'w': 0},
       'Energy_cost': {'w': 0},
       'Curtailment_Red': {'w': 0},
       'AC_losses': {'w': 0},
       'DC_losses': {'w': 0},
       'Converter_Losses': {'w': 0},
       'General_Losses': {'w': 0},
       'PZ_cost_of_generation': {'w': 0},
       'Renewable_profit': {'w': 0},
       'Gen_set_dev': {'w': 0}
    }

    # If user provides specific weights, merge them with the default
    if ObjRule is not None:
       for key in ObjRule:
           if key in weights_def:
               weights_def[key]['w'] = ObjRule[key]

    if OnlyGen == False:
        grid.OnlyGen=False
    Price_Zones = False
    if  weights_def['PZ_cost_of_generation']['w']!=0 :
        Price_Zones=True
    if  weights_def['Curtailment_Red']['w']!=0 :
        grid.CurtCost=True

    return weights_def, Price_Zones

def Optimal_PF(grid,ObjRule=None,PV_set=False,OnlyGen=True,Price_Zones=False):
    analyse_OPF(grid)

    weights_def, Price_Zones = obj_w_rule(grid,ObjRule,OnlyGen)
        
    model = pyo.ConcreteModel()
    model.name="AC/DC hybrid OPF"
    
    
    t1 = time.time()
    
    # pr = cProfile.Profile()
    # pr.enable()
    # Call your function here
    OPF_createModel_ACDC(model,grid,PV_set,Price_Zones)
    # pr.disable()
    
    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s)
    # ps.sort_stats('cumulative')  # Can also try 'time'
    # ps.print_stats()
    # print(s.getvalue())
    
    t2 = time.time()  
    t_modelcreate = t2-t1
    
    """
    """
    
    
    
    obj_rule= OPF_obj(model,grid,weights_def,OnlyGen)

    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    """
    """
    
    if grid.nn_DC!=0:

        if any(conv.OPF_fx for conv in grid.Converters_ACDC):
                    fx_conv(model, grid)
                
                
    """
    """
    model_res,solver_stats = OPF_solve(model,grid)
    
    t1 = time.time()
    # pr = cProfile.Profile()
    # pr.enable()
    # Call your function here
    ExportACDC_model_toPyflowACDC(model, grid, Price_Zones)
    # pr.disable()

    for obj in weights_def:
        weights_def[obj]['v']=calculate_objective(grid,obj,OnlyGen)
    
    # s = StringIO()
    # ps = pstats.Stats(pr, stream=s)
    # ps.sort_stats('cumulative')  # Can also try 'time'
    # ps.print_stats()
    # print(s.getvalue())
    t2 = time.time()  
    t_modelexport = t2-t1
   
       
    grid.OPF_run=True 
    grid.OPF_obj=weights_def
    timing_info = {
    "create": t_modelcreate,
    "solve": solver_stats['time'],
    "export": t_modelexport,
    }
    return model, model_res , timing_info, solver_stats


def TS_parallel_OPF(grid,idx,current_range,ObjRule=None,PV_set=False,OnlyGen=True,Price_Zones=False,print_step=False):
    from .Time_series import update_grid_data,modify_parameters
    
    weights_def, Price_Zones = obj_w_rule(grid,ObjRule,OnlyGen)
        
        
    model = pyo.ConcreteModel()
    model.name="TS MTDC AC/DC hybrid OPF"
    
    
    model.Time_frames = pyo.Set(initialize=range(idx, idx + current_range))
    model.submodel = pyo.Block(model.Time_frames)
    # Run parallel iterations
    base_model = pyo.ConcreteModel()
    base_model = OPF_createModel_ACDC(base_model,grid,PV_set=False,Price_Zones=True,TEP=True)

    for i in range(current_range):
        t = idx + i
        if print_step:
            print(t)
        base_model_copy = base_model.clone()
        model.submodel[t].transfer_attributes_from(base_model_copy)

        for ts in grid.Time_series:
            update_grid_data(grid, ts, t)
        
        if Price_Zones:
            for price_zone in grid.Price_Zones:
                if price_zone.b > 0:
                    price_zone.PGL_min -= price_zone.ImportExpand
                    price_zone.a = -price_zone.b / (2 * price_zone.PGL_min * grid.S_base) 
                    
            
        modify_parameters(grid,model.submodel[t],Price_Zones) 
        subobj = OPF_obj(model.submodel[t],grid,weights_def,OnlyGen)
        model.submodel[t].obj = pyo.Objective(rule=subobj, sense=pyo.minimize)

    obj_rule= TS_parallel_obj(model)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    model_results,elapsed_time= OPF_solve(model,grid)
    
    Current_range_res = obtain_results_TSOPF(model,grid,current_range,idx,Price_Zones)
      
    return model, Current_range_res,t,elapsed_time


def obtain_results_TSOPF(model,grid,current_range,idx,Price_Zones) :
    opt_res_P_conv_DC_list = []
    opt_res_P_conv_AC_list =[]
    opt_res_Q_conv_AC_list =[]
    opt_res_P_Load_list =[]
    opt_res_P_extGrid_list = []
    opt_res_curtailment_list = []
    opt_res_Q_extGrid_list = []
    opt_res_Loading_conv_list =[]
    opt_res_Loading_lines_list =[]
    opt_res_price_list =[]
    opt_res_Loading_grid_list=[]
    for i in range(current_range):
        
        t = idx + i
        # print(t+1)
        
        (opt_res_P_conv_DC, opt_res_P_conv_AC, opt_res_Q_conv_AC, opt_P_load,
         opt_res_P_extGrid, opt_res_Q_extGrid, opt_res_curtailment,opt_res_Loading_conv) = OPF_conv_results(model.submodel[t], grid)
        
        opt_res_Loading_lines,opt_res_Loading_grid=OPF_line_res (model.submodel[t],grid)
        
        if Price_Zones:
           opt_res_price=OPF_price_priceZone (model.submodel[t],grid)
        else:
            opt_res_price={}
            for ts in grid.Time_series:
                if ts.type == 'price':
                    opt_res_price[ts.name]=ts.data[t]
                        
        # Add the time index to the dictionaries
        opt_res_curtailment['time'] = t + 1
        opt_res_P_conv_DC['time'] = t + 1
        opt_res_P_conv_AC['time'] = t + 1
        opt_res_Q_conv_AC['time'] = t + 1
        opt_res_P_extGrid['time'] = t + 1
        opt_res_Q_extGrid['time']=t+1
        opt_P_load['time']        = t+1
        opt_res_Loading_conv['time'] = t + 1
        opt_res_Loading_lines['time'] = t + 1
        opt_res_Loading_grid['time'] =t+1
        opt_res_price['time']=t+1
        
        # Append the dictionaries to the respective lists
        opt_res_P_conv_DC_list.append(opt_res_P_conv_DC)
        opt_res_P_conv_AC_list.append(opt_res_P_conv_AC)
        opt_res_Q_conv_AC_list.append(opt_res_Q_conv_AC)
        
        opt_res_P_extGrid_list.append(opt_res_P_extGrid)
        opt_res_P_Load_list.append(opt_P_load)
        opt_res_curtailment_list.append(opt_res_curtailment)
        opt_res_Q_extGrid_list.append(opt_res_Q_extGrid)
        opt_res_Loading_conv_list.append(opt_res_Loading_conv)
        opt_res_Loading_lines_list.append(opt_res_Loading_lines)
        opt_res_price_list.append(opt_res_price)
        opt_res_Loading_grid_list.append(opt_res_Loading_grid)

    # After processing all time steps, pack the results into tuples
    touple = (opt_res_Loading_conv_list,opt_res_Loading_lines_list,opt_res_Loading_grid_list,
             opt_res_P_conv_AC_list,opt_res_Q_conv_AC_list,opt_res_P_conv_DC_list,
             opt_res_P_extGrid_list,opt_res_P_Load_list,opt_res_Q_extGrid_list,
             opt_res_curtailment_list,opt_res_price_list)
    
    
    return touple

def TS_parallel_obj(model):
   
    # Calculate the weighted social cost for each submodel (subblock)
    total_obj = 0
    for t in model.Time_frames:
        submodel_obj = model.submodel[t].obj
        model.submodel[t].obj.deactivate()
        total_obj+=submodel_obj
      
        
    return total_obj 


def fx_conv(model,grid):
    def fx_PDC(model,conv):
        if grid.Converters_ACDC[conv].OPF_fx==True and grid.Converters_ACDC[conv].OPF_fx_type=='PDC':
            return model.P_conv_DC[conv.Node_DC.nodeNumber]==grid.Converters_ACDC[conv].P_DC
        else:
            return pyo.Constraint.Skip
    def fx_PAC(model,conv):   
        if grid.Converters_ACDC[conv].OPF_fx==True and (grid.Converters_ACDC[conv].OPF_fx_type=='PQ' or grid.Converters_ACDC[conv].OPF_fx_type=='PV'):
            return model.P_conv_s_AC[conv]==grid.Converters_ACDC[conv].P_AC
        else:
            return pyo.Constraint.Skip
    def fx_QAC(model,conv):    
        if grid.Converters_ACDC[conv].OPF_fx==True and grid.Converters_ACDC[conv].OPF_fx_type=='PQ':
            return model.Q_conv_s_AC[conv]==grid.Converters_ACDC[conv].Q_AC
        else:
            return pyo.Constraint.Skip
        
    model.Conv_fx_pdc=pyo.Constraint(model.conv,rule=fx_PDC)
    model.Conv_fx_pac=pyo.Constraint(model.conv,rule=fx_PAC)
    model.Conv_fx_qac =pyo.Constraint(model.conv,rule=fx_QAC)


def OPF_solve(model,grid,solver = 'ipopt'):
    
    solver = solver.lower()

    if grid.MixedBinCont:
           # opt = pyo.SolverFactory("mindtpy")
           # results = opt.solve(model,mip_solver='glpk',nlp_solver='ipopt')
           print('PyFlow ACDC is not capable of ensuring the reliability of this solution.')
    """
    if solver_options is None:
        solver = 'ipopt' 
        tol = 1e-8
        max_iter = 3000
        print_level = 12
        acceptable_tol = 1e-6
    else:
        solver   = solver_options['solver'] if 'solver' in solver_options else 'ipopt'
        tol      = solver_options['tol'] if 'tol' in solver_options else 1e-8
        max_iter = solver_options['max_iter'] if 'max_iter' in solver_options else 3000
        print_level = solver_options['print_level'] if 'print_level' in solver_options else 12
        acceptable_tol = solver_options['acceptable_tol'] if 'acceptable_tol' in solver_options else 1e-6

    
    opt.options['max_iter']       = max_iter  # Maximum number of iterations
    opt.options['tol']            = tol   # Convergence tolerance
    opt.options['acceptable_tol'] = acceptable_tol   # Acceptable convergence tolerance
    opt.options['print_level']    = print_level      # Output verbosity (0-12)
    """

    #solver = solver_options['solver'] if 'solver' in solver_options else 'ipopt'
    #tee = solver_options['tee'] if 'tee' in solver_options else True
    #keepfiles = tee

    

    #bonmin
    #ipopt
    #logging = solver_options['logging'] if 'logging' in solver_options else True
    

    opt = pyo.SolverFactory(solver)
    #opt.options['print_level']    = solver_options['print_level'] if 'print_level' in solver_options else 3
    #if logging:
    #    results = opt.solve(model, logfile="ipopt_output.log")
        
    #    with open("ipopt_output.log", "r") as f:
    #        log_content = f.read()
    #        print("Log content:", log_content)  # Debug print

    # Print the regex match attempt
    #match = re.search(r"Number of Iterations\.+:\s*(\d+)", log_content)# Debug print
    #num_iterations = int(match.group(1)) if match else None
    results = opt.solve(model)
    num_iterations = None

    solver_stats = {
        'iterations': num_iterations,  # May not exist in IPOPT
        'best_objective': getattr(results.problem, 'upper_bound', None),  # IPOPT provides upper_bound
        'time': getattr(results.solver, 'time', None),  # May not be available
        'termination_condition': str(results.solver.termination_condition)
    }
    
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        # Set the logging level to INFO
        logging.getLogger('pyomo').setLevel(logging.INFO)

        # Now call log_infeasible_constraints
        log_infeasible_constraints(model)
    
        
    return  results, solver_stats

def OPF_updateParam(model,grid):
 
    for n in grid.nodes_AC:
        model.P_Gain_known_AC[n.nodeNumber] = n.PGi
        model.P_Load_known_AC[n.nodeNumber] = n.PLi
        model.Q_known_AC[n.nodeNumber] = n.QGi-n.QLi
        model.price[n.nodeNumber] = n.price
        
    for n in grid.nodes_DC:
        model.P_known_DC[n.nodeNumber] = n.P_DC
    

    return model

def OPF_obj(model,grid,ObjRule,OnlyGen=True):
   
    # for node in  model.nodes_AC:
    #     nAC=grid.nodes_AC[node]
    #     if nAC.Num_conv_connected >= 2:
    #         obj_expr += sum(model.Q_conv_s_AC[conv]**2 for conv in nAC.connected_conv)

   
    def formula_Min_Ext_Gen():
        if ObjRule['Ext_Gen']['w']==0:
            return 0
        return sum((model.PGi_opt[node]*grid.S_base) for node in model.nodes_AC)

    def formula_Energy_cost():
        if ObjRule['Energy_cost']['w']==0:
            return 0
        
        AC= 0
        DC= 0
        if grid.ACmode:
            AC= sum(((model.PGi_gen[gen.genNumber]*grid.S_base)**2*gen.qf+model.PGi_gen[gen.genNumber]*grid.S_base*model.lf[gen.genNumber]+model.np_gen[gen.genNumber]*gen.fc) for gen in grid.Generators)
        if grid.DCmode:
            DC= sum(((model.PGi_gen_DC[gen.genNumber_DC]*grid.S_base)**2*gen.qf+model.PGi_gen_DC[gen.genNumber_DC]*grid.S_base*model.lf_dc[gen.genNumber_DC]+model.np_gen_DC[gen.genNumber_DC]*gen.fc) for gen in grid.Generators_DC)
        
        if OnlyGen:
            return AC+DC
        
        else :
            nodes_with_RenSource = [node for node in model.nodes_AC if grid.nodes_AC[node].RenSource]
            nodes_with_conv= [node for node in model.nodes_AC if grid.nodes_AC[node].Num_conv_connected != 0]
            return AC+DC  \
                   + sum(model.PGi_ren[node]*model.price[node] for node in nodes_with_RenSource)*grid.S_base \
                   + sum(model.P_conv_AC[node]*model.price[node] for node in nodes_with_conv)*grid.S_base
    def formula_AC_losses():
        if ObjRule['AC_losses']['w']==0:
            return 0
        loss = sum(model.PAC_line_loss[line] for line in model.lines_AC)
        if grid.TAP_tf:
            loss += sum(model.tf_PAC_line_loss[tf] for tf in model.lines_AC_tf)
        if grid.TEP_AC:
            loss += sum(model.exp_PAC_line_loss[exp] for exp in model.lines_AC_exp)   
        if grid.REC_AC:
            loss += sum(model.rec_PAC_line_loss[rec] for rec in model.lines_AC_rec)
        if grid.CT_AC:
            loss += sum(model.ct_PAC_line_loss[ct] for ct in model.lines_AC_ct)
        return loss

    def formula_DC_losses():
        if ObjRule['DC_losses']['w']==0:
            return 0
        loss = sum(model.PDC_line_loss[line] for line in model.lines_DC)
        if grid.CDC:
            loss += sum(model.CDC_loss[conv] for conv in model.DCDC_conv)
        return loss

    def formula_Converter_Losses():
        if ObjRule['Converter_Losses']['w']==0:
            return 0
        return sum(model.P_conv_loss[conv]+model.P_AC_loss_conv[conv] for conv in model.conv)

    def formula_General_Losses():
        if ObjRule['General_Losses']['w']==0:
            return 0
        load = 0
        if grid.nodes_AC != []:
            load = sum(model.P_known_AC[node] for node in model.nodes_AC)
        if grid.nodes_DC != []:
            load = sum(model.P_known_DC[node] for node in model.nodes_DC)
        gen = 0
        if grid.Generators != []:
            gen = sum(model.PGi_gen[gen] for gen in model.gen_AC)
        if grid.RenSources != []:
            gen = sum(model.P_renSource[rs]*model.gamma[rs] for rs in model.ren_sources)
        return gen - load
    
    def formula_curtailment_red():
        if ObjRule['Curtailment_Red']['w']==0:
            return 0
        ac_curt=0
        dc_curt=0
        if grid.ACmode:
            ac_curt= sum((1-model.gamma[rs])*model.P_renSource[rs]*model.price[grid.rs2node['AC'].get(rs, 0)]*rs.sigma for rs in model.ren_sources)*grid.S_base
        if grid.DCmode:
            dc_curt= sum((1-model.gamma[rs])*model.P_renSource[rs]*model.price_DC[grid.rs2node['DC'].get(rs, 0)]*rs.sigma for rs in model.ren_sources)*grid.S_base
        return ac_curt+dc_curt
    def formula_CG():
       if ObjRule['PZ_cost_of_generation']['w']==0:
           return 0
       return sum(model.SocialCost[price_zone] for price_zone in model.M)
   
    def formula_Offshoreprofit():
        from .Classes import OffshorePrice_Zone
        if ObjRule['Renewable_profit']['w']==0:
            return 0
        nodes_with_RenSource = []
        convloss=0
        for price_zone in model.M:
            for conv in grid.Price_Zones[price_zone].ConvACDC:     
                convloss+=model.price_zone_price[price_zone]*(model.P_conv_loss[conv.ConvNumber]+model.P_AC_loss_conv[conv.ConvNumber])*grid.S_base
            if isinstance(grid.Price_Zones[price_zone], OffshorePrice_Zone):
                # Loop through the nodes assigned to the offshore price_zone
                for node in grid.Price_Zones[price_zone].nodes_AC:
                    # Check if the node is marked as a renewable source and add it to the list
                    if node.RenSource:
                        nodes_with_RenSource.append(node.nodeNumber)
        
        return -sum(model.PGi_ren[node]*model.price[node] for node in nodes_with_RenSource)*grid.S_base +convloss
   
    def formula_Gen_set_dev():
        if ObjRule['Gen_set_dev']['w']==0:
            return 0
        return sum((model.PGi_gen[gen.genNumber]-gen.Pset)**2 for gen in grid.Generators)
    s=1
    for key, entry in ObjRule.items():
        if key == 'Ext_Gen':
            entry['f'] = formula_Min_Ext_Gen()
        elif key == 'Energy_cost':
            entry['f'] = formula_Energy_cost()
        elif key == 'AC_losses':
            entry['f'] = formula_AC_losses()
        elif key == 'DC_losses':
            entry['f'] = formula_DC_losses()
        elif key == 'Converter_Losses':
            entry['f'] = formula_Converter_Losses()
        elif key == 'General_Losses':
            entry['f'] = formula_General_Losses()
        elif key == 'Curtailment_Red':   
            entry ['f'] = formula_curtailment_red()
        elif key == 'PZ_cost_of_generation':
            entry['f']  =formula_CG()
        elif key == 'Renewable_profit':
            entry['f']  =formula_Offshoreprofit()    
        elif key == 'Gen_set_dev':
            entry['f']  =formula_Gen_set_dev()  
        
    s=1
    total_weight = sum(entry['w'] for entry in ObjRule.values())
    if total_weight== 0:
        weighted_sum=0
    else:
        weighted_sum = sum(entry['w'] / total_weight * entry['f'] for entry in ObjRule.values())
    
    
    return weighted_sum






def Translate_pyf_OPF(grid,Price_Zones=False):
    """Translation of element wise to internal numbering"""
    AC_info, DC_info, Conv_info,DCDC_info = None, None, None,None
    ACmode= grid.ACmode
    DCmode = grid.DCmode
    "AC system info"
    lista_nodos_AC = list(range(0, grid.nn_AC))
    lista_lineas_AC = list(range(0, grid.nl_AC))
    lista_lineas_AC_exp = list(range(0, grid.nle_AC))
    lista_lineas_AC_tf = list(range(0, grid.nttf))
    lista_lineas_AC_rec = list(range(0, grid.nlr_AC))
    lista_lineas_AC_ct = list(range(0, grid.nct_AC))
    # Dictionaries for AC variables
    price, V_ini_AC, Theta_ini = {}, {}, {}
    P_renSource, P_know, Q_know = {}, {}, {}
    S_lineAC_limit,S_lineACexp_limit,S_lineACtf_limit,m_tf_og,NP_lineAC  = {}, {}, {}, {},{}
    S_lineACrec_lim, S_lineACrec_lim_new,REC_AC_act = {}, {}, {}
    lf,qf,fc,np_gen = {}, {}, {}, {}
    lf_DC,qf_DC,fc_DC,np_gen_DC = {}, {}, {}, {}

    S_lineACct_lim,cab_types_set,allowed_types = {},{},{}

    u_min_ac = list(range(0, grid.nn_AC))
    u_max_ac = list(range(0, grid.nn_AC))

    AC_slack, AC_PV = [], []

    # Fill AC node and line information
    
    for gen in grid.Generators:
        lf[gen.genNumber] = gen.lf
        qf[gen.genNumber] = gen.qf
        fc[gen.genNumber] = gen.fc
        np_gen[gen.genNumber] = gen.np_gen
    
    lista_gen = list(range(0, grid.n_gen))
    
    for gen in grid.Generators_DC:
        lf_DC[gen.genNumber_DC] = gen.lf
        qf_DC[gen.genNumber_DC] = gen.qf
        fc_DC[gen.genNumber_DC] = gen.fc
        np_gen_DC[gen.genNumber_DC] = gen.np_gen
    
    lista_gen_DC = list(range(0, grid.n_gen_DC))
       
    nn_rs=0
    for rs in grid.RenSources:
        nn_rs+=1
        P_renSource[rs.rsNumber]=rs.PGi_ren
        
    lista_rs = list(range(0, nn_rs))

    gen_AC_info = pack_variables(lf,qf,fc,np_gen,lista_gen)
    gen_DC_info = pack_variables(lf_DC,qf_DC,fc_DC,np_gen_DC,lista_gen_DC)
    gen_info = pack_variables(gen_AC_info,gen_DC_info,P_renSource,lista_rs)

    "Price zone info"
   
    price_zone_prices, price_zone_as, price_zone_bs, PGL_min, PGL_max, PL_price_zone =  {}, {}, {}, {}, {}, {}
    nn_M, lista_M = 0, []
    node2price_zone = {'DC': {}, 'AC': {}}
    price_zone2node = {'DC': {}, 'AC': {}}
    if Price_Zones:
        for m in grid.Price_Zones:
            
            nn_M += 1
            price_zone_prices[m.price_zone_num] = m.price
            price_zone_as[m.price_zone_num] = m.a
            price_zone_bs[m.price_zone_num] = m.b
            import_M = m.import_pu_L
            export_M = m.export_pu_G * (sum(sum(rs.PGi_ren for rs in node.connected_RenSource) + sum(gen.Max_pow_gen for gen in node.connected_gen) for node in m.nodes_AC))*grid.S_base
            PL_price_zone[m.price_zone_num] = 0
            
            if ACmode:
                price_zone2node['AC'][m.price_zone_num] = []
                for n in m.nodes_AC:
                    price_zone2node['AC'][m.price_zone_num].append(n.nodeNumber)
                    node2price_zone['AC'][n.nodeNumber] = m.price_zone_num
                    PL_price_zone[m.price_zone_num] += n.PLi
            
            if DCmode:
                price_zone2node['DC'][m.price_zone_num] = []
                for n in m.nodes_DC:
                    price_zone2node['DC'][m.price_zone_num].append(n.nodeNumber)
                    node2price_zone['DC'][n.nodeNumber] = m.price_zone_num
                    PL_price_zone[m.price_zone_num] += n.PLi
            PGL_min[m.price_zone_num] = max(m.PGL_min, -import_M * PL_price_zone[m.price_zone_num]*grid.S_base)
            PGL_max[m.price_zone_num] = min(m.PGL_max, export_M)
        lista_M = list(range(0, nn_M))
    
    Price_Zone_Lists = pack_variables(lista_M, node2price_zone, price_zone2node)
    Price_Zone_lim = pack_variables(price_zone_as, price_zone_bs, PGL_min, PGL_max)
    Price_Zone_info = pack_variables(Price_Zone_Lists, Price_Zone_lim)

    if ACmode:
        for n in grid.nodes_AC:
            V_ini_AC[n.nodeNumber] = n.V_ini
            Theta_ini[n.nodeNumber] = n.theta_ini
            
            P_know[n.nodeNumber] = n.PGi - n.PLi
            Q_know[n.nodeNumber] = n.QGi - n.QLi
            
            u_min_ac[n.nodeNumber] = n.Umin
            u_max_ac[n.nodeNumber] = n.Umax
            
            price[n.nodeNumber] = n.price
            
            if n.type == 'Slack':
                AC_slack.append(n.nodeNumber)
            elif n.type == 'PV':
                AC_PV.append(n.nodeNumber)
            
        
        for l in grid.lines_AC:
            S_lineAC_limit[l.lineNumber]    = l.MVA_rating / grid.S_base
        
        for l in grid.lines_AC_exp:
            S_lineACexp_limit[l.lineNumber] = l.MVA_rating / grid.S_base
            NP_lineAC[l.lineNumber]         = l.np_line

        for l in grid.lines_AC_rec:
            S_lineACrec_lim[l.lineNumber] = l.MVA_rating / grid.S_base
            S_lineACrec_lim_new[l.lineNumber] = l.MVA_rating_new / grid.S_base
            REC_AC_act[l.lineNumber] = 0 if not l.rec_branch  else 1

        for l in grid.lines_AC_tf:
            S_lineACtf_limit[l.lineNumber]  = l.MVA_rating / grid.S_base
            m_tf_og[l.lineNumber]           = l.m
            
        for l in grid.lines_AC_ct:
            for i in range(len(l.MVA_rating_list)):
                S_lineACct_lim[l.lineNumber,i] = l.MVA_rating_list[i] / grid.S_base
        if grid.Cable_options is not None and len(grid.Cable_options) > 0:
            cab_types_set = list(range(0,len(grid.Cable_options[0].cable_types)))
        else:
            cab_types_set = []
        allowed_types = grid.cab_types_allowed
        
        # Packing common AC info
        AC_Lists = pack_variables(lista_nodos_AC, lista_lineas_AC,lista_lineas_AC_tf,AC_slack, AC_PV)
        AC_nodes_info = pack_variables(u_min_ac, u_max_ac, V_ini_AC, Theta_ini, P_know, Q_know, price)
        AC_lines_info = pack_variables(S_lineAC_limit,S_lineACtf_limit,m_tf_og)
        
        EXP_info = pack_variables(lista_lineas_AC_exp,S_lineACexp_limit,NP_lineAC)
        REC_info = pack_variables(lista_lineas_AC_rec,S_lineACrec_lim,S_lineACrec_lim_new,REC_AC_act)
        CT_info = pack_variables(lista_lineas_AC_ct,S_lineACct_lim,cab_types_set,allowed_types)
        AC_info = pack_variables(AC_Lists, AC_nodes_info, AC_lines_info,EXP_info,REC_info,CT_info)
    
   
    if DCmode:

        # DC and Converter Variables (if not OnlyAC)
        lista_nodos_DC = list(range(0, grid.nn_DC))
        lista_nodos_DC_sin_cn=lista_nodos_DC
        lista_lineas_DC = list(range(0, grid.nl_DC))
        lista_conv = list(range(0, grid.nconv))


        u_min_dc = list(range(0, grid.nn_DC))
        u_max_dc = list(range(0, grid.nn_DC))
        u_c_min = list(range(0, grid.nconv))
        u_c_max = list(range(0, grid.nconv))

        V_ini_DC, P_known_DC, P_conv_limit,price_dc = {}, {}, {},{}
        P_lineDC_limit, NP_lineDC = {}, {}

        AC_nodes_connected_conv, DC_nodes_connected_conv = [], []
        S_limit_conv, NumConvP, P_conv_loss = {}, {}, {}
        DC_slack = []

        P_DCDC_limit, Pset_DCDC = {}, {}
        
        
        for n in grid.nodes_DC:
            V_ini_DC[n.nodeNumber] = n.V_ini
            P_known_DC[n.nodeNumber] = n.PGi-n.PLi
            u_min_dc[n.nodeNumber] = n.Umin
            u_max_dc[n.nodeNumber] = n.Umax
            price_dc[n.nodeNumber] = n.price
            if n.type == 'Slack':
                DC_slack.append(n.nodeNumber)

        for l in grid.lines_DC:
            P_lineDC_limit[l.lineNumber] = l.MW_rating / grid.S_base
            NP_lineDC[l.lineNumber] = l.np_line

        lista_DCDC = list(range(0, grid.ncdc_DC))

        for cn in grid.Converters_DCDC:
            P_DCDC_limit[cn.ConvNumber] = cn.MW_rating / grid.S_base
            Pset_DCDC[cn.ConvNumber] = cn.Powerto

        
        DCDC_info = pack_variables(lista_DCDC,P_DCDC_limit,Pset_DCDC)
        # Packing AC, DC, Converter, and Price_Zone info
        DC_Lists = pack_variables(lista_nodos_DC, lista_lineas_DC, DC_slack,DC_nodes_connected_conv)
        DC_nodes_info = pack_variables(u_min_dc, u_max_dc, V_ini_DC, P_known_DC,price_dc)
        DC_lines_info = pack_variables(P_lineDC_limit, NP_lineDC)
        DC_info = pack_variables(DC_Lists, DC_nodes_info, DC_lines_info,DCDC_info)
   
    if ACmode and DCmode:

        for conv in grid.Converters_ACDC:
            AC_nodes_connected_conv.append(conv.Node_AC.nodeNumber)
            DC_nodes_connected_conv.append(conv.Node_DC.nodeNumber)
            P_conv_limit[conv.Node_DC.nodeNumber] = conv.MVA_max / grid.S_base
            S_limit_conv[conv.ConvNumber] = conv.MVA_max / grid.S_base
            NumConvP[conv.ConvNumber] = conv.NumConvP
            u_c_min[conv.ConvNumber] = conv.Ucmin
            u_c_max[conv.ConvNumber] = conv.Ucmax
            P_conv_loss[conv.ConvNumber] = conv.P_loss

        Conv_Lists = pack_variables(lista_conv, NumConvP)
        Conv_Volt = pack_variables(u_c_min, u_c_max, S_limit_conv, P_conv_limit) 
        Conv_info = pack_variables(Conv_Lists, Conv_Volt)
    
   
    return pack_variables(AC_info, DC_info, Conv_info, Price_Zone_info,gen_info)




def OPF_line_res (model,grid):
    opt_res_Loading_line = {}
    opt_res_Loading_grid ={}
    loadS_AC = np.zeros(grid.Num_Grids_AC)
    loadP_DC = np.zeros(grid.Num_Grids_DC)
    

    def process_line_AC(line):
        l= line.lineNumber
        G = grid.Graph_line_to_Grid_index_AC[line]
        
        P_from = PAC_from_values[l]
        P_to   = PAC_to_values[l]
        Q_from = QAC_from_values[l]
        Q_to   = QAC_to_values[l]
        
        S_from = np.sqrt(P_from**2+Q_from**2)
        S_to = np.sqrt(P_to**2+Q_to**2)
        
        loading = max(S_from,S_to)*grid.S_base/line.MVA_rating
        # with lock:
        loadS_AC[G] += max(S_from, S_to) * grid.S_base
        opt_res_Loading_line[f'AC_Load_{line.name}'] = loading
        opt_res_Loading_line[f'AC_from_{line.name}'] = S_from * grid.S_base
        opt_res_Loading_line[f'AC_to_{line.name}'] = S_to * grid.S_base
    
    
    def process_line_DC(line):
        G = grid.Graph_line_to_Grid_index_DC[line]
        
        l= line.lineNumber
        P_from = PDC_from_values[l]
        P_to   = PDC_to_values[l]
      
        loading = max(P_from,P_to)*grid.S_base/line.MW_rating
        # with lock:
        loadP_DC[G] += max(P_from, P_to) * grid.S_base
        opt_res_Loading_line[f'DC_Load_{line.name}'] = loading
        opt_res_Loading_line[f'DC_from_{line.name}'] = P_from * grid.S_base
        opt_res_Loading_line[f'DC_to_{line.name}'] = P_to * grid.S_base
    
    if grid.lines_AC: 
        PAC_from_values= {k: np.float64(pyo.value(v)) for k, v in model.PAC_from.items()}
        PAC_to_values  = {k: np.float64(pyo.value(v)) for k, v in model.PAC_to.items()}
        QAC_from_values= {k: np.float64(pyo.value(v)) for k, v in model.QAC_from.items()}
        QAC_to_values  = {k: np.float64(pyo.value(v)) for k, v in model.QAC_to.items()}
        
        
        with ThreadPoolExecutor() as executor:
            executor.map(process_line_AC, grid.lines_AC)
    
    if grid.lines_DC:
        PDC_from_values= {k: np.float64(pyo.value(v)) for k, v in model.PDC_from.items()}
        PDC_to_values  = {k: np.float64(pyo.value(v)) for k, v in model.PDC_to.items()}
        
        with ThreadPoolExecutor() as executor:
            executor.map(process_line_DC, grid.lines_DC)
        
        
    total_loading = 0
    total_rating = sum(grid.rating_grid_AC) + sum(grid.rating_grid_DC)
    
    for g in range(grid.Num_Grids_AC):
        loading = loadS_AC[g]
        total_loading += loading
        opt_res_Loading_grid[f'Loading_Grid_AC_{g+1}'] = 0 if grid.rating_grid_AC[g] == 0 else loading / grid.rating_grid_AC[g]

    for g in range(grid.Num_Grids_DC):
        loading = loadP_DC[g]
        total_loading += loading
        opt_res_Loading_grid[f'Loading_Grid_DC_{g+1}'] = loading / grid.rating_grid_DC[g]
    opt_res_Loading_grid['Total'] = 0 if total_rating == 0 else total_loading /total_rating
    
    return opt_res_Loading_line,opt_res_Loading_grid


def OPF_price_priceZone (model,grid):
    opt_res_Loading_pz = {}
    for pz in grid.Price_Zones:
        m= pz.price_zone_num
        price = pyo.value(model.price_zone_price[m])
        opt_res_Loading_pz[pz.name]=price

    
    return opt_res_Loading_pz
 
def OPF_conv_results(model,grid):
    opt_res_P_conv_DC = {}
    opt_res_P_conv_AC = {}
    opt_res_Q_conv_AC = {}
    opt_res_Loading_conv={}
    opt_P_load = {}
    opt_res_P_extGrid = {}
    opt_res_Q_extGrid  = {}
    opt_res_curtailment ={}
   
    P_conv_DC_conv_values= {k: np.float64(pyo.value(v)) for k, v in model.P_conv_DC.items()}
    P_conv_s_AC_values   = {k: np.float64(pyo.value(v)) for k, v in model.P_conv_s_AC.items()}
    Q_conv_s_AC_values   = {k: np.float64(pyo.value(v)) for k, v in model.Q_conv_s_AC.items()}
    
    def process_converter(conv):
        nconv = conv.ConvNumber
        name = conv.name   
       
        opt_res_P_conv_DC[name] = P_conv_DC_conv_values[conv.Node_DC.nodeNumber] * conv.NumConvP
        opt_res_P_conv_AC[name] = P_conv_s_AC_values[nconv] * conv.NumConvP
        opt_res_Q_conv_AC[name] = Q_conv_s_AC_values[nconv] * conv.NumConvP
            
        
        S_AC = np.sqrt(opt_res_P_conv_AC[name]**2 + opt_res_Q_conv_AC[name]**2)
        P_DC = opt_res_P_conv_DC[name]
        
        opt_res_Loading_conv[name]=max(S_AC, np.abs(P_DC)) * grid.S_base / (conv.MVA_max*conv.NumConvP)
       
    with ThreadPoolExecutor() as executor:
        executor.map(process_converter, grid.Converters_ACDC)
    
    Pload_values = {k: np.float64(pyo.value(v)) for k, v in model.P_known_AC.items()}
    PGen_values  = {k: np.float64(pyo.value(v)) for k, v in model.PGi_gen.items()}
    QGen_values  = {k: np.float64(pyo.value(v)) for k, v in model.QGi_gen.items()}
    gamma_values = {k: np.float64(pyo.value(v)) for k, v in model.gamma.items()}
    Pren_values  = {k: np.float64(pyo.value(v)) for k, v in model.P_renSource.items()}
    Qren_values  = {k: np.float64(pyo.value(v)) for k, v in model.Q_renSource.items()}
    
    def process_load(node):
        nAC= node.nodeNumber
        name = node.name
        
        opt_P_load[name]= -Pload_values[nAC]
        
        
    with ThreadPoolExecutor() as executor:
        executor.map(process_load, grid.nodes_AC)
    
    def process_element(element):
        if hasattr(element, 'genNumber'):  # Generator
            name = element.name
            opt_res_P_extGrid [name] = PGen_values[element.genNumber]
            opt_res_Q_extGrid [name] = QGen_values[element.genNumber]

        elif hasattr(element, 'rsNumber'):  # Renewable Source
            name = element.name
            gamma=gamma_values[element.rsNumber]
            opt_res_curtailment [name] = 1-gamma
            opt_res_P_extGrid[f'RenSource_{name}'] = Pren_values[element.rsNumber]*gamma
            opt_res_Q_extGrid[f'RenSource_{name}'] = Qren_values[element.rsNumber]

    # Combine Generators and Renewable Sources into one iterable
    elements = grid.Generators + grid.RenSources
    
    # Parallelize processing
    with ThreadPoolExecutor() as executor:
        executor.map(process_element, elements)
        
            
    return (opt_res_P_conv_DC, opt_res_P_conv_AC, opt_res_Q_conv_AC, opt_P_load,
                opt_res_P_extGrid, opt_res_Q_extGrid, opt_res_curtailment, 
                opt_res_Loading_conv)


      

def calculate_objective(grid,obj,OnlyGen=True):
   
    if obj =='Ext_Gen':
        return sum((node.PGi_opt*grid.S_base) for node in grid.nodes_AC)

    if obj =='Energy_cost':
        AC= 0
        DC= 0
        if grid.ACmode:
            AC= sum(((gen.PGen*grid.S_base)**2*gen.qf+gen.PGen*grid.S_base*gen.lf+gen.np_gen*gen.fc) for gen in grid.Generators)
        if grid.DCmode:
            DC= sum(((gen.PGen*grid.S_base)**2*gen.qf+gen.PGen*grid.S_base*gen.lf+gen.np_gen*gen.fc) for gen in grid.Generators_DC)
        return AC+DC

        
    if obj =='PZ_cost_of_generation':
       return sum(pz.a*(pz.PN*grid.S_base)**2+pz.b*(pz.PN*grid.S_base) for pz in grid.Price_Zones)
   
    if obj =='AC_losses':
        return (sum(line.P_loss for line in grid.lines_AC)+
                sum(tf.P_loss for tf in grid.lines_AC_tf)+
                sum(line.P_loss for line in grid.lines_AC_exp)+
                sum(line.P_loss for line in grid.lines_AC_rec)+
                sum(line.P_loss for line in grid.lines_AC_ct))*grid.S_base

    if obj =='DC_losses':
        return (sum(line.loss for line in grid.lines_DC)+
                sum(conv.loss for conv in grid.Converters_DCDC))*grid.S_base

    if obj =='Converter_Losses':
        return sum(conv.P_loss for conv in grid.Converters_ACDC)*grid.S_base

    if obj =='General_Losses':
        return (sum(line.P_loss for line in grid.lines_AC) +
                sum(tf.P_loss for tf in grid.lines_AC_tf) +
                sum(line.P_loss for line in grid.lines_AC_exp) +
                sum(line.loss for line in grid.lines_DC) +
                sum(conv.P_loss for conv in grid.Converters_ACDC))*grid.S_base

    if obj =='Curtailment_Red':
        ac_curt=0
        dc_curt=0
        if grid.ACmode:
            ac_curt= sum((1-rs.gamma)*rs.PGi_ren*grid.nodes_AC[grid.rs2node['AC'].get(rs, 0)].price*rs.sigma for rs in grid.RenSources)*grid.S_base
        if  grid.DCmode:
            dc_curt= sum((1-rs.gamma)*rs.PGi_ren*grid.nodes_DC[grid.rs2node['DC'].get(rs, 0)].price*rs.sigma for rs in grid.RenSources)*grid.S_base
        return ac_curt+dc_curt
    
    if obj=='PZ_cost_of_generation':
           return  sum(pz.PN**2*pz.a+pz.PN*pz.b for pz in grid.Price_Zones)
   
    if obj=='Gen_set_dev':
        return sum((gen.PGen-gen.Pset)**2 for gen in grid.Generators)
    
    return 0