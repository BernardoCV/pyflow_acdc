
import numpy as np
import pyomo.environ as pyo
import pandas as pd
import time
from concurrent.futures import ThreadPoolExecutor


from .ACDC_OPF_model import OPF_createModel_ACDC,analyse_OPF,TEP_variables
from .ACDC_OPF import OPF_solve,OPF_obj,obj_w_rule,ExportACDC_model_toPyflowACDC,calculate_objective
from .ACDC_TEP import get_TEP_variables

__all__ = [
    'multi_period_TEP'
]

def pack_variables(*args):
    return args



def multi_period_TEP(grid,NPV=True,inv_periods = [1,2,3],n_years=10,Hy=8760,discount_rate=0.02,ObjRule=None,solver='bonmin'):
    ACmode,DCmode,ACadd,DCadd,GPR = analyse_OPF(grid)
    TEP_AC,TAP_tf,REC_AC,CT_AC = ACadd
    CFC = DCadd
    weights_def, PZ = obj_w_rule(grid,ObjRule,True,False)

    grid.TEP_n_years = n_years
    grid.TEP_discount_rate =discount_rate
                
    conv_var,DC_line_var,AC_line_var,gen_var = get_TEP_variables(grid)

    
    t1=time.time()

    model = pyo.ConcreteModel()
    model.name        ="Multi-period TEP MTDC AC/DC hybrid OPF"

    n_periods = len(inv_periods)
    model.inv_periods = pyo.Set(initialize=list(range(0,n_periods)))
    if DCmode:
        model.lines_DC    = pyo.Set(initialize=list(range(0, grid.nl_DC)))
    if ACmode and DCmode:
        model.conv        = pyo.Set(initialize=list(range(0, grid.nconv)))
    if TEP_AC:
        model.lines_AC_exp= pyo.Set(initialize=list(range(0,grid.nle_AC)))
    if REC_AC:
        model.lines_AC_rec= pyo.Set(initialize=list(range(0,grid.nlr_AC)))
    if CT_AC:
        model.lines_AC_ct = pyo.Set(initialize=list(range(0,grid.nct_AC)))
        model.ct_set = pyo.Set(initialize=list(range(0,len(grid.Cable_options[0].cable_types))))
    if GPR:
        model.gen_AC = pyo.Set(initialize=list(range(0,grid.ngen_AC)))

    model.inv_model = pyo.Block(model.inv_periods)

    base_model = pyo.ConcreteModel()
    OPF_createModel_ACDC(base_model,grid,PV_set=False,Price_Zones=PZ,TEP=True)

    for i in model.inv_periods:
        base_model_copy = base_model.clone()
        model.inv_model[i].transfer_attributes_from(base_model_copy)
       
        if ACmode:
            for idx in model.inv_model[i].P_known_AC:
                model.inv_model[i].P_known_AC[idx] *= inv_periods[i]
        
        if DCmode:
            for idx in model.inv_model[i].P_known_DC:
                model.inv_model[i].P_known_DC[idx] *= inv_periods[i]

        
        obj_OPF = OPF_obj(model.inv_model[i],grid,weights_def,True)
        
        present_value =   Hy*(1 - (1 + discount_rate) ** -n_years) / discount_rate
        if NPV:
            obj_OPF *=present_value
        
        model.inv_model[i].obj = pyo.Objective(rule=obj_OPF, sense=pyo.minimize)

    if GPR:
        np_gen,np_gen_max = gen_var

        def np_gen_bounds(model,gen):
            g = grid.Generators[gen]
            if g.np_gen_opf:
                return (np_gen[gen],np_gen_max[gen])
            else:
                return (np_gen[gen],np_gen[gen])
            
        model.np_gen = pyo.Var(model.gen_AC,model.inv_periods,initialize=np_gen,bounds=np_gen_bounds)

        def MP_gen_lower_bound(model,gen,i):
            if i == 0:
                return pyo.Constraint.Skip
            else:
                return model.np_gen[gen,i] >= model.np_gen[gen,i-1]
        model.MP_gen_lower_bound_constraint = pyo.Constraint(model.gen_AC,model.inv_periods,rule=MP_gen_lower_bound)
        
        def MP_gen_link(model,gen,i):
            return model.inv_model[i].np_gen[gen] == model.np_gen[gen,i]
        model.MP_gen_link_constraint = pyo.Constraint(model.gen_AC,model.inv_periods,rule=MP_gen_link)

    if ACmode:
        NP_lineAC,NP_lineAC_i,NP_lineAC_max,Line_length,REC_branch,ct_ini = AC_line_var
        
        model.NumLinesACP_base  =pyo.Param(model.lines_AC_exp,initialize=NP_lineAC)
        def MP_AC_line_bounds(model,l,i):
            return (NP_lineAC[l],NP_lineAC_max[l])
        model.ACLinesMP = pyo.Var(model.lines_AC_exp,model.inv_periods, within=pyo.NonNegativeIntegers,bounds=MP_AC_line_bounds)

        def MP_AC_line_lower_bound(model,l,i):
            if i == 0:
                return pyo.Constraint.Skip
            else:
                return model.ACLinesMP[l,i] >= model.ACLinesMP[l,i-1]
        model.MP_AC_line_lower_bound_constraint = pyo.Constraint(model.lines_AC_exp,model.inv_periods, rule=MP_AC_line_lower_bound)

        def MP_AC_line_link(model, l, i):
            return model.inv_model[i].NumLinesACP[l] == model.ACLinesMP[l, i]
        model.MP_AC_line_link_constraint = pyo.Constraint(model.lines_AC_exp, model.inv_periods, rule=MP_AC_line_link)

    if DCmode:
        NP_lineDC,NP_lineDC_i,NP_lineDC_max,Line_length = DC_line_var
        
        model.NumLinesDCP_base  =pyo.Param(model.lines_DC,initialize=NP_lineDC)
        def MP_DC_line_bounds(model,l,i):
            return (NP_lineDC[l],NP_lineDC_max[l])
        model.DCLinesMP = pyo.Var(model.lines_DC,model.inv_periods, within=pyo.NonNegativeIntegers,bounds=MP_DC_line_bounds)

        def MP_DC_line_lower_bound(model,l,i):
            if i == 0:
                return pyo.Constraint.Skip
            else:
                return model.DCLinesMP[l,i] >= model.DCLinesMP[l,i-1]
        model.MP_DC_line_lower_bound_constraint = pyo.Constraint(model.lines_DC,model.inv_periods, rule=MP_DC_line_lower_bound)

        def MP_DC_line_link(model, l, i):
            return model.inv_model[i].NumLinesDCP[l] == model.DCLinesMP[l, i]
        model.MP_DC_line_link_constraint = pyo.Constraint(model.lines_DC, model.inv_periods, rule=MP_DC_line_link)


    if ACmode and DCmode:
        NumConvP,NumConvP_i,NumConvP_max,S_limit_conv = conv_var
        model.NumConvP_base  =pyo.Param(model.conv,initialize=NumConvP)
        def MP_Conv_bounds(model,l,i):
            return (NumConvP[l],NumConvP_max[l])
        model.ConvMP = pyo.Var(model.conv,model.inv_periods, within=pyo.NonNegativeIntegers,bounds=MP_Conv_bounds)

        def MP_Conv_lower_bound(model,l,i):
            if i == 0:
                return pyo.Constraint.Skip
            else:
                return model.ConvMP[l,i] >= model.ConvMP[l,i-1]
        model.MP_Conv_lower_bound_constraint = pyo.Constraint(model.conv,model.inv_periods, rule=MP_Conv_lower_bound)

        def MP_Conv_link(model, l, i):
            return model.inv_model[i].NumConvP[l] == model.ConvMP[l, i]
        model.MP_Conv_link_constraint = pyo.Constraint(model.conv, model.inv_periods, rule=MP_Conv_link)


    net_cost = MP_TEP_obj(model,grid,n_years,discount_rate)
    model.obj = pyo.Objective(rule=net_cost, sense=pyo.minimize)
    
    t2 = time.time()

    model_results,solver_stats = OPF_solve(model,grid,solver)
    
    t3 = time.time()
    ExportACDC_model_toPyflowACDC(model.inv_model[i], grid, PZ,TEP=True)
    
    MINLP = False
    if solver != 'ipopt':
        MINLP = True
    
    export_MP_TEP_results_toPyflowACDC(model,grid,inv_periods,MINLP)
    t4 = time.time()




   
    timing_info = {
    "create": t2-t1,
    "solve": solver_stats['time'],
    "export": t4-t3,
    }

    
    
    return model, model_results ,timing_info, solver_stats
    


def MP_TEP_obj(model,grid,n_years,discount_rate):
    ACmode,DCmode,ACadd,DCadd,GPR = analyse_OPF(grid)
    TEP_AC,TAP_tf,REC_AC,CT_AC = ACadd
    CFC = DCadd
    net_cost = 0

    for i in model.inv_periods:
        inv_cost = 0
        
        if ACmode:
            AC_Inv_lines=0
            for l in model.lines_AC_exp:
                line = grid.lines_AC_exp[l]
                if i ==0:
                    AC_Inv_lines+=(model.ACLinesMP[l,i]-model.NumLinesACP_base[l])*line.base_cost
                else:
                    AC_Inv_lines+=(model.ACLinesMP[l,i]-model.ACLinesMP[l,i-1])*line.base_cost
            inv_cost += AC_Inv_lines
        if DCmode:
            DC_Inv_lines=0
            for l in model.lines_DC:
                line = grid.lines_DC[l]
                if i ==0:
                    DC_Inv_lines+=(model.DCLinesMP[l,i]-model.NumLinesDCP_base[l])*line.base_cost
                else:
                    DC_Inv_lines+=(model.DCLinesMP[l,i]-model.DCLinesMP[l,i-1])*line.base_cost
            inv_cost += DC_Inv_lines
        if ACmode and DCmode:
            Conv_Inv=0
            for l in model.conv:
                conv = grid.Converters_ACDC[l]
                if i ==0:
                    Conv_Inv+=(model.ConvMP[l,i]-model.ConvMP_base[l])*conv.base_cost
                else:
                    Conv_Inv+=(model.ConvMP[l,i]-model.ConvMP[l,i-1])*conv.base_cost
            inv_cost += Conv_Inv


        instance_cost = model.inv_model[i].obj.expr + inv_cost
        net_cost += instance_cost/(1+discount_rate)**(i*n_years)
        model.inv_model[i].obj.deactivate()

    return net_cost

    




def export_MP_TEP_results_toPyflowACDC(model,grid,inv_periods,MINLP=False):
    ACmode,DCmode,ACadd,DCadd,GPR = analyse_OPF(grid)
    TEP_AC,TAP_tf,REC_AC,CT_AC = ACadd
    CFC = DCadd
    grid.MP_TEP_run=True
    
    
    rows = []
    

    if ACmode:
        if MINLP:
            ac_lines_mp_values = {(l, i): round(pyo.value(model.ACLinesMP[l, i])) for (l, i) in model.ACLinesMP}   
        else:
            ac_lines_mp_values = {(l, i): round(pyo.value(model.ACLinesMP[l, i]),2) for (l, i) in model.ACLinesMP}   
        for line in grid.lines_AC_exp:
            l = line.lineNumber
            row = {'Element': str(line.name)}
            row['Type'] = 'AC Line'
            row['Initial'] = pyo.value(model.NumLinesACP_base[l])
            total_cost = 0
            for i in range(len(inv_periods)):
                n_val = ac_lines_mp_values[l, i]
                if i == 0:
                    cost = (n_val - pyo.value(model.NumLinesACP_base[l])) * line.base_cost
                else:
                    cost = (n_val - ac_lines_mp_values[l, i-1]) * line.base_cost
                row[f"N_{inv_periods[i]}"] = n_val
                row[f"Cost_{inv_periods[i]}"] = cost
                total_cost += cost
            row['Total_Cost'] = total_cost
            rows.append(row)
        
        


    if DCmode:
        dc_lines_mp_values = {(l, i): pyo.value(model.DCLinesMP[l, i]) for (l, i) in model.DCLinesMP}

        for line in grid.lines_DC:  
            if line.np_line_opf:
                l = line.lineNumber
                row = {'Element': str(line.name)}
                row['Type'] = 'DC Line'
                row['Initial'] = pyo.value(model.NumLinesDCP_base[l])
                total_cost = 0
                for i in range(len(inv_periods)):
                    n_val = dc_lines_mp_values[l, i]
                    if i == 0:
                        cost = (n_val - pyo.value(model.NumLinesDCP_base[l])) * line.base_cost
                    else:
                        cost = (n_val - dc_lines_mp_values[l, i-1]) * line.base_cost
                    row[f"N_{inv_periods[i]}"] = n_val
                    row[f"Cost_{inv_periods[i]}"] = cost
                    total_cost += cost
                row['Total_Cost'] = total_cost
                rows.append(row)

    if ACmode and DCmode:
        acdc_conv_mp_values = {(c, i): pyo.value(model.ACDCConvMP[c, i]) for (c, i) in model.ACDCConvMP}
        for conv in grid.Converters_ACDC:
            c = conv.ConvNumber
            row = {'Element': str(conv.name)}
            row['Type'] = 'ACDC Conv'
            row['Initial'] = pyo.value(model.NumConvP_base[c])
            total_cost = 0
            for i in range(len(inv_periods)):
                n_val = acdc_conv_mp_values[c, i]   
                if i == 0:
                    cost = (n_val - pyo.value(model.NumConvP_base[c])) * conv.base_cost
                else:
                    cost = (n_val - acdc_conv_mp_values[c, i-1]) * conv.base_cost
                row[f"N_{inv_periods[i]}"] = n_val
                row[f"Cost_{inv_periods[i]}"] = cost
                total_cost += cost
            row['Total_Cost'] = total_cost
            rows.append(row)    

    df = pd.DataFrame(rows)
    total_row = {}
    for col in df.columns:
        if col == "Element":
            total_row[col] = "Total cost"
        elif "Cost" in col:
            total_row[col] = df[col].sum()
        else:
            total_row[col] = ""
    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
    
    grid.MP_TEP_results = df  