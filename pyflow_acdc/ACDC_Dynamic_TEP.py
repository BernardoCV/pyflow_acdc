import numpy as np
import pyomo.environ as pyo
import pandas as pd
import time
import math
from concurrent.futures import ThreadPoolExecutor


from .ACDC_OPF_NL_model import OPF_create_NLModel_ACDC,TEP_variables,ExportACDC_NLmodel_toPyflowACDC
from .ACDC_OPF import OPF_solve,OPF_obj,obj_w_rule,calculate_objective
from .ACDC_Static_TEP import get_TEP_variables,initialize_links,create_scenarios
from .Class_editor import analyse_grid
from .Time_series import _modify_parameters
from .Graph_and_plot import save_network_svg, create_geometries

__all__ = [
    'dynamic_transmission_expansion',
    'multi_period_MS_TEP',
    'save_MP_TEP_period_svgs'
]

def pack_variables(*args):
    return args

def _update_grid_investment_period(grid,inv,i):
    idx = i
    typ = inv.type
    if typ == 'Load':
        for price_zone in grid.Price_Zones:
            if inv.element_name == price_zone.name:
                price_zone.PLi_inv_factor = inv.data[idx]
                break
        for node in grid.nodes_AC:
            if inv.element_name == node.name:
                node.PLi_inv_factor=inv.data[idx]
                break
        for node in grid.nodes_DC:
            if inv.element_name == node.name:
                node.PLi_inv_factor=inv.data[idx]
                break

    if typ == 'elasticity':
        for price_zone in grid.Price_Zones:
            if inv.element_name == price_zone.name:
                price_zone.elasticity = inv.data[idx]
                break

    if typ == 'import_expand':
        for price_zone in grid.Price_Zones:
            if inv.element_name == price_zone.name:
                price_zone.import_expand = inv.data[idx]
                break

    if typ in ['WPP', 'OWPP','SF','REN']:
        for zone in grid.RenSource_zones:
            if inv.element_name == zone.name:
                zone.PRGi_inv_factor = inv.data[idx]
                break  # Stop after assigning to the correct zone
        for rs in grid.RenSources:
            if inv.element_name == rs.name:
                rs.PRGi_inv_factor = inv.data[idx]
                break  # Stop after assigning to the correct node

def MP_TEP_variables(model,grid):
    
    conv_var,DC_line_var,AC_line_var,gen_var = get_TEP_variables(grid)

    if grid.GPR:
        np_gen,np_gen_max = gen_var

        model.np_gen_base = pyo.Param(model.gen_AC,initialize=np_gen)
        def np_gen_bounds(model,gen,i):
            return (np_gen[gen],np_gen_max[gen])
            
        model.np_gen = pyo.Var(model.gen_AC,model.inv_periods,within=pyo.NonNegativeIntegers,bounds=np_gen_bounds)

        def MP_gen_lower_bound(model,gen,i):
            if i == 0:
                return pyo.Constraint.Skip
            else:
                return model.np_gen[gen,i] >= model.np_gen[gen,i-1]
        model.MP_gen_lower_bound_constraint = pyo.Constraint(model.gen_AC,model.inv_periods,rule=MP_gen_lower_bound)
        
        def MP_gen_link(model,gen,i):
            return model.inv_model[i].np_gen[gen] == model.np_gen[gen,i]
        model.MP_gen_link_constraint = pyo.Constraint(model.gen_AC,model.inv_periods,rule=MP_gen_link)

    
    if grid.ACmode:
        NP_lineAC,NP_lineAC_i,NP_lineAC_max,Line_length,REC_branch,ct_ini = AC_line_var
        if grid.TEP_AC:
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

    if grid.DCmode:
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


    if grid.ACmode and grid.DCmode:
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

def dynamic_transmission_expansion(grid,inv_periods=[],n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,solver='bonmin',time_limit=99999,tee=False,export=True):

    analyse_grid(grid)
    weights_def, PZ = obj_w_rule(grid,ObjRule,True,False)

    grid.TEP_n_years = n_years
    grid.TEP_discount_rate =discount_rate
                
    t1=time.time()

    model = pyo.ConcreteModel()
    model.name        ="Dynamic TEP MTDC AC/DC hybrid OPF"

    n_periods = len(grid.inv_series[0].data)

    model.inv_periods = pyo.Set(initialize=list(range(0,n_periods)))

    model.inv_model = pyo.Block(model.inv_periods)

    base_model = pyo.ConcreteModel()
    OPF_create_NLModel_ACDC(base_model,grid,PV_set=False,Price_Zones=PZ,TEP=True)

    for element in grid.Generators + grid.lines_AC_exp + grid.lines_DC + grid.Converters_ACDC: 
        _calculate_decomision_period(element,n_years)
        

    for i in model.inv_periods:
        base_model_copy = base_model.clone()
        model.inv_model[i].transfer_attributes_from(base_model_copy)
       
        for inv in grid.inv_series:    
            _update_grid_investment_period(grid,inv,i)

        _modify_parameters(grid,model.inv_model[i],grid.ACmode,grid.DCmode,PZ)

        
        obj_OPF = OPF_obj(model.inv_model[i],grid,weights_def,True)
        
        present_value =   Hy*(1 - (1 + discount_rate) ** -n_years) / discount_rate
        
        obj_OPF *=present_value
        
        model.inv_model[i].obj = pyo.Objective(rule=obj_OPF, sense=pyo.minimize)

    initialize_links(model,grid)
    MP_TEP_variables(model,grid)
    

    net_cost = MP_TEP_obj(model,grid,n_years,discount_rate)
    model.obj = pyo.Objective(rule=net_cost, sense=pyo.minimize)
    
    t2 = time.time()

    model_results,solver_stats = OPF_solve(model,grid,solver)
    
    t3 = time.time()
    ExportACDC_NLmodel_toPyflowACDC(model.inv_model[i], grid, PZ,TEP=True)
    
    MINLP = False
    if solver != 'ipopt':
        MINLP = True
    
    export_MP_TEP_results_toPyflowACDC(model,grid,MINLP)
    t4 = time.time()




   
    timing_info = {
    "create": t2-t1,
    "solve": solver_stats['time'],
    "export": t4-t3,
    }

    
    
    return model, model_results ,timing_info, solver_stats
    


def MP_TEP_obj(model,grid,n_years,discount_rate):
    
    net_cost = 0

    for i in model.inv_periods:
        inv_gen= 0
        AC_Inv_lines=0
        DC_Inv_lines=0
        Conv_Inv=0
        if grid.GPR:
            
            for g in model.gen_AC:
                gen = grid.Generators[g]
                if i == 0:
                    inv_gen+=(model.np_gen[g,i]-model.np_gen_base[g])*gen.base_cost
                else:
                    inv_gen+=(model.np_gen[g,i]-model.np_gen[g,i-1])*gen.base_cost
        else:
            inv_gen=0


        if grid.ACmode:
            if grid.TEP_AC:
                
                for l in model.lines_AC_exp:
                    line = grid.lines_AC_exp[l]
                    if i ==0:
                        AC_Inv_lines+=(model.ACLinesMP[l,i]-model.NumLinesACP_base[l])*line.base_cost
                    else:
                        AC_Inv_lines+=(model.ACLinesMP[l,i]-model.ACLinesMP[l,i-1])*line.base_cost
                
        if grid.DCmode:
            
            for l in model.lines_DC:
                line = grid.lines_DC[l]
                if i ==0:
                    DC_Inv_lines+=(model.DCLinesMP[l,i]-model.NumLinesDCP_base[l])*line.base_cost
                else:
                    DC_Inv_lines+=(model.DCLinesMP[l,i]-model.DCLinesMP[l,i-1])*line.base_cost
            
        if grid.ACmode and grid.DCmode:
            
            for l in model.conv:
                conv = grid.Converters_ACDC[l]
                if i ==0:
                    Conv_Inv+=(model.ConvMP[l,i]-model.ConvMP_base[l])*conv.base_cost
                else:
                    Conv_Inv+=(model.ConvMP[l,i]-model.ConvMP[l,i-1])*conv.base_cost
            

        inv_cost=inv_gen+AC_Inv_lines+DC_Inv_lines+Conv_Inv
        instance_cost = model.inv_model[i].obj.expr + inv_cost
        net_cost += instance_cost/(1+discount_rate)**(i*n_years)
        model.inv_model[i].obj.deactivate()

    return net_cost

    




def export_MP_TEP_results_toPyflowACDC(model,grid,Price_Zones=False,MINLP=False):
    

    grid.MP_TEP_run=True
    
    n_periods = len(grid.inv_series[0].data)
    
    rows = []
    
    if grid.GPR:
        if MINLP:
            gen_mp_values = {(g, i): round(pyo.value(model.np_gen[g, i])) for (g, i) in model.np_gen}
        else:
            gen_mp_values = {(g, i): round(pyo.value(model.np_gen[g, i]),2) for (g, i) in model.np_gen}
        for gen in grid.Generators:
            g = gen.genNumber
            gen.np_dynamic = [gen_mp_values[g, i] for i in range(n_periods)]
            row = {'Element': str(gen.name)}
            row['Type'] = 'Generator'
            row['Initial'] = pyo.value(model.np_gen_base[g])
            total_cost = 0
            for i in range(n_periods):
                n_val = gen_mp_values[g, i]
                if i == 0:
                    cost = (n_val - pyo.value(model.np_gen_base[g])) * gen.base_cost
                else:
                    cost = (n_val - gen_mp_values[g, i-1]) * gen.base_cost
                row[f"N_{i}"] = n_val
                row[f"Cost_{i}"] = cost
                total_cost += cost
            row['Total_Cost'] = total_cost
            rows.append(row)

    if grid.ACmode:
        if grid.TEP_AC:
            if MINLP:
                ac_lines_mp_values = {(l, i): round(pyo.value(model.ACLinesMP[l, i])) for (l, i) in model.ACLinesMP}   
            else:
                ac_lines_mp_values = {(l, i): round(pyo.value(model.ACLinesMP[l, i]),2) for (l, i) in model.ACLinesMP}   
            for line in grid.lines_AC_exp:
                l = line.lineNumber
                line.np_dynamic = [ac_lines_mp_values[l, i] for i in range(n_periods)]
                row = {'Element': str(line.name)}
                row['Type'] = 'AC Line'
                row['Initial'] = pyo.value(model.NumLinesACP_base[l])
                total_cost = 0
                for i in range(n_periods):
                    n_val = ac_lines_mp_values[l, i]
                    if i == 0:
                        cost = (n_val - pyo.value(model.NumLinesACP_base[l])) * line.base_cost
                    else:
                        cost = (n_val - ac_lines_mp_values[l, i-1]) * line.base_cost
                    row[f"N_{i}"] = n_val
                    row[f"Cost_{i}"] = cost
                    total_cost += cost
                row['Total_Cost'] = total_cost
                rows.append(row)
        
        


    if grid.DCmode:
        dc_lines_mp_values = {(l, i): pyo.value(model.DCLinesMP[l, i]) for (l, i) in model.DCLinesMP}

        for line in grid.lines_DC:  
            if line.np_line_opf:
                l = line.lineNumber
                line.np_dynamic = [dc_lines_mp_values[l, i] for i in range(n_periods)]
                row = {'Element': str(line.name)}
                row['Type'] = 'DC Line'
                row['Initial'] = pyo.value(model.NumLinesDCP_base[l])
                total_cost = 0
                for i in range(n_periods):
                    n_val = dc_lines_mp_values[l, i]
                    if i == 0:
                        cost = (n_val - pyo.value(model.NumLinesDCP_base[l])) * line.base_cost
                    else:
                        cost = (n_val - dc_lines_mp_values[l, i-1]) * line.base_cost
                    row[f"N_{i}"] = n_val
                    row[f"Cost_{i}"] = cost
                    total_cost += cost
                row['Total_Cost'] = total_cost
                rows.append(row)

    if grid.ACmode and grid.DCmode:
        acdc_conv_mp_values = {(c, i): pyo.value(model.ACDCConvMP[c, i]) for (c, i) in model.ACDCConvMP}
        for conv in grid.Converters_ACDC:
            c = conv.ConvNumber
            conv.np_dynamic = [acdc_conv_mp_values[c, i] for i in range(n_periods)]
            row = {'Element': str(conv.name)}
            row['Type'] = 'ACDC Conv'
            row['Initial'] = pyo.value(model.NumConvP_base[c])
            total_cost = 0
            for i in range(n_periods):
                n_val = acdc_conv_mp_values[c, i]   
                if i == 0:
                    cost = (n_val - pyo.value(model.NumConvP_base[c])) * conv.base_cost
                else:
                    cost = (n_val - acdc_conv_mp_values[c, i-1]) * conv.base_cost
                row[f"N_{i}"] = n_val
                row[f"Cost_{i}"] = cost
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
    

    ExportACDC_NLmodel_toPyflowACDC(model.inv_model[-1],grid,Price_Zones,TEP=True)

    grid.MP_TEP_results = df  

def multi_period_MS_TEP(grid, NPV=True, n_years=10, Hy=8760, 
                       discount_rate=0.02, clustering_options=None, ObjRule=None, 
                       solver='bonmin'):
    """
    Multi-period Transmission Expansion Planning with time series clustering.
    Hierarchical model structure:
    - Level 1: Investment periods
    - Level 2: Time frames/scenarios for each investment period
    """
    # 1. Initial analysis and setup
    analyse_grid(grid)

    weights_def, Price_Zones = obj_w_rule(grid, ObjRule, True)

    # 2. Set grid parameters
    grid.TEP_n_years = n_years
    grid.TEP_discount_rate = discount_rate

    # 3. Handle time series clustering
    try:
        from .Time_series_clustering import cluster_analysis
        n_clusters,clustering = cluster_analysis(grid,clustering_options)
    except:
        n_clusters = len(grid.Time_series[0].data)
        clustering = False

    # 4. Create model sets
    t1 = time.time()
    model = pyo.ConcreteModel()
    model.name = "MP TEP MS MTDC AC/DC hybrid OPF"
    
    # Investment periods
    n_periods = len(grid.inv_series[0].data)
    model.inv_periods = pyo.Set(initialize=list(range(0, n_periods)))
    
    # Create hierarchical model structure
    model.inv_model = pyo.Block(model.inv_periods)  # Level 1: Investment periods
    for i in model.inv_periods:
        # Time frames for each investment period
        model.inv_model[i].scenario_frames = pyo.Set(initialize=range(1, n_clusters + 1))
        model.inv_model[i].submodel = pyo.Block(model.inv_model[i].scenario_frames)  # Level 2: Time frames

    # 5. Create base model and clone for each period/time frame
    base_model = pyo.ConcreteModel()
    OPF_create_NLModel_ACDC(base_model, grid, PV_set=False, Price_Zones=Price_Zones, TEP=True)

    create_scenarios(model.inv_model[i],grid,Price_Zones,weights_def,n_clusters,clustering,NPV,n_years,discount_rate,Hy)

    initialize_links(model,grid)
    MP_TEP_variables(model,grid)

    
    net_cost = MP_TEP_obj(model,grid,n_years,discount_rate)
    model.obj = pyo.Objective(rule=net_cost, sense=pyo.minimize)


    # 10. Solve the model
    t2 = time.time()
    model_results, solver_stats = OPF_solve(model, grid, solver)
    t3 = time.time()

    # 11. Export results
    MINLP = False
    if solver != 'ipopt':
        MINLP = True
    TEP_TS_res = export_MP_TEP_results_toPyflowACDC(model, grid,Price_Zones,MINLP)
    t4 = time.time()

    timing_info = {
        "create": t2-t1,
        "solve": solver_stats['time'],
        "export": t4-t3,
    }

    return model, model_results, timing_info, solver_stats, TEP_TS_res  


def save_MP_TEP_period_svgs(grid, name_prefix='grid_MP_TEP', journal=True, legend=True, square_ratio=False, poly=None, linestrings=None):
    
    periods = len(grid.inv_series[0].data)
   
    for i in range(periods):
            # From DataFrame by names
        _set_grid_to_dynamic_state(grid, i) 

        try:
            create_geometries(grid)
        except Exception:
            pass

        save_network_svg(
            grid,
            name=f"{name_prefix}_P{i}",
            journal=journal,
            legend=legend,
            square_ratio=square_ratio,
            poly=poly,
            linestrings=linestrings
        )

    return

def _set_grid_to_dynamic_state(grid, investment_period):    
    for line in grid.lines_AC_exp:
        line.np_line = line.np_dynamic[investment_period]
    for line in grid.lines_DC:
        line.np_line = line.np_dynamic[investment_period]
    for conv in grid.Converters_ACDC:
        conv.NumConvP = conv.np_dynamic[investment_period]

def _calculate_decomision_period(element,n_years):

    element.decomision_period = math.ceil(element.life_time/n_years)

    