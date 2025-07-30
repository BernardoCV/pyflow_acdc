# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:25:02 2024

@author: BernardoCastro
Gurobi version of AC OPF Linear Model
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time

__all__ = ['Optimal_L_CSS_gurobi']

from .ACDC_OPF import analyse_OPF, obj_w_rule, calculate_objective


def print_gurobi_model(model, gen_vars=None, ac_vars=None, detailed=True):
    """
    Print Gurobi model information similar to Pyomo's pprint()
    
    Args:
        model: Gurobi model object
        gen_vars: Generation variables dictionary (optional)
        ac_vars: AC variables dictionary (optional)
        detailed: If True, print detailed variable and constraint information
    """
    print("=" * 80)
    print("GUROBI MODEL SUMMARY")
    print("=" * 80)
    
    # Basic model info
    print(f"Model Name: {model.ModelName}")
    print(f"Number of Variables: {model.NumVars}")
    print(f"Number of Constraints: {model.NumConstrs}")
    print(f"Number of Binary Variables: {model.NumBinVars}")
    print(f"Number of Integer Variables: {model.NumIntVars}")
    print(f"Model Status: {model.status}")
    
    if model.status == GRB.OPTIMAL:
        print(f"Objective Value: {model.objVal:.6f}")
    
    print("\n" + "=" * 80)
    print("VARIABLES SUMMARY")
    print("=" * 80)
    
    # Print variable information
    if gen_vars and ac_vars:
        print("\nGeneration Variables:")
        for var_type, var_dict in gen_vars.items():
            print(f"  {var_type}: {len(var_dict)} variables")
            if detailed and var_dict:
                for key, var in var_dict.items():  # Show ALL variables
                    print(f"    {var.VarName}: [{var.LB:.2f}, {var.UB:.2f}]")
        
        print("\nAC Variables:")
        for var_type, var_dict in ac_vars.items():
            print(f"  {var_type}: {len(var_dict)} variables")
            if detailed and var_dict:
                for key, var in var_dict.items():  # Show ALL variables
                    print(f"    {var.VarName}: [{var.LB:.2f}, {var.UB:.2f}]")
    
    print("\n" + "=" * 80)
    print("CONSTRAINTS SUMMARY")
    print("=" * 80)
    
    # Print constraint information
    if detailed:
        print("\nALL CONSTRAINTS:")
        for i, constr in enumerate(model.getConstrs()):
            # Get the constraint expression
            row = model.getRow(constr)
            expr_str = ""
            for j in range(row.size()):
                var = row.getVar(j)
                coeff = row.getCoeff(j)
                if coeff != 0:
                    if coeff > 0 and expr_str:
                        expr_str += " + "
                    elif coeff < 0:
                        expr_str += " - " if expr_str else "-"
                    expr_str += f"{abs(coeff)}*{var.VarName}"
            
            # Add RHS to the expression
            if constr.RHS != 0:
                if constr.RHS > 0:
                    expr_str += f" + {constr.RHS}"
                else:
                    expr_str += f" - {abs(constr.RHS)}"
            
            print(f"  {i+1}: {constr.ConstrName}: {expr_str} {constr.Sense} 0")
    else:
        print(f"Total Constraints: {model.NumConstrs}")
    
    print("\n" + "=" * 80)
    print("SOLVER STATISTICS")
    print("=" * 80)
    
    # Print solver statistics
    if hasattr(model, 'Runtime'):
        print(f"Solve Time: {model.Runtime:.4f} seconds")
    if hasattr(model, 'BarrierIterCount'):
        print(f"Barrier Iterations: {model.BarrierIterCount}")
    if hasattr(model, 'NodeCount'):
        print(f"Nodes Explored: {model.NodeCount}")
    if hasattr(model, 'IterCount'):
        print(f"Simplex Iterations: {model.IterCount}")
    
    print("=" * 80)


def add_pprint_to_model(model, gen_vars=None, ac_vars=None):
    """
    Add a pprint method to the Gurobi model object for easy access
    """
    def pprint(detailed=True):
        print_gurobi_model(model, gen_vars, ac_vars, detailed)
    
    # Store the pprint function in the model's _pprint attribute
    # We'll use a try-except to handle the case where we can't add attributes
    try:
        model._pprint = pprint
    except:
        # If we can't add attributes, we'll just return the function
        pass
    return model


def debug_infeasibility(model, gen_vars=None, ac_vars=None):
    """
    Debug infeasibility by computing IIS and showing problematic constraints
    """
    print("=" * 80)
    print("INFEASIBILITY DEBUGGING")
    print("=" * 80)
    
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible. Computing IIS (Irreducible Infeasible Subsystem)...")
        
        # Compute IIS
        model.computeIIS()
        
        # Get IIS constraints
        iis_constrs = [c for c in model.getConstrs() if c.IISConstr]
        iis_vars = [v for v in model.getVars() if v.IISLB or v.IISUB]
        
        print(f"\nFound {len(iis_constrs)} constraints in IIS:")
        for constr in iis_constrs:
            print(f"  - {constr.ConstrName}: {constr.Sense} {constr.RHS}")
        
        print(f"\nFound {len(iis_vars)} variables with bounds in IIS:")
        for var in iis_vars:
            if var.IISLB:
                print(f"  - {var.VarName}: LB = {var.LB}")
            if var.IISUB:
                print(f"  - {var.VarName}: UB = {var.UB}")
        
        # Show some variable bounds that might be problematic
        if gen_vars and ac_vars:
            print("\nChecking variable bounds:")
            
            print("\nGeneration Variables:")
            for var_type, var_dict in gen_vars.items():
                for key, var in var_dict.items():
                    if var.LB == var.UB and var.LB != 0:
                        print(f"  {var.VarName}: Fixed at {var.LB}")
                    elif var.LB > var.UB:
                        print(f"  {var.VarName}: LB ({var.LB}) > UB ({var.UB}) - INVALID!")
            
            print("\nAC Variables:")
            for var_type, var_dict in ac_vars.items():
                for key, var in var_dict.items():
                    if var.LB == var.UB and var.LB != 0:
                        print(f"  {var.VarName}: Fixed at {var.LB}")
                    elif var.LB > var.UB:
                        print(f"  {var.VarName}: LB ({var.LB}) > UB ({var.UB}) - INVALID!")
    
    elif model.status == GRB.UNBOUNDED:
        print("Model is unbounded. Check for missing constraints or incorrect bounds.")
    
    else:
        print(f"Model status: {model.status}")
    
    print("=" * 80)


def Optimal_L_CSS_gurobi(grid, OPEX=True, NPV=True, n_years=25, Hy=8760, discount_rate=0.02,tee=False):
    """Main function to create and solve Gurobi model"""
    
    analyse_OPF(grid)
    if not grid.CT_AC:
        raise ValueError("No conductor size selection connections found in the grid")
    
    # Create model
    model = gp.Model("ACDC_OPF")
    
    # Enable automatic Benders decomposition - CORRECT PARAMETER NAMES
    try:
        model.setParam('BendersStrategy', 2)  # 2 = automatic Benders
        model.setParam('BendersCuts', 2)      # 2 = automatic cut generation
        print("✓ Benders decomposition enabled")
    except Exception as e:
        print(f"⚠ Benders parameters not available: {e}")
        # Fallback to general decomposition
        try:
            model.setParam('DecompositionStrategy', 1)
            print("✓ General decomposition enabled")
        except:
            print("⚠ Using default solver settings")
    
    # Set basic Gurobi parameters
    model.setParam('OutputFlag', 1 if tee else 0)        
    t1 = time.time()
    model, gen_vars, ac_vars = OPF_create_LModel_ACDC_gurobi(model,grid)
    t2 = time.time()  
    t_modelcreate = t2 - t1
    
    # Add pprint method to model for easy inspection
    add_pprint_to_model(model, gen_vars, ac_vars)
    
    # Set objective function for Gurobi model
    set_objective(model, grid,gen_vars,ac_vars,OPEX,NPV, n_years, Hy, discount_rate)
    
    t3 = time.time()
    model_res, solver_stats = solve_gurobi_model(model, grid)
    t4 = time.time()
    
    # Export results
    ExportACDC_Lmodel_toPyflowACDC_gurobi(model, grid,gen_vars,ac_vars)
    
    if OPEX:
        obj = {'Energy_cost': 1}
    else:
        obj = None

    weights_def, _ = obj_w_rule(grid, obj, True)
    # Calculate objective values
    for obj in weights_def:
        weights_def[obj]['v'] = calculate_objective(grid, obj, True)
    
    t5 = time.time()  
    t_modelexport = t5 - t4
    
    grid.OPF_run = True 
    grid.OPF_obj = weights_def
    
    timing_info = {
        "create": t_modelcreate,
        "solve": solver_stats['time'] if solver_stats['time'] is not None else t4 - t3,
        "export": t_modelexport,
    }
    
    return model, model_res, timing_info, solver_stats


def solve_gurobi_model(model, grid):
    """Solve Gurobi model and return results"""
    
    try:
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            model_res = {
                'status': 'optimal',
                'objective_value': model.objVal,
                'solver_time': model.Runtime,
                'iterations': getattr(model, 'BarrierIterCount', None),
                'nodes': getattr(model, 'NodeCount', None)
            }
        elif model.status == GRB.INFEASIBLE:
            model_res = {
                'status': 'infeasible',
                'objective_value': None,
                'solver_time': model.Runtime
            }
        elif model.status == GRB.UNBOUNDED:
            model_res = {
                'status': 'unbounded',
                'objective_value': None,
                'solver_time': model.Runtime
            }
        else:
            model_res = {
                'status': f'other_{model.status}',
                'objective_value': None,
                'solver_time': model.Runtime
            }
        
        solver_stats = {
            'time': model.Runtime,
            'status': model.status,
            'iterations': getattr(model, 'BarrierIterCount', None),
            'nodes': getattr(model, 'NodeCount', None)
        }
        
    except Exception as e:
        model_res = {
            'status': 'error',
            'error_message': str(e),
            'objective_value': None,
            'solver_time': None
        }
        solver_stats = {
            'time': None,
            'status': 'error',
            'error_message': str(e)
        }
    
    return model_res, solver_stats


def OPF_create_LModel_ACDC_gurobi(model,grid):
    """Create Gurobi model for AC DC OPF"""
    from .ACDC_OPF import Translate_pyf_OPF 
    from .ACDC_TEP import get_TEP_variables  # Add this import
    
    
    
    [AC_info,DC_info,Conv_info,Price_Zone_info,gen_info]=Translate_pyf_OPF(grid,False)
    
    # Get TEP variables for initialization
    conv_var,DC_line_var,AC_line_var,gen_var = get_TEP_variables(grid)
    NP_lineAC,NP_lineAC_i,NP_lineAC_max,Line_length,REC_branch,ct_ini = AC_line_var
    
    gen_vars = Generation_variables_gurobi(model, grid, gen_info)
    ac_vars = AC_variables_gurobi(model, grid, AC_info, ct_ini)  # Pass ct_ini

    AC_constraints_gurobi(model, grid, AC_info, gen_info, gen_vars, ac_vars)
    
    return model, gen_vars, ac_vars


def Generation_variables_gurobi(model, grid, gen_info):
    """Convert generation variables to Gurobi"""
    gen_AC_info, gen_DC_info, P_renSource, lista_rs = gen_info
    lf, qf, fc, np_gen, lista_gen = gen_AC_info
    
    # Create a dictionary to store variables
    variables = {}
    
    # Renewable sources
    variables['gamma'] = {}
    
    for rs in lista_rs:
        ren_source = grid.RenSources[rs]
        
        # Curtailment factor
        if ren_source.curtailable:
            lb, ub = ren_source.min_gamma, 1.0
        else:
            lb, ub = 1.0, 1.0
        variables['gamma'][rs] = model.addVar(lb=lb, ub=ub, name=f"gamma_{rs}")
        # Set initial value to 1 to match Pyomo
        variables['gamma'][rs].Start = 1
    

    # AC Generators
    variables['PGi_gen'] = {}
    variables['lf'] = {}
    for g in lista_gen:
        gen = grid.Generators[g]

        # Power bounds
        p_lb = gen.Min_pow_gen * gen.np_gen
        p_ub = gen.Max_pow_gen * gen.np_gen
        
        variables['PGi_gen'][g] = model.addVar(lb=p_lb, ub=p_ub, name=f"PGi_gen_{g}")
      
    return variables


def AC_variables_gurobi(model, grid, AC_info, ct_ini):
    """Convert AC variables to Gurobi"""
    AC_Lists, AC_nodes_info, AC_lines_info, EXP_info, REC_info, CT_info = AC_info
    
    lista_nodos_AC, lista_lineas_AC, lista_lineas_AC_tf, AC_slack, AC_PV = AC_Lists
    u_min_ac, u_max_ac, V_ini_AC, Theta_ini, P_know, Q_know, price = AC_nodes_info
    S_lineAC_limit, S_lineACtf_limit, m_tf_og = AC_lines_info

    lista_lineas_AC_exp, S_lineACexp_limit, NP_lineAC = EXP_info
    lista_lineas_AC_rec, S_lineACrec_lim, S_lineACrec_lim_new, grid.REC_AC_act = REC_info
    lista_lineas_AC_ct, S_lineACct_lim, cab_types_set, allowed_types = CT_info

    # Create a dictionary to store variables
    variables = {}

    # Node variables
    variables['thetha_AC'] = {}
    variables['PGi_ren'] = {}
    variables['PGi_opt'] = {}
    
    for node in lista_nodos_AC:
        nAC = grid.nodes_AC[node]
        
        # Voltage angle
        variables['thetha_AC'][node] = model.addVar(lb=-1.6, ub=1.6, name=f"thetha_AC_{node}")
        
        # Renewable power injection
        if nAC.connected_RenSource == []:
            variables['PGi_ren'][node] = model.addVar(lb=0, ub=0, name=f"PGi_ren_{node}")
        else:
            variables['PGi_ren'][node] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"PGi_ren_{node}")
        
        # Generator power injection
        if nAC.connected_gen == []:
            variables['PGi_opt'][node] = model.addVar(lb=0, ub=0, name=f"PGi_opt_{node}")
        else:
            variables['PGi_opt'][node] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"PGi_opt_{node}")
    
    # CT variables (Investment decisions - Master problem)
    variables['Pto_CT'] = {}
    variables['Pfrom_CT'] = {}
    variables['ct_branch'] = {}
    variables['ct_PAC_to'] = {}
    variables['ct_PAC_from'] = {}
    variables['z_to'] = {}
    variables['z_from'] = {}
    variables['ct_types'] = {}  
    
    for node in lista_nodos_AC:
        nAC = grid.nodes_AC[node]
        
        if nAC.connected_toCTLine:
            variables['Pto_CT'][node] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Pto_CT_{node}")
        else:
            variables['Pto_CT'][node] = model.addVar(lb=0, ub=0, name=f"Pto_CT_{node}")
        
        if nAC.connected_fromCTLine:
            variables['Pfrom_CT'][node] = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"Pfrom_CT_{node}")
        else:
            variables['Pfrom_CT'][node] = model.addVar(lb=0, ub=0, name=f"Pfrom_CT_{node}")
    
    for line in lista_lineas_AC_ct:
        for ct in cab_types_set:
            
            init_val = ct_ini[line, ct]
            
            variables['ct_branch'][line, ct] = model.addVar(
                vtype=GRB.BINARY, name=f"ct_branch_{line}_{ct}"
            )
            variables['ct_branch'][line, ct].Start = init_val
            
            # Power flow variables should allow negative values (bidirectional flow)
            max_min = max(S_lineACct_lim[line,ct] for ct in cab_types_set)
            variables['ct_PAC_to'][line, ct] = model.addVar(
                lb=-max_min, ub=max_min, name=f"ct_PAC_to_{line}_{ct}"
            )
            variables['ct_PAC_from'][line, ct] = model.addVar(
                lb=-max_min, ub=max_min, name=f"ct_PAC_from_{line}_{ct}"
            )
            variables['z_to'][line, ct] = model.addVar(
                lb=-max_min, ub=max_min, name=f"z_to_{line}_{ct}"
            )
            variables['z_from'][line, ct] = model.addVar(
                lb=-max_min, ub=max_min, name=f"z_from_{line}_{ct}"
            )
    
    # Add ct_types variables
    for ct in cab_types_set:
        variables['ct_types'][ct] = model.addVar(
            vtype=GRB.BINARY, name=f"ct_types_{ct}"
        )
    
    # Line variables (Operational decisions - Subproblems)
    variables['PAC_to'] = {}
    variables['PAC_from'] = {}
    variables['PAC_line_loss'] = {}
    
    for line in lista_lineas_AC:
        variables['PAC_to'][line] = model.addVar(
            lb=-S_lineAC_limit[line], ub=S_lineAC_limit[line],
            name=f"PAC_to_{line}"
        )
        variables['PAC_from'][line] = model.addVar(
            lb=-S_lineAC_limit[line], ub=S_lineAC_limit[line],
            name=f"PAC_from_{line}"
        )
        variables['PAC_line_loss'][line] = model.addVar(name=f"PAC_line_loss_{line}")
    
    # Slack bus constraints
    for node in AC_slack:
        model.addConstr(
            variables['thetha_AC'][node] == 0,
            name=f"slack_theta_{node}"
        )

    return variables


def AC_constraints_gurobi(model, grid, AC_info, gen_info, gen_vars, ac_vars):
    """Convert AC constraints to Gurobi"""
    AC_Lists, AC_nodes_info, AC_lines_info, EXP_info, REC_info, CT_info = AC_info
    lista_nodos_AC, lista_lineas_AC, lista_lineas_AC_tf, AC_slack, AC_PV = AC_Lists
    u_min_ac, u_max_ac, V_ini_AC, Theta_ini, P_know, Q_know, price = AC_nodes_info
    S_lineAC_limit, S_lineACtf_limit, m_tf_og = AC_lines_info

    lista_lineas_AC_exp, S_lineACexp_limit, NP_lineAC = EXP_info
    lista_lineas_AC_rec, S_lineACrec_lim, S_lineACrec_lim_new, grid.REC_AC_act = REC_info
    lista_lineas_AC_ct, S_lineACct_lim, cab_types_set, allowed_types = CT_info

    gen_AC_info,gen_DC_info,P_renSource,lista_rs = gen_info

    # Power balance constraints
    for node in lista_nodos_AC:
        nAC = grid.nodes_AC[node]
        
        # Power balance equation
        power_sum = 0
        for k in lista_nodos_AC:
            if grid.Ybus_AC[node, k] != 0:
                power_sum += -np.imag(grid.Ybus_AC[node, k]) * (
                    ac_vars['thetha_AC'][node] - ac_vars['thetha_AC'][k]
                )
        
        # Add investment-related power flows
        power_sum += ac_vars['Pto_CT'][node] + ac_vars['Pfrom_CT'][node]
        
        # Power balance constraint
        model.addConstr(
            power_sum == P_know[node] + ac_vars['PGi_ren'][node] + ac_vars['PGi_opt'][node],
            name=f"power_balance_{node}"
        )
    
        # Generator power injection
        gen_power = 0
        for gen in nAC.connected_gen:
            gen_power += gen_vars['PGi_gen'][gen.genNumber]
        model.addConstr(
            ac_vars['PGi_opt'][node] == gen_power,
            name=f"gen_power_{node}"
        )
        
        # Renewable power injection
        ren_power = 0
        for rs in nAC.connected_RenSource:
            ren_power += P_renSource[rs.rsNumber] * gen_vars['gamma'][rs.rsNumber]
        model.addConstr(
            ac_vars['PGi_ren'][node] == ren_power,
            name=f"ren_power_{node}"
        )
    
        # CT power flows
        to_ct_sum = 0
        for line in nAC.connected_toCTLine:
            for ct in cab_types_set:
                to_ct_sum += ac_vars['z_to'][line.lineNumber, ct]
        
        model.addConstr(
            ac_vars['Pto_CT'][node] == to_ct_sum,
            name=f"to_ct_{node}"
        )
        
        # From cable type lines
        from_ct_sum = 0
        for line in nAC.connected_fromCTLine:
            for ct in cab_types_set:
                from_ct_sum += ac_vars['z_from'][line.lineNumber, ct]
        model.addConstr(
            ac_vars['Pfrom_CT'][node] == from_ct_sum,
            name=f"from_ct_{node}"
        )

    # Line flow constraints
    for line in lista_lineas_AC:
        l = grid.lines_AC[line]
        f = l.fromNode.nodeNumber
        t = l.toNode.nodeNumber
        
        # Power flow equations (DC approximation)
        B = np.imag(l.Ybus_branch[0, 1])
        
        model.addConstr(
            ac_vars['PAC_to'][line] == -B * (ac_vars['thetha_AC'][t] - ac_vars['thetha_AC'][f]),
            name=f"power_flow_to_{line}"
        )
        
        model.addConstr(
            ac_vars['PAC_from'][line] == -B * (ac_vars['thetha_AC'][f] - ac_vars['thetha_AC'][t]),
            name=f"power_flow_from_{line}"
        )
    
    # Cable type constraints
    model.addConstr(
        sum(ac_vars['ct_types'][ct] for ct in cab_types_set) <= grid.cab_types_allowed,
        name="CT_limit_rule"
    )
    
    # Cable types upper bound - if cable type is selected, it must be used
    for ct in cab_types_set:
        model.addConstr(
            sum(ac_vars['ct_branch'][line, ct] for line in lista_lineas_AC_ct) <= len(lista_lineas_AC_ct) * ac_vars['ct_types'][ct],
            name=f"ct_types_upper_bound_{ct}"
        )
    
    # Cable types lower bound - if cable type is used, it must be selected
    for ct in cab_types_set:
        model.addConstr(
            ac_vars['ct_types'][ct] <= sum(ac_vars['ct_branch'][line, ct] for line in lista_lineas_AC_ct),
            name=f"ct_types_lower_bound_{ct}"
        )
    
    # Array cable type rule - at most one cable type per line (for array mode)
    for line in lista_lineas_AC_ct:
        model.addConstr(
            sum(ac_vars['ct_branch'][line, ct] for ct in cab_types_set) <= 1,
            name=f"ct_Array_cable_type_rule_{line}"
        )
   
    # Node limit rule - limit cable types per node
    for node in lista_nodos_AC:
        nAC = grid.nodes_AC[node]
        if hasattr(nAC, 'ct_limit'):
            connections = 0
            for line in nAC.connected_toCTLine:
                for ct in cab_types_set:
                    connections += ac_vars['ct_branch'][line.lineNumber, ct]
            for line in nAC.connected_fromCTLine:
                for ct in cab_types_set:
                    connections += ac_vars['ct_branch'][line.lineNumber, ct]
            model.addConstr(
                connections <= nAC.ct_limit,
                name=f"ct_node_limit_rule_{node}"
            )
   
    # Crossings rule - limit cable types in crossing groups   
    for ct_crossing in grid.crossing_groups:
        model.addConstr(
            sum(ac_vars['ct_branch'][line, ct] for line in grid.crossing_groups[ct_crossing] for ct in cab_types_set) <= 1,
            name=f"ct_crossings_rule_{ct_crossing}"
        )

    # McCormick envelope constraints for z variables
    for line in lista_lineas_AC_ct:
        l = grid.lines_AC_ct[line]
        
        # Calculate M once per line (outside the ct loop)
        M = max(S_lineACct_lim[line, ct] for ct in cab_types_set) * 1.1
        
        for ct in cab_types_set:
            # Power flow for this cable type
            f = l.fromNode.nodeNumber
            t = l.toNode.nodeNumber
            B = np.imag(l.Ybus_list[ct][0, 1])
            
            # Power flow constraints
            model.addConstr(
                ac_vars['ct_PAC_to'][line, ct] + B * (ac_vars['thetha_AC'][t] - ac_vars['thetha_AC'][f]) == 0,
                name=f"ct_power_flow_to_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['ct_PAC_from'][line, ct] + B * (ac_vars['thetha_AC'][f] - ac_vars['thetha_AC'][t]) == 0,
                name=f"ct_power_flow_from_{line}_{ct}"
            )
            
            # McCormick envelopes for z_to
            model.addConstr(
                ac_vars['z_to'][line, ct] <= ac_vars['ct_PAC_to'][line, ct] + (1 - ac_vars['ct_branch'][line, ct]) * (2*M),
                name=f"z_to_ub_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['z_to'][line, ct] >= ac_vars['ct_PAC_to'][line, ct] - (1 - ac_vars['ct_branch'][line, ct]) * (2*M),
                name=f"z_to_lb_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['z_to'][line, ct] <= S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct],
                name=f"z_to_branch_ub_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['z_to'][line, ct] >= -S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct],
                name=f"z_to_branch_lb_{line}_{ct}"
            )
            
            # McCormick envelopes for z_from
            model.addConstr(
                ac_vars['z_from'][line, ct] <= ac_vars['ct_PAC_from'][line, ct] + (1 - ac_vars['ct_branch'][line, ct]) * (2*M),
                name=f"z_from_ub_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['z_from'][line, ct] >= ac_vars['ct_PAC_from'][line, ct] - (1 - ac_vars['ct_branch'][line, ct]) * (2*M),
                name=f"z_from_lb_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['z_from'][line, ct] <= S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct],
                name=f"z_from_branch_ub_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['z_from'][line, ct] >= -S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct],
                name=f"z_from_branch_lb_{line}_{ct}"
            )
            
            # Power flow limits for ct_PAC variables
            model.addConstr(
                ac_vars['ct_PAC_to'][line, ct] <= S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct] + M * (1 - ac_vars['ct_branch'][line, ct]),
                name=f"ct_S_to_AC_limit_upper_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['ct_PAC_to'][line, ct] >= -S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct] - M * (1 - ac_vars['ct_branch'][line, ct]),
                name=f"ct_S_to_AC_limit_lower_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['ct_PAC_from'][line, ct] <= S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct] + M * (1 - ac_vars['ct_branch'][line, ct]),
                name=f"ct_S_from_AC_limit_upper_{line}_{ct}"
            )
            
            model.addConstr(
                ac_vars['ct_PAC_from'][line, ct] >= -S_lineACct_lim[line, ct] * ac_vars['ct_branch'][line, ct] - M * (1 - ac_vars['ct_branch'][line, ct]),
                name=f"ct_S_from_AC_limit_lower_{line}_{ct}"
            )

def set_objective(model, grid, gen_vars, ac_vars, OPEX=True, NPV=True, n_years=25, Hy=8760, discount_rate=0.02):
    """Set objective function for Gurobi model"""
    cab_types_set = list(range(0,len(grid.Cable_options[0]._cable_types)))
    # Investment costs
    investment_cost = 0
    
    for line in grid.lines_AC_ct:
        l = line.lineNumber
        if line.array_opf:
            if NPV:
                for ct in cab_types_set:
                    investment_cost += ac_vars['ct_branch'][l, ct] * line.base_cost[ct]
            else:
                for ct in cab_types_set:
                    investment_cost += ac_vars['ct_branch'][l, ct] * line.base_cost[ct] / line.life_time_hours
    
    # Operational costs
    operational_cost = 0
    if OPEX:
        lista_gen = list(range(0, grid.n_gen))
        for g in lista_gen:
            gen = grid.Generators[g]
            operational_cost += gen.lf * gen_vars['PGi_gen'][g]
    
    if NPV:
        present_value = Hy * (1 - (1 + discount_rate) ** -n_years) / discount_rate
        operational_cost *= present_value
    
    # Total objective
    total_cost = investment_cost + operational_cost
    model.setObjective(total_cost, GRB.MINIMIZE)


def ExportACDC_Lmodel_toPyflowACDC_gurobi(model, grid,gen_vars,ac_vars):
    """Export Gurobi results back to grid object"""
    
    if model.status != GRB.OPTIMAL:
        # Print model information to help debug infeasibility
        print_gurobi_model(model, gen_vars, ac_vars, detailed=True)
        debug_infeasibility(model, gen_vars, ac_vars)
        raise RuntimeError(f"Cannot export results: model status is {model.status}")
    cab_types_set = list(range(0,len(grid.Cable_options[0]._cable_types)))
    grid.OPF_run = True

    # Generation 
    for g in grid.Generators:
        g.PGen = gen_vars['PGi_gen'][g.genNumber].X
        g.QGen = 0.0 
    
    # Renewable sources
    for rs in grid.RenSources:
        rs.gamma = gen_vars['gamma'][rs.rsNumber].X
        rs.QGi_ren = 0.0 

    # AC bus
    grid.V_AC = np.ones(grid.nn_AC)
    grid.Theta_V_AC = np.zeros(grid.nn_AC)

    for node in grid.nodes_AC:
        nAC = node.nodeNumber
        node.V = 1.0  
        node.theta = ac_vars['thetha_AC'][nAC].X
        
        node.PGi_opt = ac_vars['PGi_opt'][nAC].X
        node.QGi_opt = 0.0 
        node.PGi_ren = ac_vars['PGi_ren'][nAC].X
        node.QGi_ren = 0.0  
        
        grid.Theta_V_AC[nAC] = node.theta

    # Power injections
    B = np.imag(grid.Ybus_AC)
    Theta = grid.Theta_V_AC
    Theta_diff = Theta[:, None] - Theta
    Pf_DC = (-B * Theta_diff).sum(axis=1)
    
    for node in grid.nodes_AC:
        i = node.nodeNumber
        node.P_INJ = Pf_DC[i]
        node.Q_INJ = 0.0
    
    for line in grid.lines_AC_ct:
        ct_selected = [ac_vars['ct_branch'][line.lineNumber, ct].X >= 0.9 for ct in cab_types_set]
        if any(ct_selected):
            line.active_config = np.where(ct_selected)[0][0]
            ct = list(cab_types_set)[line.active_config]
            line.fromS = ac_vars['ct_PAC_from'][line.lineNumber, ct].X + 1j*0
            line.toS = ac_vars['ct_PAC_to'][line.lineNumber, ct].X + 1j*0
        else:
            line.active_config = -1
            line.fromS = 0 + 1j*0
            line.toS = 0 + 1j*0
        line.loss = 0
        line.P_loss = 0

    # Standard AC lines
    Theta = grid.Theta_V_AC
    for line in grid.lines_AC:
        i = line.fromNode.nodeNumber
        j = line.toNode.nodeNumber
        
        B = -np.imag(line.Ybus_branch[0, 1])
        P_ij = B * (Theta[i] - Theta[j])
        P_ji = B * (Theta[j] - Theta[i])

        line.fromP = P_ij
        line.toP = P_ji
        line.toS = P_ji + 1j*0
        line.fromS = P_ij + 1j*0
        line.P_loss = 0
        line.loss = 0
        line.i_from = abs(P_ij)
        line.i_to = abs(P_ji)
    