import time
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math
import pyomo.environ as pyo
try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

from .ACDC_OPF_NL_model import OPF_create_NLModel_ACDC,TEP_variables
from .AC_OPF_L_model import OPF_create_LModel_ACDC,ExportACDC_Lmodel_toPyflowACDC
from .ACDC_OPF import OPF_solve,OPF_obj,OPF_obj_L,obj_w_rule,ExportACDC_NLmodel_toPyflowACDC,calculate_objective,reset_to_initialize
from .ACDC_Static_TEP import transmission_expansion, linear_transmission_expansion

from .Graph_and_plot import save_network_svg


__all__ = [
    'sequential_CSS',
    'min_sub_connections',
    'MIP_path_graph',
    'simple_CSS'
]


def sequential_CSS(grid,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,max_turbines_per_string=None,limit_crossings=True,sub_min_connections=True,MIP_solver='glpk',CSS_L_solver='glpk',CSS_NL_solver='bonmin',svg=None,max_iter=None,time_limit=300,NL=False,tee=False,fs=False):
    
    

    staring_cables = grid.Cable_options[0].cable_types
    new_cables = staring_cables.copy()
  
    results = []
    tot_timing_info = {}
    i = 0
    path_time = 0
    css_time = 0
    weights_def, PZ = obj_w_rule(grid,ObjRule,True)
    t0 = time.perf_counter()
    t_MW = grid.RenSources[0].PGi_ren_base*grid.S_base
    #print(f'DEBUG: t_MW {t_MW}')
    #print(f'DEBUG: starting max flow {max_flow}')

    if max_iter is None:
        max_iter = len(grid.Cable_options[0].cable_types)
    og_cable_types = grid.Cable_options[0].cable_types.copy()
    
    MIP_time = grid.MIP_time
    if max_turbines_per_string is not None:
        max_flow = max_turbines_per_string
    else:
        max_flow = grid.max_turbines_per_string

    flag = True

    



    while flag:
        timing_info = {}
        
        t1 = time.perf_counter()
        if sub_min_connections:
            flag, high_flow,model_MIP,feasible_solutions_MIP ,ns, sub_iter , path_time = min_sub_connections(grid, max_flow)
        else:
            flag, high_flow,model_MIP,feasible_solutions_MIP = MIP_path_graph(grid, max_flow, solver_name=MIP_solver, crossings=limit_crossings, tee=tee,callback=fs)
            
        t2 = time.perf_counter()
        timing_info['Paths'] = t2 - t1
        path_time += t2 - t1

        if not flag:
            if i == 0:
                # If MIP fails on first iteration, return None
                return None, None, None, None,i
            else:
                # If MIP fails on later iterations, break the loop
                break
        MIP_obj_value = pyo.value(model_MIP.objective)
        if  high_flow < max_flow:
            
            max_power_per_string = t_MW*high_flow 
            first_index_to_comply = next((i for i, rating in enumerate(grid.Cable_options[0].MVA_ratings) if rating >= max_power_per_string), len(grid.Cable_options[0].MVA_ratings) - 1)
            for line in grid.lines_AC_ct:
                if line.active_config > 0:
                    line.active_config = first_index_to_comply

            grid.Cable_options[0].cable_types = grid.Cable_options[0]._cable_types[:first_index_to_comply + 1]
           
            grid.max_turbines_per_string = high_flow
        iter_cab_available= grid.Cable_options[0].cable_types.copy()
        t3 = time.perf_counter()
        #print(f'DEBUG: Iteration {i}')
        model, model_results, timing_info_CSS, solver_stats = simple_CSS(grid,NPV,n_years,Hy,discount_rate,ObjRule,CSS_L_solver,CSS_NL_solver,time_limit,NL,tee,fs=fs)
        feasible_solutions_CSS = solver_stats['feasible_solutions']
        t4 = time.perf_counter()
        timing_info['CSS'] = t4 - t3
        css_time += t4 - t3
        if svg is not None:
            from .Graph_and_plot import save_network_svg
            lines_AC_CT = {k: {ct: np.float64(pyo.value(model.ct_branch[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
            lines_AC_CT_fromP = {k: {ct: np.float64(pyo.value(model.ct_PAC_from[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
            lines_AC_CT_toP = {k: {ct: np.float64(pyo.value(model.ct_PAC_to[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
            gen_active_config = {k: np.float64(pyo.value(model.ct_types[k])) for k in model.ct_set}
           
            
            grid.Cable_options[0].active_config = gen_active_config


            def process_line_AC_CT(line):
                l = line.lineNumber
                ct_selected = [lines_AC_CT[l][ct] >= 0.90  for ct in model.ct_set]
                if any(ct_selected):
                    line.active_config = np.where(ct_selected)[0][0]
                    ct = list(model.ct_set)[line.active_config]
                    Pfrom = lines_AC_CT_fromP[l][ct]
                    Pto   = lines_AC_CT_toP[l][ct]
                    Qfrom = 0.0
                    Qto   = 0.0
                else:
                    line.active_config = -1
                    Pfrom = 0
                    Pto   = 0
                    Qfrom = 0
                    Qto   = 0
                
                line.fromS = (Pfrom + 1j*Qfrom)
                line.toS = (Pto + 1j*Qto)
                line.loss = 0
                line.P_loss = 0

            with ThreadPoolExecutor() as executor:
                executor.map(process_line_AC_CT, grid.lines_AC_ct)

            
            if not os.path.exists('intermediate_networks'):
                os.makedirs('intermediate_networks')
            save_network_svg(grid, name=f'intermediate_networks/{svg}_{i}_{CSS_L_solver}', width=1000, height=1000, journal=True,square_ratio=True, legend=True)
        
        if model_results['Solver'][0]['Status'] == 'ok':
            obj_value = pyo.value(model.obj)
        else:
            obj_value = None  # or some default value

        
        
        #print('DEBUG: Iteration',i)

        used_cable_types = []
        used_cable_names = []
        
        # Analyze which cable types were used in the optimization
        if model_results['Solver'][0]['Status'] == 'ok':
            # Get the cable types that were actually used
           
            for ct in model.ct_set:
                if pyo.value(model.ct_types[ct]) > 0.5:  # Binary variable > 0.5 means it was selected
                    used_cable_types.append(ct)
                    used_cable_names.append(grid.Cable_options[0].cable_types[ct])
            #print(f'DEBUG: Used cable types: {used_cable_types}')
            
            if used_cable_types:
                # Find the largest cable type that was used
                largest_used_index = max(used_cable_types)
              
                new_cables = new_cables[:largest_used_index]
                
            else:
                # No cable types were used, remove the largest one
                new_cables.pop()
               
        else:
            # Optimization failed, remove the largest cable type
            new_cables.pop()
           
        
       
        

        
        t5 = time.perf_counter()
        timing_info['processing'] = (t5 - t1)-(timing_info['Paths']+timing_info['CSS'])
        # Create a dictionary for this iteration's results
        iteration_result = {
            'cable_length': MIP_obj_value,
            'model_obj': obj_value,  # Save the objective value
            'cable_options': iter_cab_available,  # Save a copy of the cable list
            'cables_used': used_cable_names,
            'model_results': model_results,
            'solver_stats': solver_stats,
            'timing_info': timing_info,
            'MIP_model': model_MIP,
            'CSS_model': model,
            'i': i,
            'feasible_solutions_MIP': feasible_solutions_MIP,
            'feasible_solutions_CSS': feasible_solutions_CSS
        }
        results.append(iteration_result)  # Add to the results list   
        
        if i > 0 and obj_value is not None and results[i-1]['model_obj'] is not None:
            if obj_value > results[i-1]['model_obj']:
                break
        i += 1
        if i > max_iter:
            break
        # Update grid with new cable set
        if len(new_cables) > 0:
            grid.Cable_options[0].cable_types = new_cables
            
            # Recalculate max_flow based on current cable set
            max_cable_capacity = max(grid.Cable_options[0].MVA_ratings)
            max_flow = int(max_cable_capacity / t_MW)
        else:
            #print("DEBUG: No more cable types available")
            break
       
    
    # After the while loop ends, create summary from all iterations
    summary_results = {
        'cable_length': [result['cable_length'] for result in results],
        'model_obj':    [result['model_obj'] for result in results],
        'cable_options': [result['cable_options'] for result in results],
        'cables_used':  [result['cables_used'] for result in results],
        'timing_info':  [result['timing_info'] for result in results],
        'solver_status':[result['model_results']['Solver'][0]['Status']  for result in results],
        'iteration':    [result['i'] for result in results],
        'feasible_solutions_MIP': [result['feasible_solutions_MIP'] for result in results],
        'feasible_solutions_CSS': [result['feasible_solutions_CSS'] for result in results]
    }

    # Find best result
    if len(results) > 1:
        best_result = min(results, key=lambda x: x['model_obj'])
    else:
        best_result = results[0]  # Just return the single result
    

    if fs:
        feasible_solutions_MIP = [result['feasible_solutions_MIP'] for result in results]
        feasible_solutions_CSS = [result['feasible_solutions_CSS'] for result in results]
        _plot_feasible_solutions_subplots(
            feasible_solutions_MIP,
            feasible_solutions_CSS,
            show=False,
            save_path=f'feasible_solutions_{grid.name}.png'
        )
        # Export to Excel with MIP and CSS sheets
        export_feasible_solutions_to_excel(
            feasible_solutions_MIP,
            feasible_solutions_CSS,
            save_path=f'feasible_solutions_{grid.name}.xlsx'
        )

    
    

    model = best_result['CSS_model']
    model_MIP = best_result['MIP_model']
    model_results = best_result['model_results']
    tot_timing_info['Paths'] = path_time
    tot_timing_info['CSS'] = css_time
    solver_stats = best_result['solver_stats']
    best_i = best_result['i']

    t5 = time.perf_counter()
    if NL:
        ExportACDC_NLmodel_toPyflowACDC(model, grid, PZ, TEP=True)
    else:
        ExportACDC_Lmodel_toPyflowACDC(model, grid, solver_results=model_results, tee=tee)
    
    

    grid.Cable_options[0].cable_types = og_cable_types
    gen_active_config = grid.Cable_options[0].active_config
    grid.Cable_options[0].active_config = [int(gen_active_config.get(k, 0)) for k in range(len(og_cable_types))]
    
    lines_AC_CT = {k: {ct: np.float64(pyo.value(model.ct_branch[k, ct])) for ct in model.ct_set} for k in model.lines_AC_ct}
    
    def process_line_AC_CT(line):
        l = line.lineNumber
        ct_selected = [lines_AC_CT[l][ct] >= 0.90  for ct in model.ct_set]
        if any(ct_selected):
            line.active_config = np.where(ct_selected)[0][0] 
        else:
            line.active_config = -1


    with ThreadPoolExecutor() as executor:
        executor.map(process_line_AC_CT, grid.lines_AC_ct)


    present_value = Hy*(1 - (1 + discount_rate) ** -n_years) / discount_rate
    for obj in weights_def:
        weights_def[obj]['v']=calculate_objective(grid,obj,True)
        weights_def[obj]['NPV']=weights_def[obj]['v']*present_value

    grid.TEP_run=True
    grid.OPF_obj = weights_def

    t_modelexport = time.perf_counter() - t5
    tot_timing_info['export'] = t_modelexport
    tot_timing_info['sequential'] = t5 - t0

    models = (model_MIP,model)
    return models, summary_results , tot_timing_info, solver_stats,best_i
    


def min_sub_connections(grid, max_flow=None, MIP_solver='glpk', crossings=True, tee=False, callback=False):
    tn = grid.n_ren
    sn = grid.nn_AC - grid.n_ren

    ns = math.ceil(tn/(sn* max_flow))

    flag=False
    i =0
    max_iters = 1 if ns is None else 10

    while not flag and i<max_iters:
        
        for node in grid.nodes_AC:
            if node.type == 'Slack':
                node.ct_limit = ns

        t0 = time.perf_counter()
        flag, high_flow,model_MIP,feasible_solutions_MIP = MIP_path_graph(grid, max_flow, solver_name=MIP_solver, crossings=crossings, tee=tee,callback=callback)
        t1 = time.perf_counter()
        path_time = t1 - t0
        i+=1
        if not flag:
            if ns is not None:
                ns+=1
    return flag, high_flow,model_MIP,feasible_solutions_MIP ,ns, i , path_time




def MIP_path_graph(grid, max_flow=None, solver_name='glpk', crossings=False, tee=False, callback=False):
    """Solve the master MIP problem and track feasible solutions over time (optional with Gurobi callback)."""
    model = _create_master_problem_pyomo(grid, crossings, max_flow)
    feasible_solutions = []
    feasible_solution_found = False
    high_flow = None

    # === Gurobi + Callback path ===
    if callback and solver_name == 'gurobi' and GUROBI_AVAILABLE:
        from gurobipy import GRB
        solver = pyo.SolverFactory('gurobi_persistent')
        solver.set_instance(model)
        grb_model = solver._solver_model
        
        def my_callback(model, where):
            if where == GRB.Callback.MIPSOL:
                # New feasible solution found
                time_found = model.cbGet(GRB.Callback.RUNTIME)
                obj        = model.cbGet(GRB.Callback.MIPSOL_OBJ)      # incumbent obj (this solution)
                
                # Global best bound at this moment
                bound      = model.cbGet(GRB.Callback.MIPSOL_OBJBND)   # <-- IMPORTANT

                gap = None
                # Check that we actually have a meaningful incumbent and bound
                if obj < GRB.INFINITY and bound > -GRB.INFINITY:
                    denom = abs(obj)
                    if denom < 1e-10:
                        denom = 1e-10  # avoid division by zero for tiny objectives

                    gap = abs(bound - obj) / denom   # same definition Gurobi uses

                # Store: (time, value, gap)
                feasible_solutions.append((time_found, obj, gap))

        if getattr(grid, "MIP_time", None) is not None:
            grb_model.setParam("TimeLimit", grid.MIP_time)
        
        # Settings for spanning tree / network flow MIP
        mip_focus = getattr(grid, "MIP_focus", 1)  # Focus on finding good solutions fast
        grb_model.setParam("MIPFocus", mip_focus)
        grb_model.setParam("Cuts", 2)  # Aggressive cuts
        grb_model.setParam("Heuristics", 0.2)  # More time on heuristics
        grb_model.setParam("Presolve", 2)  # Aggressive presolve
        grb_model.setParam("Threads", 0)  # Use all available cores
        grb_model.setParam("Symmetry", 2)  # Aggressive symmetry detection
        
        grb_model.optimize(my_callback)

        from pyomo.opt.results.results_ import SolverResults
        results = SolverResults()
        results.solver.status = pyo.SolverStatus.ok
        results.problem.upper_bound = grb_model.ObjVal if grb_model.SolCount > 0 else None
        results.solver.time = grb_model.Runtime

        if grb_model.Status == GRB.Status.OPTIMAL:
            results.solver.termination_condition = pyo.TerminationCondition.optimal
            feasible_solution_found = True
        elif grb_model.Status == GRB.Status.SUBOPTIMAL:
            results.solver.termination_condition = pyo.TerminationCondition.feasible
            feasible_solution_found = True
        elif grb_model.Status == GRB.Status.TIME_LIMIT:
            results.solver.termination_condition = pyo.TerminationCondition.maxTimeLimit
            feasible_solution_found = grb_model.SolCount > 0
        elif grb_model.Status == GRB.Status.INFEASIBLE:
            results.solver.termination_condition = pyo.TerminationCondition.infeasible
        else:
            results.solver.termination_condition = pyo.TerminationCondition.unknown
            feasible_solution_found = grb_model.SolCount > 0

        if feasible_solution_found:
            #sync_gurobi_solution_to_pyomo(model, solver)
            
            solver.load_vars()
            # Calculate final gap
            final_gap = None
            if grb_model.SolCount > 0:
                obj_val = grb_model.ObjVal
                obj_bound = grb_model.ObjBound
                if (obj_bound != GRB.INFINITY and obj_bound != -GRB.INFINITY and 
                    abs(obj_val) > 1e-10):
                    model_sense = grb_model.ModelSense
                    if model_sense == GRB.MINIMIZE:
                        final_gap = (obj_val - obj_bound) / abs(obj_val)
                    else:  # MAXIMIZE
                        final_gap = (obj_bound - obj_val) / abs(obj_val)
            feasible_solutions.append((grb_model.Runtime, grb_model.ObjVal, final_gap))
            
        grb_model.dispose()

    # === Other solvers (no callback) ===
    else:
        solver = pyo.SolverFactory(solver_name)
        if getattr(grid, "MIP_time", None) is not None:
            if solver_name == 'gurobi':
                solver.options['TimeLimit'] = grid.MIP_time
                # Settings for spanning tree / network flow MIP
                mip_focus = getattr(grid, "MIP_focus", 1)  # Focus on finding good solutions fast
                solver.options['MIPFocus'] = mip_focus
                solver.options['Cuts'] = 2  # Aggressive cuts
                solver.options['Heuristics'] = 0.2  # More time on heuristics
                solver.options['Presolve'] = 2  # Aggressive presolve
                solver.options['Threads'] = 0  # Use all available cores
                solver.options['Symmetry'] = 2  # Aggressive symmetry detection
            elif solver_name == 'glpk':
                solver.options['tmlim'] = grid.MIP_time

        try:
            solver.solve(model, tee=tee)
            _ = pyo.value(model.objective)
            feasible_solution_found = True
            
        except (ValueError, AttributeError):
            feasible_solution_found = False

    # === Post-solve handling ===
    if feasible_solution_found:
        flows = [abs(pyo.value(model.line_flow[line])) for line in model.lines]
        high_flow = max(flows) if flows else 0
        last_cable_type_index = len(grid.Cable_options[0]._cable_types) - 1
        for line in model.lines:
            ct_line = grid.lines_AC_ct[line]
            ct_line.active_config = last_cable_type_index if pyo.value(model.line_used[line]) > 0.5 else -1

        return True, high_flow, model, feasible_solutions

    else:
        print("✗ MIP model failed")
        return False, None, model, feasible_solutions


    
def _create_master_problem_pyomo(grid,crossings=True, max_flow=None):
        """Create master problem using Pyomo"""
        
        if max_flow is None:
            max_flow = len(grid.nodes_AC) - 1
        
        # Create model
        model = pyo.ConcreteModel()
        
        # Sets
        model.lines = pyo.Set(initialize=range(len(grid.lines_AC_ct)))
        model.nodes = pyo.Set(initialize=range(len(grid.nodes_AC)))
        
        # Find sink nodes (nodes with generators) and source nodes (nodes with renewable resources)
        sink_nodes = []
        source_nodes = []
        
        for node in model.nodes:
            nAC = grid.nodes_AC[node]
            if nAC.connected_gen:  # Node has generator (sink)
                sink_nodes.append(node)
            if nAC.connected_RenSource:  # Node has renewable resources (source)
                source_nodes.append(node)
        
        if not sink_nodes:
            raise ValueError("No generator nodes found!")
        
        model.source_nodes = pyo.Set(initialize=source_nodes)
        model.sink_nodes = pyo.Set(initialize=sink_nodes)
        

        
        # Variables
        # Binary variables: one per line (used or not)
        model.line_used = pyo.Var(model.lines, domain=pyo.Binary)
        
        def line_flow_bounds(model, line):
            line_obj = grid.lines_AC_ct[line]
            from_node = line_obj.fromNode
            to_node = line_obj.toNode
            
            # Check if either endpoint is a slack node (substation)
            from_is_slack = from_node.type == 'Slack'
            to_is_slack = to_node.type == 'Slack'
            
            # If line connects to slack node: use full capacity [-max_flow, max_flow]
            if from_is_slack or to_is_slack:
                return (-max_flow, max_flow)
            else:
                # If line connects PQ-PQ (turbine-turbine): use reduced capacity
                return (-(max_flow - 1), max_flow - 1)
        # Flow variables (integer - can carry flow in either direction)
        model.line_flow = pyo.Var(model.lines, domain=pyo.Integers, bounds=line_flow_bounds)
        model.node_flow = pyo.Var(model.nodes, domain=pyo.Integers)
        model.line_flow_dir = pyo.Var(model.lines, domain=pyo.Binary)
        # Objective: minimize total cable length
        def objective_rule(model):
            return sum(model.line_used[line] * grid.lines_AC_ct[line].Length_km 
                      for line in model.lines)
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Spanning tree constraint: exactly numNodes-1 connections
        def spanning_tree_rule(model):
            return sum(model.line_used[line] for line in model.lines) == len(model.nodes) - len(model.sink_nodes)
     
        
        model.spanning_tree = pyo.Constraint(rule=spanning_tree_rule)
        
        # Constrain connections 
        def connections_rule(model, node):
            if  grid.nodes_AC[node].ct_limit is None:
                    return pyo.Constraint.Skip
            else:
                node_connections = sum(model.line_used[line] 
                                     for line in model.lines
                                     if (grid.lines_AC_ct[line].fromNode.nodeNumber == node or 
                                         grid.lines_AC_ct[line].toNode.nodeNumber == node))
                return node_connections <= grid.nodes_AC[node].ct_limit
                
        nT= len(model.nodes) - len(model.sink_nodes)
        nS = len(model.sink_nodes)

        def connections_rule_lower(model, node):
            node_connections = sum(model.line_used[line] 
                            for line in model.lines
                            if (grid.lines_AC_ct[line].fromNode.nodeNumber == node or 
                                grid.lines_AC_ct[line].toNode.nodeNumber == node))
        
            # If node is a sink (substation), calculate minimum connections needed
            if node in model.sink_nodes:
                # Calculate minimum connections per sink based on capacity
                # Formula: ceil((non_sink_nodes) / (total_sink_capacity))
                
                min_connections = math.ceil(nT/(nS*max_flow))
                return node_connections >= min_connections
            else:
                # For non-sink nodes, minimum is 1
                return node_connections >= 1

        model.connections_rule = pyo.Constraint(model.nodes, rule=connections_rule)
        model.connect_lower = pyo.Constraint(model.nodes, rule= connections_rule_lower)




        # Flow conservation for all nodes
        def flow_conservation_rule(model, node):
            node_flow = 0
            
            for line in model.lines:
                line_obj = grid.lines_AC_ct[line]
                from_node = line_obj.fromNode.nodeNumber
                to_node = line_obj.toNode.nodeNumber
            
                if from_node == node:
                    # This node is fromNode, so positive flow leaves this node
                    node_flow += model.line_flow[line]
    
                elif to_node == node:
                    # This node is toNode, so negative flow leaves this node
                    node_flow -= model.line_flow[line]
                    
            return model.node_flow[node] == node_flow
        
        model.flow_conservation = pyo.Constraint(model.nodes, rule=flow_conservation_rule)
        
        def source_node_rule(model, node):
            return model.node_flow[node] == 1
            
        model.source_node = pyo.Constraint(model.source_nodes, rule=source_node_rule)

     
        def sink_absorption_rule(model):
            return sum(model.node_flow[n] for n in model.sink_nodes) == -len(model.source_nodes)
        model.total_sink_absorption = pyo.Constraint(rule=sink_absorption_rule)
        
        # Intermediate nodes: net flow = 0 (conservation)
        def intermediate_node_rule(model, node):
            if node not in model.source_nodes and node not in model.sink_nodes:
                return model.node_flow[node] == 0
            else:
                return pyo.Constraint.Skip
        
        model.intermediate_node = pyo.Constraint(model.nodes, rule=intermediate_node_rule)
        
        # Link flow to investment: can only use lines we invest in
        def flow_investment_rule(model, line):
            return model.line_flow[line] <= max_flow * model.line_used[line]

        def flow_investment_rule_2(model, line):
            return model.line_flow[line] >= -max_flow * model.line_used[line]

        # NEW: If line is used, flow must be >= 1 (when positive) OR <= -1 (when negative)
        def flow_nonzero_positive(model, line):
            # If line_used=1 and line_flow_dir=1, then line_flow >= 1
            # If line_used=0 or line_flow_dir=0, this constraint is relaxed
            M = max_flow + 1
            return model.line_flow[line] >= 1 - M * (1 - model.line_used[line]) - M * (1 - model.line_flow_dir[line])

        def flow_nonzero_negative(model, line):
            # If line_used=1 and line_flow_dir=0, then line_flow <= -1
            # If line_used=0 or line_flow_dir=1, this constraint is relaxed
            M = max_flow + 1
            return model.line_flow[line] <= -1 + M * (1 - model.line_used[line]) + M * model.line_flow_dir[line]

        # Ensure direction variable is only active when line is used
        def flow_dir_active(model, line):
            return model.line_flow_dir[line] <= model.line_used[line]

        model.flow_investment_link = pyo.Constraint(model.lines, rule=flow_investment_rule)
        model.flow_investment_link_2 = pyo.Constraint(model.lines, rule=flow_investment_rule_2)
        model.flow_nonzero_pos = pyo.Constraint(model.lines, rule=flow_nonzero_positive)
        model.flow_nonzero_neg = pyo.Constraint(model.lines, rule=flow_nonzero_negative)
        model.flow_dir_active = pyo.Constraint(model.lines, rule=flow_dir_active)



        # Add crossing constraints if crossings=True
        if crossings and hasattr(grid, 'crossing_groups') and grid.crossing_groups:
            # Create a set for crossing groups
            model.crossing_groups = pyo.Set(initialize=range(len(grid.crossing_groups)))
            
            # Constraint: for each crossing group, only one line can be active
            def crossing_constraint_rule(model, group_idx):
                group = grid.crossing_groups[group_idx]
                # Sum of all line_used variables in this crossing group must be <= 1
                return sum(model.line_used[line] for line in model.lines 
                          if grid.lines_AC_ct[line].lineNumber in group) <= 1
            
            model.crossing_constraints = pyo.Constraint(model.crossing_groups, rule=crossing_constraint_rule)
            
        return model
    

def _plot_feasible_solutions(results,type='solution', plot_type='MIP', suptitle=None, show=True, save_path=None, width_mm=None):
    import matplotlib.pyplot as plt
    # local import to ensure availability regardless of module-level imports
    import os
    FS = 10
    
    # Determine which column to extract based on type parameter
    # Tuples are always (time, solution, gap)
    if type == 'gap':
        col_idx = 2  # Third column (gap)
    else:  # type == 'solution'
        col_idx = 1  # Second column (solution)
    
    # Normalize input: accept a single feasible_solutions list, a list of those,
    # a dict with key 'feasible_solutions_MIP', or a list of such dicts
    def _is_pair_list(seq):
        try:
            return isinstance(seq, (list, tuple)) and len(seq) > 0 and isinstance(seq[0], (list, tuple)) and len(seq[0]) >= 2
        except Exception:
            return False
    
    if results is None:
        normalized_results = []
    elif isinstance(results, dict) and 'feasible_solutions_MIP' in results:
        normalized_results = [results.get('feasible_solutions_MIP', [])]
    elif isinstance(results, list):
        if len(results) > 0 and isinstance(results[0], dict) and 'feasible_solutions_MIP' in results[0]:
            normalized_results = [r.get('feasible_solutions_MIP', []) for r in results]
        elif _is_pair_list(results):
            # single run provided as list of (time, obj)
            normalized_results = [results]
        else:
            # assume already in the expected list-of-runs format
            normalized_results = results
    else:
        # Fallback: treat as empty
        normalized_results = []
    
    if width_mm is not None:
        fig_w_in = width_mm / 25.4
        fig_h_in = fig_w_in 
    else:
        fig_w_in = 6.0
        fig_h_in = fig_w_in 

    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in), sharex=False, sharey=False, constrained_layout=True)

    # Normalize plot_type and set axis label
    ptype = (plot_type or 'MIP').upper()
    if ptype == 'CSS':
        y_axis_label = 'Objective [M€]'
    else:
        ptype = 'MIP'
        y_axis_label = 'Cable length [km]'
    
    # Update y-axis label based on type
    if type == 'gap':
        y_axis_label = 'Gap'

    # plotting logic mirroring the subplots helper
    if not normalized_results:
        ax.set_title(ptype, fontsize=FS)
        ax.set_xlabel('Time (s)', fontsize=FS)
        ax.set_ylabel(y_axis_label, fontsize=FS)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=FS)
    else:
        has_any = False
        for i, feas in enumerate(normalized_results):
            if not feas:
                continue
            has_any = True
            feas_sorted = sorted(feas, key=lambda x: x[0])
            times = []
            values = []
            for f in feas_sorted:
                # Tuples are always (time, solution, gap)
                t = f[0]
                v = f[col_idx]
                # For gap, skip None values
                if col_idx == 2 and v is None:
                    continue
                times.append(t)
                values.append(v)
            
            if not times:  # Skip if no valid data points
                continue
            
            if ptype == 'CSS' and col_idx == 1:
                values = [v / 1e6 for v in values]
            ax.plot(times, values, 'o-', label=f'i={i} (s={len(values)})', markersize=5, linewidth=2)
        ax.set_title(ptype, fontsize=FS*1.2)
        ax.set_xlabel('Time (s)', fontsize=FS*1.1)
        ax.set_ylabel(y_axis_label, fontsize=FS*1.1)
        if has_any:
            ax.legend(prop={'size': FS}, loc='upper right', frameon=False)
        ax.tick_params(labelsize=FS)
        ax.grid(True, alpha=0.3)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=FS*1.3)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.22)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.18)

    if save_path is not None:
        dir_, base = os.path.split(save_path)
        root, ext = os.path.splitext(base)
        base = (root + ext.lower()).lower()
        save_path = os.path.join(dir_, base)
        if save_path.endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, format='png', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)

def _plot_feasible_solutions_subplots(results_mip, results_css, suptitle=None, show=True, save_path=None, width_mm=None):
    import matplotlib.pyplot as plt
    FS = 10
    # Maintain 40:20 aspect ratio regardless of absolute size (taller axes)
    ratio = 20.0 / 40.0
    if width_mm is not None:
        fig_w_in = width_mm / 25.4
        fig_h_in = fig_w_in * ratio
    else:
        fig_w_in = 6.0
        fig_h_in = fig_w_in * ratio
    figsize = (fig_w_in, fig_h_in)
    # Two subplots side-by-side: MIP (left), CSS (right)
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharex=False, sharey=False, constrained_layout=True)

    def _plot(ax, results, title, yaxis, plot_gap=False):
        if not results:
            ax.set_title(title, fontsize=FS)
            ax.set_xlabel('Time (s)', fontsize=FS)
            ax.set_ylabel(yaxis, fontsize=FS)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=FS)
            return
        has_any = False
        for i, feas in enumerate(results):
            if not feas:
                continue
            has_any = True
            feas_sorted = sorted(feas, key=lambda x: x[0])
            times = [t for t, o, g in feas_sorted]
            if plot_gap:
                # Plot gap (tuple[2]) as percentage
                values = [g * 100 if g is not None else None for t, o, g in feas_sorted]
                # Filter out None values
                valid_data = [(t, v) for t, v in zip(times, values) if v is not None]
                if valid_data:
                    times = [t for t, v in valid_data]
                    values = [v for t, v in valid_data]
            else:
                values = [o for t, o, g in feas_sorted]
                if title == 'CSS':
                    values = [o/1e6 for o in values]
            if times and values:
                ax.plot(times, values, 'o-', label=f'i={i} (s={len(values)})', markersize=5, linewidth=2)
        ax.set_title(title, fontsize=FS*1.2)
        ax.set_xlabel('Time (s)', fontsize=FS*1.1)
        ax.set_ylabel(yaxis, fontsize=FS*1.1)
        if has_any:
            ax.legend(prop={'size': FS}, loc='upper right', frameon=False)
        ax.tick_params(labelsize=FS)
        ax.grid(True, alpha=0.3)

    # Plot time vs gap for MIP, time vs objective for CSS
    _plot(axes[0], results_mip, 'MIP', 'Gap [%]', plot_gap=True)
    _plot(axes[1], results_css, 'CSS', 'Objective [M€]', plot_gap=True)

    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=FS*1.3)
        fig.subplots_adjust(left=0.10, right=0.99, top=0.80, bottom=0.22, wspace=0.22)
    else:
        fig.subplots_adjust(left=0.08, right=0.99, top=0.98, bottom=0.18, wspace=0.18)

    if save_path is not None:
        dir_, base = os.path.split(save_path)
        root, ext = os.path.splitext(base)
        base = (root + ext.lower()).lower()  # lowercase name + extension
        save_path = os.path.join(dir_, base)
        
        if save_path.endswith('.svg'):
            fig.savefig(save_path, format='svg', bbox_inches='tight')
        else:
            fig.savefig(save_path, format='png', bbox_inches='tight')

    if show:
        plt.show()
    else:
        plt.close(fig)


def export_feasible_solutions_to_excel(results_mip, results_css, save_path):
    """
    Export MIP and CSS feasible solutions to Excel with two sheets.
    
    Each iteration's solutions are in columns: {i}_t, {i}_obj, {i}_gap
    Rows correspond to solution index within that iteration.
    
    Args:
        results_mip: List of lists, each inner list contains (time, obj, gap) tuples for MIP
        results_css: List of lists, each inner list contains (time, obj, gap) tuples for CSS
        save_path: Path to save the Excel file
    """
    import pandas as pd
    
    def _create_dataframe(results, name):
        """Create a DataFrame from results with columns for each iteration."""
        if not results:
            return pd.DataFrame()
        
        # Find max number of solutions across all iterations
        max_solutions = max(len(feas) if feas else 0 for feas in results)
        if max_solutions == 0:
            return pd.DataFrame()
        
        data = {}
        for i, feas in enumerate(results):
            if not feas:
                continue
            # Sort by time
            feas_sorted = sorted(feas, key=lambda x: x[0])
            
            times = [t for t, o, g in feas_sorted]
            objs = [o for t, o, g in feas_sorted]
            gaps = [g * 100 if g is not None else None for t, o, g in feas_sorted]
            
            # Pad to max_solutions length
            times.extend([None] * (max_solutions - len(times)))
            objs.extend([None] * (max_solutions - len(objs)))
            gaps.extend([None] * (max_solutions - len(gaps)))
            
            data[f'{i+1}_t'] = times
            data[f'{i+1}_obj'] = objs
            data[f'{i+1}_gap'] = gaps
        
        return pd.DataFrame(data)
    
    mip_df = _create_dataframe(results_mip, 'MIP')
    css_df = _create_dataframe(results_css, 'CSS')
    
    with pd.ExcelWriter(save_path, engine='xlsxwriter') as writer:
        if not mip_df.empty:
            mip_df.to_excel(writer, sheet_name='MIP', index=False)
        if not css_df.empty:
            css_df.to_excel(writer, sheet_name='CSS', index=False)
    
    print(f"Feasible solutions saved to: {save_path}")


def simple_CSS(grid,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,CSS_L_solver='gurobi',CSS_NL_solver='bonmin',time_limit=300,NL=False,tee=False,export=True,fs=False):

    grid.Array_opf = False
    if NL:
        model, model_results , timing_info, solver_stats= transmission_expansion(grid,NPV,n_years,Hy,discount_rate,ObjRule,CSS_NL_solver,time_limit,tee,export)
    else:
        model, model_results , timing_info, solver_stats= linear_transmission_expansion(grid,NPV,n_years,Hy,discount_rate,None,CSS_L_solver,time_limit,tee,export,fs)

    return model, model_results , timing_info, solver_stats
