import time
import os
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import math
import pyomo.environ as pyo
import pandas as pd
try:
    import gurobipy
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False

try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False

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


def sequential_CSS(grid,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,max_turbines_per_string=None,limit_crossings=True,sub_min_connections=True,MIP_solver='glpk',CSS_L_solver='glpk',CSS_NL_solver='bonmin',svg=None,max_iter=None,time_limit=300,NL=False,tee=False,fs=False,save_path=None,MIP_gap=0.01,backend='pyomo'):
    
    # Determine save directory: create "sequential_CSS" folder
    if save_path is not None and os.path.isdir(save_path):
        # If save_path is provided and is a directory, create "sequential_CSS" inside it
        save_dir = os.path.join(save_path, 'sequential_CSS')
    else:
        # If save_path is None or not a directory, create "sequential_CSS" in current working directory
        save_dir = 'sequential_CSS'
    
    # Create the directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    

    staring_cables = grid.Cable_options[0].cable_types
    new_cables = staring_cables.copy()
  
    results = []
    tot_timing_info = {}
    i = 0
    seq_path_time = 0
    seq_css_time = 0
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

    if tee:
        print(f'Starting sequential CSS for {grid.name}')

    while flag:
        timing_info = {}
        
        t1 = time.perf_counter()
        if sub_min_connections:
            if tee and i==0:
                print(f'Using min sub connections iterationfor sequential CSS')
            flag, high_flow,model_MIP,feasible_solutions_MIP ,ns, sub_iter , path_time = min_sub_connections(grid, max_flow,solver_name=MIP_solver, crossings=limit_crossings, tee=tee,callback=fs,MIP_gap=MIP_gap,backend=backend)
        else:
            if tee and i==0:
                print(f'Using user defined substation limit path graph for sequential CSS')
            flag, high_flow,model_MIP,feasible_solutions_MIP = MIP_path_graph(grid, max_flow, solver_name=MIP_solver, crossings=limit_crossings, tee=tee,callback=fs,MIP_gap=MIP_gap,backend=backend)
            sub_iter = 1
        
        t2 = time.perf_counter()
        if tee:     
            print(f'Iteration {i} MIP finished in {t2 - t1} seconds')
        timing_info['Paths'] = t2 - t1
        seq_path_time += t2 - t1

        if not flag:
            if i == 0:
                # If MIP fails on first iteration, return None
                if tee:
                    print(f'MIP failed on first iteration, returning None')
                return None, None, None, None,i
            else:
                # If MIP fails on later iterations, break the loop
                if tee:
                    print(f'MIP failed on iteration {i}, breaking loop')
                break
        # Handle both Pyomo model and OR-Tools MockModel
        if hasattr(model_MIP, 'objective'):
            # Pyomo model
            MIP_obj_value = pyo.value(model_MIP.objective)
        elif hasattr(model_MIP, 'objective_value'):
            # OR-Tools MockModel
            MIP_obj_value = model_MIP.objective_value
        else:
            raise AttributeError("model_MIP must have either 'objective' (Pyomo) or 'objective_value' (OR-Tools) attribute")
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
        if tee:
            print(f'Iteration {i} CSS finished in {t4 - t3} seconds')
        timing_info['CSS'] = t4 - t3
        seq_css_time += t4 - t3
        if svg is not None:
            if tee:
                print(f'Iteration {i} saving SVG')
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

            
            # Save SVG in the sequential_CSS folder
            intermediate_dir = os.path.join(save_dir, 'intermediate_networks')
            if not os.path.exists(intermediate_dir):
                os.makedirs(intermediate_dir)
            save_network_svg(grid, name=f'{intermediate_dir}/{svg}_{i}_{CSS_L_solver}', width=1000, height=1000, journal=True,square_ratio=True, legend=True)
        
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
            'sub_iter': sub_iter,
            'i': i,
            'feasible_solutions_MIP': feasible_solutions_MIP,
            'feasible_solutions_CSS': feasible_solutions_CSS
        }
        results.append(iteration_result)  # Add to the results list   
        
        if i > 0 and obj_value is not None and results[i-1]['model_obj'] is not None:
            if obj_value > results[i-1]['model_obj']:
                if tee:
                    print(f'Iteration {i} objective value increased, breaking loop')
                break
        i += 1
        if i > max_iter:
            if tee:
                print(f'Iteration {i} max iterations reached, breaking loop')
            break
        # Update grid with new cable set
        if len(new_cables) > 0:
            grid.Cable_options[0].cable_types = new_cables
            
            # Recalculate max_flow based on current cable set
            max_cable_capacity = max(grid.Cable_options[0].MVA_ratings)
            max_flow = int(max_cable_capacity / t_MW)
            if tee:
                print(f'Iteration {i} max flow updated to {max_flow}')
        else:
            if tee:
                print(f'Iteration {i} no more cable types available, breaking loop')
            break
        if tee:
            print(f'Iteration {i} finished updating grid with new cable set')
    if tee:
        print(f'Sequential CSS finished in {time.perf_counter() - t0} seconds')
    # After the while loop ends, create summary from all iterations
    summary_results = {
        'cable_length': [result['cable_length'] for result in results],
        'model_obj':    [result['model_obj'] for result in results],
        'cable_options': [result['cable_options'] for result in results],
        'cables_used':  [result['cables_used'] for result in results],
        'timing_info':  [result['timing_info'] for result in results],
        'solver_status':[result['model_results']['Solver'][0]['Status']  for result in results],
        'iteration':    [result['i'] for result in results],
        'sub_iter':     [result['sub_iter'] for result in results],
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
        # Save feasible solutions plot in the sequential_CSS folder
        feasible_sol_gap_path = os.path.join(save_dir, f'feasible_solutions_{grid.name}_gap.png')
        feasible_sol_obj_path = os.path.join(save_dir, f'feasible_solutions_{grid.name}_obj.png')
        _plot_feasible_solutions_subplots(
            feasible_solutions_MIP,
            feasible_solutions_CSS,
            show=False,
            save_path=feasible_sol_obj_path,
            type='obj'
        )
        _plot_feasible_solutions_subplots(
            feasible_solutions_MIP,
            feasible_solutions_CSS,
            show=False,
            save_path=feasible_sol_gap_path,
            type='gap'
        )
        # Export feasible solutions to Excel/CSV
        feasible_sol_excel_path = os.path.join(save_dir, f'feasible_solutions_{grid.name}.csv')
        _export_feasible_solutions_to_excel(
            feasible_solutions_MIP,
            feasible_solutions_CSS,
            save_path=feasible_sol_excel_path
        )

    
    

    model = best_result['CSS_model']
    model_MIP = best_result['MIP_model']
    model_results = best_result['model_results']
    tot_timing_info['Paths'] = seq_path_time
    tot_timing_info['CSS'] = seq_css_time
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
    


def min_sub_connections(grid, max_flow=None, solver_name='glpk', crossings=True, tee=False, callback=False,MIP_gap=None, backend='pyomo'):
    tn = grid.n_ren
    sn = grid.nn_AC - grid.n_ren

    ns = math.ceil(tn/(sn* max_flow))

    flag=False
    i =0
    max_iters = 1 if ns is None else 10
    if tee:
        print(f'Starting min sub connections for {grid.name}')
    while not flag and i<max_iters:
        if tee:
            print(f'Iteration sub-{i} starting min sub connections')
        for node in grid.nodes_AC:
            if node.type == 'Slack':
                node.ct_limit = ns

        t0 = time.perf_counter()
        flag, high_flow,model_MIP,feasible_solutions_MIP = MIP_path_graph(grid, max_flow, solver_name, crossings, tee,callback,MIP_gap,backend)
        t1 = time.perf_counter()
        path_time = t1 - t0
        i+=1
        if not flag:
            if ns is not None:
                ns+=1
            if tee:
                print(f'Iteration sub-{i} ns increased to {ns}')

    if tee:
        print(f'Min sub connections finished in {time.perf_counter() - t0} seconds')
        print(f'Final ns: {ns}')
    return flag, high_flow,model_MIP,feasible_solutions_MIP ,ns, i , path_time




def MIP_path_graph(grid, max_flow=None, solver_name='glpk', crossings=False, tee=False, callback=False, MIP_gap=None, backend='pyomo',
                   enable_cable_types=False, t_MW=None, cab_types_allowed=None):
    """
    Solve the master MIP problem and track feasible solutions over time.
    
    Parameters:
    -----------
    backend : str, optional
        Backend to use: 'pyomo' (default) or 'ortools'
        - 'pyomo': Uses Pyomo with external solver (GLPK, Gurobi, etc.)
        - 'ortools': Uses OR-Tools CP-SAT solver (built-in, faster)
    enable_cable_types : bool
        If True, enable individual cable type selection per line
    t_MW : float
        Turbine MW rating (needed to calculate flow capacity from MVA ratings)
    cab_types_allowed : int, optional
        Maximum number of cable types that can be used (linking constraint)
    
    Returns:
    --------
    success : bool
        True if solution found, False otherwise
    high_flow : float or None
        Maximum flow value in solution
    model : Pyomo model or MockModel
        Solved model object
    feasible_solutions : list
        List of (time, objective_value, gap) tuples if callback=True
    """
    # Route to appropriate backend
    if backend.lower() == 'ortools':
        if not ORTOOLS_AVAILABLE:
            raise ImportError(
                "OR-Tools is not installed. Please install it with: pip install ortools\n"
                "Alternatively, use backend='pyomo' (default) which uses Pyomo with external solvers."
            )
        # Note: OR-Tools version doesn't support cable types yet
        if enable_cable_types:
            raise NotImplementedError("Cable type selection not yet implemented for OR-Tools backend")
        return MIP_path_graph_ortools(grid, max_flow=max_flow, crossings=crossings, 
                                      tee=tee, callback=callback, MIP_gap=MIP_gap)
    elif backend.lower() != 'pyomo':
        raise ValueError(f"Unknown backend: {backend}. Must be 'pyomo' or 'ortools'")
    
    # Original Pyomo implementation
    model = _create_master_problem_pyomo(grid, crossings, max_flow, 
                                         enable_cable_types=enable_cable_types,
                                         t_MW=t_MW,
                                         cab_types_allowed=cab_types_allowed)
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
        
        # MIPFocus: 0=balanced, 1=feasibility, 2=optimality, 3=bound improvement
        # For faster gap reduction, use 2 (optimality) or 3 (bound improvement)
        # Default was 1 (feasibility) which prioritizes finding solutions over closing gap
        mip_focus = getattr(grid, "MIP_focus", 2)  # Default to 2 for better gap reduction
        grb_model.setParam("MIPFocus", mip_focus)
        
       
        # Additional parameters to improve gap reduction:
        # Increase cutting planes to strengthen LP relaxation
        grb_model.setParam("Cuts", 2)  # 2=aggressive cutting
        
        # Increase heuristics to find better solutions faster
        grb_model.setParam("Heuristics", 0.05)  # Spend 5% of time on heuristics
        
        # Improve presolve to reduce problem size
        grb_model.setParam("Presolve", 2)  # 2=aggressive presolve
        
        if MIP_gap is not None:
            grb_model.setParam("MIPGap", MIP_gap)

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
            
            # For OPTIMAL/SUBOPTIMAL, SolCount should always be > 0, but check defensively
            if grb_model.SolCount > 0:
                solver.load_vars()
                # Calculate final gap
                final_gap = None
                obj_val = grb_model.ObjVal
                obj_bound = grb_model.ObjBound
                if (obj_bound != GRB.INFINITY and obj_bound != -GRB.INFINITY and 
                    abs(obj_val) > 1e-10):
                    model_sense = grb_model.ModelSense
                    if model_sense == GRB.MINIMIZE:
                        final_gap = (obj_val - obj_bound) / abs(obj_val)
                    else:  # MAXIMIZE
                        final_gap = (obj_bound - obj_val) / abs(obj_val)
                feasible_solutions.append((grb_model.Runtime, obj_val, final_gap))
            else:
                # This shouldn't happen for OPTIMAL/SUBOPTIMAL, but handle gracefully
                feasible_solution_found = False
            
        grb_model.dispose()

    # === Other solvers (no callback) ===
    else:
        solver = pyo.SolverFactory(solver_name)
        if solver_name == 'gurobi':
            # Set Gurobi-specific parameters regardless of MIP_time
            # Use MIPFocus=2 for better gap reduction instead of 1
            mip_focus = getattr(grid, "MIP_focus", 2)
            solver.options['MIPFocus'] = mip_focus
            solver.options['Cuts'] = 2  # Aggressive cutting
            solver.options['Heuristics'] = 0.05
            solver.options['Presolve'] = 2
            if MIP_gap is not None:
                solver.options['MIPGap'] = MIP_gap
            if getattr(grid, "MIP_time", None) is not None:
                solver.options['TimeLimit'] = grid.MIP_time
        elif solver_name == 'glpk':
            if getattr(grid, "MIP_time", None) is not None:
                solver.options['tmlim'] = grid.MIP_time
        elif solver_name == 'cbc':
            if getattr(grid, "MIP_time", None) is not None:
                solver.options['seconds'] = grid.MIP_time
            if MIP_gap is not None:
                solver.options['ratioGap'] = MIP_gap
        elif solver_name == 'highs':
            if getattr(grid, "MIP_time", None) is not None:
                solver.options['time_limit'] = grid.MIP_time
            if MIP_gap is not None:
                solver.options['mip_rel_gap'] = MIP_gap
        try:
            results = solver.solve(model, tee=tee)
            # Check solver results to determine if solve was successful
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                feasible_solution_found = True
            elif results.solver.termination_condition == pyo.TerminationCondition.feasible:
                feasible_solution_found = True
            elif results.solver.termination_condition == pyo.TerminationCondition.maxTimeLimit:
                # Check if we have a solution even if time limit was hit
                try:
                    _ = pyo.value(model.objective)
                    feasible_solution_found = True
                except (ValueError, AttributeError):
                    feasible_solution_found = False
            elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
                feasible_solution_found = False
            else:
                # Unknown status - try to access objective to see if solution exists
                try:
                    _ = pyo.value(model.objective)
                    feasible_solution_found = True
                except (ValueError, AttributeError):
                    feasible_solution_found = False
            
        except (ValueError, AttributeError):
            feasible_solution_found = False

    # === Post-solve handling ===
    if feasible_solution_found:
        flows = [abs(pyo.value(model.line_flow[line])) for line in model.lines]
        high_flow = max(flows) if flows else 0
        
        # Assign cable types to lines
        for line in model.lines:
            ct_line = grid.lines_AC_ct[line]
            line_used = pyo.value(model.line_used[line]) > 0.5
            
            if not line_used:
                ct_line.active_config = -1
            elif enable_cable_types and hasattr(model, 'ct_branch'):
                # Read selected cable type from ct_branch
                selected_ct = None
                for ct in model.ct_set:
                    if pyo.value(model.ct_branch[line, ct]) > 0.5:
                        selected_ct = ct
                        break
                if selected_ct is not None:
                    ct_line.active_config = selected_ct
                else:
                    # Fallback: use last cable type if no selection found
                    last_cable_type_index = len(grid.Cable_options[0]._cable_types) - 1
                    ct_line.active_config = last_cable_type_index
            else:
                # No cable type selection: use last cable type as default
                last_cable_type_index = len(grid.Cable_options[0]._cable_types) - 1
                ct_line.active_config = last_cable_type_index

        return True, high_flow, model, feasible_solutions

    else:
        print("✗ MIP model failed")
        return False, None, None, feasible_solutions

def MIP_path_graph_ortools(grid, max_flow=None, crossings=False, tee=False, callback=False, MIP_gap=None):
    """Solve the master MIP problem using OR-Tools CP-SAT solver."""
    if not ORTOOLS_AVAILABLE:
        raise ImportError(
            "OR-Tools is not installed. Please install it with: pip install ortools"
        )
    length_scale = 1000
    from ortools.sat.python import cp_model
    
    # Create model
    model, vars_dict = _create_master_problem_ortools(grid, crossings, max_flow, length_scale)
    
    feasible_solutions = []
    feasible_solution_found = False
    high_flow = None
    
    # Create solver
    solver = cp_model.CpSolver()
    
    # Set solver parameters
    if tee:
        solver.parameters.log_search_progress = True
    
    # Set time limit if specified
    if hasattr(grid, "MIP_time") and grid.MIP_time is not None:
        solver.parameters.max_time_in_seconds = grid.MIP_time
    
    # Set MIP gap if specified (CP-SAT uses relative gap)
    if MIP_gap is not None:
        solver.parameters.relative_gap_limit = MIP_gap
    
    # Callback for tracking feasible solutions (if requested)
    if callback:
        class SolutionCallback(cp_model.CpSolverSolutionCallback):
            def __init__(self, vars_dict, feasible_solutions):
                cp_model.CpSolverSolutionCallback.__init__(self)
                self.vars_dict = vars_dict
                self.feasible_solutions = feasible_solutions
                self.solution_count = 0
            
            def on_solution_callback(self):
                self.solution_count += 1
                runtime = self.WallTime()
                objective = self.ObjectiveValue()
                
                # Try to get bound if available (CP-SAT may not provide in callback)
                try:
                    bound = self.BestObjectiveBound()
                except:
                    bound = None
                
                # Calculate relative gap
                relgap = None
                if bound is not None and objective is not None and abs(objective) > 1e-10:
                    relgap = 1.0 - bound / objective
                
                # Store as tuple for compatibility: (time, objective, gap)
                self.feasible_solutions.append((runtime, objective, relgap))
        
        callback_obj = SolutionCallback(vars_dict, feasible_solutions)
        status = solver.Solve(model, callback_obj)
    else:
        status = solver.Solve(model)
    
    # Check solution status
    if status == cp_model.OPTIMAL:
        feasible_solution_found = True
    elif status == cp_model.FEASIBLE:
        feasible_solution_found = True
    elif status == cp_model.INFEASIBLE:
        feasible_solution_found = False
    elif status == cp_model.MODEL_INVALID:
        feasible_solution_found = False
    else:
        # TIMEOUT or other status - check if we have a solution by checking objective value
        # CP-SAT returns a very large value if no solution found
        try:
            obj_val = solver.ObjectiveValue()
            # If objective is reasonable (not infinity), we have a solution
            feasible_solution_found = obj_val < 1e20
        except:
            feasible_solution_found = False
    
    # Post-solve handling
    if feasible_solution_found:
        # Extract solution values
        line_used_vals = {}
        line_flow_vals = {}
        
        for l in vars_dict["line_used"]:
            line_used_vals[l] = solver.Value(vars_dict["line_used"][l])
            line_flow_vals[l] = solver.Value(vars_dict["line_flow"][l])
        
        # Calculate high flow
        flows = [abs(line_flow_vals[l]) for l in line_flow_vals]
        high_flow = max(flows) if flows else 0
        
        # Update grid with solution
        last_cable_type_index = len(grid.Cable_options[0]._cable_types) - 1
        for l in line_used_vals:
            ct_line = grid.lines_AC_ct[l]
            ct_line.active_config = last_cable_type_index if line_used_vals[l] > 0 else -1
        
        # Create SolutionInfo for final solution
        objective = solver.ObjectiveValue()
        runtime = solver.WallTime()
        
        # Get bound if available
        try:
            bound = solver.BestObjectiveBound()
        except:
            bound = None
        
        # Calculate relative gap
        relgap = None
        if bound is not None and objective is not None and abs(objective) > 1e-10:
            relgap = 1.0 - bound / objective
        elif status == cp_model.OPTIMAL:
            relgap = 0.0  # Optimal means gap is 0
        
        # Get termination status name
        termination_map = {
            cp_model.OPTIMAL: 'OPTIMAL',
            cp_model.FEASIBLE: 'FEASIBLE',
            cp_model.INFEASIBLE: 'INFEASIBLE',
            cp_model.MODEL_INVALID: 'MODEL_INVALID',
            cp_model.UNKNOWN: 'UNKNOWN'
        }
        termination = termination_map.get(status, 'UNKNOWN')
    
        # Add final solution to feasible_solutions if callback was used
        if callback:
            # Store as tuple for compatibility: (time, objective, gap)
            feasible_solutions.append((runtime, objective, relgap))
        
        # Create a mock model-like object for compatibility with Pyomo version
        class MockModel:
            def __init__(self, vars_dict, line_used_vals, line_flow_vals, objective_value):
                self.vars_dict = vars_dict
                self.line_used_vals = line_used_vals
                self.line_flow_vals = line_flow_vals
                self.objective_value = objective_value
             
        
        mock_model = MockModel(vars_dict, line_used_vals, line_flow_vals, objective)
        
        return True, high_flow, mock_model, feasible_solutions
    else:
        print("✗ MIP model failed (OR-Tools)")
        return False, None, None, feasible_solutions

def _create_master_problem_pyomo(grid,crossings=True, max_flow=None, 
                                  enable_cable_types=False, 
                                  t_MW=None,
                                  cab_types_allowed=None):
        """Create master problem using Pyomo
        
        Parameters:
        -----------
        enable_cable_types : bool
            If True, enable individual cable type selection per line
        t_MW : float
            Turbine MW rating (needed to calculate flow capacity from MVA ratings)
        cab_types_allowed : int, optional
            Maximum number of cable types that can be used (linking constraint)
        """
        
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
        
        # Cable type selection (if enabled)
        if enable_cable_types:
            if grid.Cable_options is None or len(grid.Cable_options) == 0:
                raise ValueError("enable_cable_types=True but no Cable_options found in grid")
            if t_MW is None:
                raise ValueError("t_MW must be provided when enable_cable_types=True")
            
            # Cable type set
            model.ct_set = pyo.Set(initialize=range(len(grid.Cable_options[0]._cable_types)))
            
            # Calculate flow capacity for each cable type (in turbine units)
            # Flow capacity = int(MVA_rating / t_MW)
            ct_flow_capacity = {}
            for ct in model.ct_set:
                mva_rating = grid.Cable_options[0].MVA_ratings[ct]
                ct_flow_capacity[ct] = int(mva_rating / t_MW)
            model.ct_flow_capacity = pyo.Param(model.ct_set, initialize=ct_flow_capacity)
            
            # Cable type selection: ct_branch[line, ct] = 1 if cable type ct selected for line
            model.ct_branch = pyo.Var(model.lines, model.ct_set, domain=pyo.Binary)
            
            # Optional: Global cable type indicator (Z_k style from image)
            # ct_types[ct] = 1 if cable type ct is used anywhere
            model.ct_types = pyo.Var(model.ct_set, domain=pyo.Binary)
            
            # LINKING: line_used and ct_branch are linked via cable_type_selection_rule constraint:
            # sum(ct_branch[line, ct]) == line_used[line]
            # This ensures:
            # - If line_used[line] = 1: exactly one ct_branch[line, ct] = 1 (one cable type selected)
            # - If line_used[line] = 0: all ct_branch[line, ct] = 0 (no cable type selected)
        
        def line_flow_bounds(model, line):
            line_obj = grid.lines_AC_ct[line]
            from_node = line_obj.fromNode
            to_node = line_obj.toNode
            
            # Check if either endpoint is a slack node (substation)
            from_is_slack = from_node.type == 'Slack'
            to_is_slack = to_node.type == 'Slack'
            
            # If cable types enabled, use max capacity from all cable types
            if enable_cable_types:
                max_ct_flow = max(model.ct_flow_capacity[ct] for ct in model.ct_set)
                # If line connects to slack node: use full capacity
                if from_is_slack or to_is_slack:
                    return (-max_ct_flow, max_ct_flow)
                else:
                    # If line connects PQ-PQ (turbine-turbine): use reduced capacity
                    return (-(max_ct_flow - 1), max_ct_flow - 1)
            else:
                # Original bounds (no cable type selection)
                if from_is_slack or to_is_slack:
                    return (-max_flow, max_flow)
                else:
                    return (-(max_flow - 1), max_flow - 1)
        
        # Flow variables (integer - can carry flow in either direction)
        model.line_flow = pyo.Var(model.lines, domain=pyo.Integers, bounds=line_flow_bounds)
        model.node_flow = pyo.Var(model.nodes, domain=pyo.Integers)
        model.line_flow_dir = pyo.Var(model.lines, domain=pyo.Binary)
        # Objective: minimize total cable length (+ optional investment cost)
        def objective_rule(model):
           
            if enable_cable_types:
                # Add cable type investment costs
                investment_cost = 0
                for line in model.lines:
                    line_obj = grid.lines_AC_ct[line]
                    for ct in model.ct_set:
                        investment_cost += model.ct_branch[line, ct] * line_obj.base_cost[ct]
                return investment_cost
            else:
                length_cost = sum(model.line_used[line] * grid.lines_AC_ct[line].Length_km 
                             for line in model.lines)
            
                return length_cost
        
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
                if enable_cable_types:
                    # Use max cable type capacity for sink capacity calculation
                    max_ct_flow = max(model.ct_flow_capacity[ct] for ct in model.ct_set)
                    min_connections = math.ceil(nT/(nS*max_ct_flow))
                else:
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
        if enable_cable_types:
            # Flow capacity linked to selected cable type
            def flow_capacity_upper_rule(model, line):
                # Flow <= sum(flow_capacity[ct] * ct_branch[line, ct])
                return model.line_flow[line] <= sum(model.ct_flow_capacity[ct] * model.ct_branch[line, ct] 
                                                   for ct in model.ct_set)
            
            def flow_capacity_lower_rule(model, line):
                # Flow >= -sum(flow_capacity[line, ct] * ct_branch[line, ct])
                return model.line_flow[line] >= -sum(model.ct_flow_capacity[ct] * model.ct_branch[line, ct] 
                                                    for ct in model.ct_set)
            
            # Cable type selection: each used line must have exactly one cable type
            # CONSTRAINT LINKING line_used and ct_branch:
            # sum(ct_branch[line, ct]) == line_used[line]
            # - If line_used[line] = 1: exactly one ct_branch[line, ct] = 1 (one cable type selected)
            # - If line_used[line] = 0: all ct_branch[line, ct] = 0 (no cable type selected)
            def cable_type_selection_rule(model, line):
                return sum(model.ct_branch[line, ct] for ct in model.ct_set) == model.line_used[line]
            
            model.flow_capacity_upper = pyo.Constraint(model.lines, rule=flow_capacity_upper_rule)
            model.flow_capacity_lower = pyo.Constraint(model.lines, rule=flow_capacity_lower_rule)
            model.cable_type_selection = pyo.Constraint(model.lines, rule=cable_type_selection_rule)
            
            # Link ct_types to ct_branch using homogeneity constraint from image formulation
            # Constraint: sum(ct_branch[line, ct]) - (NN-1) * ct_types[ct] <= 0
            # Which is: sum(ct_branch[line, ct]) <= (NN-1) * ct_types[ct]
            # Where NN-1 = len(nodes) - len(sink_nodes) (spanning tree has exactly this many lines)
            # This constraint enforces BOTH directions:
            # - If ct_types[ct] = 0: then sum(ct_branch) <= 0, so no lines use ct
            # - If sum(ct_branch) > 0 (any line uses ct): then ct_types[ct] must be 1
            #   (because if ct_types[ct] = 0, we'd have sum(ct_branch) <= 0, contradiction)
            # So the lower bound constraint is NOT needed - homogeneity constraint is sufficient!
            NN_minus_1 = len(model.nodes) - len(model.sink_nodes)  # Number of lines in spanning tree
            
            def ct_types_homogeneity_rule(model, ct):
                # Image formulation: sum X_i,j,k - (NN-1) * Z_k <= 0
                return sum(model.ct_branch[line, ct] for line in model.lines) - NN_minus_1 * model.ct_types[ct] <= 0
            
            model.ct_types_homogeneity = pyo.Constraint(model.ct_set, rule=ct_types_homogeneity_rule)
            
            # Optional: Limit total cable types used (linking constraint)
            if cab_types_allowed is not None:
                def ct_limit_rule(model):
                    return sum(model.ct_types[ct] for ct in model.ct_set) <= cab_types_allowed
                model.ct_limit = pyo.Constraint(rule=ct_limit_rule)
        else:
            # Original flow constraints (no cable type selection)
            def flow_investment_rule(model, line):
                return model.line_flow[line] <= max_flow * model.line_used[line]

            def flow_investment_rule_2(model, line):
                return model.line_flow[line] >= -max_flow * model.line_used[line]
            
            model.flow_investment_link = pyo.Constraint(model.lines, rule=flow_investment_rule)
            model.flow_investment_link_2 = pyo.Constraint(model.lines, rule=flow_investment_rule_2)

        # NEW: If line is used, flow must be >= 1 (when positive) OR <= -1 (when negative)
        def flow_nonzero_positive(model, line):
            # If line_used=1 and line_flow_dir=1, then line_flow >= 1
            # If line_used=0 or line_flow_dir=0, this constraint is relaxed
            if enable_cable_types:
                max_ct_flow = max(model.ct_flow_capacity[ct] for ct in model.ct_set)
                M = max_ct_flow + 1
            else:
                M = max_flow + 1
            return model.line_flow[line] >= 1 - M * (1 - model.line_used[line]) - M * (1 - model.line_flow_dir[line])

        def flow_nonzero_negative(model, line):
            # If line_used=1 and line_flow_dir=0, then line_flow <= -1
            # If line_used=0 or line_flow_dir=1, this constraint is relaxed
            if enable_cable_types:
                max_ct_flow = max(model.ct_flow_capacity[ct] for ct in model.ct_set)
                M = max_ct_flow + 1
            else:
                M = max_flow + 1
            return model.line_flow[line] <= -1 + M * (1 - model.line_used[line]) + M * model.line_flow_dir[line]

        # Ensure direction variable is only active when line is used
        def flow_dir_active(model, line):
            return model.line_flow_dir[line] <= model.line_used[line]

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
    




def _create_master_problem_ortools(grid, crossings=True, max_flow=None, length_scale=1000):
    """
    OR-Tools version of _create_master_problem_pyomo(grid, crossings=True, max_flow=None)

    Returns:
        model: cp_model.CpModel
        vars_dict: {
            "line_used": {line: BoolVar},
            "line_flow": {line: IntVar},
            "line_flow_dir": {line: BoolVar},
            "node_flow": {node: IntVar},
            "source_nodes": list[int],
            "sink_nodes": list[int],
        }
    """
    if not ORTOOLS_AVAILABLE:
        raise ImportError(
            "OR-Tools is not installed. Please install it with: pip install ortools"
        )
    
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()

    # --------------------
    # Basic sets and parameters
    # --------------------
    num_lines = len(grid.lines_AC_ct)
    num_nodes = len(grid.nodes_AC)

    lines = range(num_lines)
    nodes = range(num_nodes)

    if max_flow is None:
        max_flow = num_nodes - 1

    # Identify sources (renewables) and sinks (generators / substations)
    source_nodes = []
    sink_nodes = []

    for n in nodes:
        nAC = grid.nodes_AC[n]
        if getattr(nAC, "connected_gen", False):
            sink_nodes.append(n)
        if getattr(nAC, "connected_RenSource", False):
            source_nodes.append(n)

    if not sink_nodes:
        raise ValueError("No generator nodes found!")

    source_nodes_set = set(source_nodes)
    sink_nodes_set = set(sink_nodes)

    # Non-sink nodes count
    nT = num_nodes - len(sink_nodes)
    nS = len(sink_nodes)

    # Minimum connections for sink nodes (same formula as Pyomo version)
    # ceil(non_sink_nodes / (total_sink_capacity))
    if nS > 0:
        min_connections_per_sink = math.ceil(nT / (nS * max_flow))
    else:
        min_connections_per_sink = 0

    # --------------------
    # Precompute line incidence and bounds
    # --------------------
    incident_lines = {n: [] for n in nodes}
    line_from = {}
    line_to = {}
    line_lb = {}
    line_ub = {}

    for l in lines:
        line_obj = grid.lines_AC_ct[l]
        from_node = line_obj.fromNode
        to_node = line_obj.toNode

        from_idx = from_node.nodeNumber
        to_idx = to_node.nodeNumber

        line_from[l] = from_idx
        line_to[l] = to_idx

        incident_lines[from_idx].append(l)
        incident_lines[to_idx].append(l)

        # Bounds depend on slack nodes (substations)
        from_is_slack = getattr(from_node, "type", None) == "Slack"
        to_is_slack = getattr(to_node, "type", None) == "Slack"

        if from_is_slack or to_is_slack:
            lb, ub = -max_flow, max_flow
        else:
            lb, ub = -(max_flow - 1), (max_flow - 1)

        line_lb[l] = lb
        line_ub[l] = ub

    # Node flow bounds: each source contributes +1, sinks absorb −1
    max_node_flow = len(source_nodes)

    # --------------------
    # Variables
    # --------------------
    # Binary: line_used[l] = 1 if line is active
    line_used = {
        l: model.NewBoolVar(f"line_used[{l}]")
        for l in lines
    }

    # Binary: line_flow_dir[l] = 1 for "positive" orientation, 0 for "negative"
    line_flow_dir = {
        l: model.NewBoolVar(f"line_flow_dir[{l}]")
        for l in lines
    }

    # Integer flow on each line (can be negative)
    line_flow = {
        l: model.NewIntVar(line_lb[l], line_ub[l], f"line_flow[{l}]")
        for l in lines
    }

    # Integer net flow at each node
    node_flow = {
        n: model.NewIntVar(-max_node_flow, max_node_flow, f"node_flow[{n}]")
        for n in nodes
    }

    # --------------------
    # Objective: minimize total cable length
    # CP-SAT needs integer coefficients: we scale Length_km by length_scale.
    # Using math.ceil to round up to nearest meter integer (conservative approach).
    # Effective objective = sum(line_used[l] * Length_km[l]) up to a constant factor.
    # --------------------
    coeffs = []
    for l in lines:
        length_km = grid.lines_AC_ct[l].Length_km
        coeff = math.ceil(length_km * length_scale)  # Round up to nearest meter
        coeffs.append(coeff)

    model.Minimize(sum(coeffs[l] * line_used[l] for l in lines))

    # --------------------
    # Spanning tree constraint:
    #   sum(line_used) == num_nodes - num_sink_nodes
    # (same as original)
    # --------------------
    model.Add(sum(line_used[l] for l in lines) == num_nodes - len(sink_nodes))

    # --------------------
    # Connection constraints
    # --------------------
    # Upper bound: ct_limit per node if given
    for n in nodes:
        nAC = grid.nodes_AC[n]
        ct_limit = getattr(nAC, "ct_limit", None)
        if ct_limit is not None:
            model.Add(
                sum(line_used[l] for l in incident_lines[n]) <= ct_limit
            )

    # Lower bound:
    # - For sinks: >= min_connections_per_sink
    # - For non-sinks: >= 1
    for n in nodes:
        deg = sum(line_used[l] for l in incident_lines[n])

        if n in sink_nodes_set:
            if min_connections_per_sink > 0:
                model.Add(deg >= min_connections_per_sink)
        else:
            model.Add(deg >= 1)

    # --------------------
    # Flow conservation: node_flow[n] = sum(outgoing) - sum(ingoing)
    # --------------------
    for n in nodes:
        expr_terms = []
        for l in incident_lines[n]:
            if line_from[l] == n:
                expr_terms.append(line_flow[l])
            elif line_to[l] == n:
                expr_terms.append(-line_flow[l])
            else:
                # Should not happen
                pass
        if expr_terms:
            model.Add(node_flow[n] == sum(expr_terms))
        else:
            model.Add(node_flow[n] == 0)

    # --------------------
    # Source nodes: node_flow[n] == 1
    # --------------------
    for n in source_nodes:
        model.Add(node_flow[n] == 1)

    # --------------------
    # Intermediate nodes: net flow = 0 (not source, not sink)
    # --------------------
    for n in nodes:
        if n not in source_nodes_set and n not in sink_nodes_set:
            model.Add(node_flow[n] == 0)

    # --------------------
    # Total sink absorption: sum(node_flow[sink]) == -len(source_nodes)
    # --------------------
    if sink_nodes:
        model.Add(
            sum(node_flow[n] for n in sink_nodes) == -len(source_nodes)
        )

    # --------------------
    # Link flow to investment: |line_flow| <= max_flow * line_used
    # --------------------
    for l in lines:
        model.Add(line_flow[l] <= max_flow * line_used[l])
        model.Add(line_flow[l] >= -max_flow * line_used[l])

    # --------------------
    # Nonzero flow when line is used:
    #   If line_used=1 and line_flow_dir=1 -> line_flow >= 1
    #   If line_used=1 and line_flow_dir=0 -> line_flow <= -1
    #
    # Using Big-M linearization (expanded to avoid product of vars)
    # --------------------
    M = max_flow + 1

    # flow_nonzero_positive:
    # line_flow >= 1 - M*(1 - line_used) - M*(1 - line_flow_dir)
    # Expand:
    #   = 1 - M + M*line_used - M + M*line_flow_dir
    #   = (1 - 2M) + M*line_used + M*line_flow_dir
    for l in lines:
        model.Add(
            line_flow[l] >=
            (1 - 2 * M)
            + M * line_used[l]
            + M * line_flow_dir[l]
        )

    # flow_nonzero_negative:
    # line_flow <= -1 + M*(1 - line_used) + M*(line_flow_dir)
    # Expand:
    #   = -1 + M - M*line_used + M*line_flow_dir
    for l in lines:
        model.Add(
            line_flow[l] <=
            (-1 + M)
            - M * line_used[l]
            + M * line_flow_dir[l]
        )

    # Direction variable only active when line is used:
    # line_flow_dir <= line_used
    for l in lines:
        model.Add(line_flow_dir[l] <= line_used[l])

    # --------------------
    # Crossing constraints: at most one line per crossing group
    # --------------------
    if crossings and hasattr(grid, "crossing_groups") and grid.crossing_groups:
        # Map lineNumber -> index
        line_number_to_idx = {
            grid.lines_AC_ct[l].lineNumber: l
            for l in lines
        }

        for group_idx, group in enumerate(grid.crossing_groups):
            # group is a collection of lineNumbers
            line_indices = [
                line_number_to_idx[ln]
                for ln in group
                if ln in line_number_to_idx
            ]
            if line_indices:
                model.Add(
                    sum(line_used[l] for l in line_indices) <= 1
                )

    # --------------------
    # Return model + handy variable dict
    # --------------------
    vars_dict = {
        "line_used": line_used,
        "line_flow": line_flow,
        "line_flow_dir": line_flow_dir,
        "node_flow": node_flow,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
    }

    return model, vars_dict


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

def _export_feasible_solutions_to_excel(results_mip, results_css, save_path):
    """
    Export feasible solutions to separate CSV files (one for MIP, one for CSS) or Excel.
    Format: 0_t, 0_obj, 0_gap, 1_t, 1_obj, 1_gap, ...
    
    Args:
        results_mip: List of feasible solutions for MIP (each is list of (time, obj, gap) tuples)
        results_css: List of feasible solutions for CSS (each is list of (time, obj, gap) tuples)
        save_path: Base path (will create _MIP.csv and _CSS.csv, or single .xlsx file)
    """
    
    # Determine format from extension
    base_path = save_path
    if save_path.lower().endswith('.csv'):
        base_path = save_path[:-4]  # Remove .csv extension
        file_format = 'csv'
    elif save_path.lower().endswith('.xlsx'):
        file_format = 'excel'
    else:
        # Default to CSV
        file_format = 'csv'
    
    def _export_single(results, suffix, max_solutions):
        """Export a single set of results (MIP or CSS)"""
        if not results:
            return None
        
        data = {}
        max_iter = len(results)
        
        for i in range(max_iter):
            if results[i]:
                feas_sorted = sorted(results[i], key=lambda x: x[0])  # Sort by time
                # Pad to max_solutions with NaN
                times = [t for t, _, _ in feas_sorted] + [np.nan] * (max_solutions - len(feas_sorted))
                objs = [o for _, o, _ in feas_sorted] + [np.nan] * (max_solutions - len(feas_sorted))
                gaps = [g for _, _, g in feas_sorted] + [np.nan] * (max_solutions - len(feas_sorted))
                
                data[f'{i}_t'] = times[:max_solutions]
                data[f'{i}_obj'] = objs[:max_solutions]
                data[f'{i}_gap'] = gaps[:max_solutions]
        
        if not data:
            return None
        
        df = pd.DataFrame(data)
        
        if file_format == 'csv':
            csv_path = f"{base_path}_{suffix}.csv"
            df.to_csv(csv_path, index=False)
            print(f"Feasible solutions ({suffix}) exported to CSV: {csv_path}")
            return None
        else:
            # For Excel, return DataFrame to write to sheet
            return df
    
    # Find maximum number of feasible solutions for each type
    max_mip_solutions = max((len(feas) for feas in results_mip if feas), default=0)
    max_css_solutions = max((len(feas) for feas in results_css if feas), default=0)
    
    if max_mip_solutions == 0 and max_css_solutions == 0:
        print("Warning: No feasible solutions to export")
        return
    
    if file_format == 'csv':
        # Save separate CSV files
        if max_mip_solutions > 0:
            _export_single(results_mip, 'MIP', max_mip_solutions)
        if max_css_solutions > 0:
            _export_single(results_css, 'CSS', max_css_solutions)
    else:
        # Save to Excel with separate sheets
        excel_path = f"{base_path}.xlsx" if not base_path.endswith('.xlsx') else base_path
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            if max_mip_solutions > 0:
                df_mip = _export_single(results_mip, 'MIP', max_mip_solutions)
                if df_mip is not None:
                    df_mip.to_excel(writer, sheet_name='MIP', index=False)
            if max_css_solutions > 0:
                df_css = _export_single(results_css, 'CSS', max_css_solutions)
                if df_css is not None:
                    df_css.to_excel(writer, sheet_name='CSS', index=False)
        print(f"Feasible solutions exported to Excel: {excel_path}")


def _plot_feasible_solutions_subplots(results_mip, results_css, suptitle=None, show=True, save_path=None, width_mm=None,type='gap'):
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

    def _plot(ax, results, title,yaxis,type):
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
            # Unpack (time, objective, gap) tuples
            times = [t for t, _, _ in feas_sorted]
            if type == 'gap':
                # Handle None gaps - convert to percentage, use 0 if None
                gap = [(g * 100 if g is not None else 0) for _, _, g in feas_sorted]
            else:
                gap = [o for _, o, _ in feas_sorted]
                if title == 'CSS':
                    gap = [o/1e6 for o in gap]
            ax.plot(times, gap, 'o-', label=f'i={i} (s={len(gap)})', markersize=5, linewidth=2)
        ax.set_title(title, fontsize=FS*1.2)
        ax.set_xlabel('Time (s)', fontsize=FS*1.1)
        ax.set_ylabel(yaxis, fontsize=FS*1.1)
        if has_any:
            ax.legend(prop={'size': FS}, loc='upper right', frameon=False)
        ax.tick_params(labelsize=FS)
        ax.grid(True, alpha=0.3)
    
    if type == 'gap':
        _plot(axes[0], results_mip, 'MIP', 'Gap [%]',type)
        _plot(axes[1], results_css, 'CSS', 'Gap [%]',type)
    else:
        _plot(axes[0], results_mip, 'MIP', 'Cable length [km]',type)
        _plot(axes[1], results_css, 'CSS', 'Objective [M€]',type)
   

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



def simple_CSS(grid,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,CSS_L_solver='gurobi',CSS_NL_solver='bonmin',time_limit=300,NL=False,tee=False,export=True,fs=False):

    grid.Array_opf = False
    if NL:
        model, model_results , timing_info, solver_stats= transmission_expansion(grid,NPV,n_years,Hy,discount_rate,ObjRule,CSS_NL_solver,time_limit,tee,export)
    else:
        model, model_results , timing_info, solver_stats= linear_transmission_expansion(grid,NPV,n_years,Hy,discount_rate,None,CSS_L_solver,time_limit,tee,export,fs)

    return model, model_results , timing_info, solver_stats
