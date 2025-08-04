# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 18:25:02 2024

@author: BernardoCastro
Master problem with network flow for connected spanning tree
"""

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from .ACDC_OPF import analyse_OPF
from .Graph_and_plot import save_network_svg

# Optional Pyomo import
try:
    import pyomo.environ as pyo
    PYOMO_AVAILABLE = True
except ImportError:
    PYOMO_AVAILABLE = False
    print("Pyomo not available. Install with: pip install pyomo")



def create_master_problem_gurobi(grid, max_flow=None):
    master = gp.Model("Master")
    if max_flow is None:
        max_flow = len(grid.nodes_AC) - 1
    lista_lineas_AC_ct = list(range(0, len(grid.lines_AC_ct)))
    lista_nodos_AC = list(range(0, len(grid.nodes_AC)))
    
    # Binary variables: one per line (used or not)
    line_vars = {}
    
    for line in lista_lineas_AC_ct:
        line_vars[line] = master.addVar(
            vtype=GRB.BINARY,
            name=f"line_used_{line}"
        )

    
    # Objective: minimize total cable length
    investment_cost = 0
    for line in grid.lines_AC_ct:
        l = line.lineNumber
        line_cost = line.Length_km
        investment_cost += line_vars[l] * line_cost
          
    
    # Spanning tree constraint: exactly numNodes-1 connections
    total_connections = sum(line_vars[line] for line in lista_lineas_AC_ct)
    master.addConstr(
        total_connections == len(lista_nodos_AC) - 1,
        name="spanning_tree_connections"
    )
    
  
    # Find sink nodes (nodes with generators) and source nodes (nodes with renewable resources)
    sink_nodes = []
    source_nodes = []
    
    for node in lista_nodos_AC:
        nAC = grid.nodes_AC[node]
        if nAC.connected_gen:  # Node has generator (sink)
            sink_nodes.append(node)
        if nAC.connected_RenSource:  # Node has renewable resources (source)
            source_nodes.append(node)
    
    if not sink_nodes:
        raise ValueError("No generator nodes found!")
      # Constrain connections for source nodes (renewable nodes)
    for node in source_nodes:
        # Count how many lines are connected to this source node
        node_connections = sum(line_vars[line] 
                              for line in lista_lineas_AC_ct
                              if (grid.lines_AC_ct[line].fromNode.nodeNumber == node or 
                                  grid.lines_AC_ct[line].toNode.nodeNumber == node))
        
        # Limit to node.ct_limit
        nAC = grid.nodes_AC[node]
        master.addConstr(
            node_connections <= nAC.ct_limit,
            name=f"source_connections_limit_{node}"
        )
 
    # Flow variables (integer - can carry flow in either direction)
    flow_vars = {}
    node_flow_vars = {}
    
    for line in lista_lineas_AC_ct:
        # Signed flow variable: positive = flow from fromNode to toNode, negative = reverse
        flow_vars[line] = master.addVar(
            vtype=GRB.INTEGER,
            lb=-max_flow,
            ub=max_flow,
            name=f"flow_{line}"
        )
    
    for node in lista_nodos_AC:
        # Net flow out of each node
        node_flow_vars[node] = master.addVar(
            vtype=GRB.INTEGER,
            name=f"node_flow_{node}"
        )
        
        # Calculate net flow out of this node
        net_flow = 0
        
        for line in lista_lineas_AC_ct:
            line_obj = grid.lines_AC_ct[line]
            from_node = line_obj.fromNode.nodeNumber
            to_node = line_obj.toNode.nodeNumber
            
            if from_node == node:
                # Flow leaving this node (positive)
                net_flow += flow_vars[line]
            elif to_node == node:
                # Flow entering this node (negative, so we add it to flow_out)
                net_flow -= flow_vars[line]
        
        # Set the net flow out of this node
        master.addConstr(
            node_flow_vars[node] == net_flow,
            name=f"flow_conservation_{node}"
        )
    
    # Source nodes: net flow out = 1 (supply)
    for node in source_nodes:
        master.addConstr(
            node_flow_vars[node] == 1,
            name=f"source_node_{node}"
        )
    
    # Sink nodes: total net flow out = -num_sources (demand)
    master.addConstr(
        sum(node_flow_vars[node] for node in sink_nodes) == -len(source_nodes),
        name="total_sink_absorption"
    )
    
    # Intermediate nodes: net flow = 0 (conservation)
    for node in lista_nodos_AC:
        if node not in source_nodes and node not in sink_nodes:
            master.addConstr(
                node_flow_vars[node] == 0,
                name=f"intermediate_node_{node}"
            )
    
    # Link flow to investment: can only use lines we invest in
    for line in lista_lineas_AC_ct:
        # Flow must be zero if line not invested
        master.addConstr(
            flow_vars[line] <= max_flow * line_vars[line],
            name=f"flow_investment_link_upper_{line}"
        )
        master.addConstr(
            flow_vars[line] >= -max_flow * line_vars[line],
            name=f"flow_investment_link_lower_{line}"
        )
    





    master.setObjective(investment_cost, GRB.MINIMIZE)
    master.update()
    return master, line_vars, flow_vars, node_flow_vars


def test_master_problem_gurobi(grid, max_flow=None):
    """Simple test for master problem"""
    print("Testing Master Problem...")
    
    # Create and solve master problem
    master, line_vars, flow_vars,node_flow_vars = create_master_problem_gurobi(grid, max_flow)
    
    # Try to solve (should be feasible)
    master.setParam('OutputFlag', 1)  # Show output for debugging
    master.optimize()
    
    if master.status == GRB.OPTIMAL:
        print(f"✓ Master problem is feasible!")
        print(f"  Objective: {master.objVal:.2f}")
        print(f"  Variables: {master.NumVars}")
        print(f"  Constraints: {master.NumConstrs}")
        
        # Count and display investments
        investments = 0
        print("\n=== NETWORK FLOW ANALYSIS ===")
        print("Invested lines and their flows:")
        for line in range(len(grid.lines_AC_ct)):
            if line_vars[line].X > 0.8:
                investments += 1
                line_obj = grid.lines_AC_ct[line]
                flow_value = flow_vars[line].X
                print(f"  Line {line}: {line_obj.fromNode.nodeNumber} -> {line_obj.toNode.nodeNumber}, Flow: {flow_value}")
        
        print("\nNode flows:")
        for node in range(len(grid.nodes_AC)):
            node_flow = node_flow_vars[node].X
            node_type = ""
            if node in [n for n in range(len(grid.nodes_AC)) if grid.nodes_AC[n].connected_RenSource]:
                node_type = " (SOURCE)"
            elif node in [n for n in range(len(grid.nodes_AC)) if grid.nodes_AC[n].connected_gen]:
                node_type = " (SINK)"
            else:
                node_type = " (INTERMEDIATE)"
            print(f"  Node {node}: Net flow = {node_flow}{node_type}")
        
        # Set active configurations
        for line in range(len(grid.lines_AC_ct)):
            ct_line = grid.lines_AC_ct[line]
            if line_vars[line].X > 0.5:
                ct_line.active_config = 0
            else:
                ct_line.active_config = -1
        
        print(f"\n=== SUMMARY ===")
        print(f"  Total investments: {investments}")
        print(f"  Expected (numNodes-1): {len(grid.nodes_AC) - 1}")
        
        # Verify flow conservation
        source_nodes = [n for n in range(len(grid.nodes_AC)) if grid.nodes_AC[n].connected_RenSource]
        sink_nodes = [n for n in range(len(grid.nodes_AC)) if grid.nodes_AC[n].connected_gen]
        total_source_flow = sum(node_flow_vars[node].X for node in source_nodes)
        total_sink_flow = sum(node_flow_vars[node].X for node in sink_nodes)
        print(f"  Total source flow: {total_source_flow}")
        print(f"  Total sink flow: {total_sink_flow}")
        print(f"  Flow conservation check: {total_source_flow + total_sink_flow == 0}")
        
        save_network_svg(grid, name='grid_network_investments')
        return True
    
        
    else:
        print(f"✗ Master problem failed: status {master.status}")
        return False


# Pyomo versions (if available)
if PYOMO_AVAILABLE:
    
    def create_master_problem_pyomo(grid, max_flow=None):
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
        
        # Flow variables (integer - can carry flow in either direction)
        model.line_flow = pyo.Var(model.lines, domain=pyo.Integers, bounds=(-max_flow, max_flow))
        model.node_flow = pyo.Var(model.nodes, domain=pyo.Integers)

        # Objective: minimize total cable length
        def objective_rule(model):
            return sum(model.line_used[line] * grid.lines_AC_ct[line].Length_km 
                      for line in model.lines)
        
        model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
        
        # Spanning tree constraint: exactly numNodes-1 connections
        def spanning_tree_rule(model):
            return sum(model.line_used[line] for line in model.lines) == len(model.nodes) - 1
     
        
        model.spanning_tree = pyo.Constraint(rule=spanning_tree_rule)
        
        # Constrain connections for source nodes (renewable nodes)
        def source_connections_rule(model, node):
            if node in model.source_nodes:
                node_connections = sum(model.line_used[line] 
                                     for line in model.lines
                                     if (grid.lines_AC_ct[line].fromNode.nodeNumber == node or 
                                         grid.lines_AC_ct[line].toNode.nodeNumber == node))
                return node_connections <= grid.nodes_AC[node].ct_limit
            else:
                return pyo.Constraint.Skip
        
        model.source_connections_limit = pyo.Constraint(model.nodes, rule=source_connections_rule)
        
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
       
        model.flow_investment_link = pyo.Constraint(model.lines, rule=flow_investment_rule)
        model.flow_investment_link_2 = pyo.Constraint(model.lines, rule=flow_investment_rule_2)
        
        return model
    
    
    def test_master_problem_pyomo(grid, max_flow=None, solver_name='glpk'):
        """Test master problem using Pyomo with open-source solver"""
        print(f"Testing Master Problem with Pyomo and {solver_name}...")
        
        # Create model
        model = create_master_problem_pyomo(grid, max_flow)
        
        # Create solver
        solver = pyo.SolverFactory(solver_name)
        
        # Solve
        results = solver.solve(model, tee=True)
        
        # Check results
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            print(f"✓ Master problem is optimal!")
            print(f"  Objective: {pyo.value(model.objective):.2f}")
           
            # Count and display investments
            investments = 0
            print("\n=== NETWORK FLOW ANALYSIS ===")
            print("Invested lines and their flows:")
            for line in model.lines:
                if pyo.value(model.line_used[line]) > 0.8:
                    investments += 1
                    line_obj = grid.lines_AC_ct[line]
                    flow_value = pyo.value(model.line_flow[line])
                    print(f"  Line {line}: {line_obj.fromNode.nodeNumber} -> {line_obj.toNode.nodeNumber}, Flow: {flow_value}")
            
            print("\nNode flows:")
            for node in model.nodes:
                node_flow = pyo.value(model.node_flow[node])
                node_type = ""
                if node in model.source_nodes:
                    node_type = " (SOURCE)"
                elif node in model.sink_nodes:
                    node_type = " (SINK)"
                else:
                    node_type = " (INTERMEDIATE)"
                print(f"  Node {node}: Net flow = {node_flow}{node_type}")
            
            # Set active configurations
            for line in model.lines:
                ct_line = grid.lines_AC_ct[line]
                if pyo.value(model.line_used[line]) > 0.5:
                    ct_line.active_config = 0
                else:
                    ct_line.active_config = -1
            
            print(f"\n=== SUMMARY ===")
            print(f"  Total investments: {investments}")
            print(f"  Expected (numNodes-1): {len(grid.nodes_AC) - 1}")
            
            # Verify flow conservation
            total_source_flow = sum(pyo.value(model.node_flow[node]) for node in model.source_nodes)
            total_sink_flow = sum(pyo.value(model.node_flow[node]) for node in model.sink_nodes)
            print(f"  Total source flow: {total_source_flow}")
            print(f"  Total sink flow: {total_sink_flow}")
            print(f"  Flow conservation check: {total_source_flow + total_sink_flow == 0}")
            
            save_network_svg(grid, name='grid_network_investments_pyomo')
            return True
        
        else:
            print(f"✗ Master problem failed: {results.solver.termination_condition}")
            return False
    
    
    def solve_with_different_solvers(grid, max_flow=None):
        """Test with different open-source solvers"""
        solvers_to_try = ['glpk', 'cbc', 'ipopt']
        
        for solver_name in solvers_to_try:
            print(f"\n{'='*50}")
            print(f"Testing with {solver_name.upper()}")
            print(f"{'='*50}")
            
            try:
                success = test_master_problem_pyomo(grid, max_flow, solver_name)
                if success:
                    print(f"✓ {solver_name.upper()} solved successfully!")
                else:
                    print(f"✗ {solver_name.upper()} failed")
            except Exception as e:
                print(f"✗ {solver_name.upper()} error: {e}")
        
        print(f"\n{'='*50}")
        print("Solver comparison complete")
        print(f"{'='*50}")


# Convenience functions that work with both solvers
def create_master_problem(grid, max_flow=None, solver='gurobi'):
    """Create master problem with specified solver"""
    if solver.lower() == 'gurobi':
        return create_master_problem_gurobi(grid, max_flow)
    elif solver.lower() == 'pyomo' and PYOMO_AVAILABLE:
        return create_master_problem_pyomo(grid, max_flow)
    else:
        raise ValueError(f"Solver '{solver}' not available")


def test_master_problem(grid, max_flow=None, solver='gurobi'):
    """Test master problem with specified solver"""
    if solver.lower() == 'gurobi':
        return test_master_problem_gurobi(grid, max_flow)
    elif solver.lower() == 'pyomo' and PYOMO_AVAILABLE:
        return test_master_problem_pyomo(grid, max_flow)
    else:
        raise ValueError(f"Solver '{solver}' not available")
    