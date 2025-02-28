import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio
import logging
import itertools
import base64
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os

from .Classes import Node_AC


__all__ = ['plot_Graph',
    'plot_neighbour_graph',
    'plot_TS_res',
    'save_network_svg']

def update_ACnode_hovertext(node,S_base,text):
    # print(f"Updating hover text for node: {node.name}")
    dec= 2
    if text =='data':
        name = node.name
        typ = node.type
        Load = np.round(node.PLi, decimals=dec)
        x_cord = node.x_coord
        y_cord = node.y_coord
        PZ = node.PZ
        node.hover_text = f"Node: {name}<br>coord: {x_cord},{y_cord}<br>Type: {typ}<br>Load: {Load}<br>Area: {PZ}"

    elif text=='inPu':
            name = node.name
            V = np.round(node.V, decimals=dec)
            theta = np.round(node.theta, decimals=dec)
            PGi= node.PGi+node.PGi_ren*node.curtailment +node.PGi_opt
            Gen =  np.round(PGi, decimals=dec)
            Load = np.round(node.PLi, decimals=dec)
            conv = np.round(node.P_s, decimals=dec)
            PZ = node.PZ
            node.hover_text = f"Node: {name}<br>Voltage: {V}<br>Angle: {theta}<br>Generation: {Gen}<br>Load: {Load}<br>Converters: {conv}<br>PZ: {PZ}"
    else:
            name = node.name
            V = int(np.round(node.V*node.kV_base, decimals=0))
            theta = int(np.round(np.degrees(node.theta), decimals=0))
            PGi= node.PGi+node.PGi_ren*node.curtailment  +node.PGi_opt
            Gen =  int(np.round(PGi*S_base, decimals=0))
            Load = int(np.round(node.PLi*S_base, decimals=0))
            conv = int(np.round(node.P_s*S_base, decimals=0))
            PZ = node.PZ
            node.hover_text = f"Node: {name}<br>Voltage: {V}kV<br>Angle: {theta}°<br>Generation: {Gen}MW<br>Load: {Load}MW<br>Converters: {conv}MW<br>PZ: {PZ}"
                
                    
def update_DCnode_hovertext(node,S_base,text):            
    dec= 2
    if text =='data':
        name = node.name
        typ = node.type
        Load = np.round(node.PLi, decimals=dec)
        x_cord = node.x_coord
        y_cord = node.y_coord
        PZ = node.PZ
        node.hover_text = f"Node: {name}<br>coord: {x_cord},{y_cord}<br>Type: {typ}<br>Load: {Load}<br>Area: {PZ}"

    elif text=='inPu':   
            name = node.name
            V = np.round(node.V, decimals=dec)
            conv  = np.round(node.Pconv, decimals=dec)
            node.hover_text = f"Node: {name}<br>Voltage: {V}<br><br>Converter: {conv}"
           
        
    else:
        name = node.name
        V = np.round(node.V*node.kV_base, decimals=0).astype(int)
        
        if node.ConvInv and node.Nconv >= 0.00001:
            conv  = np.round(node.P*S_base, decimals=0).astype(int)
            nconv = np.round(node.Nconv,decimals=2)
            load = int(np.round(node.conv_loading * S_base / (node.conv_MW * node.Nconv) * 100))
            node.hover_text = f"Node: {name}<br>Voltage: {V}kV<br>Converter:{conv}MW<br>Number Converter: {nconv}<br>Converters loading: {load}%"
        else:
            node.hover_text = f"Node: {name}<br>Voltage: {V}kV"
     
            
            
def update_lineAC_hovertext(line,S_base,text):
    dec=2
    line.direction = 'from' if line.fromS >= 0 else 'to'
    if text =='data':
        name = line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        l = int(line.Length_km)
        z= np.round(line.Z,decimals=5)
        y= np.round(line.Y,decimals=5)
        rating = line.MVA_rating
        rating = np.round(rating,decimals=0)
        Line_tf = 'Transformer' if line.isTf else 'Line'
        line.hover_text = f"{Line_tf}: {name}<br> Z:{z}<br>Y:{y}<br>Length: {l}km<br>Rating: {rating}MVA"

    elif text=='inPu':
        
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Sfrom= np.round(line.fromS, decimals=dec)
        Sto = np.round(line.toS, decimals=dec)
        load = max(np.abs(Sfrom), np.abs(Sto))*S_base/line.MVA_rating*100
        Loading = np.round(load, decimals=dec)
        Line_tf = 'Transformer' if line.isTf else 'Line'
        if np.real(Sfrom) > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        line.hover_text = f"{Line_tf}: {name}<br> {line_string}<br>S from: {Sfrom}<br>S to: {Sto}<br>Loading: {Loading}%"
    else:
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Sfrom= np.round(line.fromS*S_base, decimals=0)
        Sto = np.round(line.toS*S_base, decimals=0)
        load = max(np.abs(line.fromS), np.abs(line.toS))*S_base/line.MVA_rating*100
        Loading = np.round(load, decimals=0).astype(int)
        Line_tf = 'Transformer' if line.isTf else 'Line'
        if np.real(Sfrom) > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        line.hover_text = f"{Line_tf}: {name}<br>  {line_string}<br>S from: {Sfrom}MVA<br>S to: {Sto}MVA<br>Loading: {Loading}%"
              
def update_lineDC_hovertext(line,S_base,text):            
    dec=2
    line.direction = 'from' if line.fromP >= 0 else 'to'
    if text =='data':
        name = line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        
        r= np.round(line.R,decimals=5)
        l = int(line.Length_km)
        rating = line.MW_rating
        rating = np.round(rating,decimals=0)
        line.hover_text = f"Line: {name}<br> R:{r}<br>Length:{l}km<br>Rating: {rating}MW"

    elif text=='inPu':
     
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Pfrom= np.round(line.fromP, decimals=dec)
        Pto = np.round(line.toP, decimals=dec)
        np_line = np.round(line.np_line, decimals=1)
        if np_line == 0:
            load = 0
        else:
            load = max(np.abs(Pfrom), np.abs(Pto))*S_base/(line.MW_rating*line.np_line)*100
        Loading = np.round(load, decimals=dec)
        if Pfrom > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        line.hover_text = f"Line: {name}<br>  {line_string}<br>P from: {Pfrom}<br>P to: {Pto}<br>Loading: {Loading}%"
            
    else:
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Pfrom= np.round(line.fromP*S_base, decimals=0).astype(int)
        Pto = np.round(line.toP*S_base, decimals=0).astype(int)
        np_line = np.round(line.np_line, decimals=1)
        if np_line == 0:
            load = 0
        else:
            load = max(np.abs(Pfrom), np.abs(Pto))/(line.MW_rating*line.np_line)*100
        Loading = np.round(load, decimals=0).astype(int)
        
        if Pfrom > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        line.hover_text = f"Line: {name}<br>  {line_string}<br>P from: {Pfrom}MW<br>P to: {Pto}MW<br>Loading: {Loading}%<br>Number Lines: {np_line}"



def update_lineACexp_hovertext(line,S_base,text):        
    dec=2
    line.direction = 'from' if line.fromS >= 0 else 'to'
    if text =='data':
        name = line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        l = int(line.Length_km)
        z= np.round(line.Z,decimals=5)
        y= np.round(line.Y,decimals=5)
        rating = line.MVA_rating
        rating = np.round(rating,decimals=0)
        Line_tf = 'Transformer' if line.isTf else 'Line'
        line.hover_text = f"{Line_tf}: {name}<br> Z:{z}<br>Y:{y}<br>Length: {l}km<br>Rating: {rating}MVA"

    elif text=='inPu':
        
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Sfrom= np.round(line.fromS, decimals=dec)
        Sto = np.round(line.toS, decimals=dec)
        load = max(np.abs(Sfrom), np.abs(Sto))*S_base/line.MVA_rating*100
        np_line = np.round(line.np_line, decimals=1)
        if np_line == 0:
            load = 0
        else:
            load = max(np.abs(line.fromS), np.abs(line.toS))*S_base/(line.MVA_rating*line.np_line)*100
        Loading = np.round(load, decimals=0).astype(int)    
        if np.real(Sfrom) > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        Line_tf = 'Transformer' if line.isTf else 'Line'
        line.hover_text = f"{Line_tf}: {name}<br> {line_string}<br>S from: {Sfrom}<br>S to: {Sto}<br>Loading: {Loading}%<br>Lines: {np_line}"
    else:
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Sfrom= np.round(line.fromS*S_base, decimals=0)
        Sto = np.round(line.toS*S_base, decimals=0)
        np_line = np.round(line.np_line, decimals=1)
        if np_line == 0:
            load = 0
        else:
            load = max(np.abs(line.fromS), np.abs(line.toS))*S_base/(line.MVA_rating*line.np_line)*100
        Loading = np.round(load, decimals=0).astype(int)
        if np.real(Sfrom) > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        Line_tf = 'Transformer' if line.isTf else 'Line'
        line.hover_text = f"Line: {name}<br>  {line_string}<br>S from: {Sfrom}MVA<br>S to: {Sto}MVA<br>Loading: {Loading}%<br>Lines: {np_line}"


def update_tf_hovertext(line,S_base,text):            
     dec=2
     line.direction = 'from' if line.fromS >= 0 else 'to'
     if text =='data':
         name = line.name
         fromnode = line.fromNode.name
         tonode = line.toNode.name
         l = int(line.Length_km)
         z= np.round(line.Z,decimals=5)
         y= np.round(line.Y,decimals=5)
         rating = line.MVA_rating
         rating = np.round(rating,decimals=0)
         Line_tf = 'Transformer' if line.isTf else 'Line'
         line.hover_text = f"{Line_tf}: {name}<br> Z:{z}<br>Y:{y}<br>Length: {l}km<br>Rating: {rating}MVA"

     elif text=='inPu':
         
         name= line.name
         fromnode = line.fromNode.name
         tonode = line.toNode.name
         Sfrom= np.round(line.fromS, decimals=dec)
         Sto = np.round(line.toS, decimals=dec)
         load = max(np.abs(Sfrom), np.abs(Sto))*S_base/line.MVA_rating*100
         Loading = np.round(load, decimals=dec)
         if np.real(Sfrom) > 0:
             line_string = f"{fromnode} -> {tonode}"
         else:
             line_string = f"{fromnode} <- {tonode}"
         line.hover_text = f"Transformer: {name}<br> {line_string}<br>S from: {Sfrom}<br>S to: {Sto}<br>Loading: {Loading}%"
     else:
        name= line.name
        fromnode = line.fromNode.name
        tonode = line.toNode.name
        Sfrom= np.round(line.fromS*S_base, decimals=0)
        Sto = np.round(line.toS*S_base, decimals=0)
        load = max(np.abs(line.fromS), np.abs(line.toS))*S_base/line.MVA_rating*100
        Loading = np.round(load, decimals=0).astype(int)
        if np.real(Sfrom) > 0:
            line_string = f"{fromnode} -> {tonode}"
        else:
            line_string = f"{fromnode} <- {tonode}"
        line.hover_text = f"Transformer: {name}<br>  {line_string}<br>S from: {Sfrom}MVA<br>S to: {Sto}MVA<br>Loading: {Loading}%"
  
def update_conv_hovertext(conv,S_base,text):            
     if text =='data':
         name= conv.name
         fromnode = conv.Node_DC.name
         tonode = conv.Node_AC.name
         rating = conv.MVA_max
         rating = np.round(rating,decimals=0)
         conv.hover_text = f"Converter: {name}<br>DC node: {fromnode}<br>AC node: {tonode}<br>Rating: {rating}"    
         
     elif text=='inPu':
         name= conv.name
         fromnode = conv.Node_DC.name
         tonode = conv.Node_AC.name
         Sfrom= np.round(conv.P_DC, decimals=0)
         Sto = np.round(np.sqrt(conv.P_AC**2 + conv.Q_AC**2) * np.sign(conv.P_AC), decimals=0)
         load = max(np.abs(Sfrom), np.abs(Sto))*S_base/conv.MVA_max*100
         Loading = np.round(load, decimals=0).astype(int)
         if np.real(Sfrom) > 0:
             conv_string = f"{fromnode} -> {tonode}"
         else:
             conv_string = f"{fromnode} <- {tonode}"
         conv.hover_text = f"Converter: {name}<br>  {conv_string}<br>P DC: {Sfrom}<br>S AC: {Sto}<br>Loading: {Loading}%"    
         
     else:    
        name= conv.name
        fromnode = conv.Node_DC.name
        tonode = conv.Node_AC.name
        Sfrom= np.round(conv.P_DC*S_base, decimals=0)
        Sto = np.round(np.sqrt(conv.P_AC**2+conv.Q_AC**2)*S_base*(conv.P_AC/np.abs(conv.P_AC)), decimals=0)
        load = max(np.abs(Sfrom), np.abs(Sto))*S_base/conv.MVA_max*100
        Loading = np.round(load, decimals=0).astype(int)
        if np.real(Sfrom) > 0:
            conv_string = f"{fromnode} -> {tonode}"
        else:
            conv_string = f"{fromnode} <- {tonode}"
        conv.hover_text = f"Converter: {name}<br>  {conv_string}<br>P DC: {Sfrom}MVA<br>S AC: {Sto}MVA<br>Loading: {Loading}%"    
        
def update_gen_hovertext(gen,S_base,text):            
     if text =='data':
         name= gen.name
         node = gen.Node_AC
         if gen.Max_S is None:
             rating = gen.Max_pow_gen
         else:
            rating = gen.Max_S*S_base
         rating = np.round(rating,decimals=0)
         
         gen.hover_text = f"Generator: {name}<br>AC node: {node}<br>Rating: {rating}<br>Fuel: {gen.gen_type}"    
         
     elif text =='inPu':
         name= gen.name
         Pto = np.round(gen.PGen, decimals=0)
         Qto = np.round(gen.QGen, decimals=0)
         if gen.Max_S is None:
             rating = gen.Max_pow_gen
         else:
            rating = gen.Max_S
         load = np.sqrt(Pto**2+Qto**2)/rating*100
         Loading = np.round(load, decimals=0).astype(int)
         
         gen.hover_text = f"Generator: {name}<br> P gen: {Pto}<br>Q Gen: {Qto}<br>Loading: {Loading}%"   
     else:
        name= gen.name
        Pto = np.round(gen.PGen*S_base, decimals=0)
        Qto = np.round(gen.QGen*S_base, decimals=0)
        if gen.Max_S is None:
            rating = gen.Max_pow_gen*S_base
        else:
           rating = gen.Max_S*S_base
        load = np.sqrt(Pto**2+Qto**2)/rating*100
        Loading = np.round(load, decimals=0).astype(int)
        
        gen.hover_text = f"Generator: {name}<br> P gen: {Pto*S_base}MW<br>Q Gen: {Qto*S_base}MVAR<br>Loading: {Loading}%"    
        
def update_renSource_hovertext(renSource,S_base,text):            
     if text =='data':
         name= renSource.name
         node = renSource.Node
         rating = renSource.PGi_ren_base
         rating = np.round(rating,decimals=0)
         renSource.hover_text = f"Ren Source: {name}<br>AC node: {node}<br>Rating: {rating}<br>Tech: {renSource.rs_type}"    
         
     elif text=='inPu':
         name= renSource.name
         Pto= np.round(renSource.PGi_ren, decimals=0)
         Curt = np.round((1-renSource.gamma)*100, decimals=0)
         renSource.hover_text = f"Ren Source: {name}<br>  P : {Pto}<br>Curtailment: {Curt}%"    
     else:
         
        name= renSource.name
        Pto= np.round(renSource.PGi_ren*S_base, decimals=0)
        Curt = np.round((1-renSource.gamma)*100, decimals=0)
        renSource.hover_text = f"Ren Source: {name}<br>  P : {Pto}MW<br>Curtailment: {Curt}%"    
        
    
                            
                            
                             
def update_hovertexts(grid,text):
    S_base= grid.S_base        
    with ThreadPoolExecutor() as executor:
        futures = []
        if grid.nodes_AC is not None:
            # Update hover texts for nodes
            for node in grid.nodes_AC:
                futures.append(executor.submit(update_ACnode_hovertext, node, S_base, text))
        if grid.nodes_DC is not None:
            for node in grid.nodes_DC:
                futures.append(executor.submit(update_DCnode_hovertext, node, S_base, text))
        if grid.lines_AC is not None:
            # Update hover texts for lines
            for line in grid.lines_AC:
                futures.append(executor.submit(update_lineAC_hovertext, line, S_base, text))
        if grid.lines_DC is not None:
            for line in grid.lines_DC:
                futures.append(executor.submit(update_lineDC_hovertext, line, S_base, text))
        if grid.lines_AC_exp is not None:    
            for line in grid.lines_AC_exp:
                futures.append(executor.submit(update_lineACexp_hovertext, line, S_base, text))
        if grid.lines_AC_tf is not None:    
            for line in grid.lines_AC_tf:
                futures.append(executor.submit(update_tf_hovertext, line, S_base, text))
        if grid.Converters_ACDC is not None:    
            for conv in grid.Converters_ACDC:
                futures.append(executor.submit(update_conv_hovertext, conv, S_base, text))
        if grid.Generators is not None:    
            for gen in grid.Generators:
                futures.append(executor.submit(update_gen_hovertext, gen, S_base, text))
        if grid.RenSources is not None:    
            for renSource in grid.RenSources:
                futures.append(executor.submit(update_renSource_hovertext, renSource, S_base, text))        

        # Wait for all futures to complete
        for future in futures:
            try:
                future.result()  # This will block until the task is finished
            except Exception as e:
                print(f"Error in thread: {e}")
        
def initialize_positions(Grid):
    """Initialize positions for the grid nodes."""
    return Grid.node_positions if Grid.node_positions is not None else {}

def assign_layout_to_missing_nodes(G, pos):
    """Assign layout to nodes missing positions."""
    missing_nodes = [
        node for node in G.nodes if node not in pos or pos[node][0] is None or pos[node][1] is None]
    if missing_nodes:
        try:
            # Attempt to apply planar layout to missing nodes
            pos_missing = nx.planar_layout(G.subgraph(missing_nodes))
            pos.update(pos_missing)
        except nx.NetworkXException as e:
            logging.warning("Planar layout failed, falling back to Kamada-Kawai layout.")
            # Fall back to Kamada-Kawai layout
            pos_missing = nx.kamada_kawai_layout(G.subgraph(missing_nodes))
            pos.update(pos_missing)
    return pos

def assign_converter_positions(Grid, pos):
    """Assign positions for DC nodes using corresponding AC node positions."""
    if Grid.Converters_ACDC is not None:
        for conv in Grid.Converters_ACDC:
            dc_node = conv.Node_DC
            ac_node = conv.Node_AC
            if ac_node in pos:
                pos[dc_node] = pos[ac_node]
            else:
                logging.warning(f"AC node {ac_node} for converter {conv.name} is missing in positions.")
    return pos

def calculate_positions(G, Grid):
    """Calculate positions for nodes in the graph."""
    # Step 1: Initialize positions
    pos = initialize_positions(Grid)
    
    # Step 2: Assign layout to missing nodes
    pos = assign_layout_to_missing_nodes(G, pos)
    
    # Step 3: Assign positions for converters
    pos = assign_converter_positions(Grid, pos)
    
    return pos


def plot_neighbour_graph(grid,node=None,node_name=None,base_node_size=10, proximity=1):
    G = grid.Graph_toPlot
    if node is not None:
        Gn = nx.ego_graph(G,node,proximity)
    elif node_name is not None:
        node= next((node for node in grid.nodes_AC if node.name == node_name), None)
        Gn = nx.ego_graph(G,node,proximity)
    if node is None: 
        print('Node name provided not found')
        return
    plot_Graph(grid,base_node_size=base_node_size,G=Gn)

        
def plot_Graph(Grid,image_path=None,dec=3,text='inPu',grid_names=None,base_node_size=10,G=None):
    
    if G is None:
        G = Grid.Graph_toPlot
    
    update_hovertexts(Grid, text) 
    
    # Initialize pos with node_positions if provided, else empty dict
    pos = calculate_positions(G, Grid)
 
    lines_ac = Grid.lines_AC if Grid.lines_AC is not None else []
    lines_ac_exp = Grid.lines_AC_exp if Grid.lines_AC_exp is not None else []
    lines_dc = Grid.lines_DC if Grid.lines_DC is not None else []
    nodes_DC = Grid.nodes_DC if Grid.nodes_DC is not None else []
    lines_dc_set = set(lines_dc)
    lines_ac_exp_set = set(lines_ac_exp)


    pio.renderers.default = 'browser'
    # Define a color palette for the subgraphs
    color_palette = itertools.cycle([
    'red', 'blue', 'green', 'purple', 'orange', 
    'cyan', 'magenta', 'brown', 'gray', 
    'black', 'lime', 'navy', 'teal',
    'violet', 'indigo', 'turquoise', 'beige', 'coral', 'salmon', 'olive'])
    # 
    # Find connected components (subgraphs)
    connected_components = list(nx.connected_components(G))
    
    
    pos_cache = pos
    node_traces_data = []
    edge_traces_data = []
    mnode_x_data = []
    mnode_y_data = []
    mnode_txt_data = []

    # Create traces for each subgraph with a unique color
    edge_traces = []
    node_traces = []
    mnode_trace = []
    
    
    for idx, subgraph_nodes in enumerate(connected_components):
        color = next(color_palette)
        
        # Create edge trace for the current subgraph
        for edge in G.subgraph(subgraph_nodes).edges(data=True):
            line = edge[2]['line']
            
            # Skip lines with np_line == 0
            if (line in lines_dc_set and line.np_line == 0) or (line in lines_ac_exp_set and line.np_line == 0):
                continue  # Skip plotting for lines where np_line == 0

            # Set line width based on line type
            if line in lines_dc_set:
                line_width = line.np_line
            elif line in lines_ac_exp_set:
                line_width = line.np_line
            else:
                line_width = 1
            
            # Cache positions to avoid repeated access
            x0, y0 = pos_cache[edge[0]]
            x1, y1 = pos_cache[edge[1]]
            
            # Collect midpoint data for marker
            mnode_x_data.append((x0 + x1) / 2)
            mnode_y_data.append((y0 + y1) / 2)
            mnode_txt_data.append(line.hover_text)
            
            # Append edge trace data
            edge_traces_data.append((x0, y0, x1, y1, line_width, color))
        
        # Process nodes for the current subgraph
        x_subgraph_nodes = []
        y_subgraph_nodes = []
        hover_texts_nodes_sub = []
        node_sizes = []
        node_opacities = []
        
        for node in subgraph_nodes:
            x_subgraph_nodes.append(pos_cache[node][0])
            y_subgraph_nodes.append(pos_cache[node][1])
            
            # Adjust for DC nodes
            
            if node in nodes_DC:
              if Grid.TEP_run:  
                node_size = max(base_node_size * (node.Nconv - node.Nconv_i) + base_node_size, base_node_size)
                node_opacity = min(node.Nconv, 1.0) if node.ConvInv else 1.0
            else:
                node_size = base_node_size
                node_opacity = 1.0
            
            hover_texts_nodes_sub.append(node.hover_text)
            node_sizes.append(node_size)
            node_opacities.append(node_opacity)
        
        # Collect node trace data
        node_traces_data.append((x_subgraph_nodes, y_subgraph_nodes, node_sizes, node_opacities, hover_texts_nodes_sub, color))

    
    # After the loops, create all traces in bulk
    # Edge Traces
    for (x0, y0, x1, y1, line_width, color) in edge_traces_data:
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=line_width, color=color),
            visible=True,
            text="hover_text_placeholder",  # Replace with actual hover text
            hoverinfo='text'
        ))

    # Node Traces
    for (x_subgraph_nodes, y_subgraph_nodes, node_sizes, node_opacities, hover_texts_nodes_sub, color) in node_traces_data:
        node_traces.append(go.Scatter(
            x=x_subgraph_nodes,
            y=y_subgraph_nodes,
            mode='markers',
            marker=dict(
                size=node_sizes,
                color=color,
                opacity=node_opacities,
                line=dict(width=2)
            ),
            text=hover_texts_nodes_sub,
            hoverinfo='text',
            visible=True
        ))

    # Create mnode_trace (midpoint node trace) only after processing edges
    mnode_trace = go.Scatter(
        x=mnode_x_data,
        y=mnode_y_data,
        mode="markers",
        showlegend=False,
        hovertemplate="%{hovertext}<extra></extra>",
        visible=True,
        hovertext=mnode_txt_data,
        marker=dict(
            opacity=0,
            size=10,
            color=color
        )
    )
    

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        width=600,  # Set width
        height=600,
        # updatemenus=updatemenus
    )
    
    
    
    # Create figure
    fig = go.Figure(data=edge_traces + node_traces + [mnode_trace], layout=layout)
        
    
    if image_path is not None:
        # Load the image
        with open(image_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode()
 
           # Add background image
        fig.update_layout(
        images=[
            dict(
                source=f'data:image/png;base64,{encoded_image}',
                xref='paper', yref='paper',
                x=0, y=1,
                sizex=1, sizey=1,
                sizing='stretch',
                opacity=0.5,
                layer='below'
                    )
                ]
            )
   

       

    
    # Display plot
    pio.show(fig)
    s=1
    return fig
 
def plot_TS_res(grid, start, end, plotting_choice=None, grid_names=None):
    Plot = {
        'Power Generation by price zone': False,
        'Power Generation by generator': False,
        'Curtailment': False,
        'Market Prices': False,
        'AC line loading': False,
        'DC line loading': False,
        'AC/DC Converters': False,
        'Power Generation by generator area chart': False,
        'Power Generation by price zone area chart': False,
    }

    # If plotting_choice is None, ask user to choose
    if plotting_choice is None:
        print("Please choose a plotting option:")
        print("1: Power Generation by price zone")
        print("2: Power Generation by generator")
        print("3: Curtailment")
        print("4: Market Prices")
        print("5: AC line loading")
        print("6: DC line loading")
        print("7: AC/DC Converters")
        print("8: Power Generation by generator area chart")
        print("9: Power Generation by price zone area chart")
        
        choice = int(input("Enter a number between 1 and 9: "))
        if choice == 1:
            plotting_choice = 'Power Generation by price zone'
        elif choice == 2:
            plotting_choice = 'Power Generation by generator'
        elif choice == 3:
            plotting_choice = 'Curtailment'
        elif choice == 4:
            plotting_choice = 'Market Prices'
        elif choice == 5:
            plotting_choice = 'AC line loading'
        elif choice == 6:
            plotting_choice = 'DC line loading'
        elif choice == 7:
            plotting_choice = 'AC/DC Converters'
        elif choice == 8:
            plotting_choice = 'Power Generation by generator area chart'
        elif choice == 9:
            plotting_choice = 'Power Generation by price zone area chart'
        else:
            print("Invalid choice. Please choose a valid option.")
            return

    # Verify that the choice is valid
    if plotting_choice not in Plot:
        print(f"Invalid plotting option: {plotting_choice}")
        return

    pio.renderers.default = 'browser'
    
    # Retrieve the time series data for curtailment
    
    if plotting_choice == 'Curtailment':
        df = grid.time_series_results['curtailment'].iloc[start:end]*100
    elif plotting_choice in ['Power Generation by generator','Power Generation by generator area chart']:
        df = grid.time_series_results['real_power_opf'].iloc[start:end]*grid.S_base
    elif plotting_choice in ['Power Generation by price zone','Power Generation by price zone area chart'] :
        df = grid.time_series_results['real_power_by_zone'].iloc[start:end] * grid.S_base
    elif plotting_choice == 'Market Prices':
        df = grid.time_series_results['prices_by_zone'].iloc[start:end]
    elif plotting_choice == 'AC line loading':
        df = grid.time_series_results['ac_line_loading'].iloc[start:end]*100
    elif plotting_choice == 'DC line loading':
        df = grid.time_series_results['dc_line_loading'].iloc[start:end]*100
    elif plotting_choice == 'AC/DC Converters':
        df = grid.time_series_results['converter_loading'].iloc[start:end] * grid.S_base

        
        
        
    columns = df.columns  # Correct way to get DataFrame columns
    time = df.index  # Assuming the DataFrame index is time
    
    
    layout = dict(
        title=f"Time Series Plot: {plotting_choice}",  # Set title based on user choice
        hovermode="x"
    )

    cumulative_sum = None
    fig = go.Figure()
    # Check if we need to stack the areas for specific plotting choices
    stack_areas = plotting_choice in ['Power Generation by generator area chart', 'Power Generation by price zone area chart']


    # Adding traces to the subplots
    for col in columns:
        y_values = df[col]

        if stack_areas:
            # print(stack_areas)
            # If stacking, add the current values to the cumulative sum
            if cumulative_sum is None:
                cumulative_sum = y_values.copy()  # Start cumulative sum with the first selected row
                fig.add_trace(
                    go.Scatter(x=time, y=y_values, name=col, hoverinfo='x+y+name', fill='tozeroy')
                )
            else:
                y_values = cumulative_sum + y_values  # Stack current on top of cumulative sum
                cumulative_sum = y_values  # Update cumulative sum
                fig.add_trace(
                    go.Scatter(x=time, y=y_values, name=col, hoverinfo='x+y+name', fill='tonexty')
                )
        else:
            # Plot normally (no stacking)
            fig.add_trace(
                go.Scatter(x=time, y=y_values, name=col, hoverinfo='x+y+name')
            )

    # Update layout
    fig.update_layout(layout)
    
    # Show figure
    fig.show()


def create_subgraph_color_dict(G):
    
    
    
    color_palette_0 = itertools.cycle([
     'violet', 'limegreen',  'salmon',
    'burlywood', 'pink', 'cyan'
    ])
    
    color_palette_1 = itertools.cycle([
     'darkviolet', 'green',  'red',
    'darkoragne', 'hotpink', 'lightseagreen'
    ])
    
    color_palette_2 = itertools.cycle([
         'darkmagenta', 'darkolivegreen',  'brown',
        'darkgoldenrod', 'crimson', 'darkcyan'
    ])

    color_palette_3 = itertools.cycle([
     'orchid', 'lightgreen',  'navajowhite',
    'tan', 'lightpink', 'paleturquoise'
    ])
    
    # Get connected components (subgraphs) of the graph G
    connected_components = list(nx.connected_components(G))
    subgraph_color_dict = {'MV':{},'HV': {}, 'EHV': {}, 'UHV': {}}
    
    # Loop through the connected components and assign colors
    for idx, subgraph_nodes in enumerate(connected_components):
        subgraph_color_dict['MV'][idx] = next(color_palette_0) 
        subgraph_color_dict['HV'][idx] = next(color_palette_1) 
        subgraph_color_dict['EHV'][idx] = next(color_palette_2) 
        subgraph_color_dict['UHV'][idx] = next(color_palette_3) 
    return subgraph_color_dict





def save_network_svg(grid, name='grid_network', width=1000, height=800):
    """Save the network as SVG file"""
    try:
        import svgwrite

        print(f"Current working directory: {os.getcwd()}")
        print(f"Will save as: {os.path.abspath(f'{name}.svg')}")
        # Create SVG drawing
        dwg = svgwrite.Drawing(f"{name}.svg", size=(f'{width}px', f'{height}px'), profile='tiny')
        
        # Get all geometries and their bounds
        all_bounds = []
        
        # Add lines
        for line in grid.lines_AC + grid.lines_AC_tf + grid.lines_DC:
            if hasattr(line, 'geometry') and line.geometry:
                all_bounds.append(line.geometry.bounds)
                
        # Add nodes
        for node in grid.nodes_AC + grid.nodes_DC:
            if hasattr(node, 'geometry') and node.geometry:
                all_bounds.append(node.geometry.bounds)
                
        # Add generators and renewable sources
        for gen in grid.Generators + grid.RenSources:
            if hasattr(gen, 'geometry') and gen.geometry:
                all_bounds.append(gen.geometry.bounds)
        
        # Calculate overall bounds
        if all_bounds:
            minx = min(bound[0] for bound in all_bounds)
            miny = min(bound[1] for bound in all_bounds)
            maxx = max(bound[2] for bound in all_bounds)
            maxy = max(bound[3] for bound in all_bounds)
        else:
            print("No geometries found to plot")
            return

        # Calculate scaling factors
        padding = 50  # pixels of padding
        scale_x = (width - 2*padding) / (maxx - minx)
        scale_y = (height - 2*padding) / (maxy - miny)
        scale = min(scale_x, scale_y)
        
        def transform_coords(x, y):
            """Transform coordinates to SVG space"""
            return (
                padding + (x - minx) * scale,
                height - (padding + (y - miny) * scale)  # Flip Y axis
            )
        
        # Draw AC lines
        for line in grid.lines_AC + grid.lines_AC_tf:
            if hasattr(line, 'geometry') and line.geometry:
                coords = list(line.geometry.coords)
                path_data = "M "
                for x, y in coords:
                    svg_x, svg_y = transform_coords(x, y)
                    path_data += f"{svg_x},{svg_y} L "
                path_data = path_data[:-2]  # Remove last "L "
                
                color = "black" if getattr(line, 'isTf', False) else "red"
                dwg.add(dwg.path(d=path_data, stroke=color, stroke_width=2, fill='none'))
        
        # Draw DC lines
        for line in grid.lines_DC:
            if hasattr(line, 'geometry') and line.geometry:
                coords = list(line.geometry.coords)
                path_data = "M "
                for x, y in coords:
                    svg_x, svg_y = transform_coords(x, y)
                    path_data += f"{svg_x},{svg_y} L "
                path_data = path_data[:-2]
                dwg.add(dwg.path(d=path_data, stroke='blue', stroke_width=2, fill='none'))
        
        # Draw nodes
        for node in grid.nodes_AC + grid.nodes_DC:
            if hasattr(node, 'geometry') and node.geometry:
                x, y = node.geometry.x, node.geometry.y
                svg_x, svg_y = transform_coords(x, y)
                color = "black" if isinstance(node, Node_AC) else "purple"
                dwg.add(dwg.circle(center=(svg_x, svg_y), r=3, 
                                 fill=color, stroke=color))
        
                
        # Save the SVG file
        dwg.save()
        print(f"Network saved as {name}.svg")
        
    except ImportError as e:
        print(f"Could not save SVG: {e}. Please install svgwrite package.")


    return 


    
