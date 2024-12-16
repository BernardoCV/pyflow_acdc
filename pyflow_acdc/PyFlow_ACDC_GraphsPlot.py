import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

import itertools
import base64
import numpy as np
import pandas as pd
from plotly import data
from plotly.subplots import make_subplots



def plot_Graph(Grid,image_path=None,dec=3,InPu=True,grid_names=None,base_node_size=10):
    G = Grid.Graph_toPlot
    
    hover_texts_nodes = {}
    hover_texts_lines = {}
    if InPu:
       
        for node in G.nodes():
            if node in Grid.nodes_AC:
                name = node.name
                V = np.round(node.V, decimals=dec)
                theta = np.round(node.theta, decimals=dec)
                PGi= node.PGi+node.PGi_ren*node.curtailment +node.PGi_opt
                Gen =  np.round(PGi, decimals=dec)
                Load = np.round(node.PLi, decimals=dec)
                conv = np.round(node.P_s, decimals=dec)
                hover_texts_nodes[node] = f"Node: {name}<br>Voltage: {V}<br>Angle: {theta}<br>Generation: {Gen}<br>Load: {Load}<br>Converters: {conv}"
                
                
            elif node in Grid.nodes_DC:
                name = node.name
                V = np.round(node.V, decimals=dec)
                conv  = np.round(node.Pconv, decimals=dec)
                hover_texts_nodes[node] = f"Node: {name}<br>Voltage: {V}<br><br>Converter: {conv}"
            else:
                hover_texts_nodes[node] = f"Node: {node}<br>No additional data available"
        for edge in G.edges(data=True):
            line=edge[2]['line']
            s=1
            if line in Grid.lines_AC:
                name= line.name
                fromnode = line.fromNode.name
                tonode = line.toNode.name
                Sfrom= np.round(line.fromS, decimals=dec)
                Sto = np.round(line.toS, decimals=dec)
                load = max(np.abs(Sfrom), np.abs(Sto))*Grid.S_base/line.MVA_rating*100
                Loading = np.round(load, decimals=dec)
                if np.real(Sfrom) > 0:
                    line_string = f"{fromnode} -> {tonode}"
                else:
                    line_string = f"{fromnode} <- {tonode}"
                hover_texts_lines[line] = f"Line: {name}<br> {line_string}<br>S from: {Sfrom}<br>S to: {Sto}<br>Loading: {Loading}%"
                
                
            elif line in Grid.lines_DC:
                name= line.name
                fromnode = line.fromNode.name
                tonode = line.toNode.name
                Pfrom= np.round(line.fromP/line.np_line, decimals=dec)
                Pto = np.round(line.toP/line.np_line, decimals=dec)
                
                load = max(np.abs(Pfrom), np.abs(Pto))*Grid.S_base/(line.MW_rating)*100
                Loading = np.round(load, decimals=dec)
                if Pfrom > 0:
                    line_string = f"{fromnode} -> {tonode}"
                else:
                    line_string = f"{fromnode} <- {tonode}"
                hover_texts_lines[line] = f"Line: {name}<br>  {line_string}<br>P from: {Pfrom}<br>P to: {Pto}<br>Loading: {Loading}%"
                
            else:
                hover_texts_nodes[node] = f"Node: {node}<br>No additional data available"

    else:
        
        for node in G.nodes():
            if node in Grid.nodes_AC:
                name = node.name
                V = np.round(node.V*node.kV_base, decimals=0).astype(int)
                theta = np.round(np.degrees(node.theta), decimals=0).astype(int)
                PGi= node.PGi+node.PGi_ren*node.curtailment  +node.PGi_opt
                Gen =  np.round(PGi*Grid.S_base, decimals=0).astype(int)
                Load = np.round(node.PLi*Grid.S_base, decimals=0).astype(int)
                conv = np.round(node.P_s*Grid.S_base, decimals=0).astype(int)
                hover_texts_nodes[node] = f"Node: {name}<br>Voltage: {V}kV<br>Angle: {theta}°<br>Generation: {Gen}MW<br>Load: {Load}MW<br>Converters: {conv}MW"
                
                
            elif node in Grid.nodes_DC:
                name = node.name
                V = np.round(node.V*node.kV_base, decimals=0).astype(int)
                
                if node.ConvInv and node.Nconv >= 0.00001:
                    conv  = np.round(node.P*Grid.S_base, decimals=0).astype(int)
                    nconv = np.round(node.Nconv,decimals=2)
                    load = np.round(node.conv_loading*Grid.S_base/(node.conv_MW*node.Nconv)*100,decimals = 0).astype(int)
                    hover_texts_nodes[node] = f"Node: {name}<br>Voltage: {V}kV<br>Converter:{conv}MW<br>Number Converter: {nconv}<br>Converters loading: {load}%"
                else:
                    hover_texts_nodes[node] = f"Node: {name}<br>Voltage: {V}kV"
                
            else:
                hover_texts_nodes[node] = f"Node: {node}<br>No additional data available"
                
        for edge in G.edges(data=True):
            line=edge[2]['line']
            
            if line in Grid.lines_AC:
                name= line.name
                fromnode = line.fromNode.name
                tonode = line.toNode.name
                Sfrom= np.round(line.fromS*Grid.S_base, decimals=0)
                Sto = np.round(line.toS*Grid.S_base, decimals=0)
                load = max(np.abs(line.fromS), np.abs(line.toS))*Grid.S_base/line.MVA_rating*100
                Loading = np.round(load, decimals=0).astype(int)
                if np.real(Sfrom) > 0:
                    line_string = f"{fromnode} -> {tonode}"
                else:
                    line_string = f"{fromnode} <- {tonode}"
                hover_texts_lines[line] = f"Line: {name}<br>  {line_string}<br>S from: {Sfrom}MVA<br>S to: {Sto}MVA<br>Loading: {Loading}%"
                
                
            elif line in Grid.lines_DC:
                if line.np_line==0:
                    continue
                name= line.name
                fromnode = line.fromNode.name
                tonode = line.toNode.name
                Pfrom= np.round(line.fromP*Grid.S_base, decimals=0).astype(int)
                Pto = np.round(line.toP*Grid.S_base, decimals=0).astype(int)
                
                
                load = max(Pfrom, Pto)/(line.MW_rating*line.np_line)*100
                Loading = np.round(load, decimals=0).astype(int)
                np_line = np.round(line.np_line, decimals=1)
                if Pfrom > 0:
                    line_string = f"{fromnode} -> {tonode}"
                else:
                    line_string = f"{fromnode} <- {tonode}"
                hover_texts_lines[line] = f"Line: {name}<br>  {line_string}<br>P from: {Pfrom}MW<br>P to: {Pto}MW<br>Loading: {Loading}%<br>Number Lines: {np_line}"
                       
                
    pio.renderers.default = 'browser'
    
    # Initialize pos with node_positions if provided, else empty dict
    pos = Grid.node_positions if Grid.node_positions is not None else {}
    s=1
    # Identify nodes without positions and apply spring layout to them
    missing_nodes = [node for node in G.nodes if node not in pos or pos[node][0] is None or pos[node][1] is None]
    if missing_nodes:
        try:
            # Attempt to apply planar layout to missing nodes
            pos_missing = nx.planar_layout(G.subgraph(missing_nodes))
            pos.update(pos_missing)
        except nx.NetworkXException:
            # If planar layout fails, fall back to Kamada-Kawai layout
            pos_missing = nx.kamada_kawai_layout(G.subgraph(missing_nodes))
            pos.update(pos_missing)
            
    if Grid.Converters_ACDC is not None:
        for conv in Grid.Converters_ACDC:
            dc_node= conv.Node_DC
            ac_node= conv.Node_AC
            # print(f'{ac_node.name}--{conv.name}--{dc_node.name}')
            pos[dc_node] = pos[ac_node]
    # Extract node positions
    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]
    
    # Define a color palette for the subgraphs
    color_palette = itertools.cycle([
    'red', 'blue', 'green', 'purple', 'orange', 
    'cyan', 'magenta', 'brown', 'gray', 
    'black', 'lime', 'navy', 'teal',
    'violet', 'indigo', 'turquoise', 'beige', 'coral', 'salmon', 'olive'])
    # 
    # Find connected components (subgraphs)
    connected_components = list(nx.connected_components(G))
    
    # Create traces for each subgraph with a unique color
    edge_traces = []
    node_traces = []
    mnode_trace = []
    traces_edge= np.zeros((Grid.nl_AC+Grid.nl_DC,len(connected_components)))
    trace_num_edge=0
    traces_nodes= np.zeros((len(connected_components),len(connected_components)))
    trace_num_nodes=0
    
    mnode_x, mnode_y, mnode_txt = [], [], []

    for idx, subgraph_nodes in enumerate(connected_components):
        color = next(color_palette)
        
        # Create edge trace for the current subgraph
        for edge in G.subgraph(subgraph_nodes).edges(data=True):
            line = edge[2]['line']
            
            # Skip DC lines with np_line == 0
            if Grid.lines_DC is not None:
                if line in Grid.lines_DC :
                    if line.np_line == 0:
                        continue  # Skip plotting for DC lines where np_line == 0
        
                    line_width=line.np_line
                else:
                    line_width=1
            else:
                line_width=1
            
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            
            mnode_x.extend([(x0 + x1)/2]) # assuming values positive/get midpoint
            mnode_y.extend([(y0 + y1)/2]) # assumes positive vals/get midpoint
            mnode_txt.extend([hover_texts_lines[edge[2]['line']]]) # hovertext
            
            
            mnode_trace.append(go.Scatter(x = mnode_x, y = mnode_y, mode = "markers", showlegend = False,
                         hovertemplate = "%{hovertext}<extra></extra>",visible= True,
                         hovertext = mnode_txt, marker=dict(
                             opacity=0,
                              size=10,
                              color=color),
                                 )   
                               )         
            
            edge_traces.append(
                go.Scatter(
                    x=[x0, x1, None], 
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=line_width, color=color),
                    visible= True,  # Toggle visibility
                    text=hover_texts_lines[edge[2]['line']],
                    hoverinfo='text'
                )
            )
            traces_edge[trace_num_edge,idx]=1
            trace_num_edge+=1
        # Create node trace for the current subgraph
        x_subgraph_nodes = []
        y_subgraph_nodes = []
        hover_texts_nodes_sub = []
        node_sizes = []
        node_opacities = []
        
        for node in subgraph_nodes:
            x_subgraph_nodes.append(pos[node][0])
            y_subgraph_nodes.append(pos[node][1])
            
            if Grid.lines_DC is not None:
                if node in Grid.nodes_DC:
                    # Modify size and opacity based on Nconv for DC nodes
                    node_size = max(base_node_size *(node.Nconv-node.Nconv_i)+base_node_size,base_node_size)
                    if node.ConvInv:
                        node_opacity = min(node.Nconv, 1.0)  # Example: adjust opacity based on Nconv
                        hover_texts_nodes_sub.append(hover_texts_nodes[node])
                    else:
                        node_opacity = 1.0
                        hover_texts_nodes_sub.append(hover_texts_nodes[node])
                else:
                    # Default size and opacity for non-DC nodes (e.g., AC nodes)
                    node_size = base_node_size
                    node_opacity = 1.0
                    hover_texts_nodes_sub.append(hover_texts_nodes[node])
            else:
                # Default size and opacity for non-DC nodes (e.g., AC nodes)
                node_size = base_node_size
                node_opacity = 1.0
                hover_texts_nodes_sub.append(hover_texts_nodes[node])
            node_sizes.append(node_size)
            node_opacities.append(node_opacity)

        # hover_texts_nodes_sub = [hover_texts_nodes[node] for node in subgraph_nodes]
        
        node_traces.append(
            go.Scatter(
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
                visible= True  # Toggle visibility
             )
        )
        traces_nodes[trace_num_nodes,idx]=1
        trace_num_nodes+=1

    trace_TF=np.vstack((traces_edge,traces_nodes,traces_edge))
    # Create layout with checkbox
    checkbox_items = []
    visibility_all = [True] * len(edge_traces + node_traces)
    visibility_sub={}
   

    checkbox_items.append(
        dict(label="Toggle All",
             method='update',
             args=[{"visible":visibility_all},  # Toggle visibility
                   {"title": "Toggle All Subgraphs"}])
    )

    for idx, subgraph_nodes in enumerate(connected_components):
        column = trace_TF[:, idx]
        visibility_sub[idx] = column.astype(bool).tolist()
     
        if grid_names is not None:
           try: 
               label_str= f'{grid_names[idx]}'
           except: 
               label_str=f'Subgraph {idx+1}'
        else: 
            label_str=f'Subgraph {idx+1}'
        
        checkbox_items.append(
            dict(label=label_str,               
                method='update',
                args=[
                    {"visible": visibility_sub[idx]},
                    {"title": label_str + ' visibility'}
                ]
            )
        )

    

    # Create layout with updatemenus for subgraph checkboxes
    # updatemenus = [
    #     dict(
    #         type="buttons",
    #         buttons=checkbox_items,
    #         direction="down",
    #         pad={"r": 10, "t": 10},
    #         showactive=True,
    #         x=-0.05,
    #         xanchor="left",
    #         y=1.15,
    #         yanchor="top"
    #     )
    # ]

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        # updatemenus=updatemenus
    )
    # Create figure
    fig = go.Figure(data=edge_traces + node_traces+mnode_trace, layout=layout)
    
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
        print("8: Power Generation by generator (area chart)")
        print("9: Power Generation by price zone (area chart)")
        
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