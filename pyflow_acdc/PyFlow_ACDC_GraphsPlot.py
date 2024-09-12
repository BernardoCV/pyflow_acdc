import networkx as nx
import plotly.graph_objs as go
import plotly.io as pio

import itertools
import base64
import numpy as np
import pandas as pd
from plotly import data
from plotly.subplots import make_subplots


from PyFlow_ACDC_TS import export_TS


def plot_Graph(Grid,image_path=None,dec=3,InPu=True,grid_names=None):
    G = Grid.Graph_toPlot
    
    hover_texts_nodes = {}
    hover_texts_lines = {}
    if InPu == True:
       
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
                conv  = np.round(node.P, decimals=dec)
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
                Pfrom= np.round(line.fromP, decimals=dec)
                Pto = np.round(line.toP, decimals=dec)
                load = max(Pfrom, Pto)*Grid.S_base/line.MW_rating*100
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
                conv  = np.round(node.P*Grid.S_base, decimals=0).astype(int)
                hover_texts_nodes[node] = f"Node: {name}<br>Voltage: {V}kV<br>Converter:.{conv}MW"
                
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
                Loading = np.round(load, decimals=dec).astype(int)
                if np.real(Sfrom) > 0:
                    line_string = f"{fromnode} -> {tonode}"
                else:
                    line_string = f"{fromnode} <- {tonode}"
                hover_texts_lines[line] = f"Line: {name}<br>  {line_string}<br>S from: {Sfrom}MVA<br>S to: {Sto}MVA<br>Loading: {Loading}%"
                
                
            elif line in Grid.lines_DC:
                name= line.name
                fromnode = line.fromNode.name
                tonode = line.toNode.name
                Pfrom= np.round(line.fromP*Grid.S_base, decimals=0).astype(int)
                Pto = np.round(line.toP*Grid.S_base, decimals=0).astype(int)
                load = max(Pfrom, Pto)/line.MW_rating*100
                Loading = np.round(load, decimals=0).astype(int)
                if Pfrom > 0:
                    line_string = f"{fromnode} -> {tonode}"
                else:
                    line_string = f"{fromnode} <- {tonode}"
                hover_texts_lines[line] = f"Line: {name}<br>  {line_string}<br>P from: {Pfrom}MW<br>P to: {Pto}MW<br>Loading: {Loading}%"
                       
                
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

    # Extract node positions
    x_nodes = [pos[k][0] for k in G.nodes()]
    y_nodes = [pos[k][1] for k in G.nodes()]
    
    # Define a color palette for the subgraphs
    color_palette = itertools.cycle([
    'red', 'blue', 'green', 'purple', 'orange', 
    'cyan', 'magenta', 'pink', 'brown', 'gray', 
    'black', 'maroon', 'lime', 'navy', 'teal',
    'violet', 'indigo', 'turquoise', 'beige', 'coral', 'salmon', 'olive'
])
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
                    line=dict(width=1, color=color),
                    visible= True,  # Toggle visibility
                    text=hover_texts_lines[edge[2]['line']],
                    hoverinfo='text'
                )
            )
            traces_edge[trace_num_edge,idx]=1
            trace_num_edge+=1
        # Create node trace for the current subgraph
        x_subgraph_nodes = [pos[node][0] for node in subgraph_nodes]
        y_subgraph_nodes = [pos[node][1] for node in subgraph_nodes]
        
        
        hover_texts_nodes_sub = [hover_texts_nodes[node] for node in subgraph_nodes]
        
        node_traces.append(
            go.Scatter(
                x=x_subgraph_nodes,
                y=y_subgraph_nodes,
                mode='markers',
                marker=dict(
                    size=10,
                    color=color,
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
    updatemenus = [
        dict(
            type="buttons",
            buttons=checkbox_items,
            direction="down",
            pad={"r": 10, "t": 10},
            showactive=True,
            x=-0.05,
            xanchor="left",
            y=1.15,
            yanchor="top"
        )
    ]

    layout = go.Layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        updatemenus=updatemenus
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
 
def plot_TS_res(grid,start,end,grid_names=None):
    pio.renderers.default = 'browser'
    # start=np.copy(start)
    # Assuming export_TS is a function that takes these parameters and returns a dictionary with a key 'TS curtailment'
    case_res = export_TS(grid, start, end, grid_names=grid_names)
    
    # Retrieve the time series data for curtailment
    df = case_res['TS curtailment']

    
    
    columns = df.columns  # Correct way to get DataFrame columns
    time = df.index  # Assuming the DataFrame index is time
    
    
    # Define the layout for subplots
    layout = dict(
        title="Time Series Curtailment",
        hovermode="x",
        # hoversubplots="axis",
        grid=dict(rows=len(columns), columns=1)
    )
    
    
    # Creating subplots
    fig = make_subplots(rows=len(columns), cols=1, shared_xaxes=True, subplot_titles=grid_names)
    
    # Adding traces to the subplots
    for i, col in enumerate(columns):
        fig.add_trace(
            go.Scatter(x=time, y=df[col], name=col,hoverinfo='x+y+name'),
            row=i+1, col=1
        )

    # Update layout
    fig.update_layout(layout)
    
    # Show figure
    fig.show()