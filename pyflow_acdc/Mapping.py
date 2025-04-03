import networkx as nx
import numpy as np
import os
from importlib import resources

import geopandas as gpd
import folium
import branca
from folium.plugins import Draw,MarkerCluster,AntPath
import webbrowser

from .Graph_and_plot import update_hovertexts, create_subgraph_color_dict
from .Classes import Node_AC

def plot_folium(grid, text='inPu', name=None,tiles="CartoDB Positron",polygon=None,ant_path='None',clustering=True,coloring=None):
    # "OpenStreetMap",     "CartoDB Positron"     "Cartodb dark_matter" 
    if name is None:
        name = grid.name
    update_hovertexts(grid, text) 

    # Initialize the map, centred around the North Sea
    m = folium.Map(location=[56, 10], tiles=tiles,zoom_start=5)
    
    
    G = grid.Graph_toPlot  # Assuming this is your main graph object
    subgraph_colors= create_subgraph_color_dict(G)
    subgraph_dict = {} 
    
    # Map each line to its subgraph index
    for idx, subgraph_nodes in enumerate(nx.connected_components(G)):
        for edge in G.subgraph(subgraph_nodes).edges(data=True):
            line = edge[2]['line']
            subgraph_dict[line] = idx
        for node in subgraph_nodes:    
            subgraph_dict[node] = idx
            connected_gens = getattr(node, 'connected_gen', [])  
            connected_renSources = getattr(node, 'connected_RenSource', [])  
            subgraph_dict.update({gen: idx for gen in connected_gens})
            subgraph_dict.update({rs:  idx for rs  in connected_renSources})
    
    # Extract line data (AC and HVDC) into a GeoDataFrame
    def extract_line_data(lines, line_type):
        line_data = []

        if line_type == 'DC': 
            subgraph_dc_counts = {}
            for line_obj in lines:
                subgraph_idx = subgraph_dict.get(line_obj)  # Avoid KeyError
                if subgraph_idx is not None:  # Ensure the line is in subgraph_dict
                    subgraph_dc_counts[subgraph_idx] = subgraph_dc_counts.get(subgraph_idx, 0) + 1
        
        if coloring == 'loss':
            min_loss = min(np.real(line.loss) for line in lines)
            max_loss = max(np.real(line.loss) for line in lines)
            if min_loss == max_loss:
                max_loss += 0.1 
            colormap = branca.colormap.LinearColormap(
                colors=["green", "yellow", "red"],
                vmin=min_loss, 
                vmax=max_loss
                )
        if coloring == 'Efficiency':
           colormap = branca.colormap.LinearColormap(
               colors=["red", "yellow","green"],
               vmin=70, 
               vmax=100
               )
        # test_values = [min_loss, (min_loss + max_loss) / 2, max_loss]
        # for val in test_values:
        #     print(f"Loss: {val}, Color: {colormap(val)}")
        for line_obj in lines:
            subgraph_idx = subgraph_dict.get(line_obj)
            geometry = getattr(line_obj, 'geometry', None)  # Ensure geometry exists
            VL = 'MV' if line_obj.toNode.kV_base < 110 else \
                 'HV' if line_obj.toNode.kV_base < 300 else \
                 'EHV' if line_obj.toNode.kV_base < 500 else \
                 'UHV'
                 
            line_type_indv= line_type    
            
            if line_type_indv == 'DC' and subgraph_dc_counts.get(subgraph_idx, 0) >= 2:
               line_type_indv = 'MTDC'
            
            
            area = line_obj.toNode.PZ if line_obj.toNode.PZ == line_obj.fromNode.PZ else 'ICL'
            ant_v = False
            
            if area == 'ICL' or line_type == 'DC':
                ant_v = True
            if ant_path == 'All' and VL != 'MV':
                ant_v = True
           
            if coloring == 'loss':
                color = colormap(np.real(line_obj.loss))
                # print(f'{np.real(line.loss)} - {color}')
            elif coloring == 'Efficiency':
                loss =np.real(line_obj.loss)
                if line_type== 'DC':
                    power=max(np.abs(line_obj.fromP),np.abs(line_obj.toP))
                else:
                    power =max(np.abs(np.real(line_obj.fromS)),np.abs(np.real(line_obj.toS)))
                eff=(1-loss/power)*100 if power != 0 else 0
                color= colormap(eff)
                # print(f'{eff} - {color}')
            else:
                color=('black' if getattr(line_obj, 'isTf', False)  # Defaults to False if 'isTF' does not exist/
                        else subgraph_colors[VL].get(subgraph_idx, "black") if line_type == 'AC' 
                        else 'darkblue' if line_type_indv == 'MTDC' 
                        else 'royalblue')
            if geometry and not geometry.is_empty:
                line_data.append({
                    "geometry": geometry,
                    "type": line_type_indv,
                    "name": getattr(line_obj, 'name', 'Unknown'),
                    "Direction": line_obj.direction,
                    "ant_viable": ant_v, 
                    "thck": getattr(line_obj, 'np_line', 1),
                    "VL" :VL,
                    "area":area,
                    "tf": getattr(line_obj, 'isTf', False),
                    "hover_text": getattr(line_obj, 'hover_text', 'No info'),
                    "color":color
                })
        
        if lines:  # Using if lines instead of if lines != [] is more pythonic
            return gpd.GeoDataFrame(line_data, geometry="geometry")
        else:
            # Create an empty GeoDataFrame with the expected columns
            return gpd.GeoDataFrame(columns=['geometry', 'type', 'name', 'Direction', 'ant_viable', 
                                           'thck', 'VL', 'area', 'tf', 'hover_text', 'color'], 
                                  geometry='geometry')
   
   
    # Create GeoDataFrames for AC and HVDC lines
    gdf_lines_AC = extract_line_data(grid.lines_AC+grid.lines_AC_tf, "AC")
    if grid.lines_AC_exp != []:
        gdf_lines_AC_exp = extract_line_data(grid.lines_AC_exp, "AC")
    else:
        gdf_lines_AC_exp = gpd.GeoDataFrame(columns=["geometry", "type", "name", "VL", "tf", "hover_text", "color"])

    
    def filter_vl_and_tf(gdf):
    # Filter lines based on Voltage Level (VL)
        AC_mv = gdf[gdf['VL'] == 'MV']    
        AC_hv = gdf[gdf['VL'] == 'HV']
        AC_ehv = gdf[gdf['VL'] == 'EHV']
        AC_uhv = gdf[gdf['VL'] == 'UHV']
    
        # Filter transformer lines (isTf == True)
        AC_tf = gdf[gdf['tf'] == True] if 'tf' in gdf.columns else None

        return AC_mv,AC_hv, AC_ehv, AC_uhv, AC_tf
   
    gdf_lines_AC_mv,gdf_lines_AC_hv, gdf_lines_AC_ehv, gdf_lines_AC_uhv, gdf_lines_AC_tf=filter_vl_and_tf(gdf_lines_AC)
 
    if grid.lines_DC != []:
        gdf_lines_HVDC = extract_line_data(grid.lines_DC, "DC")
    else:
        gdf_lines_HVDC = gpd.GeoDataFrame(columns=["geometry", "type", "name", "VL", "tf", "hover_text", "color"])
        
        
    def extract_conv_data(converters):
        line_data = []
        for conv_obj in converters:
            geometry = getattr(conv_obj, 'geometry', None)  # Ensure geometry exists
            if geometry and not geometry.is_empty:
                line_data.append({
                    "geometry": geometry,
                    "type": "conv",
                    "area":conv_obj.Node_DC.PZ,
                    "ant_viable":False,
                    "thck": getattr(conv_obj, 'NumConvP', 1),
                    "name": getattr(conv_obj, 'name', 'Unknown'),
                    "hover_text": getattr(conv_obj, 'hover_text', 'No info'),
                    "color": 'purple'
                })
        return gpd.GeoDataFrame(line_data, geometry="geometry")
    
    
    if grid.Converters_ACDC != []:
        gdf_conv = extract_conv_data(grid.Converters_ACDC)
    else:
        gdf_conv = gpd.GeoDataFrame(columns=["geometry", "type", "area", "name","hover_text", "color"])
    
    # Extract node data into a GeoDataFrame
    def extract_node_data(nodes):
        
        node_data = []
        for node in nodes:
            subgraph_idx = subgraph_dict.get(node, None)
            geometry = getattr(node, 'geometry', None)
            VL = 'MV' if node.kV_base < 110 else \
                 'HV' if node.kV_base < 300 else \
                 'EHV' if node.kV_base < 500 else \
                 'UHV'
            if geometry and not geometry.is_empty:
                node_data.append({
                    "geometry": geometry,
                    "name": getattr(node, 'name', 'Unknown'),
                    "VL" :VL,
                    "area":node.PZ,
                    "hover_text": getattr(node, 'hover_text', 'No info'),
                    "type": "AC" if isinstance(node, Node_AC) else "DC",
                    "color": subgraph_colors[VL].get(subgraph_idx, "black") if isinstance(node, Node_AC) else "blue"
                })
        return gpd.GeoDataFrame(node_data, geometry="geometry")

    # Create GeoDataFrame for nodes
    gdf_nodes_AC = extract_node_data(grid.nodes_AC)
    
    gdf_nodes_AC_mv,gdf_nodes_AC_hv, gdf_nodes_AC_ehv, gdf_nodes_AC_uhv, _=filter_vl_and_tf(gdf_nodes_AC)
    
    if grid.nodes_DC != []:
        gdf_nodes_DC = extract_node_data(grid.nodes_DC)
    else:
        gdf_nodes_DC = gpd.GeoDataFrame(columns=["geometry", "name", "VL", "area","hover_text","type","color"])
        
        
    def extract_gen_data(gens):
        gen_data = []
        for gen in gens:
            subgraph_idx = subgraph_dict.get(gen, None)
            geometry = getattr(gen, 'geometry', None)
            VL = 'HV' if gen.kV_base < 300 else \
                 'EHV' if gen.kV_base < 500 else \
                 'UHV'
            if geometry and not geometry.is_empty:
                gen_data.append({
                    "geometry": geometry,
                    "name": getattr(gen, 'name', 'Unknown'),
                    "VL" :VL,
                    "area":gen.PZ,
                    "hover_text": getattr(gen, 'hover_text', 'No info'),
                    "type": gen.gen_type,
                    "color": subgraph_colors[VL].get(subgraph_idx, "black") 
                })
        return gpd.GeoDataFrame(gen_data, geometry="geometry")
    
    
    if grid.Generators != []:
        gdf_gens = extract_gen_data(grid.Generators)
    else:
        gdf_gens = gpd.GeoDataFrame(columns=["geometry", "name", "VL", "area","hover_text","type","color"])
    
    
    def extract_renSource_data(renSources):
        gen_data = []
        for rs in renSources:
            subgraph_idx = subgraph_dict.get(rs, None)
            geometry = getattr(rs, 'geometry', None)
            VL = 'HV' if rs.kV_base < 300 else \
                 'EHV' if rs.kV_base < 500 else \
                 'UHV'
            if geometry and not geometry.is_empty:
                gen_data.append({
                    "geometry": geometry,
                    "name": getattr(rs, 'name', 'Unknown'),
                    "VL" :VL,
                    "area":rs.PZ,
                    "hover_text": getattr(rs, 'hover_text', 'No info'),
                    "type": rs.rs_type,
                    "color": subgraph_colors[VL].get(subgraph_idx, "black") 
                })
        return gpd.GeoDataFrame(gen_data, geometry="geometry")
    
    
    if grid.RenSources != []:
        gdf_rsSources = extract_renSource_data(grid.RenSources)
    else:
        gdf_rsSources = gpd.GeoDataFrame(columns=["geometry", "name", "VL", "area","hover_text","type","color"])

    
    # Function to add LineString geometries to the map
    def add_lines(gdf, tech_name,ant):
        
        for _, row in gdf.iterrows():
            
            coords = [(lat, lon) for lon, lat in row.geometry.coords]  # Folium needs (lat, lon) order
            
            if ant and row["ant_viable"]:
                if row["Direction"] == "to":
                    coords = coords[::-1]
                # Add animated AntPath
                AntPath(
                    locations=coords,
                    color=row["color"],
                    weight=3*row["thck"] if row["type"] == "HVDC" else 2*row["thck"],  # HVDC lines slightly thicker
                    opacity=0.8,
                    delay=400,  # Adjust animation speed
                    popup=row["hover_text"]
                ).add_to(tech_name)
    
            else:
        
                folium.PolyLine(
                    coords,
                    color=row["color"],
                    weight=3*row["thck"] if row["type"] == "HVDC" else 2*row["thck"],  # HVDC lines slightly thicker
                    opacity=0.8,
                    popup=row["hover_text"]
                ).add_to(tech_name)
           
    
    
    # Function to add nodes with filtering by type and zone
    def add_nodes(gdf, tech_name):
        for _, row in gdf.iterrows():
            # Check if the node matches the filter criteria (both type and zone)
            folium.CircleMarker(
                location=(row.geometry.y, row.geometry.x),  # (lat, lon)
                radius=2 if row["type"] == "AC" else 3,  # DC nodes slightly larger
                color=row["color"],
                fill=True,
                fill_opacity=0.9,
                popup=row["hover_text"]
            ).add_to(tech_name)
    
    def add_markers(gdf, tech_name):  
        
        if clustering == True:
            cluster = MarkerCluster().add_to(tech_name)  # Add clustering per type
        else:
            cluster = tech_name
        for _, row in gdf.iterrows():
            
            typ = row['type']
            # Ensure valid coordinates (lat, lon)
            if row['geometry'] and not row['geometry'].is_empty:
                lat, lon = row['geometry'].y, row['geometry'].x
                try:
                    # For Python 3.9+
                    with resources.files('pyflow_acdc').joinpath('folium_images').joinpath(f'{typ}.png') as icon_path:
                        icon_path = str(icon_path)
                except Exception:
                    # Fallback for older Python versions
                    icon_path = os.path.join(os.path.dirname(__file__), 'folium_images', f'{typ}.png')
                    
                folium.Marker(
                    location=(lat, lon),  # (lat, lon)
                    popup=row["hover_text"],  # Display name on click
                    icon=folium.CustomIcon(
                        icon_image=icon_path,  
                    )
                ).add_to(cluster)
                
    
    
    mv_AC  = folium.FeatureGroup(name="MVAC Lines <110kV")
    hv_AC  = folium.FeatureGroup(name="HVAC Lines <300kV")
    ehv_AC = folium.FeatureGroup(name="HVAC Lines <500kV")
    uhv_AC = folium.FeatureGroup(name="HVAC Lines")
    hvdc   = folium.FeatureGroup(name="HVDC Lines")
    convs  = folium.FeatureGroup(name="Converters")
    transformers = folium.FeatureGroup(name="Transformers")
    exp_lines = folium.FeatureGroup(name="Exp Lines")
    
    
    if ant_path == 'All' or ant_path == 'Reduced':
        ant = True
    else:
        ant = False
        
    add_lines(gdf_lines_AC_mv, mv_AC,ant)    
    add_lines(gdf_lines_AC_hv, hv_AC,ant)
    add_lines(gdf_lines_AC_ehv, ehv_AC,ant)
    add_lines(gdf_lines_AC_uhv, uhv_AC,ant)
    add_lines(gdf_lines_AC_exp, exp_lines,ant)
    add_lines(gdf_lines_AC_tf, transformers,ant)
    add_lines(gdf_lines_HVDC, hvdc,ant)
    add_lines(gdf_conv, convs, ant)
    
    add_nodes(gdf_nodes_AC_mv, mv_AC)
    add_nodes(gdf_nodes_AC_hv, hv_AC)
    add_nodes(gdf_nodes_AC_ehv, ehv_AC)
    add_nodes(gdf_nodes_AC_uhv, uhv_AC)
    add_nodes(gdf_nodes_DC, hvdc)

    layer_names = [
    "Nuclear", "Hard Coal", "Hydro", "Oil", "Lignite", "Natural Gas",
    "Solid Biomass", "Wind", "Other", "Solar", "Waste", "Biogas", "Geothermal"
    ]
    # Dictionary to store FeatureGroups for each generation type
    layers = {name: folium.FeatureGroup(name=name, show=False) for name in layer_names}
    
    
    # Add filtered layers to map
    mv_AC.add_to(m)  if len(mv_AC._children) > 0 else None
    hv_AC.add_to(m)  if len(hv_AC._children) > 0 else None
    ehv_AC.add_to(m) if len(ehv_AC._children) > 0 else None
    uhv_AC.add_to(m) if len(uhv_AC._children) > 0 else None
    hvdc.add_to(m)   if len(hvdc._children) > 0 else None
    convs.add_to(m)  if len(convs._children) > 0 else None
    transformers.add_to(m) if len(transformers._children) > 0 else None
    exp_lines.add_to(m)    if len(exp_lines._children) > 0 else None
        
    # Split gdf_gens by type and add markers for each type
    for gen_type, subset in gdf_gens.groupby('type'):  # Split by 'type'
        if gen_type in layers:
            add_markers(subset, layers[gen_type])
    
    for gen_type, subset in gdf_rsSources.groupby('type'):  # Split by 'type'
        if gen_type in layers:
            add_markers(subset, layers[gen_type])
    for layer in layers.values():
        if len(layer._children) > 0:  # Check if the layer has children
            layer.add_to(m)

    if polygon is not None:
        folium.GeoJson(
            polygon,
            name="Area to Study",
            style_function=lambda x: {"color": "blue", "weight": 2, "opacity": 0.6},
            show=False
        ).add_to(m)

    Draw(   export=True,  # Allows downloading edited layers
            edit_options={'poly': {'allowIntersection': False}},  # Prevents self-intersecting edits
            draw_options={'polygon': True, 'polyline': True, 'rectangle': True, 'circle': False},
        ).add_to(m)
    # Draw().add_to(m)
    if coloring == 'Efficiency':
        colormap = branca.colormap.LinearColormap(
            colors=["red","yellow", "green"],
            vmin=70, 
            vmax=100
            )
        colormap.caption = "Efficiency Scale"  # Optional: Set a caption for clarity
        m.add_child(colormap)
        
    # Add layer control
    folium.LayerControl().add_to(m)
    # Save and display the map
    map_filename = f"{name}.html"
    # Save and display the map
    m.save(map_filename)  # Open this file in a browser to viewm
    abs_map_filename = os.path.abspath(map_filename)
    
    # Automatically open the map in the default web browser
    webbrowser.open(f"file://{abs_map_filename}")
    return m