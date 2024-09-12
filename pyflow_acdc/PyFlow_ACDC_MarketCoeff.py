# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:46:59 2024

@author: BernardoCastro
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad



def market_data_pd(data):
    df= pd.DataFrame(columns=['time','a', 'b', 'c','price','PGL_min','PGL_max']) 
    
    
    for i in data:
        hour=i['Hour']
        a=i['poly']['a_SC']
        b=i['poly']['b_SC']
        c=i['poly']['c_SC']
        price = i['Market_price']
        PGL_min=i['poly']['P_min']
        PGL_max=i['poly']['P_max']
        new_row = pd.DataFrame({'time':[hour],'a':[a],'b':[b],'c':[c],'price':[price],'PGL_min': [PGL_min],'PGL_max':[PGL_max]})
        df = pd.concat([df, new_row], ignore_index=True)
    df.set_index('time', inplace=True)
    return df   
        
def market_coef_data(df,start,end):
    data = []
    # Loop through the range from 1 to 8760
    for i in range(1, 8761):
        # Create a dictionary for each hour
        hour_data = {
            'Date': i,
            'Hour': i,
            'Sell': pd.DataFrame(columns=['volume', 'price']),
            'Purchase': pd.DataFrame(columns=['volume', 'price']),
            'Load':0,
            'Gen':0
        }
        # Append the dictionary to the list
        data.append(hour_data)
    
    min_hour=8761
    max_hour=1
        
    for index, row in df.iterrows():
        date = row['Date']
        week = row['Week']-1
        week_day = row['Week Day']-1
        hour = row['Hour']
        
        volume = row['Volume']
        if volume==0:
            continue
        price = row['Price']
        Sale_purchase = row['Sale/Purchase']
        
        point= (week*7+week_day)*24+hour
        new_row = pd.DataFrame({'volume': [volume], 'price': [price]})
        min_hour=min(min_hour,point)
        max_hour=max(max_hour,point)
        # Append the new row to the appropriate DataFrame
        data[point-1][Sale_purchase] = pd.concat([data[point-1][Sale_purchase], new_row], ignore_index=True)
        data[point-1]['Date']=date
    
    
    
    min_hour=max(min_hour,start)
    max_hour=min(max_hour,end)
    
    
    small_data=data[min_hour-1: max_hour]
    s=1
    for entry in small_data:
        if not entry['Sell'].empty:
            entry['Gen_data_points']= entry['Sell']['volume'].count()
            entry['max_gen'] = entry['Sell']['volume'].max()
        else:
            entry['max_gen'] = 0
        
        if not entry['Purchase'].empty:
            entry['Dem_data_points']= entry['Purchase']['volume'].count()
            entry['min_demand'] = entry['Purchase']['volume'].min()
        else:
            entry['min_demand'] = 0
        hour= entry['Hour']   
        
        social_cost_curve(data, hour)
        
        
    return small_data


def social_cost_curve(data, hour,limit=0.05,percent_range = .5):
    chosen_entry = data[hour - 1]
    # Extract Sell and Purchase data
    supply_data = chosen_entry['Sell']
    demand_data = chosen_entry['Purchase']
    
    

    # Group by volume and calculate the mean price for each volume
    supply_df = pd.DataFrame(supply_data).groupby('volume').mean().reset_index()
    demand_df = pd.DataFrame(demand_data).groupby('volume').mean().reset_index()
    
    
    
    min_volume = max(min(supply_df['volume']), min(demand_df['volume']))
    max_volume = min(max(supply_df['volume']), max(demand_df['volume']))
    
    supply_0 = {'volume':0,'price':supply_df['price'][0]}
    demand_0 = {'volume':0,'price':demand_data['price'][0]}
    
    supply_0 =  pd.DataFrame([supply_0])
    demand_0 =  pd.DataFrame([demand_0])
    
    # supply_df =  pd.concat([supply_0, supply_df]).reset_index(drop=True)
    # demand_df =  pd.concat([demand_0, demand_df]).reset_index(drop=True)
    
    volumes = np.linspace(min_volume, max_volume, num=1000)

    # Create interpolated functions for supply and demand
    supply_interp = interp1d(supply_df['volume'], supply_df['price'], kind='linear', fill_value="extrapolate")
    demand_interp = interp1d(demand_df['volume'], demand_df['price'], kind='linear', fill_value="extrapolate")

    # Calculate the price difference (supply - demand)
    price_diff = supply_interp(volumes) - demand_interp(volumes)

    # Remove NaN values from price_diff and volumes
    nan_indices = np.isnan(price_diff)
    price_diff = price_diff[~nan_indices]
    volumes = volumes[~nan_indices]

    # Find the equilibrium price and volume
    eq_index = np.argmin(np.abs(price_diff))
    eq_volume = volumes[eq_index]
    eq_price = supply_interp(eq_volume).item()

    cumulative_cost = [0]

    # Integrate between each consecutive pair of volumes and save the cumulative integration
    for i in range(1, len(supply_df)):
        V_prev = supply_df['volume'][i-1]
        V = supply_df['volume'][i]
        cost_of_generation, _ = quad(lambda x: supply_interp(x), V_prev, V)
        cumulative_cost.append(cumulative_cost[-1] + cost_of_generation)

    supply_df['cumulative_cost'] = cumulative_cost
    
    cumulative_benfit=[0]
        
    for i in range(1, len(demand_df)):
        V_prev = demand_df['volume'][i-1]
        V = demand_df['volume'][i]
        benefit_of_consumption, _ = quad(lambda x: demand_interp(x), V_prev, V)
        cumulative_benfit.append(cumulative_benfit[-1] + benefit_of_consumption)
        
    demand_df['cumulative_benefit'] = cumulative_benfit
     
    volume_linspace = np.linspace(min_volume, eq_volume, num=100)

    # Interpolate cumulative cost and benefit over the linspace
    cumulative_cost_interp = interp1d(supply_df['volume'], supply_df['cumulative_cost'], fill_value="extrapolate")
    cumulative_benefit_interp = interp1d(demand_df['volume'], demand_df['cumulative_benefit'], fill_value="extrapolate")
    
    cumulative_cost_values = cumulative_cost_interp(volume_linspace)
    cumulative_benefit_values = cumulative_benefit_interp(volume_linspace)
    
    # Subtract the interpolated cumulative benefit from the interpolated cumulative cost
    net_cost_benefit = cumulative_cost_values - cumulative_benefit_values 
    
    
    
    net_cost_benefit = net_cost_benefit[::-1]
    volume_linspace -= min_volume
    
        
    num_points = len(volume_linspace)
        
    range_start = 0
    range_end = int(percent_range * num_points)
    range_end = min(range_end, num_points)
    
    # Extract the subset of data within the defined range
    volume_linspace_subset = volume_linspace[range_start:range_end]
    net_cost_benefit_subset = net_cost_benefit[range_start:range_end]
    
    
    # Perform the polynomial fit on the subset of data
    coefficients_0 = np.polyfit(volume_linspace_subset, net_cost_benefit_subset, 2)
    a_0 = coefficients_0[0]
    b_0 = coefficients_0[1]
    c_0 = coefficients_0[2]
    
   
    x0 = 0  # replace with the specific x value where Gen = Load
    d = eq_price  # replace with the desired derivative
    
    x_original = (d - b_0) / (2 * a_0)

    # Calculate the shift
    shift =  x_original-x0
  
 
    
    # Generate predicted_SC values based on the fit
    
    predicted_SC = np.polyval(coefficients_0, volume_linspace)
      
    
    shifted_V = volume_linspace -shift
    
    coefficients_SC = np.polyfit(shifted_V, predicted_SC, 2)
    a_SC = coefficients_SC[0]
    b_SC = coefficients_SC[1]
    c_SC = coefficients_SC[2]


        
    if shift<=0:
        # s1hift=200
        # shifted_V=volume_linspace-shift
        s=1
        
      
    predicted_SC = np.polyval(coefficients_SC, shifted_V)
    # predicted_CC = np.polyval(coefficients_CC, shifted_V)
    # predicted_BC = np.polyval(coefficients_BC, shifted_V)
    
    price = 2*a_SC*shifted_V+b_SC
    
    SC_min_curve = -b_SC / (2 * a_SC)
   
    difference_SC = abs(predicted_SC -net_cost_benefit)
     
    
    # #To prevent div by 0 from data startin at Volume=0
    # if (net_cost_benefit==0).any():
    #    net_cost_benefit_0 = net_cost_benefit[:-1]
    #    difference_SC_0 = difference_SC[:-1]
    #    s=1
    #    err_SC = difference_SC_0 /abs(net_cost_benefit_0)
       
    # else:
    err_SC = difference_SC /abs(net_cost_benefit)
        
    valid_indices_SC = np.where(err_SC < limit)[0]
    
    
   
    positive_price = np.where(price > 0)[0]
    first_positive_price = positive_price[0]
    
    
    
    positive_shifted_V_indices = np.where(shifted_V > 0)[0]
    if len(positive_shifted_V_indices) == 0:
      s=1
      first_positive_shifted_V_index=0
    else:
        first_positive_shifted_V_index = positive_shifted_V_indices[0]
  
    
    
    min_valid_index_SC = min(first_positive_shifted_V_index, first_positive_price)
    max_valid_index_SC = np.max(valid_indices_SC)
  
    
    # Filter the arrays based on these indices
    filtered_predicted_SC = predicted_SC[min_valid_index_SC:max_valid_index_SC + 1]
    # filtered_predicted_CC = predicted_CC[min_valid_index_SC:max_valid_index_SC + 1]
    # filtered_predicted_BC = predicted_BC[min_valid_index_SC:max_valid_index_SC + 1]
    
    filtered_shifted_V_SC = shifted_V[min_valid_index_SC:max_valid_index_SC + 1]
    filtered_predicted_price=price[min_valid_index_SC:max_valid_index_SC + 1]
    
    
    if len(filtered_shifted_V_SC) == 0:
      s=1
    
    # P_min= min(filtered_shifted_V_SC[0],0)
    P_min_c= -b_SC/(2*a_SC)

    P_min= P_min_c# filtered_shifted_V_SC[0]
    P_max= filtered_shifted_V_SC[-1]
     
    if P_max<0:
        P_max=0
    cs_dataset = pd.DataFrame({
        'volume':supply_df['volume'] ,
        'cumulative_cost':supply_df['cumulative_cost'],
        }) 
    cd_dataset = pd.DataFrame({
        'volume':demand_df['volume'] ,
        'cumulative_benefit':demand_df['cumulative_benefit'],
        }) 
    poly = {
        'a_SC':a_SC ,
        'b_SC':b_SC,
        'c_SC':c_SC,
        'SC_min_curve':SC_min_curve,
        'P_min':P_min,
        'P_max':P_max,
        'shift':shift,
        'x0':x0,
        
        # 'R':r2
        }
    
    
    sc_dataset = pd.DataFrame({
        'Volume_surplus_values': volume_linspace,
        'cost_of_generation_values': cumulative_cost_values,
        'benefit_of_consumption_values': cumulative_benefit_values,
        'social_cost_values': net_cost_benefit -c_SC       
    })
    
    pred_SC = pd.DataFrame({
        'volume': filtered_shifted_V_SC,
        'predicted_SC':filtered_predicted_SC-c_SC,
        # 'predicted_CC':filtered_predicted_CC-c_CC,
        # 'predicted_BC':filtered_predicted_BC-c_BC,      
        'price':filtered_predicted_price
        })
   
    chosen_entry['SC_dataset'] = sc_dataset
    chosen_entry['prediction_SC']=pred_SC
    
    chosen_entry['cs_dataset'] = cs_dataset
    chosen_entry['cd_dataset'] = cd_dataset
    chosen_entry['poly']=poly
    # Save equilibrium price and volume
    chosen_entry['Market_price'] = eq_price
    chosen_entry['Eq_volume'] = eq_volume

    return a_SC,b_SC,c_SC    

def plot_curves(data, hour):
    chosen_entry = data[hour - 1]
    # Extract Sell, Purchase, and Social Cost data
    sell_data = chosen_entry['Sell']
    purchase_data = chosen_entry['Purchase']
    SC = chosen_entry['SC_dataset']
    cs = chosen_entry['cs_dataset'] 
    cd = chosen_entry['cd_dataset']
    pred_SC= chosen_entry['prediction_SC']
   
    curve_data= chosen_entry['poly']
    # Create a subplot figure with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.1, column_widths=[0.7, 0.3],
                        subplot_titles=('Supply and Demand', 'Social Cost', 'Integrated supply and demand','Price'))
    
    eq_price=chosen_entry['Market_price'] 
    eq_volume= chosen_entry['Eq_volume'] 
    
    # Add scatter trace for Sell data
    if not sell_data.empty:
        fig.add_trace(go.Scatter(x=sell_data['volume'], y=sell_data['price'], mode='lines+markers', name='Supply'), row=1, col=1)
        
    # Add scatter trace for Purchase data
    if not purchase_data.empty:
        fig.add_trace(go.Scatter(x=purchase_data['volume'], y=purchase_data['price'], mode='lines+markers', name='Demand'), row=1, col=1)
        
    if not sell_data.empty and not  purchase_data.empty:
        # Add vertical line at equilibrium volume
        fig.add_trace(go.Scatter(x=[eq_volume, eq_volume], y=[min(min(sell_data['price']), min(purchase_data['price'])),
                                                              max(max(sell_data['price']), max(purchase_data['price']))],
                                 mode='lines', line=dict(color='Green', width=2, dash='dash'), name='Equilibrium Volume'), row=1, col=1)
        
        # Add horizontal line at equilibrium price
        fig.add_trace(go.Scatter(x=[min(min(sell_data['volume']), min(purchase_data['volume'])),
                                    max(max(sell_data['volume']), max(purchase_data['volume']))],
                                 y=[eq_price, eq_price],
                                 mode='lines', line=dict(color='Red', width=2, dash='dash'), name='Equilibrium Price'), row=1, col=1)
    
    # Add scatter trace for Social Cost data
    if not SC.empty:
        fig.add_trace(go.Scatter(x=cs['volume'], y=cs['cumulative_cost'], mode='lines+markers', name='Cost of Gen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=cd['volume'], y=cd['cumulative_benefit'], mode='lines+markers', name='Benefit of consumption'), row=2, col=1)
        fig.add_trace(go.Scatter(x=SC['Volume_surplus_values']-curve_data['shift'], y=SC['social_cost_values'], mode='lines+markers', name='Social Cost'), row=1, col=2)
        fig.add_trace(go.Scatter(x=pred_SC['volume'], y=pred_SC['predicted_SC'], mode='lines+markers', name='predicted_SC'), row=1, col=2)
        # fig.add_trace(go.Scatter(x=pred_SC['volume'], y=pred_SC['predicted_CC'], mode='lines+markers', name='predicted_CC'), row=1, col=2)
        # fig.add_trace(go.Scatter(x=pred_SC['volume'], y=pred_SC['predicted_BC'], mode='lines+markers', name='predicted_BC'), row=1, col=2)
        fig.add_trace(go.Scatter(x=pred_SC['volume'], y=pred_SC['price'], mode='lines+markers', name='predicted_SC'), row=2, col=2)
        
    mp = np.round(chosen_entry['Market_price'], decimals=2)
    # Add vertical line at equilibrium volume and horizontal line at equilibrium price
    
    
    for line_data, line_color, line_name in [ #(eq_volume-curve_data['shift'],'Green',''),            
                                            (curve_data['P_min'] , 'Blue', 'P min'),
                                              (curve_data['P_max'] , 'Blue', 'P max')]:
        fig.add_trace(go.Scatter(x=[line_data, line_data], 
                                  y=[min(pred_SC['price']),
                                    max(pred_SC['price'])],
                                  mode='lines', line=dict(color=line_color, width=2, dash='dash'), name=line_name), row=2, col=2)
    
    
  
    # Update layout with titles and axis labels
    fig.update_layout(        title=f'Supply and Demand for hour {hour}; Market Price: {mp}'           )
    
    # Update xaxis properties
    # fig.update_xaxes(title_text="Volume [MWh]", row=1, col=1)
    # fig.update_xaxes(title_text="P balance = Gen - Load [MW]", row=1, col=2)
    fig.update_xaxes(title_text="Volume [MWh]", row=2, col=1)
    fig.update_xaxes(title_text="P balance = Gen - Load [MW]", row=2, col=2)
    
    # Update yaxis properties
    fig.update_yaxes(title_text="Offer Price [€/MWh]", row=1, col=1)
    fig.update_yaxes(title_text="Social Cost [€]", row=1, col=2)
    fig.update_yaxes(title_text="Integrated [€]", row=2, col=1)
    fig.update_yaxes(title_text="Market Price [€]", row=2, col=2)


    
    # Display the figure in a web browser
    pio.show(fig, renderer='browser')
   
    return fig


    
# plot_curves(data, hour)    
       