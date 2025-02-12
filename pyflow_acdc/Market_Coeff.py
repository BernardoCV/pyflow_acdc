# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 09:46:59 2024

@author: BernardoCastro
"""

import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from scipy.optimize import minimize


def price_zone_data_pd(data,save_csv=None):
    df= pd.DataFrame(columns=['time','a_BC', 'b_BC', 'c_BC','a_CG', 'b_CG', 'c_CG','price','volume','PGL_min','PGL_max']) 
    
    
    for i in data:
        hour=i['Hour']
        a_BC=i['poly']['a_BC']
        b_BC=i['poly']['b_BC']
        c_BC=i['poly']['c_BC']
        a_CG=i['poly']['a_CG']
        b_CG=i['poly']['b_CG']
        c_CG=i['poly']['c_CG']
        price = i['Market_price']
        volume = i['Volume_eq']
        PGL_min=i['poly']['P_min']
        PGL_max=i['poly']['P_max']
        new_row = pd.DataFrame({'time':[hour],'a_BC':[a_BC],'b_BC':[b_BC],'c_BC':[c_BC],'a_CG':[a_CG],'b_CG':[b_CG],'c_CG':[c_CG],'price':[price],'volume':[volume],'PGL_min': [PGL_min],'PGL_max':[PGL_max]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    
    df.set_index('time', inplace=True)
    if save_csv is not None:
        df.to_csv(save_csv, index=True)
    return df   
  
def is_leap_year(year):
    """Check if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
      
def price_zone_coef_data(df,start,end,increase_eq_price=50):
    
    first_date = pd.to_datetime(df.iloc[0]['Date'], format='%d/%m/%Y')
    year = first_date.year
    is_leap = is_leap_year(year)
    total_hours = 8784 if is_leap else 8760
   
    data = [
    {
        'Date': None,  # Placeholder for the date
        'Hour': i,
        'Sell': pd.DataFrame(columns=['volume', 'price']),
        'Purchase': pd.DataFrame(columns=['volume', 'price']),
        'Load': 0,
        'Gen': 0
    }
    for i in range(1, total_hours + 1)
    ]
    
    min_hour= total_hours
    max_hour=1
    hour_prev = df.iloc[0]['Hour']
    date_prev = df.iloc[0]['Date']
    summer_time_offset = 0
    dst_active = False
    data_storage = {'Sell': [[] for _ in range(total_hours)], 'Purchase': [[] for _ in range(total_hours)], 'Dates': [None] * total_hours}
    t1= time.time()    
    
    for row in df.itertuples(index=False):
        
        # print(row)
        date = row.Date
        hour = row.Hour
        
        # print(f'{date}----{dst_active}')
        # Detect DST changes and handle "B" hour (e.g., "3B")
        if "B" not in str(hour):
            try:
                if date == date_prev and abs(int(hour) - int(hour_prev)) > 1 and not dst_active:
                    # Spring forward (e.g., 2 -> 4), activate summer time offset
                    
                    summer_time_offset = -1
                    hour_prev = int(hour) - 1  # Ensure hour is an integer for calculations
                    # print(hour)
                    dst_active = True
            except ValueError:
                # Handle the case where the conversion to int fails
                print(f'Subtraction not possible at {hour}')
        elif "B" in str(hour):
            # Autumn fallback, remove summer time offset
            summer_time_offset = 0
            hour = int(hour.replace("B", ""))  # Convert "3B" to 3, or any "XB" to X
            dst_active = False
            
        volume = row.Volume
        if volume==0:
            continue
        price = row.Price
        Sale_purchase = row._6
        
        
        date_obj = pd.to_datetime(date, format='%d/%m/%Y')
        day_of_year = date_obj.dayofyear
        # print(date)
        # print(hour)
        hour_of_year = (day_of_year - 1) * 24 + int(hour) + summer_time_offset
        # print(hour_of_year)
    
        # point= (week*7+week_day)*24+hour+24
        new_row = pd.DataFrame({'volume': [volume], 'price': [price]})
        min_hour=min(min_hour,hour_of_year)
        max_hour=max(max_hour,hour_of_year)
        # Append the new row to the appropriate DataFrame
        data_storage[Sale_purchase][hour_of_year - 1].append({'volume': volume, 'price': price})
        data_storage['Dates'][hour_of_year - 1] = date

        
        date_prev = date
        hour_prev = hour
        
    for i in range(8760):
        data[i]['Date'] = data_storage['Dates'][i]
        data[i]['Sell'] = pd.DataFrame(data_storage['Sell'][i])
        data[i]['Purchase'] = pd.DataFrame(data_storage['Purchase'][i])
    
    
    
    
    
    t2= time.time()    
    t_loaddata = t2-t1    
    
    min_hour=max(min_hour,start)
    max_hour=min(max_hour,end)
    
    
    small_data=data[min_hour-1: max_hour]
    s=1
    count = 0
    
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
       
        cost_generation_curve(data, hour,increase_eq_price)
        count+=1
        # print(hour)
    
        
    t3 = time.time()
    t_process = t3-t2
    t_avprocess = t_process/count
    timing_info = {
    "load data": t_loaddata,
    "avg process": t_avprocess,
    "tot process": t_process } 
    return small_data , timing_info
    
    
    
def cost_generation_curve(data, hour,increase_eq_price):
    chosen_entry, supply_df, demand_df, eq_volume, eq_price,volumes , supply_interp, demand_interp,min_volume,max_volume = prepare_data_and_interpolate(data, hour)
    
    max_price=eq_price+increase_eq_price
    if max_price<10:
        max_price = 10
    # Calculate benefits and costs separately
    coefficients_BC, volume_linspace_demand,cumulative_benefit_interp = calculate_benefit_of_consumption(demand_df, eq_volume, demand_interp,min_volume,max_volume,eq_price,max_price)
    coefficients_CG, volume_linspace_supply,cumulative_cost_interp ,min_volume_s,first_positive_volume= calculate_cost_of_generation(supply_df, min_volume,eq_volume,max_volume,max_price, supply_interp,eq_price)
    
    a_CG = coefficients_CG[0]
    b_CG = coefficients_CG[1]
    c_CG = coefficients_CG[2]
    
    a_BC = coefficients_BC[0]
    b_BC = coefficients_BC[1]
    c_BC = coefficients_BC[2]
    
    P_min_curve= -b_CG/(2*a_CG)
    P_min_data =min_volume_s-eq_volume
    P_max= volume_linspace_supply[-1]
    
    P_min=max(P_min_data,P_min_curve)
    
    
    if P_min >0:
        P_min=0
    if P_max<0:
        P_max=0
        
    volume_linspace_supply = np.linspace(P_min,P_max, num=100)
    
    #Price is determined as the derivation of cost of generation
    price = 2*a_CG*(volume_linspace_supply)+b_CG
    
    cumulative_cost_values_all = cumulative_cost_interp(volumes)
    cumulative_benefit_values_all = cumulative_benefit_interp(volumes)
    
    net_benefit_all =  cumulative_benefit_values_all - cumulative_cost_values_all
     
    
    
    
    
    
    
    
    
    chosen_entry['Integrated_sets'] = pd.DataFrame({
        'volume': volumes,
        'cost_of_generation_values': cumulative_cost_values_all,
        'social_welfare_all':net_benefit_all,
        'benefit_of_consumption_values': cumulative_benefit_values_all})

    chosen_entry['prediction_BC'] = pd.DataFrame({
        'Volume_surplus_values': volume_linspace_demand,
        'real_benefit_prodcution':cumulative_benefit_interp(volume_linspace_demand+eq_volume) ,
        'predicted_benefit_consumption': np.polyval([a_BC, b_BC, c_BC], volume_linspace_demand)
    })
    
    chosen_entry['prediction_CG'] = pd.DataFrame({
        'Volume_surplus_values': volume_linspace_supply,
        'real_cost_gen' : cumulative_cost_interp(volume_linspace_supply+eq_volume) ,
        'predicted_cost_of_gen': np.polyval([a_CG, b_CG, 0], volume_linspace_supply),
        'price':price
    })
    
    
    chosen_entry['poly'] = {
        'a_BC': a_BC,
        'b_BC': b_BC,
        'c_BC': c_BC,
        'a_CG': a_CG,
        'b_CG': b_CG,
        'c_CG': c_CG,
        'P_min': P_min,  # Minimum power output
        'P_max': P_max,  # Maximum power output
    }
    
    
    chosen_entry['Market_price'] = eq_price
    chosen_entry['Volume_eq'] = eq_volume
    chosen_entry['Volume_0'] = first_positive_volume

    
    return coefficients_BC, coefficients_CG



def prepare_data_and_interpolate(data, hour):
    chosen_entry = data[hour - 1]
    supply_data = chosen_entry['Sell']
    demand_data = chosen_entry['Purchase']

    supply_df = pd.DataFrame(supply_data).groupby('volume').mean().reset_index()
    demand_df = pd.DataFrame(demand_data).groupby('volume').mean().reset_index()
    # print(hour-1)
    min_volume = max(min(supply_df['volume']), min(demand_df['volume']))
    max_volume = min(max(supply_df['volume']), max(demand_df['volume']))

    volumes = np.linspace(min_volume, max_volume, num=1000)

    supply_interp = interp1d(supply_df['volume'], supply_df['price'], kind='linear', fill_value="extrapolate")
    demand_interp = interp1d(demand_df['volume'], demand_df['price'], kind='linear', fill_value="extrapolate")

    price_diff = supply_interp(volumes) - demand_interp(volumes)
    nan_indices = np.isnan(price_diff)
    price_diff = price_diff[~nan_indices]
    volumes = volumes[~nan_indices]

    eq_index = np.argmin(np.abs(price_diff))
    eq_volume = volumes[eq_index]
    eq_price = supply_interp(eq_volume).item()
    
    
    return chosen_entry, supply_df, demand_df, eq_volume, eq_price,volumes , supply_interp, demand_interp,min_volume,max_volume


def calculate_benefit_of_consumption(demand_df, eq_volume, demand_interp,min_volume,max_volume,eq_price,max_price):
    cumulative_benfit = [0]
    for i in range(1, len(demand_df)):
        V_prev = demand_df['volume'][i-1]
        V = demand_df['volume'][i]
        benefit_of_consumption, _ = quad(lambda x: demand_interp(x), V_prev, V)
        cumulative_benfit.append(cumulative_benfit[-1] + benefit_of_consumption)
        
    demand_df['cumulative_benefit'] = cumulative_benfit
    
    positive_demand_df = demand_df[demand_df['price'] > 0]
    
   
    
    filtered_demand_df = demand_df[demand_df['price'] < max_price]
    last_positive_price_vol = positive_demand_df['volume'].iloc[-1]
    volume_at_max_price = filtered_demand_df['volume'].iloc[0]
    
    
    if eq_price >0:
        max_volume_s= last_positive_price_vol
    else:
        max_volume_s=max_volume  
    
    
    
    cumulative_benefit_interp = interp1d(demand_df['volume'], demand_df['cumulative_benefit'], fill_value="extrapolate")
    volume_linspace_demand   = np.linspace(volume_at_max_price, max_volume_s, num=100)
    BC_gain_values =  cumulative_benefit_interp(volume_linspace_demand)
    
    #shifting demand to equilibrium point
    # volume_linspace_demand-=eq_volume
    coefficients_BC = np.polyfit(volume_linspace_demand, BC_gain_values, 2)
        
    return coefficients_BC, volume_linspace_demand ,cumulative_benefit_interp

def calculate_cost_of_generation(supply_df, min_volume,eq_volume,max_volume,max_price, supply_interp,eq_price):
    cumulative_cost = [0]

    for i in range(1, len(supply_df)):
        V_prev = supply_df['volume'][i-1]
        V = supply_df['volume'][i]
        cost_of_generation, _ = quad(lambda x: supply_interp(x), V_prev, V)
        cumulative_cost.append(cumulative_cost[-1] + cost_of_generation)

    supply_df['cumulative_cost'] = cumulative_cost
   
    positive_supply_df = supply_df[supply_df['price'] > 1]
    first_positive_price_index = supply_df[supply_df['price'] > 0].index[0]
    first_positive_volume = supply_df.loc[first_positive_price_index, 'volume']
    
    def fit_with_vertex_constraint(coeffs):
        # Coefficients of the quadratic: y = ax^2 + bx + c
        a, b, c = coeffs
        
        # Compute the vertex x-coordinate: x = -b / (2a)
        vertex_x = -b / (2 * a)
        
        # Compute the quadratic fit values
        fitted_curve = a * volume_linspace_supply**2 + b * volume_linspace_supply + c
        
        # Calculate the error in fitting
        fit_error = np.sum((fitted_curve - CG_cost_values) ** 2)/ len(CG_cost_values)
        
        # Add a penalty for the vertex not aligning with the first positive volume
        vertex_penalty = (vertex_x - first_positive_volume) ** 2
        
        # Combine the fit error and the penalty
        penalty_weight = 10  # Adjust this weight to balance the fit and vertex alignment
        # print("Fitting Error:", fit_error)
        # print("Penalty:", vertex_penalty)
        return fit_error + penalty_weight * vertex_penalty
    
    filtered_supply_df = supply_df[supply_df['price'] < max_price]
    
    if positive_supply_df.index[0] > 0:  # Ensure there is a previous index
        if eq_price <100:
            previous_price_vol = supply_df['volume'].iloc[positive_supply_df.index[0]]
        else:
            previous_price_vol = supply_df['volume'].iloc[positive_supply_df.index[0]]
    else:
        previous_price_vol = supply_df['volume'].iloc[positive_supply_df.index[0]]
    
 
    
    first_positive_price_vol=previous_price_vol
    volume_at_max_price = filtered_supply_df['volume'].iloc[-1]
    
    
    if eq_price >0:
        min_volume_s= first_positive_price_vol
    else:
        min_volume_s=eq_volume  
    # max_volume_s=min(volume_at_max_price,max_volume)    
        
    cumulative_cost_interp = interp1d(supply_df['volume'], supply_df['cumulative_cost'], fill_value="extrapolate")
    volume_linspace_supply = np.linspace(min_volume_s,volume_at_max_price, num=100)
    CG_cost_values =  cumulative_cost_interp(volume_linspace_supply)
    
    
    #shifting supply to equilibrium point
    volume_linspace_supply-=eq_volume
    # coefficients_CG = np.polyfit(volume_linspace_supply, CG_cost_values, 2)
    
    # Initial guess for coefficients [a, b, c]
    initial_guess = np.polyfit(volume_linspace_supply, CG_cost_values, 2)
    
    if eq_price >15: 
        bounds = [(1e-5, None),  # 'a' > 0 (positive)
              (1e-5, np.inf),  # 'b' can be any real number
              (-np.inf, np.inf)]  # 'c' can be any real number
        # Optimise the fit
        result = minimize(fit_with_vertex_constraint, initial_guess, bounds=bounds,method='L-BFGS-B')
        # Extract optimised coefficients
        optimized_coeffs = result.x
        a_CG, b_CG, c_CG = optimized_coeffs
        coefficients_CG = optimized_coeffs
    else:
        coefficients_CG = initial_guess
    return coefficients_CG, volume_linspace_supply,cumulative_cost_interp,min_volume_s,first_positive_volume




def plot_curves(data, hour, name=None):
    chosen_entry = data[hour - 1]
    # Extract Sell, Purchase, and Social Cost data
    sell_data = chosen_entry['Sell']
    purchase_data = chosen_entry['Purchase']
    intSet = chosen_entry['Integrated_sets'] 
    pcg = chosen_entry['prediction_CG']
    pbc = chosen_entry['prediction_BC']

    curve_data= chosen_entry['poly']
    # Create a subplot figure with 2 rows and 1 column
    fig = make_subplots(rows=2, cols=2, shared_xaxes=True, vertical_spacing=0.1, column_widths=[0.7, 0.3],
                        subplot_titles=('Supply and Demand', 'Cost of Generation', 'Integrated supply and demand','Price'))
    
    eq_price=chosen_entry['Market_price'] 
    eq_volume= chosen_entry['Volume_eq'] 
    
    c_CG=curve_data['c_CG']
    
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
    if not intSet.empty:
        fig.add_trace(go.Scatter(x=intSet['volume'], y=intSet['cost_of_generation_values'], mode='lines+markers', name='Cost of Gen'), row=2, col=1)
        fig.add_trace(go.Scatter(x=intSet['volume'], y=intSet['benefit_of_consumption_values'], mode='lines+markers', name='Benefit of consumption'), row=2, col=1)
        fig.add_trace(go.Scatter(x=intSet['volume'], y=intSet['social_welfare_all'], mode='lines+markers', name='Social Welfare'), row=2, col=1)
        
        # fig.add_trace(go.Scatter(x=pbc['Volume_surplus_values'], y=pbc['predicted_benefit_consumption'], mode='lines+markers', name='predicted_benefit_consumption'), row=2, col=1)

        
        fig.add_trace(go.Scatter(x=pcg['Volume_surplus_values'], y=pcg['real_cost_gen']-c_CG, mode='lines+markers', name='Cost of Generation'), row=1, col=2)
        fig.add_trace(go.Scatter(x=pcg['Volume_surplus_values'], y=pcg['predicted_cost_of_gen'], mode='lines+markers', name='Predicted cost of gen'), row=1, col=2)
        
        fig.add_trace(go.Scatter(x=pcg['Volume_surplus_values'], y=pcg['price'], mode='lines+markers', name='Predicted price CoG'), row=2, col=2)
        
    mp = np.round(chosen_entry['Market_price'], decimals=2)
    # Add vertical line at equilibrium volume and horizontal line at equilibrium price
    
    
    for line_data, line_color, line_name in [ #(eq_volume-curve_data['shift'],'Green',''),            
                                            (curve_data['P_min'] , 'Blue', 'P min'),
                                              (curve_data['P_max'] , 'Blue', 'P max')]:
        fig.add_trace(go.Scatter(x=[line_data, line_data], 
                                  y=[min(pcg['price']),
                                    max(pcg['price'])],
                                  mode='lines', line=dict(color=line_color, width=2, dash='dash'), name=line_name), row=2, col=2)
    
    
  
    # Update layout with titles and axis labels
    if name is not None:
        fig.update_layout(        title=f'Supply and Demand for hour {hour}; Market Price: {mp}, Price Zone: {name}'           )
    
    else:
        fig.update_layout(        title=f'Supply and Demand for hour {hour}; Market Price: {mp}'           )
    
    # Update xaxis properties
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


