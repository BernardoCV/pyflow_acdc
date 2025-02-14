# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 12:03:23 2024

@author: BernardoCastro
"""

import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go

__all__ = [
    'run_dash'
]


def plot_TS_res(grid, plotting_choice, selected_rows, x_limits=None, y_limits=None):
    # Select the appropriate DataFrame based on plotting_choice
   
    
    if plotting_choice == 'Curtailment':
        df = grid.time_series_results['curtailment']* 100
    elif plotting_choice in ['Power Generation by generator','Power Generation by generator area chart']:
        df = grid.time_series_results['real_power_opf']*grid.S_base
    elif plotting_choice in ['Power Generation by price zone','Power Generation by price zone area chart'] :
        df = grid.time_series_results['real_power_by_zone']*grid.S_base
    elif plotting_choice == 'Market Prices':
        df = grid.time_series_results['prices_by_zone']
    elif plotting_choice == 'AC line loading':
        df = grid.time_series_results['ac_line_loading']* 100
    elif plotting_choice == 'DC line loading':
        df = grid.time_series_results['dc_line_loading']* 100
    elif plotting_choice == 'AC/DC Converters':
        df = grid.time_series_results['converter_loading']*100

    # print(plotting_choice)
    
    # Prepare the time index
    time = df.index

    # Create a figure
    fig = go.Figure()

    # Initialize a variable for cumulative sum if stacking is required
    cumulative_sum = None
    
    # Check if we need to stack the areas for specific plotting choices
    stack_areas = plotting_choice in ['Power Generation by generator area chart', 'Power Generation by price zone area chart']

    # Adding traces to the figure for selected rows
    for col in selected_rows:
        if col in df.columns:  # Check if the column exists
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

    # Update layout for the figure
    fig.update_layout(
        title="Time Series Data",
        xaxis_title="Time",
        yaxis_title="Values",
        hovermode="x",
    )

    # Set x-axis limits if provided
    if x_limits is None:
        x_limits = (df.index[0], df.index[-1])
    fig.update_xaxes(range=x_limits)
    
    # Set y-axis limits if provided
    if y_limits and len(y_limits) == 2:
        fig.update_yaxes(range=y_limits)

    return fig

def create_dash_app(grid):
    app = dash.Dash(__name__)

    # Define layout
    app.layout = html.Div([
        dcc.Dropdown(
            id='plotting-choice',
            options=[
                {'label': 'Power Generation by price zone', 'value': 'Power Generation by price zone'},
                {'label': 'Power Generation by generator', 'value': 'Power Generation by generator'},
                {'label': 'Power Generation by price zone area chart', 'value': 'Power Generation by price zone area chart'},
                {'label': 'Power Generation by generator area chart', 'value': 'Power Generation by generator area chart'},
                {'label': 'Market Prices', 'value': 'Market Prices'},
                {'label': 'AC line loading', 'value': 'AC line loading'},
                {'label': 'DC line loading', 'value': 'DC line loading'},
                {'label': 'AC/DC Converters', 'value': 'AC/DC Converters'},
                {'label': 'Curtailment', 'value': 'Curtailment'}
            ],
            value='Power Generation by price zone'  # Default value
        ),
        dcc.Checklist(
            id='subplot-selection',
            options=[],  # This will be updated dynamically
            value=[],  # Default to show no columns initially
            inline=True
        ),
        html.Div([
            html.Label('X-axis limits:'),
            dcc.Input(id='x-min', type='number', placeholder='X min', value=None),
            dcc.Input(id='x-max', type='number', placeholder='X max', value=None)
        ]),
        html.Div([
            html.Label('Y-axis limits:'),
            dcc.Input(id='y-min', type='number', placeholder='Y min', value=0),
            dcc.Input(id='y-max', type='number', placeholder='Y max', value=100)
        ]),
        dcc.Graph(id='plot-output')
    ])

    @app.callback(
        Output('subplot-selection', 'options'),
        Output('subplot-selection', 'value'),
        Input('plotting-choice', 'value')
    )
    def update_subplot_options(plotting_choice):
        # Get available columns based on the selected plotting choice
        available_columns = []

        if plotting_choice == 'Curtailment':
            available_columns = grid.time_series_results['curtailment'].columns.tolist()
        elif plotting_choice in ['Power Generation by generator','Power Generation by generator area chart']:
            available_columns = grid.time_series_results['real_power_opf'].columns.tolist()
        elif plotting_choice in ['Power Generation by price zone','Power Generation by price zone area chart']:
            available_columns = grid.time_series_results['real_power_by_zone'].columns.tolist()
        elif plotting_choice == 'Market Prices':
            available_columns = grid.time_series_results['prices_by_zone'].columns.tolist()
        elif plotting_choice == 'AC line loading':
            available_columns = grid.time_series_results['ac_line_loading'].columns.tolist()
        elif plotting_choice == 'DC line loading':
            available_columns = grid.time_series_results['dc_line_loading'].columns.tolist()
        elif plotting_choice == 'AC/DC Converters':
            available_columns = grid.time_series_results['converter_loading'].columns.tolist()


        # Set the default value to all available columns
        default_value = available_columns if available_columns else []

        # Return options and default selected value
        return (
            [{'label': col, 'value': col} for col in available_columns],
            default_value
        )
    @app.callback(
     Output('y-min', 'value'),
     Output('y-max', 'value'),
     Input('plotting-choice', 'value')
 )
    def update_limits(plotting_choice):
         # Get the data for the selected plotting choice to determine limits
         if plotting_choice == 'Curtailment':
             data = grid.time_series_results['curtailment']* 100
         elif plotting_choice in ['Power Generation by generator','Power Generation by generator area chart']:
             data = grid.time_series_results['real_power_opf']*grid.S_base
         elif plotting_choice in ['Power Generation by price zone','Power Generation by price zone area chart'] :
             data = grid.time_series_results['real_power_by_zone']*grid.S_base
         elif plotting_choice == 'Market Prices':
             data = grid.time_series_results['prices_by_zone']
         elif plotting_choice == 'AC line loading':
             data = grid.time_series_results['ac_line_loading']* 100
         elif plotting_choice == 'DC line loading':
             data = grid.time_series_results['dc_line_loading']* 100
         elif plotting_choice == 'AC/DC Converters':
             data = grid.time_series_results['converter_loading']*100
         else:
             data = None
    
         if data is not None and not data.empty:
            y_min = int(min(0, data.min().min() - 5))
    
            # Determine if we are in a stacking scenario
            if plotting_choice in ['Power Generation by generator area chart', 'Power Generation by price zone area chart']:
                # For stacking, calculate the cumulative sum of all columns (stacked area)
                cumulative_sum = data.sum(axis=1)
                y_max = int(cumulative_sum.max() + 10)  # Set y_max to the max cumulative sum plus a buffer
            elif plotting_choice in ['AC line loading', 'DC line loading', 'Curtailment']:
                y_max = int(min(data.max().max() + 10, 100))  # Cap y_max at 100 for these cases
            else:
                y_max = int(data.max().max() + 10)
         else:
             y_min, y_max =  0, 1  # Default if no data
    
         return y_min, y_max

    @app.callback(
        Output('plot-output', 'figure'),
        [Input('plotting-choice', 'value'),
         Input('subplot-selection', 'value'),
         Input('x-min', 'value'),
         Input('x-max', 'value'),
         Input('y-min', 'value'),
         Input('y-max', 'value')]
    )
    def update_graph(plotting_choice, selected_rows, x_min, x_max, y_min, y_max):
        # Create x and y limits as tuples
        x_limits = (x_min, x_max) if x_min is not None and x_max is not None else None
        y_limits = (y_min, y_max) if y_min is not None and y_max is not None else None

        # Call the plotting function with the selected option, rows, and limits
        return plot_TS_res(grid, plotting_choice, selected_rows, x_limits=x_limits, y_limits=y_limits)

    return app

def run_dash(grid):
    app = create_dash_app(grid)
    app.run_server(debug=True)

