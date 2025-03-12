Plotting Module
==============

This page has been pre-filled with the functions that are available in the Plotting module by AI, please check the code for more details.

This module provides functions for visualizing grid components and results.

functions are found in pyflow_acdc.Graph_and_plot

Time Series Results
-------------------

This function is used to plot the time series results of the grid.

.. py:function:: plot_TS_res(grid, start, end, plotting_choices=[],show=True,path=None,save_format='svg')

   Creates plots for time series results. The possible plotting choices are:

   - 'Power Generation by price zone'
   - 'Power Generation by generator'
   - 'Curtailment'
   - 'Market Prices'
   - 'AC line loading'
   - 'DC line loading'
   - 'AC/DC Converters'
   - 'Power Generation by generator area chart'
   - 'Power Generation by price zone area chart'


   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid with results
        - Required
      * - ``start``
        - str
        - Start date
        - None
      * - ``end``
        - str
        - End date
        - None
      * - ``plotting_choices``
        - list
        - Results types to plot
        - None
      * - ``show``    
        - bool
        - Whether to show the plot in browser
        - True
      * - ``path``  
        - str
        - Path to save the plot
        - Current working directory
      * - ``save_format``
        - str
        - Format to save the plot
        - 'svg'

   **Example**

   .. code-block:: python

       import pyflow_acdc as pyf
       [grid,results] = pyf.NS_MTDC()

       start = 5750
       end = 6000
       obj = {'Energy_cost': 1}

       market_prices_url = "https://raw.githubusercontent.com/BernardoCV/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_marketPrices_data_sd2024.csv"
       TS_MK = pd.read_csv(market_prices_url)
       pyf.add_TimeSeries(grid,TS_MK)

       wind_load_url = "https://raw.githubusercontent.com/BernardoCV/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_WL_data2024.csv"
       TS_wl = pd.read_csv(wind_load_url)
       pyf.add_TimeSeries(grid,TS_wl)

       times=pyf.TS_ACDC_OPF(grid,start,end,ObjRule=obj)  

       grid.plot_TS_res(grid,start,end)



Network Graph Visualization
---------------------------

plot_Graph
^^^^^^^^^^

.. py:function:: plot_Graph(Grid, image_path=None, dec=3, text='inPu', grid_names=None, base_node_size=10, G=None)

   Creates an interactive network graph visualization using Plotly.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``Grid``
        - Grid
        - Grid to visualize
        - Required
        - -
      * - ``image_path``
        - str
        - Path to save image
        - None
        - -
      * - ``dec``
        - int
        - Decimal places
        - 3
        - -
      * - ``text``
        - str
        - Hover text format ('inPu' or 'abs')
        - 'inPu'
        - -
      * - ``grid_names``
        - dict
        - Custom node names
        - None
        - -
      * - ``base_node_size``
        - int
        - Base size for nodes
        - 10
        - -

   **Example**

   .. code-block:: python

       grid.plot_Graph(text='abs')



Neighbor Graph
------------

plot_neighbour_graph
^^^^^^^^^^^^^^^^^^

.. py:function:: plot_neighbour_graph(grid, node, depth=1)

   Creates a graph visualization of a node's neighbors.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid to analyze
        - Required
        - -
      * - ``node``
        - Node
        - Central node
        - Required
        - -
      * - ``depth``
        - int
        - Neighbor depth
        - 1
        - -

   **Example**

   .. code-block:: python

       grid.plot_neighbour_graph(node, depth=2)

Interactive Dashboard
------------------

run_dash
^^^^^^^

.. py:function:: run_dash(grid)

   Creates and runs an interactive Dash web application for visualizing time series results.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid with time series results
        - Required
        - -

   **Features**:

   - Interactive plot selection:
     - Power Generation by price zone
     - Power Generation by generator
     - Power Generation by price zone (area chart)
     - Power Generation by generator (area chart)
     - Market Prices
     - AC line loading
     - DC line loading
     - AC/DC Converters
     - Curtailment
   - Dynamic axis limits
   - Component selection checklist
   - Real-time plot updates

   **Example**

   .. code-block:: python

       pyf.run_dash(grid)

Dashboard Components
-----------------

plot_TS_res
^^^^^^^^^^

.. py:function:: plot_TS_res(grid, plotting_choice, selected_rows, x_limits=None, y_limits=None)

   Creates plots for the Dash dashboard.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid with results
        - Required
        - -
      * - ``plotting_choice``
        - str
        - Type of plot
        - Required
        - -
      * - ``selected_rows``
        - list
        - Components to plot
        - Required
        - -
      * - ``x_limits``
        - tuple
        - X-axis limits
        - None
        - -
      * - ``y_limits``
        - tuple
        - Y-axis limits
        - None
        - -

   **Available Plot Types**:

   - Line plots
   - Stacked area charts
   - Time series data
   - Loading percentages
   - Price curves

   Returns a Plotly figure object for the dashboard.