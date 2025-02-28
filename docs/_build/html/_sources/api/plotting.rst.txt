Plotting Module
==============

This page has been pre-filled with the functions that are available in the Plotting module by AI, please check the code for more details.

This module provides functions for visualizing grid components and results.

functions are found in pyflow_acdc.Graph_and_plot

Network Graph Visualization
-------------------------

plot_Graph
^^^^^^^^^

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

Map Visualization
---------------

plot_folium
^^^^^^^^^^

.. py:function:: plot_folium(grid, text='inPu', name='grid_map', tiles="CartoDB Positron", polygon=None, ant_path='None', clustering=True, coloring=None)

   Creates an interactive map visualization using Folium.

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
        - Grid to visualize
        - Required
        - -
      * - ``text``
        - str
        - Hover text format ('inPu' or 'abs')
        - 'inPu'
        - -
      * - ``name``
        - str
        - Output file name
        - 'grid_map'
        - -
      * - ``tiles``
        - str
        - Map style
        - "CartoDB Positron"
        - -
      * - ``ant_path``
        - str
        - Animated paths
        - 'None'
        - -
      * - ``clustering``
        - bool
        - Enable marker clustering
        - True
        - -

   **Features**:

   - Interactive map with zoom/pan
   - Voltage level filtering
   - Component type layers:
     - MVAC Lines (<110kV)
     - HVAC Lines (<300kV)
     - HVAC Lines (<500kV)
     - HVDC Lines
     - Converters
     - Transformers
   - Marker clustering for generators/loads
   - Hover information for components
   - Optional animated power flows

   **Example**

   .. code-block:: python

       grid.plot_folium(name='my_map', text='inPu')

Time Series Results
-----------------

plot_TS_res
^^^^^^^^^^

.. py:function:: plot_TS_res(grid, results_to_plot=None)

   Creates plots for time series results.

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
      * - ``results_to_plot``
        - list
        - Results types to plot
        - None
        - -

   **Available Result Types**:

   - Line loadings
   - Node voltages
   - Power flows
   - Generator dispatch
   - Load profiles
   - Converter flows

   **Example**

   .. code-block:: python

       grid.plot_TS_res(['line_loading', 'voltages'])

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