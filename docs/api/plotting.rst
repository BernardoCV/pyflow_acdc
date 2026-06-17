Plotting Module
===============

This page has been pre-filled with the functions that are available in the Plotting module by AI, please check the code for more details.

This module provides functions for visualizing grid components and results.

functions are found in pyflow_acdc.Graph_and_plot

Time Series
-----------

Time series results
^^^^^^^^^^^^^^^^^^^

This function is used to plot the time series results of the grid.

.. autofunction:: pyflow_acdc.plot_TS_res

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
        - int
        - Start timeframe
        - Required
      * - ``end``
        - int
        - End timeframe
        - Required
      * - ``plotting_choices``
        - list
        - Results types to plot
        - All
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
        - Format to save the plot, if None, the plot are not saved
        - None

   **Example**

   .. code-block:: python

       import pyflow_acdc as pyf
       import pandas as pd

       [grid,results] = pyf.NS_MTDC()

       start = 5750
       end = 6000
       obj = {'Energy_cost': 1}

       market_prices_url = "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_marketPrices_data_sd2024.csv"
       TS_MK = pd.read_csv(market_prices_url)
       pyf.add_TimeSeries(grid,TS_MK)

       wind_load_url = "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_WL_data2024.csv"
       TS_wl = pd.read_csv(wind_load_url)
       pyf.add_TimeSeries(grid,TS_wl)

       times=pyf.TS_ACDC_OPF(grid,start,end,ObjRule=obj)  

       pyf.plot_TS_res(grid,start,end,save_format='svg')

   Plot shown in browser:

   .. themed-figure:: ts_plot_browser
      :alt: Time Series Plot: Power Generation by price zone
      :align: center
      :width: 70%

   Plot saved in current working directory:

   .. themed-figure:: ts_plot_save
      :alt: Time Series Plot: Power Generation by price zone
      :align: center
      :width: 70%

Time series probability
^^^^^^^^^^^^^^^^^^^^^^^

This function is used to plot the probability of the time series parameters or results of the grid. Results currently available are:

- 'Power Generation by generator'
- 'Prices by price zone'
- 'AC line loading'
- 'DC line loading'
- 'AC/DC Converters loading'

.. autofunction:: pyflow_acdc.time_series_prob

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid Class
        - Grid to analyze
        - Required
      * - ``element_name``
        - str
        - Name of element to analyze
        - Required
      * - ``save_format``
        - str
        - Format to save the plot, if None, the plot are not saved
        - None
      * - ``path``
        - str
        - Path to save the plot
        - Current working directory


   **Example**

   .. code-block:: python

      import pyflow_acdc as pyf
      import pandas as pd

      [grid,results] = pyf.NS_MTDC()

      start = 5750
      end = 6000
      obj = {'Energy_cost': 1}

      market_prices_url = "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_marketPrices_data_sd2024.csv"
      TS_MK = pd.read_csv(market_prices_url)
      pyf.add_TimeSeries(grid,TS_MK)

      wind_load_url = "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/NS_MTDC_TS/NS_TS_WL_data2024.csv"
      TS_wl = pd.read_csv(wind_load_url)
      pyf.add_TimeSeries(grid,TS_wl)

      pyf.time_series_prob(grid,'OWPP_BE',save_format='svg')
      pyf.time_series_prob(grid,'BE_price',save_format='svg')
      pyf.time_series_prob(grid,'L_BE',save_format='svg')

   .. list-table::
      :widths: 50 50 50 

      * - .. themed-figure:: owpp_be_distribution
           :align: center
        - .. themed-figure:: be_price_distribution
           :align: center
        - .. themed-figure:: l_be_distribution
           :align: center

Network Graph Visualization
---------------------------

Full grid visualization as a network graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.plot_graph

   Creates an interactive network graph visualization using Plotly.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``Grid``
        - Grid
        - Grid to visualize
        - Required
      * - ``text``
        - str
        - Hover text format ('data' or 'inPu' or 'abs')
        - 'inPu'
      * - ``base_node_size``
        - int
        - Base size for nodes
        - 10
      * - ``G``
        - Graph
        - Graph to visualize
        - Full grid

   **Example**

   .. code-block:: python

       import pyflow_acdc as pyf

       grid,res = pyf.case24_3zones_acdc()

       pyf.plot_graph(grid)

   .. themed-figure:: case24acdc_full
      :alt: case24_3zones_acdc_graph
      :align: center
      :width: 70%





Neighbor Graph
^^^^^^^^^^^^^^

This function is used to plot the neighbor graph of a node. You can either provide a node or a node name, one or the other must be provided.

.. autofunction:: pyflow_acdc.plot_neighbour_graph

   Creates a graph visualization of a node's neighbors.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid to analyze
        - Required
      * - ``node``
        - Node
        - Node object
        - None
      * - ``node_name``
        - str
        - Node name
        - None
      * - ``base_node_size``
        - int
        - Base size for nodes
        - 10
      * - ``proximity``
        - int
        - Proximity
        - 1

   **Example**

   .. code-block:: python

       import pyflow_acdc as pyf

       grid,res = pyf.case24_3zones_acdc()

       pyf.plot_neighbour_graph(grid,node_name='111.0')

   .. themed-figure:: case24acdc_neig
      :alt: case24_3zones_acdc neighbour graph of node 111.0
      :align: center
      :width: 70%

Saving the Network Graph
^^^^^^^^^^^^^^^^^^^^^^^^^

For this function, you need to have the svgwrite library installed. You can install it using pip install svgwrite. ``geometry`` of objects is required.

.. autofunction:: pyflow_acdc.save_network_svg

   Saves the network graph as an SVG file.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid to save
        - Required
      * - ``name``
        - str
        - Name of the file  
        - 'grid_network'
      * - ``width``
        - int
        - Width of the file
        - 1000
      * - ``height``
        - int
        - Height of the file
        - 800

   **Example**

   .. code-block:: python 

       import pyflow_acdc as pyf

       grid,res = pyf.NS_MTDC()

       pyf.save_network_svg(grid)

   .. themed-figure:: grid_network
      :alt: grid_network
      :align: center
      :width: 70%


Solver Diagnostics
------------------

.. autofunction:: pyflow_acdc.plot_model_feasibility

   Plots solver feasible-solution progress and objective evolution.

3D Grid Plot
------------

.. autofunction:: pyflow_acdc.plot_3D

   Generates a 3D network visualization of the grid and selected element
   attributes.

