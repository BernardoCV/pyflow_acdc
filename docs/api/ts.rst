Time Series Module
==================

This module provides functions for time series analysis of power flows and optimal power flow.

functions are found in pyflow_acdc.Time_series


Sequential AC/DC Time Series Power Flow
---------------------------------------

.. py:function:: TS_ACDC_PF(grid, start=1, end=99999, print_step=False)

   Performs sequential AC/DC power flow for time series data.

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
      * - ``start``
        - int
        - Start time step
        - 1
      * - ``end``
        - int
        - End time step
        - 99999
      * - ``print_step``
        - bool
        - Print progress
        - False

   **Returns**

   Results are stored in ``grid.time_series_results`` dictionary with the following keys:

   - ``PF_results``: Node voltages and power flows
   - ``line_loading``: Line loading percentages
   - ``ac_line_loading``: AC line loading percentages
   - ``dc_line_loading``: DC line loading percentages
   - ``converter_loading``: Converter loading percentages
   - ``grid_loading``: Overall grid loading

   **Example**

   .. code-block:: python

       pyf.TS_ACDC_PF(grid, start=1, end=24)

Optimal Power Flow Time Series
------------------------------

.. py:function:: TS_ACDC_OPF(grid,start=1,end=99999,ObjRule=None ,price_zone_restrictions=False,expand=False,print_step=False)
    
   Performs time series optimal power flow analysis.

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
      * - ``start``
        - int
        - Start time step 
        - 1
      * - ``end``
        - int
        - End time step
        - 99999 
      * - ``ObjRule``
        - dict
        - Objective rule, check :ref:`Objective Functions <obj_functions>` for more details
        - None
      * - ``price_zone_restrictions``
        - bool
        - Price zone restrictions, adds price zone restrictions to the model [1]_
        - False
      * - ``expand``
        - bool
        - Expand price zone import limits
        - False
      * - ``print_step``
        - bool
        - Print step in the terminal
        - False

   **Returns**

   Results are stored in ``grid.time_series_results`` dictionary with the following keys:

   * ``converter_p_dc`` - Converter power in DC side
   * ``converter_q_ac`` - Converter power in AC side
   * ``converter_p_ac`` - Converter power in AC side
   * ``converter_loading`` - Converter loading percentages
   * ``real_load_opf`` - Real load per node
   * ``real_power_opf`` - Real power per generator
   * ``reactive_power_opf`` - Reactive power per generator
   * ``curtailment`` - Curtailment values
   * ``line_loading`` - Line loading percentages
   * ``grid_loading`` - Loading by unsynchronized grids
   * ``prices_by_zone`` - Prices by price zone
   * ``prices_by_zone_total`` - Total prices by price zone
   * ``ac_line_loading`` - AC line loading percentages
   * ``dc_line_loading`` - DC line loading percentages
   * ``real_load_by_zone`` - Real load per price zone
   * ``real_power_by_zone`` - Real power per price zone

   It also returns a dictionary with the timing information.

   **Example**

   .. code-block:: python

       timing_info = pyf.TS_ACDC_OPF(grid, ObjRule={'Energy_cost': 1.0})

Parallel Time Series OPF
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: TS_ACDC_OPF_parallel(grid, ObjRule=None, PV_set=False, OnlyGen=True, Price_Zones=False)

   Performs parallel time series optimal power flow analysis. Creates parallel sub-models to speed up the calculation.

   **Returns**
   Results are saved in ``grid.time_series_results`` and the average elapsed time is returned.

   **Example**

   .. code-block:: python

       average_elapsed_time=pyf.TS_ACDC_OPF_parallel(grid)

Statistical Analysis
--------------------

.. py:function:: Time_series_statistics(grid, curtail=0.99, over_loading=0.9)

   Calculates statistical metrics for time series results.

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
      * - ``curtail``
        - float
        - Curtailment percentile
        - 0.99
      * - ``over_loading``
        - float
        - Overloading threshold
        - 0.9

   Calculates for each time series:

   - Mean
   - Median
   - Maximum/Minimum
   - Mode
   - IQR
   - Percentiles

Results Export
--------------

.. py:function:: results_TS_OPF(grid, excel_file_path, grid_names=None, stats=None, times=None)

   Exports time series results to Excel file.

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
      * - ``excel_file_path``
        - str
        - Output file path
        - Required
      * - ``grid_names``
        - dict
        - Grid name mappings
        - None
      * - ``stats``
        - DataFrame
        - Statistical results
        - None
      * - ``times``
        - dict
        - Computation times
        - None

   Exports sheets for:

   - ``Time`` - Timing information
   - ``All line loadings (AC/DC)`` - All line loadings (AC/DC)
   - ``AC line loadings`` - AC line loading percentages
   - ``DC line loadings`` - DC line loading percentages
   - ``Grid loadings`` - Grid loading percentages
   - ``Converter DC power`` - Converter power in DC side
   - ``Converter AC power`` - Converter power in AC side
   - ``Converter AC reactive power`` - Converter reactive power in AC side
   - ``Real load per node`` - Real load per node
   - ``Real power per generator`` - Real power per generator
   - ``Reactive power per generator`` - Reactive power per generator
   - ``Curtailment`` - Curtailment
   - ``Converter loading`` - Converter loading percentages
   - ``Real load by zone`` - Real load by zone
   - ``Real power by zone`` - Real power by zone
   - ``Reactive power by zone`` - Reactive power by zone
   - ``Prices by zone`` - Prices by zone
   - ``Statistics`` - Statistics

   **Example**

   .. code-block:: python

       pyf.results_TS_OPF(grid, "results", stats=stats_df)

References
----------

.. [1] B. C. Valerio, V. A. Lacerda, M. Cheah-Mane, P. Gebraad and O. Gomis-Bellmunt,
       "Optimizing Offshore Wind Integration through Multi-Terminal DC Grids: A
       Market-Based OPF Framework for the North Sea Interconnectors"

