Time Series Module
=================

This module provides functions for time series analysis of power flows and optimal power flow.

functions are found in pyflow_acdc.Time_series

Power Flow Time Series
--------------------

.. py:function:: Time_series_PF(grid)

   Performs time series power flow analysis.

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

   **Example**

   .. code-block:: python

       pyf.Time_series_PF(grid)

Sequential AC/DC Time Series
--------------------------

.. py:function:: TS_ACDC_PF(grid, start=1, end=99999, print_step=False)

   Performs sequential AC/DC power flow for time series data.

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
      * - ``start``
        - int
        - Start time step
        - 1
        - -
      * - ``end``
        - int
        - End time step
        - 99999
        - -
      * - ``print_step``
        - bool
        - Print progress
        - False
        - -

   Results are stored in grid.time_series_results containing:

   - PF_results: Node voltages and power flows
   - line_loading: Line loading percentages
   - ac_line_loading: AC line loading percentages
   - dc_line_loading: DC line loading percentages
   - converter_loading: Converter loading percentages
   - grid_loading: Overall grid loading

   **Example**

   .. code-block:: python

       pyf.TS_ACDC_PF(grid, start=1, end=24)

Optimal Power Flow Time Series
----------------------------

.. py:function:: TS_ACDC_OPF(grid, ObjRule=None, PV_set=False, OnlyGen=True, Price_Zones=False)

   Performs time series optimal power flow analysis.

   Parameters same as OPF_ACDC plus time series specific options.

   **Example**

   .. code-block:: python

       pyf.TS_ACDC_OPF(grid, ObjRule={'Energy_cost': 1.0})

Parallel Time Series OPF
^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: TS_ACDC_OPF_parallel(grid, ObjRule=None, PV_set=False, OnlyGen=True, Price_Zones=False)

   Performs parallel time series optimal power flow analysis.

   Uses parallel processing to speed up time series calculations.

   **Example**

   .. code-block:: python

       pyf.TS_ACDC_OPF_parallel(grid)

Statistical Analysis
------------------

.. py:function:: Time_series_statistics(grid, curtail=0.99, over_loading=0.9)

   Calculates statistical metrics for time series results.

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
      * - ``curtail``
        - float
        - Curtailment percentile
        - 0.99
        - -
      * - ``over_loading``
        - float
        - Overloading threshold
        - 0.9
        - -

   Calculates for each time series:

   - Mean
   - Median
   - Maximum/Minimum
   - Mode
   - IQR
   - Percentiles

Results Export
-------------

.. py:function:: results_TS_OPF(grid, excel_file_path, grid_names=None, stats=None, times=None)

   Exports time series results to Excel file.

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
      * - ``excel_file_path``
        - str
        - Output file path
        - Required
        - -
      * - ``grid_names``
        - dict
        - Grid name mappings
        - None
        - -
      * - ``stats``
        - DataFrame
        - Statistical results
        - None
        - -
      * - ``times``
        - dict
        - Computation times
        - None
        - -

   Exports sheets for:

   - Line loadings (AC/DC)
   - Grid loadings
   - Converter flows
   - Generator dispatch
   - Load profiles
   - Curtailment
   - Price zones
   - Statistics

   **Example**

   .. code-block:: python

       pyf.results_TS_OPF(grid, "results.xlsx", stats=stats_df)
