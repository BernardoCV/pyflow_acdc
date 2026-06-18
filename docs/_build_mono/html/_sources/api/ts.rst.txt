Time Series Module
==================

This module provides functions for time series analysis of power flows and optimal power flow.

functions are found in pyflow_acdc.Time_series


Sequential AC/DC Time Series Power Flow
---------------------------------------

Cross-sectional time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.ts_acdc_pf

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

       pyf.ts_acdc_pf(grid, start=1, end=24)

Simple Time-Series Power Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.time_series_pf

Grid Data Update Helper
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.Time_series.update_grid_data

   Internal helper that applies one time-step (or one clustered state) to grid
   data before solving PF/OPF routines.

Optimal Power Flow Time Series
------------------------------

Cross-sectional time series
^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. autofunction:: pyflow_acdc.ts_acdc_opf

   Objective rule (``ObjRule``) — see :ref:`Objective Functions <obj_functions>`.
   ``price_zone_restrictions`` adds price-zone restrictions to the model [1]_.

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

       times=pyf.ts_acdc_opf(grid,start,end,ObjRule=obj)

       res_dict = grid.time_series_results


Data handling
-------------

Statistical Analysis
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.time_series_statistics

   Calculates for each time series:

   - Mean
   - Median
   - Maximum/Minimum
   - Mode
   - IQR
   - Percentiles

Results Export
^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.results_ts_opf

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

       pyf.results_ts_opf(grid, "results", stats=stats_df)

**References**

.. [1] B. C. Valerio, V. A. Lacerda, M. Cheah-Mañe, P. Gebraad, and O. Gomis-Bellmunt,
       "Optimizing offshore wind integration through multi-terminal DC grids: a market-based
       OPF framework for the North Sea interconnectors," IET Conference Proceedings, vol. 2025,
       no. 6, pp. 150–155, 2025. doi: 10.1049/icp.2025.1198

