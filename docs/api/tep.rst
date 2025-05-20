Transmission Expansion Planning Module
====================================

This module provides functions for transmission expansion planning (TEP) analysis of AC/DC hybrid power systems. [1]_

Functions are found in pyflow_acdc.ACDC_TEP

Transmission Expansion Planning
-----------------------------

Running the TEP
^^^^^^^^^^^^^^

.. py:function:: transmission_expansion(grid, NPV=True, n_years=25, Hy=8760, discount_rate=0.02, ObjRule=None, solver='bonmin', initial_guess=0)

   Performs transmission expansion planning analysis.

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
      * - ``NPV``
        - bool
        - Calculate net present value
        - True
      * - ``n_years``
        - int
        - Number of years for NPV calculation
        - 25
      * - ``Hy``
        - int
        - Hours per year
        - 8760
      * - ``discount_rate``
        - float
        - Discount rate for NPV
        - 0.02
      * - ``ObjRule``
        - dict
        - Objective function weights
        - None
      * - ``solver``
        - str
        - Solver to use
        - 'bonmin'
      * - ``initial_guess``
        - int
        - Initial guess for optimization
        - 0

   **Returns**

   Returns a tuple containing:
   
   - Model object
   - Model results
   - Timing information
   - Solver statistics

   **Example**

   .. code-block:: python

       model, results, timing, stats = pyf.transmission_expansion(grid)

Time Series TEP
^^^^^^^^^^^^^^

.. py:function:: transmission_expansion_TS(grid, increase_Pmin=False, NPV=True, n_years=25, Hy=8760, discount_rate=0.02, clustering_options=None, ObjRule=None, solver='bonmin')

   Performs time series transmission expansion planning analysis.

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
      * - ``increase_Pmin``
        - bool
        - Increase minimum power limit
        - False
      * - ``NPV``
        - bool
        - Calculate net present value
        - True
      * - ``n_years``
        - int
        - Number of years for NPV
        - 25
      * - ``Hy``
        - int
        - Hours per year
        - 8760
      * - ``discount_rate``
        - float
        - Discount rate for NPV
        - 0.02
      * - ``clustering_options``
        - dict
        - Time series clustering options
        - None
      * - ``ObjRule``
        - dict
        - Objective function weights
        - None
      * - ``solver``
        - str
        - Solver to use
        - 'bonmin'

   **Returns**

   Returns a tuple containing:
   
   - Model object
   - Model results
   - Timing information
   - Solver statistics
   - TEP time series results

   **Example**

   .. code-block:: python

       model, results, timing, stats, ts_results = pyf.transmission_expansion_TS(grid)

Export Results
^^^^^^^^^^^^^

.. py:function:: export_TEP_TS_results_to_excel(grid, export)

   Exports time series TEP results to Excel file.

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
      * - ``export``
        - str
        - Export file path
        - Required

   **Example**

   .. code-block:: python

       pyf.export_TEP_TS_results_to_excel(grid, "results.xlsx")

References
----------

.. [1] B. C. Valerio, M. Cheah-Mane, V. A. Lacerda, P. Gebraad and O. Gomis-Bellmunt,
       "Transmission expansion planning for hybrid AC/DC grids using a
       Mixed-Integer Non-linear Programming approach"
