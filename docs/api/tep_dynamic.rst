Multi period Transmission Expansion Planning Module
====================================================

This module is under development 


This module provides functions for multi-period transmission expansion planning
with investment states applied over time.

Functions are found in `pyflow_acdc.ACDC_MultiPeriod_TEP`.

Multi-period Multi-scenario Dynamic TEP
---------------------------------------

.. autofunction:: pyflow_acdc.multi_period_transmission_expansion

.. autofunction:: pyflow_acdc.multi_period_MS_TEP

   Solves dynamic transmission expansion planning across investment periods
   using clustered time frames/scenarios.

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
        - Include NPV formulation for operational costs
        - True
      * - ``n_years``
        - int
        - Number of years for discounting
        - 10
      * - ``Hy``
        - int
        - Hours per year
        - 8760
      * - ``discount_rate``
        - float
        - Discount rate
        - 0.02
      * - ``clustering_options``
        - dict
        - Time-series clustering configuration
        - None
      * - ``ObjRule``
        - dict
        - OPF objective weights
        - None
      * - ``solver``
        - str
        - Pyomo solver name
        - 'bonmin'
      * - ``obj_scaling``
        - float
        - Objective scaling factor
        - 1.0

   **Returns**

   - Model object
   - Model results
   - Timing information dictionary
   - Solver statistics dictionary
   - Dynamic TEP time-series results

Export Dynamic Investment Period Plots
--------------------------------------

.. autofunction:: pyflow_acdc.export_and_save_inv_period_svgs

   Exports one SVG network plot per investment period using the solved dynamic
   investment states.

Run OPF on One Investment Period
--------------------------------

.. autofunction:: pyflow_acdc.run_opf_for_investment_period

   Applies one dynamic investment-period state to the grid, runs OPF, and
   optionally exports period results to Excel.

Run OPF on All Investment Periods
---------------------------------

.. autofunction:: pyflow_acdc.run_opf_for_all_investment_periods

   Runs OPF for every dynamic investment period and exports one result file per
   period.