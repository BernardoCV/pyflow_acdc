Multi period Transmission Expansion Planning Module
====================================================

This module provides functions for multi-period transmission expansion planning
with investment states applied over time.

Functions are found in `pyflow_acdc.ACDC_MultiPeriod_TEP`.

For workflow-oriented notes:

- MS TEP (single investment snapshot): :doc:`../usage_tep`
- MP TEP, sequential STEP, and MP+MS TEP: :doc:`../usage_mp_tep`

Multi-period Multi-scenario Dynamic TEP
---------------------------------------

.. autofunction:: pyflow_acdc.multi_period_transmission_expansion

   Example on ``case24_MP`` (grid + planning CSVs, model build with **ipopt**; use
   **bonmin** and omit ``build_only`` for a full MINLP solve):

   .. literalinclude:: ../../pyflow_tests/doc_examples/tep_mp/01_multi_period_tep_case24.py
      :language: python
      :lines: 2-

.. autofunction:: pyflow_acdc.multi_period_MS_TEP

   Solves dynamic transmission expansion planning across investment periods
   using clustered time frames/scenarios.

   Example on ``NS_MTDC_2025`` (full solve with **ipopt**; use **bonmin** for
   production MINLP with binary expansion):

   .. literalinclude:: ../../pyflow_tests/doc_examples/tep_mp/02_multi_period_multi_scenario_dynamic_tep.py
      :language: python
      :lines: 2-

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
        - OPF objective weights (see :doc:`opf`)
        - None
      * - ``solver``
        - str
        - Pyomo solver name
        - 'bonmin'
      * - ``obj_scaling``
        - float
        - Objective scaling factor
        - 1.0
      * - ``build_only``
        - bool
        - Build model and return without solving or exporting
        - False

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

Run TS-OPF on One Investment Period
-----------------------------------

.. autofunction:: pyflow_acdc.run_ts_opf_for_investment_period

   Applies one MP investment-period state to the grid (or nominal base when
   ``nominal_base=True``), runs :func:`~pyflow_acdc.ts_acdc_opf` over
   ``[start, end]``, and optionally exports time-series OPF tables to Excel via
   :func:`~pyflow_acdc.results_ts_opf`. Use after
   :func:`~pyflow_acdc.multi_period_MS_TEP` or
   :func:`~pyflow_acdc.multi_period_transmission_expansion` when you need
   operating trajectories for a single built-out year.

Export Multi-scenario TEP Time Series
-------------------------------------

.. autofunction:: pyflow_acdc.export_TEP_multiScenario_results_to_excel
   :no-index:

   Excel export for MS and MP+MS scenario tables on ``grid.TEP_multiScenario_res``.

.. function:: pyflow_acdc.export_TEP_TS_results_to_excel
   :no-index:

   **Deprecated alias** for :func:`export_TEP_multiScenario_results_to_excel`.
   Prefer the canonical name in new scripts.