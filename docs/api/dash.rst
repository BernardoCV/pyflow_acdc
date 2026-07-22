Dashboards
===========

For the use of this module, you need to have the optional dependendency pyflow_acdc[Dash] installed.

Interactive Dashboard
^^^^^^^^^^^^^^^^^^^^^^

For now you have to run the time series **or** a window NL OPF to have results
to plot. Then run the dashboard. Once the dashboard is running, you can see the
plots by selecting the desired plot type. And choose axis limits to zoom in or out.


.. autofunction:: pyflow_acdc.run_dash

.. autofunction:: pyflow_acdc.run_ts_dash

.. autofunction:: pyflow_acdc.run_window_dash

.. autofunction:: pyflow_acdc.run_mp_ts_dash

Plot Helpers
^^^^^^^^^^^^

Low-level Plotly figure builders used by the Dash apps (also usable standalone).

.. autofunction:: pyflow_acdc.plot_TS_res_from_ts

.. autofunction:: pyflow_acdc.plot_TS_res_dash

.. autofunction:: pyflow_acdc.plot_window_res_dash

Multi-Period Dash Builder
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.create_mp_ts_dash

   Creates and runs an interactive Dash web application for visualizing time series results.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid with time series results
        - Required

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

 


Once the dashboard is created, you can see it in your browser under the url:

.. code-block:: bash

   http://127.0.0.1:8050/



.. literalinclude:: ../../pyflow_tests/doc_examples/dash/01_multi_period_dash_builder.py
   :language: python
   :lines: 2-


.. themed-figure:: dash_example
   :width: 100%
   :alt: Dash Example

   Example of the Dash dashboard.


