Mapping
=======

For this module, you need to have the optional dependency ``pyflow-acdc[mapping]`` installed.

Interactive map
---------------

.. autofunction:: pyflow_acdc.plot_folium

Network map (static topology)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.plot_folium_network

.. themed-figure:: north_sea_folium
   :alt: Example of the Folium map.
   :align: center
   :width: 80%
   
Time-series and investment maps (under development)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These helpers read solved results already stored on the ``Grid`` and build
animated Folium maps. The API may change in future releases.

.. autofunction:: pyflow_acdc.plot_folium_ts_results

   Line-loading animation from ``grid.time_series_results`` (after
   :func:`~pyflow_acdc.ts_acdc_opf` or :func:`~pyflow_acdc.run_ts_opf_for_investment_period`).

.. autofunction:: pyflow_acdc.plot_folium_inv_results

   Expansion/decommission overlay from ``grid.MP_TEP_results``,
   ``grid.Seq_STEP_results``, or ``grid.Seq_MS_STEP_results`` (``source="auto"``
   picks the first non-empty table).


