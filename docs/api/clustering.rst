Time-Series Clustering
======================

Functions in :mod:`pyflow_acdc.Time_series_clustering`. Representative-period
clustering reduces long time-series inputs to a weighted set of scenarios for
:func:`~pyflow_acdc.multi_scenario_TEP`, :func:`~pyflow_acdc.multi_period_MS_TEP`,
and related TEP drivers.

Workflow guides: :doc:`../usage_tep` (TEP and MS TEP), :doc:`../usage_mp_tep`
(MP TEP and MP+MS TEP).

``clustering_options``
----------------------

TEP functions accept a ``clustering_options`` dict, normally processed by
:func:`~pyflow_acdc.cluster_analysis`:

.. list-table::
   :widths: 28 52
   :header-rows: 1

   * - Key
     - Role
   * - ``n_clusters``
     - Number of representative periods
   * - ``time_series``
     - TS labels to include (e.g. ``["price", "Load", "WPP"]``)
   * - ``central_market``
     - Price-zone names treated as central markets
   * - ``thresholds``
     - ``[cv_threshold, correlation_threshold]``
   * - ``correlation_decisions``
     - ``[clean, method, scale_groups]`` for :func:`~pyflow_acdc.identify_correlations`
   * - ``cluster_algorithm``
     - e.g. ``kmedoids``, ``kmeans_medoids``, ``Kmeans``
   * - ``precomputed_clusters_path``
     - JSON path; skips re-clustering when set (see
       :func:`~pyflow_acdc.load_precomputed_clusters_to_grid`)
   * - ``print_details``
     - Verbose clustering output

Cluster analysis
----------------

.. autofunction:: pyflow_acdc.cluster_analysis

   Main entry used inside TEP when ``clustering_options`` is passed.

.. autofunction:: pyflow_acdc.cluster_TS

.. autofunction:: pyflow_acdc.identify_correlations

Precomputed clusters
--------------------

.. autofunction:: pyflow_acdc.load_precomputed_clusters_to_grid

Exploratory analysis
--------------------

.. autofunction:: pyflow_acdc.run_clustering_analysis_and_plot
