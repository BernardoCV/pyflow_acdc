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
:func:`~pyflow_acdc.cluster_analysis`. The multi-scenario TEP doc example
(:file:`pyflow_tests/doc_examples/tep/02_multi_scenario_tep.py`) uses:

.. code-block:: python

   clustering_options = {
       "n_clusters": 4,
       "precomputed_clusters_path": "examples/North_Sea_grid_data/clusters_kmeans_medoids_k4.json",
   }

Use with ``years_data="23,24"`` on ``NS_MTDC_2025`` so time-series length matches the
JSON payload.

.. list-table::
   :widths: 28 52
   :header-rows: 1

   * - Key
     - Role
   * - ``n_clusters``
     - Number of representative periods (scenarios) passed to the clustering
       algorithm.
   * - ``time_series``
     - TS **types** to include (e.g. ``["price", "Load", "WPP"]``). Series on
       the grid whose ``type`` is not listed are dropped before clustering.
   * - ``central_market``
     - List of price-zone names whose attached series are kept for market-linked
       types (``price``, ``Load``, ``PGL_min``, ``PGL_max``, ``a_CG``, ``b_CG``,
       ``c_CG``). An **empty list** (``[]``) keeps all zones in the grid — this
       is the usual choice on multi-zone cases such as ``NS_MTDC_2025``. To
       cluster only around selected hubs, pass e.g. ``["NL", "DE"]``; series
       tied to other zones are then excluded.
   * - ``thresholds``
     - Two-element list ``[cv_threshold, correlation_threshold]`` used in
       :func:`~pyflow_acdc.filter_data` and
       :func:`~pyflow_acdc.identify_correlations``. With ``[0, 0.8]`` (typical):
       no CV pre-filtering (``0`` disables it), and pairs with
       ``|correlation| > 0.8`` form correlated groups. When ``cv_threshold > 0``,
       series whose coefficient of variation **exceeds** that value are removed
       before clustering.
   * - ``correlation_decisions``
     - Three-element list ``[clean, method, scale_groups]`` for
       :func:`~pyflow_acdc.identify_correlations``. ``[True, 3, True]`` means:
       reduce redundant series in each correlated group (**clean**),
       use **method** ``3`` (PCA representative: keep the member most aligned
       with the group's first principal component), and **scale** the kept series
       by ``sqrt(group size)`` so merged information is not under-weighted.
       Set ``clean`` to ``False`` to skip correlation reduction. Methods ``1``
       and ``2`` keep the highest-variance member or replace the group with a
       single PC1 column respectively (see
       :func:`~pyflow_acdc.identify_correlations`).
   * - ``cluster_algorithm``
     - e.g. ``kmedoids``, ``kmeans_medoids``, ``Kmeans``
   * - ``precomputed_clusters_path``
     - JSON path; skips re-clustering when set (see
       :func:`~pyflow_acdc.load_precomputed_clusters_to_grid`)
   * - ``print_details``
     - When ``True``, print filtering statistics (mean, std, CV per series),
       excluded columns, correlated groups, deduplication choices, and
       clustering diagnostics to stdout. Use ``False`` in batch tests to keep
       output quiet.

The four keys highlighted in the doc example work together as a preprocessing
pipeline before the chosen ``cluster_algorithm`` runs: restrict which series
enter the feature matrix (``time_series``, ``central_market``), optionally drop
high-CV or highly correlated columns (``thresholds``,
``correlation_decisions``), then form ``n_clusters`` representative periods.
Set ``print_details=True`` while tuning a case; set it ``False`` once options
are fixed.

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

.. autofunction:: pyflow_acdc.run_clustering_analysis

   Sweeps clustering algorithms and cluster counts on the attached time series,
   records quality metrics (coefficient of variation, inertia, Davies–Bouldin),
   and writes ``clustering_summary_<identifier>.csv`` under ``save_path``. Set
   ``plotting=True`` to save representative-period plots while sweeping. Use
   this to tune ``clustering_options`` before calling TEP; production solves
   normally use :func:`~pyflow_acdc.cluster_analysis` inside the TEP drivers.

.. autofunction:: pyflow_acdc.run_clustering_analysis_and_plot
