Time Series Clustering
=====================

This page has been pre-filled with the functions that are available in the Time Series Clustering module by AI, please check the code for more details.

This module provides functions for clustering time series data using various algorithms [2]_.

functions are found in pyflow_acdc.Time_series_clustering

Main Clustering Function
----------------------

.. py:function:: cluster_TS(grid, n_clusters, time_series=[], central_market=[], algorithm='Kmeans', cv_threshold=0, correlation_threshold=0.8, print_details=False, corrolation_decisions=[])

   Main function for clustering time series data using various algorithms.

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
        - Grid with time series
        - Required
        - -
      * - ``n_clusters``
        - int
        - Number of clusters
        - Required
        - -
      * - ``time_series``
        - list
        - Time series to cluster
        - []
        - -
      * - ``algorithm``
        - str
        - Clustering algorithm
        - 'Kmeans'
        - -
      * - ``cv_threshold``
        - float
        - Coefficient of variation threshold
        - 0
        - -
      * - ``correlation_threshold``
        - float
        - Correlation threshold
        - 0.8
        - -

   **Available Algorithms**:

   - kmeans
   - ward
   - dbscan
   - optics
   - kmedoids
   - spectral
   - hdbscan

   **Example**

   .. code-block:: python

       n_clusters, clusters, metrics, labels = pyf.cluster_TS(grid, n_clusters=4, algorithm='kmeans')

Clustering Analysis
-----------------

.. py:function:: run_clustering_analysis(grid, save_path='clustering_results', algorithms=['kmeans', 'kmedoids', 'ward', 'dbscan', 'hdbscan'], n_clusters_list=[1, 4, 8, 16, 24, 48], time_series=[], print_details=False)

   Runs comprehensive clustering analysis using multiple algorithms and cluster numbers.

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
        - Grid with time series
        - Required
        - -
      * - ``save_path``
        - str
        - Path to save results
        - 'clustering_results'
        - -
      * - ``algorithms``
        - list
        - Algorithms to test
        - ['kmeans',...]
        - -
      * - ``n_clusters_list``
        - list
        - Numbers of clusters to test
        - [1,4,8,...]
        - -

   Computes metrics including:

   - Computation time
   - Coefficient of variation
   - Inertia (for k-means)
   - Silhouette score
   - Dunn index
   - Davies-Bouldin index

Visualization
------------

.. py:function:: plot_clustering_results(df=None, results_path='clustering_results', format='svg')

   Creates visualization plots for clustering results.

   Generates plots for:

   - Time comparison
   - Coefficient of variation
   - Inertia (k-means and k-medoids)
   - Silhouette score
   - Dunn index
   - Davies-Bouldin index

Time Series Relationships
-----------------------

.. py:function:: Time_series_cluster_relationship(grid, ts1_name=None, ts2_name=None, price_zone=None, ts_type=None, algorithm='kmeans', take_into_account_time_series=[], number_of_clusters=2, path='clustering_results', format='svg', print_details=False)

   Analyzes and visualizes relationships between time series using clustering.

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
        - Grid with time series
        - Required
        - -
      * - ``ts1_name``
        - str
        - First time series name
        - None
        - -
      * - ``ts2_name``
        - str
        - Second time series name
        - None
        - -
      * - ``ts_type``
        - str
        - Type of time series to analyze
        - None
        - -
      * - ``algorithm``
        - str
        - Clustering algorithm
        - 'kmeans'
        - -
      * - ``number_of_clusters``
        - int
        - Number of clusters
        - 2
        - -

   **Example**

   .. code-block:: python

       pyf.Time_series_cluster_relationship(grid, ts_type='Load', number_of_clusters=4)

   Creates scatter plots showing relationships between time series colored by cluster. 