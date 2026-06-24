"""Docs: api/clustering.rst — Live clustering"""
import pyflow_acdc as pyf

grid, _ = pyf.cases["NS_MTDC_2025"](years_data="24", expandable=False, online=False)
clustering_options = {
    "n_clusters": 2,
    "time_series": ["price", "Load"],
    "central_market": [],
    "thresholds": [0, 0.8],
    "correlation_decisions": [False, "1", False],
    "cluster_algorithm": "Kmeans",
    "print_details": False,
}
n_clusters, clustered = pyf.cluster_analysis(grid, clustering_options)

assert clustered is True
assert n_clusters == 2
assert 2 in grid.Clusters
