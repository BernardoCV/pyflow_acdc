"""Docs: api/clustering.rst — Precomputed clusters"""
import pyflow_acdc as pyf
from pyflow_tests.test_constants import north_sea_ms_clustering_options

grid, _ = pyf.cases["NS_MTDC_2025"](years_data="23,24", expandable=False, online=False)
n_clusters, clustered = pyf.cluster_analysis(grid, north_sea_ms_clustering_options())

assert clustered is True
assert n_clusters == 4
assert 4 in grid.Clusters
