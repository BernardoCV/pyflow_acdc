"""Docs: api/clustering.rst — Exploratory clustering sweep"""
import tempfile

import pyflow_acdc as pyf
from pyflow_acdc.Time_series_clustering import run_clustering_analysis

grid, _ = pyf.cases["NS_MTDC_2025"](years_data="24", expandable=False, online=False)

with tempfile.TemporaryDirectory() as save_path:
    results = run_clustering_analysis(
        grid,
        save_path=save_path,
        algorithms=["kmeans", "kmeans_medoids", "kmedoids"],
        n_clusters_list=[2, 4],
        time_series=["price", "Load"],
        print_details=False,
        ts_options=[None, 0, 0.8],
        correlation_decisions=[False, "1", False],
        plotting=False,
        identifier="doc_example",
    )

assert len(results) >= 3
assert set(results["algorithm"]) >= {"kmeans", "kmeans_medoids", "kmedoids"}
