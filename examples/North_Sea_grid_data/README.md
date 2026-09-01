North Sea MTDC multi-period TEP example data for `pyf.cases["NS_MTDC_2025"]()`.

Grid topology CSVs, expandable-element tables (``Expandable_elements.csv`` for
MP TEP, ``Expandable_elements_step.csv`` for sequential / MS TEP step mode),
investment-period series, hourly market/load time series (2023–2025), and
precomputed MS clustering (``clusters_kmeans_medoids_k4.json`` and
``clusters_kmeans_medoids_k24.json`` for 2023+2024).

Usage:

```python
import pyflow_acdc as pyf

# Multi-period TEP (default)
grid, res = pyf.cases["NS_MTDC_2025"](years_data="23,24", expandable="mp")

# Multi-scenario / sequential STEP (no MP investment series)
grid, res = pyf.cases["NS_MTDC_2025"](years_data="23,24", expandable="step")
```

Raw GitHub links:

- https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/North_Sea_grid_data/NS_TS_marketPrices_data_sd2024.csv
- https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/North_Sea_grid_data/NS_TS_WL_data2024.csv
- https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/North_Sea_grid_data/clusters_kmeans_medoids_k4.json
- https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/North_Sea_grid_data/clusters_kmeans_medoids_k24.json
