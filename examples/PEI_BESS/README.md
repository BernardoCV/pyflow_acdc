PEI BESS seasonal time-series data
==================================

24 h wind and export-price windows for the Princess Elisabeth Island BESS + H₂
case (Useche-Arteaga et al. 2026). Sourced from Mario’s ``18414805`` seasonal
folders.

Layout::

    PEI_BESS/
      Spring|Summer|Autumn|Winter/
        power_matrix.csv   # 160 turbines × 24 h, MW
        BE_Price.csv       # Belgium PZ b_CG → node.lf → gen.lf (quadratic link)
        GB_Price.csv       # Great Britain PZ b_CG
        DK_Price.csv       # Denmark PZ b_CG

GitHub raw base (``main``)::

    https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/PEI_BESS/

``PEI_grid(..., storage=True, hydrogen=True, data=...)`` loads seasonal or
``Full_data`` series (local ``examples/PEI_BESS`` checkout, else GitHub raw URL).

Usage::

    import pyflow_acdc as pyf

    # Default Autumn 24 h + BESS + H₂
    grid, _ = pyf.cases["PEI_grid"](
        include_countries=["GB", "DK"],
        storage=True,
        hydrogen=True,
        data="season_comparison",
    )

    # All four seasons concatenated (96 h)
    grid, _ = pyf.cases["PEI_grid"](
        include_countries=["GB", "DK"],
        storage=True,
        hydrogen=True,
        data="season_comparison",
        seasons=("Spring", "Summer", "Autumn", "Winter"),
    )

Season-compare + Dash: see docs page ``usage_window_opf`` and
``pyflow_tests/doc_examples/window_opf/01_pei_season_compare_dash.py``.
