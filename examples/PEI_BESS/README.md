PEI BESS seasonal time-series data
==================================

24 h wind and export-price windows for the Princess Elisabeth Island BESS + H₂
case (Useche-Arteaga et al. 2026). Sourced from Mario’s ``18414805`` seasonal
folders.

Layout::

    PEI_BESS/
      Spring|Summer|Autumn|Winter/
        power_matrix.csv   # 160 turbines × 24 h, MW
        BE_Price.csv       # BE_ON, EUR/MWh
        GB_Price.csv       # Na_AC_UK, EUR/MWh
        DK_Price.csv       # Tr_AC_DK, EUR/MWh

<<<<<<< Updated upstream
Usage (via ``pyflow_tests._bess_h2_pei_data``)::
=======
GitHub raw base (``main``)::

    https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/PEI_BESS/

``PEI_grid(..., storage=True, hydrogen=True, data=...)`` loads seasonal or
``Full_data`` series (local ``examples/PEI_BESS`` checkout, else GitHub raw URL).

Usage::
>>>>>>> Stashed changes

    import pyflow_acdc as pyf

    # Default Autumn 24 h + BESS + H₂
    grid, _ = pyf.cases["PEI_grid"](
        include_countries=["GB", "DK"],
        storage=True,
        hydrogen=True,
        data="season_comparison",
    )

    # All four seasons concatenated (96 h)
<<<<<<< Updated upstream
    grid = build_pei_bess_h2_grid(seasons=PEI_SEASONS)
=======
    grid, _ = pyf.cases["PEI_grid"](
        include_countries=["GB", "DK"],
        storage=True,
        hydrogen=True,
        data="season_comparison",
        seasons=("Spring", "Summer", "Autumn", "Winter"),
    )

Season-compare + Dash: see docs page ``usage_window_opf`` and
``pyflow_tests/doc_examples/window_opf/01_pei_season_compare_dash.py``.
>>>>>>> Stashed changes
