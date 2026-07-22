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

Usage (via ``pyflow_tests._bess_h2_pei_data``)::

    from pyflow_tests._bess_h2_pei_data import build_pei_bess_h2_grid, PEI_SEASONS

    # Default: Autumn only (24 h)
    grid = build_pei_bess_h2_grid()

    # All four seasons concatenated (96 h)
    grid = build_pei_bess_h2_grid(seasons=PEI_SEASONS)
