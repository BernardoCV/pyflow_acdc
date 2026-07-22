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

Usage (via ``pyflow_tests._bess_h2_pei_data``)::

    from pyflow_tests._bess_h2_pei_data import build_pei_bess_h2_grid, PEI_SEASONS

    # Default: Autumn only (24 h)
    grid = build_pei_bess_h2_grid()

    # All four seasons concatenated (96 h)
    grid = build_pei_bess_h2_grid(seasons=PEI_SEASONS)

Season compare (separate 24 h solves, Dash overlay)::

    python my_tests/pei_window_nl_opf_bess_h2.py --compare-seasons --dash
    python my_tests/pei_window_nl_opf_bess_h2.py --compare-seasons --seasons Summer Winter --dash
