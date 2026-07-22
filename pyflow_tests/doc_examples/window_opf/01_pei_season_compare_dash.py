# -*- coding: utf-8 -*-
"""PEI: solve each season window, attach season-compare, open Dash.

Heavy: four 24 h IPOPT windows. Docs CI does not execute this file; it is the
literalinclude for ``docs/usage_window_opf.rst``. Run interactively, or use
``my_tests/pei_window_nl_opf_bess_h2.py --compare-seasons --dash``.
"""

import pyflow_acdc as pyf
from pyflow_tests._bess_h2_pei_data import (
    PEI_OBJ_RULE,
    PEI_SEASONS,
    WINDOW_START,
    build_pei_bess_h2_grid,
    window_end,
)

if not pyf.is_pyomo_solver_available("ipopt"):
    raise SystemExit(0)

season_results = {}
grid = None
for season in PEI_SEASONS:
    grid = build_pei_bess_h2_grid(seasons=(season,))
    pyf.window_nl_opf(
        grid,
        start=WINDOW_START,
        end=window_end((season,)),
        ObjRule=PEI_OBJ_RULE,
        solver="ipopt",
    )
    season_results[season] = grid.window_opf_results

pyf.attach_season_window_compare(grid, season_results)
grid.name = "PEI BESS H2"
# Interactive browser: pyf.run_dash(grid)
app = pyf.create_season_compare_dash_app(grid)
assert app.layout is not None
