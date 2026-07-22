# -*- coding: utf-8 -*-
"""PEI 24 h coupled window NL OPF with BESS and electrolyser (build-only)."""

import pyflow_acdc as pyf

from pyflow_tests._bess_h2_pei_data import (
    PEI_OBJ_RULE,
    WINDOW_END,
    WINDOW_START,
    build_pei_bess_h2_grid,
)

grid = build_pei_bess_h2_grid(attach_wind=True)
pyf.window_nl_opf(
    grid,
    start=WINDOW_START,
    end=WINDOW_END,
    ObjRule=PEI_OBJ_RULE,
    build_only=True,
)
