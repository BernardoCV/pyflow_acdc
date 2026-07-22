# -*- coding: utf-8 -*-
"""Docs smoke for window OPF page (CSV load + example present).

Does not run the heavy four-season IPOPT example; that file is for
literalinclude / interactive use only.
"""

from pathlib import Path

from pyflow_tests._bess_h2_pei_data import PEI_SEASONS, build_pei_bess_h2_grid
from pyflow_tests._test_solver_deps import (
    dash_missing_for_run_test,
    require_dash,
)

EXAMPLE = (
    Path(__file__).resolve().parent
    / "doc_examples"
    / "window_opf"
    / "01_pei_season_compare_dash.py"
)


def test_docs_window_opf():
    require_dash()
    run_test()


def run_test():
    if dash_missing_for_run_test():
        return
    if not EXAMPLE.is_file():
        raise FileNotFoundError(f"Missing window OPF doc example: {EXAMPLE}")
    for season in PEI_SEASONS:
        build_pei_bess_h2_grid(seasons=(season,))
    print("OK window_opf doc smoke passed")


if __name__ == "__main__":
    run_test()
