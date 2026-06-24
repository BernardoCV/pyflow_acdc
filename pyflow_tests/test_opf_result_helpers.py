# -*- coding: utf-8 -*-
"""OPF post-solve result helpers (public API not called from res.all())."""

import pyflow_acdc as pyf

from pyflow_tests._test_solver_deps import pyomo_missing_for_run_test, require_pyomo


def _solve_case39_acdc():
    grid, _ = pyf.cases["case39_acdc"]()
    model, model_results, _, solver_stats = pyf.optimal_pf(
        grid,
        ObjRule={"Energy_cost": 1},
    )
    assert model_results is not None
    assert solver_stats.get("solution_found") is not False
    return grid, model


def test_opf_line_res_case39_acdc():
    require_pyomo()

    grid, model = _solve_case39_acdc()
    line_res, grid_res = pyf.opf_line_res(model, grid)

    assert len(line_res) > 0
    assert any(key.startswith("AC_Load_") for key in line_res)
    assert "Total" in grid_res


def test_opf_step_results_case39_acdc():
    require_pyomo()

    grid, model = _solve_case39_acdc()
    (
        conv_dc,
        conv_ac,
        conv_q_ac,
        p_load,
        p_ext,
        q_ext,
        curtailment,
        loading_conv,
    ) = pyf.opf_step_results(model, grid)

    assert len(p_load) > 0
    assert grid.ACmode and grid.DCmode
    assert len(conv_dc) == len(grid.Converters_ACDC)
    assert len(loading_conv) == len(grid.Converters_ACDC)


def test_opf_price_zone_case24_3zones():
    require_pyomo()

    grid, _ = pyf.cases["case24_3zones_acdc"]()
    assert len(grid.Price_Zones) >= 1

    model, model_results, _, solver_stats = pyf.optimal_pf(
        grid,
        ObjRule={"PZ_cost_of_generation": 1},
    )
    assert model_results is not None
    assert solver_stats.get("solution_found") is not False
    assert hasattr(model, "price_zone_price")

    pz_prices = pyf.opf_price_price_zone(model, grid)
    assert len(pz_prices) == len(grid.Price_Zones)
    for pz in grid.Price_Zones:
        assert pz.name in pz_prices


def run_test():
    if pyomo_missing_for_run_test():
        return
    test_opf_line_res_case39_acdc()
    test_opf_step_results_case39_acdc()
    test_opf_price_zone_case24_3zones()
    print("✓ OPF result helper tests passed")


if __name__ == "__main__":
    run_test()
