# -*- coding: utf-8 -*-
"""Fast multi-period multi-scenario TEP tests on the NS_MTDC_2025 example grid."""

import pytest
import pyflow_acdc as pyf
from pyflow_tests.test_constants import north_sea_ms_clustering_options

NS_MP_MS_OBJ_RULE = {
    "Energy_cost": 0,
    "PZ_cost_of_generation": 1,
    "Renewable_profit": 0,
}


def _ns_mtdc_mp_grid():
    grid, res = pyf.cases["NS_MTDC_2025"](
        years_data="23,24",
        tee=False,
        expandable="mp",
        online=False,
    )
    return grid, res


def test_ns_mtdc_2025_case_is_registered():
    assert "NS_MTDC_2025" in pyf.cases
    grid, res = pyf.cases["NS_MTDC_2025"](years_data="24", expandable=False, online=False)
    assert len(grid.nodes_AC) > 0
    assert len(grid.nodes_DC) > 0
    assert res is not None


def test_ns_mtdc_mp_grid_has_investment_load_series():
    grid, _ = _ns_mtdc_mp_grid()
    load_series = grid.Price_Zones[0].investment_decisions.get("Load", [])
    assert len(load_series) >= 1
    assert len(grid.Time_series) >= 1


def test_ns_mtdc_multi_period_ms_tep_build_only():
    pytest.importorskip("pyomo")

    grid, _ = _ns_mtdc_mp_grid()
    mp_load_series = list(grid.Price_Zones[0].investment_decisions.get("Load", []))

    model, model_results, timing_info, solver_stats, mp_ms_results = pyf.multi_period_MS_TEP(
        grid,
        inv_periods=mp_load_series,
        NPV=True,
        n_years=10,
        Hy=8760,
        discount_rate=0.02,
        clustering_options=north_sea_ms_clustering_options(),
        ObjRule=NS_MP_MS_OBJ_RULE,
        solver="bonmin",
        tee=False,
        obj_scaling=1e10,
        save_period_svgs=False,
        build_only=True,
    )

    assert hasattr(model, "inv_periods")
    assert len(model.inv_periods) >= 1
    assert model_results is None
    assert mp_ms_results == {}
    assert timing_info["create"] >= 0
    assert timing_info["solve"] is None
    assert solver_stats["termination_condition"] == "build_only"


def test_ns_mtdc_sequential_ms_step_orchestration_fake_solve(monkeypatch):
    pytest.importorskip("pyomo")

    solve_calls = []

    def _fake_solve(*args, **kwargs):
        solve_calls.append(1)
        return None, {
            "solution_found": False,
            "termination_condition": "unknown",
            "solver_message": "mocked in test",
            "time": 0.0,
        }

    monkeypatch.setattr("pyflow_acdc.ACDC_Static_TEP.pyomo_model_solve", _fake_solve)

    grid, _ = _ns_mtdc_mp_grid()
    run_results = pyf.sequential_MS_STEP(
        grid=grid,
        n_years=10,
        Hy=8760,
        discount_rate=0.02,
        clustering_options=north_sea_ms_clustering_options(),
        ObjRule=NS_MP_MS_OBJ_RULE,
        solver="bonmin",
        tee=False,
        obj_scaling=1e10,
        save_svgs=False,
        export_steps=False,
    )

    assert len(solve_calls) == 1
    assert run_results["_meta"]["aborted"] is True
    assert "no feasible solution" in run_results["_meta"]["abort_reason"]


def run_test():
    exit_code = pytest.main([__file__, "-q"])
    if exit_code == 0:
        print("✓ NS_MTDC MP+MS multi-period tests passed")
    else:
        print("✗ NS_MTDC MP+MS multi-period tests failed")


if __name__ == "__main__":
    run_test()
