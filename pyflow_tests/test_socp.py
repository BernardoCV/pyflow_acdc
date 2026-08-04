# -*- coding: utf-8 -*-
"""Sparse SOCP smoke and parity checks."""

import pandas as pd
import pytest
import pyflow_acdc as pyf

from pyflow_tests._test_solver_deps import require_ipopt, require_socp, socp_solver

SOCP_ENERGY = {"Energy_cost": {"w": 1}}
SOCP_ENERGY_H2 = {"Energy_cost": {"w": 1}, "H2_sale": {"w": 1}}


def _case39_socp_grid():
    return pyf.cases["case39_acdc"]()[0]


def _case39_socp_grid_with_flex_and_ts(n_frames=3):
    grid = _case39_socp_grid()
    pyf.add_storage(
        grid,
        "30",
        E_max_MWh=100.0,
        P_charge_MW=33.0,
        P_discharge_MW=33.0,
        eta_charge=0.85,
        eta_discharge=0.90,
        soc_initial=0.5,
    )
    pyf.add_electrolyser(
        grid,
        "30",
        P_max_MW=20.0,
        P_min_MW=0.0,
        b_h=0.02,
        c_h=0.0,
        H2_mass_max_kg=1000.0,
        H2_mass_initial_kg=0.0,
        h2_price=2.0,
    )
    pyf.add_TimeSeries(
        grid,
        pd.DataFrame({"load": [1.0, 0.92, 1.05][:n_frames]}),
        associated="30",
        TS_type="Load",
    )
    return grid


def test_socp_builds_case39_acdc():
    require_socp()
    grid = _case39_socp_grid()
    problem, variables, timing, stats = pyf.socp_optimise(
        grid,
        build_only=True,
        weights_def=SOCP_ENERGY,
    )

    assert problem is not None
    assert variables.ac is not None
    assert variables.dc is not None
    assert variables.conv is not None
    assert stats["n_vars"] > 0
    assert stats["n_constr"] > 0
    assert timing["build"] >= 0


def test_socp_solves_case39_acdc():
    require_socp()
    grid = _case39_socp_grid()
    _, _, _, stats = pyf.socp_optimise(
        grid,
        solver=socp_solver(),
        weights_def=SOCP_ENERGY,
    )

    assert stats["status"] in ("optimal", "optimal_inaccurate")
    assert stats["solver"] == socp_solver()
    assert grid.socp_run is True
    assert grid.nodes_AC[0].V_AC is not None
    assert grid.Converters_ACDC[0].P_loss >= 0


def test_soc_window_optimisation_solves_case39_with_bess_h2():
    require_socp()
    grid = _case39_socp_grid_with_flex_and_ts()
    _, variables, _, stats = pyf.soc_window_optimisation(
        grid,
        frame_ids=[0, 1, 2],
        solver=socp_solver(),
        weights_def=SOCP_ENERGY_H2,
    )

    assert stats["status"] in ("optimal", "optimal_inaccurate")
    assert variables.storage is not None
    assert variables.hydrogen is not None
    assert grid.storage_elements[0].SoC is not None
    assert grid.electrolysers[0].mass_H2 >= 0


def test_pei_soc_window_builds_short_horizon():
    require_socp()
    grid, _ = pyf.cases["PEI_grid"](
        include_countries=["GB", "DK"],
        storage=True,
        hydrogen=True,
        data="season_comparison",
    )
    problem, variables, _, stats = pyf.soc_window_optimisation(
        grid,
        frame_ids=[0, 1, 2],
        build_only=True,
        weights_def=SOCP_ENERGY,
    )

    assert problem is not None
    assert variables.storage is not None
    assert variables.hydrogen is not None
    assert stats["n_vars"] > 0
    assert stats["n_constr"] > 0


@pytest.mark.slow
def test_socp_and_nl_case39_voltage_are_close_when_ipopt_available():
    require_socp()
    require_ipopt()

    grid_socp = _case39_socp_grid()
    grid_nl = _case39_socp_grid()

    pyf.socp_optimise(grid_socp, solver=socp_solver(), weights_def=SOCP_ENERGY)
    pyf.optimal_pf(grid_nl, ObjRule={"Energy_cost": 1}, solver="ipopt", tee=False)

    node_name = "30"
    v_socp = next(n.V_AC for n in grid_socp.nodes_AC if n.name == node_name)
    v_nl = next(n.V for n in grid_nl.nodes_AC if n.name == node_name)

    assert abs(v_socp - v_nl) < 0.08
