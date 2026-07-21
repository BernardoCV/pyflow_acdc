# -*- coding: utf-8 -*-
"""Snapshot NL OPF with AC and DC BESS (Phase 2)."""

import pytest
import pyflow_acdc as pyf

from pyflow_tests._test_solver_deps import require_pyomo


def _grid_with_storage():
    grid, _ = pyf.cases["case39_acdc"]()
    pyf.add_storage(
        grid,
        "30",
        E_max_MWh=100.0,
        P_charge_MW=33.0,
        P_discharge_MW=33.0,
        eta_charge=0.85,
        eta_discharge=0.90,
        soc_initial=0.5,
        soc_final=0.5,
    )
    dc_node = next(n for n in grid.nodes_DC if n.name == "1")
    pyf.add_storage(
        grid,
        dc_node,
        E_max_MWh=50.0,
        P_charge_MW=10.0,
        P_discharge_MW=10.0,
        eta_charge=0.9,
        eta_discharge=0.95,
        soc_initial=0.5,
    )
    return grid


def test_storage_nl_model_builds():
    require_pyomo()
    grid = _grid_with_storage()
    pyf.analyse_grid(grid)
    assert grid.ESS is True

    model, _, _, _ = pyf.optimal_pf(
        grid,
        ObjRule={"Energy_cost": 1},
        build_only=True,
    )

    assert hasattr(model, "storage_AC")
    assert hasattr(model, "storage_DC")
    assert hasattr(model, "P_storage_charge")
    assert hasattr(model, "P_storage_charge_DC")
    assert hasattr(model, "S_storage_AC_limit_constraint")
    assert hasattr(model, "P_storage_DC_net_upper_constraint")
    assert hasattr(model, "Gen_Pstorage_constraint")
    assert hasattr(model, "Gen_Pstorage_DC_constraint")
    assert len(model.storage_AC) == 1
    assert len(model.storage_DC) == 1


def test_ext_storage_reporting():
    require_pyomo()
    grid = _grid_with_storage()
    pyf.optimal_pf(grid, ObjRule={"Energy_cost": 1}, build_only=True)

    res = pyf.Results(grid)
    df = res.ext_storage(print_table=False)

    assert "Ext_storage" in res.tables
    assert len(df) == 3  # AC + DC + Total
    assert set(df.loc[df["Name"] != "Total", "Side"]) == {"AC", "DC"}
    assert "P charge (MW)" in df.columns
    assert "SoC (pu)" in df.columns
    ac_row = df[df["Side"] == "AC"].iloc[0]
    dc_row = df[df["Side"] == "DC"].iloc[0]
    assert ac_row["Q (MVAR)"] != "----"
    assert dc_row["Q (MVAR)"] == "----"


def test_storage_opf_solves_when_ipopt_available():
    require_pyomo()
    if not pyf.is_pyomo_solver_available("ipopt"):
        return

    grid = _grid_with_storage()
    model, model_res, _, solver_stats = pyf.optimal_pf(
        grid,
        ObjRule={"Energy_cost": 1},
        solver="ipopt",
        tee=False,
    )
    assert model_res is not None
    assert solver_stats.get("solution_found") is not False

    ac_storage = next(s for s in grid.storage_elements if s.connected.value == "AC")
    dc_storage = next(s for s in grid.storage_elements if s.connected.value == "DC")
    assert ac_storage.SoC == pytest.approx(ac_storage.soc_final, rel=0, abs=1e-4)
    tol = 1e-5
    assert -tol <= ac_storage.P_charge <= ac_storage.P_charge_max + tol
    assert -tol <= ac_storage.P_discharge <= ac_storage.P_discharge_max + tol
    assert -tol <= dc_storage.P_charge <= dc_storage.P_charge_max + tol
    assert -tol <= dc_storage.P_discharge <= dc_storage.P_discharge_max + tol
    assert hasattr(ac_storage, "Q")
    assert model.obj is not None


def run_test():
    test_storage_nl_model_builds()
    test_ext_storage_reporting()
    test_storage_opf_solves_when_ipopt_available()
    print("OK test_storage_opf")


if __name__ == "__main__":
    run_test()
