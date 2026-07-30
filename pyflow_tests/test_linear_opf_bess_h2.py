# -*- coding: utf-8 -*-
"""Linear AC OPF with AC BESS (P-only) and AC electrolyser."""

import pytest
import pyflow_acdc as pyf

from pyflow_acdc.constants import ObjComponent
from pyflow_tests._test_solver_deps import (
    lopf_solver,
    require_pyomo,
    require_pyomo_mip_css_solvers,
)


def _grid_ac_bess_h2():
    grid, _ = pyf.cases["case39"]()
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
        P_max_MW=150.0,
        P_min_MW=22.5,
        b_h=16.0585,
        c_h=8.2195,
        H2_mass_max_kg=43448.0,
        H2_mass_initial_kg=0.0,
        h2_price=2.0,
    )
    return grid


def test_linear_opf_bess_h2_builds():
    require_pyomo()
    grid = _grid_ac_bess_h2()

    model, _, _, stats = pyf.optimal_l_pf(
        grid,
        ObjRule={"Energy_cost": 1, "H2_sale": 1},
        build_only=True,
    )

    assert stats["termination_condition"] == "build_only"
    assert hasattr(model, "storage_AC")
    assert hasattr(model, "P_storage_charge")
    assert hasattr(model, "SoC")
    assert hasattr(model, "P_storage_AC_net_upper_constraint")
    assert hasattr(model, "Gen_Pstorage_constraint")
    assert not hasattr(model, "Q_storage")

    assert hasattr(model, "electrolyser")
    assert hasattr(model, "P_electrolyser")
    assert hasattr(model, "mass_H2")
    assert hasattr(model, "hydrogen_mass_h2_balance_constraint")
    assert hasattr(model, "Gen_Pelectrolyser_constraint")
    assert not hasattr(model, "Q_electrolyser")

    assert len(model.storage_AC) == 1
    assert len(model.electrolyser) == 1

    st = grid.storage_elements[0]
    assert st.Q == 0.0
    el = grid.electrolysers[0]
    assert el.Q_electrolyser == 0.0


def test_linear_opf_dc_grid_rejected():
    require_pyomo()
    grid, _ = pyf.cases["case39_acdc"]()
    with pytest.raises(ValueError, match="not ready for DC"):
        pyf.optimal_l_pf(grid, ObjRule={"Energy_cost": 1}, build_only=True)


def test_linear_opf_soc_deviation_rejected():
    require_pyomo()
    grid = _grid_ac_bess_h2()
    with pytest.raises(ValueError, match="SoC_deviation"):
        pyf.optimal_l_pf(
            grid,
            ObjRule={
                ObjComponent.ENERGY_COST: 1,
                ObjComponent.SOC_DEVIATION: 1,
            },
            build_only=True,
        )


def test_linear_opf_bess_h2_solves_when_lopf_available():
    require_pyomo_mip_css_solvers()
    grid = _grid_ac_bess_h2()
    model, _, _, stats = pyf.optimal_l_pf(
        grid,
        ObjRule={"Energy_cost": 1, "H2_sale": 1},
        solver=lopf_solver(),
    )
    assert str(stats["termination_condition"]).lower() in (
        "optimal",
        "feasible",
        "locallyoptimal",
    )
    assert hasattr(model, "SoC")
    assert grid.storage_elements[0].SoC is not None
    assert grid.electrolysers[0].mass_H2 is not None
