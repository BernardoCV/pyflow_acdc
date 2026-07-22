# -*- coding: utf-8 -*-
"""Phase 6: PEI 24 h window_nl_opf with coupled BESS and electrolyser.

Uses the full ``PEI_grid`` with BE, UK, and DK export links. Validates model
assembly, Table 1 asset parameters, and (when IPOPT is available) a coupled solve.
"""

import pytest
import pyflow_acdc as pyf

from pyflow_tests._bess_h2_pei_data import (
    BESS_E_MAX_MWH,
    BESS_P_NOM_MW,
    EXPORT_PRICE_NODES,
    HUB_NODE,
    PEI_OBJ_RULE,
    WINDOW_END,
    WINDOW_START,
    build_pei_bess_h2_grid,
    h2_mass_final_kg,
    load_pei_export_prices,
    load_pei_power_matrix,
)
from pyflow_tests._test_solver_deps import require_pyomo


def test_pei_power_matrix_available():
    assert load_pei_power_matrix().shape == (160, 24)


def test_pei_export_prices_available():
    prices = load_pei_export_prices()
    assert set(prices) == set(EXPORT_PRICE_NODES)
    for node, series in prices.items():
        assert series.shape == (24,), node


def test_pei_bess_h2_grid_assets():
    grid = build_pei_bess_h2_grid(attach_wind=False)
    pyf.analyse_grid(grid)

    assert grid.ESS is True
    assert grid.H2 is True
    assert len(grid.storage_elements) == 1
    assert len(grid.electrolysers) == 1
    price_ts = {
        ts.element_name
        for ts in grid.Time_series
        if ts.type == "price"
    }
    for node_name in EXPORT_PRICE_NODES:
        assert node_name in price_ts
        ext = next(
            g for g in grid.Generators
            if getattr(g, "is_ext_grid", False) and g._node.name == node_name
        )
        assert ext.price_link is True
    bess = grid.storage_elements[0]
    assert bess.Node == HUB_NODE
    assert bess.E_max == pytest.approx(BESS_E_MAX_MWH)
    assert bess.P_charge_max * grid.S_base == pytest.approx(BESS_P_NOM_MW)
    assert bess.P_discharge_max * grid.S_base == pytest.approx(BESS_P_NOM_MW)
    assert grid.electrolysers[0].Node == HUB_NODE
    assert grid.electrolysers[0].H2_mass_final == pytest.approx(
        h2_mass_final_kg(), rel=1e-6
    )


def test_pei_window_nl_opf_bess_h2_builds():
    require_pyomo()
    grid = build_pei_bess_h2_grid(attach_wind=True)
    assert len(grid.Time_series) == 163

    model, _, _, _ = pyf.window_nl_opf(
        grid,
        start=WINDOW_START,
        end=WINDOW_END,
        ObjRule=PEI_OBJ_RULE,
        build_only=True,
    )

    assert hasattr(model, "window_soc_constraint")
    assert hasattr(model, "window_h2_constraint")
    assert len(model.frames) == 24

    block0 = model.frame_model[WINDOW_START]
    assert hasattr(block0, "storage_AC")
    assert hasattr(block0, "electrolyser")
    assert len(block0.storage_AC) == 1
    assert len(block0.electrolyser) == 1


@pytest.mark.slow
def test_pei_window_nl_opf_bess_h2_solves_when_ipopt_available():
    require_pyomo()
    if not pyf.is_pyomo_solver_available("ipopt"):
        pytest.skip("IPOPT not available")

    grid = build_pei_bess_h2_grid(attach_wind=True)
    model, _, _, stats = pyf.window_nl_opf(
        grid,
        start=WINDOW_START,
        end=WINDOW_END,
        ObjRule=PEI_OBJ_RULE,
        solver="ipopt",
        tee=False,
    )

    assert stats is not None
    assert hasattr(model, "obj")
