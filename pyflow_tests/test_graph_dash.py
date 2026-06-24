# -*- coding: utf-8 -*-
"""Graph_Dash unit tests with synthetic time-series results (no OPF solve, no server)."""

import pandas as pd
import plotly.graph_objects as go
import pytest

import pyflow_acdc as pyf
from pyflow_tests._test_solver_deps import dash_missing_for_run_test, require_dash


def _dash_installed():
    try:
        __import__("dash")
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _dash_installed(),
    reason="dash is not installed",
)

if _dash_installed():
    from pyflow_acdc.Graph_Dash import (
        _MP_PLOT_CHOICES,
        create_dash_app,
        create_mp_ts_dash,
        plot_TS_res_dash,
        plot_TS_res_from_ts,
        run_dash,
    )
else:
    _MP_PLOT_CHOICES = []


def _synthetic_ts_results(n=6):
    """Minimal ``time_series_results`` dict covering all Dash plot types."""
    idx = pd.RangeIndex(0, n)
    return {
        "curtailment": pd.DataFrame({"rs1": [0.1, 0.2, 0.15, 0.05, 0.0, 0.1]}, index=idx),
        "net_price_zone_power": pd.DataFrame(
            {"pz1": [10.0] * n, "o_aux": [0.0] * n}, index=idx
        ),
        "grid_loading": pd.DataFrame({"load": [0.5, 0.6, 0.55, 0.7, 0.65, 0.6]}, index=idx),
        "real_load_opf": pd.DataFrame({"n1": [-0.2, -0.25, -0.22, -0.3, -0.28, -0.26]}, index=idx),
        "real_load_known_by_zone": pd.DataFrame({"z1": [0.3] * n}, index=idx),
        "PN_min": pd.DataFrame({"pz1": [1.0, 2.0, 1.5, 2.5, 2.0, 1.8]}, index=idx),
        "PN_max": pd.DataFrame({"pz1": [5.0, 6.0, 5.5, 7.0, 6.5, 6.0]}, index=idx),
        "real_power_opf": pd.DataFrame({"g1": [0.4, 0.5, 0.45, 0.55, 0.5, 0.48]}, index=idx),
        "real_power_by_zone": pd.DataFrame({"z1": [0.4, 0.5, 0.45, 0.55, 0.5, 0.48]}, index=idx),
        "prices_by_zone": pd.DataFrame(
            {"pz1": [50.0, 55.0, 52.0, 58.0, 54.0, 51.0], "o_hidden": [0.0] * n},
            index=idx,
        ),
        "ac_loading": pd.DataFrame({"l1": [0.6, 0.7, 0.65, 0.75, 0.7, 0.68]}, index=idx),
        "dc_loading": pd.DataFrame({"l2": [0.5, 0.55, 0.52, 0.6, 0.58, 0.56]}, index=idx),
        "converter_loading": pd.DataFrame({"c1": [0.8, 0.85, 0.82, 0.9, 0.88, 0.86]}, index=idx),
    }


def _grid_with_ts_results():
    grid = pyf.Grid(S_base=100)
    grid.name = "dash_test"
    grid.time_series_results = _synthetic_ts_results()
    grid.Time_series_ran = True
    return grid


def _ts_inv_snapshot():
    return {"time_series_results": _synthetic_ts_results(), "S_base": 100.0}


def _cols_for_plot_choice(plotting_choice, ts):
    if plotting_choice in (
        "Power Generation by generator",
        "Power Generation by generator area chart",
    ):
        return list(ts["real_power_opf"].columns)
    if plotting_choice in (
        "Power Generation by price zone",
        "Power Generation by price zone area chart",
    ):
        return list(ts["real_power_by_zone"].columns)
    if plotting_choice in ("Market Prices", "PN"):
        return ["pz1"]
    if plotting_choice == "PN_min":
        return list(ts["PN_min"].columns)
    if plotting_choice == "PN_max":
        return list(ts["PN_max"].columns)
    if plotting_choice == "Grid loading":
        return list(ts["grid_loading"].columns)
    if plotting_choice == "Real load":
        return list(ts["real_load_opf"].columns)
    if plotting_choice == "Known load by zone":
        return list(ts["real_load_known_by_zone"].columns)
    if plotting_choice == "AC line loading":
        return list(ts["ac_loading"].columns)
    if plotting_choice == "DC line loading":
        return list(ts["dc_loading"].columns)
    if plotting_choice == "AC/DC Converters":
        return list(ts["converter_loading"].columns)
    if plotting_choice == "Curtailment":
        return list(ts["curtailment"].columns)
    return []


@pytest.mark.parametrize("plotting_choice", _MP_PLOT_CHOICES)
def test_plot_TS_res_from_ts_all_choices(plotting_choice):
    ts = _synthetic_ts_results()
    cols = _cols_for_plot_choice(plotting_choice, ts)

    fig = plot_TS_res_from_ts(
        ts,
        S_base=100.0,
        plotting_choice=plotting_choice,
        selected_rows=cols,
        x_limits=(0, 5),
        y_limits=(0, 100),
    )
    assert isinstance(fig, go.Figure)
    assert len(fig.data) >= 1


def test_plot_TS_res_from_ts_empty_and_unknown():
    ts = _synthetic_ts_results()
    empty_fig = plot_TS_res_from_ts(
        ts, 100.0, "unknown plot", selected_rows=[],
    )
    assert isinstance(empty_fig, go.Figure)
    assert len(empty_fig.data) == 0

    missing_pn = dict(ts)
    missing_pn.pop("net_price_zone_power")
    fig = plot_TS_res_from_ts(
        missing_pn, 100.0, "PN", selected_rows=["pz1"],
    )
    assert len(fig.data) == 0


def test_plot_TS_res_dash_wraps_grid():
    grid = _grid_with_ts_results()
    fig = plot_TS_res_dash(
        grid,
        "Power Generation by generator",
        selected_rows=["g1"],
        x_limits=(0, 5),
        y_limits=(0, 50),
    )
    assert len(fig.data) == 1


def test_create_dash_app_layout():
    app = create_dash_app(_grid_with_ts_results())
    assert app.layout is not None


def test_create_mp_ts_dash_layout_and_errors():
    ts_inv = {"base": _ts_inv_snapshot(), 1: _ts_inv_snapshot()}
    app = create_mp_ts_dash(ts_inv, grid_name="MP test")
    assert app.layout is not None

    with pytest.raises(ValueError, match="ts_inv is empty"):
        create_mp_ts_dash({})
    with pytest.raises(ValueError, match="no period keys"):
        create_mp_ts_dash({"bad": 1})


def test_run_dash_routing_errors():
    grid = pyf.Grid(S_base=100)
    with pytest.raises(ValueError, match="run_dash \\(auto\\)"):
        run_dash(grid)

    grid.dash_mode = "mp_ts"
    with pytest.raises(ValueError, match="dash_mode=mp_ts"):
        run_dash(grid)


def _callback_fn(app, key):
    """Return the undecorated Dash callback (Dash 3 wraps with context)."""
    return app.callback_map[key]["callback"].__wrapped__


def test_create_dash_app_callbacks_fire():
    """Invoke registered callbacks without starting a server."""
    grid = _grid_with_ts_results()
    app = create_dash_app(grid)

    toggle = _callback_fn(
        app, "..plot-2-controls.style...plot-2-container.style..",
    )
    style_controls, style_container = toggle(True)
    assert style_controls["display"] == "block"
    style_controls, style_container = toggle(False)
    assert style_controls["display"] == "none"

    update_opts = _callback_fn(
        app,
        "..subplot-selection-1.options...subplot-selection-1.value..."
        "subplot-selection-2.options...subplot-selection-2.value..",
    )
    opts1, vals1, opts2, vals2 = update_opts(
        "Power Generation by price zone",
        "Market Prices",
    )
    assert vals1 == ["z1"]
    assert vals2 == ["pz1"]

    update_graphs = _callback_fn(
        app, "..plot-output-1.figure...plot-output-2.figure..",
    )
    fig1, fig2 = update_graphs(
        "Power Generation by generator",
        "Market Prices",
        ["g1"],
        ["pz1"],
        0,
        5,
        0,
        50,
        0,
        100,
        False,
    )
    assert len(fig1.data) >= 1
    assert len(fig2.data) == 0


def test_create_mp_ts_dash_callbacks_fire():
    ts_inv = {"base": _ts_inv_snapshot(), 1: _ts_inv_snapshot(), 2: _ts_inv_snapshot()}
    app = create_mp_ts_dash(ts_inv, grid_name="MP callbacks")

    toggle_mode = _callback_fn(
        app, "..mp-compare-row.style...mp-single-row.style..",
    )
    compare_row, single_row = toggle_mode("compare")
    assert compare_row["display"] == "block"
    compare_row, single_row = toggle_mode("single")
    assert single_row["display"] == "block"

    update_fig = _callback_fn(app, "..mp-graph.figure...mp-graph-2.figure..")
    fig1, fig2 = update_fig(
        "single",
        "base",
        "base",
        "base",
        2,
        "Power Generation by price zone",
        ["z1"],
        False,
        "Market Prices",
        ["pz1"],
        0,
        5,
        0,
        100,
    )
    assert len(fig1.data) >= 1
    assert len(fig2.data) == 0

    fig_cmp, fig2_cmp = update_fig(
        "compare",
        "base",
        "base",
        1,
        2,
        "Power Generation by generator",
        ["g1"],
        False,
        "Market Prices",
        [],
        None,
        None,
        None,
        None,
    )
    assert len(fig_cmp.data) >= 1

    fig1, fig2 = update_fig(
        "single",
        "base",
        "base",
        "base",
        2,
        "Power Generation by price zone",
        ["z1"],
        True,
        "Curtailment",
        ["rs1"],
        0,
        5,
        0,
        100,
    )
    assert len(fig1.data) >= 1
    assert len(fig2.data) >= 1


def run_test():
    if dash_missing_for_run_test():
        return
    require_dash()
    test_plot_TS_res_from_ts_all_choices("Market Prices")
    test_plot_TS_res_from_ts_empty_and_unknown()
    test_plot_TS_res_dash_wraps_grid()
    test_create_dash_app_layout()
    test_create_mp_ts_dash_layout_and_errors()
    test_run_dash_routing_errors()
    test_create_dash_app_callbacks_fire()
    test_create_mp_ts_dash_callbacks_fire()
    print("✓ graph_dash tests passed")


if __name__ == "__main__":
    run_test()
