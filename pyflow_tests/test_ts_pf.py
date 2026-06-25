# -*- coding: utf-8 -*-
"""Time-series power flow (``time_series_pf``) smoke tests."""

import pandas as pd
import pytest
import pyflow_acdc as pyf

TS_FACTORS = [1.0, 0.9, 0.8, 0.9, 1.0]
N_STEPS = len(TS_FACTORS)


def _factor_ts(values, column_name):
    return pd.DataFrame({column_name: values})


def _assert_ts_pf_results(grid, *, expect_ac=False, expect_dc=False, expect_conv=False):
    assert grid.Time_series_ran is True
    pf = grid.time_series_results["PF_results"]
    assert len(pf) == N_STEPS
    assert pf.index.tolist() == list(range(1, N_STEPS + 1))

    if expect_ac:
        ac_loading = grid.time_series_results["ac_loading"]
        assert len(ac_loading) == N_STEPS
        assert not ac_loading.isna().any().any()

    if expect_dc:
        dc_loading = grid.time_series_results["dc_loading"]
        assert len(dc_loading) == N_STEPS
        assert not dc_loading.isna().any().any()

    if expect_conv:
        conv_loading = grid.time_series_results["converter_loading"]
        assert len(conv_loading) == N_STEPS
        assert not conv_loading.isna().any().any()


def test_time_series_pf_stagg5_acdc_load():
    """Hybrid MATACDC case 5: load-factor time series via ``ts_acdc_pf`` dispatch."""
    grid, _res = pyf.cases["Stagg5MATACDC"]()
    load_node = "4"
    pyf.add_TimeSeries(
        grid,
        _factor_ts(TS_FACTORS, "load_4"),
        associated=load_node,
        TS_type="Load",
    )

    pyf.time_series_pf(grid)

    node = grid.nodes_AC_dict[load_node]
    assert node.PLi_factor == pytest.approx(TS_FACTORS[-1])
    _assert_ts_pf_results(grid, expect_ac=True, expect_dc=True, expect_conv=True)


def test_time_series_pf_case24_ac_load():
    """IEEE RTS-24 (``case24_OPF``): pure AC load-factor time series."""
    grid, _res = pyf.cases["case24_OPF"]()
    load_node = "4"
    pyf.add_TimeSeries(
        grid,
        _factor_ts(TS_FACTORS, "load_4"),
        associated=load_node,
        TS_type="Load",
    )

    pyf.time_series_pf(grid)

    node = grid.nodes_AC_dict[load_node]
    assert node.PLi_factor == pytest.approx(TS_FACTORS[-1])
    _assert_ts_pf_results(grid, expect_ac=True)


def test_time_series_pf_dc_opf_simple_load():
    """Small DC grid (``DC_OPF_simple``): DC load-factor time series."""
    grid, _res = pyf.cases["DC_OPF_simple"]()
    load_node = "Node_2"
    pyf.add_TimeSeries(
        grid,
        _factor_ts(TS_FACTORS, "load_n2"),
        associated=load_node,
        TS_type="Load",
    )

    pyf.time_series_pf(grid)

    node = grid.nodes_DC_dict[load_node]
    assert node.PLi_factor == pytest.approx(TS_FACTORS[-1])
    _assert_ts_pf_results(grid, expect_dc=True)


def run_test():
    """Script entrypoint for ``pyflow-acdc-test``."""
    test_time_series_pf_stagg5_acdc_load()
    test_time_series_pf_case24_ac_load()
    test_time_series_pf_dc_opf_simple_load()
    print("✓ Time-series PF tests passed")


if __name__ == "__main__":
    run_test()
