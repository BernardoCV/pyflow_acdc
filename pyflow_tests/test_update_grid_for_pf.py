# -*- coding: utf-8 -*-
"""Tests for update_grid_for_pf converter setpoint wiring."""

import pandas as pd
import pytest
import pyflow_acdc as pyf

from pyflow_acdc.constants import ConverterDCType, NodeType, TSType
from pyflow_acdc.Classes import TimeSeries


def _pu_ts(values, column_name):
    return pd.DataFrame({column_name: values})


def test_known_converter_pf_setpoints_by_type():
    grid, _ = pyf.cases["CigreB4_ACDC"]()
    assert grid.Converters_ACDC

    for conv in grid.Converters_ACDC:
        known = pyf.known_converter_pf_setpoints(conv)
        if conv.type in (ConverterDCType.P, ConverterDCType.DROOP):
            assert "P_DC" in known
            assert "P_AC" not in known
        elif conv.type == ConverterDCType.PAC:
            assert "P_AC" in known
            assert "P_DC" not in known
        elif conv.type == ConverterDCType.SLACK:
            assert "P_DC" not in known
            assert "P_AC" not in known
        if conv.AC_type == NodeType.PQ:
            assert "Q_AC" in known
        else:
            assert "Q_AC" not in known


def test_update_grid_for_pf_sets_known_p_dc_and_rejects_unknown():
    grid, _ = pyf.cases["CigreB4_ACDC"]()
    conv = next(
        c for c in grid.Converters_ACDC
        if c.type in (ConverterDCType.P, ConverterDCType.DROOP)
    )
    series = _pu_ts([0.1, 0.2, 0.3], "p_dc")
    pyf.add_TimeSeries(
        grid,
        series,
        associated=conv.name,
        TS_type=TSType.CONV_P_DC,
    )
    ts = grid.Time_series[-1]

    pyf.update_grid_for_pf(grid, ts, 1)
    assert conv.P_DC == pytest.approx(0.2)
    assert conv.Node_DC.Pconv == pytest.approx(0.2)

    if "P_AC" not in pyf.known_converter_pf_setpoints(conv):
        bad = TimeSeries(
            TSType.CONV_P_AC.value,
            conv.name,
            series["p_dc"].to_numpy(),
            name="bad_pac",
        )
        with pytest.raises(ValueError, match="P_AC"):
            pyf.update_grid_for_pf(grid, bad, 0)


def test_update_grid_for_pf_sets_q_ac_for_pq():
    grid, _ = pyf.cases["CigreB4_ACDC"]()
    conv = next(c for c in grid.Converters_ACDC if c.AC_type == NodeType.PQ)
    series = _pu_ts([-0.05, -0.1], "q_ac")
    pyf.add_TimeSeries(
        grid,
        series,
        associated=conv.name,
        TS_type="conv_Q_AC",
    )
    ts = grid.Time_series[-1]
    pyf.update_grid_for_pf(grid, ts, 1)
    assert conv.Q_AC == pytest.approx(-0.1)
