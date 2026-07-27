# -*- coding: utf-8 -*-
"""Smoke tests for the optional desktop GUI (Phase 0 bones)."""

import importlib.util

import pytest

pytestmark = pytest.mark.gui


def test_power_flow_job():
    import pyflow_acdc as pyf
    from pyflow_acdc.gui.jobs import run_power_flow_job

    grid, _ = pyf.cases["CigreB4_ACDC"]()
    results = run_power_flow_job(grid)
    assert "AC_Powerflow" in results.tables
    assert not results.tables["AC_Powerflow"].empty


@pytest.mark.skipif(
    importlib.util.find_spec("PySide6") is None,
    reason="PySide6 not installed",
)
def test_gui_import_and_session():
    try:
        from PySide6.QtCore import QObject  # noqa: F401
    except ImportError:
        pytest.skip("PySide6 installed but Qt runtime failed to load")

    from pyflow_acdc.gui import HAS_GUI, launch
    from pyflow_acdc.gui.session import Session

    assert HAS_GUI is True
    assert callable(launch)

    session = Session()
    assert session.grid is None
