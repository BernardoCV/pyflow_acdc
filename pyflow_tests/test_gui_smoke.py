# -*- coding: utf-8 -*-
"""Smoke tests for the optional desktop GUI (Phase 0 bones)."""

import importlib.util

import pytest

pytestmark = pytest.mark.gui


def test_power_flow_job():
    import pyflow_acdc as pyf
    from pyflow_acdc.gui.jobs import run_power_flow_job

    grid, _ = pyf.cases["CigreB4_ACDC"]()
    results, report = run_power_flow_job(grid)
    assert "AC_Powerflow" in results.tables
    assert not results.tables["AC_Powerflow"].empty
    assert report.kind == "pf_acdc"
    assert report.tracker is not None
    assert report.tracker.get("sequential_iterations")


def test_grid_inventory_and_code_runner():
    import pyflow_acdc as pyf
    from pyflow_acdc.gui.grid.code_runner import run_grid_code
    from pyflow_acdc.gui.grid.inventory import grid_element_counts

    grid, _ = pyf.cases["pglib_opf_case5_pjm"]()
    counts = dict(grid_element_counts(grid))
    assert counts["AC nodes"] == 5
    assert counts["DC nodes"] == 0

    built = run_grid_code(
        'grid, res = pyf.cases["pglib_opf_case5_pjm"]()\ngrid.name = "x"\n'
    )
    assert built.name == "x"
    assert len(built.nodes_AC) == 5


def test_figure_from_results_table():
    import pyflow_acdc as pyf
    from pyflow_acdc.gui.jobs import run_power_flow_job
    from pyflow_acdc.gui.results.plot_builder import (
        figure_from_results_table,
        table_plot_options,
    )
    from pyflow_acdc.gui.studies.solve_report import figure_from_study_report

    grid, _ = pyf.cases["pglib_opf_case5_pjm"]()
    results, report = run_power_flow_job(grid)
    keys = table_plot_options(results)
    assert "AC_Powerflow" in keys
    fig = figure_from_results_table(
        results.tables["AC_Powerflow"],
        columns=["Power Gen (MW)"],
        title="AC_Powerflow",
    )
    assert len(fig.data) == 1
    progress = figure_from_study_report(report)
    assert progress.layout.title.text


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
