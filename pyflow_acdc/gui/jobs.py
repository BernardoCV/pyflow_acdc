# -*- coding: utf-8 -*-
"""Study jobs callable without Qt (unit-tested)."""

from __future__ import annotations

import contextlib
import io

from pyflow_acdc.ACDC_PF import (
    PF_SEQ_TOL_FACTOR,
    ac_power_flow,
    acdc_sequential,
    dc_power_flow,
)
from pyflow_acdc.Classes import Grid
from pyflow_acdc.Results_class import Results
from pyflow_acdc.constants import DEFAULT_PF_MAX_ITER, DEFAULT_TOLERANCE
from pyflow_acdc.grid_analysis import analyse_grid
from pyflow_acdc.gui.studies.solve_report import StudyReport


def _capture_io(fn):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        result = fn()
    return result, buf.getvalue()


def run_power_flow_job(grid: Grid) -> tuple[Results, StudyReport]:
    """Run PF, build Results tables, and return a StudyReport for progress plots."""

    def _solve():
        analyse_grid(grid)
        grid.iter_flow_AC = []
        if grid.ACmode and grid.DCmode:
            t, tracker, _ps = acdc_sequential(
                grid,
                tol_lim=DEFAULT_TOLERANCE * PF_SEQ_TOL_FACTOR,
                internal_tol=DEFAULT_TOLERANCE,
                maxIter=DEFAULT_PF_MAX_ITER,
                Droop_PF=True,
            )
            return StudyReport(
                kind="pf_acdc",
                elapsed_s=t,
                final_tol=tracker.get("final_sequential_tolerance"),
                tracker=tracker,
                ac_iters=list(grid.iter_flow_AC),
            )
        if grid.ACmode:
            t, tol = ac_power_flow(grid, DEFAULT_TOLERANCE, DEFAULT_PF_MAX_ITER)
            return StudyReport(
                kind="pf_ac",
                elapsed_s=t,
                final_tol=tol,
                ac_iters=list(grid.iter_flow_AC),
            )
        if grid.DCmode:
            t, tol = dc_power_flow(
                grid, DEFAULT_TOLERANCE, DEFAULT_PF_MAX_ITER, Droop_PF=True
            )
            return StudyReport(kind="pf_dc", elapsed_s=t, final_tol=tol)
        raise RuntimeError("Grid has neither AC nor DC mode set")

    report, log = _capture_io(_solve)
    report.log = log
    results = Results(grid, save_res=False)
    results.all(print_table=False)
    return results, report


def run_optimal_pf_job(grid: Grid, solver: str = "ipopt") -> tuple[Results, StudyReport]:
    """Run nonlinear OPF and return Results + feasibility StudyReport."""

    def _solve():
        from pyflow_acdc.ACDC_OPF import optimal_pf

        _model, _model_res, timing_info, solver_stats = optimal_pf(
            grid,
            solver=solver,
            tee=False,
            callback=False,
        )
        return StudyReport(
            kind="opf",
            elapsed_s=(timing_info or {}).get("solve"),
            solver_stats=solver_stats,
            timing_info=timing_info,
        )

    report, log = _capture_io(_solve)
    report.log = log
    results = Results(grid, save_res=False)
    results.all(print_table=False)
    return results, report
