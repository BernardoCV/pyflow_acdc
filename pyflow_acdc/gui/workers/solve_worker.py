# -*- coding: utf-8 -*-
"""Background study runners (keep Pyomo/PF off the UI thread)."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from pyflow_acdc.Classes import Grid
from pyflow_acdc.gui.jobs import run_optimal_pf_job, run_power_flow_job


class PowerFlowWorker(QThread):
    finished_ok = Signal(object, object)  # Results, StudyReport
    failed = Signal(str)

    def __init__(self, grid: Grid):
        super().__init__()
        self._grid = grid

    def run(self) -> None:
        try:
            results, report = run_power_flow_job(self._grid)
            self.finished_ok.emit(results, report)
        except Exception:
            self.failed.emit(traceback.format_exc())


class OptimalPfWorker(QThread):
    finished_ok = Signal(object, object)
    failed = Signal(str)

    def __init__(self, grid: Grid, solver: str = "ipopt"):
        super().__init__()
        self._grid = grid
        self._solver = solver

    def run(self) -> None:
        try:
            results, report = run_optimal_pf_job(self._grid, solver=self._solver)
            self.finished_ok.emit(results, report)
        except Exception:
            self.failed.emit(traceback.format_exc())
