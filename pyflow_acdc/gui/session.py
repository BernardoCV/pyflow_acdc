# -*- coding: utf-8 -*-
"""In-memory GUI session: one Grid and optional Results."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from pyflow_acdc.Classes import Grid
from pyflow_acdc.Results_class import Results


class Session(QObject):
    """Shared state across Grid / Tests / Results tabs."""

    grid_changed = Signal()
    results_changed = Signal()
    log_message = Signal(str)
    status_changed = Signal(str)
    busy_changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.grid: Grid | None = None
        self.results: Results | None = None
        self.busy = False
        self.status = "Idle"

    def set_busy(self, busy: bool, status: str) -> None:
        self.busy = busy
        self.status = status
        self.status_changed.emit(status)
        self.busy_changed.emit(busy)

    def set_grid(self, grid: Grid) -> None:
        self.grid = grid
        self.results = None
        self.grid_changed.emit()
        self.results_changed.emit()
        self.log_message.emit(self.grid_summary())

    def set_results(self, results: Results) -> None:
        self.results = results
        self.results_changed.emit()

    def grid_summary(self) -> str:
        if self.grid is None:
            return "No grid loaded."
        g = self.grid
        name = getattr(g, "name", None) or "(unnamed)"
        return (
            f"Grid '{name}': {len(g.nodes_AC)} AC nodes, {len(g.nodes_DC)} DC nodes, "
            f"{len(g.lines_AC)} AC lines, {len(g.lines_DC)} DC lines, "
            f"{len(g.Time_series)} time series"
        )
