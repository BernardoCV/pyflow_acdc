# -*- coding: utf-8 -*-
"""Launch Dash in a background thread (full interactive UI in browser)."""

from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal

from pyflow_acdc.Classes import Grid


class DashLaunchWorker(QThread):
    failed = Signal(str)
    started_ok = Signal(str)

    def __init__(self, grid: Grid, host: str = "127.0.0.1", port: int = 8050):
        super().__init__()
        self._grid = grid
        self._host = host
        self._port = port

    def run(self) -> None:
        try:
            from pyflow_acdc.Graph_Dash import run_dash

            self.started_ok.emit(f"http://{self._host}:{self._port}")
            run_dash(self._grid, debug=False, use_reloader=False)
        except Exception:
            self.failed.emit(traceback.format_exc())
