# -*- coding: utf-8 -*-
"""Tab 2 — run studies; solve-progress plot on the right."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyflow_acdc.gui.results.plotly_view import PlotlyView
from pyflow_acdc.gui.session import Session
from pyflow_acdc.gui.studies.solve_report import figure_from_study_report
from pyflow_acdc.gui.workers.solve_worker import OptimalPfWorker, PowerFlowWorker


class TestsTab(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._worker = None

        self._status = QLabel("Status: Idle")
        self._hint = QLabel("Load a grid in the left sidebar first.")
        self._run_pf = QPushButton("Run power flow")
        self._run_pf.clicked.connect(self._run_power_flow)
        self._run_pf.setEnabled(False)

        self._run_opf = QPushButton("Run OPF (ipopt)")
        self._run_opf.clicked.connect(self._run_opf_job)
        self._run_opf.setEnabled(False)
        self._run_opf.setToolTip("Needs pyomo + ipopt. Shows feasibility / NLP progress.")

        run_group = QGroupBox("Studies")
        run_layout = QVBoxLayout(run_group)
        run_layout.addWidget(self._status)
        run_layout.addWidget(self._hint)
        run_layout.addWidget(self._run_pf)
        run_layout.addWidget(self._run_opf)
        run_layout.addWidget(QLabel("Window OPF, rolling, TEP — planned."))
        run_layout.addStretch()

        self._plot = PlotlyView(empty_message="Solver progress appears here after a run.")
        plot_group = QGroupBox("Solve progress")
        plot_layout = QVBoxLayout(plot_group)
        plot_layout.addWidget(self._plot)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(run_group)
        splitter.addWidget(plot_group)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 700])

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

        session.grid_changed.connect(self._on_grid_changed)
        session.busy_changed.connect(self._on_busy_changed)
        session.status_changed.connect(self._on_status_changed)
        session.study_changed.connect(self._refresh_plot)

        self._detect_opf()
        self._refresh_plot()

    def _log(self, text: str) -> None:
        self._session.log_message.emit(text)

    def _detect_opf(self) -> None:
        try:
            import pyomo  # noqa: F401

            self._opf_available = True
        except ImportError:
            self._opf_available = False
            self._run_opf.setToolTip("Install pyflow_acdc[OPF] (pyomo) + ipopt to enable.")

    def _on_status_changed(self, status: str) -> None:
        self._status.setText(f"Status: {status}")

    def _on_busy_changed(self, busy: bool) -> None:
        self._update_run_enabled()

    def _on_grid_changed(self) -> None:
        has_grid = self._session.grid is not None
        self._hint.setVisible(not has_grid)
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        ready = self._session.grid is not None and not self._session.busy
        self._run_pf.setEnabled(ready)
        self._run_opf.setEnabled(ready and self._opf_available)

    def _refresh_plot(self) -> None:
        fig = figure_from_study_report(self._session.last_study)
        self._plot.set_figure(fig, open_browser_if_needed=False)

    def _start_worker(self, worker) -> None:
        self._worker = worker
        worker.finished_ok.connect(self._on_study_ok)
        worker.failed.connect(self._on_study_failed)
        worker.finished.connect(self._on_worker_finished)
        worker.start()

    def _run_power_flow(self) -> None:
        if self._session.grid is None:
            return
        if self._session.busy:
            QMessageBox.warning(self, "Busy", f"Wait — {self._session.status}")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A study is already running.")
            return

        self._session.set_busy(True, "Running power flow…")
        self._log("--- power_flow started ---")
        self._start_worker(PowerFlowWorker(self._session.grid))

    def _run_opf_job(self) -> None:
        if self._session.grid is None:
            return
        if self._session.busy:
            QMessageBox.warning(self, "Busy", f"Wait — {self._session.status}")
            return
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(self, "Busy", "A study is already running.")
            return

        self._session.set_busy(True, "Running OPF…")
        self._log("--- optimal_pf (ipopt) started ---")
        self._start_worker(OptimalPfWorker(self._session.grid, solver="ipopt"))

    def _on_study_ok(self, results, report) -> None:
        self._session.set_results(results, study=report)
        for line in report.summary_lines():
            self._log(line)
        if report.log.strip():
            self._log(report.log.strip())
        self._log("--- study finished OK ---")
        self._session.set_busy(False, "Done")

    def _on_study_failed(self, tb: str) -> None:
        self._log(tb)
        self._session.set_busy(False, "Failed")
        QMessageBox.critical(self, "Study failed", "See log for details.")

    def _on_worker_finished(self) -> None:
        self._update_run_enabled()
