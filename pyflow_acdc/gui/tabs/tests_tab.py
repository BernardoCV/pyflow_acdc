# -*- coding: utf-8 -*-
"""Tab 2 — run studies on the current grid."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pyflow_acdc.gui.session import Session
from pyflow_acdc.gui.workers.solve_worker import PowerFlowWorker


class TestsTab(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._worker: PowerFlowWorker | None = None

        self._status = QLabel("Status: Idle")
        self._hint = QLabel("Load a grid on the Grid tab first.")
        self._run_pf = QPushButton("Run power flow")
        self._run_pf.clicked.connect(self._run_power_flow)
        self._run_pf.setEnabled(False)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setPlaceholderText("Solver log and messages…")

        run_group = QGroupBox("Studies")
        run_layout = QVBoxLayout(run_group)
        run_layout.addWidget(self._status)
        run_layout.addWidget(self._hint)
        run_layout.addWidget(self._run_pf)
        run_layout.addWidget(QLabel("Window OPF, rolling, TEP — planned."))

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self._log)

        layout = QVBoxLayout(self)
        layout.addWidget(run_group)
        layout.addWidget(log_group)

        session.grid_changed.connect(self._on_grid_changed)
        session.log_message.connect(self._append_log)
        session.busy_changed.connect(self._on_busy_changed)
        session.status_changed.connect(self._on_status_changed)

    def _on_status_changed(self, status: str) -> None:
        self._status.setText(f"Status: {status}")

    def _on_busy_changed(self, busy: bool) -> None:
        self._update_run_enabled()

    def _on_grid_changed(self) -> None:
        has_grid = self._session.grid is not None
        self._hint.setVisible(not has_grid)
        self._update_run_enabled()

    def _update_run_enabled(self) -> None:
        self._run_pf.setEnabled(
            self._session.grid is not None and not self._session.busy
        )

    def _append_log(self, text: str) -> None:
        self._log.append(text)

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
        self._log.append("--- power_flow started ---")

        self._worker = PowerFlowWorker(self._session.grid)
        self._worker.finished_ok.connect(self._on_pf_ok)
        self._worker.failed.connect(self._on_pf_failed)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _on_pf_ok(self, results) -> None:
        self._session.set_results(results)
        self._log.append("--- power_flow finished OK ---")
        self._session.set_busy(False, "Done")

    def _on_pf_failed(self, tb: str) -> None:
        self._log.append(tb)
        self._session.set_busy(False, "Failed")
        QMessageBox.critical(self, "Power flow failed", "See log for details.")

    def _on_worker_finished(self) -> None:
        # Busy flag already cleared in ok/failed handlers; keep button sync.
        self._update_run_enabled()
