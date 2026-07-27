# -*- coding: utf-8 -*-
"""Tab 1 — load or sample grid (builder / code: placeholders)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import pyflow_acdc as pyf

from pyflow_acdc.gui.session import Session


class GridTab(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session

        self._status = QLabel("Status: Idle")
        self._summary = QLabel("No grid loaded.")
        self._summary.setWordWrap(True)

        self._load_pickle_btn = QPushButton("Load pickle…")
        self._load_pickle_btn.clicked.connect(self._load_pickle)

        self._sample_btn = QPushButton("Load sample case (IEEE PJM 5-bus AC)")
        self._sample_btn.clicked.connect(self._load_sample_case)

        load_group = QGroupBox("Load file")
        load_layout = QVBoxLayout(load_group)
        load_layout.addWidget(self._status)
        load_layout.addWidget(self._load_pickle_btn)
        load_layout.addWidget(self._sample_btn)

        placeholder = QGroupBox("Coming soon")
        placeholder_layout = QVBoxLayout(placeholder)
        placeholder_layout.addWidget(
            QLabel("Interactive add_* builder and paste-code editor will go here.")
        )

        inspector = QGroupBox("Inspector")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.addWidget(self._summary)

        layout = QVBoxLayout(self)
        layout.addWidget(load_group)
        layout.addWidget(placeholder)
        layout.addWidget(inspector)
        layout.addStretch()

        session.grid_changed.connect(self._refresh_summary)
        session.busy_changed.connect(self._on_busy_changed)
        session.status_changed.connect(self._on_status_changed)

    def _on_busy_changed(self, busy: bool) -> None:
        self._load_pickle_btn.setEnabled(not busy)
        self._sample_btn.setEnabled(not busy)

    def _on_status_changed(self, status: str) -> None:
        self._status.setText(f"Status: {status}")

    def _refresh_summary(self) -> None:
        self._summary.setText(self._session.grid_summary())

    def _begin_load(self, status: str) -> bool:
        if self._session.busy:
            QMessageBox.warning(self, "Busy", f"Wait — {self._session.status}")
            return False
        self._session.set_busy(True, status)
        QApplication.processEvents()
        return True

    def _end_load(self, ok_status: str) -> None:
        self._session.set_busy(False, ok_status)

    def _load_pickle(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open grid pickle",
            "",
            "Pickle files (*.pkl *.pickle);;All files (*)",
        )
        if not path:
            return
        if not self._begin_load("Loading pickle…"):
            return
        try:
            grid = pyf.create_grid_from_pickle(path)
            self._session.set_grid(grid)
            self._end_load("Loaded")
        except Exception as exc:
            self._end_load("Load failed")
            QMessageBox.critical(self, "Load failed", str(exc))

    def _load_sample_case(self) -> None:
        if not self._begin_load("Loading sample case…"):
            return
        try:
            grid, _res = pyf.cases["pglib_opf_case5_pjm"]()
            grid.name = "pglib_opf_case5_pjm"
            self._session.set_grid(grid)
            self._end_load("Loaded")
        except Exception as exc:
            self._end_load("Load failed")
            QMessageBox.critical(self, "Load failed", str(exc))
