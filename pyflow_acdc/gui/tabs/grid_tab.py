# -*- coding: utf-8 -*-
"""Left Grid column — logo, load, inventory, session log."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import pyflow_acdc as pyf

from pyflow_acdc.gui.dialogs.cases_dialog import CasesDialog
from pyflow_acdc.gui.dialogs.code_dialog import PasteCodeDialog
from pyflow_acdc.gui.grid.inventory import format_grid_inventory
from pyflow_acdc.gui.session import Session

_LOGO_LIGHT = (
    Path(__file__).resolve().parents[2] / "assets" / "pyflow_logo_light.svg"
)


def _sidebar_logo() -> QWidget:
    """Light-background logo for the Grid column header."""
    wrap = QWidget()
    layout = QVBoxLayout(wrap)
    layout.setContentsMargins(4, 4, 4, 0)

    if _LOGO_LIGHT.is_file():
        try:
            from PySide6.QtSvgWidgets import QSvgWidget

            logo = QSvgWidget(str(_LOGO_LIGHT))
            logo.setObjectName("sidebarLogo")
            logo.setFixedHeight(56)
            logo.setMinimumWidth(180)
            logo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            layout.addWidget(logo, alignment=Qt.AlignHCenter)
            return wrap
        except ImportError:
            pass

    title = QLabel("<b>pyflow-acdc</b>")
    title.setAlignment(Qt.AlignCenter)
    layout.addWidget(title)
    return wrap


class GridTab(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session

        self._status = QLabel("Status: Idle")

        self._load_pickle_btn = QPushButton("Load pickle…")
        self._load_pickle_btn.clicked.connect(self._load_pickle)

        self._load_cases_btn = QPushButton("Load from cases…")
        self._load_cases_btn.clicked.connect(self._load_from_cases)

        self._load_code_btn = QPushButton("Load from code…")
        self._load_code_btn.clicked.connect(self._load_from_code)

        load_group = QGroupBox("Load")
        load_layout = QVBoxLayout(load_group)
        load_layout.addWidget(self._status)
        load_layout.addWidget(self._load_pickle_btn)
        load_layout.addWidget(self._load_cases_btn)
        load_layout.addWidget(self._load_code_btn)

        size_group = QGroupBox("Grid size")
        size_layout = QVBoxLayout(size_group)
        self._inventory = QTextEdit()
        self._inventory.setReadOnly(True)
        self._inventory.setMinimumHeight(160)
        self._inventory.setPlainText("No grid loaded.")
        size_layout.addWidget(self._inventory)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setPlaceholderText("Solver log and messages…")
        log_layout.addWidget(self._log)

        layout = QVBoxLayout(self)
        layout.addWidget(_sidebar_logo())
        layout.addWidget(load_group)
        layout.addWidget(size_group, 1)
        layout.addWidget(log_group, 1)

        session.grid_changed.connect(self._refresh_inventory)
        session.busy_changed.connect(self._on_busy_changed)
        session.status_changed.connect(self._on_status_changed)
        session.log_message.connect(self._append_log)

    def _append_log(self, text: str) -> None:
        self._log.append(text)

    def _load_buttons(self):
        return (
            self._load_pickle_btn,
            self._load_cases_btn,
            self._load_code_btn,
        )

    def _on_busy_changed(self, busy: bool) -> None:
        for btn in self._load_buttons():
            btn.setEnabled(not busy)

    def _on_status_changed(self, status: str) -> None:
        self._status.setText(f"Status: {status}")

    def _refresh_inventory(self) -> None:
        self._inventory.setPlainText(format_grid_inventory(self._session.grid))

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

    def _load_from_cases(self) -> None:
        if self._session.busy:
            QMessageBox.warning(self, "Busy", f"Wait — {self._session.status}")
            return
        dialog = CasesDialog(self)
        if dialog.exec() != CasesDialog.Accepted or dialog.grid is None:
            return
        if not self._begin_load("Loading case…"):
            return
        try:
            self._session.set_grid(dialog.grid)
            self._end_load("Loaded")
        except Exception as exc:
            self._end_load("Load failed")
            QMessageBox.critical(self, "Load failed", str(exc))

    def _load_from_code(self) -> None:
        if self._session.busy:
            QMessageBox.warning(self, "Busy", f"Wait — {self._session.status}")
            return
        dialog = PasteCodeDialog(self, existing_grid=self._session.grid)
        if dialog.exec() != PasteCodeDialog.Accepted or dialog.grid is None:
            return
        if not self._begin_load("Running pasted code…"):
            return
        try:
            self._session.set_grid(dialog.grid)
            self._end_load("Loaded")
        except Exception as exc:
            self._end_load("Load failed")
            QMessageBox.critical(self, "Load failed", str(exc))
