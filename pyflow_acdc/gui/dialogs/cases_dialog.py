# -*- coding: utf-8 -*-
"""Dialog: pick a bundled case from ``pyf.cases``."""

from __future__ import annotations

import inspect
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QVBoxLayout,
)

import pyflow_acdc as pyf
from pyflow_acdc.Classes import Grid


def _case_folder(name: str) -> str:
    factory = pyf.cases.get(name)
    if factory is None:
        return "?"
    try:
        path = Path(inspect.getfile(factory)).resolve()
    except (TypeError, OSError):
        return "?"
    parts = path.parts
    for folder in ("PF", "OPF", "TEP", "Wind_Array"):
        if folder in parts:
            return folder
    return path.parent.name


class CasesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load from cases")
        self.resize(520, 520)
        self.grid: Grid | None = None

        self._filter = QLineEdit()
        self._filter.setPlaceholderText("Filter cases…")
        self._filter.textChanged.connect(self._apply_filter)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(lambda _item: self._load())

        self._detail = QLabel("Select a case.")
        self._detail.setWordWrap(True)
        self._list.currentItemChanged.connect(self._on_selection)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._load)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Bundled example grids (pyf.cases):"))
        layout.addWidget(self._filter)
        layout.addWidget(self._list, 1)
        layout.addWidget(self._detail)
        layout.addWidget(buttons)

        self._all_names = sorted(pyf.cases.keys())
        self._apply_filter("")

    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()
        self._list.clear()
        for name in self._all_names:
            if needle and needle not in name.lower():
                continue
            folder = _case_folder(name)
            item = QListWidgetItem(f"{name}  [{folder}]")
            item.setData(Qt.UserRole, name)
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)

    def _on_selection(self, current: QListWidgetItem | None, _previous) -> None:
        if current is None:
            self._detail.setText("Select a case.")
            return
        name = current.data(Qt.UserRole)
        factory = pyf.cases[name]
        try:
            path = inspect.getfile(factory)
        except (TypeError, OSError):
            path = "(unknown)"
        self._detail.setText(f"{name}\nFile: {path}")

    def _load(self) -> None:
        item = self._list.currentItem()
        if item is None:
            QMessageBox.warning(self, "No selection", "Select a case first.")
            return
        name = item.data(Qt.UserRole)
        try:
            result = pyf.cases[name]()
            if isinstance(result, tuple):
                grid = result[0]
            else:
                grid = result
            if not isinstance(grid, Grid):
                raise TypeError(f"{name}() did not return a Grid")
            if not getattr(grid, "name", None):
                grid.name = name
            self.grid = grid
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))
            return
        self.accept()
