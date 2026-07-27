# -*- coding: utf-8 -*-
"""Dialog: paste Python that builds ``grid``."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
)

from pyflow_acdc.Classes import Grid
from pyflow_acdc.gui.grid.code_runner import run_grid_code

_PLACEHOLDER = '''# Example — assign a Grid to variable "grid"
import pyflow_acdc as pyf

grid, res = pyf.cases["pglib_opf_case5_pjm"]()
grid.name = "from_paste"
'''


class PasteCodeDialog(QDialog):
    def __init__(self, parent=None, existing_grid: Grid | None = None):
        super().__init__(parent)
        self.setWindowTitle("Load grid from code")
        self.resize(640, 480)
        self._existing_grid = existing_grid
        self.grid: Grid | None = None

        self._editor = QPlainTextEdit()
        self._editor.setPlainText(_PLACEHOLDER)
        self._editor.setPlaceholderText(
            'Paste case-style Python. Must assign a Grid to "grid".'
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._run)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(
                "Paste or edit Python below, then OK. "
                "Available: pyf, pd, NodeType, ConverterDCType, Polarity, DataInput."
            )
        )
        layout.addWidget(self._editor, 1)
        layout.addWidget(buttons)

    def _run(self) -> None:
        try:
            self.grid = run_grid_code(
                self._editor.toPlainText(),
                existing_grid=self._existing_grid,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Code failed", str(exc))
            return
        self.accept()
