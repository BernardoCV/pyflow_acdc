# -*- coding: utf-8 -*-
"""QApplication bootstrap."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pyflow_acdc.gui.main_window import MainWindow
from pyflow_acdc.gui.session import Session


def _load_stylesheet(app: QApplication) -> None:
    qss = Path(__file__).resolve().parent / "assets" / "gui.qss"
    if qss.is_file():
        app.setStyleSheet(qss.read_text(encoding="utf-8"))


def run_app() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("pyflow-acdc")
    _load_stylesheet(app)

    session = Session()
    window = MainWindow(session)
    window.show()

    sys.exit(app.exec())
