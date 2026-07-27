# -*- coding: utf-8 -*-
"""Main window: left Grid sidebar + Tests / Results tabs."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pyflow_acdc.gui.session import Session
from pyflow_acdc.gui.tabs.grid_tab import GridTab
from pyflow_acdc.gui.tabs.results_tab import ResultsTab
from pyflow_acdc.gui.tabs.tests_tab import TestsTab

_SIDEBAR_WIDTH = 340


class MainWindow(QMainWindow):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self.setWindowTitle("pyflow-acdc")
        self.resize(1200, 720)

        self._grid_open = True

        self._toggle = QPushButton("Hide grid ◀")
        self._toggle.setObjectName("sidebarToggle")
        self._toggle.setFixedWidth(110)
        self._toggle.clicked.connect(self._toggle_grid)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 6, 8, 4)
        toolbar_layout.addWidget(self._toggle)
        toolbar_layout.addWidget(QLabel("Grid sidebar · Tests / Results on the right"))
        toolbar_layout.addStretch()

        self._grid_panel = QWidget()
        self._grid_panel.setObjectName("gridSidebar")
        self._grid_panel.setMinimumWidth(260)
        grid_layout = QVBoxLayout(self._grid_panel)
        grid_layout.setContentsMargins(8, 4, 8, 8)
        grid_layout.addWidget(QLabel("<b>Grid</b>"))
        grid_layout.addWidget(GridTab(session))

        tabs = QTabWidget()
        tabs.addTab(TestsTab(session), "Tests")
        tabs.addTab(ResultsTab(session), "Results")

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._grid_panel)
        self._splitter.addWidget(tabs)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._splitter.setSizes([_SIDEBAR_WIDTH, 860])

        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        central_layout.addWidget(toolbar)
        central_layout.addWidget(self._splitter, 1)
        self.setCentralWidget(central)

        status = QStatusBar()
        self.setStatusBar(status)
        session.status_changed.connect(status.showMessage)
        session.grid_changed.connect(
            lambda: status.showMessage(f"Loaded — {session.grid_summary()}")
        )
        session.log_message.connect(lambda msg: status.showMessage(msg, 5000))
        status.showMessage("Idle — load a grid to begin.")

    def _toggle_grid(self) -> None:
        self._grid_open = not self._grid_open
        self._grid_panel.setVisible(self._grid_open)
        if self._grid_open:
            self._toggle.setText("Hide grid ◀")
            self._splitter.setSizes([_SIDEBAR_WIDTH, max(400, self.width() - _SIDEBAR_WIDTH)])
        else:
            self._toggle.setText("Show grid ▶")
