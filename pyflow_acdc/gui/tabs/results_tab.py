# -*- coding: utf-8 -*-
"""Tab 3 — Results tables (plots / Dash: placeholders)."""

from __future__ import annotations

import pandas as pd
from PySide6.QtWidgets import (
    QGroupBox,
    QLabel,
    QListWidget,
    QSplitter,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from pyflow_acdc.gui.session import Session
from pyflow_acdc.gui.widgets.pandas_model import PandasTableModel


class ResultsTab(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._model = PandasTableModel()

        self._empty = QLabel("Run a study on the Tests tab to populate results.")
        self._table_list = QListWidget()
        self._table_list.currentTextChanged.connect(self._show_table)

        self._table_view = QTableView()
        self._table_view.setModel(self._model)
        self._table_view.setAlternatingRowColors(True)

        tables_group = QGroupBox("Results tables")
        tables_layout = QVBoxLayout(tables_group)
        tables_layout.addWidget(self._empty)

        splitter = QSplitter()
        splitter.addWidget(self._table_list)
        splitter.addWidget(self._table_view)
        splitter.setStretchFactor(1, 1)
        tables_layout.addWidget(splitter)

        plots_group = QGroupBox("Visualisation")
        plots_layout = QVBoxLayout(plots_group)
        plots_layout.addWidget(
            QLabel("Plotly, Folium, and Open Dash will be added in a later phase.")
        )

        layout = QVBoxLayout(self)
        layout.addWidget(tables_group)
        layout.addWidget(plots_group)

        session.results_changed.connect(self._refresh_tables)

    def _refresh_tables(self) -> None:
        self._table_list.clear()
        results = self._session.results
        if results is None or not results.tables:
            self._empty.setVisible(True)
            self._model.set_dataframe(pd.DataFrame({"message": ["No tables yet."]}))
            return

        self._empty.setVisible(False)
        for key in sorted(results.tables):
            self._table_list.addItem(key)
        self._table_list.setCurrentRow(0)

    def _show_table(self, key: str) -> None:
        if not key or self._session.results is None:
            return
        df = self._session.results.tables.get(key)
        if df is None:
            return
        self._model.set_dataframe(df)
        self._table_view.resizeColumnsToContents()
