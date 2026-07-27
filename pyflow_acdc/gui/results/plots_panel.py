# -*- coding: utf-8 -*-
"""Interactive Plotly panel (Results tab)."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyflow_acdc.gui.results.dash_launcher import DashLaunchWorker
from pyflow_acdc.gui.results.plot_builder import (
    available_ts_plot_choices,
    dash_usable,
    figure_from_results_table,
    figure_from_ts_choice,
    table_plot_options,
)
from pyflow_acdc.gui.results.plotly_view import PlotlyView
from pyflow_acdc.gui.session import Session


class PlotsPanel(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._dash_worker: DashLaunchWorker | None = None

        self._mode = QComboBox()
        self._mode.addItem("Results table", "table")
        self._mode.addItem("Time series (Dash helpers)", "ts")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)

        self._preset = QComboBox()
        self._preset.addItem("Custom columns", "custom")
        self._preset.addItem("AC voltages (all nodes)", "ac_voltage")
        self._preset.addItem("AC powerflow injections", "ac_powerflow")
        self._preset.currentIndexChanged.connect(self._on_preset_changed)

        self._table_combo = QComboBox()
        self._table_combo.currentTextChanged.connect(self._on_table_changed)

        self._columns = QListWidget()
        self._columns.setSelectionMode(QListWidget.MultiSelection)
        self._columns.setMinimumHeight(100)

        self._ts_choice = QComboBox()
        self._ts_choice.currentTextChanged.connect(self._on_ts_choice_changed)
        self._ts_series = QListWidget()
        self._ts_series.setSelectionMode(QListWidget.MultiSelection)
        self._ts_series.setMinimumHeight(100)

        self._plot_btn = QPushButton("Update plot")
        self._plot_btn.clicked.connect(self._update_plot)

        self._browser_btn = QPushButton("Open plot in browser")
        self._browser_btn.clicked.connect(self._open_browser)

        self._dash_btn = QPushButton("Open full Dash…")
        self._dash_btn.clicked.connect(self._open_dash)
        self._dash_btn.setEnabled(False)

        controls = QGroupBox("Plot controls")
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(QLabel("Source"))
        controls_layout.addWidget(self._mode)
        controls_layout.addWidget(QLabel("Quick plot (Results)"))
        controls_layout.addWidget(self._preset)
        controls_layout.addWidget(QLabel("Results table"))
        controls_layout.addWidget(self._table_combo)
        controls_layout.addWidget(QLabel("Columns (multi-select)"))
        controls_layout.addWidget(self._columns)
        controls_layout.addWidget(QLabel("Time-series metric"))
        controls_layout.addWidget(self._ts_choice)
        controls_layout.addWidget(QLabel("Series (multi-select)"))
        controls_layout.addWidget(self._ts_series)
        row = QHBoxLayout()
        row.addWidget(self._plot_btn)
        row.addWidget(self._browser_btn)
        controls_layout.addLayout(row)
        controls_layout.addWidget(self._dash_btn)

        self._view = PlotlyView()

        layout = QHBoxLayout(self)
        layout.addWidget(controls, 0)
        layout.addWidget(self._view, 1)

        session.results_changed.connect(self.refresh)
        session.grid_changed.connect(self.refresh)
        self._on_mode_changed()
        self.refresh()

    def refresh(self) -> None:
        tables = table_plot_options(self._session.results)
        current = self._table_combo.currentText()
        self._table_combo.blockSignals(True)
        self._table_combo.clear()
        self._table_combo.addItems(tables)
        if current in tables:
            self._table_combo.setCurrentText(current)
        self._table_combo.blockSignals(False)
        self._on_table_changed(self._table_combo.currentText())

        grid = self._session.grid
        ts_choices = available_ts_plot_choices(grid) if grid is not None else []
        cur_ts = self._ts_choice.currentText()
        self._ts_choice.blockSignals(True)
        self._ts_choice.clear()
        self._ts_choice.addItems(ts_choices)
        if cur_ts in ts_choices:
            self._ts_choice.setCurrentText(cur_ts)
        self._ts_choice.blockSignals(False)
        self._on_ts_choice_changed(self._ts_choice.currentText())

        self._dash_btn.setEnabled(dash_usable(grid))
        self._on_mode_changed()

    def _on_mode_changed(self) -> None:
        mode = self._mode.currentData()
        is_table = mode == "table"
        self._preset.setEnabled(is_table)
        self._table_combo.setEnabled(is_table)
        self._columns.setEnabled(is_table and self._preset.currentData() == "custom")
        self._ts_choice.setEnabled(not is_table)
        self._ts_series.setEnabled(not is_table)

    def _on_preset_changed(self) -> None:
        preset = self._preset.currentData()
        tables = [self._table_combo.itemText(i) for i in range(self._table_combo.count())]
        self._table_combo.blockSignals(True)
        if preset == "ac_voltage" and "AC_voltage" in tables:
            self._table_combo.setCurrentText("AC_voltage")
        elif preset == "ac_powerflow" and "AC_Powerflow" in tables:
            self._table_combo.setCurrentText("AC_Powerflow")
        self._table_combo.blockSignals(False)
        self._fill_columns_for_current_table()
        if preset == "ac_voltage":
            self._select_columns(["Voltage (pu)", "Voltage angle (deg)"])
        elif preset == "ac_powerflow":
            self._select_columns(
                [
                    "Power Gen (MW)",
                    "Power Load (MW)",
                    "Power injected  (MW)",
                ]
            )
        elif preset == "custom":
            self._columns.selectAll()
        self._columns.setEnabled(preset == "custom" and self._mode.currentData() == "table")

    def _select_columns(self, names: list[str]) -> None:
        want = set(names)
        self._columns.clearSelection()
        for i in range(self._columns.count()):
            item = self._columns.item(i)
            if item.text() in want:
                item.setSelected(True)
        if not self._columns.selectedItems() and self._columns.count():
            self._columns.selectAll()

    def _fill_columns_for_current_table(self) -> None:
        self._columns.clear()
        key = self._table_combo.currentText()
        if not key or self._session.results is None:
            return
        df = self._session.results.tables.get(key)
        if df is None:
            return
        from pyflow_acdc.gui.results.plot_builder import numeric_columns

        for col in numeric_columns(df):
            self._columns.addItem(col)

    def _on_table_changed(self, key: str) -> None:
        self._fill_columns_for_current_table()
        if self._preset.currentData() == "custom":
            self._columns.selectAll()
        else:
            preset = self._preset.currentData()
            if preset == "ac_voltage":
                self._select_columns(["Voltage (pu)", "Voltage angle (deg)"])
            elif preset == "ac_powerflow":
                self._select_columns(
                    [
                        "Power Gen (MW)",
                        "Power Load (MW)",
                        "Power injected  (MW)",
                    ]
                )
    def _on_ts_choice_changed(self, choice: str) -> None:
        self._ts_series.clear()
        grid = self._session.grid
        if not choice or grid is None:
            return
        ts = getattr(grid, "time_series_results", None) or {}
        df = ts.get(choice)
        if df is None:
            return
        for col in df.columns:
            self._ts_series.addItem(str(col))
        self._ts_series.selectAll()

    def _selected_list_texts(self, widget: QListWidget) -> list[str]:
        return [item.text() for item in widget.selectedItems()]

    def _update_plot(self) -> None:
        try:
            mode = self._mode.currentData()
            if mode == "table":
                key = self._table_combo.currentText()
                if not key or self._session.results is None:
                    raise ValueError("Select a results table (run a study first).")
                df = self._session.results.tables[key]
                cols = self._selected_list_texts(self._columns)
                if not cols:
                    raise ValueError("Select at least one column.")
                fig = figure_from_results_table(df, columns=cols, title=key)
            else:
                grid = self._session.grid
                if grid is None:
                    raise ValueError("No grid loaded.")
                choice = self._ts_choice.currentText()
                if not choice:
                    raise ValueError(
                        "No time-series results on this grid. "
                        "Use Results table mode after PF, or run a TS/window study."
                    )
                rows = self._selected_list_texts(self._ts_series)
                if not rows:
                    raise ValueError("Select at least one series.")
                fig = figure_from_ts_choice(grid, choice, rows)
            # Prefer in-window (PNG if no WebEngine); browser is optional.
            self._view.set_figure(fig, open_browser_if_needed=False)
        except Exception as exc:
            QMessageBox.critical(self, "Plot failed", str(exc))

    def _open_browser(self) -> None:
        try:
            self._view.open_in_browser()
        except Exception as exc:
            QMessageBox.warning(self, "No plot", str(exc))

    def _open_dash(self) -> None:
        grid = self._session.grid
        if grid is None or not dash_usable(grid):
            QMessageBox.warning(
                self,
                "Dash unavailable",
                "Full Dash needs TS / window / rolling / season-compare results on the grid. "
                "Snapshot PF tables use the embedded Plotly plot instead.",
            )
            return
        if self._dash_worker is not None and self._dash_worker.isRunning():
            QMessageBox.information(self, "Dash", "Dash is already starting or running.")
            return
        try:
            import dash  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self,
                "Dash missing",
                "Install with: pip install 'pyflow_acdc[Dash]'",
            )
            return

        self._dash_worker = DashLaunchWorker(grid)
        self._dash_worker.started_ok.connect(
            lambda url: QMessageBox.information(
                self,
                "Dash",
                f"Dash server starting.\nOpen: {url}\n(Leave this dialog; server runs in background.)",
            )
        )
        self._dash_worker.failed.connect(
            lambda tb: QMessageBox.critical(self, "Dash failed", tb)
        )
        self._dash_worker.start()
