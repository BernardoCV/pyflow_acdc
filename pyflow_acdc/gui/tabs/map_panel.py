# -*- coding: utf-8 -*-
"""Folium map panel — WebEngine embed, else static schematic PNG + browser Folium."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyflow_acdc.gui.results.html_view import HAS_WEBENGINE, HtmlView
from pyflow_acdc.gui.results.plotly_view import PlotlyView
from pyflow_acdc.gui.results.schematic import schematic_figure_from_grid
from pyflow_acdc.gui.session import Session


class MapPanel(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        self._session = session
        self._folium_path: Path | None = None

        self._text = QComboBox()
        self._text.addItem("Data (hover_text)", "data")
        self._text.addItem("Per-unit", "inPu")
        self._text.addItem("Absolute", "abs")

        self._planar = QCheckBox("Planar coordinates (schematic, no geo)")
        self._planar.setChecked(True)
        self._planar.setToolTip(
            "Use when the case has x/y but not lat/lon. "
            "Uncheck for geographic cases (PEI, NS_MTDC, …)."
        )

        self._render_btn = QPushButton("Update map")
        self._render_btn.clicked.connect(self._update_map)

        self._browser_btn = QPushButton("Open in browser")
        self._browser_btn.clicked.connect(self._open_browser)

        controls = QGroupBox("Map controls")
        controls_layout = QVBoxLayout(controls)
        controls_layout.addWidget(
            QLabel("Folium network · tiles=None (no basemap) · click elements for details")
        )
        if not HAS_WEBENGINE:
            controls_layout.addWidget(
                QLabel(
                    "WebEngine unavailable — in-window view is a static schematic (kaleido). "
                    "Use «Open in browser» for interactive Folium."
                )
            )
        controls_layout.addWidget(QLabel("Hover / popup text mode"))
        controls_layout.addWidget(self._text)
        controls_layout.addWidget(self._planar)
        row = QHBoxLayout()
        row.addWidget(self._render_btn)
        row.addWidget(self._browser_btn)
        controls_layout.addLayout(row)
        controls_layout.addStretch()

        self._html = HtmlView(empty_message="Load a grid, then Update map.")
        self._plot = PlotlyView(
            empty_message="Load a grid, then Update map (static schematic)."
        )
        if HAS_WEBENGINE:
            self._plot.hide()
        else:
            self._html.hide()

        layout = QHBoxLayout(self)
        layout.addWidget(controls, 0)
        layout.addWidget(self._html, 1)
        layout.addWidget(self._plot, 1)

        session.grid_changed.connect(self._on_grid_changed)
        session.results_changed.connect(self._on_grid_changed)
        self._on_grid_changed()

    def _on_grid_changed(self) -> None:
        has = self._session.grid is not None
        self._render_btn.setEnabled(has)
        if not has:
            self._folium_path = None
            self._html.clear()
            self._plot.clear()

    def _update_map(self) -> None:
        grid = self._session.grid
        if grid is None:
            QMessageBox.warning(self, "No grid", "Load a grid first.")
            return

        try:
            import folium  # noqa: F401
        except ImportError:
            QMessageBox.critical(
                self,
                "Folium missing",
                "Install with: pip install 'pyflow_acdc[mapping]'",
            )
            return

        try:
            from pyflow_acdc.Mapping import plot_folium_network

            stamp = Path(tempfile.gettempdir()) / "pyflow_acdc_gui_folium"
            m = plot_folium_network(
                grid,
                text=self._text.currentData(),
                name=str(stamp),
                tiles=None,
                show=False,
                planar=self._planar.isChecked(),
                clustering=False,
            )
            out = Path(tempfile.gettempdir()) / "pyflow_acdc_gui_map.html"
            m.save(str(out))
            self._folium_path = out
            html = out.read_text(encoding="utf-8")

            if HAS_WEBENGINE:
                self._html.set_html(html, open_browser_if_needed=False)
            else:
                fig = schematic_figure_from_grid(
                    grid,
                    title=f"{getattr(grid, 'name', None) or 'Grid'} (static schematic)",
                )
                self._plot.set_figure(fig, open_browser_if_needed=False)
        except Exception as exc:
            QMessageBox.critical(self, "Map failed", str(exc))

    def _open_browser(self) -> None:
        if self._folium_path is not None and self._folium_path.is_file():
            webbrowser.open(self._folium_path.as_uri())
            return
        try:
            if HAS_WEBENGINE:
                self._html.open_in_browser()
            else:
                self._plot.open_in_browser()
        except Exception as exc:
            QMessageBox.warning(self, "No map", str(exc))
