# -*- coding: utf-8 -*-
"""Embed Plotly figures: WebEngine HTML, else static PNG in-window."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
import plotly.graph_objects as go

from pyflow_acdc.gui.results.html_view import HAS_WEBENGINE, HtmlView


def figure_to_html(fig: go.Figure) -> str:
    return fig.to_html(
        include_plotlyjs=True,
        full_html=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["sendDataToCloud"],
        },
    )


def figure_to_png_bytes(fig: go.Figure) -> bytes:
    try:
        return fig.to_image(format="png", scale=2)
    except Exception as exc:
        raise RuntimeError(
            "In-window Plotly needs kaleido when Qt WebEngine is unavailable. "
            "Install with: pip install kaleido"
        ) from exc


class PlotlyView(QWidget):
    def __init__(self, parent=None, empty_message: str = "No plot yet."):
        super().__init__(parent)
        self._empty_message = empty_message
        self._last_html: str | None = None
        self._last_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._hint = QLabel(empty_message)
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self._html: HtmlView | None = None
        self._image = QLabel()
        self._image.setAlignment(Qt.AlignCenter)
        self._image.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._image)

        if HAS_WEBENGINE:
            self._html = HtmlView(empty_message=empty_message)
            layout.addWidget(self._html, 1)
            self._scroll.hide()
        else:
            layout.addWidget(self._scroll, 1)
            self._hint.setText(
                f"{empty_message}\n\n"
                "Qt WebEngine unavailable — showing static PNG in-window "
                "(install kaleido). Use «Open in browser» for interactive Plotly."
            )

    def clear(self) -> None:
        self._last_html = None
        self._last_path = None
        self._image.clear()
        if self._html is not None:
            self._html.clear()
        self._hint.setText(self._empty_message)

    def set_figure(self, fig: go.Figure, *, open_browser_if_needed: bool = False):
        html = figure_to_html(fig)
        self._last_html = html
        path = Path(tempfile.gettempdir()) / "pyflow_acdc_gui_view.html"
        path.write_text(html, encoding="utf-8")
        self._last_path = path

        if self._html is not None:
            return self._html.set_html(html, open_browser_if_needed=open_browser_if_needed)

        png = figure_to_png_bytes(fig)
        pix = QPixmap()
        if not pix.loadFromData(png):
            raise RuntimeError("Failed to decode Plotly PNG for in-window display.")
        self._image.setPixmap(pix)
        self._hint.setText(
            "Static plot (WebEngine unavailable). "
            "Use «Open in browser» for hover / zoom."
        )
        if open_browser_if_needed:
            webbrowser.open(path.as_uri())
        return path

    def open_in_browser(self) -> None:
        if self._last_path is None or not self._last_path.is_file():
            if not self._last_html:
                raise RuntimeError("Nothing to open yet.")
            path = Path(tempfile.gettempdir()) / "pyflow_acdc_gui_view.html"
            path.write_text(self._last_html, encoding="utf-8")
            self._last_path = path
        webbrowser.open(self._last_path.as_uri())
