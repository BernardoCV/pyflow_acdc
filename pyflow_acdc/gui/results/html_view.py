# -*- coding: utf-8 -*-
"""Embed HTML (Plotly / Folium) via Qt WebEngine or browser fallback."""

from __future__ import annotations

import tempfile
import webbrowser
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView

    HAS_WEBENGINE = True
except Exception:
    # ImportError or Windows DLL load failure
    HAS_WEBENGINE = False
    QWebEngineView = None  # type: ignore[misc, assignment]


class HtmlView(QWidget):
    def __init__(self, parent=None, empty_message: str = "Nothing to show yet."):
        super().__init__(parent)
        self._empty_message = empty_message
        self._last_html: str | None = None
        self._last_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._hint = QLabel(empty_message)
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self._view = None
        if HAS_WEBENGINE:
            self._view = QWebEngineView()
            layout.addWidget(self._view, 1)
        else:
            self._hint.setText(
                f"{empty_message}\n\n"
                "Qt WebEngine not available — use «Open in browser».\n"
                "Optional: conda install -c conda-forge pyside6-qtwebengine"
            )

    def clear(self) -> None:
        self._last_html = None
        self._last_path = None
        if self._view is not None:
            self._view.setHtml(
                f"<html><body style='font-family:sans-serif;color:#333'>"
                f"{self._empty_message}</body></html>"
            )

    def set_html(self, html: str, *, open_browser_if_needed: bool = True) -> Path:
        self._last_html = html
        path = Path(tempfile.gettempdir()) / "pyflow_acdc_gui_view.html"
        path.write_text(html, encoding="utf-8")
        self._last_path = path

        if self._view is not None:
            self._view.setUrl(QUrl.fromLocalFile(str(path.resolve())))
            self._hint.setText("Interactive view (zoom / pan / click for details).")
            return path

        self._hint.setText(
            f"Saved: {path}\nClick «Open in browser» for hover / zoom."
        )
        if open_browser_if_needed:
            webbrowser.open(path.as_uri())
        return path

    def open_in_browser(self) -> None:
        if self._last_path is None or not self._last_path.is_file():
            if not self._last_html:
                raise RuntimeError("Nothing to open yet.")
            self.set_html(self._last_html, open_browser_if_needed=False)
        webbrowser.open(self._last_path.as_uri())
