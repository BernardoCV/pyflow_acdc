# -*- coding: utf-8 -*-
"""Visualize tab — Folium map (Results-based Plotly lives under Results → Plots)."""

from __future__ import annotations

from PySide6.QtWidgets import QVBoxLayout, QWidget

from pyflow_acdc.gui.session import Session
from pyflow_acdc.gui.tabs.map_panel import MapPanel


class VisualizeTab(QWidget):
    def __init__(self, session: Session):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.addWidget(MapPanel(session))
