# -*- coding: utf-8 -*-
"""Static network schematic from node/line geometries (GUI-only, no Mapping changes)."""

from __future__ import annotations

import plotly.graph_objects as go

from pyflow_acdc.Classes import Grid


def schematic_figure_from_grid(grid: Grid, *, title: str = "Network schematic") -> go.Figure:
    """Build a planar Plotly network figure from element ``geometry`` attributes."""
    fig = go.Figure()

    line_groups = (
        ("AC lines", "#000080", list(grid.lines_AC or [])),
        ("AC tf", "#4169e1", list(grid.lines_AC_tf or [])),
        ("AC exp", "#708090", list(grid.lines_AC_exp or [])),
        ("DC lines", "#008000", list(grid.lines_DC or [])),
        ("Converters", "#ff6600", list(grid.Converters_ACDC or [])),
    )
    for name, color, lines in line_groups:
        xs: list[float | None] = []
        ys: list[float | None] = []
        for line in lines:
            geom = getattr(line, "geometry", None)
            if geom is None or getattr(geom, "is_empty", False):
                continue
            coords = list(geom.coords)
            if len(coords) < 2:
                continue
            for x, y in coords:
                xs.append(float(x))
                ys.append(float(y))
            xs.append(None)
            ys.append(None)
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=2),
                    hoverinfo="skip",
                )
            )

    node_groups = (
        ("AC nodes", "#000080", list(grid.nodes_AC or [])),
        ("DC nodes", "#008000", list(grid.nodes_DC or [])),
    )
    for name, color, nodes in node_groups:
        xs = []
        ys = []
        texts = []
        for node in nodes:
            geom = getattr(node, "geometry", None)
            if geom is None or getattr(geom, "is_empty", False):
                continue
            xs.append(float(geom.x))
            ys.append(float(geom.y))
            texts.append(str(getattr(node, "name", "")))
        if xs:
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers+text",
                    name=name,
                    text=texts,
                    textposition="top center",
                    marker=dict(size=10, color=color),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    if not fig.data:
        raise ValueError(
            "No geometries on this grid. Load a case in the GUI so layout "
            "coordinates are created, then Update map again."
        )

    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="x",
        yaxis_title="y",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        legend_title_text="Elements",
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig
