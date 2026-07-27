# -*- coding: utf-8 -*-
"""Build Plotly figures for the GUI Results tab."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from pyflow_acdc.Classes import Grid
from pyflow_acdc.Results_class import Results


def numeric_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(str(col))
            continue
        try:
            pd.to_numeric(df[col], errors="raise")
            cols.append(str(col))
        except (TypeError, ValueError):
            continue
    return cols


def _x_labels(df: pd.DataFrame) -> list[str]:
    for candidate in ("Node", "name", "Name", "Line", "Converter", "Bus"):
        if candidate in df.columns:
            return df[candidate].astype(str).tolist()
    return [str(i) for i in df.index]


def figure_from_results_table(
    df: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    title: str = "",
) -> go.Figure:
    """Interactive bar chart of numeric columns (snapshot Results tables)."""
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title or "Empty table")
        return fig

    numeric = numeric_columns(df)
    if columns:
        missing = [c for c in columns if c not in numeric]
        if missing:
            raise ValueError(f"Non-numeric or missing columns: {missing}")
        numeric = list(columns)
    if not numeric:
        raise ValueError("No numeric columns to plot")

    x = _x_labels(df)
    fig = go.Figure()
    for col in numeric:
        y = pd.to_numeric(df[col], errors="coerce")
        fig.add_trace(go.Bar(name=str(col), x=x, y=y, hovertemplate="%{x}<br>%{y}<extra>%{fullData.name}</extra>"))

    fig.update_layout(
        title=title or None,
        barmode="group",
        xaxis_title="",
        yaxis_title="Value",
        legend_title_text="Series",
        hovermode="x unified",
        template="plotly_white",
    )
    return fig


def available_ts_plot_choices(grid: Grid) -> list[str]:
    """Classic Dash TS plot keys when ``time_series_results`` exists."""
    ts = getattr(grid, "time_series_results", None)
    if not isinstance(ts, dict) or not ts:
        return []
    return sorted(str(k) for k in ts.keys() if isinstance(ts.get(k), pd.DataFrame) and not ts[k].empty)


def figure_from_ts_choice(grid: Grid, plotting_choice: str, selected_rows: list[str] | None = None) -> go.Figure:
    from pyflow_acdc.Graph_Dash import plot_TS_res_dash

    ts = grid.time_series_results
    df = ts.get(plotting_choice)
    if df is None or df.empty:
        raise ValueError(f"No data for {plotting_choice!r}")
    rows = selected_rows if selected_rows else list(df.columns)
    return plot_TS_res_dash(grid, plotting_choice, rows)


def dash_usable(grid: Grid | None) -> bool:
    if grid is None:
        return False
    if getattr(grid, "season_window_compare_run", False) and getattr(grid, "season_window_compare", None):
        return True
    if getattr(grid, "rolling_window_opf_run", False) and getattr(grid, "window_opf_results", None):
        return True
    if getattr(grid, "window_opf_run", False) and getattr(grid, "window_opf_results", None):
        return True
    if getattr(grid, "Time_series_ran", False) and getattr(grid, "time_series_results", None):
        return True
    ts_inv = getattr(grid, "ts_inv", None)
    if ts_inv:
        return True
    return False


def table_plot_options(results: Results | None) -> list[str]:
    if results is None or not results.tables:
        return []
    out = []
    for key, df in results.tables.items():
        if isinstance(df, pd.DataFrame) and not df.empty and numeric_columns(df):
            out.append(key)
    return sorted(out)
