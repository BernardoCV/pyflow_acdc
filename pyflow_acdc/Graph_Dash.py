# -*- coding: utf-8 -*-
"""Interactive Dash applications.

Builds interactive (Dash/Plotly) apps for exploring grids and time-series /
multi-period results.

Owns: interactive web-app figures and callbacks.
Does not own: static plotting (see ``Graph_and_plot``).
"""

import dash
from dash import dcc, html, ctx
from dash.dependencies import Input, Output, State, ALL, MATCH
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

__all__ = [
    'run_dash',
    'run_ts_dash',
    'run_window_dash',
    'run_season_compare_dash',
    'run_mp_ts_dash',
    'create_mp_ts_dash',
    'create_dash_app',
    'create_window_dash_app',
    'create_season_compare_dash_app',
    'build_season_window_compare',
    'attach_season_window_compare',
    'plot_TS_res_from_ts',
    'plot_TS_res_dash',
    'plot_window_res_dash',
    'plot_season_compare_dash',
]

# Docs Furo "SCADA panel" light palette (docs/conf.py light_css_variables).
_THEME = {
    'sidebar_bg': '#142033',
    'sidebar_border': '#253a57',
    'sidebar_text': '#d0dbe7',
    'sidebar_text_top': '#60a5fa',
    'sidebar_caption': '#8da2bb',
    'bg_primary': '#f7f9fc',
    'bg_secondary': '#eef3f8',
    'bg_hover': '#e1e8f0',
    'border': '#d6dee8',
    'text_primary': '#172033',
    'text_secondary': '#3f4f63',
    'text_muted': '#6b7788',
    'accent': '#2563eb',
    'accent_hover': '#1d4ed8',
    'accent_visited': '#6d28d9',
    'header_tint': '#dde6f1',
    'fig_paper': '#f7f9fc',
    'fig_plot': 'white',
    'fig_grid': '#d6dee8',
    'fig_zero': '#172033',
    'fig_font': '#172033',
}

_SERIES_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]

_TS_PLOT_CHOICES = [
    'Power Generation by price zone',
    'Power Generation by generator',
    'Power Generation by price zone area chart',
    'Power Generation by generator area chart',
    'Market Prices',
    'AC line loading',
    'DC line loading',
    'AC/DC Converters',
    'Curtailment',
]

_WINDOW_PLOT_CHOICES = [
    'Total power',
    'Storage SoC',
    'Storage power',
    'Storage Q',
    'Hydrogen mass',
    'Hydrogen power',
    'Generator power',
    'Generator price',
    'Renewable power',
    'Renewable price',
    'Curtailment',
    'AC line loading',
    'DC line loading',
    'AC/DC Converters',
]

_WINDOW_TOTAL_POWER_SERIES = (
    ('Total ren', 'ren_power'),
    ('Total gen', 'gen_power'),
    ('Total H2', 'hydrogen_P_e'),
    ('Total BESS', 'storage_power'),
)

_SEASON_COMPARE_PLOT_CHOICES = [label for label, _ in _WINDOW_TOTAL_POWER_SERIES]

_MP_PLOT_CHOICES = [
    'Power Generation by price zone',
    'Power Generation by generator',
    'Power Generation by price zone area chart',
    'Power Generation by generator area chart',
    'Market Prices',
    'PN',
    'PN_min',
    'PN_max',
    'Grid loading',
    'Real load',
    'Known load by zone',
    'AC line loading',
    'DC line loading',
    'AC/DC Converters',
    'Curtailment',
]


def _card_style(**overrides):
    style = {
        'backgroundColor': _THEME['bg_secondary'],
        'border': f"1px solid {_THEME['border']}",
        'borderRadius': '10px',
        'padding': '16px',
        'marginBottom': '16px',
        'boxShadow': '0 1px 3px rgba(23,32,51,0.08)',
    }
    style.update(overrides)
    return style


def _label_style(on_dark=False):
    return {
        'fontWeight': '600',
        'display': 'block',
        'marginBottom': '6px',
        'color': _THEME['sidebar_caption'] if on_dark else _THEME['text_secondary'],
    }


_SIDEBAR_WIDTH_PX = 420
_DEFAULT_PLOT_HEIGHT = 480


def _normalize_plot_height(height):
    if height is None:
        return _DEFAULT_PLOT_HEIGHT
    try:
        h = int(height)
    except (TypeError, ValueError):
        raise ValueError(f'plot height must be an integer, got {height!r}')
    if h < 200:
        raise ValueError(f'plot height must be >= 200, got {h}')
    return h


def _apply_plot_height(fig, height):
    fig.update_layout(height=_normalize_plot_height(height))
    return fig


def _graph_style(height):
    return {'height': f'{_normalize_plot_height(height)}px'}


def _sidebar_style(is_open):
    # Do not set ``color`` here — it cascades into Dropdown menus (white bg +
    # pale inherited text). Labels / radios / checklists set their own colors.
    return {
        'position': 'fixed',
        'top': 0,
        'left': 0,
        'height': '100vh',
        'width': f'{_SIDEBAR_WIDTH_PX}px',
        'overflowY': 'auto',
        'padding': '18px',
        'boxSizing': 'border-box',
        'zIndex': 1000,
        'backgroundColor': _THEME['sidebar_bg'],
        'borderRight': f"1px solid {_THEME['sidebar_border']}",
        'transition': 'transform .25s ease',
        'transform': 'translateX(0)' if is_open else 'translateX(-100%)',
    }


def _content_style(is_open):
    return {
        'marginLeft': f'{_SIDEBAR_WIDTH_PX}px' if is_open else '0',
        'transition': 'margin-left .25s ease',
        'padding': '24px',
        'backgroundColor': _THEME['bg_primary'],
        'minHeight': '100vh',
        'boxSizing': 'border-box',
        'fontFamily': 'Arial, sans-serif',
        'color': _THEME['text_primary'],
    }


_DASH_CUSTOM_CSS = f"""
/* Dropdown / select: dark text on light surfaces (readable over navy sidebar) */
#sidebar .Select-control,
#sidebar .Select-menu-outer,
#sidebar .Select-value-label,
#sidebar .Select-placeholder,
#sidebar .Select-input > input {{
    color: {_THEME['text_primary']} !important;
}}
#sidebar .Select-menu-outer .VirtualizedSelectOption,
#sidebar .Select-menu-outer .VirtualizedSelectFocusedOption {{
    color: {_THEME['text_primary']} !important;
    background-color: #ffffff;
}}
#sidebar .Select-menu-outer .VirtualizedSelectFocusedOption {{
    background-color: {_THEME['header_tint']} !important;
}}
/* Dash / react-select v4+ */
#sidebar div[class*="-control"],
#sidebar div[class*="-menu"],
#sidebar div[class*="-option"],
#sidebar div[class*="-singleValue"],
#sidebar div[class*="-placeholder"],
#sidebar div[class*="-Input"] input {{
    color: {_THEME['text_primary']} !important;
}}
#sidebar div[class*="-option"] {{
    background-color: #ffffff;
}}
#sidebar div[class*="-option"]:hover,
#sidebar div[class*="-option"][aria-selected="true"],
#sidebar div[class*="-option"][class*="focused"] {{
    background-color: {_THEME['header_tint']} !important;
    color: {_THEME['text_primary']} !important;
}}
#sidebar input[type="number"] {{
    color: {_THEME['text_primary']} !important;
    background-color: #ffffff;
}}
#sidebar .dash-checklist label,
#sidebar .dash-radio-items label {{
    color: {_THEME['sidebar_text']} !important;
}}
"""


def _attach_app_css(app):
    app.index_string = f"""<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{{%title%}}</title>
        {{%favicon%}}
        {{%css%}}
        <style>{_DASH_CUSTOM_CSS}</style>
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
    </body>
</html>"""



def _btn_style(**overrides):
    style = {
        'backgroundColor': _THEME['accent'],
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'padding': '8px 12px',
        'cursor': 'pointer',
        'fontWeight': '600',
        'marginBottom': '12px',
        'width': '100%',
    }
    style.update(overrides)
    return style


def _toggle_btn_style():
    return {
        'backgroundColor': _THEME['accent'],
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'padding': '8px 14px',
        'cursor': 'pointer',
        'fontWeight': '600',
        'marginBottom': '16px',
    }


def _remove_btn_style():
    return {
        'backgroundColor': 'transparent',
        'color': _THEME['sidebar_caption'],
        'border': f"1px solid {_THEME['sidebar_border']}",
        'borderRadius': '4px',
        'padding': '2px 8px',
        'cursor': 'pointer',
        'fontSize': '12px',
    }


def _sidebar_panel_card_style():
    return {
        'backgroundColor': '#1a2a40',
        'border': f"1px solid {_THEME['sidebar_border']}",
        'borderRadius': '8px',
        'padding': '12px',
        'marginBottom': '12px',
    }


def _apply_fig_theme(fig, *, show_title=True):
    fig.update_layout(
        paper_bgcolor=_THEME['fig_paper'],
        plot_bgcolor=_THEME['fig_plot'],
        font=dict(family='Arial, sans-serif', color=_THEME['fig_font']),
        hovermode='x unified',
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor=_THEME['border'],
            borderwidth=1,
        ),
        margin=dict(l=60, r=30, t=80 if show_title else 40, b=60),
    )
    axis = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor=_THEME['fig_grid'],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=_THEME['fig_zero'],
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _hex_to_rgba(hex_color, alpha=0.5):
    h = hex_color.lstrip('#')
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f'rgba({r},{g},{b},{alpha})'


def _get_df_and_label_from_ts(time_series_results, S_base, plotting_choice):
    """Resolve (dataframe, y-axis label) from a time_series_results mapping (not grid)."""
    if plotting_choice == 'Curtailment':
        return time_series_results['curtailment'] * 100, 'Curtailment %'
    if plotting_choice == 'PN':
        df = time_series_results.get('net_price_zone_power')
        if df is None:
            return None, ''
        # Keep behavior similar to "Market Prices": drop "o_*" helper columns if present.
        if hasattr(df, "columns"):
            df = df.loc[:, ~df.columns.str.startswith('o_')]
        return df, 'PN (MW)'
    if plotting_choice == 'Grid loading':
        df = time_series_results.get('grid_loading')
        return (df * 100.0) if df is not None else None, 'Grid loading %' if df is not None else ''
    if plotting_choice == 'Real load':
        # OPF step stores load with a sign convention (negative), so flip it to show positive load.
        df = time_series_results.get('real_load_opf')
        return (-df * S_base) if df is not None else None, 'Real load (MW)' if df is not None else ''
    if plotting_choice == 'Known load by zone':
        df = time_series_results.get('real_load_known_by_zone')
        return (df * S_base) if df is not None else None, 'P_known_AC by zone (MW)' if df is not None else ''
    if plotting_choice == 'PN_min':
        df = time_series_results.get('PN_min')
        return df, 'PN_min (MW)' if df is not None else ''
    if plotting_choice == 'PN_max':
        df = time_series_results.get('PN_max')
        return df, 'PN_max (MW)' if df is not None else ''
    if plotting_choice in ['Power Generation by generator', 'Power Generation by generator area chart']:
        return time_series_results['real_power_opf'] * S_base, 'Power Generation (MW)'
    if plotting_choice in ['Power Generation by price zone', 'Power Generation by price zone area chart']:
        return time_series_results['real_power_by_zone'] * S_base, 'Power Generation (MW)'
    if plotting_choice == 'Market Prices':
        df = time_series_results['prices_by_zone']
        df = df.loc[:, ~df.columns.str.startswith('o_')]
        return df, 'Market Prices (€/MWh)'
    if plotting_choice == 'AC line loading':
        return time_series_results['ac_loading'] * 100, 'AC Line Loading %'
    if plotting_choice == 'DC line loading':
        return time_series_results['dc_loading'] * 100, 'DC Line Loading %'
    if plotting_choice == 'AC/DC Converters':
        return time_series_results['converter_loading'] * 100, 'AC/DC Converters loading %'
    return None, ''


def _frame_column_to_index(df):
    """Turn window result tables (``frame`` column) into TS-style indexed frames."""
    if df is None:
        return None
    if 'frame' not in getattr(df, 'columns', []):
        return df
    out = df.set_index('frame')
    out.index.name = None
    return out


def _window_total_power_df(window_opf_results):
    """Sum ren / gen / H₂ / BESS power into one multi-series frame (MW)."""
    series = {}
    for label, key in _WINDOW_TOTAL_POWER_SERIES:
        df = _frame_column_to_index(window_opf_results.get(key))
        if df is None or df.empty:
            continue
        series[label] = df.sum(axis=1)
    if not series:
        return None
    return pd.DataFrame(series)


def build_season_window_compare(season_to_window_results):
    """Build compare tables: metric → DataFrame (index=hour, columns=season).

    Parameters
    ----------
    season_to_window_results : dict
        Mapping ``season_name -> window_opf_results`` (as on ``grid.window_opf_results``).
    """
    if not season_to_window_results:
        raise ValueError("season_to_window_results is empty")

    per_season_totals = {}
    for season, res in season_to_window_results.items():
        totals = _window_total_power_df(res)
        if totals is None or totals.empty:
            raise ValueError(
                f"Season {season!r} has no total-power series in window_opf_results"
            )
        # Align to hour-of-day for overlay (drop leading SoC-only frames if present).
        totals = totals.copy()
        totals.index = range(len(totals))
        per_season_totals[season] = totals

    compare = {}
    for metric, _ in _WINDOW_TOTAL_POWER_SERIES:
        cols = {}
        for season, totals in per_season_totals.items():
            if metric in totals.columns:
                cols[season] = totals[metric]
        if cols:
            compare[metric] = pd.DataFrame(cols)
    if not compare:
        raise ValueError("No overlapping total-power metrics across seasons")
    return compare


def attach_season_window_compare(grid, season_to_window_results):
    """Store season-compare tables on ``grid`` for Dash (``run_dash`` auto)."""
    grid.season_window_compare = build_season_window_compare(season_to_window_results)
    grid.season_window_compare_run = True
    return grid


def _season_compare_usable(grid):
    return (
        getattr(grid, 'season_window_compare_run', False)
        and isinstance(getattr(grid, 'season_window_compare', None), dict)
        and bool(grid.season_window_compare)
    )


def _available_season_compare_plot_choices(grid):
    choices = []
    for choice in _SEASON_COMPARE_PLOT_CHOICES:
        df = grid.season_window_compare.get(choice)
        if df is not None and not df.empty:
            choices.append(choice)
    return choices


def _get_df_and_label_from_season_compare(season_window_compare, plotting_choice):
    df = season_window_compare.get(plotting_choice)
    if df is None:
        return None, ''
    return df, f'{plotting_choice} (MW)'


def _get_df_and_label_from_window(window_opf_results, plotting_choice):
    """Resolve (dataframe, y-axis label) from ``grid.window_opf_results``."""
    if plotting_choice == 'Total power':
        df = _window_total_power_df(window_opf_results)
        return df, 'Power (MW)' if df is not None else ''
    if plotting_choice == 'Curtailment':
        df = _frame_column_to_index(window_opf_results.get('curtailment'))
        return (df * 100.0) if df is not None else None, (
            'Curtailment %' if df is not None else ''
        )
    if plotting_choice == 'AC line loading':
        df = _frame_column_to_index(window_opf_results.get('ac_loading'))
        return (df * 100.0) if df is not None else None, (
            'AC Line Loading %' if df is not None else ''
        )
    if plotting_choice == 'DC line loading':
        df = _frame_column_to_index(window_opf_results.get('dc_loading'))
        return (df * 100.0) if df is not None else None, (
            'DC Line Loading %' if df is not None else ''
        )
    if plotting_choice == 'AC/DC Converters':
        df = _frame_column_to_index(window_opf_results.get('converter_loading'))
        return (df * 100.0) if df is not None else None, (
            'AC/DC Converters loading %' if df is not None else ''
        )

    key_map = {
        'Storage SoC': ('storage_soc', 'SoC'),
        'Storage power': ('storage_power', 'Storage P (MW, +discharge/−charge)'),
        'Storage Q': ('storage_Q', 'Q (MVAr)'),
        'Hydrogen mass': ('hydrogen_mass_H2', 'H₂ mass (kg)'),
        'Hydrogen power': ('hydrogen_P_e', 'Electrolyser P (MW)'),
        'Generator power': ('gen_power', 'Generator P (MW)'),
        'Generator price': ('gen_price', 'Generator price (EUR/MWh)'),
        'Renewable power': ('ren_power', 'Renewable P (MW)'),
        'Renewable price': ('ren_price', 'Renewable price (EUR/MWh)'),
    }
    entry = key_map.get(plotting_choice)
    if entry is None:
        return None, ''
    key, label = entry
    df = window_opf_results.get(key)
    return _frame_column_to_index(df), label if df is not None else ''


def _get_df_and_label(grid, plotting_choice):
    if _season_compare_usable(grid):
        return _get_df_and_label_from_season_compare(
            grid.season_window_compare, plotting_choice
        )
    if getattr(grid, 'window_opf_run', False) and getattr(grid, 'window_opf_results', None):
        return _get_df_and_label_from_window(grid.window_opf_results, plotting_choice)
    return _get_df_and_label_from_ts(grid.time_series_results, grid.S_base, plotting_choice)


def _window_opf_usable(grid):
    return (
        getattr(grid, 'window_opf_run', False)
        and isinstance(getattr(grid, 'window_opf_results', None), dict)
        and bool(grid.window_opf_results)
    )


def _available_window_plot_choices(grid):
    res = grid.window_opf_results
    choices = []
    for choice in _WINDOW_PLOT_CHOICES:
        df, _ = _get_df_and_label_from_window(res, choice)
        if df is not None and not df.empty:
            choices.append(choice)
    return choices


def _columns_for_choice(grid, plotting_choice):
    df, _ = _get_df_and_label(grid, plotting_choice)
    if df is None or df.empty:
        return []
    return df.columns.tolist()


def _auto_ylimits(grid, plotting_choice):
    data, _ = _get_df_and_label(grid, plotting_choice)
    if data is None or data.empty:
        return 0, 1
    y_min = int(min(0, data.min().min() - 5))
    if plotting_choice in [
        'Power Generation by generator area chart',
        'Power Generation by price zone area chart',
    ]:
        y_max = int(data.sum(axis=1).max() + 10)
    elif plotting_choice in ['AC line loading', 'DC line loading', 'Curtailment']:
        y_max = int(min(data.max().max() + 10, 100))
    elif plotting_choice == 'Storage SoC':
        y_min, y_max = 0, 1
    else:
        y_max = int(data.max().max() + 10)
    return y_min, y_max


def plot_TS_res_from_ts(
    time_series_results,
    S_base,
    plotting_choice,
    selected_rows,
    x_limits=None,
    y_limits=None,
    show_title=True,
    legend_prefix='',
):
    """Build one Plotly figure from stored TS results (e.g. one investment period)."""
    df, y_label = _get_df_and_label_from_ts(time_series_results, S_base, plotting_choice)
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=(f"Time Series: {plotting_choice}" if show_title else None),
            xaxis_title="Time",
            yaxis_title=y_label if y_label else "Value",
        )
        return _apply_fig_theme(fig, show_title=show_title)

    time = df.index

    fig = go.Figure()
    cumulative_sum = None
    stack_areas = plotting_choice in [
        'Power Generation by generator area chart',
        'Power Generation by price zone area chart',
    ]

    for i, col in enumerate(selected_rows):
        if col in df.columns:
            y_values = df[col]
            color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
            trace_name = f'{legend_prefix}{col}' if legend_prefix else col

            if stack_areas:
                if cumulative_sum is None:
                    cumulative_sum = y_values.copy()
                    fig.add_trace(
                        go.Scatter(
                            x=time, y=y_values, name=trace_name, hoverinfo='x+y+name',
                            fill='tozeroy', line=dict(color=color),
                            fillcolor=_hex_to_rgba(color),
                        )
                    )
                else:
                    y_values = cumulative_sum + y_values
                    cumulative_sum = y_values
                    fig.add_trace(
                        go.Scatter(
                            x=time, y=y_values, name=trace_name, hoverinfo='x+y+name',
                            fill='tonexty', line=dict(color=color),
                            fillcolor=_hex_to_rgba(color),
                        )
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=time, y=y_values, name=trace_name, hoverinfo='x+y+name',
                        line=dict(color=color, width=2),
                    )
                )

    title_block = None
    if show_title:
        title_block = dict(
            text=f"Time Series: {plotting_choice}",
            font=dict(size=24, color=_THEME['text_primary']),
            x=0.5,
            xanchor='center',
        )
    fig.update_layout(
        title=title_block,
        xaxis_title=dict(text="Time", font=dict(size=14)),
        yaxis_title=dict(text=y_label, font=dict(size=14)),
        showlegend=True,
    )
    _apply_fig_theme(fig, show_title=show_title)

    if x_limits is None:
        x_limits = (df.index[0], df.index[-1])
    fig.update_xaxes(range=x_limits)

    if y_limits and len(y_limits) == 2:
        fig.update_yaxes(range=y_limits)

    return fig


def plot_TS_res_dash(grid, plotting_choice, selected_rows, x_limits=None, y_limits=None):
    """Build one Plotly figure from ``grid.time_series_results`` (Dash callback helper)."""
    return plot_TS_res_from_ts(
        grid.time_series_results,
        grid.S_base,
        plotting_choice,
        selected_rows,
        x_limits=x_limits,
        y_limits=y_limits,
        show_title=True,
        legend_prefix='',
    )


def plot_window_res_dash(grid, plotting_choice, selected_rows, x_limits=None, y_limits=None):
    """Build one Plotly figure from ``grid.window_opf_results`` (Dash callback helper)."""
    df, y_label = _get_df_and_label_from_window(grid.window_opf_results, plotting_choice)
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Window OPF: {plotting_choice}",
            xaxis_title="Frame",
            yaxis_title=y_label if y_label else "Value",
        )
        return _apply_fig_theme(fig)

    time = df.index
    fig = go.Figure()
    cols = selected_rows if selected_rows else list(df.columns)
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=time, y=df[col], mode='lines', name=str(col),
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title=dict(
            text=f"Window OPF: {plotting_choice}",
            font=dict(size=24, color=_THEME['text_primary']),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title="Frame",
        yaxis_title=y_label,
        legend_title="Elements",
        showlegend=True,
    )
    _apply_fig_theme(fig)
    if x_limits is not None and x_limits[0] is not None and x_limits[1] is not None:
        fig.update_xaxes(range=list(x_limits))
    if y_limits is not None and y_limits[0] is not None and y_limits[1] is not None:
        fig.update_yaxes(range=list(y_limits))
    return fig


def plot_season_compare_dash(grid, plotting_choice, selected_rows, x_limits=None, y_limits=None):
    """Build one Plotly figure comparing seasons (one line per window)."""
    df, y_label = _get_df_and_label_from_season_compare(
        grid.season_window_compare, plotting_choice
    )
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Season compare: {plotting_choice}",
            xaxis_title="Hour",
            yaxis_title=y_label if y_label else "Value",
        )
        return _apply_fig_theme(fig)

    time = df.index
    fig = go.Figure()
    cols = selected_rows if selected_rows else list(df.columns)
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=time, y=df[col], mode='lines', name=str(col),
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title=dict(
            text=f"Season compare: {plotting_choice}",
            font=dict(size=24, color=_THEME['text_primary']),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title="Hour",
        yaxis_title=y_label,
        legend_title="Season",
        showlegend=True,
    )
    _apply_fig_theme(fig)
    if x_limits is not None and x_limits[0] is not None and x_limits[1] is not None:
        fig.update_xaxes(range=list(x_limits))
    if y_limits is not None and y_limits[0] is not None and y_limits[1] is not None:
        fig.update_yaxes(range=list(y_limits))
    return fig


def _panel_control_card(i, default_choice, dd_options):
    return html.Div(style=_sidebar_panel_card_style(), children=[
        html.Div(
            style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center',
                   'marginBottom': '10px'},
            children=[
                html.Span(f'Plot {i + 1}', style={
                    'fontWeight': '700', 'color': _THEME['sidebar_text_top'],
                }),
                html.Button(
                    '✕',
                    id={'type': 'plot-remove', 'index': i},
                    n_clicks=0,
                    style=_remove_btn_style(),
                ),
            ],
        ),
        html.Label('Plot type', style=_label_style(on_dark=True)),
        dcc.Dropdown(
            id={'type': 'plot-choice', 'index': i},
            options=dd_options,
            value=default_choice,
            clearable=False,
            style={'marginBottom': '10px'},
        ),
        html.Label('Components', style=_label_style(on_dark=True)),
        dcc.Checklist(
            id={'type': 'plot-series', 'index': i},
            options=[],
            value=[],
            style={'marginBottom': '10px', 'color': _THEME['sidebar_text']},
        ),
        html.Label('Y-axis limits', style=_label_style(on_dark=True)),
        html.Div(style={'display': 'flex', 'gap': '8px'}, children=[
            dcc.Input(
                id={'type': 'plot-ymin', 'index': i},
                type='number',
                placeholder='Min',
                style={'flex': 1, 'padding': '5px'},
            ),
            dcc.Input(
                id={'type': 'plot-ymax', 'index': i},
                type='number',
                placeholder='Max',
                style={'flex': 1, 'padding': '5px'},
            ),
        ]),
    ])


def _panel_graph_card(i, height=None):
    return html.Div(
        style=_card_style(backgroundColor='white'),
        children=[
            dcc.Graph(
                id={'type': 'plot-graph', 'index': i},
                style=_graph_style(height),
            ),
        ],
    )


def _build_dual_plot_dash_app(
    grid,
    *,
    title,
    plot_choices,
    default_choice_1,
    default_choice_2,
    plot_fn,
    x_axis_label='Time',
):
    """Shared multi-plot Dash layout used by TS and window apps."""
    if not plot_choices:
        raise ValueError('plot_choices is empty')

    app = dash.Dash(__name__)
    _attach_app_css(app)
    dd_options = [{'label': c, 'value': c} for c in plot_choices]

    app.layout = html.Div([
        dcc.Store(id='sidebar-open', data=True),
        dcc.Store(id='plot-panels', data=[0]),
        html.Div(id='sidebar', style=_sidebar_style(True), children=[
            html.Div(
                style={'display': 'flex', 'justifyContent': 'space-between',
                       'alignItems': 'flex-start', 'gap': '8px', 'marginBottom': '12px'},
                children=[
                    html.H2(title, style={
                        'color': _THEME['sidebar_text_top'],
                        'fontSize': '18px',
                        'margin': 0,
                        'flex': 1,
                    }),
                    html.Button(
                        'Hide ✕',
                        id='hide-sidebar',
                        n_clicks=0,
                        style=_remove_btn_style(),
                    ),
                ],
            ),
            html.Div(id='global-controls', children=[
                html.Label(f'X-axis limits ({x_axis_label})', style=_label_style(on_dark=True)),
                html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '12px'}, children=[
                    dcc.Input(
                        id='x-min', type='number', placeholder='Min',
                        style={'flex': 1, 'padding': '5px'},
                    ),
                    dcc.Input(
                        id='x-max', type='number', placeholder='Max',
                        style={'flex': 1, 'padding': '5px'},
                    ),
                ]),
                html.Label('Plot height (px)', style=_label_style(on_dark=True)),
                dcc.Input(
                    id='plot-height',
                    type='number',
                    value=_DEFAULT_PLOT_HEIGHT,
                    min=200,
                    step=20,
                    style={'width': '100%', 'padding': '5px', 'marginBottom': '12px',
                           'boxSizing': 'border-box'},
                ),
            ]),
            html.Button('➕ Add plot', id='add-plot', n_clicks=0, style=_btn_style()),
            html.Div(id='panel-controls'),
        ]),
        html.Div(id='content', style=_content_style(True), children=[
            html.Button(
                '☰ Hide options',
                id='toggle-sidebar',
                n_clicks=0,
                style=_toggle_btn_style(),
            ),
            html.Div(id='panel-graphs'),
        ]),
    ])

    @app.callback(
        [Output('sidebar', 'style'),
         Output('content', 'style'),
         Output('sidebar-open', 'data'),
         Output('toggle-sidebar', 'children')],
        [Input('toggle-sidebar', 'n_clicks'),
         Input('hide-sidebar', 'n_clicks')],
        [State('sidebar-open', 'data')],
    )
    def _toggle_sidebar(n_toggle, n_hide, is_open):
        try:
            trig = ctx.triggered_id
        except dash.exceptions.MissingCallbackContextException:
            if (n_toggle or 0) > 0:
                trig = 'toggle-sidebar'
            elif (n_hide or 0) > 0:
                trig = 'hide-sidebar'
            else:
                trig = None
        if trig in ('toggle-sidebar', 'hide-sidebar'):
            new_open = not bool(is_open)
        else:
            new_open = True if is_open is None else bool(is_open)
        label = '☰ Show options' if not new_open else '☰ Hide options'
        return _sidebar_style(new_open), _content_style(new_open), new_open, label

    @app.callback(
        Output('plot-panels', 'data'),
        [Input('add-plot', 'n_clicks'),
         Input({'type': 'plot-remove', 'index': ALL}, 'n_clicks')],
        [State('plot-panels', 'data')],
        prevent_initial_call=True,
    )
    def _manage_panels(add_clicks, remove_clicks, panels):
        panels = list(panels or [0])
        trig = ctx.triggered_id
        if trig == 'add-plot':
            next_id = (max(panels) + 1) if panels else 0
            return panels + [next_id]
        if isinstance(trig, dict) and trig.get('type') == 'plot-remove':
            rid = trig['index']
            remaining = [p for p in panels if p != rid]
            return remaining if remaining else panels
        return panels

    @app.callback(
        [Output('panel-controls', 'children'),
         Output('panel-graphs', 'children')],
        [Input('plot-panels', 'data')],
    )
    def _render_panels(panels):
        panels = panels or [0]
        defaults = [default_choice_1, default_choice_2]
        ctrls, graphs = [], []
        for pos, i in enumerate(panels):
            dc = defaults[pos] if pos < len(defaults) else plot_choices[0]
            ctrls.append(_panel_control_card(i, dc, dd_options))
            graphs.append(_panel_graph_card(i))
        return ctrls, graphs

    @app.callback(
        [Output({'type': 'plot-series', 'index': MATCH}, 'options'),
         Output({'type': 'plot-series', 'index': MATCH}, 'value'),
         Output({'type': 'plot-ymin', 'index': MATCH}, 'value'),
         Output({'type': 'plot-ymax', 'index': MATCH}, 'value')],
        [Input({'type': 'plot-choice', 'index': MATCH}, 'value')],
    )
    def _panel_series(choice):
        cols = _columns_for_choice(grid, choice)
        ymin, ymax = _auto_ylimits(grid, choice)
        return [{'label': c, 'value': c} for c in cols], cols, ymin, ymax

    @app.callback(
        [Output({'type': 'plot-graph', 'index': ALL}, 'figure'),
         Output({'type': 'plot-graph', 'index': ALL}, 'style')],
        [Input({'type': 'plot-choice', 'index': ALL}, 'value'),
         Input({'type': 'plot-series', 'index': ALL}, 'value'),
         Input({'type': 'plot-ymin', 'index': ALL}, 'value'),
         Input({'type': 'plot-ymax', 'index': ALL}, 'value'),
         Input('x-min', 'value'),
         Input('x-max', 'value'),
         Input('plot-height', 'value')],
    )
    def _draw(choices, series, ymins, ymaxs, xmin, xmax, plot_height):
        x_limits = (xmin, xmax) if xmin is not None and xmax is not None else None
        height = _normalize_plot_height(
            plot_height if plot_height is not None else _DEFAULT_PLOT_HEIGHT
        )
        figs = []
        styles = []
        for choice, sel, ymin, ymax in zip(choices or [], series or [], ymins or [], ymaxs or []):
            y_limits = (ymin, ymax) if ymin is not None and ymax is not None else None
            fig = plot_fn(grid, choice, sel or [], x_limits=x_limits, y_limits=y_limits)
            figs.append(_apply_plot_height(fig, height))
            styles.append(_graph_style(height))
        return figs, styles

    return app


def create_dash_app(grid):
    """Dash app for sequential TS OPF results (``grid.time_series_results``)."""
    return _build_dual_plot_dash_app(
        grid,
        title=f"{grid.name} Time Series Dashboard",
        plot_choices=_TS_PLOT_CHOICES,
        default_choice_1='Power Generation by price zone',
        default_choice_2='Market Prices',
        plot_fn=plot_TS_res_dash,
        x_axis_label='Time',
    )


def create_window_dash_app(grid):
    """Dash app for coupled window NL OPF results (``grid.window_opf_results``)."""
    if not _window_opf_usable(grid):
        raise ValueError(
            "create_window_dash_app requires grid.window_opf_run and grid.window_opf_results"
        )
    choices = _available_window_plot_choices(grid)
    if not choices:
        raise ValueError("window_opf_results has no plottable storage/hydrogen series")
    default_1 = choices[0]
    default_2 = choices[1] if len(choices) > 1 else choices[0]
    return _build_dual_plot_dash_app(
        grid,
        title=f"{getattr(grid, 'name', 'grid')} Window OPF Dashboard",
        plot_choices=choices,
        default_choice_1=default_1,
        default_choice_2=default_2,
        plot_fn=plot_window_res_dash,
        x_axis_label='Frame',
    )


def create_season_compare_dash_app(grid):
    """Dash app comparing seasonal window totals (one line per season)."""
    if not _season_compare_usable(grid):
        raise ValueError(
            "create_season_compare_dash_app requires grid.season_window_compare_run "
            "and grid.season_window_compare"
        )
    choices = _available_season_compare_plot_choices(grid)
    if not choices:
        raise ValueError("season_window_compare has no plottable metrics")
    default_1 = choices[0]
    default_2 = choices[1] if len(choices) > 1 else choices[0]
    return _build_dual_plot_dash_app(
        grid,
        title=f"{getattr(grid, 'name', 'grid')} Season Window Compare",
        plot_choices=choices,
        default_choice_1=default_1,
        default_choice_2=default_2,
        plot_fn=plot_season_compare_dash,
        x_axis_label='Hour',
    )


def _ts_inv_usable(grid):
    ts_inv = getattr(grid, 'ts_inv', None)
    return isinstance(ts_inv, dict) and bool(ts_inv)


def run_ts_dash(grid, debug=True, use_reloader=False):
    """Run the single-grid TS Dash app (requires ``grid.Time_series_ran``)."""
    app = create_dash_app(grid)
    app.run(debug=debug, use_reloader=use_reloader)


def run_window_dash(grid, debug=True, use_reloader=False):
    """Run the window NL OPF Dash app (requires ``grid.window_opf_run``)."""
    app = create_window_dash_app(grid)
    app.run(debug=debug, use_reloader=use_reloader)


def run_season_compare_dash(grid, debug=True, use_reloader=False):
    """Run the season-compare Dash app (requires ``grid.season_window_compare_run``)."""
    app = create_season_compare_dash_app(grid)
    app.run(debug=debug, use_reloader=use_reloader)


def run_dash(grid, debug=True, use_reloader=False):
    """
    Start the appropriate Dash app from grid run flags (same family as ``Grid.reset_run_flags``).

    * ``grid.dash_mode`` optional: ``'auto'`` (default), ``'mp_ts'``, ``'single_ts'``,
      ``'window'``, or ``'season_compare'``.

    **auto** (precedence):

    1. ``season_window_compare_run`` with ``grid.season_window_compare`` → season compare.
    2. ``window_opf_run`` with ``grid.window_opf_results`` → window OPF dashboard.
    3. ``MP_TEP_run`` or ``MP_MS_TEP_run`` and ``grid.ts_inv`` populated (MS TS-OPF post-processing)
       → multi-period TS dashboard.
    4. Else ``Time_series_ran`` → single-grid TS dashboard.
    5. Else raise ``ValueError``.
    """
    mode = getattr(grid, 'dash_mode', 'auto')
    if mode not in ('auto', 'mp_ts', 'single_ts', 'window', 'season_compare'):
        mode = 'auto'

    if mode == 'season_compare':
        if not _season_compare_usable(grid):
            raise ValueError(
                'run_dash: dash_mode=season_compare requires '
                'grid.season_window_compare_run and grid.season_window_compare'
            )
        return run_season_compare_dash(grid, debug=debug, use_reloader=use_reloader)
    if mode == 'window':
        if not _window_opf_usable(grid):
            raise ValueError(
                'run_dash: dash_mode=window requires grid.window_opf_run and grid.window_opf_results'
            )
        return run_window_dash(grid, debug=debug, use_reloader=use_reloader)
    if mode == 'mp_ts':
        if not _ts_inv_usable(grid):
            raise ValueError('run_dash: dash_mode=mp_ts requires grid.ts_inv from MS TS-OPF (run_opf_for_all_investment_periods MS=True).')
        return run_mp_ts_dash(
            grid.ts_inv,
            grid_name=getattr(grid, 'name', 'grid'),
            debug=debug,
            use_reloader=use_reloader,
        )
    if mode == 'single_ts':
        return run_ts_dash(grid, debug=debug, use_reloader=use_reloader)

    # auto
    if _season_compare_usable(grid):
        return run_season_compare_dash(grid, debug=debug, use_reloader=use_reloader)
    if _window_opf_usable(grid):
        return run_window_dash(grid, debug=debug, use_reloader=use_reloader)
    if (getattr(grid, 'MP_TEP_run', False) or getattr(grid, 'MP_MS_TEP_run', False)) and _ts_inv_usable(grid):
        return run_mp_ts_dash(
            grid.ts_inv,
            grid_name=getattr(grid, 'name', 'grid'),
            debug=debug,
            use_reloader=use_reloader,
        )
    if getattr(grid, 'Time_series_ran', False):
        return run_ts_dash(grid, debug=debug, use_reloader=use_reloader)

    raise ValueError(
        'run_dash (auto): need season_window_compare_run, '
        'window_opf_run with window_opf_results, '
        'or (MP_TEP_run or MP_MS_TEP_run) with grid.ts_inv, '
        'or Time_series_ran after TS_ACDC_OPF. '
        "Override with grid.dash_mode='season_compare', 'window', 'mp_ts', or 'single_ts'."
    )


def create_mp_ts_dash(ts_inv, grid_name='MP time series'):
    """
    Dash app for TS results saved per investment period (see run_opf_for_all_investment_periods MS mode).

    ts_inv: mapping with optional ``'base'`` (nominal np) and int keys for MP periods;
    values are ``{'time_series_results': ..., 'S_base': float, ...}``.
    """
    if not ts_inv:
        raise ValueError('ts_inv is empty')

    has_base = 'base' in ts_inv
    int_periods = sorted(k for k in ts_inv.keys() if isinstance(k, int))
    if not has_base and not int_periods:
        raise ValueError('ts_inv has no period keys')

    def _period_order():
        out = []
        if has_base:
            out.append('base')
        out.extend(int_periods)
        return out

    period_order = _period_order()

    plot_dd_options = [{'label': x, 'value': x} for x in _MP_PLOT_CHOICES]

    def _ref_snapshot():
        return ts_inv[period_order[0]]

    def _columns_for(plot_type, ref_snap):
        df, _ = _get_df_and_label_from_ts(
            ref_snap['time_series_results'], ref_snap['S_base'], plot_type
        )
        if df is None or df.empty:
            return [], []
        cols = df.columns.tolist()
        return [{'label': c, 'value': c} for c in cols], cols

    def _auto_ylimits_snap(plot_type, snap):
        data, _ = _get_df_and_label_from_ts(
            snap['time_series_results'], snap['S_base'], plot_type
        )
        if data is None or data.empty:
            return 0, 1
        y_min = int(min(0, data.min().min() - 5))
        if plot_type in [
            'Power Generation by generator area chart',
            'Power Generation by price zone area chart',
        ]:
            y_max = int(data.sum(axis=1).max() + 10)
        elif plot_type in ['AC line loading', 'DC line loading', 'Curtailment']:
            y_max = int(min(data.max().max() + 10, 100))
        else:
            y_max = int(data.max().max() + 10)
        return y_min, y_max

    period_opts = []
    for p in period_order:
        if p == 'base':
            period_opts.append({'label': 'Base (nominal np)', 'value': 'base'})
        else:
            period_opts.append({'label': f'Period {p}', 'value': p})
    period_opts_skip = period_opts + [{'label': '—', 'value': -1}]

    def _default_triple():
        a = period_order[0]
        b = period_order[min(1, len(period_order) - 1)]
        c = period_order[min(2, len(period_order) - 1)]
        return a, b, c

    d1, d2, d3 = _default_triple()
    default_choice_1 = 'Power Generation by price zone'
    default_choice_2 = 'Market Prices'

    def _build_compare_fig(plot_type, cols, p1, p2, p3, x_limits, y_limits):
        titles = []
        for p in (p1, p2, p3):
            if p == -1:
                titles.append('—')
            elif p == 'base':
                titles.append('Base (nominal np)')
            else:
                titles.append(f'Period {p}')
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=titles,
            shared_yaxes=True,
        )
        any_trace = False
        shown_legends = set()
        for ci, p in enumerate((p1, p2, p3)):
            if p == -1 or p not in ts_inv:
                continue
            snap = ts_inv[p]
            sub = plot_TS_res_from_ts(
                snap['time_series_results'],
                snap['S_base'],
                plot_type,
                cols,
                x_limits=x_limits,
                y_limits=y_limits,
                show_title=False,
                legend_prefix='',
            )
            for tr in sub.data:
                tr.showlegend = tr.name not in shown_legends
                shown_legends.add(tr.name)
                fig.add_trace(tr, row=1, col=ci + 1)
                any_trace = True
        if not any_trace:
            fig.add_annotation(
                text='Select at least one period column',
                xref='paper',
                yref='paper',
                x=0.5,
                y=0.5,
                showarrow=False,
            )
        fig.update_layout(height=520, showlegend=True)
        _apply_fig_theme(fig, show_title=False)
        if x_limits is not None:
            for c in range(1, 4):
                fig.update_xaxes(range=x_limits, row=1, col=c)
        if y_limits is not None:
            fig.update_yaxes(range=y_limits, row=1, col=1)
        return fig

    app = dash.Dash(__name__)
    _attach_app_css(app)
    app.layout = html.Div([
        dcc.Store(id='sidebar-open', data=True),
        dcc.Store(id='plot-panels', data=[0]),
        html.Div(id='sidebar', style=_sidebar_style(True), children=[
            html.Div(
                style={'display': 'flex', 'justifyContent': 'space-between',
                       'alignItems': 'flex-start', 'gap': '8px', 'marginBottom': '12px'},
                children=[
                    html.H2(
                        f'{grid_name} — TS by investment period',
                        style={
                            'color': _THEME['sidebar_text_top'],
                            'fontSize': '18px',
                            'margin': 0,
                            'flex': 1,
                        },
                    ),
                    html.Button(
                        'Hide ✕',
                        id='hide-sidebar',
                        n_clicks=0,
                        style=_remove_btn_style(),
                    ),
                ],
            ),
            html.Label('Mode', style=_label_style(on_dark=True)),
            dcc.RadioItems(
                id='mp-mode',
                options=[
                    {'label': ' Single period', 'value': 'single'},
                    {'label': ' Compare three periods', 'value': 'compare'},
                ],
                value='single',
                style={'marginBottom': '12px', 'color': _THEME['sidebar_text']},
            ),
            html.Div(id='mp-single-row', children=[
                html.Label('Period', style=_label_style(on_dark=True)),
                dcc.Dropdown(
                    id='mp-period-single',
                    options=period_opts,
                    value=period_order[0],
                    clearable=False,
                    style={'marginBottom': '12px'},
                ),
            ]),
            html.Div(id='mp-compare-row', style={'display': 'none'}, children=[
                html.Label('Column 1', style=_label_style(on_dark=True)),
                dcc.Dropdown(
                    id='mp-p1', options=period_opts_skip, value=d1, clearable=False,
                    style={'marginBottom': '8px'},
                ),
                html.Label('Column 2', style=_label_style(on_dark=True)),
                dcc.Dropdown(
                    id='mp-p2', options=period_opts_skip, value=d2, clearable=False,
                    style={'marginBottom': '8px'},
                ),
                html.Label('Column 3', style=_label_style(on_dark=True)),
                dcc.Dropdown(
                    id='mp-p3', options=period_opts_skip, value=d3, clearable=False,
                    style={'marginBottom': '12px'},
                ),
            ]),
            html.Label('X-axis limits', style=_label_style(on_dark=True)),
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '8px'}, children=[
                dcc.Input(id='mp-xmin', type='number', placeholder='auto', style={'flex': 1}),
                dcc.Input(id='mp-xmax', type='number', placeholder='auto', style={'flex': 1}),
            ]),
            html.Label('Y-axis limits (global fallback)', style=_label_style(on_dark=True)),
            html.Div(style={'display': 'flex', 'gap': '8px', 'marginBottom': '8px'}, children=[
                dcc.Input(id='mp-ymin', type='number', placeholder='auto', style={'flex': 1}),
                dcc.Input(id='mp-ymax', type='number', placeholder='auto', style={'flex': 1}),
            ]),
            html.Label('Plot height (px)', style=_label_style(on_dark=True)),
            dcc.Input(
                id='plot-height',
                type='number',
                value=_DEFAULT_PLOT_HEIGHT,
                min=200,
                step=20,
                style={'width': '100%', 'padding': '5px', 'marginBottom': '12px',
                       'boxSizing': 'border-box'},
            ),
            html.Button('➕ Add plot', id='add-plot', n_clicks=0, style=_btn_style()),
            html.Div(id='panel-controls'),
        ]),
        html.Div(id='content', style=_content_style(True), children=[
            html.Button(
                '☰ Hide options',
                id='toggle-sidebar',
                n_clicks=0,
                style=_toggle_btn_style(),
            ),
            html.Div(id='panel-graphs'),
        ]),
    ])

    @app.callback(
        [Output('sidebar', 'style'),
         Output('content', 'style'),
         Output('sidebar-open', 'data'),
         Output('toggle-sidebar', 'children')],
        [Input('toggle-sidebar', 'n_clicks'),
         Input('hide-sidebar', 'n_clicks')],
        [State('sidebar-open', 'data')],
    )
    def _toggle_sidebar(n_toggle, n_hide, is_open):
        try:
            trig = ctx.triggered_id
        except dash.exceptions.MissingCallbackContextException:
            if (n_toggle or 0) > 0:
                trig = 'toggle-sidebar'
            elif (n_hide or 0) > 0:
                trig = 'hide-sidebar'
            else:
                trig = None
        if trig in ('toggle-sidebar', 'hide-sidebar'):
            new_open = not bool(is_open)
        else:
            new_open = True if is_open is None else bool(is_open)
        label = '☰ Show options' if not new_open else '☰ Hide options'
        return _sidebar_style(new_open), _content_style(new_open), new_open, label

    @app.callback(
        [Output('mp-compare-row', 'style'),
         Output('mp-single-row', 'style')],
        [Input('mp-mode', 'value')],
    )
    def _toggle_mode(mode):
        if mode == 'compare':
            return {'display': 'block'}, {'display': 'none'}
        return {'display': 'none'}, {'display': 'block'}

    @app.callback(
        Output('plot-panels', 'data'),
        [Input('add-plot', 'n_clicks'),
         Input({'type': 'plot-remove', 'index': ALL}, 'n_clicks')],
        [State('plot-panels', 'data')],
        prevent_initial_call=True,
    )
    def _manage_panels(add_clicks, remove_clicks, panels):
        panels = list(panels or [0])
        trig = ctx.triggered_id
        if trig == 'add-plot':
            next_id = (max(panels) + 1) if panels else 0
            return panels + [next_id]
        if isinstance(trig, dict) and trig.get('type') == 'plot-remove':
            rid = trig['index']
            remaining = [p for p in panels if p != rid]
            return remaining if remaining else panels
        return panels

    @app.callback(
        [Output('panel-controls', 'children'),
         Output('panel-graphs', 'children')],
        [Input('plot-panels', 'data')],
    )
    def _render_panels(panels):
        panels = panels or [0]
        defaults = [default_choice_1, default_choice_2]
        ctrls, graphs = [], []
        for pos, i in enumerate(panels):
            dc = defaults[pos] if pos < len(defaults) else _MP_PLOT_CHOICES[0]
            ctrls.append(_panel_control_card(i, dc, plot_dd_options))
            graphs.append(_panel_graph_card(i))
        return ctrls, graphs

    @app.callback(
        [Output({'type': 'plot-series', 'index': MATCH}, 'options'),
         Output({'type': 'plot-series', 'index': MATCH}, 'value'),
         Output({'type': 'plot-ymin', 'index': MATCH}, 'value'),
         Output({'type': 'plot-ymax', 'index': MATCH}, 'value')],
        [Input({'type': 'plot-choice', 'index': MATCH}, 'value'),
         Input('mp-mode', 'value'),
         Input('mp-period-single', 'value')],
    )
    def _panel_series(choice, mode, period_single):
        ref = _ref_snapshot()
        if mode == 'single' and period_single is not None and period_single in ts_inv:
            ref = ts_inv[period_single]
        opts, cols = _columns_for(choice, ref)
        ymin, ymax = _auto_ylimits_snap(choice, ref)
        return opts, cols, ymin, ymax

    @app.callback(
        [Output({'type': 'plot-graph', 'index': ALL}, 'figure'),
         Output({'type': 'plot-graph', 'index': ALL}, 'style')],
        [Input('mp-mode', 'value'),
         Input('mp-period-single', 'value'),
         Input('mp-p1', 'value'),
         Input('mp-p2', 'value'),
         Input('mp-p3', 'value'),
         Input({'type': 'plot-choice', 'index': ALL}, 'value'),
         Input({'type': 'plot-series', 'index': ALL}, 'value'),
         Input({'type': 'plot-ymin', 'index': ALL}, 'value'),
         Input({'type': 'plot-ymax', 'index': ALL}, 'value'),
         Input('mp-xmin', 'value'),
         Input('mp-xmax', 'value'),
         Input('mp-ymin', 'value'),
         Input('mp-ymax', 'value'),
         Input('plot-height', 'value')],
    )
    def _draw(mode, ps, p1, p2, p3, choices, series_list, ymins, ymaxs, xmin, xmax, gmin, gmax, plot_height):
        choices = choices or []
        series_list = series_list or []
        ymins = ymins or []
        ymaxs = ymaxs or []
        x_limits = (xmin, xmax) if xmin is not None and xmax is not None else None
        global_y = (gmin, gmax) if gmin is not None and gmax is not None else None
        height = _normalize_plot_height(
            plot_height if plot_height is not None else _DEFAULT_PLOT_HEIGHT
        )

        figs = []
        styles = []
        for choice, cols, ymin, ymax in zip(choices, series_list, ymins, ymaxs):
            cols = cols or []
            if ymin is not None and ymax is not None:
                y_limits = (ymin, ymax)
            else:
                y_limits = global_y

            if mode == 'single':
                if ps is None or ps not in ts_inv:
                    fig = go.Figure()
                    fig.update_layout(title='Invalid period')
                    figs.append(_apply_plot_height(_apply_fig_theme(fig), height))
                else:
                    snap = ts_inv[ps]
                    fig = plot_TS_res_from_ts(
                        snap['time_series_results'],
                        snap['S_base'],
                        choice,
                        cols,
                        x_limits=x_limits,
                        y_limits=y_limits,
                        show_title=True,
                        legend_prefix='',
                    )
                    figs.append(_apply_plot_height(fig, height))
            else:
                figs.append(_apply_plot_height(
                    _build_compare_fig(choice, cols, p1, p2, p3, x_limits, y_limits),
                    height,
                ))
            styles.append(_graph_style(height))
        return figs, styles

    return app


def run_mp_ts_dash(ts_inv, grid_name='MP time series', debug=True, use_reloader=False):
    app = create_mp_ts_dash(ts_inv, grid_name=grid_name)
    app.run(debug=debug, use_reloader=use_reloader)
