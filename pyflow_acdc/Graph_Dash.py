# -*- coding: utf-8 -*-
"""Interactive Dash applications.

Builds interactive (Dash/Plotly) apps for exploring grids and time-series /
multi-period results.

Owns: interactive web-app figures and callbacks.
Does not own: static plotting (see ``Graph_and_plot``).
"""

import colorsys
import os

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
    'resolve_family_df',
    'resolve_season_family_df',
    'available_dash_families',
    'available_family_aggregations',
    'plot_TS_res_from_ts',
    'plot_TS_res_dash',
    'plot_window_res_dash',
    'plot_window_family_dash',
    'plot_season_family_dash',
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

# Theme-aligned series palette (readable on white plot bg).
_SERIES_COLORS = [
    '#2563eb', '#ea580c', '#059669', '#dc2626', '#7c3aed',
    '#0891b2', '#ca8a04', '#db2777', '#4b5563', '#65a30d',
]

_FONT_STACK = '"DM Sans", "Segoe UI", system-ui, sans-serif'

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

# Family + aggregation for window / season-compare Dash.
# Aggregation order: entity → node → zone/pz → total.
_FAMILY_SPECS = {
    # Overview: Total ren / gen / H₂ / BESS on one plot (window results only).
    'Power': {
        'key': None,
        'ylabel': 'Power (MW)',
        'entity_agg': 'source',
        'node_agg': 'node',
        'zone_agg': 'zone',
        'reduce': 'sum',
        'kind': 'power',
    },
    'Ren Sources': {
        'key': 'ren_power',
        'ylabel': 'Power (MW)',
        'entity_agg': 'ren_source',
        'node_agg': 'node',
        'zone_agg': 'zone',
        'reduce': 'sum',
        'kind': 'ren',
    },
    'Generators': {
        'key': 'gen_power',
        'ts_key': 'real_power_opf',
        'ts_scale': 'S_base',
        'ylabel': 'Power (MW)',
        'entity_agg': 'gen',
        'node_agg': 'nodes',
        'zone_agg': 'pz',
        'reduce': 'sum',
        'kind': 'gen',
    },
    'Prices': {
        'key': 'gen_price',
        'ylabel': 'Price (EUR/MWh)',
        'entity_agg': 'gen',
        'node_agg': 'nodes',
        'zone_agg': 'pz',
        'reduce': 'price',
        'kind': 'gen',
    },
    'Storage': {
        'key': 'storage_power',
        'ylabel': 'Storage P (MW, +discharge/−charge)',
        'entity_agg': 'storage',
        'node_agg': 'node',
        'zone_agg': 'pz',
        'reduce': 'sum',
        'kind': 'storage',
    },
    'SoC': {
        'key': 'storage_soc',
        'ylabel': 'SoC',
        'entity_agg': 'storage',
        'node_agg': 'node',
        'zone_agg': 'pz',
        'reduce': 'mean',
        'kind': 'storage',
    },
    'H2': {
        'key': 'hydrogen_P_e',
        'ylabel': 'Electrolyser P (MW)',
        'entity_agg': 'h2',
        'node_agg': 'node',
        'zone_agg': 'pz',
        'reduce': 'sum',
        'kind': 'h2',
    },
    'H2 mass': {
        'key': 'hydrogen_mass_H2',
        'ylabel': 'H₂ mass (kg)',
        'entity_agg': 'h2',
        'node_agg': 'node',
        'zone_agg': 'pz',
        'reduce': 'sum',
        'kind': 'h2',
    },
}

_FAMILY_ORDER = (
    'Power', 'Ren Sources', 'Generators', 'Prices', 'Storage', 'SoC', 'H2', 'H2 mass',
)

# Distinct hues for seasons; elements within a season vary lightness on that hue.
_SEASON_HUES = (0.58, 0.08, 0.33, 0.0, 0.75, 0.45)

_ENTITY_DEFAULT_SELECT = 8
_SEASON_ELEMENT_SEP = ' | '

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
        'fontFamily': _FONT_STACK,
        'color': _THEME['text_primary'],
    }


_LOGO_FILENAME = 'pyflow_logo.svg'


def _assets_logo_path():
    return os.path.join(os.path.dirname(__file__), 'assets', _LOGO_FILENAME)


def _attach_app_css(app):
    # Dropdown/hover/scrollbar CSS is auto-loaded from pyflow_acdc/assets/dashboard.css.
    # Favicon + DM Sans (loaded once for all Dash apps in this package).
    app.index_string = """<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link rel="icon" type="image/svg+xml" href="/assets/pyflow_logo.svg">
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&display=swap" rel="stylesheet">
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>"""


def _sidebar_header(app, title, subtitle=None):
    """Sidebar header: full-width logo, then title (+ optional subtitle) + Hide."""
    logo_path = _assets_logo_path()
    if not os.path.isfile(logo_path):
        raise FileNotFoundError(f'Dash logo asset missing: {logo_path}')
    children = [
        html.Div(
            style={'display': 'flex', 'justifyContent': 'flex-end', 'marginBottom': '10px'},
            children=[
                html.Button('Hide ✕', id='hide-sidebar', n_clicks=0, style=_remove_btn_style()),
            ],
        ),
        html.Img(
            src=app.get_asset_url(_LOGO_FILENAME),
            style={
                'width': '100%',
                'height': 'auto',
                'display': 'block',
                'marginBottom': '14px',
            },
        ),
        html.H2(title, style={
            'color': _THEME['sidebar_text_top'],
            'fontSize': '18px',
            'fontWeight': '600',
            'fontFamily': _FONT_STACK,
            'margin': '0 0 4px 0',
            'lineHeight': '1.3',
        }),
    ]
    if subtitle:
        children.append(html.Div(subtitle, style={
            'color': _THEME['sidebar_caption'],
            'fontSize': '12px',
            'fontFamily': _FONT_STACK,
            'marginBottom': '14px',
            'letterSpacing': '0.02em',
        }))
    else:
        children.append(html.Div(style={'marginBottom': '10px'}))
    children.append(html.Hr(style={
        'border': 'none',
        'borderTop': f"1px solid {_THEME['sidebar_border']}",
        'margin': '0 0 14px 0',
    }))
    return html.Div(children=children)


def _content_shell(page_title=None):
    """Shared content region: sticky top bar + panel-graphs container."""
    bar_children = [
        html.Button('☰ Hide options', id='toggle-sidebar', n_clicks=0,
                    style=_toggle_btn_style()),
    ]
    if page_title:
        bar_children.insert(0, html.Div(page_title, style={
            'flex': 1,
            'fontFamily': _FONT_STACK,
            'fontSize': '15px',
            'fontWeight': '600',
            'color': _THEME['text_primary'],
            'alignSelf': 'center',
        }))
    return html.Div(id='content', style=_content_style(True), children=[
        html.Div(
            style={
                'position': 'sticky',
                'top': 0,
                'zIndex': 5,
                'display': 'flex',
                'alignItems': 'center',
                'gap': '12px',
                'justifyContent': 'space-between',
                'backgroundColor': _THEME['bg_secondary'],
                'borderBottom': f"1px solid {_THEME['border']}",
                'padding': '10px 4px 12px 4px',
                'marginBottom': '16px',
            },
            children=bar_children,
        ),
        html.Div(id='panel-graphs'),
    ])


def _register_sidebar_toggle(app):
    """Attach the shared show/hide sidebar callback (identical for every app)."""
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


def _register_panel_manager(app):
    """Attach the shared add/remove plot-panel callback (identical for every app)."""
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
        'fontFamily': _FONT_STACK,
        'marginBottom': '0',
        'flex': '0 0 auto',
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
        font=dict(family=_FONT_STACK, color=_THEME['fig_font'], size=13),
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='left',
            x=0,
            bgcolor='rgba(255,255,255,0.92)',
            bordercolor=_THEME['border'],
            borderwidth=1,
            font=dict(size=12),
        ),
        margin=dict(l=56, r=24, t=88 if show_title else 56, b=52),
        title=dict(font=dict(family=_FONT_STACK, size=20, color=_THEME['text_primary'])),
    )
    axis = dict(
        showgrid=True,
        gridwidth=1,
        gridcolor=_THEME['fig_grid'],
        zeroline=True,
        zerolinewidth=1,
        zerolinecolor=_THEME['border'],
        linecolor=_THEME['border'],
        tickfont=dict(size=11, color=_THEME['text_secondary']),
        title_font=dict(size=12, color=_THEME['text_secondary']),
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


def _hex_from_hls(h, lightness, saturation):
    r, g, b = colorsys.hls_to_rgb(h, lightness, saturation)
    return f'#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}'


def _season_trace_color(season, element, seasons, elements, aggregation):
    """One hue per season; lightness steps for elements within that season."""
    seasons = list(seasons)
    try:
        si = seasons.index(season)
    except ValueError:
        si = 0
    hue = _SEASON_HUES[si % len(_SEASON_HUES)]
    if aggregation == 'total' or not elements or element is None:
        return _hex_from_hls(hue, 0.45, 0.85)
    elements = list(elements)
    try:
        ei = elements.index(element)
    except ValueError:
        ei = 0
    n = len(elements)
    light = 0.30 if n <= 1 else 0.28 + 0.44 * ei / (n - 1)
    return _hex_from_hls(hue, light, 0.85)


def _zone_name(value):
    if value is None:
        return None
    return value.name if hasattr(value, 'name') else value


def _family_entity_records(grid, family):
    """Topology records for a family: name, node, zone/pz, is_ext_grid."""
    if family not in _FAMILY_SPECS:
        raise ValueError(f"Unknown family {family!r}")
    kind = _FAMILY_SPECS[family]['kind']
    records = []
    if kind == 'power':
        # Overview family has no topology entities; columns come from totals.
        return []
    if kind == 'ren':
        for rs in getattr(grid, 'RenSources', None) or []:
            records.append({
                'name': rs.name,
                'node': rs.Node,
                'zone': _zone_name(rs.Ren_source_zone),
                'pz': _zone_name(rs.PZ),
                'is_ext_grid': False,
            })
    elif kind == 'gen':
        gens = list(getattr(grid, 'Generators', None) or []) + list(
            getattr(grid, 'Generators_DC', None) or []
        )
        for gen in gens:
            node = getattr(gen, 'Node_AC', None) or getattr(gen, 'Node_DC', None)
            records.append({
                'name': gen.name,
                'node': node,
                'zone': None,
                'pz': _zone_name(getattr(gen, 'PZ', None)),
                'is_ext_grid': bool(getattr(gen, 'is_ext_grid', False)),
            })
    elif kind == 'storage':
        for st in getattr(grid, 'storage_elements', None) or []:
            records.append({
                'name': st.name,
                'node': st.Node,
                'zone': None,
                'pz': _zone_name(getattr(st, 'PZ', None)),
                'is_ext_grid': False,
            })
    elif kind == 'h2':
        for el in getattr(grid, 'electrolysers', None) or []:
            records.append({
                'name': el.name,
                'node': el.Node,
                'zone': None,
                'pz': _zone_name(getattr(el, 'PZ', None)),
                'is_ext_grid': False,
            })
    else:
        raise ValueError(f"Unknown family kind {_FAMILY_SPECS[family]['kind']!r}")
    return records


def _family_group_key(record, family, aggregation):
    spec = _FAMILY_SPECS[family]
    if aggregation == 'total':
        return 'total'
    if aggregation == spec['entity_agg']:
        return record['name']
    if aggregation == spec['node_agg']:
        if record['node'] is None:
            return None
        return record['node']
    if aggregation == spec['zone_agg']:
        key = record['zone'] if spec['zone_agg'] == 'zone' else record['pz']
        return key
    raise ValueError(
        f"Aggregation {aggregation!r} is not valid for family {family!r}"
    )


def _family_result_df(results, grid, family, source, s_base=None):
    """Raw per-element dataframe for a family from a results dict, by ``source``.

    ``source='window'`` reads ``_FAMILY_SPECS[family]['key']`` (frame-indexed).
    ``source='ts'`` reads the family's ``ts_key`` from ``time_series_results`` and
    scales it (``ts_scale='S_base'`` → ×``s_base`` [default ``grid.S_base``]).
    Returns ``None`` when the family has no series for that source.
    """
    spec = _FAMILY_SPECS[family]
    if spec.get('kind') == 'power':
        if source != 'window':
            return None
        return _window_power_overview_df(results, grid)
    if source == 'window':
        return _frame_column_to_index(results.get(spec['key']))
    if source == 'ts':
        ts_key = spec.get('ts_key')
        if ts_key is None:
            return None
        df = results.get(ts_key)
        if df is None:
            return None
        base = grid.S_base if s_base is None else s_base
        scale = base if spec.get('ts_scale') == 'S_base' else 1.0
        return df * scale
    raise ValueError(f"Unknown family source {source!r}")


def available_family_aggregations(grid, family, window_opf_results=None, source='window', s_base=None):
    """Return aggregation levels available for ``family`` given grid/results."""
    if family not in _FAMILY_SPECS:
        raise ValueError(f"Unknown family {family!r}")
    spec = _FAMILY_SPECS[family]
    if spec.get('kind') == 'power':
        if window_opf_results is None:
            return [spec['entity_agg']]
        df = _family_result_df(window_opf_results, grid, family, source, s_base=s_base)
        if df is None or df.empty:
            return []
        return [spec['entity_agg']]
    ordered = [spec['entity_agg'], spec['node_agg'], spec['zone_agg'], 'total']
    if window_opf_results is None:
        return ordered
    df = _family_result_df(window_opf_results, grid, family, source, s_base=s_base)
    if df is None or df.empty:
        return []
    records = [
        r for r in _family_entity_records(grid, family) if r['name'] in df.columns
    ]
    if not records:
        # Results columns with no topology match: entity + total only.
        return [spec['entity_agg'], 'total']
    out = [spec['entity_agg']]
    if any(r['node'] for r in records):
        out.append(spec['node_agg'])
    zone_field = 'zone' if spec['zone_agg'] == 'zone' else 'pz'
    if any(r.get(zone_field) for r in records):
        out.append(spec['zone_agg'])
    out.append('total')
    return out


def available_dash_families(grid, window_opf_results, source='window', s_base=None):
    """Families that have at least one series in ``window_opf_results`` for ``source``."""
    if not window_opf_results:
        return []
    out = []
    for family in _FAMILY_ORDER:
        df = _family_result_df(window_opf_results, grid, family, source, s_base=s_base)
        if df is not None and not df.empty:
            out.append(family)
    return out


def _reduce_group(df, names, reduce, records_by_name):
    sub = df[names]
    if reduce == 'sum':
        return sub.sum(axis=1)
    if reduce == 'mean':
        return sub.mean(axis=1)
    if reduce == 'price':
        ext = [n for n in names if records_by_name.get(n, {}).get('is_ext_grid')]
        pick = ext[0] if ext else names[0]
        return sub[pick]
    raise ValueError(f"Unknown reduce mode {reduce!r}")


def resolve_family_df(window_opf_results, grid, family, aggregation, source='window', s_base=None):
    """Aggregate one result into columns at ``aggregation`` level.

    Returns ``(DataFrame, ylabel)``. Columns are group keys (``total`` for total).
    ``source`` selects window (frame-indexed) vs ts (``time_series_results``) keys.
    """
    if family not in _FAMILY_SPECS:
        raise ValueError(f"Unknown family {family!r}")
    spec = _FAMILY_SPECS[family]
    aggs = available_family_aggregations(grid, family, window_opf_results, source=source, s_base=s_base)
    if aggregation not in aggs:
        raise ValueError(
            f"Aggregation {aggregation!r} not available for family {family!r}; "
            f"available={aggs}"
        )
    df = _family_result_df(window_opf_results, grid, family, source, s_base=s_base)
    if df is None or df.empty:
        return None, spec['ylabel']

    records = _family_entity_records(grid, family)
    records_by_name = {r['name']: r for r in records}
    present = [r for r in records if r['name'] in df.columns]

    if not present:
        # Fall back to raw columns as entities (no topology grouping).
        if aggregation == spec['entity_agg']:
            return df.copy(), spec['ylabel']
        if aggregation == 'total':
            reduce = 'mean' if spec['reduce'] in ('mean', 'price') else 'sum'
            if reduce == 'sum':
                out = df.sum(axis=1).to_frame('total')
            else:
                out = df.mean(axis=1).to_frame('total')
            return out, spec['ylabel']
        raise ValueError(
            f"No topology records for family {family!r}; "
            f"only aggregations {spec['entity_agg']!r} and 'total' work"
        )

    groups = {}
    for rec in present:
        gk = _family_group_key(rec, family, aggregation)
        if gk is None:
            continue
        groups.setdefault(gk, []).append(rec['name'])
    if not groups:
        raise ValueError(
            f"No groups for family={family!r} aggregation={aggregation!r}"
        )

    cols = {
        gk: _reduce_group(df, names, spec['reduce'], records_by_name)
        for gk, names in groups.items()
    }
    return pd.DataFrame(cols), spec['ylabel']


def family_element_options(grid, window_opf_results, family, aggregation, source='window', s_base=None):
    """Element names for the Components checklist (empty when aggregation is total)."""
    if aggregation == 'total':
        return []
    df, _ = resolve_family_df(window_opf_results, grid, family, aggregation, source=source, s_base=s_base)
    if df is None or df.empty:
        return []
    return list(df.columns)


def resolve_season_family_df(grid, family, aggregation, seasons=None, elements=None):
    """Build season-compare frame: columns = seasons, or ``season | element``.

    Parameters
    ----------
    seasons : list, optional
        Seasons to include (default: all).
    elements : list, optional
        Element keys when aggregation is not ``total``. Ignored for ``total``.
    """
    raw = getattr(grid, 'season_window_compare_raw', None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError("grid.season_window_compare_raw is missing or empty")
    season_names = list(seasons) if seasons is not None else list(raw.keys())
    if not season_names:
        raise ValueError("No seasons selected")

    series = {}
    ylabel = _FAMILY_SPECS[family]['ylabel']
    for season in season_names:
        if season not in raw:
            raise ValueError(f"Unknown season {season!r}")
        df, ylabel = resolve_family_df(raw[season], grid, family, aggregation)
        if df is None or df.empty:
            continue
        aligned = df.copy()
        aligned.index = range(len(aligned))
        if aggregation == 'total':
            col = 'total' if 'total' in aligned.columns else aligned.columns[0]
            series[season] = aligned[col]
            continue
        el_list = list(elements) if elements is not None else list(aligned.columns)
        for el in el_list:
            if el not in aligned.columns:
                continue
            series[f'{season}{_SEASON_ELEMENT_SEP}{el}'] = aligned[el]

    if not series:
        return None, ylabel
    return pd.DataFrame(series), ylabel


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


def _window_power_overview_df(window_opf_results, grid):
    """Power-family overview: one series per input class (ren, gen_AC, gen_DC, H2, BESS).

    Generators are split using ``grid.Generators`` vs ``grid.Generators_DC``.
    Raises if ``gen_power`` is present but no column matches either list.
    """
    series = {}
    for label, key in (
        ('Total ren', 'ren_power'),
        ('Total H2', 'hydrogen_P_e'),
        ('Total BESS', 'storage_power'),
    ):
        df = _frame_column_to_index(window_opf_results.get(key))
        if df is None or df.empty:
            continue
        series[label] = df.sum(axis=1)

    gdf = _frame_column_to_index(window_opf_results.get('gen_power'))
    if gdf is not None and not gdf.empty:
        ac_names = [
            g.name for g in (getattr(grid, 'Generators', None) or [])
            if g.name in gdf.columns
        ]
        dc_names = [
            g.name for g in (getattr(grid, 'Generators_DC', None) or [])
            if g.name in gdf.columns
        ]
        if not ac_names and not dc_names:
            raise ValueError(
                "Power family: gen_power has columns but none match "
                "grid.Generators or grid.Generators_DC"
            )
        if ac_names:
            series['Total gen_AC'] = gdf[ac_names].sum(axis=1)
        if dc_names:
            series['Total gen_DC'] = gdf[dc_names].sum(axis=1)

    if not series:
        return None
    return pd.DataFrame(series)


def _hour_aligned_sum_series(df):
    """Sum columns of a frame-indexed df and reindex to 0…n−1 for overlay."""
    if df is None or df.empty:
        return None
    s = df.sum(axis=1) if len(df.columns) else None
    if s is None or s.empty:
        return None
    out = s.copy()
    out.index = range(len(out))
    return out


def _price_series_by_label(window_opf_results, grid):
    """Return {label: Series} of hourly prices for season compare.

    With price zones: one series per zone (from a gen on a zone bus, preferring
    extgrids). Without zones: one series per ``gen_price`` column (prefer
    extgrids when present).
    """
    gprice = _frame_column_to_index(window_opf_results.get('gen_price'))
    if gprice is None or gprice.empty:
        return {}

    out = {}
    price_zones = list(getattr(grid, 'Price_Zones', None) or [])
    if price_zones:
        for pz in price_zones:
            picked = None
            # Prefer extgrid on a zone node, then any gen with a gen_price column.
            candidates = []
            for node in getattr(pz, 'nodes_AC', []):
                for gen in getattr(node, 'connected_gen', []):
                    if gen.name not in gprice.columns:
                        continue
                    candidates.append(gen)
            ext = [g for g in candidates if getattr(g, 'is_ext_grid', False)]
            ordered = ext + [g for g in candidates if g not in ext]
            if ordered:
                picked = gprice[ordered[0].name]
            if picked is not None:
                s = picked.copy()
                s.index = range(len(s))
                out[f'Price: {pz.name}'] = s
        return out

    gens = list(getattr(grid, 'Generators', None) or []) + list(
        getattr(grid, 'Generators_DC', None) or []
    )
    ext_names = {
        g.name for g in gens if getattr(g, 'is_ext_grid', False) and g.name in gprice.columns
    }
    cols = [c for c in gprice.columns if c in ext_names] or list(gprice.columns)
    for col in cols:
        s = gprice[col].copy()
        s.index = range(len(s))
        out[f'Price: {col}'] = s
    return out


def _ren_by_zone_series(window_opf_results, grid):
    """Return {label: Series} of total ren power per RenSource zone."""
    ren = _frame_column_to_index(window_opf_results.get('ren_power'))
    if ren is None or ren.empty:
        return {}
    zones = list(getattr(grid, 'RenSource_zones', None) or [])
    if not zones:
        return {}
    out = {}
    for zone in zones:
        names = [
            rs.name
            for rs in getattr(zone, 'RenSources', [])
            if rs.name in ren.columns
        ]
        if not names:
            continue
        s = ren[names].sum(axis=1).copy()
        s.index = range(len(s))
        out[f'Ren: {zone.name}'] = s
    return out


def build_season_window_compare(season_to_window_results, grid=None):
    """Build compare tables: metric → DataFrame (index=hour, columns=season).

    Always includes total-power overlays when present. With ``grid``, also adds
    SoC, H₂ mass, prices (PZ or gen), and ren-by-zone when data exist.

    Parameters
    ----------
    season_to_window_results : dict
        Mapping ``season_name -> window_opf_results``.
    grid : Grid, optional
        Topology for price-zone / ren-zone aggregation.
    """
    if not season_to_window_results:
        raise ValueError("season_to_window_results is empty")

    seasons = list(season_to_window_results.keys())
    compare = {}
    ylabels = {}

    # --- Total power (required base) ---
    per_season_totals = {}
    for season, res in season_to_window_results.items():
        totals = _window_total_power_df(res)
        if totals is None or totals.empty:
            raise ValueError(
                f"Season {season!r} has no total-power series in window_opf_results"
            )
        totals = totals.copy()
        totals.index = range(len(totals))
        per_season_totals[season] = totals

    for metric, _ in _WINDOW_TOTAL_POWER_SERIES:
        cols = {}
        for season, totals in per_season_totals.items():
            if metric in totals.columns:
                cols[season] = totals[metric]
        if cols:
            compare[metric] = pd.DataFrame(cols)
            ylabels[metric] = 'Power (MW)'

    # --- SoC / H2 mass (one line per season) ---
    for metric, key, ylab in (
        ('SoC', 'storage_soc', 'SoC'),
        ('H2 mass', 'hydrogen_mass_H2', 'H₂ mass (kg)'),
    ):
        cols = {}
        for season, res in season_to_window_results.items():
            s = _hour_aligned_sum_series(
                _frame_column_to_index(res.get(key))
            )
            if s is not None:
                cols[season] = s
        if cols:
            compare[metric] = pd.DataFrame(cols)
            ylabels[metric] = ylab

    if grid is not None:
        # --- Prices: one plot choice per zone/gen, columns = seasons ---
        price_by_season = {
            season: _price_series_by_label(res, grid)
            for season, res in season_to_window_results.items()
        }
        price_labels = set()
        for d in price_by_season.values():
            price_labels.update(d.keys())
        for label in sorted(price_labels):
            cols = {}
            for season in seasons:
                s = price_by_season[season].get(label)
                if s is not None:
                    cols[season] = s
            if cols:
                compare[label] = pd.DataFrame(cols)
                ylabels[label] = 'Price (EUR/MWh)'

        # --- Ren by zone ---
        ren_by_season = {
            season: _ren_by_zone_series(res, grid)
            for season, res in season_to_window_results.items()
        }
        ren_labels = set()
        for d in ren_by_season.values():
            ren_labels.update(d.keys())
        for label in sorted(ren_labels):
            cols = {}
            for season in seasons:
                s = ren_by_season[season].get(label)
                if s is not None:
                    cols[season] = s
            if cols:
                compare[label] = pd.DataFrame(cols)
                ylabels[label] = 'Power (MW)'

    if not compare:
        raise ValueError("No overlapping metrics across seasons")
    return compare, ylabels


def attach_season_window_compare(grid, season_to_window_results):
    """Store season-compare tables + raw results on ``grid`` for Dash."""
    if not season_to_window_results:
        raise ValueError("season_to_window_results is empty")
    grid.season_window_compare_raw = dict(season_to_window_results)
    compare, ylabels = build_season_window_compare(
        season_to_window_results, grid=grid
    )
    grid.season_window_compare = compare
    grid.season_window_compare_ylabels = ylabels
    grid.season_window_compare_run = True
    return grid


def _season_compare_usable(grid):
    raw = getattr(grid, 'season_window_compare_raw', None)
    flat = getattr(grid, 'season_window_compare', None)
    return (
        getattr(grid, 'season_window_compare_run', False)
        and (
            (isinstance(raw, dict) and bool(raw))
            or (isinstance(flat, dict) and bool(flat))
        )
    )


def _get_df_and_label_from_season_compare(grid, plotting_choice):
    compare = grid.season_window_compare
    df = compare.get(plotting_choice)
    if df is None:
        return None, ''
    labels = getattr(grid, 'season_window_compare_ylabels', None) or {}
    return df, labels.get(plotting_choice, plotting_choice)


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
        return _get_df_and_label_from_season_compare(grid, plotting_choice)
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
    elif plotting_choice in ('Storage SoC', 'SoC'):
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


def plot_season_family_dash(
    grid,
    family,
    aggregation,
    seasons,
    elements,
    x_limits=None,
    y_limits=None,
    layout='overlay',
):
    """Season-compare figure with family aggregation and season color map.

    ``layout='overlay'`` draws all seasons on one axes (default); ``'split'``
    draws one horizontal subplot per season.
    """
    df, y_label = resolve_season_family_df(
        grid, family, aggregation, seasons=seasons, elements=elements
    )
    title = f"Season compare: {family} / {aggregation}"
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title="Hour", yaxis_title=y_label or "Value")
        return _apply_fig_theme(fig)

    season_order = list(seasons) if seasons else []
    element_order = list(elements) if elements else []

    def _split_col(col):
        if aggregation == 'total' or _SEASON_ELEMENT_SEP not in col:
            return col, None
        return tuple(col.split(_SEASON_ELEMENT_SEP, 1))

    if layout == 'split':
        seasons_present = season_order or sorted({_split_col(c)[0] for c in df.columns})
        # Split: subplots already separate seasons, so color by the *variable*
        # (element) and keep it identical across every season subplot.
        elements_present = element_order or list(dict.fromkeys(
            e for e in (_split_col(c)[1] for c in df.columns) if e is not None
        ))
        fig = make_subplots(
            rows=1, cols=max(len(seasons_present), 1),
            subplot_titles=seasons_present, shared_yaxes=True,
        )
        shown = set()
        for ci, season in enumerate(seasons_present):
            for col in df.columns:
                s, element = _split_col(col)
                if s != season:
                    continue
                if element is None:
                    name = str(family)
                    color = _SERIES_COLORS[0]
                else:
                    name = str(element)
                    idx = elements_present.index(element) if element in elements_present else 0
                    color = _SERIES_COLORS[idx % len(_SERIES_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=df.index, y=df[col], mode='lines', name=name,
                        line=dict(color=color, width=2),
                        showlegend=name not in shown,
                    ),
                    row=1, col=ci + 1,
                )
                shown.add(name)
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=24, color=_THEME['text_primary']),
                x=0.5, xanchor='center',
            ),
            showlegend=True,
        )
        _apply_fig_theme(fig)
        fig.update_xaxes(title_text="Hour")
        fig.update_yaxes(title_text=y_label, row=1, col=1)
        if x_limits is not None and x_limits[0] is not None and x_limits[1] is not None:
            fig.update_xaxes(range=list(x_limits))
        if y_limits is not None and y_limits[0] is not None and y_limits[1] is not None:
            fig.update_yaxes(range=list(y_limits))
        return fig

    fig = go.Figure()
    for col in df.columns:
        season, element = _split_col(col)
        color = _season_trace_color(
            season, element, season_order, element_order, aggregation
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col], mode='lines', name=str(col),
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, color=_THEME['text_primary']),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title="Hour",
        yaxis_title=y_label,
        legend_title="Season" if aggregation == 'total' else "Season | element",
        showlegend=True,
    )
    _apply_fig_theme(fig)
    if x_limits is not None and x_limits[0] is not None and x_limits[1] is not None:
        fig.update_xaxes(range=list(x_limits))
    if y_limits is not None and y_limits[0] is not None and y_limits[1] is not None:
        fig.update_yaxes(range=list(y_limits))
    return fig


def plot_window_family_dash(
    grid,
    family,
    aggregation,
    elements,
    x_limits=None,
    y_limits=None,
    *,
    results=None,
    source='window',
    x_axis_label='Frame',
    s_base=None,
):
    """Family-aggregation figure for window (``source='window'``) or TS (``source='ts'``) results."""
    if results is None:
        results = grid.window_opf_results
    df, y_label = resolve_family_df(results, grid, family, aggregation, source=source, s_base=s_base)
    title = f"{family} / {aggregation}"
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(title=title, xaxis_title=x_axis_label, yaxis_title=y_label or "Value")
        return _apply_fig_theme(fig)

    cols = list(elements) if elements else list(df.columns)
    if aggregation == 'total':
        cols = list(df.columns)
    fig = go.Figure()
    for i, col in enumerate(cols):
        if col not in df.columns:
            continue
        color = _SERIES_COLORS[i % len(_SERIES_COLORS)]
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df[col], mode='lines', name=str(col),
                line=dict(color=color, width=2),
            )
        )
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=24, color=_THEME['text_primary']),
            x=0.5,
            xanchor='center',
        ),
        xaxis_title=x_axis_label,
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


def _family_panel_control_card(i, *, compare, default_family, family_options):
    """Sidebar card: family + aggregation (+ seasons when compare)."""
    children = [
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
        html.Label('Family', style=_label_style(on_dark=True)),
        dcc.Dropdown(
            id={'type': 'plot-family', 'index': i},
            options=[{'label': f, 'value': f} for f in family_options],
            value=default_family,
            clearable=False,
            style={'marginBottom': '10px'},
        ),
        html.Label('Aggregation', style=_label_style(on_dark=True)),
        dcc.Dropdown(
            id={'type': 'plot-agg', 'index': i},
            options=[],
            value=None,
            clearable=False,
            style={'marginBottom': '10px'},
        ),
    ]
    if compare:
        children.extend([
            html.Label('Seasons', style=_label_style(on_dark=True)),
            dcc.Checklist(
                id={'type': 'plot-seasons', 'index': i},
                options=[],
                value=[],
                style={'marginBottom': '10px', 'color': _THEME['sidebar_text']},
            ),
        ])
    children.extend([
        html.Label('Elements', style=_label_style(on_dark=True)),
        dcc.Checklist(
            id={'type': 'plot-elements', 'index': i},
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
    return html.Div(style=_sidebar_panel_card_style(), children=children)


def _default_element_selection(options, aggregation):
    if aggregation == 'total' or not options:
        return []
    entity_aggs = {s['entity_agg'] for s in _FAMILY_SPECS.values()}
    if aggregation in entity_aggs and len(options) > _ENTITY_DEFAULT_SELECT:
        return options[:_ENTITY_DEFAULT_SELECT]
    return list(options)


def _family_auto_ylimits(df, family):
    if df is None or df.empty:
        return 0, 1
    y_min = int(min(0, df.min().min() - 5))
    if family == 'SoC':
        return 0, 1
    y_max = int(df.max().max() + 10)
    return y_min, y_max


def _panel_graph_card(i, height=None):
    return html.Div(
        style=_card_style(
            backgroundColor='white',
            borderLeft=f"3px solid {_THEME['accent']}",
        ),
        children=[
            dcc.Loading(
                type='dot',
                color=_THEME['accent'],
                children=dcc.Graph(
                    id={'type': 'plot-graph', 'index': i},
                    style=_graph_style(height),
                    responsive=True,
                    config={'displaylogo': False},
                ),
            ),
        ],
    )


def create_dash_app(grid):
    """Dash app for sequential TS OPF results (``grid.time_series_results``).

    Family mode exposes the **Generators** family (from ``real_power_opf``) with
    entity/total aggregation; Classic mode keeps the full TS plot-type list.
    """
    name = getattr(grid, 'name', None) or 'grid'
    return _build_family_dash_app(
        grid,
        title=name,
        subtitle='Time series',
        page_title='Time series dashboard',
        compare=False,
        sample_results=grid.time_series_results,
        x_axis_label='Time',
        allow_classic=True,
        classic_choices=_TS_PLOT_CHOICES,
        classic_plot_fn=plot_TS_res_dash,
        source='ts',
    )


def _build_family_dash_app(
    grid,
    *,
    title,
    compare,
    sample_results,
    x_axis_label,
    allow_classic=False,
    classic_choices=None,
    classic_plot_fn=None,
    source='window',
    subtitle=None,
    page_title=None,
):
    """Multi-plot Dash with Family / Aggregation (+ Seasons when compare).

    When ``allow_classic`` is True, a Mode radio switches between classic plot
    types and family aggregation. ``source`` selects window vs ts family keys.
    """
    families = available_dash_families(grid, sample_results, source=source)
    if not families and not (allow_classic and classic_choices):
        raise ValueError('No plottable families in results')
    default_family = families[0] if families else None
    all_seasons = list(getattr(grid, 'season_window_compare_raw', {}) or {})

    app = dash.Dash(__name__)
    _attach_app_css(app)

    global_children = []
    if allow_classic and classic_choices:
        global_children.append(html.Label('Mode', style=_label_style(on_dark=True)))
        global_children.append(dcc.RadioItems(
            id='view-mode',
            options=[
                {'label': 'Classic', 'value': 'classic'},
                {'label': 'Family', 'value': 'family'},
            ],
            value='family' if families else 'classic',
            style={'marginBottom': '12px', 'color': _THEME['sidebar_text']},
        ))
    else:
        global_children.append(dcc.Store(id='view-mode', data='family'))

    if compare:
        global_children.append(html.Label('Compare layout', style=_label_style(on_dark=True)))
        global_children.append(dcc.RadioItems(
            id='compare-layout',
            options=[
                {'label': ' Overlay', 'value': 'overlay'},
                {'label': ' Split (subplots)', 'value': 'split'},
            ],
            value='overlay',
            style={'marginBottom': '12px', 'color': _THEME['sidebar_text']},
        ))

    global_children.extend([
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
    ])

    app.layout = html.Div([
        dcc.Store(id='sidebar-open', data=True),
        dcc.Store(id='plot-panels', data=[0]),
        html.Div(id='sidebar', style=_sidebar_style(True), children=[
            _sidebar_header(app, title, subtitle=subtitle),
            html.Div(id='global-controls', children=global_children),
            html.Button('➕ Add plot', id='add-plot', n_clicks=0, style=_btn_style()),
            html.Div(id='panel-controls'),
        ]),
        _content_shell(page_title=page_title or title),
    ])

    _register_sidebar_toggle(app)
    _register_panel_manager(app)

    classic_dd = [{'label': c, 'value': c} for c in (classic_choices or [])]

    @app.callback(
        [Output('panel-controls', 'children'),
         Output('panel-graphs', 'children')],
        [Input('plot-panels', 'data'),
         Input('view-mode', 'data') if not (allow_classic and classic_choices)
         else Input('view-mode', 'value')],
    )
    def _render_panels(panels, view_mode):
        panels = panels or [0]
        mode = view_mode or 'family'
        ctrls, graphs = [], []
        for pos, i in enumerate(panels):
            if mode == 'classic' and classic_dd:
                dc = classic_choices[min(pos, len(classic_choices) - 1)]
                ctrls.append(_panel_control_card(i, dc, classic_dd))
            else:
                fam = default_family if default_family else families[0]
                if pos == 1 and len(families) > 1:
                    fam = families[1]
                ctrls.append(_family_panel_control_card(
                    i, compare=compare, default_family=fam, family_options=families,
                ))
            graphs.append(_panel_graph_card(i))
        return ctrls, graphs

    # --- Family panel: aggregation options when family changes ---
    @app.callback(
        [Output({'type': 'plot-agg', 'index': MATCH}, 'options'),
         Output({'type': 'plot-agg', 'index': MATCH}, 'value')],
        [Input({'type': 'plot-family', 'index': MATCH}, 'value')],
    )
    def _family_aggs(family):
        aggs = available_family_aggregations(grid, family, sample_results, source=source)
        default = 'total' if 'total' in aggs else (aggs[0] if aggs else None)
        return [{'label': a, 'value': a} for a in aggs], default

    if compare:
        @app.callback(
            [Output({'type': 'plot-seasons', 'index': MATCH}, 'options'),
             Output({'type': 'plot-seasons', 'index': MATCH}, 'value'),
             Output({'type': 'plot-elements', 'index': MATCH}, 'options'),
             Output({'type': 'plot-elements', 'index': MATCH}, 'value'),
             Output({'type': 'plot-ymin', 'index': MATCH}, 'value'),
             Output({'type': 'plot-ymax', 'index': MATCH}, 'value')],
            [Input({'type': 'plot-family', 'index': MATCH}, 'value'),
             Input({'type': 'plot-agg', 'index': MATCH}, 'value')],
        )
        def _season_series(family, aggregation):
            season_opts = [{'label': s, 'value': s} for s in all_seasons]
            season_vals = list(all_seasons)
            if not family or not aggregation:
                return season_opts, season_vals, [], [], 0, 1
            # Element options from first season's topology/results
            first = all_seasons[0] if all_seasons else None
            el_opts_list = []
            if first is not None and aggregation != 'total':
                el_opts_list = family_element_options(
                    grid, grid.season_window_compare_raw[first], family, aggregation
                )
            el_opts = [{'label': e, 'value': e} for e in el_opts_list]
            el_vals = _default_element_selection(el_opts_list, aggregation)
            df, _ = resolve_season_family_df(
                grid, family, aggregation, seasons=season_vals, elements=el_vals or None
            )
            ymin, ymax = _family_auto_ylimits(df, family)
            return season_opts, season_vals, el_opts, el_vals, ymin, ymax

        @app.callback(
            [Output({'type': 'plot-graph', 'index': ALL}, 'figure'),
             Output({'type': 'plot-graph', 'index': ALL}, 'style')],
            [Input({'type': 'plot-family', 'index': ALL}, 'value'),
             Input({'type': 'plot-agg', 'index': ALL}, 'value'),
             Input({'type': 'plot-seasons', 'index': ALL}, 'value'),
             Input({'type': 'plot-elements', 'index': ALL}, 'value'),
             Input({'type': 'plot-ymin', 'index': ALL}, 'value'),
             Input({'type': 'plot-ymax', 'index': ALL}, 'value'),
             Input('x-min', 'value'),
             Input('x-max', 'value'),
             Input('plot-height', 'value'),
             Input('compare-layout', 'value')],
        )
        def _draw_compare(families_sel, aggs, seasons_sel, elements_sel,
                          ymins, ymaxs, xmin, xmax, plot_height, compare_layout):
            x_limits = (xmin, xmax) if xmin is not None and xmax is not None else None
            height = _normalize_plot_height(
                plot_height if plot_height is not None else _DEFAULT_PLOT_HEIGHT
            )
            layout = compare_layout or 'overlay'
            figs, styles = [], []
            for family, agg, seas, els, ymin, ymax in zip(
                families_sel or [], aggs or [], seasons_sel or [],
                elements_sel or [], ymins or [], ymaxs or [],
            ):
                y_limits = (ymin, ymax) if ymin is not None and ymax is not None else None
                if not family or not agg:
                    fig = go.Figure()
                    _apply_fig_theme(fig)
                else:
                    fig = plot_season_family_dash(
                        grid, family, agg, seas or [], els or [],
                        x_limits=x_limits, y_limits=y_limits, layout=layout,
                    )
                figs.append(_apply_plot_height(fig, height))
                styles.append(_graph_style(height))
            return figs, styles
    else:
        @app.callback(
            [Output({'type': 'plot-elements', 'index': MATCH}, 'options'),
             Output({'type': 'plot-elements', 'index': MATCH}, 'value'),
             Output({'type': 'plot-ymin', 'index': MATCH}, 'value'),
             Output({'type': 'plot-ymax', 'index': MATCH}, 'value')],
            [Input({'type': 'plot-family', 'index': MATCH}, 'value'),
             Input({'type': 'plot-agg', 'index': MATCH}, 'value')],
        )
        def _window_family_series(family, aggregation):
            if not family or not aggregation:
                return [], [], 0, 1
            el_opts_list = family_element_options(
                grid, sample_results, family, aggregation, source=source
            )
            el_opts = [{'label': e, 'value': e} for e in el_opts_list]
            el_vals = _default_element_selection(el_opts_list, aggregation)
            df, _ = resolve_family_df(sample_results, grid, family, aggregation, source=source)
            ymin, ymax = _family_auto_ylimits(df, family)
            return el_opts, el_vals, ymin, ymax

        # Classic series callback (only when classic cards exist)
        if allow_classic and classic_choices:
            @app.callback(
                [Output({'type': 'plot-series', 'index': MATCH}, 'options'),
                 Output({'type': 'plot-series', 'index': MATCH}, 'value'),
                 Output({'type': 'plot-ymin', 'index': MATCH}, 'value'),
                 Output({'type': 'plot-ymax', 'index': MATCH}, 'value')],
                [Input({'type': 'plot-choice', 'index': MATCH}, 'value')],
            )
            def _classic_series(choice):
                cols = _columns_for_choice(grid, choice)
                ymin, ymax = _auto_ylimits(grid, choice)
                return [{'label': c, 'value': c} for c in cols], cols, ymin, ymax

        @app.callback(
            [Output({'type': 'plot-graph', 'index': ALL}, 'figure'),
             Output({'type': 'plot-graph', 'index': ALL}, 'style')],
            [Input('view-mode', 'value') if allow_classic and classic_choices
             else Input('view-mode', 'data'),
             Input({'type': 'plot-family', 'index': ALL}, 'value'),
             Input({'type': 'plot-agg', 'index': ALL}, 'value'),
             Input({'type': 'plot-elements', 'index': ALL}, 'value'),
             Input({'type': 'plot-choice', 'index': ALL}, 'value'),
             Input({'type': 'plot-series', 'index': ALL}, 'value'),
             Input({'type': 'plot-ymin', 'index': ALL}, 'value'),
             Input({'type': 'plot-ymax', 'index': ALL}, 'value'),
             Input('x-min', 'value'),
             Input('x-max', 'value'),
             Input('plot-height', 'value')],
        )
        def _draw_window(view_mode, families_sel, aggs, elements_sel,
                         choices, series, ymins, ymaxs, xmin, xmax, plot_height):
            x_limits = (xmin, xmax) if xmin is not None and xmax is not None else None
            height = _normalize_plot_height(
                plot_height if plot_height is not None else _DEFAULT_PLOT_HEIGHT
            )
            mode = view_mode or 'family'
            figs, styles = [], []
            if mode == 'classic' and classic_plot_fn is not None:
                for choice, sel, ymin, ymax in zip(
                    choices or [], series or [], ymins or [], ymaxs or []
                ):
                    y_limits = (ymin, ymax) if ymin is not None and ymax is not None else None
                    fig = classic_plot_fn(
                        grid, choice, sel or [], x_limits=x_limits, y_limits=y_limits
                    )
                    figs.append(_apply_plot_height(fig, height))
                    styles.append(_graph_style(height))
            else:
                for family, agg, els, ymin, ymax in zip(
                    families_sel or [], aggs or [], elements_sel or [],
                    ymins or [], ymaxs or [],
                ):
                    y_limits = (ymin, ymax) if ymin is not None and ymax is not None else None
                    if not family or not agg:
                        fig = go.Figure()
                        _apply_fig_theme(fig)
                    else:
                        fig = plot_window_family_dash(
                            grid, family, agg, els or [],
                            x_limits=x_limits, y_limits=y_limits,
                            results=sample_results, source=source,
                            x_axis_label=x_axis_label,
                        )
                    figs.append(_apply_plot_height(fig, height))
                    styles.append(_graph_style(height))
            return figs, styles

    return app


def create_window_dash_app(grid):
    """Dash app for coupled window NL OPF results (``grid.window_opf_results``).

    Mode **Family** (default when families exist): Family + Aggregation + Elements.
    Mode **Classic**: legacy flat plot-type list.
    """
    if not _window_opf_usable(grid):
        raise ValueError(
            "create_window_dash_app requires grid.window_opf_run and grid.window_opf_results"
        )
    choices = _available_window_plot_choices(grid)
    families = available_dash_families(grid, grid.window_opf_results)
    if not choices and not families:
        raise ValueError("window_opf_results has no plottable series")
    return _build_family_dash_app(
        grid,
        title=getattr(grid, 'name', None) or 'grid',
        subtitle='Window OPF',
        page_title='Window OPF dashboard',
        compare=False,
        sample_results=grid.window_opf_results,
        x_axis_label='Frame',
        allow_classic=bool(choices),
        classic_choices=choices,
        classic_plot_fn=plot_window_res_dash,
    )


def create_season_compare_dash_app(grid):
    """Dash app comparing seasons with Family / Aggregation / Seasons / Elements."""
    if not _season_compare_usable(grid):
        raise ValueError(
            "create_season_compare_dash_app requires grid.season_window_compare_run "
            "and grid.season_window_compare_raw (or season_window_compare)"
        )
    raw = getattr(grid, 'season_window_compare_raw', None)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(
            "create_season_compare_dash_app requires grid.season_window_compare_raw"
        )
    sample = next(iter(raw.values()))
    families = available_dash_families(grid, sample)
    if not families:
        raise ValueError("season_window_compare_raw has no plottable families")
    return _build_family_dash_app(
        grid,
        title=getattr(grid, 'name', None) or 'grid',
        subtitle='Season comparison',
        page_title='Season comparison',
        compare=True,
        sample_results=sample,
        x_axis_label='Hour',
        allow_classic=False,
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

    def _period_label(p):
        if p == 'base':
            return 'Base'
        return f'Period {p}'

    def _build_compare_fig(plot_type, cols, p1, p2, p3, x_limits, y_limits, layout='split'):
        if layout == 'overlay':
            fig = go.Figure()
            any_trace = False
            active = [p for p in (p1, p2, p3) if p != -1 and p in ts_inv]
            period_color = {
                p: _SERIES_COLORS[i % len(_SERIES_COLORS)] for i, p in enumerate(active)
            }
            for p in (p1, p2, p3):
                if p == -1 or p not in ts_inv:
                    continue
                snap = ts_inv[p]
                sub = plot_TS_res_from_ts(
                    snap['time_series_results'], snap['S_base'], plot_type, cols,
                    x_limits=x_limits, y_limits=y_limits, show_title=False,
                    legend_prefix=f'{_period_label(p)} | ',
                )
                # Overlay: one color per period so the periods are distinguishable.
                for tr in sub.data:
                    tr.line.color = period_color[p]
                    fig.add_trace(tr)
                    any_trace = True
            if not any_trace:
                fig.add_annotation(
                    text='Select at least one period column',
                    xref='paper', yref='paper', x=0.5, y=0.5, showarrow=False,
                )
            fig.update_layout(showlegend=True)
            _apply_fig_theme(fig, show_title=False)
            if x_limits is not None:
                fig.update_xaxes(range=x_limits)
            if y_limits is not None:
                fig.update_yaxes(range=y_limits)
            return fig

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
            _sidebar_header(app, grid_name or 'MP time series', subtitle='Investment periods'),
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
                html.Label('Compare layout', style=_label_style(on_dark=True)),
                dcc.RadioItems(
                    id='mp-compare-layout',
                    options=[
                        {'label': ' Split (subplots)', 'value': 'split'},
                        {'label': ' Overlay', 'value': 'overlay'},
                    ],
                    value='split',
                    style={'marginBottom': '12px', 'color': _THEME['sidebar_text']},
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
        _content_shell(page_title='TS by investment period'),
    ])

    _register_sidebar_toggle(app)

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
         Input('plot-height', 'value'),
         Input('mp-compare-layout', 'value')],
    )
    def _draw(mode, ps, p1, p2, p3, choices, series_list, ymins, ymaxs, xmin, xmax, gmin, gmax, plot_height, compare_layout):
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
                    _build_compare_fig(
                        choice, cols, p1, p2, p3, x_limits, y_limits,
                        layout=compare_layout or 'split',
                    ),
                    height,
                ))
            styles.append(_graph_style(height))
        return figs, styles

    return app


def run_mp_ts_dash(ts_inv, grid_name='MP time series', debug=True, use_reloader=False):
    app = create_mp_ts_dash(ts_inv, grid_name=grid_name)
    app.run(debug=debug, use_reloader=use_reloader)
