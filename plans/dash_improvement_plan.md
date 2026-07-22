# Dash visual improvement plan for pyflow_acdc

Redesign plan for the interactive Dash apps in
[`pyflow_acdc/Graph_Dash.py`](../pyflow_acdc/Graph_Dash.py).

Goal: keep current functionality, but make the apps look good and more capable:

1. **Left options menu** — move all controls into a collapsible / pop-up sidebar on the left.
2. **Better colors** — one shared theme, derived from the docs (Furo "SCADA panel" scheme).
3. **More than 2 graphs** — unlimited add/remove plot panels instead of the hard-coded 2.

Related tests: [`pyflow_tests/test_graph_dash.py`](../pyflow_tests/test_graph_dash.py).
Docs API page: [`docs/api/dash.rst`](../docs/api/dash.rst).

---

## 1. Decisions (resolved)

| ID | Topic | Decision |
|----|-------|----------|
| D1 | Scope | **All three apps**: dual-plot (`_build_dual_plot_dash_app` → TS + window) **and** multi-period (`create_mp_ts_dash`). |
| D2 | Plot count | **Unlimited** add/remove plot panels. Remove the boolean `show-plot-2` pattern. |
| D3 | Dependencies | **Core Dash only** (`dash`, `dcc`, `html`, `plotly`). No `dash-bootstrap-components` / `dash-mantine-components`. Inline styles + one shared theme dict. |
| D4 | Palette source | **Reuse the docs palette** (`docs/conf.py` Furo `light_css_variables`, "SCADA panel"). |
| D5 | Series colors | Keep the existing 10-color Plotly trace cycle for data series (already in `plot_TS_res_from_ts`). Theme only restyles chrome + figure frame (paper/plot bg, gridlines, fonts). |
| D6 | Public API | **No signature changes** to `run_dash` / `run_ts_dash` / `run_window_dash` / `run_mp_ts_dash` / `create_*`. Purely internal layout/callback refactor. |
| D7 | Behavior parity | Existing plot logic (`_get_df_and_label*`, limit auto-calc, stacked area, MP compare-3-columns) is preserved. |

---

## 2. Shared theme (Phase 0)

Add a module-level `_THEME` dict near the top of `Graph_Dash.py`, sourced from
`docs/conf.py` `light_css_variables`:

```python
_THEME = {
    # chrome
    'sidebar_bg':        '#142033',
    'sidebar_border':    '#253a57',
    'sidebar_text':      '#d0dbe7',
    'sidebar_text_top':  '#60a5fa',
    'sidebar_caption':   '#8da2bb',
    'bg_primary':        '#f7f9fc',
    'bg_secondary':      '#eef3f8',   # cards
    'bg_hover':          '#e1e8f0',
    'border':            '#d6dee8',
    'text_primary':      '#172033',
    'text_secondary':    '#3f4f63',
    'text_muted':        '#6b7788',
    'accent':            '#2563eb',   # brand primary
    'accent_hover':      '#1d4ed8',   # brand content
    'accent_visited':    '#6d28d9',
    'header_tint':       '#dde6f1',
    # figure frame (mapped from the same palette)
    'fig_paper':         '#f7f9fc',
    'fig_plot':          'white',
    'fig_grid':          '#d6dee8',
    'fig_zero':          '#172033',
    'fig_font':          '#172033',
}

# data-series color cycle (unchanged)
_SERIES_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
```

Small style-builder helpers (kept minimal per repo style) to avoid repeating dicts:

- `_card_style()` — card container (bg_secondary, border, radius, padding, shadow).
- `_label_style()` — bold labels using `text_secondary`.
- `_sidebar_style(open)` — sidebar container + open/closed transform.
- `_apply_fig_theme(fig)` — apply paper/plot bg, gridlines, zeroline, font to any figure.

Then replace the scattered `#f5f6fa` / `#2c3e50` / `#e1e1e1` literals in
`plot_TS_res_from_ts`, `plot_window_res_dash`, and the MP figure layouts with
`_apply_fig_theme` + `_THEME` values.

**Acceptance:** no visual regression in figures; all hardcoded chrome hexes replaced by `_THEME`.

---

## 3. Left sidebar layout (Phase 1)

Replace the single centered stacked `html.Div` layout with a two-region shell:

```
html.Div(app-shell, flex row)
├── html.Div(sidebar)        # fixed left, scrollable, dark navy
│    ├── header (title)
│    ├── global controls (x-limits, add-plot button, mode toggles)
│    └── per-plot control cards (stacked)
└── html.Div(content)        # graphs, flex-grow
     └── plot panels
```

Details:

- **Sidebar container**: `position: fixed; top:0; left:0; height:100vh; width:320px; overflowY:auto;`
  background `sidebar_bg`, text `sidebar_text`, right border `sidebar_border`.
- **Pop-up / collapse**: a `☰ Options` toggle button (top-left of content) drives a
  `dcc.Store(id='sidebar-open')`; a callback returns the sidebar style with
  `transform: translateX(0)` (open) or `translateX(-100%)` (hidden), plus a
  content `marginLeft` of `320px`/`0`. `transition: transform .25s ease` for the slide.
- **Controls restyled for dark sidebar**: labels use `sidebar_caption`; dropdowns/inputs
  keep light backgrounds for readability; section cards use a slightly lighter navy panel.
- **Content area**: background `bg_primary`, graphs in `_card_style()` cards.

**Acceptance:** all controls that were previously stacked in the body now live in the
left sidebar; toggle button shows/hides it with a slide; graphs occupy the main area.

---

## 4. Unlimited plots — dual-plot app (Phase 2)

Rework `_build_dual_plot_dash_app` from fixed `plot 1 / optional plot 2` to a dynamic list.

State:

- `dcc.Store(id='plot-panels')` holding a list of integer panel ids (start `[0]`).
- `➕ Add plot` button in the sidebar appends a new id; each panel card has a `✕ Remove`.

Dynamic components use **pattern-matching IDs** (`dash.dependencies.ALL` / `MATCH`):

```python
from dash.dependencies import Input, Output, State, ALL, MATCH
from dash import ctx

# per-panel control ids
{'type': 'plot-choice',    'index': i}
{'type': 'plot-series',    'index': i}
{'type': 'plot-ymin',      'index': i}
{'type': 'plot-ymax',      'index': i}
{'type': 'plot-remove',    'index': i}
{'type': 'plot-graph',     'index': i}
```

Callbacks:

1. **Manage panel list** — `Add plot` / `{'plot-remove', ALL}` → update `plot-panels`
   store (append / drop id). `ctx.triggered_id` identifies the source.
2. **Render control cards + graph containers** — `plot-panels` store → build one control
   card (in sidebar) and one graph container (in content) per id. New panels default to
   the first plot choice; guard against empty choice list (fail fast with clear message if
   `plot_choices` is empty).
3. **Populate series options + auto limits per panel** — `{'plot-choice', MATCH}` →
   `{'plot-series', MATCH}` options/value and `{'plot-ymin/ymax', MATCH}` using the
   existing `_get_df_and_label` + limit-calc logic (extracted into a small helper reused
   by all apps).
4. **Draw figures** — `Input({'plot-graph'... via choice/series/limits ALL}` + global
   x-limits → one figure per panel via `plot_fn`.

Remove: `show-plot-2` radio, `plot-2-controls`, `plot-2-container`, `toggle_plot_2`, and the
fixed `y-min-2/y-max-2` block.

**Acceptance:** can add ≥3 plots, each independently configurable; removing a middle panel
keeps the rest working; TS and window apps both work (they share this builder).

---

## 5. Unlimited plots — MP app (Phase 3)

Apply the same dynamic-panel model to `create_mp_ts_dash`, preserving its two modes:

- **Single period**: N independent panels (same pattern as Phase 2), each with its own
  plot type + series + y-limits; shared period selector + x-limits stay global in sidebar.
- **Compare (3 columns)**: keep the `make_subplots(rows=1, cols=3)` per-period comparison,
  but allow **N such comparison rows** (one per added plot), each a `dcc.Graph`. The
  column period selectors (`mp-p1/p2/p3`) remain global.

Move `mp-mode`, period selectors, plot-type, series, and x/y-limit controls into the
sidebar; graphs into the content area. Reuse `_apply_fig_theme` for both single and
subplot figures.

**Acceptance:** single mode supports >2 stacked plots; compare mode still renders the
3-column period comparison and supports multiple comparison rows; legends de-duplicated as
today.

---

## 6. Manual test checklist (Phase 4)

Run each app and verify (extends `pyflow_tests/test_graph_dash.py`, which currently checks
layout/callbacks construct without error):

- [ ] TS dashboard (`run_ts_dash`): sidebar toggles; add/remove ≥3 plots; series + limits per plot.
- [ ] Window dashboard (`run_window_dash`): same, with `Frame` x-axis label.
- [ ] MP dashboard (`run_mp_ts_dash`): single mode N plots; compare mode 3-column rows.
- [ ] Colors match docs palette (dark navy sidebar, light content, blue accents).
- [ ] Figures readable (gridlines, fonts, backgrounds themed).
- [ ] Existing tests in `test_graph_dash.py` still pass; add cases for dynamic-panel callbacks.

---

## 7. File-by-file impact

| File | Change |
|------|--------|
| `pyflow_acdc/Graph_Dash.py` | Add `_THEME`, `_SERIES_COLORS`, style helpers, `_apply_fig_theme`; rewrite `_build_dual_plot_dash_app` and `create_mp_ts_dash` layouts + callbacks; restyle figure builders. Public functions unchanged. |
| `pyflow_tests/test_graph_dash.py` | Add tests for dynamic add/remove panels and sidebar toggle callback. |
| `docs/api/dash.rst` | Optional: refresh screenshot / note new sidebar UI. |

## 8. Risks / notes

- Pattern-matching callbacks (`ALL`/`MATCH`) are the main new concept; the `getattr`-on-model
  rule does **not** apply here (no Pyomo model involved), but keep dict-based component id
  maps explicit.
- Fail fast: if a plot choice list is empty or a df is missing, render a figure with a clear
  message (as today) rather than silently blank chrome.
- Keep the refactor internal — no changes to `run_dash` auto-detection precedence.

---

## 9. Detailed implementation

This section describes, step by step, how each phase is built in
[`pyflow_acdc/Graph_Dash.py`](../pyflow_acdc/Graph_Dash.py). It references the current
line ranges so the diff scope is clear.

### 9.0 Theme + helpers (Phase 0)

**Where:** new module-level block after the imports (current lines 11–15) and before
`__all__` (line 17). No change to `__all__` (helpers are private, `_`-prefixed).

**Add constants** `_THEME` and `_SERIES_COLORS` exactly as in §2.

**Add five private helpers** (kept tiny, plain dicts — no class layer):

```python
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


def _sidebar_style(is_open):
    return {
        'position': 'fixed', 'top': 0, 'left': 0, 'height': '100vh', 'width': '320px',
        'overflowY': 'auto', 'padding': '18px', 'boxSizing': 'border-box', 'zIndex': 1000,
        'backgroundColor': _THEME['sidebar_bg'],
        'color': _THEME['sidebar_text'],
        'borderRight': f"1px solid {_THEME['sidebar_border']}",
        'transition': 'transform .25s ease',
        'transform': 'translateX(0)' if is_open else 'translateX(-100%)',
    }


def _content_style(is_open):
    return {
        'marginLeft': '320px' if is_open else '0',
        'transition': 'margin-left .25s ease',
        'padding': '24px',
        'backgroundColor': _THEME['bg_primary'],
        'minHeight': '100vh',
    }


def _apply_fig_theme(fig, *, show_title=True):
    fig.update_layout(
        paper_bgcolor=_THEME['fig_paper'],
        plot_bgcolor=_THEME['fig_plot'],
        font=dict(family='Arial, sans-serif', color=_THEME['fig_font']),
        hovermode='x unified',
        legend=dict(bgcolor='rgba(255,255,255,0.9)',
                    bordercolor=_THEME['border'], borderwidth=1),
        margin=dict(l=60, r=30, t=80 if show_title else 40, b=60),
    )
    axis = dict(showgrid=True, gridwidth=1, gridcolor=_THEME['fig_grid'],
                zeroline=True, zerolinewidth=1, zerolinecolor=_THEME['fig_zero'])
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig
```

**Refactor figure builders to use them:**

- `plot_TS_res_from_ts` (lines 176–283): swap the local `colors` list for `_SERIES_COLORS`;
  replace the `fig.update_layout(...)` + `update_xaxes/update_yaxes` chrome block
  (lines 241–274) with a single `_apply_fig_theme(fig, show_title=show_title)` call, keeping
  the title/axis-title/range logic. Title font color → `_THEME['text_primary']`.
- `plot_window_res_dash` (lines 300–332): after building traces, call `_apply_fig_theme(fig)`
  and set trace colors from `_SERIES_COLORS` by index.
- MP figures in `create_mp_ts_dash` (the two `fig.update_layout(...)` blocks at lines
  947–954 and 1002–1009): replace the inline `plot_bgcolor='white'` etc. with
  `_apply_fig_theme(fig)`.

**Result:** every hardcoded chrome hex (`#f5f6fa`, `#2c3e50`, `#e1e1e1`) is gone; series
colors unchanged.

### 9.1 Sidebar shell (Phase 1)

**Where:** the `app.layout = html.Div(...)` in `_build_dual_plot_dash_app` (lines 349–450)
and in `create_mp_ts_dash` (lines 707–797).

**New layout skeleton** (shared shape for all three apps):

```python
app.layout = html.Div([
    dcc.Store(id='sidebar-open', data=True),
    dcc.Store(id='plot-panels', data=[0]),          # Phase 2/3
    html.Div(id='sidebar', style=_sidebar_style(True), children=[
        html.H2(title, style={'color': _THEME['sidebar_text_top'], 'fontSize': '18px'}),
        html.Div(id='global-controls', children=[ ... x-limits, mode toggles ... ]),
        html.Button('➕ Add plot', id='add-plot', n_clicks=0, style=_btn_style()),
        html.Div(id='panel-controls'),               # per-panel cards injected here
    ]),
    html.Div(id='content', style=_content_style(True), children=[
        html.Button('☰ Options', id='toggle-sidebar', n_clicks=0, style=_toggle_btn_style()),
        html.Div(id='panel-graphs'),                 # per-panel graphs injected here
    ]),
])
```

`_btn_style()` / `_toggle_btn_style()` are two more tiny helpers using `_THEME['accent']`
background, white text, rounded corners, `accent_hover` on `:hover` is not possible inline,
so hover is skipped (acceptable for core-Dash-only).

**Toggle callback:**

```python
@app.callback(
    [Output('sidebar', 'style'),
     Output('content', 'style'),
     Output('sidebar-open', 'data')],
    [Input('toggle-sidebar', 'n_clicks')],
    [State('sidebar-open', 'data')],
)
def _toggle_sidebar(n, is_open):
    new_open = not is_open if n else is_open
    return _sidebar_style(new_open), _content_style(new_open), new_open
```

The `x-min`/`x-max` inputs (dual-plot lines 430–436; MP lines 783–792) move verbatim into
`global-controls`. Mode radio in the MP app (`mp-mode`, lines 715–724) also moves there.

### 9.2 Dynamic panels — dual-plot app (Phase 2)

**Where:** rewrite the body of `_build_dual_plot_dash_app` (lines 335–539). The signature
(`grid, *, title, plot_choices, default_choice_1, default_choice_2, plot_fn, x_axis_label`)
is kept for API compatibility; `default_choice_2` simply becomes the default for the 2nd
added panel (or ignored if unused).

**Imports:** extend line 13 to
`from dash.dependencies import Input, Output, State, ALL, MATCH` and add `from dash import ctx`.

**Panel id component map** (explicit dicts, per §4). A helper builds one control card and one
graph card for a given index `i`:

```python
def _panel_control_card(i, default_choice):
    return html.Div(style=_card_style(), children=[
        html.Div([html.Span(f'Plot {i+1}', style={'fontWeight': '700'}),
                  html.Button('✕', id={'type': 'plot-remove', 'index': i}, n_clicks=0,
                              style=_remove_btn_style())],
                 style={'display': 'flex', 'justifyContent': 'space-between'}),
        html.Label('Plot type', style=_label_style(on_dark=True)),
        dcc.Dropdown(id={'type': 'plot-choice', 'index': i}, options=dd_options,
                     value=default_choice, clearable=False),
        html.Label('Components', style=_label_style(on_dark=True)),
        dcc.Checklist(id={'type': 'plot-series', 'index': i}, options=[], value=[]),
        html.Label('Y-axis limits', style=_label_style(on_dark=True)),
        dcc.Input(id={'type': 'plot-ymin', 'index': i}, type='number', placeholder='Min'),
        dcc.Input(id={'type': 'plot-ymax', 'index': i}, type='number', placeholder='Max'),
    ])

def _panel_graph_card(i):
    return html.Div(style=_card_style(backgroundColor='white'),
                    children=[dcc.Graph(id={'type': 'plot-graph', 'index': i})])
```

**Callback 1 — manage the panel list** (append / remove ids in the `plot-panels` store):

```python
@app.callback(
    Output('plot-panels', 'data'),
    [Input('add-plot', 'n_clicks'),
     Input({'type': 'plot-remove', 'index': ALL}, 'n_clicks')],
    [State('plot-panels', 'data')],
)
def _manage_panels(add_clicks, remove_clicks, panels):
    trig = ctx.triggered_id
    if trig == 'add-plot':
        next_id = (max(panels) + 1) if panels else 0
        return panels + [next_id]
    if isinstance(trig, dict) and trig.get('type') == 'plot-remove':
        # only drop if this remove button was actually clicked (n_clicks > 0)
        rid = trig['index']
        remaining = [p for p in panels if p != rid]
        return remaining or panels   # never allow zero panels
    return panels
```

Guard: keep at least one panel (fail-soft on the last remove).

**Callback 2 — render control cards + graph cards** from the store:

```python
@app.callback(
    [Output('panel-controls', 'children'),
     Output('panel-graphs', 'children')],
    [Input('plot-panels', 'data')],
)
def _render_panels(panels):
    if not dd_options:
        raise PreventUpdate  # or render a clear "no plot choices" message
    defaults = [default_choice_1, default_choice_2]
    ctrls, graphs = [], []
    for pos, i in enumerate(panels):
        dc = defaults[pos] if pos < len(defaults) else plot_choices[0]
        ctrls.append(_panel_control_card(i, dc))
        graphs.append(_panel_graph_card(i))
    return ctrls, graphs
```

**Callback 3 — per-panel series options + auto y-limits** (`MATCH`), reusing the existing
column + limit logic (extract the current `get_columns` at 471–475 and `get_limits` at
492–505 into module-level `_columns_for_choice(grid, choice)` and
`_auto_ylimits(grid, choice)` so all apps share them):

```python
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
```

**Callback 4 — draw every figure** (`ALL`), one output figure per graph:

```python
@app.callback(
    Output({'type': 'plot-graph', 'index': ALL}, 'figure'),
    [Input({'type': 'plot-choice', 'index': ALL}, 'value'),
     Input({'type': 'plot-series', 'index': ALL}, 'value'),
     Input({'type': 'plot-ymin', 'index': ALL}, 'value'),
     Input({'type': 'plot-ymax', 'index': ALL}, 'value'),
     Input('x-min', 'value'),
     Input('x-max', 'value')],
)
def _draw(choices, series, ymins, ymaxs, xmin, xmax):
    x_limits = (xmin, xmax) if xmin is not None and xmax is not None else None
    figs = []
    for choice, sel, ymin, ymax in zip(choices, series, ymins, ymaxs):
        y_limits = (ymin, ymax) if ymin is not None and ymax is not None else None
        figs.append(plot_fn(grid, choice, sel or [], x_limits=x_limits, y_limits=y_limits))
    return figs
```

**Delete:** `show-plot-2` radio + card (388–398), `plot-2-controls` block (400–428),
`plot-2-container` (446–448), `toggle_plot_2` (452–460), and the old fixed
`update_subplot_options` / `update_limits` / `update_graphs` callbacks (462–537) — their
logic is now in callbacks 2–4.

Both `create_dash_app` (542–552) and `create_window_dash_app` (555–574) are unchanged since
they just call `_build_dual_plot_dash_app`.

### 9.3 Dynamic panels — MP app (Phase 3)

**Where:** `create_mp_ts_dash` (lines 652–1018). Reuse the same `plot-panels` store +
`add-plot` + pattern-matching ids, but the draw callback branches on `mp-mode`:

- **Global controls in sidebar:** `mp-mode` radio, single-period dropdown (`mp-period-single`),
  the three compare dropdowns (`mp-p1/p2/p3`), and x/y-limits. Keep the existing
  `_toggle_mode` callback (804–807) but retarget its outputs to sidebar sub-divs.
- **Per-panel controls** (`plot-choice`, `plot-series` via `MATCH`) identical to Phase 2,
  built with `_MP_PLOT_CHOICES`. Reuse `_columns_for` (681–688) via the shared
  `_columns_for_choice` helper.
- **Draw callback** (replaces `_update_mp_fig`, 847–1016): for each panel id, if
  `mode == 'single'` build one `plot_TS_res_from_ts` figure from the selected snapshot; if
  `mode == 'compare'` build a `make_subplots(rows=1, cols=3)` comparison **for that panel's
  plot type** across `p1/p2/p3`, preserving the legend de-duplication logic (931–936) and
  per-column x-range / shared y-range handling (955–959). Output is
  `Output({'type': 'plot-graph', 'index': ALL}, 'figure')` — a list of N figures, each either
  single or a 3-column subplot.
- Apply `_apply_fig_theme` to every produced figure.

**Delete:** `mp-show-plot-2` + `mp-plot-2-controls` + `mp-graph-2-container` and
`_toggle_plot_2` (809–817), plus the second-plot half of `_update_mp_fig` (963–1016), since
"second plot" is now just "add another panel".

### 9.4 Shared helpers extracted (used by all phases)

To avoid duplicating logic across the two builders, promote these to module level:

- `_columns_for_choice(grid_or_snap, choice)` — wraps `_get_df_and_label` / the MP
  `_columns_for`; returns a column-name list.
- `_auto_ylimits(grid, choice)` — the current `get_limits` body (492–505) verbatim.
- `_apply_fig_theme(fig)` — from §9.0.

These keep the per-app callbacks thin and behavior identical to today.

### 9.5 Tests (Phase 4)

In [`pyflow_tests/test_graph_dash.py`](../pyflow_tests/test_graph_dash.py):

- Assert `app.layout` contains `sidebar`, `content`, `plot-panels`, `add-plot`,
  `toggle-sidebar`.
- Unit-test `_manage_panels` logic (add appends unique id; remove drops id; never empty).
- Unit-test `_render_panels` returns matching counts of control + graph cards.
- Keep existing construction/no-error tests for all three `create_*` factories.
