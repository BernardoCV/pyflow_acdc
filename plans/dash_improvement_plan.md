# Dash visual improvement plan for pyflow_acdc

Living plan for the interactive Dash apps in
[`pyflow_acdc/Graph_Dash.py`](../pyflow_acdc/Graph_Dash.py).

Related tests: [`pyflow_tests/test_graph_dash.py`](../pyflow_tests/test_graph_dash.py).
Docs: [`docs/api/dash.rst`](../docs/api/dash.rst),
[`docs/usage_window_opf.rst`](../docs/usage_window_opf.rst).

Original goal (3 items) is **done** (§A). Phases 5–8 below have **mostly shipped**; post-plan
work (season-compare polish, rolling Dash, PEI docs) is also **done** (§B). The one remaining
functional item is **MP family panels** (§1.5).

---

## A. Shipped (done — do not re-plan)

All green: `pytest pyflow_tests/test_graph_dash.py` (29 passed).

| Area | What shipped | Key symbols |
|------|--------------|-------------|
| Theme | Docs Furo "SCADA panel" palette + series colors; every figure restyled. | `_THEME`, `_SERIES_COLORS`, `_apply_fig_theme` |
| Left sidebar | Fixed navy sidebar + content area + open/close store. | `_sidebar_style`, `_content_style`, `dcc.Store('sidebar-open')` |
| Collapse / hide | `☰ Hide/Show options` in content + `Hide ✕` in sidebar. | `_register_sidebar_toggle` (shared) |
| Unlimited plots | Add/remove panels via pattern-matching IDs. | `dcc.Store('plot-panels')`, `_register_panel_manager`, `_render_panels`, `_panel_control_card`, `_panel_graph_card` |
| Sidebar width | `420px`. | `_SIDEBAR_WIDTH_PX` |
| Plot height | Global input, validated. | `plot-height`, `_normalize_plot_height`, `_apply_plot_height`, `_graph_style` |
| **Family / aggregation — TS, window, season** | Generalized via a source adapter; TS exposes **Generators** (`real_power_opf × S_base`), plus Classic mode; window/season share the family builder. | `_family_result_df`, `resolve_family_df`, `available_dash_families`, `available_family_aggregations`, `family_element_options`, `_build_family_dash_app(source=...)`, `plot_window_family_dash(results, source, x_axis_label, s_base)` |
| **Branding** | pyflow logo at top of every sidebar + SVG favicon; fail-fast if asset missing. | `pyflow_acdc/assets/pyflow_logo.svg`, `_LOGO_FILENAME`, `_sidebar_header`, `_attach_app_css` |
| **CSS → assets** | Dropdown readability + button hover/active + sidebar scrollbar in an auto-loaded stylesheet. | `pyflow_acdc/assets/dashboard.css` |
| **Polish** | `dcc.Loading(type='dot')` per graph, accent left-border cards, `responsive=True` + `displaylogo:False`, sticky `☰` toolbar. | `_panel_graph_card`, `_content_shell` |
| **Dedup (R1)** | Shared header, content shell, sidebar-toggle and panel-manager registrars; `_build_dual_plot_dash_app` deleted (TS now uses the family builder). | `_sidebar_header`, `_content_shell`, `_register_sidebar_toggle`, `_register_panel_manager` |
| Packaging | `assets/*.svg` + `assets/*.css` added to `package-data`. | `pyproject.toml` |

---

## B. Shipped after original plan (season / rolling / docs)

| Area | What shipped | Key symbols |
|------|--------------|-------------|
| **Power family (window)** | Overview plot: `Total ren`, `Total gen_AC`, `Total gen_DC`, `Total H2`, `Total BESS` via `source` aggregation; hard error if `gen_power` columns match neither AC nor DC generators. | `_window_power_overview_df`, `_FAMILY_SPECS['Power']` |
| **Season compare — layout** | Sidebar **Compare layout** radio: **Overlay** (season hues) vs **Split (subplots)** (variable hues per subplot). | `compare-layout`, `plot_season_family_dash(..., layout='overlay'|'split')` |
| **Season compare — data** | Attach per-season `window_opf_results`, build compare tables + y-labels on `grid`. | `attach_season_window_compare`, `build_season_window_compare`, `grid.season_window_compare_*` |
| **Season compare — app** | Family panels (Power, SoC, …) via shared `_build_family_dash_app(compare=True)`; `run_dash` auto-routes when `season_window_compare_run`. | `create_season_compare_dash_app`, `run_season_compare_dash` |
| **Dead code removed** | Legacy single-metric season plot helper deleted; season UI uses family builder only. | *(removed `plot_season_compare_dash`)* |
| **Rolling window Dash** | Separate app for `rolling_window_opf_run`: commit-window start + visible-window count map to frame x-limits; family panels over stitched `window_opf_results`. Self-contained (own panel store / callbacks). | `create_rolling_dash_app`, `run_rolling_dash`, `_rolling_frame_limits`, `roll-plot-panels` |
| **`run_dash` routing** | `dash_mode='auto'` precedence: season_compare → rolling → window → MP → single TS. | `run_dash`, `grid.dash_mode` |
| **PEI docs** | Usage page + screenshot + doc example (season loop → attach → Dash); API cross-links. | `docs/usage_window_opf.rst`, `docs/images/examples/pei_season_compare_dash.png`, `pyflow_tests/doc_examples/window_opf/01_pei_season_compare_dash.py`, `test_docs_window_opf.py` |

**PEI season-compare screenshot** (split layout, Power + SoC) is in the docs page above.

---

## 0. Remaining work

Only **MP family panels** (§1.5) and the **optional** refactors R3/R4 (§4) are left.

---

## 1. Generalize family options to ALL dashboards (Phase 5 — MOSTLY SHIPPED)

**Status:** steps 1.3.1–1.3.4 shipped (source adapter + TS migration + logo + polish + dedup).
**Only 1.5 (MP family panels) remains.** The feasibility analysis (1.1) and decisions (1.2) below
are kept for reference and still hold.

**Requirement (user):** the Family / Aggregation control set should be available in *every*
Dash app "if possible", not just the window and season-compare apps.

### 1.1 Feasibility (evaluation of current code)

Families are resolved by `resolve_family_df(results_dict, grid, family, aggregation)`, which
reads `results_dict[_FAMILY_SPECS[family]['key']]` (columns = element names) and groups them
using topology from `grid` (`RenSources` / `Generators(+_DC)` / `storage_elements` /
`electrolysers`). The window app passes `grid.window_opf_results`; the season app passes each
season's raw dict via `grid.season_window_compare_raw`.

The blocker for TS/MP is that the **result key names and column semantics differ** between
`window_opf_results` and `time_series_results`:

| Family | `_FAMILY_SPECS` key (window) | TS equivalent (`time_series_results`) | Maps to TS? |
|--------|------------------------------|----------------------------------------|-------------|
| Power | *(synthetic from ren/gen/H₂/BESS)* | *(none)* | ❌ window/season only |
| Generators | `gen_power` (per-gen, MW) | `real_power_opf` (per-gen, **pu → ×S_base**) | ✅ (entity/node/pz) |
| Ren Sources | `ren_power` (per-ren) | *(none per-element)* | ❌ |
| Prices | `gen_price` (per-gen) | `prices_by_zone` (**per-zone**, already aggregated) | ❌ (no per-entity data) |
| Storage / SoC | `storage_power` / `storage_soc` | *(none — window-only)* | ❌ |
| H2 / H2 mass | `hydrogen_P_e` / `hydrogen_mass_H2` | *(none — window-only)* | ❌ |

**Conclusion / decision:** "generalize families everywhere" is done by making **one universal
builder** and exposing, per dashboard, **only the families that actually resolve for that
data source**. In practice TS/MP will expose the **Generators** family; window/season also
get **Power** and storage/H₂ families. Everything a family can't express stays in the existing
**Classic** mode (loading %, curtailment, market prices, loads, MP-specific choices). This is
why `_build_family_dash_app` already has `allow_classic` — we reuse it.

### 1.2 Decisions
| ID | Topic | Decision |
|----|-------|----------|
| F1 | One builder | Route **all** apps through `_build_family_dash_app` (TS, window, season; MP pending). Delete `_build_dual_plot_dash_app` after TS is migrated (done). Rolling uses a **separate** bespoke builder (commit-window controls). |
| F2 | Source adapter | Add a `source` concept so family resolution knows which keys/scaling to use for a given results dict. TS uses per-gen power `real_power_opf × S_base`. |
| F3 | Availability | Per app, `families = families_for_source(grid, results, source)`; if empty, the app still runs in Classic-only mode (no regression). |
| F4 | MP interaction | MP keeps its period selector + 3-column compare. Family panels resolve against the **selected period's** snapshot; compare mode aggregates the family per period column. (Larger; see 1.4 step 4.) |
| F5 | No API break | `create_dash_app` / `create_mp_ts_dash` / `run_*` signatures unchanged; only internals swap to the family builder. `__all__` extended for season/rolling helpers. |

### 1.3 Detailed code changes

1. **Extend `_FAMILY_SPECS` with per-source keys + scale.** For each family add optional
   `ts_key` and `ts_scale` (callable or `'S_base'` sentinel). Only `Generators` gets a TS
   mapping now. **Power** uses `key: None` and `_window_power_overview_df` (shipped §B).

2. **Add a source resolver** — shipped as `_family_result_df` / `resolve_family_df(..., source=...)`.

3. **`_build_family_dash_app` gains `source` + `compare` + `subtitle`.** Season compare sets
   `compare=True` (Overlay/Split + season/element checklists). Window/TS use `compare=False`.

4. **Migrate `create_dash_app` (TS)** — **done** (see §A).

### 1.5 Migrate `create_mp_ts_dash` (MP) — **REMAINING**

The source adapter is ready: `resolve_family_df(..., source='ts', s_base=<snap S_base>)` and
`plot_window_family_dash(..., results=snap['time_series_results'], source='ts', s_base=..., x_axis_label='Time')`
work with `grid=None` (topology records fall back to entity/total). What's left is the **MP UI**:

- Add a **View** radio (`Classic` / `Family`), shown only in single-period mode.
- When View=Family & single: `_render_panels` emits `_family_panel_control_card`s; add the
  `_family_aggs` (MATCH) + `_window_family_series` (MATCH) callbacks; `_draw` gains family
  `ALL` inputs and a family branch that calls `plot_window_family_dash(...)` against the
  selected period snapshot with its `S_base`.
- Compare mode stays Classic (per-period columns). Family-across-compare is a **future** idea.

**Why deferred:** MP is a bespoke 3-column compare builder; adding a second (Classic/Family)
mode dimension changes `_draw`'s signature and needs visual verification. It's isolated from the
TS/window/season/rolling apps, so shipping it separately carries no regression risk to them.

### 1.4 Tests
- Unit: `_family_result_df` / `test_family_source_ts_generators` — **done**.
- `test_power_family_overview` — gen_AC / gen_DC split — **done**.
- Layout: `create_dash_app` family + classic — **done**.
- Season: `test_create_season_compare_dash_app_layout_and_plot`, `test_plot_season_family_overlay_vs_split` — **done**.
- Rolling: `test_rolling_window_dash_controls_and_frame_limits` — **done**.
- MP: family panel in single mode draws; compare mode returns 3-column subplot — **pending** (§1.5).

---

## 2. Branding — pyflow logo + favicon (Phase 6 — SHIPPED)

Goal: logo at the top of every sidebar + a favicon, reusing the docs asset. **Done** exactly as
below: `_sidebar_header(app, title, subtitle=None)` shared by all builders; favicon `<link>` added in
`_attach_app_css`; asset copied to `pyflow_acdc/assets/pyflow_logo.svg` (fail-fast on missing).

### Decisions
| ID | Topic | Decision |
|----|-------|----------|
| B1 | Asset | Reuse [`docs/_static/logo_dark.svg`](../docs/_static/logo_dark.svg) (light-ink logo for dark backgrounds — matches the navy sidebar). |
| B2 | Serving | Copy to `pyflow_acdc/assets/pyflow_logo.svg`; Dash auto-serves `assets/` next to the app module (`dash.Dash(__name__)` root = `pyflow_acdc/`). |
| B3 | Placement | Left of the sidebar title, in the existing header flex row; `Hide ✕` stays right. Optional `subtitle` under title (season / rolling apps). |
| B4 | Fail-fast | Startup check raises `FileNotFoundError` if the asset is missing (no silently broken `<img>`). |

### Tests
- `_sidebar_header` returns a tree with one `html.Img` whose `src` ends with `pyflow_logo.svg`
  and contains `hide-sidebar`.
- Each `create_*` layout has exactly one sidebar `Img`.

---

## 3. Visual polish (Phase 7 — MOSTLY SHIPPED)

| ID | Idea | Status |
|----|------|--------|
| V1 | Button hover/active | ✅ in `dashboard.css` (`#add-plot`, `#toggle-sidebar`, `#hide-sidebar`). |
| V2 | Collapsible sections (`html.Details`) | ⬜ not done (optional). |
| V3 | Loading feedback | ✅ `dcc.Loading(type='dot', color=accent)` around every graph. |
| V4 | Empty-state cards | ⬜ not done (optional; empty figures still render themed). |
| V5 | Scrollbar + rhythm | ✅ themed `::-webkit-scrollbar` in `dashboard.css`. |
| V6 | Panel accent border | ✅ `3px solid accent` left border on graph cards. |
| V7 | Cleaner charts | ✅ `responsive=True`, `config={'displaylogo': False}`. |
| V8 | Sticky toolbar | ✅ `☰` toolbar `position: sticky; top: 0` in `_content_shell`. |
| V9 | Season overlay vs split color maps | ✅ season hues (overlay) vs variable hues (split subplots). |

---

## 4. Code-quality refactor (Phase 8)

`Graph_Dash.py` is ~3.0k lines with multiple app builders duplicating large blocks. Reduce
duplication **without changing behavior or the public API**. (§1 F1 already deletes one builder.)

### Findings
| # | Issue | Status |
|---|-------|--------|
| Q1 | `_toggle_sidebar` copy-pasted 3×. | ✅ `_register_sidebar_toggle`. |
| Q2 | `_manage_panels` duplicated 3×. | ✅ `_register_panel_manager` (family builder); rolling has its own panel store. |
| Q3 | Sidebar header + content shell duplicated 3×. | ✅ `_sidebar_header` + `_content_shell`. |
| Q4 | Global controls (x-limits + plot-height) duplicated. | ⬜ not extracted (minor; rolling duplicates subset). |
| Q5 | Near-duplicate y-limit helpers (`_auto_ylimits` / `_auto_ylimits_snap` / `_family_auto_ylimits`). | ⬜ R3 (optional). |
| Q6 | CSS in a Python f-string + manual `index_string`. | ✅ moved to `assets/dashboard.css`. |
| Q7 | ~3k-line module mixes concerns. | ⬜ R4 (optional). |
| Q8 | Confirm `colorsys` / `pandas` used. | ✅ both used (`_hex_from_hls`, dataframes). |

### Remaining (optional) steps
- **R3 — unify y-limits:** one `_auto_ylimits(df, choice_or_family)` taking a dataframe; collapse
  the three variants.
- **R4 — split module** into `pyflow_acdc/dash/` (`theme.py`, `data.py`, `figures.py`, `apps.py`)
  with `Graph_Dash.py` re-exporting for back-compat. Only if the file keeps growing.

### Guardrails
- `__all__` and all `create_*` / `run_*` signatures stay identical (extensions OK).
- `pytest pyflow_tests/test_graph_dash.py` green after every step.

---

## 5. Remaining order

1. **§1.5 MP family panels** — the only functional gap.
2. **R3 / R4** — optional cleanups, only if desired.
