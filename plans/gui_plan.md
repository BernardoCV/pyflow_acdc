# Desktop GUI plan for pyflow_acdc

**Repository:** In-repo links target the [`mario_integration`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration) branch ([`plans/`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration/plans)).

Living plan for an **optional open-source desktop shell** around pyflow_acdc: build or
load a grid, run studies, inspect `Results` and plots — in one window.

Complements (does not replace) Dash ([`plans/dash_improvement_plan.md`](dash_improvement_plan.md)).

**Status:** product shape defined (3-tab workflow); **PySide6 locked** as shell;
**no implementation started**. Remaining open questions in §10.

---

## 0. Design principles (locked)

| Principle | Meaning |
|-----------|---------|
| **pyflow stays a library** | Core purpose unchanged: Python API for hybrid AC/DC design and analysis (BSD). Scripting, notebooks, and Dash remain first-class. |
| **GUI is optional** | Extra `pyflow_acdc[GUI]` only. Core install must not require Qt. `HAS_GUI` / ImportError pattern like Dash and mapping. |
| **Thin shell only** | Widgets call existing `add_*`, `create_grid_*`, solvers, `Results`, plot helpers. No parallel grid model, no solver rewrite. |
| **Open-source shell** | **PySide6 (LGPL)** — not PyQt6 GPL. Keeps distribution compatible with pyflow’s OSS posture. |
| **Does not redefine purpose** | No commercial lock-in via GUI toolkit license; no requirement that users use the GUI to use pyflow. |

```
┌─────────────────────────────────────────────────────────────────┐
│  Open-source desktop shell (NEW, optional)                       │
│  PySide6 + WebEngine · tabs · forms · QThread                    │
├─────────────────────────────────────────────────────────────────┤
│  pyflow_acdc backend (UNCHANGED purpose / APIs)                  │
│  Grid · add_* · grid_creator · PF/OPF/window/TEP · Results       │
│  Graph_and_plot · Graph_Dash (figures) · Mapping                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 1. Recommended OSS stack

| Layer | Package | License | Role |
|-------|---------|---------|------|
| **Shell** | [PySide6](https://doc.qt.io/qtforpython/) (+ Qt WebEngine) | LGPL | Tabs, forms, file dialogs, threads, HTML embed |
| **Backend** | pyflow_acdc (this repo) | BSD | All domain logic |
| **Tables** | pandas (already core) | BSD | `Results.tables` → `QTableView` |
| **Plots** | plotly, matplotlib (already core) | MIT / PSF | Embed via `QWebEngineView` / `FigureCanvasQTAgg` |
| **Maps** | folium (`[mapping]`) | MIT | HTML → WebEngine |
| **Dash UI** | dash (`[Dash]`) | MIT | Launch in browser; reuse figure builders in-process |
| **GUI tests** | pytest-qt | MIT | Dev / CI marker `gui` |

**Explicit non-choices:**

| Reject | Why |
|--------|-----|
| **PyQt6** | GPL v3 / commercial — would pressure pyflow’s distribution story |
| **GridCal / pandapower / PowSyBl as engine** | Different data model; would change pyflow’s purpose |
| **Electron / Tauri** | Extra stack; Python backend still required |
| **New plotting stack** | pyflow already standardizes on Plotly + Folium + Dash |

Install sketch (when implemented):

```bash
pip install -e ".[GUI]"            # PySide6 + WebEngine
pip install -e ".[GUI,OPF,Dash,mapping]"   # full desktop experience
```

---

## 2. Target UX — three main tabs

The app is organized around a **single session `Grid`** held in memory for the whole
session. All tabs read/write that object.

```
┌──────────────────────────────────────────────────────────────────────────┐
│  pyflow-acdc GUI                                                          │
├──────────────┬──────────────────────┬────────────────────────────────────┤
│  1. GRID     │  2. TESTS            │  3. RESULTS                         │
│              │                      │                                     │
│  Build /     │  Run studies on      │  Results tables + all visualisation │
│  load /      │  current grid        │  (static, Folium, Dash)             │
│  paste code  │                      │                                     │
└──────────────┴──────────────────────┴────────────────────────────────────┘
         │                    │                         │
         └────────────────────┴─────────────────────────┘
                           Grid (session)
```

### Tab 1 — Grid

Three **input modes** (sub-tabs or left rail); user picks one path to obtain a `Grid`.
After any successful load/build, show a **read-only inspector** (tree + counts) and
**validate** via `analyse_grid`.

| Mode | User action | pyflow API |
|------|-------------|------------|
| **A. Interactive builder** | Forms / wizards per element type; “Add AC node”, “Add line”, … | [`grid_modifications.py`](../pyflow_acdc/grid_modifications.py) `add_*` |
| **B. Paste code** | Multi-line editor; run snippet that must leave `grid` in namespace | Same as case factories — e.g. [`PEI_grid.py`](../pyflow_acdc/example_grids/PF/PEI_grid.py) |
| **C. Load file** | File picker + format selector | [`grid_creator.py`](../pyflow_acdc/grid_creator.py) |

#### Mode A — Interactive builder (`add_*`)

Call public `add_*` functions **directly** — no parallel grid model in the GUI.

| Widget group | Functions (initial set) | Notes |
|--------------|-------------------------|-------|
| Nodes | `add_AC_node`, `add_DC_node` | Node picker combos elsewhere use `grid.nodes_AC` / `nodes_DC` names |
| Branches | `add_line_AC`, `add_line_DC`, `add_ACDC_converter`, `add_DCDC_converter` | `fromNode` / `toNode` as dropdowns |
| Generators | `add_gen`, `add_gen_DC`, `add_extgrid`, `add_generators` (CSV path) | |
| Renewables / zones | `add_RenSource`, `add_RenSource_zone`, `add_price_zone`, `add_MTDC_price_zone`, `add_offshore_price_zone` | |
| Storage / H₂ | `add_storage`, `add_electrolyser` | Requires `[OPF]` |
| Time / investment | `add_TimeSeries`, `add_inv_series`, `add_gen_mix_limits` | CSV picker for TS layout (see usage guide) |
| Cable DB | `add_cable_option`, `add_line_sizing` | Later phase |

**Form generation:** introspect `inspect.signature` on each `add_*` and map types to Qt
widgets (`float` → `QDoubleSpinBox`, `NodeType` → enum combo from `constants.py`).
Required arg `grid` is injected from session; `name=None` → optional text field.

**New grid:** “Empty grid” button → `Grid(S_base=…)` then add elements.

**Undo:** v1 **no undo stack** — user reloads file or re-runs code. (Optional later.)

#### Mode B — Paste code

- Editor (monospace) + **Run** button.
- Executed in a **restricted namespace**:

  ```python
  namespace = {
      "pyf": pyflow_acdc,
      "grid": session.grid_or_none,  # may start None
      "pd": pandas,
      # constants used in cases: NodeType, ConverterDCType, Polarity, ...
  }
  exec(user_code, namespace)
  grid = namespace.get("grid")
  if grid is None or not isinstance(grid, Grid):
      raise ValueError("Code must assign a pyflow_acdc Grid to variable 'grid'")
  ```

- **Presets:** dropdown to load template from `pyf.cases['PEI_grid']` body or doc
  examples (`pyflow_tests/doc_examples/usage/*.py`).
- **Security:** trusted local use only; document that pasted code runs with full user
  privileges (same as a script). No sandbox beyond explicit namespace.

#### Mode C — Load file

| Format | Loader | UI |
|--------|--------|-----|
| **Pickle** | `create_grid_from_pickle(path)` / `load_pickle` | Single file dialog |
| **CSV tables** | `create_grid_from_data` or `extend_grid_from_data` | Multi-file wizard: `S_base`, then optional sheets for AC nodes, AC lines, DC nodes, DC lines, converters (matches [`docs/usage.rst`](../docs/usage.rst) CSV import) |
| **MATPOWER .mat** | `create_grid_from_mat` | Single file (optional phase) |
| **Bundled case** | `pyf.cases[name](**kwargs)` | Case browser with kwargs form (shortcut, not a file) |

After load: enable Tabs 2–3; refresh inspector.

**Save:** pickle session grid; optional “Export Python” via [`Export_files.py`](../pyflow_acdc/Export_files.py).

---

### Tab 2 — Tests

Run analysis on the **current grid**. Each study runs on a **worker thread**; log output
in a dockable pane; on success refresh Tab 3.

| Study | Entry point | Requires | GUI controls (examples) |
|-------|-------------|----------|---------------------------|
| **Power flow** | `power_flow` / `ac_power_flow` / `dc_power_flow` | — | Method selector |
| **OPF (snapshot)** | `optimal_pf` | `[OPF]`, solver | Solver name, linear vs NL if applicable |
| **Window NL OPF** | `window_nl_opf` | `[OPF]`, TS on grid | Hour range / window length |
| **Rolling window** | `rolling_window_nl_opf` | `[OPF]`, TS | Commit length, horizon |
| **Time series OPF** | `ts_acdc_opf` | `[OPF]`, TS | Myopic series options |
| **Static TEP** | `static_transmission_expansion` | `[OPF]` | Scenario kwargs |
| **Multi-period TEP** | `multi_period_transmission_expansion` | `[OPF]` | Period / investment series |

Rows **disabled** (with tooltip) when optional dep or grid preconditions fail — e.g.
window OPF greyed out if `grid.Time_series` is empty.

**Pre-run checks:** `analyse_grid`; fail-fast dialog on missing solver (`solver_utils`).

---

### Tab 3 — Results

Two regions: **tables** (from `Results`) and **visualisation** (sub-tabs).

#### Tables — `Results` class

[`Results_class.py`](../pyflow_acdc/Results_class.py):

```python
res = Results(grid, save_res=False)
res.all(print_table=False)   # fills res.tables without terminal spam
```

- Left: list of table keys (`AC_Powerflow`, `AC_voltage`, `Ext_storage`, `TEP_MS_PN`, …).
- Right: `QTableView` bound to `pandas.DataFrame`.
- Toolbar: export CSV/Excel (`Results` export options), copy selection.

Show only tables that exist after the last run (empty until first solve).

#### Visualisation — sub-tabs

| Sub-tab | Source | Widget |
|---------|--------|--------|
| **Network (Plotly)** | `Graph_and_plot` | `QWebEngineView` or matplotlib canvas |
| **Map (Folium)** | `plot_folium`, `plot_folium_network`, `plot_folium_ts_results` | `QWebEngineView` + temp HTML (`[mapping]` extra) |
| **Time series (Plotly)** | `plot_TS_res_dash`, `plot_window_family_dash`, … | Reuse Dash **figure builders** only — not callbacks |
| **Dash (full)** | `run_dash` / `create_*_dash_app` | **Open in browser** button (embedded Dash server optional later) |

**Dash routing:** mirror `run_dash` auto mode (`grid.dash_mode`, `season_window_compare_run`,
`rolling_window_opf_run`, …) — see [`Graph_Dash.py`](../pyflow_acdc/Graph_Dash.py).

**Plotly offline:** bundle `plotly.min.js` in assets for air-gapped use (§10 Q-5).

---

## 3. Executive summary

| Question | Answer |
|----------|--------|
| **Feasible?** | **Yes** — optional `pyflow_acdc[GUI]` with **PySide6**; thin shell over existing APIs. |
| **Purpose impact?** | **None** — pyflow remains a Python library; GUI is an optional front-end. |
| **Core idea** | **Grid → Tests → Results**; no duplicate solver or results logic. |
| **vs Dash** | Dash stays the rich interactive TS UI; GUI integrates tables + launch + static/embed plots. |
| **Hardest part** | Mode A forms staying in sync with every `add_*` signature; Mode B is cheap. |
| **First milestone** | Tab shell + Mode C pickle + PF + `Results.all` table view. |

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  pyflow_acdc.gui                                                 │
│  SessionState(grid, last_results: Results | None, run_log)       │
│  Tab Grid │ Tab Tests │ Tab Results                              │
├─────────────────────────────────────────────────────────────────┤
│  grid_modifications · grid_creator · ACDC_PF/OPF · window_opf   │
│  Results_class · Graph_and_plot · Graph_Dash (figures) · Mapping │
└─────────────────────────────────────────────────────────────────┘
```

**Reuse rules (unchanged):**

| Do | Don't |
|----|-------|
| Call `add_*`, `create_grid_*`, solvers, `Results` as-is | Duplicate parameter validation in GUI beyond UX defaults |
| Import Dash **figure/data** helpers | Reimplement Dash callbacks in Qt |
| `constants.py` enums in combos | Magic strings in forms |
| Worker thread for every solve | Block UI on Pyomo |
| Keep core free of Qt imports | Import PySide6 inside `pyflow_acdc.gui` only |

Optional refactor: dash plan **R4** (`pyflow_acdc/dash/data.py`) so Tab 3 and Dash share one import path.

---

## 5. Feasibility notes

| Concern | Mitigation |
|---------|------------|
| **Many `add_*` functions** | Phase forms incrementally; codegen from signatures; “Advanced” raw kwargs JSON only if needed |
| **Paste code** | Same workflow as today’s case `.py` files; templates from `pyf.cases` |
| **CSV import** | Reuse documented multi-table layout; wizard picks files per table |
| **Long solves** | `QThread` + log redirect; disable Run while busy |
| **Toolkit license** | **PySide6 (LGPL) locked** — no PyQt6 |
| **GUI tests** | `pytest-qt`, marker `gui`, offscreen CI |

---

## 6. Implementation phases

Phases follow **Tab 1 → 2 → 3**, not “results first”.

### Phase 0 — Spike (3–5 days)

| ID | Deliverable |
|----|-------------|
| P0-1 | `QTabWidget` shell + `SessionState` (PySide6) |
| P0-2 | Mode C: load pickle → inspector shows node count |
| P0-3 | Tab 2: `power_flow` on worker |
| P0-4 | Tab 3: `Results.all(print_table=False)` → one `QTableView` |

### Phase 1 — Grid tab (MVP load paths)

| ID | Deliverable |
|----|-------------|
| P1-1 | Mode C: CSV wizard → `create_grid_from_data` |
| P1-2 | Mode B: code editor + exec → `grid` |
| P1-3 | Case browser (`pyf.cases`) as code template / direct load |
| P1-4 | Inspector tree (nodes, lines, gens, storage, TS count) |
| P1-5 | Save pickle / export |

### Phase 2 — Grid tab (builder)

| ID | Deliverable |
|----|-------------|
| P2-1 | Empty grid + `add_AC_node` / `add_DC_node` forms |
| P2-2 | `add_line_AC`, `add_line_DC`, converters |
| P2-3 | `add_gen`, `add_RenSource`, `add_extgrid` |
| P2-4 | `add_storage`, `add_electrolyser`, `add_TimeSeries` |
| P2-5 | Price zones + remaining `add_*` from §2 Tab 1 table |

### Phase 3 — Tests tab

| ID | Deliverable |
|----|-------------|
| P3-1 | PF + OPF snapshot with solver picker |
| P3-2 | `window_nl_opf` + `rolling_window_nl_opf` |
| P3-3 | `ts_acdc_opf` |
| P3-4 | TEP entries (static / multi-period) behind `[OPF]` |
| P3-5 | Preconditions + log pane |

### Phase 4 — Results visualisation

| ID | Deliverable |
|----|-------------|
| P4-1 | All `Results.tables` navigable + export |
| P4-2 | Plotly network / TS embed (`Graph_and_plot`, window family figures) |
| P4-3 | Folium embed (`[mapping]`) |
| P4-4 | **Open Dash** + route by `grid` state |

### Phase 5 — Polish

| ID | Deliverable |
|----|-------------|
| P5-1 | SCADA theme (`gui.qss` from Dash palette) |
| P5-2 | `docs/usage_gui.rst` + screenshot |
| P5-3 | `pyflow-acdc-gui` entry point, `[GUI] = ["PySide6", …]` in `pyproject.toml` |

---

## 7. Module layout

```
pyflow_acdc/gui/
  __init__.py              # HAS_GUI
  __main__.py              # pyflow-acdc-gui
  app.py
  session.py               # SessionState(grid, results, logs)
  main_window.py           # QTabWidget: grid | tests | results
  tabs/
    grid_tab.py            # modes A / B / C
    tests_tab.py
    results_tab.py
  grid/
    builder_forms.py       # signature → widgets; calls add_*
    code_runner.py         # exec namespace for Mode B
    file_loaders.py        # pickle, CSV, mat
    inspector.py           # tree + analyse_grid
  tests/
    study_registry.py      # study metadata + preconditions
    solve_worker.py        # QThread jobs
    log_view.py
  results/
    tables_panel.py        # Results.tables → QTableView
    plots_panel.py         # WebEngine / matplotlib
    dash_launcher.py       # run_dash in subprocess or thread
  widgets/                 # shared combos (node picker, enum combo)
  assets/gui.qss
```

---

## 8. Threading and errors

| Rule | Implementation |
|------|----------------|
| Solves off UI thread | `SolveWorker(QThread)` |
| Invalid grid / missing TS | Disable study row; `QMessageBox` with explicit message |
| Solver failure | Log + no refresh of Results; show Pyomo table if present |
| Grid edits during solve | Lock Tab 1 + Tests Run |
| Code run errors | Show traceback in Mode B output pane |

---

## 9. Testing

| Layer | Tests |
|-------|-------|
| `code_runner` | exec sample snippet → `Grid` assigned |
| `file_loaders` | pickle round-trip |
| `study_registry` | preconditions for synthetic grids |
| `solve_worker` | PF job without Qt |
| Widgets | `pytest-qt` smoke |
| CI | `gui` marker; not in `--quick` |

---

## 10. Open questions

| ID | Question | Default if unset |
|----|----------|------------------|
| ~~Q-1~~ | ~~PyQt6 vs PySide6~~ | **Resolved: PySide6 (LGPL)** |
| Q-2 | Mode A: all `add_*` in v1 or tiered (nodes/lines first)? | Tiered per Phase 2 |
| Q-3 | Dash: browser only vs embed `QWebEngineView` + local server | Browser only in P4-4 |
| Q-4 | Paste code: allow `import` from user env? | Yes — same trust as script |
| Q-5 | Project file: pickle only vs YAML pointing to CSV paths | Pickle v1 |
| Q-6 | Plotly JS: bundle vs CDN | Bundle in package |
| Q-7 | Convex SOCP studies in Tests tab | When [`convex_acdc_socp_plan.md`](convex_acdc_socp_plan.md) ships |

---

## 11. Success criteria

| Milestone | User can … |
|-----------|------------|
| Phase 0 | Load pickle, run PF, see `AC_Powerflow` table |
| Phase 1 | Build grid from CSV or pasted `PEI_grid`-style code |
| Phase 2 | Add nodes/lines/gens via forms without a script |
| Phase 3 | Run `window_nl_opf` from GUI |
| Phase 4 | View Folium map + open season-compare Dash from same session |

---

## 12. References

| Resource | Link |
|----------|------|
| Grid modifications API | [`pyflow_acdc/grid_modifications.py`](../pyflow_acdc/grid_modifications.py) |
| Grid creation / import | [`pyflow_acdc/grid_creator.py`](../pyflow_acdc/grid_creator.py) |
| Results | [`pyflow_acdc/Results_class.py`](../pyflow_acdc/Results_class.py) |
| Dash plan | [`plans/dash_improvement_plan.md`](dash_improvement_plan.md) |
| CSV usage | [`docs/usage.rst`](../docs/usage.rst) |
| Architecture | [`docs/architecture.md`](../docs/architecture.md) |
| PySide6 | https://doc.qt.io/qtforpython/ |

---

## 13. Changelog

| Date | Change |
|------|--------|
| 2026-07-27 | Initial feasibility evaluation (`pyqt6_gui_plan.md`). |
| 2026-07-27 | Reworked around 3-tab UX: Grid (builder / code / file) → Tests → Results. |
| 2026-07-27 | Renamed to `gui_plan.md`; locked **PySide6** OSS shell; pyflow purpose unchanged; added §0–§1 stack. |
