# Architecture

This page describes what each module in `pyflow_acdc` owns, and how the
layers depend on one another. It is a developer-oriented map, not an API
reference (see the API pages for signatures).

## Layering overview

```
constants  →  Classes  →  grid_creator / grid_modifications / grid_analysis
                  │
                  ├─→  ACDC_PF                      (power flow)
                  ├─→  pyomo_model_solve              (generic Pyomo solve)
                  ├─→  ACDC_OPF (+ NL_models / L_models)  (optimal power flow)
                  ├─→  Array_OPT / ACDC_TEP_pymoo         (planning / sizing orchestration)
                  ├─→  Time_series(_clustering)           (time-series studies)
                  └─→  Results_class / Export_files / Graph_* / Mapping  (output)
```

Lower layers never import from higher layers. `constants` is a leaf module
(no intra-package imports); `Classes` is the shared data model that almost
everything else builds on.

## Core model and construction

- **`constants.py`** — Single source of truth for domain constants and
  string enums (`NodeType`, `ConverterDCType`, `DataInput`, `Polarity`,
  `PowerLossModel`, `CableType`, `ConverterOpfFxType`, `AcDcSide`,
  `PriceZoneCategory`, `ObjComponent`, `CssMode`, `MIPBackend`,
  `PricingStrategy`, `TSType`), default tuples, tolerances/iteration caps,
  and shared helpers (`present_value_factor`, `default_obj_weights`).
  Owns: magic-value/string centralisation. Does not own: any grid logic.
- **`Classes.py`** — The data model: `Grid` plus element classes
  (`Node_AC/DC`, `Line_AC/DC`, `AC_DC_converter`, `Gen_AC/DC`, `Storage_AC/DC`,
  `Ren_Source`, `Price_Zone` and subclasses, `TimeSeries`, cable/sizing helpers). Owns
  per-object state and derived electrical quantities (e.g. `Ybus`).
- **`grid_creator.py`** — Build a `Grid` from data tables, MATPOWER `.mat`
  files, pickles, or a turbine graph. Owns import/parsing.
- **`grid_modifications.py`** — Add or mutate elements after creation
  (`add_*`, line-type conversions, time/investment series wiring).
- **`grid_analysis.py`** — Topology/analysis utilities: `analyse_grid`,
  coordinate transforms (`pol2cart`/`cart2pol`/…), `Cable_parameters`,
  `Converter_parameters`, fuel-mix distribution.

## Analysis engines

- **`ACDC_PF.py`** — Power flow: `power_flow`, `ac_power_flow`,
  `dc_power_flow`, `acdc_sequential` (aliases `Power_flow`, etc. remain for
  backward compatibility). Pure Numpy no pyomo needed.
- **`ACDC_OPF.py`** — OPF orchestration: `optimal_pf` / `optimal_l_pf`,
  objective assembly (`obj_w_rule`, `opf_obj`), result translation back onto the
  `Grid` (`translate_pyf_opf`, `opf_line_res`, …). Calls
  `pyomo_model_solve` for the actual solve.
- **`pyomo_model_solve.py`** — Generic Pyomo solve layer shared by OPF, TEP,
  array, and time-series drivers: `pyomo_model_solve`, solver log parsers,
  feasibility checks, `reset_to_initialize`, `export_solver_progress_to_excel`.
  Distinct from `solver_utils.py` (environment probe only).
- **`NL_models/`** — Nonlinear model builders and planning drivers:
  `ACDC_OPF_NL_model` (full AC/DC Pyomo model, converters, price zones, TEP
  variables, BESS / H₂), `ACDC_Static_TEP`, `ACDC_MultiPeriod_TEP`,
  `ACDC_sequential_STEP`, and `window_opf` (coupled / rolling multi-hour NL OPF).
- **`L_models/`** — Linearised model builders and drivers: `AC_OPF_L_model`
  (McCormick cable-type selection; optional BESS P-only and electrolyser
  inventory), `ACDC_L_TEP`, and `AC_L_CSS_ortools`.
- **`convex_model/`** — CVXPY SOCP / MI-SOCP model builders: `convex_model`
  (`build_socp_data`, `socp_model`). Runners live in `ACDC_convex.py`.

## Planning and sizing

- **`ACDC_TEP_pymoo.py`** — Population-based (pymoo) outer TEP with OPF
  subproblems (orchestration; uses `NL_models`).
- **`Array_OPT.py`** — Offshore inter-array sizing orchestration: **route**
  MIP (``MIP_path_graph``, Pyomo or OR-Tools CP-SAT; optional joint cable types
  via ``enable_cable_types``) and **CSS** dispatch (``wind_farm_CSS``,
  ``sequential_CSS``). Owns spanning-tree / flow / ``ct_limit`` constraints.
  Install ``[OPF]`` for Pyomo CSS; ``[LINEAR_ARRAY]`` for OR-Tools MIP/CSS + HiGHS.
  Uses both `NL_models` and `L_models`.
- **`_tep_utils.py`** — Shared TEP economics (annuity/present-value factor).

## Time series

- **`Time_series.py`** — Time-series power flow / OPF drivers and result
  aggregation.
- **`Time_series_clustering.py`** — Representative-period clustering of
  time-series inputs.
- **`Market_Coeff.py`** — Price-zone quadratic cost curves from EPEX order books
  and ENTSO-E day-ahead CSVs (`price_zone_coef_data`, `clean_entsoe_data`, …).

## Output and visualisation

- **`Results_class.py`** — `Results` container and reporting tables.
- **`Export_files.py`** — Export a grid to runnable Python, MATLAB, or
  pickle; code generation for loaders.
- **`Graph_and_plot.py`** — Static / Plotly network and result plots.
- **`Graph_Dash.py`** — Interactive Dash applications.
- **`Mapping.py`** — Geographic (folium) maps.

## Support

- **`windfarm_loader.py`** — Load a bundled wind-farm case grid plus its
  GeoJSON context.
- **`solver_utils.py`** — Detect available Pyomo solvers and OR-Tools backends
  (does not run models).
- **`__init__.py`** — Public API surface (`__all__`), optional-dependency
  guards, and the `cases` example-grid loader.

## Conventions

- Optional dependencies are imported behind `try/except ImportError` with
  feature flags (`HAS_OPF`, `HAS_DASH`, …). New optional features follow the
  same pattern.
- Public string values that form a closed vocabulary live in `constants.py`
  as `(str, Enum)` members. Stored attributes / dict keys use `.value`
  (plain strings) to preserve serialization; comparisons use the enum member.
- See `CONTRIBUTING.md` for coding and testing expectations.
