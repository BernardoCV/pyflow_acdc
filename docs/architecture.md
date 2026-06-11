# Architecture

This page describes what each module in `pyflow_acdc` owns, and how the
layers depend on one another. It is a developer-oriented map, not an API
reference (see the API pages for signatures).

## Layering overview

```
constants  →  Classes  →  grid_creator / grid_modifications / grid_analysis
                  │
                  ├─→  ACDC_PF                      (power flow)
                  ├─→  ACDC_OPF (+ NL / L models)   (optimal power flow)
                  ├─→  ACDC_*_TEP / Array_OPT        (planning / sizing)
                  ├─→  Time_series(_clustering)      (time-series studies)
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
  (`Node_AC/DC`, `Line_AC/DC`, `AC_DC_converter`, `Gen_AC/DC`, `Ren_Source`,
  `Price_Zone` and subclasses, `TimeSeries`, cable/sizing helpers). Owns
  per-object state and derived electrical quantities (e.g. `Ybus`).
- **`grid_creator.py`** — Build a `Grid` from data tables, MATPOWER `.mat`
  files, pickles, or a turbine graph. Owns import/parsing.
- **`grid_modifications.py`** — Add or mutate elements after creation
  (`add_*`, line-type conversions, time/investment series wiring).
- **`grid_analysis.py`** — Topology/analysis utilities: `analyse_grid`,
  coordinate transforms (`pol2cart`/`cart2pol`/…), `Cable_parameters`,
  `Converter_parameters`, fuel-mix distribution.

## Analysis engines

- **`ACDC_PF.py`** — Power flow: `Power_flow`, `AC_PowerFlow`,
  `DC_PowerFlow`, `ACDC_sequential`. Pure numpy/Newton-style solve, no Pyomo.
- **`ACDC_OPF.py`** — OPF orchestration: `Optimal_PF` / `Optimal_L_PF`,
  objective assembly (`obj_w_rule`, `OPF_obj`), solve (`pyomo_model_solve`),
  and result translation back onto the `Grid`.
- **`ACDC_OPF_NL_model.py`** — Builds the non-linear Pyomo model (full AC/DC
  physics, converters, price zones, optional TEP variables).
- **`AC_OPF_L_model.py`** — Builds the linear (DC-style) Pyomo model with
  McCormick linearisations for cable-type selection.

## Planning and sizing

- **`ACDC_Static_TEP.py`** — Static transmission-expansion planning and
  multi-scenario TEP; sensitivity analyses.
- **`ACDC_MultiPeriod_TEP.py`** — Multi-period (investment-horizon) TEP.
- **`ACDC_sequential_STEP.py`** — Sequential static TEP runs over time.
- **`ACDC_TEP_pymoo.py`** — Population-based (pymoo) outer TEP with OPF
  subproblems.
- **`Array_OPT.py`** — Offshore-array inter-array cable sizing: MIP path
  graphs and the cable-string-sizing (CSS) loop, with `pyomo`/`ortools`
  backends.
- **`AC_L_CSS_gurobi.py` / `AC_L_CSS_ortools.py`** — Solver-specific linear
  cable-sizing implementations.
- **`_tep_utils.py`** — Shared TEP economics (annuity/present-value factor).

## Time series

- **`Time_series.py`** — Time-series power flow / OPF drivers and result
  aggregation.
- **`Time_series_clustering.py`** — Representative-period clustering of
  time-series inputs.

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
- **`solver_utils.py`** — Detect available Pyomo solvers.
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
