# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

> This changelog was introduced during a maintenance/hardening effort; entries
> for releases prior to its creation are not reconstructed here. The current
> packaged version is **0.5.1**.

## [Unreleased]

### Added
- `CONTRIBUTING.md`, `CHANGELOG.md`, and `docs/architecture.md`.
- Centralised string-as-enum constants in `pyflow_acdc/constants.py`:
  `ObjComponent`, `CssMode`, `MIPBackend`, `PricingStrategy`, `TSType`
  (plus the `TS_RENEWABLE_TYPES` group), and a `default_obj_weights()` factory.
- `__all__` is now defined on every module, making the public surface explicit.
- `pyflow_acdc/depreciation_methods.py`: deprecated mixed-case aliases (both
  module-level functions and `Results` methods) that forward to the snake_case
  implementations and emit `DeprecationWarning`.
- `pyproject.toml`: `keywords`, a `Homepage` URL, and `pytest-cov` in the
  `[tests]` extra.

### Changed
- Objective-weight defaults are now built from a single factory instead of
  three duplicated literal dicts.
- **Namespace narrowing:** accidentally re-exported internals (e.g.
  `pyflow_acdc.NodeType`) are no longer in the top-level namespace now that
  every module declares `__all__`. Use `pyflow_acdc.constants.<Name>` or a
  direct import. The documented `pyflow_acdc.__all__` API is unchanged.
- Docs: corrected class references (`rec_Line_AC`, `Size_selection`),
  refreshed the `Optimal_PF` signature, removed the unused `sphinx-rtd-theme`
  dependency, and updated the copyright year.
- **snake_case is now the default for the public API.** Public functions were
  renamed to snake_case (e.g. `Power_flow→power_flow`, `Create_grid_from_data→
  create_grid_from_data`, `Optimal_PF→optimal_pf`, `OPF_obj→opf_obj`,
  `Expand_element→expand_element`, `Translate_pyf_OPF→translate_pyf_opf`,
  `plot_Graph→plot_graph`, `results_TS_OPF→results_ts_opf`). The legacy
  mixed-case names remain importable as deprecated aliases (see below). The
  `grid.OPF_obj` attribute is unchanged (only the same-named function moved).
- **`Results` methods renamed to snake_case** (e.g. `res.All()→res.all()`,
  `res.AC_Powerflow()→res.ac_powerflow()`, `res.TEP_N()→res.tep_n()`). Old
  method names remain as deprecated aliases. Excel-sheet/table keys, `Grid`
  attributes, and the `Price_Zone` class are unchanged.
- **`Grid` methods renamed to snake_case** (e.g. `grid.Update_Graph_AC()→grid.update_graph_ac()`,
  `grid.Line_AC_calc()→grid.line_ac_calc()`, `grid.Check_SlacknDroop()→grid.check_slack_n_droop()`).
  Old method names remain as deprecated aliases on `Grid`.
- **Internal OPF/PF/TEP helpers renamed to snake_case** (e.g.
  `OPF_create_NLModel_ACDC→opf_create_nl_model_acdc`,
  `load_flow_AC→load_flow_ac`, `TEP_obj→tep_obj`,
  `Jacobian_conv→jacobian_conv`, `OPF_create_LModel_AC_gurobi→opf_create_l_model_ac_gurobi`).
  Submodule-only; no deprecation aliases added.
- **Multi-scenario TEP economics (`weighted_subobj`):** scenario weights `w[t]`
  carry each frame's share of the year; `Hy` (default 8760 h/y) scales per-hour
  OPF costs to annual. `create_scenarios` passes the full
  `present_value_factor(Hy, discount_rate, n_years)` (or `Hy` when `NPV=False`)
  into `weighted_subobj` — not a separate bug. Prior audit/CHANGELOG note about
  “missing `Hy`” was incorrect (looked at the helper body, not the caller).

### Fixed
- `Export_files` no longer emits an unquoted `pricing_strategy=` value in
  generated loader code (which raised `NameError` when re-run).
- Removed a duplicate/unreachable `PZ_cost_of_generation` branch in
  `calculate_objective` (kept the `S_base`-scaled formula).
- Removed the invalid `Programming Language :: C` classifier from packaging
  metadata.
- `kappa_sensitivity` no longer references undefined `model.discount_rate`;
  it now uses `present_value_factor(Hy, discount_rate, n_years)` like the other
  TEP sensitivity helpers (fixes `AttributeError` at runtime).
