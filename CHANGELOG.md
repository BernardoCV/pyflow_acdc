# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to follow [Semantic Versioning](https://semver.org/).

> This changelog was introduced during a maintenance/hardening effort; entries
> for releases prior to its creation are not reconstructed here. The current
> packaged version is **0.6.0**.

## [Unreleased]

### Added
- **CI**: Codecov upload on push/PR to ``main`` (``coverage`` job in
  ``pr-tests.yml``; set ``CODECOV_TOKEN`` in repository secrets). Coverage
  reports and the README badge are maintained on Codecov.
- **`pyomo_model_solve.py`**: extracted generic Pyomo solve layer from
  `ACDC_OPF.py` (`pyomo_model_solve`, log parsers, feasibility checks,
  `reset_to_initialize`, `export_solver_progress_to_excel`). `ACDC_OPF` re-exports
  for backward compatibility.
- **Tests**: `test_solver_utils.py` (mocked Pyomo/OR-Tools solver probes),
  `test_opf_result_helpers.py` (OPF result helpers after Ipopt solve),
  `test_market_coeff.py`, clustering doc examples (`test_docs_clustering`),
  `test_graph_dash.py` (synthetic TS + Dash callback unit tests).
- **`ipopt_available` / `require_ipopt`** helpers in `pyflow_tests/_test_solver_deps.py`.

### Removed
- **`TEST_COVERAGE.md`**: removed in favor of Codecov-only coverage tracking.

### Changed
- **`dill`** is a required base dependency; removed optional-import fallbacks in
  `grid_creator` / `Export_files` and `require_dill` test skips.
- **`Market_Coeff`**: module and public-function docstrings; expanded
  `docs/api/market_coef.rst` (EPEX CSV schema, workflow, ENTSO-E layout).
  `clean_entsoe_data` now returns the output Excel path.
- Docs and user-facing strings: **pyflow-acdc** / **pyflow_acdc** naming
  (replacing mixed ``PyFlow-ACDC`` / ``PyFlow ACDC`` variants).

## [0.6.0] - 2026-06-16

### Added
- **Sequential STEP** (`ACDC_sequential_STEP.py`): `sequential_STEP` and
  `sequential_MS_STEP` run static transmission expansion sequentially across
  time frames (single- and multi-scenario), tracking per-element installation
  (`np_gen`/`np_rsgen`/`np_line`/`np_conv`) over the horizon; `export_results_to_csv`
  persists the run results.
- **Solver utilities** (`solver_utils.py`): `check_available_solvers` detects which
  Pyomo solvers and OR-Tools backends are actually installed.
- **Multi-period TEP — time-series OPF per investment period**:
  `run_ts_opf_for_investment_period` (alongside the existing
  `run_opf_for_investment_period`).
- **Dash / interactive visualization** (`Graph_Dash.py`): `run_ts_dash`,
  `run_mp_ts_dash`, `create_mp_ts_dash`, `plot_TS_res_from_ts`, `plot_TS_res_dash`
  for time-series and multi-period result exploration.
- **Folium map plotting**: `plot_folium` and `plot_folium_network` for geographic
  network rendering.
- **Clustering**: `load_precomputed_clusters_to_grid` plus forced-cluster support
  and correlation-identification improvements in `Time_series_clustering.py`.
- **Example grids reorganized** into `PF/`, `OPF/`, `TEP/`, and `Wind_Array/`
  subfolders; the case loader now walks these folders (qualified module names).
  New cases: `case1888rte`, `case3120sp_acdc` (OPF) and `CigreB4_ACDC` (PF). Added
  `Moray_East` wind-farm data; removed the obsolete `generate_grids.py`.
- **Tests**: fast fake-solve harness (`_quick_fake_solve.py`,
  `test_OPF_quick_runner.py`) and plotting tests (`test_plot.py`).
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
- `ts_ac_pf` and `ts_dc_pf`: AC-only and DC-only time-series power-flow
  helpers (mirroring `ts_acdc_pf`), exported at the top level.

### Changed
- **`pyomo_model_solve` reworked**: supports solver options and callbacks, a
  robust retry path, constraint-tightening/feasibility handling for hard
  multi-period runs, and richer solver-statistics reporting.
- **Multi-period TEP substantially extended** (`ACDC_MultiPeriod_TEP.py`):
  installation ranges over planned values, decommissioning periods, time-series
  net-power reporting, and per-investment-period result exports / SVG output.
- **PGL min/max physical constraints** added for renewable/load power bounds in
  the OPF/TEP formulations.
- **Array optimization** (`Array_OPT.py`): added an array-loss objective term and
  fixed cable-sizing/CSS behavior.
- **OPF/TS multiplicity (`np`) handling aligned** across the OPF and time-series
  paths, and time-series line outputs were cleaned up for consistent reporting.
- **Mapping/Folium cleanup**: planar-layout fix and SVG export scaling.
- Objective-weight defaults are now built from a single factory instead of
  three duplicated literal dicts.
- Node active/reactive generation expressions are centralized on `Node_AC`
  (`gen_P_injection`, `gen_Q_injection`, `gen_P_total`, `gen_Q_total`,
  `gen_P_node_aggregate`), replacing duplicated formulas in `Time_series`,
  `Results_class`, and `Graph_and_plot`. The slack and PV reactive
  back-calculations (which were identical) now share `gen_Q_injection`.
  **Behavior change:** the non-OPF forward generation sums now scale per-unit
  generator/renewable output by the parallel-unit counts (`np_gen` /
  `np_rsgen`); previously these counts were ignored outside the OPF path.
- Power flow now populates per-node generation aggregates as a single source of
  truth: `Node_AC.PGi_ren`/`QGi_ren` (renewable: `Σ rs.PGi_ren*rs.gamma*rs.np_rsgen`
  active, `Σ rs.QGi_ren*rs.np_rsgen` reactive — reactive is not curtailed) and
  `Node_AC.PGi_opt`/`QGi_opt` (connected-generator dispatch: `Σ gen.PGen`/`Σ gen.QGen`).
  The known-power vectors are now `PGi + PGi_ren + PGi_opt - PLi` (and the reactive
  analog). The DC path mirrors the active side, and `Node_DC` gained `PGi_ren` /
  `PGi_opt` attributes. The converter `P_AC`/`Q_AC` back-calculation in
  `ACDC_PF.acdc_sequential` now reads these node attributes instead of recomputing
  the sums inline. **Behavior change:** for renewable sources with `np_rsgen > 1`
  the power-flow injection now includes the parallel-unit count (and the renewable
  reactive contribution, previously omitted entirely from `Q_AC`).
- `Gen_AC`/`Gen_DC` now initialize `PGen`/`QGen` as the **total** output
  (`Pset * np_gen` / `Qset * np_gen`) instead of the per-unit setpoint, making
  `PGen` consistently the total dispatch both before and after an OPF solve
  (`Pset`/`Qset` remain the per-unit inputs). The power-flow known-power vectors
  therefore use `Σ gen.PGen` / `Σ gen.QGen` directly. **Behavior change:** for
  generators with `np_gen > 1`, the power-flow injection and `PGen`-derived
  reporting (e.g. loading) now reflect the full parallel-unit output, which was
  previously omitted in the pre-OPF state.

### Removed
- `Node_AC.curtailment` (a node-level scalar that was always `1` and never
  assigned): curtailment is modeled per `Ren_Source` via `gamma`/`min_gamma`,
  and `PGi_ren` already reflects it, so the redundant factor was dropped from
  the node generation expressions.
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
- `plot_model_feasebility` renamed to `plot_model_feasibility` (public typo;
  old name kept as a deprecated alias).
- DC power-flow corrections and renewable/generator parallel-unit (`np_rsgen` /
  `np_gen`) export fixes.
- Sequential/multi-period solver robustness: `bonmin` exit handling and retry on
  solver failures during STEP/MP runs.
- `Export_files` no longer emits an unquoted `pricing_strategy=` value in
  generated loader code (which raised `NameError` when re-run).
- Removed a duplicate/unreachable `PZ_cost_of_generation` branch in
  `calculate_objective` (kept the `S_base`-scaled formula).
- Removed the invalid `Programming Language :: C` classifier from packaging
  metadata.
- `kappa_sensitivity` no longer references undefined `model.discount_rate`;
  it now uses `present_value_factor(Hy, discount_rate, n_years)` like the other
  TEP sensitivity helpers (fixes `AttributeError` at runtime).
- `Gen_set_dev` objective now compares like units: the setpoint deviation uses
  `gen.Pset * gen.np_gen` (total) instead of the per-unit `gen.Pset`, so the
  penalty is zero at the setpoint for generators with `np_gen > 1` (both the
  Pyomo objective and the post-solve `calculate_objective` path).
- `transmission_expansion_pymoo` now wires its `NPV` flag into
  `TEPOuterProblem`: when `NPV=False` the OPEX (and CAPEX) are no longer scaled
  by the present-value factor. Previously the flag was ignored and present value
  was always applied.
- `time_series_pf` no longer calls the non-existent `grid.TS_ACDC_PF` attribute
  (which raised `AttributeError` for AC/DC grids); it now routes to the
  module-level `ts_dc_pf` (DC-only), `ts_ac_pf` (AC-only), or `ts_acdc_pf`
  (sequential). Also replaced `== None` comparisons with `is None`.
