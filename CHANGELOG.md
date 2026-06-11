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

### Fixed
- `Export_files` no longer emits an unquoted `pricing_strategy=` value in
  generated loader code (which raised `NameError` when re-run).
- Removed a duplicate/unreachable `PZ_cost_of_generation` branch in
  `calculate_objective` (kept the `S_base`-scaled formula).
- Removed the invalid `Programming Language :: C` classifier from packaging
  metadata.

### Known issues
- `kappa_sensitivity` references `model.discount_rate` without creating it
  (raises `AttributeError` at runtime) — pending fix.
- `weighted_subobj` omits the `Hy` (hours/year) factor used elsewhere —
  pending review.
