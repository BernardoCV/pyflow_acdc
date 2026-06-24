# Test coverage snapshot

Last updated from Linux CI-style run (`Python 3.10.20`, pytest 9.0.2, cov 7.1.0).

```
collected 157 items (includes test_solver_utils; +9 vs prior snapshot)
146+ passed, 1 skipped — refresh after next full Linux cov run
overall coverage: ~66% baseline (25775 stmts, 8791 miss) — refresh header after full run
```

Regenerate after test changes:

```bash
pytest pyflow_tests/ --cov=pyflow_acdc --cov-report=term-missing
```

Update this file when opening a PR that adds, removes, or materially changes tests
(see `CONTRIBUTING.md`). For each affected module, refresh the **missing lines** block
and the **function / block** table (status column).

---

## Summary by module (`pyflow_acdc/`)

Sorted by miss count (core package only; `example_grids/` omitted unless noted).

| Module | Cover | Miss | Dedicated test module |
|--------|-------|------|------------------------|
| `ACDC_Static_TEP.py` | 48% | 746 | Partial (static TEP solves; MS export cold) |
| `ACDC_OPF.py` | ~49% | — | Partial (`test_opf_result_helpers`; OPF-only code) |
| `pyomo_model_solve.py` | — | — | Partial (`test_solver_utils` mocks probe only; solve layer untested) |
| `Time_series_clustering.py` | 40% | 595 | Partial (`test_docs_clustering`) |
| `Graph_and_plot.py` | 57% | 590 | Partial (`test_plot`, doc plotting) |
| `Array_OPT.py` | 54% | 538 | Partial (array / ortools tests) |
| `grid_modifications.py` | 54% | 496 | Indirect (grid / case tests) |
| `grid_creator.py` | 57% | 515 | `test_grid_creation`, csv import docs |
| `Mapping.py` | 44% | 426 | Partial (`test_folium`) |
| `Time_series.py` | 53% | 421 | Partial (`test_docs_ts`) |
| `AC_OPF_L_model.py` | 38% | 423 | Indirect (OPF / LOPF cases) |
| `ACDC_MultiPeriod_TEP.py` | 69% | 436 | Partial (MP solve; MP+MS build-only) |
| `Results_class.py` | 60% | 758 | Indirect (`res.all()` in case tests) |
| `Graph_Dash.py` | 10% | 265 | Smoke (`test_docs_dash`) |
| `Export_files.py` | 38% | 167 | Indirect (export flags rare in CI) |
| `Market_Coeff.py` | 63% | 143 | Partial (`test_market_coeff`) |
| `solver_utils.py` | 81% | 23 | `test_solver_utils` (mocked) |
| `ACDC_sequential_STEP.py` | 61% | 120 | Partial (fake-solve abort only) |

---

## Functions not covered (or without dedicated tests)

Public API (documented or commonly imported) that has **no focused test module** and is
**not exercised** (or only lightly) by case/doc tests. Private helpers (`_…`) are listed
when they account for large uncovered blocks.

| Module | Function(s) | Status | Why / how to test |
|--------|-------------|--------|-------------------|
| `Market_Coeff` | `clean_entsoe_data` | **Not tested** | Needs ENTSO-E CSV tree on disk |
| `Market_Coeff` | `compute_hour_of_year` | **Not tested** | Synthetic ENTSO-E-shaped DataFrame |
| `Graph_Dash` | `run_ts_dash`, `run_mp_ts_dash`, `create_mp_ts_dash`, … | Smoke only | Dash UI; `test_docs_dash` scratches surface |
| `Time_series_clustering` | `run_elbow_analysis`, `Time_series_cluster_relationship` | **Not tested** | Elbow / relationship workflows |
| `Time_series_clustering` | `run_clustering_analysis_and_plot` | **Not tested** | Plotting + analysis wrapper |
| `Time_series_clustering` | `cluster_Ward`, `cluster_PAM_Hierarchical` | **Not tested** | Doc example with `cluster_algorithm='ward'` |
| `ACDC_Static_TEP` | `multi_scenario_TEP` (full solve) | **Not tested** | MS Ipopt solve too slow for CI |
| `ACDC_Static_TEP` | `export_acdc_tep_ms_to_pyflow_acdc` | **Not tested** | MS solve + `mutate_grid=True` |
| `ACDC_Static_TEP` | `alpha_pareto`, `*\_sensitivity`, `comprehensive_sensitivity_analysis` | **Not tested** | Pymoo / sensitivity workflows |
| `ACDC_MultiPeriod_TEP` | `multi_period_MS_TEP` (full solve) | **Not tested** | MP+MS solve too slow for CI |
| `ACDC_MultiPeriod_TEP` | `run_opf_for_investment_period`, `run_ts_opf_for_investment_period` | **Not tested** | Post-MP TS-OPF per period |
| `ACDC_OPF` | `fx_conv` | **Not tested** | OPF with fixed converter setpoints |
| `ACDC_OPF` | `opf_line_res`, `opf_step_results`, `opf_price_price_zone` | **Tested** | `test_opf_result_helpers` (Ipopt) |
| `pyomo_model_solve` | `log_infeasible_constraints_limited` | **Not tested** | Deliberately infeasible OPF |
| `pyomo_model_solve` | `_parse_bonmin_log`, `_gurobi_callback`, `_parse_highs_log` | **Not tested** | Alternate solvers (Bonmin/Gurobi/HiGHS) |
| `pyomo_model_solve` | `pyomo_model_solve` (core) | Partial | Ipopt path via case tests; parser/callback branches cold |
| `pyomo_model_solve` | `export_solver_progress_to_excel` | **Not tested** | Solve with progress + Excel export |
| `pyomo_model_solve` | `reset_to_initialize` | Partial | TEP/TS retry paths |
| `Export_files` | Excel export helpers | Partial | TEP/MP `export` flags on solve |
| `Mapping` | Geo layout / neighbour graph bulk | Partial | `test_folium` smoke only |
| `solver_utils` | `check_pyomo_solvers`, `check_ortools_backends`, `format_solver_report` | **Tested** | `test_solver_utils` (mocked) |

---

## Coverage gaps by module

Each subsection lists raw **missing lines** from `term-missing`, then maps them to
**functions / blocks** and whether CI currently hits them.

### `ACDC_Static_TEP.py` — 48%

```
61, 63, 70, 72, 74, 76, 90, 103-106, 124-125, 134-135, 138-139, 149-150, 177-184, 191, 239, 241-242, 247-248, 302, 305, 307-308, 312-313, 315, 348-349, 383, 391, 418-419, 425, 441, 451, 463-468, 476-504, 684, 687, 706, 709, 728, 731, 747-763, 772, 775, 794, 797, 833-836, 855-859, 866, 872, 875-879, 882-886, 889-893, 896, 903, 905, 907, 910-914, 916, 920-924, 927, 969, 1188-1248, 1258, 1260, 1262-1263, 1270-1341, 1345-1400, 1404-1459, 1487-1565, 1569-1572, 1603-1606, 1693-1695, 1729, 1755-1777, 1812, 1824, 1835, 1847-1848, 1861, 1896, 1936-2009, 2022-2046, 2049-2120, 2123-2142, 2145, 2148-2172, 2176-2391, 2417-2565, 2569-2579
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 61–504 | `identify_standalone_rs_conv_pairs`, `update_grid_scenario_frame`, `expand_elements_from_pd`, `repurpose_element_from_pd`, `update_attributes`, `expand_element`, `translate_pd_tep` | Partial — static TEP cases |
| 684–927 | `_TEP_install_variables`, `_TEP_install_constraints`, `MS_TEP_constraints` | **Not tested** — needs MS full solve |
| 1188–1248 | `linear_transmission_expansion` | **Not tested** |
| 1270–1565 | `alpha_pareto`, `rate_sensitivity`, `kappa_sensitivity`, `comprehensive_sensitivity_analysis` | **Not tested** |
| 1610–1777 | `multi_scenario_TEP` body | **Not tested** — MS Ipopt full solve |
| 1936–2172 | `get_price_zone_data`, `get_curtailment_data`, `get_line_data`, `get_converter_data`, `get_gen_data` | **Not tested** — MS + price zones + clustered TS |
| 2176–2391 | `export_acdc_tep_ms_to_pyflow_acdc` | **Not tested** |
| 2417–2579 | `export_TEP_multiScenario_results_to_excel`, `calculate_STEP_objective_from_model` | **Not tested** — MS + export flags |

---

### `ACDC_OPF.py` — OPF-only (refresh % after full cov run)

OPF orchestration, objectives, and result helpers. Solver infrastructure moved to
`pyomo_model_solve.py`.

| Lines | Function / block | Status |
|-------|------------------|--------|
| — | `fx_conv` | **Not tested** — fixed converter setpoints |
| — | `opf_line_res`, `opf_price_price_zone`, `opf_step_results` | **Tested** — `test_opf_result_helpers` |
| — | `optimal_pf` / `optimal_l_pf`, `opf_obj*`, `translate_pyf_opf` | Partial — case OPF tests |

---

### `pyomo_model_solve.py` — generic solve layer (refresh % after full cov run)

```
(refresh from: pytest ... --cov=pyflow_acdc.pyomo_model_solve --cov-report=term-missing)
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| — | `log_infeasible_constraints_limited` | **Not tested** — infeasible OPF |
| — | `_gurobi_callback` | **Not tested** — Gurobi MIP/L |
| — | `_parse_bonmin_log`, `_parse_highs_log`, `_parse_ipopt_log` | Partial — Ipopt via case solves |
| — | `_solver_progress` | **Not tested** — `callback=True` |
| — | `pyomo_model_solve` | Partial — Ipopt; alternate solvers / failure paths cold |
| — | `export_solver_progress_to_excel` | **Not tested** |
| — | `reset_to_initialize` | Partial — TEP/TS retries |

---

### `ACDC_MultiPeriod_TEP.py` — 69%

```
40, 83, 89, 102, 111, 113, 117, 172, 248, 274, 300, 317, 325, 329, 341-400, 415, 417, 446, 456, 475, 532, 534, 557, 561, 579, 603, 607, 645, 649, 663-665, 687, 691, 705-709, 730, 734, 748-752, 767, 776, 784, 794-796, 801-803, 808-810, 815-817, 823, 826, 836-837, 847, 851, 856, 858, 949-953, 955, 972, 1072-1074, 1084, 1110, 1220-1244, 1248-1253, 1263, 1270-1287, 1315-1317, 1342-1344, 1371-1373, 1400-1426, 1429-1453, 1486-1491, 1496-1497, 1500, 1502, 1511-1513, 1521, 1528, 1535-1537, 1543, 1546, 1553-1556, 1562, 1565-1568, 1574, 1580, 1583, 1591, 1644, 1755, 1761-1766, 1864-1962, 1977-1998, 2022-2062, 2067-2078, 2092-2104, 2163-2233, 2277-2386, 2395, 2400, 2402, 2426-2431, 2443-2445, 2448, 2453-2457
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 40–972 | `_fill_investment_decisions`, `_MP_TEP_*` constraint/var branches | Partial — `case24_MP` full solve hits much |
| 1220–1453 | `export_mp_tep_results_to_pyflow_acdc`, capex budget | Partial — MP solve + export flags |
| 1486–1665 | `_resolve_mp_ms_clustering`, `_build_period_scenario_block` | **Not tested** — MP+MS full solve |
| 1864–1962 | `multi_period_MS_TEP` solve/export tail | **Not tested** |
| 2001–2233 | `run_opf_for_investment_period`, `run_ts_opf_for_investment_period`, `run_opf_for_all_investment_periods` | **Not tested** |
| 2277–2457 | `_set_grid_to_multiperiod_state`, decommission helpers | **Not tested** — MP+MS multi-period |

---

### `Time_series_clustering.py` — 40%

```
74-78, 107, 141-142, 154-155, 166, 182-191, 222, 241-262, 297-318, 321-330, 332-335, 363-505, 517-575, 586-607, 647-648, 658-701, 746, 752-756, 806-819, 823-830, 867-891, 925, 935-938, 944, 980-1003, 1015-1016, 1050, 1060-1063, 1069, 1091-1147, 1166-1230, 1237, 1261-1267, 1278-1298, 1340, 1342, 1344, 1346, 1348, 1378, 1388-1399, 1412-1414, 1444-1520, 1523-1537, 1557-1671, 1680-1746, 1751-1781, 1787-1789, 1840, 1850, 1893-1915, 1925-1956, 2016, 2024, 2030, 2035
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 74–335 | `filter_data`, `identify_correlations` (methods 1–3) | Partial — live cluster uses `correlation_decisions=[False,…]` |
| 363–505 | `identify_correlations` deep branches | **Not tested** — `correlation_decisions=[True, 3, True]` |
| 517–607 | `plot_time_series`, `plot_correlation_matrix` | **Not tested** — `plotting=True` |
| 647–701 | `cluster_TS` critical_idx / split clustering | **Not tested** — MS TEP with `critical_idx` |
| 752–756, 1091–1230 | `cluster_Ward`, `cluster_PAM_Hierarchical` | **Not tested** |
| 1340–1414 | `run_clustering_analysis` plotting branch | **Not tested** — `plotting=True` |
| 1444–1537 | `plot_clustering_results`, `run_clustering_analysis_and_plot` | **Not tested** |
| 1557–1781 | `run_elbow_analysis`, `Time_series_cluster_relationship` | **Not tested** |
| 2016, 2024, 2030, 2035 | `cluster_analysis` error paths | **Not tested** — invalid payload tests |

---

### `Market_Coeff.py` — 63%

```
50, 62, 101-112, 116, 161, 167, 191, 212, 214, 319, 373-376, 389, 415, 515, 540-654, 661-737
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 101–112 | DST spring/fall (`3B` hour) | **Not tested** — synthetic CSV with DST rows |
| 319, 373–376, 389, 415 | `eq_price` branches in coef fit | **Not tested** — order book with `eq_price > 15` |
| 540–654 | `compute_hour_of_year` | **Not tested** |
| 661–737 | `clean_entsoe_data` | **Not tested** — ENTSO-E files on disk |
| (other) | `price_zone_coef_data`, `price_zone_data_pd`, `plot_curves` | **Tested** — `test_market_coeff` |

---

### `solver_utils.py` — 81%

```
116-118, 128-129, 140, 145-147, 212, 215-218, 227-231, 234, 240-242, 246
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 56–149, 175–236 | `check_pyomo_solvers`, `check_ortools_backends`, `check_available_solvers` | **Tested** — mocked in `test_solver_utils` |
| 116–118, 128–129, 140, 145–147 | verbose OR-Tools probe branches | **Not tested** |
| 212, 215–218, 227–234, 240–246 | `format_solver_report` edge cases | Partial |

---

### `ACDC_sequential_STEP.py` — 61%

```
23-27, 38, 40, 50-52, 76, 78, 80, 82, 88, 108, 112, 116, 129, 134, 148, 155-156, 173-186, 190-206, 218-239, 285, 297, 337-340, 353, 356-427, 491-508, 613-641
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 356–427 | `sequential_STEP` main loop (success path) | **Not tested** — full solve (not fake-solve abort) |
| 491–508, 613–641 | Export / SVG / step results | **Not tested** — `export_steps=True`, `save_svgs=True` |

---

### Other modules (missing lines only)

**`Graph_Dash.py` 10%:** `47-87, 91, 105-201, 206, 360-362, 373-385, 396-416, 435-447, 453-454, 459-460, 476-503, 516-875, 879-880` — almost entire module; Dash callbacks **not tested**.

**`Export_files.py` 38%:** `31-32, 51, 79-90, 102, 111, 134-138, 155-167, 250, 300-316, 330-332, 359-362, 422-444, 486-495, 517, 526-740, 752-845` — Excel export paths; partial via TEP/MP `export` flags.

**`Mapping.py` 44%:** `48-113, 125, 242-246, 252, 258, 287, 291, 294-303, 306-312, 342, 370, 375, 466, 486, 495, 502, 522, 531, 566-570, 576, 586-591, 622, 627-629, 632-633, 643, 670-671, 691-693, 716-724, 732, 759, 806, 812, 833, 855-861, 877-889, 893-898, 919-1126, 1148-1541, 1552, 1559-1561, 1568, 1570, 1572, 1574, 1582, 1611, 1613` — geo/plot helpers; `test_folium` partial.

**`AC_OPF_L_model.py` 38%:** model-builder branches for grid features not in current OPF cases (TEP install vars, REC, array losses, price zones). Grows when OPF/TEP tests use richer grids — not isolated unit-test friendly.

**`Time_series.py` 53%, `Results_class.py` 60%, `Array_OPT.py` 54%, `Graph_and_plot.py` 57%** — large indirect gaps; see `term-missing` output after full cov run.

---

## CI skips (expected)

| Test | Reason |
|------|--------|
| `test_sequential_array.py` | No Pyomo MIP/CSS-L solver (Gurobi/GLPK) |

---

## Internal planning doc

For backlog / prioritized next tests, see `my_tests/pyflow_acdc_coverage_improvement_map.md`
(workspace only; not shipped with the package).
