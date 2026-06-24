# Test coverage snapshot

Last updated from Linux CI-style run (`Python 3.10.20`, pytest 9.0.2, cov 7.1.0, pluggy 1.6.0).

```
collected 179 items
178 passed, 1 skipped
total time: ~470s
overall coverage: 67% (25769 stmts, 8528 miss)
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
| `Results_class.py` | 60% | 758 | Indirect (`res.all()` in case tests) |
| `Time_series_clustering.py` | 40% | 596 | Partial (`test_docs_clustering`) |
| `Graph_and_plot.py` | 57% | 590 | Partial (`test_plot`, doc plotting) |
| `pyomo_model_solve.py` | 19% | 522 | Partial (Ipopt via case tests; `test_solver_utils` is env probe only) |
| `Array_OPT.py` | 54% | 538 | Partial (array / ortools tests) |
| `grid_creator.py` | 57% | 511 | `test_grid_creation`, csv import docs |
| `grid_modifications.py` | 54% | 496 | Indirect (grid / case tests) |
| `Mapping.py` | 44% | 426 | Partial (`test_folium`) |
| `AC_OPF_L_model.py` | 38% | 423 | Indirect (OPF / LOPF cases) |
| `Time_series.py` | 54% | 409 | Partial (`test_docs_ts`) |
| `ACDC_MultiPeriod_TEP.py` | 69% | 436 | Partial (MP solve; MP+MS build-only) |
| `ACDC_OPF_NL_model.py` | 86% | 224 | Indirect (OPF case tests) |
| `ACDC_PF.py` | 73% | 193 | `test_cigreb4_pf`, case PF smoke |
| `ACDC_TEP_pymoo.py` | 55% | 175 | Partial (`test_docs_tep_pymoo`) |
| `Export_files.py` | 38% | 165 | Indirect (export flags rare in CI) |
| `Market_Coeff.py` | 63% | 144 | Partial (`test_market_coeff`) |
| `Graph_Dash.py` | 72% | 82 | **`test_graph_dash`** + smoke (`test_docs_dash`) |
| `ACDC_sequential_STEP.py` | 61% | 120 | Partial (fake-solve abort only) |
| `ACDC_OPF.py` | 83% | 102 | Partial (`test_opf_result_helpers`; OPF-only after solve split) |
| `Classes.py` | 86% | 328 | Indirect (all case tests) |
| `solver_utils.py` | 81% | 23 | `test_solver_utils` (mocked) |
| `AC_L_CSS_ortools.py` | 87% | 43 | `test_sequential_array_ortools` |

---

## Functions not covered (or without dedicated tests)

| Module | Function(s) | Status | Why / how to test |
|--------|-------------|--------|-------------------|
| `Market_Coeff` | `clean_entsoe_data`, `compute_hour_of_year` | **Not tested** | ENTSO-E CSV tree or synthetic fixtures |
| `Graph_Dash` | `run_ts_dash`, `run_mp_ts_dash` (server) | **Not tested** | `app.run()` blocks |
| `Graph_Dash` | `plot_TS_res_from_ts`, `create_dash_app`, `create_mp_ts_dash` | **Tested** | `test_graph_dash` |
| `Time_series_clustering` | `run_elbow_analysis`, ward/PAM, `run_clustering_analysis_and_plot` | **Not tested** | Doc examples / `plotting=True` |
| `ACDC_Static_TEP` | `multi_scenario_TEP` (full solve), MS export/sensitivity | **Not tested** | MS solve too slow for CI |
| `ACDC_MultiPeriod_TEP` | `multi_period_MS_TEP` (solve), post-MP TS-OPF | **Not tested** | MP+MS / per-period OPF too slow |
| `ACDC_OPF` | `fx_conv` | **Not tested** | OPF with fixed converter setpoints |
| `ACDC_OPF` | `opf_line_res`, `opf_step_results`, `opf_price_price_zone` | **Tested** | `test_opf_result_helpers` |
| `pyomo_model_solve` | parsers, callbacks, infeasibility logs | **Not tested** / Partial | Mocked unit tests or alternate solvers |
| `pyomo_model_solve` | `pyomo_model_solve` (core Ipopt path) | Partial | All OPF/TEP case solves |
| `pyomo_model_solve` | `export_solver_progress_to_excel` | **Not tested** | Progress + Excel export |
| `solver_utils` | probe helpers | **Tested** | `test_solver_utils` (mocked) |

---

## Coverage gaps by module

### `ACDC_Static_TEP.py` — 48%

```
61, 63, 70, 72, 74, 76, 90, 103-106, 124-125, 134-135, 138-139, 149-150, 177-184, 191, 239, 241-242, 247-248, 302, 305, 307-308, 312-313, 315, 348-349, 383, 391, 418-419, 425, 441, 451, 463-468, 476-504, 684, 687, 706, 709, 728, 731, 747-763, 772, 775, 794, 797, 833-836, 855-859, 866, 872, 875-879, 882-886, 889-893, 896, 903, 905, 907, 910-914, 916, 920-924, 927, 969, 1188-1248, 1258, 1260, 1262-1263, 1270-1341, 1345-1400, 1404-1459, 1487-1565, 1569-1572, 1603-1606, 1693-1695, 1729, 1755-1777, 1812, 1824, 1835, 1847-1848, 1861, 1896, 1936-2009, 2022-2046, 2049-2120, 2123-2142, 2145, 2148-2172, 2176-2391, 2417-2565, 2569-2579
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 684–927, 1610–1777, 2176–2579 | MS TEP solve, export, sensitivity | **Not tested** |
| 61–504, 1188–1248 | Static TEP / linear TEP | Partial |

---

### `ACDC_OPF.py` — 83% (OPF-only; solver code in `pyomo_model_solve.py`)

```
49, 64, 69, 119-122, 143, 236, 245, 273-291, 297-307, 312, 320-331, 360, 370, 380-382, 388-397, 402-405, 410, 415-425, 430-438, 443-449, 459-471, 476, 597-598, 646-647, 865, 879, 905, 939
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 273–291 | `fx_conv` | **Not tested** |
| 297–471 | `opf_obj*` branches (array losses, price zones, REC, …) | Partial |
| 597–647, 865–939 | `translate_pyf_opf`, `calculate_objective*` | Partial |
| — | `opf_line_res`, `opf_price_price_zone`, `opf_step_results` | **Tested** — `test_opf_result_helpers` |
| — | `optimal_pf` / `optimal_l_pf` | Partial — case OPF tests |

---

### `pyomo_model_solve.py` — 19%

```
16, 33-122, 143-224, 231-362, 390-483, 494-545, 560-653, 674, 680, 682, 689, 699-701, 721-724, 727-730, 795, 805-852, 856, 859-871, 876-881, 888, 890, 893-900, 904, 908-909, 944-945, 970-971, 999-1000, 1011, 1016-1060, 1077-1078, 1097-1098, 1107-1108, 1113-1114, 1118-1127, 1140-1146, 1149-1152, 1166-1220
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 33–122 | `log_infeasible_constraints_limited` | **Not tested** |
| 143–362 | `_gurobi_callback`, `_parse_*_log`, `_solver_progress` | **Not tested** / Partial (Ipopt log parse via solves) |
| 390–653 | `reset_to_initialize`, `_quick_feasible_point_check`, `_store_pyomo_results_on_grid` | Partial |
| 674–1220 | `pyomo_model_solve` (NLP warmstart, callbacks, failure paths) | Partial — happy Ipopt path only |
| 1166–1220 | `export_solver_progress_to_excel` | **Not tested** |

Note: low **line %** is expected — module is large and mostly alternate-solver / failure / callback paths; the main Ipopt path runs on every OPF/TEP solve but does not cover all branches.

---

### `ACDC_MultiPeriod_TEP.py` — 69%

```
40, 83, 89, 102, 111, 113, 117, 172, 248, 274, 300, 317, 325, 329, 341-400, 415, 417, 446, 456, 475, 532, 534, 557, 561, 579, 603, 607, 645, 649, 663-665, 687, 691, 705-709, 730, 734, 748-752, 767, 776, 784, 794-796, 801-803, 808-810, 815-817, 823, 826, 836-837, 847, 851, 856, 858, 949-953, 955, 972, 1072-1074, 1084, 1110, 1220-1244, 1248-1253, 1263, 1270-1287, 1315-1317, 1342-1344, 1371-1373, 1400-1426, 1429-1453, 1486-1491, 1496-1497, 1500, 1502, 1511-1513, 1521, 1528, 1535-1537, 1543, 1546, 1553-1556, 1562, 1565-1568, 1574, 1580, 1583, 1591, 1644, 1755, 1761-1766, 1864-1962, 1977-1998, 2022-2062, 2067-2078, 2092-2104, 2163-2233, 2277-2386, 2395, 2400, 2402, 2426-2431, 2443-2445, 2448, 2453-2457
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 40–972, 1220–1453 | MP TEP core + export | Partial — `case24_MP` full solve |
| 1486–2457 | MP+MS, post-MP OPF, decommission | **Not tested** |

---

### `Time_series_clustering.py` — 40%

```
74-78, 107, 141-142, 154-155, 166, 182-191, 222, 241-262, 297-318, 321-330, 332-335, 363-505, 517-575, 586-607, 647-648, 658-701, 746, 752-756, 806-819, 823-830, 867-891, 925, 935-938, 944, 980-1003, 1015-1016, 1050, 1060-1063, 1069, 1091-1147, 1166-1230, 1237, 1261-1267, 1278-1298, 1340, 1342, 1344, 1346, 1348, 1378, 1388-1399, 1412-1414, 1444-1520, 1523-1537, 1557-1671, 1680-1746, 1751-1781, 1787-1789, 1829, 1840, 1850, 1893-1915, 1925-1956, 2016, 2024, 2030, 2035
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 74–335, 363–505 | `identify_correlations` | Partial |
| 517–1781 | plotting, ward/PAM, elbow, relationship | **Not tested** |

---

### `Market_Coeff.py` — 63%

```
80, 153, 192-203, 207, 252, 258, 282, 303, 305, 410, 464-467, 480, 506, 612, 637-751, 791-869
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 192–203 | DST spring/fall (`3B` hour) | **Not tested** |
| 410, 464–467 | `eq_price` branches (`eq_price > 15`) | **Not tested** |
| 637–751 | `compute_hour_of_year` | **Not tested** |
| 791–869 | `clean_entsoe_data` | **Not tested** |
| (other) | `price_zone_coef_data`, `price_zone_data_pd`, `plot_curves` | **Tested** — `test_market_coeff` |

---

### `Graph_Dash.py` — 72%

```
138-140, 376, 396-416, 443, 459-460, 478, 483, 490, 494, 501, 536, 539-545, 672-674, 684-688, 698-702, 730-732, 761, 776, 796, 813-814, 816, 821-873, 879-880
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 47–201 | `plot_TS_res_from_ts` (all plot types) | **Tested** — `test_graph_dash` |
| 218–447 | `create_dash_app` callbacks | Partial — toggle, options, graphs tested |
| 396–416 | `update_limits` callback | **Not tested** (pandas edge on synthetic frame) |
| 459–501, 879–880 | `run_dash` / `run_ts_dash` / `run_mp_ts_dash` | **Not tested** — `app.run()` |
| 672–873 | MP dash compare mode + second plot | Partial |

---

### `solver_utils.py` — 81%

```
116-118, 128-129, 140, 145-147, 212, 215-218, 227-231, 234, 240-242, 246
```

| Lines | Function / block | Status |
|-------|------------------|--------|
| 56–236 | `check_*`, `format_solver_report` | **Tested** / Partial |
| 116–147 | verbose OR-Tools branches | **Not tested** |

---

### Other modules (missing lines only)

**`Export_files.py` 38%:** `47, 75-86, 98, 107, 130-134, 151-163, 246, 296-312, 326-328, 355-358, 418-440, 482-491, 513, 522-736, 748-841`

**`Mapping.py` 44%:** `48-113, 125, 242-246, 252, 258, 287, 291, 294-303, 306-312, 342, 370, 375, 466, 486, 495, 502, 522, 531, 566-570, 576, 586-591, 622, 627-629, 632-633, 643, 670-671, 691-693, 716-724, 732, 759, 806, 812, 833, 855-861, 877-889, 893-898, 919-1126, 1148-1541, 1552, 1559-1561, 1568, 1570, 1572, 1574, 1582, 1611, 1613`

**`AC_OPF_L_model.py` 38%, `Time_series.py` 54%, `Results_class.py` 60%, `Array_OPT.py` 54%, `Graph_and_plot.py` 57%** — see `term-missing` from full run.

---

## CI skips (expected)

| Test | Reason |
|------|--------|
| `test_sequential_array.py` | No Pyomo MIP/CSS-L solver (Gurobi/GLPK) |

---

## Internal planning doc

For backlog / prioritized next tests, see `my_tests/pyflow_acdc_coverage_improvement_map.md`
(workspace only; not shipped with the package).
