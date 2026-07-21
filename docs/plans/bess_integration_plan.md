# BESS integration plan for pyflow_acdc

Implementation plan for adding Battery Energy Storage Systems (BESS) to the nonlinear AC/DC OPF, based on:

- Mario Useche-Arteaga et al., *Wind Energ. Sci.* 11, 349–372 (2026) — §3.3 BESS model
- Reference script: `mario_implementation/18414805/OPF_ACDC_Energy_Islands.py`

## Documentation (GitHub / Read the Docs)

| Document | Path | Read the Docs |
|----------|------|---------------|
| User guide | [docs/usage_storage.rst](../usage_storage.rst) | [usage_storage](https://pyflow-acdc.readthedocs.io/en/latest/usage_storage.html) |
| API reference | [docs/api/storage.rst](../api/storage.rst) | [api/storage](https://pyflow-acdc.readthedocs.io/en/latest/api/storage.html) |
| This plan | [docs/plans/bess_integration_plan.md](bess_integration_plan.md) | — |

Permanent GitHub links (``main`` branch):

- User guide: https://github.com/CITCEA-UPC/pyflow_acdc/blob/main/docs/usage_storage.rst
- API: https://github.com/CITCEA-UPC/pyflow_acdc/blob/main/docs/api/storage.rst
- Plan: https://github.com/CITCEA-UPC/pyflow_acdc/blob/main/docs/plans/bess_integration_plan.md

---

## 1. Decisions (resolved gaps)

| ID | Topic | Decision |
|----|--------|----------|
| G1 | Scope | **BESS only** for now (no hydrogen / electrolyzer) |
| G2 | OPF modes | **Nonlinear OPF only** (no linear OPF, no TEP sizing) |
| G3 | Multi-hour optimization | **Coupled horizon** in a new top-level script `window_opf.py`; BESS constraints live **inside the NL model** (not a post-processing layer) |
| G4 | Time series (`ts_acdc_opf`) | **Deferred** — when added later, use **myopic** SoC propagation only (one hour at a time) |
| G5 | SoC units | **SoC in pu** in the Pyomo model; physical capacity via class attribute **`E_max` [MWh]** (enables future degradation modelling on `E_max`) |
| G6 | Simultaneous charge/discharge | Keep Mario/paper formulation (separate `P_charge` / `P_discharge` vars, no exclusivity binary). Optimizer should cancel overlap in practice. **Add code comment: revisit later** if artefacts appear |
| G7 | Apparent-power limit | **Per element**, side-dependent (mirrors `Ren_Source`): AC — `(P_dis − P_ch)² + Q² ≤ S_max²`; DC — active power only, `|P_dis − P_ch| ≤ P_max` (no `Q`, no S-circle) |
| G8 | Validation | **Yes** — reproduce Mario 24 h BESS dispatch on PEI case |
| G9 | PEI hub node | Mario uses index `0` (offshore hub). In `PEI_grid.py` the equivalent bus is **`PE_Island`** (220 kV PV hub). Confirm index mapping during validation setup |
| G10 | Public API | **`add_storage(grid, node, ...)`** in `grid_modifications.py` |
| G11 | Economics | **Operation only** — no CAPEX / investment variables |

### Phase 4 implementation (locked)

| ID | Topic | Decision |
|----|--------|----------|
| P4-1 | SoC coupling | **Parent `window_soc_links` only** — chain SoC across `hour_model[t]`; each block keeps **power limits only** (charge/discharge bounds, AC S-circle, DC `P_max`). |
| P4-1a | `window_block=True` | **Skip** in-block use of `soc_initial` / `SoC_prev ← soc_initial` and **`storage_soc_balance`** (parent owns dynamics). **Remove** `storage_soc_final_*` in blocks. Standalone `optimal_pf` (no flag) keeps Phase 2 snapshot behaviour including `soc_initial` / optional `soc_final`. |
| P4-1b | Energy state (future) | Comment in code/plan: parent links may later use **actual energy** [MWh] (or `SoC × E_max_eff`) instead of pu SoC, to support **capacity degradation** / time-varying `E_max`. Phase 4 v1 stays **pu SoC** links. |
| P4-2 | Builder API | **`opf_create_nl_model_acdc(..., window_block=True)`** — snapshot builder with block-specific SoC omissions; **`window_opf.py`** assembles blocks + parent links + objective. |
| P4-3 | Objective | **No new objective terms** — `model.obj = sum_t opf_obj(hour_model[t], …)` using existing per-block operational objectives only. |
| P4-4 | Hour indexing | **Python convention:** block indices `t ∈ {0, …, T−1}` aligned with `Time_series` row indexing (0-based). |
| P4-5 | Block build | **Clone structure** (cf. `multi_scenario_TEP`: build template once, `clone()` per hour, patch mutable params). |
| P4-6 | Window scope (v1) | **Single coupled window per call** (e.g. one 24 h solve). **Rolling / sliding window** over a long horizon → **next phase** after Phase 4 v1 + Phase 5. |

### Architecture principle

Hooks mirror **`Gen_AC` / `Gen_DC` / `Ren_Source`**:

- `Storage_AC` / `Storage_DC` classes (`storageNumber` / `storageNumber_DC`)
- `Node_AC.connected_storage` and `Node_DC.connected_storage`
- `Grid.storage_elements` (both sides in one list, like `RenSources`)
- Nodal aggregation: `PGi_storage` / `QGi_storage` (AC) and `PGi_storage_DC` (DC only)
- `connected` flag (`AcDcSide.AC` / `AcDcSide.DC`) — same branching pattern as `Ren_Source.connected` in the NL model

### Two run modes (by design)

```
┌─────────────────────────────────────────────────────────────────┐
│  window_opf.py          Coupled multi-hour NLP (paper-faithful) │
│  ─────────────────        Parent SoC links across hour blocks     │
│  hour_model[t] blocks     Primary path for Mario validation       │
│  (Phase 4 v1: one window)                                         │
├─────────────────────────────────────────────────────────────────┤
│  ts_acdc_opf (later)      Myopic sequential hours (G4 deferred)   │
│  ─────────────────        SoC_prev param updated each hour        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical model (NL)

Reference: paper Eqs. (24)–(31), Mario script lines 584–605 (AC hub BESS only).

### 2.1 Shared (AC and DC)

**Variables (pu on `grid.S_base`, per element `s`, per time `t` when window OPF)**

| Symbol | Pyomo name (proposed) | AC | DC |
|--------|----------------------|----|----|
| `P^c_{s,t}` | `P_storage_charge[...]` | ✓ | ✓ |
| `P^d_{s,t}` | `P_storage_discharge[...]` | ✓ | ✓ |
| `Q^b_{s,t}` | `Q_storage[...]` | ✓ | — |
| `e_{s,t}` | `SoC[...]` | ✓ | ✓ |

**Parameters (class attributes)**

| Symbol | `Storage_AC` | `Storage_DC` |
|--------|--------------|--------------|
| `E_max` [MWh] | ✓ | ✓ |
| `η_c`, `η_d` | ✓ | ✓ |
| `soc_min`, `soc_max`, `soc_initial`, `soc_final` | ✓ | ✓ |
| `P_charge_max`, `P_discharge_max` [pu] | ✓ | ✓ |
| `S_max` [pu] | apparent-power rating | — |
| `P_max` [pu] | — | max active-power rating |
| `Δt` | `dt_hours` | `dt_hours` |

**SoC dynamics** (same both sides):

```
SoC[s,t] = SoC[s,t-1] + (dt / E_max[s]) * (eta_c * P_charge[s,t] - P_discharge[s,t] / eta_d)
```

Plus terminal SoC, bounds on SoC, and separate charge/discharge power bounds (G6 note unchanged).

**Net active injection** (generation convention):

```
P_storage_net[s,t] = P_discharge[s,t] - P_charge[s,t]
```

### 2.2 AC vs DC power limits — mirror `Ren_Source`

Today in `ACDC_OPF_NL_model.py`, renewables branch on `ren_source.connected`:

```python
# Q bounds: AC only (DC → (0, 0))
# S-circle: S_renS_AC_limit_rule skips when connected == 'DC' or Max_S is None
#   (P*gamma)^2 + Q^2 <= Max_S^2   # AC only
# DC nodal sum → PGi_ren_DC[node]; no QGi_ren on DC
```

Storage should follow the **same split**:

| Aspect | AC (`Storage_AC`) | DC (`Storage_DC`) |
|--------|-------------------|-------------------|
| Reactive var `Q` | `Q_storage[s]` with free bounds (or ±`S_max`) | **None** — do not create `Q_storage` for DC indices |
| Converter / S-limit | Paper Eq. (30)–(31): `(P_d − P_c)² + Q² ≤ S_max²` per element | **No S-circle.** Net active limit: `|P_d − P_c| ≤ P_max` (equivalently two linear bounds, or rely on `P_charge_max` / `P_discharge_max` if `P_max = max(...)` ) |
| Nodal P aggregation | `PGi_storage[node]` = Σ `P_storage_net` over `node.connected_storage` (AC elements only) | `PGi_storage_DC[node]` = Σ `P_storage_net` (DC elements only) — parallel to `PGi_ren` / `PGi_ren_DC` |
| Nodal Q aggregation | `QGi_storage[node]` = Σ `Q_storage` → added to `Q_AC_node_rule` | **Skip** — no term in DC balance |
| Pyomo sets | `model.storage_AC` indexed by `storageNumber` | `model.storage_DC` indexed by `storageNumber_DC` |
| Export | write `P_charge`, `P_discharge`, `Q`, `SoC` on element | write `P_charge`, `P_discharge`, `SoC` only |

**AC apparent-power constraint** (element-level, not nodal — G7):

```
(P_discharge[s,t] - P_charge[s,t])**2 + Q_storage[s,t]**2 <= S_max[s]**2
```

**DC active-power constraint** (element-level):

```
P_discharge[s,t] - P_charge[s,t] <= P_max[s]
P_charge[s,t] - P_discharge[s,t] <= P_max[s]
```

(or `Constraint.Skip` if `P_max` already equals separate charge/discharge caps and bounds are sufficient — prefer explicit `P_max` rule for parity with class `loading` and future asymmetric ratings)

**Reference mapping in existing code**

| Ren_Source pattern | Storage equivalent (Phase 2) |
|--------------------|------------------------------|
| `S_renS_AC_limit_rule` | `S_storage_AC_limit_rule` on `model.storage_AC` |
| `Qren_bounds` → AC only | `Q_storage` vars / bounds on AC set only |
| `PGi_ren` + `Gen_PREN_rule` (AC nodes) | `PGi_storage` + `Gen_Pstorage_AC_rule` |
| `PGi_ren_DC` + `Gen_PREN_rule` (DC nodes) | `PGi_storage_DC` + `Gen_Pstorage_DC_rule` |
| `P_AC_node_rule`: `P_var += …` | `P_var += PGi_storage[node]` |
| `P_DC_node_rule`: `P_var += …` | `P_var += PGi_storage_DC[node]` |
| `Q_AC_node_rule`: `Q_var += …` | `Q_var += QGi_storage[node]` |

Mario PEI case remains **AC-only** (`Storage_AC` on `PE_Island`); DC storage is for general hybrid grids.

### 2.3 Nodal balance (summary)

AC (`P_AC_node_rule` / `Q_AC_node_rule`):

```
P_var += PGi_storage[node,t]
Q_var += QGi_storage[node,t]
```

DC (`P_DC_node_rule`):

```
P_var += PGi_storage_DC[node,t]
```

---

## 3. Implementation phases

### Phase 0 — Design freeze

**Goal:** Lock interfaces before coding.

**Tasks**

- [x] Confirm `PE_Island` as Mario hub mapping for validation (G9)
- [x] Fix sign convention docstring on `Storage_AC` (discharge = positive injection)
- [x] Define `dt_hours` default (= 1) on `Storage_AC`; window length remains a `window_opf` argument

**Deliverable:** This document approved; no open blocking questions.

---

### Phase 1 — Storage classes and grid hooks ✅

**Files:** `Classes.py`, `grid_modifications.py`, `grid_analysis.py`, `__init__.py`

#### 1.1 `Storage_AC` / `Storage_DC` (`Classes.py`)

Mirror `Gen_AC` / `Gen_DC` + `Ren_Source` connectivity:

```python
class Storage_AC:
    storageNumber, connected = AcDcSide.AC
    # S_max, Q, Node_AC, ...

class Storage_DC:
    storageNumber_DC, connected = AcDcSide.DC
    # P_max (no Q), Node_DC, ...
```

- `node.connected_storage.append(self)` on AC or DC host
- `reset_class()` for tests
- `S_base` setter rescales pu ratings

#### 1.2 `Grid` hooks

```python
self.storage_elements = []  # both Storage_AC and Storage_DC
```

- Property `nstorage`
- TS result keys: `storage_soc`, `storage_p_charge`, `storage_p_discharge`, `storage_q` (Q columns N/A or zero for DC in export)

#### 1.3 `Node_AC` / `Node_DC`

```python
self.connected_storage = []
```

#### 1.4 `add_storage()` (`grid_modifications.py`)

- `_look_up_node(..., ac_or_dc="any")` → `Storage_AC` or `Storage_DC`
- AC: `S_max_MVA` → apparent-power rating
- DC: `S_max_MVA` interpreted as **max active power MW** → `P_max`

#### 1.5 `analyse_grid`

- `grid.ESS = bool(grid.storage_elements)`

**Exit criteria:** Grid with `add_storage` builds; element appears on correct node; no OPF yet. ✅ Done (Phase 1)

---

### Phase 2 — NL model: BESS in `ACDC_OPF_NL_model.py`

**Goal:** BESS variables and constraints inside the existing NL builder (snapshot / single hour). **Branch AC/DC like `Ren_Source`** (§2.2 mathematical split).

#### 2.1 Snapshot mode — single hour ✅

When `grid.storage_elements` non-empty:

**Sets**

- `model.storage_AC` ← `[s.storageNumber for s in grid.storage_elements if s.connected == AcDcSide.AC]`
- `model.storage_DC` ← `[s.storageNumber_DC for s in grid.storage_elements if s.connected == AcDcSide.DC]`

**Variables**

| Var | Index | AC | DC |
|-----|-------|----|----|
| `P_storage_charge` | element id | ✓ | ✓ |
| `P_storage_discharge` | element id | ✓ | ✓ |
| `Q_storage` | `storageNumber` | ✓ | — |
| `SoC` | element id | ✓ | ✓ |
| `SoC_prev` | Param, mutable | ✓ | ✓ |

**Element constraints**

- SoC balance (one step), SoC bounds, charge/discharge bounds — both sides
- **AC:** `S_storage_AC_limit_rule` — `(P_d - P_c)² + Q² ≤ S_max²` (cf. `S_renS_AC_limit_rule`)
- **DC:** `P_storage_DC_net_limit_rule` — `|P_d - P_c| ≤ P_max` (cf. ren DC skipping S-circle; active limits only)
- G6 comment on charge/discharge overlap at both constraint sites

**Nodal aggregation** (cf. `Gen_PREN_rule` / `Gen_PREN_rule` on DC)

- AC: `PGi_storage[node]`, `QGi_storage[node]` from `node.connected_storage` (filter AC elements)
- DC: `PGi_storage_DC[node]` from DC-connected storage only

**Balance hooks**

- `P_AC_node_rule` / `Q_AC_node_rule`: add `PGi_storage`, `QGi_storage`
- `P_DC_node_rule`: add `PGi_storage_DC`

Gate entire block on `grid.ESS` (same as checking `RenSources` before building ren sets).

#### 2.2 Export (`export_acdc_nl_model_to_pyflow_acdc`) ✅

Map solved values back to each element:

| Field | `Storage_AC` | `Storage_DC` |
|-------|--------------|--------------|
| `P_charge`, `P_discharge`, `SoC` | ✓ | ✓ |
| `Q` | ✓ | — (leave unset / 0) |

Loop `grid.storage_elements` and branch on `connected` (same pattern as ren export at ~line 2259).

**Exit criteria:** Single-hour OPF with AC and DC storage on a hybrid toy grid; AC S-circle and DC P-only limits satisfied; export to `storage_elements`. ✅ Done (Phase 2)

---

### Phase 3 — `Results_class.py`

**Goal:** Reporting consistent with `ext_gen` / `ext_ren`.

#### 3.1 `ext_storage(print_table=True)`

Snapshot table per element:

| Column | Unit |
|--------|------|
| Name, Node, Side (AC/DC) | — |
| P charge / P discharge | MW |
| Q | MVAr (AC only; omit or “—” for DC) |
| SoC | pu (and MWh = SoC × E_max) |
| Loading | % (`S_max` AC / `P_max` DC) |

#### 3.2 `storage_window(print_table=True)` (window OPF)

- Hourly SoC, P_charge, P_discharge, Q per element
- Summary: energy charged/discharged [MWh], round-trip efficiency

#### 3.3 Wire into `Results.all()`

```python
if self.Grid.storage_elements:
    self.ext_storage(print_table=print_table)
if getattr(self.Grid, "window_opf_run", False):
    self.storage_window(print_table=print_table)
```

#### 3.4 Excel / CSV

- Add sheets to existing export paths when `save_res=True`

**Exit criteria:** `Results(grid).all()` prints storage table after OPF. ✅ Done (Phase 3.1)

---

### Phase 4 — Coupled horizon: time-indexed NL model + `window_opf.py`

**Goal:** Paper-faithful multi-hour NLP — extend the NL builder with a time index **and** add a separate top-level runner (not inside `ts_acdc_opf`).

> Snapshot BESS (Phase 2) stays the default for `optimal_pf`. Phase 4 adds `horizon=T` and `window_opf.py` together.

#### 4.1 Coupled window model — hour blocks + parent SoC links

**Pattern:** mirror `multi_scenario_TEP` (`model.hour_model[t] = pyo.Block(...)`) — **not** a full `(node, t)` re-index of the NL model.

```
model
├── hour_model[t]     ← opf_create_nl_model_acdc(..., window_block=True), t ∈ [0, T−1]
│   └── snapshot NL OPF + storage power limits (no in-block SoC boundaries)
└── window_soc_links  ← parent level only (P4-1)
    ├── t = 0:        anchor from soc_initial (+ dynamics using hour 0 dispatch)
    ├── t = 1…T−1:    SoC[s,t] = SoC[s,t−1] + Δt/E_max · (η_c P_c − P_d/η_d)
    └── t = T−1:      SoC[s,T−1] = soc_final[s]  (when set)
    # Future (P4-1b): link actual energy [MWh] or SoC×E_max_eff for degradation
    # Future: window_dc_ramp_links — same parent pattern (not Phase 4 v1):
    #   |P_line[t] − P_line[t−1]| ≤ ramp_max[line] · Δt  on selected DC lines
```

**Per-block vs parent (P4-1, P4-1a)**

| In each `hour_model[t]` (`window_block=True`) | At parent only |
|-----------------------------------------------|----------------|
| `P_charge`, `P_discharge`, `Q`, `SoC` vars + bounds | `soc_initial` @ **t = 0** |
| AC S-circle, DC `P_max`, charge/discharge caps | SoC chain @ **t = 1…T−1** |
| **Skip** `SoC_prev ← soc_initial` and **`storage_soc_balance`** | `soc_final` @ **t = T−1** (when set) |
| **Omit** `storage_soc_final_*` | |

Standalone `optimal_pf` (no `window_block`) unchanged: `SoC_prev = soc_initial`, one-step balance, optional `storage_soc_final_*`.

**Build (P4-2, P4-5)**

1. Build one template snapshot via `opf_create_nl_model_acdc(..., window_block=True)`.
2. For each `t`: `clone()` → `hour_model[t]`, patch hourly params from `Time_series` (cf. `_modify_parameters`).
3. Add parent `window_soc_constraints` + `sum_t` objective (P4-3).

Reference: `ACDC_Static_TEP.multi_scenario_TEP` + `MS_TEP_constraints`.

> **Future — same hour-block / parent-link pattern:** **DC line ramp-rate limits** between consecutive hours, e.g. on `hour_model[t].PDC_to[line]` / `PDC_from[line]`:
>
> `|P_line[t] − P_line[t−1]| ≤ ramp_max[line] · Δt_hours` (pu or MW rating on the line class).
>
> Defer until after BESS window OPF is stable; requires `ramp_max` (or similar) on `Line_DC` / grid data.

#### 4.2 Runner — `window_opf.py` (single window, v1)

**New file:** `pyflow_acdc/window_opf.py`

```python
def window_opf(
    grid,
    start=0,
    end=23,
    ObjRule=None,
    solver="ipopt",
    tee=False,
    ...
):
    """
    Build one coupled NL model over hours start…end (0-based, inclusive).
    Phase 4 v1: single window per call (e.g. 24 h). Rolling windows → Phase 4.3.
    Updates ren availability and prices per hour from grid.Time_series.
    Solves once; exports trajectory to grid + storage_elements.
    Sets grid.window_opf_run = True.
    """
```

**Responsibilities**

1. `analyse_grid(grid)`
2. Apply hourly data from `Time_series` into each block (cf. `update_grid_scenario_frame` + `_modify_parameters`)
3. Clone template → `model.hour_model[t]` for `t ∈ range(start, end + 1)` with `window_block=True`
4. Add parent `window_soc_constraints` (P4-1)
5. `model.obj = sum_t opf_obj(hour_model[t], …)` — **no new objective terms** (P4-3)
6. `pyomo_model_solve`
7. Window export → `grid.window_opf_results` DataFrames

**Grid flag:** `grid.window_opf_run = True` (set in `reset_run_flags` companion or window entry)

**Export in `__init__.py`:** `from .window_opf import window_opf`

**Exit criteria:** One coupled 24 h window on PEI completes; SoC obeys terminal constraint.

#### 4.3 Rolling window (deferred — after Phase 4 v1 / Phase 5)

**Not in Phase 4 v1.** `window_opf` solves **one** contiguous horizon per invocation (P4-6).

Later: **rolling / sliding window** over a long `Time_series` — e.g. advance `start/end` by `step` hours, warm-start or carry `soc_initial` from previous window’s terminal SoC. Separate API (e.g. `rolling_window_opf`) or extended kwargs; design after Mario validation.

---

### Phase 5 — PEI validation vs Mario (G8)

**Files:** `pyflow_tests/...` or `example_grids/OPF/PEI_window_opf.py`

**Setup**

1. Start from `PEI_grid()` (`S_base` may need alignment with Mario's 3500 MVA — document any base mismatch)
2. `add_storage(grid, 'PE_Island', ...)` with Mario parameters:
   - `P_nom = 0.33 pu`, `η_c = 0.85`, `η_d = 0.9`
   - `soc_initial = soc_final = 0.5`, `soc_min = 0.1`, `soc_max = 1.0`
3. Attach wind `Time_series` from `power_matrix.csv`
4. Attach hourly prices (BE, GB, DK)
5. Run `window_opf(grid, start=0, end=23)`  # 0-based, 24 h window

**Compare**

| Quantity | Tolerance |
|----------|-----------|
| `P_charge`, `P_discharge` per hour | Document Δ (expect small if grid/base aligned) |
| `SoC` trajectory | Same |
| Export powers / revenue | Qualitative match |

**Note:** Exact numeric match may require harmonizing `S_base`, slack specification, and converter loss coefficients with Mario's script.

**Exit criteria:** Validation notebook or test committed; differences explained in test docstring.

---

### Phase 6 — Documentation

**Goal:** User-facing docs and citations for BESS + `window_opf`, aligned with existing Sphinx layout. Cite Mario’s energy-island paper as the modelling reference.

**When:** After Phase 4–5 (API and PEI example stable enough to literalinclude). Docstrings in Phases 1–4 can be drafted earlier; RST pages land here.

#### 6.1 API reference

**New file:** `docs/api/storage.rst` ✅

- `.. autoclass:: pyflow_acdc.Storage_AC`
- `.. autofunction:: pyflow_acdc.add_storage`
- `.. autofunction:: pyflow_acdc.window_opf` (once implemented)
- BESS constraint summary (SoC dynamics, S-circle) with cross-ref to usage page
- Sign convention table (charge / discharge / injection)

**Update:** `docs/api/grid_mod.rst` — link to `storage.rst` ✅

**Update:** `docs/api/opf.rst` — short note that NL OPF supports `storage_elements`; coupled multi-hour runs use `window_opf` (Phase 4)

#### 6.2 Usage guide

**New file:** `docs/usage_storage.rst` ✅

Sections:

1. **Overview** — coupled horizon vs sequential `ts_acdc_opf` (deferred myopic mode)
2. **Adding a BESS** — `add_storage` workflow on any AC node
3. **Running `window_opf`** — time series, prices, `start` / `end`, objective (pending Phase 4)
4. **Results** — `Results.ext_storage`, `Results.storage_window`, Excel export (pending Phase 3)
5. **PEI example** — Princess Elisabeth Energy Island case; hub bus `PE_Island`
6. **Modelling note** — BESS formulation follows Useche-Arteaga et al. (2026) §3.3; implementation adapted to pyflow_acdc element/node hooks

**Literalinclude examples** (under `pyflow_tests/doc_examples/`):

- `storage/01_add_storage.py` — minimal `add_storage` on a small case ✅
- `storage/02_window_opf_pei.py` — PEI 24 h window OPF (validation-aligned) (pending Phase 5)

**Update:** `docs/index.rst` — add `usage_storage` and `api/storage` to toctree ✅

**Update:** `docs/usage.rst` — catalogue entry pointing to storage page ✅

#### 6.3 Citing

**Update:** `docs/citing.rst` — new subsection *For BESS / energy-island operation* ✅

> If you use the BESS or `window_opf` functionality, please also cite:
>
> M. Useche-Arteaga, P. Gebraad, V. A. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming – a case study based on the Princess Elisabeth Energy Island*, Wind Energ. Sci., 11, 349–372, https://doi.org/10.5194/wes-11-349-2026, 2026.

BibTeX block:

```bibtex
@article{usechearteaga2026energyislands,
  author  = {Useche-Arteaga, Mario and Gebraad, Pieter and Lacerda, Vinicius A. and Cheah-Mane, Marc and Gomis-Bellmunt, Oriol},
  title   = {Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the {Princess Elisabeth Energy Island}},
  journal = {Wind Energy Science},
  volume  = {11},
  pages   = {349--372},
  year    = {2026},
  doi     = {10.5194/wes-11-349-2026},
  url     = {https://doi.org/10.5194/wes-11-349-2026}
}
```

Also add a short acknowledgement in `usage_window_opf.rst` and in `Storage_AC` / `window_opf` module docstrings: *BESS model based on Useche-Arteaga et al. (2026).*

#### 6.4 Doc tests

**New:** `pyflow_tests/test_docs_storage.py` — smoke-run literalinclude examples ✅

**Exit criteria:**

- [x] `usage_storage.rst` and `api/storage.rst` build in Sphinx without warnings
- [x] Mario paper cited in `citing.rst`, usage page, and plan §7
- [x] Doc examples execute in CI (`test_docs_storage.py`)
- [ ] Cross-links between plan and docs on GitHub / Read the Docs ✅

---

### Phase 7 — Deferred: myopic time series (G4)

**Not in v1.** When implemented in `Time_series.py`:

- Reuse **snapshot** NL model (Phase 2.1) with `SoC_prev` param
- Each hour in `ts_acdc_opf`: set `SoC_prev` from previous solution → solve → record
- No terminal `soc_final` except optionally on last hour of run
- Document that this is **not** equivalent to `window_opf` global optimum

---

### Phase 8 — Deferred: hydrogen, TEP, linear OPF

Out of scope per G1, G2, G11.

---

## 4. File touch list (summary)

| File | Changes |
|------|---------|
| `Classes.py` | `Storage_AC`, `Storage_DC`, `Grid.storage_elements`, `Node_AC/DC.connected_storage` |
| `grid_modifications.py` | `add_storage()` |
| `grid_analysis.py` | `ESS` flag |
| `ACDC_OPF_NL_model.py` | Snapshot storage (Phase 2 ✅); `window_block=True` skips in-block SoC boundaries (P4-1a) |
| `window_opf.py` | **New** — clone hour blocks, parent SoC links, single-window runner (Phase 4.2) |
| `ACDC_OPF.py` | `storage_info` in `translate_pyf_opf`; pass horizon if needed |
| `Results_class.py` | `ext_storage()`, `storage_window()` |
| `Time_series.py` | Phase 7 only (deferred) |
| `__init__.py` | Export new symbols |
| `pyflow_tests/...` | PEI validation test |
| `docs/api/storage.rst` | **New** — `Storage_AC`, `add_storage`, `window_opf` API |
| `docs/usage_storage.rst` | **New** — usage guide + Mario modelling note + plan link |
| `docs/citing.rst` | BibTeX + citation for Useche-Arteaga et al. (2026) |
| `docs/index.rst`, `docs/usage.rst` | Toctree / catalogue links |
| `pyflow_tests/doc_examples/storage/` | Literalinclude examples |
| `pyflow_tests/test_docs_storage.py` | Doc example smoke tests |

---

## 5. Other considerations

### Nonlinearities

- S-circle (AC) and AC power flow → **IPOPT** only; DC storage adds linear `P_max` bounds only
- SoC dynamics are linear in pu when `E_max` is fixed

### Code conventions

- No `getattr` on Pyomo model objects (workspace rule)
- Fail fast on missing / invalid storage parameters
- Prefer explicit model attribute access

### Future: degradation

- `E_max` on `Storage_AC` is the degradation hook (e.g. `E_max_eff = E_max * health_factor`)
- SoC stays in pu; MWh state = `SoC * E_max_eff`

### Future: exclusivity (G6 revisit)

If simultaneous charge/discharge appears in results:

- Option A: post-solve assertion / warning
- Option B: complementarity or small penalty
- Option C: single net power variable with sign (changes model structure)

---

## 6. Suggested implementation order

1. Phase 0 — design freeze  
2. Phase 1 — class + `add_storage`  
3. Phase 2 — snapshot NL model ✅  
4. Phase 3 — `Results.ext_storage` ✅ (3.1); `storage_window` with Phase 4  
5. Phase 4 — single-window `window_opf` + `window_block` (4.1–4.2)  
6. Phase 5 — Mario validation  
7. Phase 6 — documentation + citations  
8. Phase 4.3 — rolling window (later)  
9. Phase 7 — myopic TS (later)

---

## 7. References

### Primary reference (BESS / energy-island operation)

M. Useche-Arteaga, P. Gebraad, V. A. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt, *Optimizing the operation of energy islands with predictive nonlinear programming – a case study based on the Princess Elisabeth Energy Island*, Wind Energ. Sci., 11, 349–372, 2026, https://doi.org/10.5194/wes-11-349-2026.

- BESS model: paper §3.3 (Eqs. 24–31) — implemented as `Storage_AC` + NL constraints
- Coupled multi-hour operation: paper §3.5–3.6 — implemented via `window_opf.py`
- Local PDF: `mario_implementation/wes-11-349-2026.pdf`
- Reference implementation: `mario_implementation/18414805/OPF_ACDC_Energy_Islands.py`

### pyflow_acdc assets

- PEI grid — `pyflow_acdc/example_grids/PF/PEI_grid.py` (hub: **`PE_Island`**)
- User guide — [docs/usage_storage.rst](../usage_storage.rst) · [Read the Docs](https://pyflow-acdc.readthedocs.io/en/latest/usage_storage.html)
- API — [docs/api/storage.rst](../api/storage.rst) · [Read the Docs](https://pyflow-acdc.readthedocs.io/en/latest/api/storage.html)
- Integration plan — [docs/plans/bess_integration_plan.md](bess_integration_plan.md)
