# Convex AC/DC SOCP plan for pyflow_acdc

**Repository:** In-repo links target the [`mario_integration`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration) branch ([`plans/`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration/plans)).

Living implementation plan for **sparse second-order cone programming (SOCP)**,
**mixed-integer SOCP (MI-SOCP)**, and **chance-constrained programming (CCP)** from
Mario Useche-Arteaga et al. (SEGAN 2026 energy-hub paper).

Publication-grounded; remaining open items are in §7–§8. Owner decisions already locked
are in **§0.0** (L1–L28) and progress is in **§0.4** (do not re-litigate without an explicit change).

**Status:** Phase **0–3 scaffolding largely landed** in tree (2026-07-29). Deterministic
**sparse** SOCP builder + runners exist; **no end-to-end solve validation yet**. BESS / H₂ /
CCP remain deferred (L18), but **`T`-indexed CVXPY variables are kept** so interperiod
coupling can be added later without reformulating the model layout.

**Handoff note (2026-07-29):** work paused to switch tasks. Resume from **§0.4 Implementation
log** + locked decisions **L21–L28**. Do not re-litigate L1–L28 without an explicit change.

**Primary reference**

M. Useche-Arteaga, W. Gil-González, P. Gebraad, M. Cheah-Mane, V. Lacerda, and
O. Gomis-Bellmunt, *Efficient AC/DC energy hubs operation using sparse SOCP relaxation
and chance-constrained optimization*, Sustainable Energy, Grids and Networks **46**, 102217
(2026). https://doi.org/10.1016/j.segan.2026.102217

**Related pyflow_acdc assets**

| Document | Link |
|----------|------|
| BESS / H₂ operation (NLP, shipped) | [bess_integration_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_integration_plan.md) |
| BESS sizing (separate build) | [bess_sizing_ramp_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_sizing_ramp_plan.md) |
| OPTYMADD integration (separate) | [optymadd_integration_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/optymadd_integration_plan.md) |
| Architecture map | [docs/architecture.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/docs/architecture.md) |
| Example PEI grid (optional later) | [`PEI_grid.py`](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/pyflow_acdc/example_grids/PF/PEI_grid.py) |
| PEI window OPF tests (NLP peer) | [`test_pei_window_nl_opf_bess_h2.py`](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/pyflow_tests/test_pei_window_nl_opf_bess_h2.py) |

Local PDF (user-provided): `efficient.pdf` (2026).

Mario reference (workspace): `mario_implementation/scop/ACDC_PE_Simplified_Efficient_SOCP_Multiperiodo.py`
(+ local Autumn price CSVs; `wpp_forecast.csv` may still be needed for smoke runs).

---

## 0. Scope of this plan

### 0.0 Locked owner decisions

| ID | Decision |
|----|----------|
| L1 | **Scope = Paper A only** (SEGAN 2026 sparse SOCP + CCP for AC/DC energy hubs). ADN / robust EMP paper is **out of scope** and deleted from this plan. |
| L2 | Stack is **case-agnostic**: `socp_optimise(grid, …)` must load **any** pyflow `Grid` (+ time series / options). Do not design the API or builder around a single validation case. Case-specific regression (PEI tables, etc.) is deferred. |
| L3 | Modeling language is **CVXPY**, not Pyomo. |
| L4 | Files: **`convex_model.py`** (CVXPY build) + **`ACDC_convex.py`** (pyflow call layer). |
| L5 | Public entry point: **`socp_optimise`** (British spelling). |
| L6 | Optional dependency extra: **`[SOCP]`** in `pyproject.toml` (`cvxpy` + chosen solvers). Guarded import in `__init__.py`, same pattern as `[OPF]`. |
| L7 | Reference script: `mario_implementation/scop/ACDC_PE_Simplified_Efficient_SOCP_Multiperiodo.py` (sparse multiperiod). Align formulation with this + paper; case data stays outside the API. |
| L8 | **Ybus for SOCP (v1 = option B):** use existing AC Ybus / `Line_AC` topology. **Ignore** converter AC-side RL branches in the SOCP graph. Converters enter only as **AC↔DC power + loss + rating**. Later **option A:** a `CONVEX_Ybus` that adds AC nodes for converter elements. |
| L9 | Included AC series elements share one Ybus (lines / trafos / cables as line-class topology). |
| L10 | Bus roles from pyflow types: gen / export (ext-grid) / converter AC / zero-injection — generalise Mario’s hard-coded node lists. |
| L11 | Nodal balance uses **`conj(S) == flow`** (Mario: simpler conj-on-V bookkeeping). |
| L12 | Renewables v1: **`gamma` as parameter**, start with `gamma = 1` (P fixed to forecast×gamma, Q = 0). |
| L13 | Converter loss v1: **always-`+`** form `Ploss = a + c·…` (worst-case; same spirit as NL OPF using `c_rect`). Defer inverter/rectifier sign flip. |
| L14 | DC polarity: **`pol = pcn`** as in Mario. |
| L15 | **AC and DC thermal / rating limits are mandatory** (Paper A Eqs. 4–7 → SOCP 40–43). Mario’s script omitted AC limits — **do not** omit them in pyflow. |
| L16 | Exports = ext-grid **negative** injections (pyflow). Paper/Mario revenue form: **`min Σ Re(S_export)·price`**. **Superseded for default runner by L27** (`Energy_cost`); `Ext_Gen` weight still available for priced-export style. |
| L17 | Mario Autumn price CSVs are **demo only**, not a canonical paper-table target. |
| L18 | **No BESS constraints in first SOCP build** (defer MI exclusivity / SoC). Keep **`T`-indexed** variables so BESS coupling can be added later (paper has BESS). |
| L19 | **Sparse only — GREEN LIGHT.** Ship sparse edge-set SOCP exclusively. Do **not** plan or implement a dense SOCP path (paper Table 2 comparison is optional later research, not a product requirement). |
| L20 | Model is **grid pu only**; € scaling only in objective / reporting via `S_base`. |
| L21 | **One CVXPY model with `(…, T)` indexing**, not Pyomo-style per-frame submodels. Single-period = `T=1`; window = ordered `frame_ids` with `T = len(frame_ids)`. |
| L22 | Internal builder name: **`socp_model(grid, d)`**. Public runners: **`socp_optimise`** (single) + **`soc_window_optimisation`** (multiperiod / window). |
| L23 | Builder consumes a **prepared SOCP data object** only (no ad hoc `P_ren` / `P_ext_bounds` kwargs on the builder). Translation owns `T`, `frame_ids`, profiles. |
| L24 | **AC generators (incl. ext-grids) are decision variables** (`PGi_gen`, `QGi_gen`) with OPF-NL-style bounds. Ext-grids are **not** a separate asset class — same gen family, different limits / sell logic. |
| L25 | **Renewables stay separate** from gens: availability from TS → fixed (parameter) injection v1 (`gamma` param, L12). |
| L26 | Time series come from **`grid.Time_series`** (`type` + `element_name` + `data`). Elements do **not** hold arrays (only `TS_dict` links). `grid.Time_series` is always present (`[]` if empty) — access directly, no soft `getattr`. |
| L27 | Objective uses NLP **`ObjComponent` / `weights_def` / `default_obj_weights()`**. Supported v1: `Energy_cost`, `Ext_Gen`, `AC_losses`, `DC_losses`, `Converter_Losses`. Unsupported active weights → **hard `NotImplementedError`**. Default = `Energy_cost` w=1. |
| L28 | Flags **`grid.ACmode` / `grid.DCmode`** (from `analyse_grid`) gate DC / converter blocks. Converters require **both** AC and DC modes plus `conv_data`. |

### 0.1 Problem class (Paper A)

| Aspect | Paper A (SEGAN 2026) |
|--------|----------------------|
| System | Hybrid **AC/DC energy hub** (offshore island in the paper; any Grid in pyflow) |
| Objective | Paper: maximize **export revenue** (Eq. 1 / 37). pyflow v1 default: NLP-style **`Energy_cost`** (L27); priced export via `Ext_Gen` / L16 |
| Network | AC collection + HVDC + converters + export links |
| Assets (paper) | Wind, BESS (MI), H₂, CCP |
| Assets (pyflow v1) | Gens as vars (L24), wind/`gamma` (L12/L25), converters, AC/DC + limits — **no BESS constraints** (L18); `T` kept |
| Convex core | **Sparse SOCP** only (L19) |
| Uncertainty | CCP later; not required for first deterministic build |
| Time | Multi-period when TS present |

SOCP lifting: `h_k = |v_k|²`, sparse complex `w_km` (Mario), rotated SOC inequality.

### 0.2 In scope / out of scope

| In scope (v1+) | Out of scope / deferred |
|----------------|-------------------------|
| Sparse SOCP AC/DC power flow from any `Grid` | Modifying `ACDC_OPF_NL_model.py` in place |
| Sparse edge-set formulation (Paper A §3) | BESS sizing; TEP investment |
| **AC + DC thermal limits** (Eqs. 40–43; L15) | ADN EMP / Paper B |
| Converter AC↔DC coupling + loss (L8/L13) + rating | `CONVEX_Ybus` converter AC nodes (option A — later) |
| `ObjComponent` objective (L27); `gamma` (L12); `T`-indexed vars (L21) | **BESS / H₂ / CCP constraints in first build** (add later) |
| `socp_optimise` + `soc_window_optimisation` (L22) | Dense SOCP (L19); Pyomo port; case-specific API design |
| Optional NLP benchmark later | Modifying `window_opf.py` / NLP builders |

### 0.3 Distinction from existing pyflow_acdc stacks

| Stack | Formulation | Relation to this plan |
|-------|-------------|----------------------|
| `optimal_pf` + `ACDC_OPF_NL_model.py` | **Nonlinear** polar OPF (`V`, `θ`) | Optional “exact” baseline later |
| `window_nl_opf` | Coupled multi-hour **NLP** on full grid + BESS + H₂ | Operational peer (Pyomo/IPOPT) |
| `soc_window_optimisation` | Coupled multi-hour **SOCP** (`T`-indexed CVXPY); no BESS yet | This plan (L22) |
| `bess_sizing.py` (planned) | Plant-level POI NLP | Separate problem class |

**This plan adds a dedicated CVXPY SOCP / MI-SOCP stack** with its own model builder and
runner — not a patch to the shipped NLP builders.

### 0.4 Implementation log (2026-07-29) — resume here

**Files in tree**

| File | Role |
|------|------|
| [`pyflow_acdc/convex_model.py`](../pyflow_acdc/convex_model.py) | `build_socp_data`, `socp_model`, subsystem `*_variables` / `*_constraints` |
| [`pyflow_acdc/ACDC_convex.py`](../pyflow_acdc/ACDC_convex.py) | `translate_pyf_socp`, `socp_optimise`, `soc_window_optimisation`, weighted objective, export |
| [`pyproject.toml`](../pyproject.toml) | `SOCP = ["cvxpy"]` (+ folded into `All`) |
| [`__init__.py`](../pyflow_acdc/__init__.py) | Guarded export: `socp_optimise`, `soc_window_optimisation`, `translate_pyf_socp` |

**Data / call flow**

```
analyse_grid(grid)
        │
        ▼
translate_pyf_socp(grid, gamma, frame_ids, P_ext_bounds)
  • build_socp_data(grid)  → topology, Ybus edges, gen_data_AC, ren_nodes_AC, …
  • read grid.Time_series  → P_ren[node,t], prices[node,t]
  • attach T, frame_ids, P_ext_bounds
        │
        ├── socp_optimise → frame_ids=[frame_id] (default 0) → T=1
        └── soc_window_optimisation → full horizon or explicit frame_ids
                │
                ▼
        socp_model(grid, d)   # CVXPY vars shaped (…, T)
                │
                ▼
        _build_objective(variables, d, grid, weights_def)
                │
                ▼
        cp.Problem.solve → _export_to_grid → grid.socp_results
```

**`convex_model.py` structure (mirrors NL OPF split)**

- `build_socp_data(grid)` → `SimpleNamespace` (static)
- `generator_variables` / `generator_constraints` — `PGi_gen`, `QGi_gen`
- `ac_variables` / `ac_constraints` — `h_AC`, sparse complex `w_AC`, balance, thermals
- `dc_variables` / `dc_constraints` — `h_DC`, sparse real `w_DC`, `P_DC`, thermals
- `converter_variables` / `converter_constraints` — `Ss`, `Ploss`
- `socp_model(grid, d)` — orchestrator; reads `d.T`, `d.P_ren`, `d.P_ext_bounds`

**`translate_pyf_socp` rules**

- Infer `T` / `frame_ids` from `grid.Time_series` (or `T=1` if empty).
- Renewable TS types (`WPP`, `OWPP`, …) match `RenSource` / `Ren_source_zone` by `element_name`.
- Injection: `PGi_ren_base * availability[t] * np_rsgen * gamma` (sum if multiple RS on one node).
- Price TS match price zone / AC node by `element_name`.
- RenSource host bus: **`rs.Node`** (name) → `nodes_AC` node number (not `rs.Node_AC`).

**Objective v1 (not Mario export-only only)**

- Default `Energy_cost`: quadratic/linear/fixed gen cost coeffs × `S_base` (OPF-NL style).
- Also: `Ext_Gen`, `AC_losses`, `DC_losses`, `Converter_Losses`.
- Mario-style export revenue can still be expressed later / via weights; current default follows NL OPF `Energy_cost`.

**Done vs remaining**

| Done | Remaining / next |
|------|------------------|
| Scaffold + sparse AC/DC/conv + thermals | Install `cvxpy` + solver; smoke `build_only` / solve |
| Gen variables + ren from `grid.Time_series` | Load TS into balance if needed; validate vs Mario / NLP |
| Window + single runners + `T` indexing | BESS SoC chain across `t` (lifts L18); H₂; CCP |
| `[SOCP]` + guarded exports | Docs, Results section name, CI without commercial solver |
| Weighted `ObjComponent` objective | Map Paper A AC thermal expressions precisely (U-A8) |

**Explicit non-goals still in force:** no dense SOCP; do not modify `ACDC_OPF_NL_model.py` / `window_opf.py`.

---

## 1. Paper summary — SEGAN 2026: sparse SOCP + CCP for AC/DC energy hubs

### 1.1 Problem class

Mixed-integer **nonlinear** operation model (Eqs. 1–26) for hybrid AC/DC hub:

- **Objective (Eq. 1):** `max Σ_{t,i} C_{i,t} · p_exp_{i,t}` — revenue from exports to
  onshore countries.
- **Continuous vars:** AC complex voltages `v_{k,t}`, DC voltages `v^dc_{k,t}`, complex
  injections `s_{k,t}`, BESS charge/discharge, H₂ inventory, wind dispatch `p^w_{k,t}`, …
- **Binary vars:** BESS mode `y^c_{k,t}`, `y^d_{k,t}` — no simultaneous charge/discharge
  (Eqs. 22–23).

Subsystems in the MINLP:

| Block | Eqs. | Notes |
|-------|------|-------|
| AC nodal balance | (2) | Bilinear `v* Y v` |
| DC nodal balance | (3) | Bilinear DC voltage products |
| AC / DC thermal limits | (4)–(7) | Apparent / active power on lines |
| Voltage bounds | (8)–(9) | AC magnitude, DC voltage |
| VSC-HVDC losses + rating | (10)–(12) | Affine loss `a + b·ℜ{s^c}` |
| H₂ production + storage | (13)–(16) | Linear electrolyser map; mass balance |
| BESS SoC + exclusivity | (17)–(25) | MI coupling; reactive `q` on BESS |
| Wind cap | (26) | `p^w ≤ P_rated` |

### 1.2 Conventional dense SOCP (§2.2)

Lift bilinear voltage products via:

- `h_{k,t} = |v_{k,t}|²` (Eq. 27)
- `w_{km,t} = v*_{k,t} v_{m,t}` (Eq. 28) — defined for **all node pairs** in dense form
- `h^dc_{k,t}`, `w^dc_{km,t}` for DC (Eqs. 33–34)

Rotated SOC relaxation (Eq. 31 / 44–45):

```
‖ [2 w_{km}; h_k − h_m] ‖₂ ≤ h_k + h_m
```

Linearized balances (Eqs. 32, 36, 38–39). Line limits reformulated in `h`, `w` (Eqs. 40–43).
Voltage bounds on `h` (Eqs. 46–47). Remaining blocks unchanged → **MI-SOCP** (Eqs. 37–48).

**Complexity:** O(N²) auxiliary variables per period — prohibitive at PEI scale (§2.2: ~90k
scalar vars for one period at 172 nodes).

### 1.3 Sparse SOCP (§3) — key contribution

Exploit topology:

- Edge sets `E_AC = {(k,m) | branch exists, k < m}`, `E_DC` analogously.
- Hermitian symmetry (AC): `w_{mk} = w*_{km}` — optimize upper triangle only.
- Symmetry (DC): `w^dc_{mk} = w^dc_{km}`.

Sparse balance (Eq. 54) uses diagonal admittance + off-diagonal `w` or `w*` by index order.
SOC constraints only on `(k,m) ∈ E` (Eqs. 55, 57). **Mathematically equivalent** to dense
SOCP; fewer variables/constraints.

Reported gains (Table 2, simplified PEI 12 AC / 4 DC, 24 h):

| Metric | Reduction vs dense SOCP |
|--------|-------------------------|
| Variables | 79.2 % |
| Constraints | 56.2 % |
| Solve time | 86.3 % |

Accuracy vs NLP: objective gap < 0.001 % (Table 3).

### 1.4 Chance-constrained programming (§4)

Individual chance constraints at confidence `γ`:

```
Pr(g(x, ξ) ≤ 0) ≥ γ  →  g(x) ≤ Q_{1−γ}(ξ)     (Eqs. 58–60)
```

**Wind (Eqs. 61–65):** `p^w_{k,t} ≤ p^p_{w,k,t} + Q_{1−γ}(ε_w)` with truncated normal
forecast error (σ = 10 % of forecast; bounded by turbine rating).

**Price (Eqs. 66–69):** `C_{k,t} ≤ C^p_{k,t} + Q_{1−γ}(ε_c)` — conservative revenue valuation;
truncated normal (±3σ).

Combined deterministic sparse SOCP (Eqs. 70–72) + remaining physics.

Revenue impact vs deterministic: 1.2 %–17 % reduction depending on γ and scenario
(Table 4). Wind uncertainty dominates price uncertainty.

### 1.5 Paper case-study anchors (reference only — not v1 design drivers)

Paper configurations used in publication (for later optional parity checks / Mario
alignment). **Do not** hard-wire the API or builder to these cases (see L2).

| Configuration | Nodes | Paper purpose |
|---------------|-------|---------------|
| Simplified (Fig. 2) | 12 AC, 4 DC | Dense vs sparse vs NLP benchmark |
| Full PEI (Fig. 6) | 172 AC, 4 DC | Scalability + BESS + H₂ deterministic |
| CCP full PEI | 172 AC, 4 DC | γ = 0.7, 0.8, 0.9 scenarios |

Deterministic full-system revenue reference in paper: **7,112,732 €** / 24 h (§5.2).

---

## 2. What pyflow_acdc already has

### 2.1 Shipped — nonlinear operational stack (overlap with Paper A assets)

| Layer | Location | Paper A mapping | Gap |
|-------|----------|-----------------|-----|
| `Grid`, `Ybus_AC`, `Ybus_DC` | `Classes.py`, `analyse_grid` | Admittance for sparse edge construction | ✅ |
| Polar NL OPF (`V`, `θ`) | `ACDC_OPF_NL_model.py` | NLP baseline | Different formulation — not rectangular SOC lift |
| `Storage` + `storage_*` | `Classes.py`, NL model | Eqs. 17–25 | **No** charge/discharge binaries; overlap allowed (G6) |
| `Electrolyser` + `hydrogen_*` | `Classes.py`, NL model | Eqs. 13–16 | Linear H₂ map exists; daily quota constraint not verified |
| `Ren_Source` | `Classes.py` | Wind dispatch `p^w` | Curtailment via `gamma`; no CCP layer |
| `Price_Zone` / export gens | NL model | Export revenue objective | Quadratic zone costs — not Paper A linear `C·p_exp` |
| `window_nl_opf` | `window_opf.py` | Multi-period coupled NLP | IPOPT/Pyomo path; no SOCP |
| Example grids | `example_grids/` | Optional later checks | Not required for API design |

### 2.2 Not shipped — this plan's subjects

| Subject | pyflow status |
|---------|---------------|
| CVXPY SOCP / MI-SOCP model builder | **None** |
| Sparse edge-set `E_AC`, `E_DC` | **None** |
| Rectangular complex `w_{km}` SOC constraints | **None** (NL uses polar) |
| CCP quantile constraints | **None** |
| Conic solver path via CVXPY (MOSEK / others) | **None** in OPF stack |
| `ACDC_convex.py` / `convex_model.py` / `socp_optimise` | **None** |
| `[SOCP]` optional extra | **None** |

---

## 3. Architecture: `convex_model.py` + `ACDC_convex.py` (CVXPY)

**Locked (L3–L6):** implement Mario's SOCP methodologies as **new, self-contained modules**
using **CVXPY**. Do **not** modify `ACDC_OPF_NL_model.py`, TEP builders, or `window_opf.py`
internals.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  EXISTING (unchanged)              │  NEW SOCP stack                           │
│  ACDC_OPF_NL_model (polar NLP)     │  convex_model.py — CVXPY SOCP/MI-SOCP    │
│  optimal_pf / window_nl_opf        │  ACDC_convex.py — socp_optimise /          │
│  IPOPT / [OPF]=pyomo               │  soc_window_optimisation + I/O             │
│                                    │  solve via CVXPY backends ([SOCP] extra)    │
└──────────────────────────────────────────────────────────────────────────────┘
         │                                      │
         │  optional benchmark / cross-check    │
         └──────────────────────────────────────┘
              same Grid in, compare objectives & voltages vs NLP (later)
```

### 3.1 Module split

| File | Responsibility | Analogue in codebase |
|------|----------------|----------------------|
| **`convex_model.py`** | `build_socp_data` + `socp_model`: edge sets, `h`/`w`, SOC, AC/DC balance, **AC+DC thermals**, converters, gens; later BESS/H₂/CCP | `ACDC_OPF_NL_model.py` |
| **`ACDC_convex.py`** | `translate_pyf_socp`; public `socp_optimise` + `soc_window_optimisation`; weighted objective; solve; export | `ACDC_OPF.py` (`optimal_pf`, `window_nl_opf`, …) |

Rationale for **two files**:

1. Even without BESS/H₂, AC + DC + converter + thermals + gens warrants **build** vs **run** split.
2. Mirrors existing `ACDC_OPF.py` / `ACDC_OPF_NL_model.py` split.
3. Keeps CVXPY optional: `ACDC_convex` / `__init__` can fail cleanly when `[SOCP]` is missing.
4. **One** `socp_model` with `(…, T)` indexing (L21); runners only differ in how `frame_ids` / `T` are prepared.

### 3.2 Public API

| Item | Locked / status |
|------|-----------------|
| Entry points | **`socp_optimise`** (single / `T=1`) + **`soc_window_optimisation`** (multiperiod) (L22) |
| Translate | **`translate_pyf_socp`** — dedicated packer (Q-7 locked) |
| Modes | Deterministic sparse SOCP first; CCP variants later — exact kwargs TBD |
| Case binding | **None** — any analysed `Grid` with required assets/params |
| Builder | Internal **`socp_model(grid, d)`** only — prepared data object (L23) |

### 3.3 Optional dependency `[SOCP]`

Mirror `[OPF] = ["pyomo"]` in `pyproject.toml`:

```toml
SOCP = ["cvxpy"]   # solvers TBD (e.g. mosek); may also document [Gurobi]
```

- Guarded import in `__init__.py`: export `socp_optimise`, `soc_window_optimisation`,
  `translate_pyf_socp` when CVXPY is available (**landed**).
- Folded into `[All]` (**landed** as `cvxpy` alone).
- Solvers packaged inside `[SOCP]` vs documented separately: **open** (U-S1).

### 3.4 What stays outside the SOCP build

| Module | Policy |
|--------|--------|
| `ACDC_OPF_NL_model.py` | Unchanged — NLP remains default “exact physics” path |
| `ACDC_Static_TEP.py`, `ACDC_MultiPeriod_TEP.py` | Unchanged |
| `bess_sizing.py` | Sibling plan — no overlap |
| `window_opf.py` | Optional downstream benchmark only |
| `pyomo_model_solve.py` | **Not** used by this stack (CVXPY solve path) |

### 3.5 Rationale

1. SOCP is a **different mathematical object** from polar NLP — own builder, not a branch
   inside `ACDC_OPF_NL_model.py`.
2. Paper A reference stack is **CVXPY+MOSEK** — matching that avoids a double port
   (paper → Pyomo).
3. Conic solvers are not the IPOPT-centric `[OPF]` default — `[SOCP]` isolates optional
   dependency and licensing.
4. Sparse formulation is an **implementation strategy** inside `convex_model.py`, not a
   change to `Grid` topology storage.
5. Case-agnostic API (L2) keeps the stack reusable beyond PEI.

---

## 4. pyflow_acdc mapping: Classes → Model → Problem → Solution

### 4.1 Classes (`Classes.py`, `grid_modifications.py`, `constants.py`)

**Role:** parameters, topology, and post-solve results. SOCP **decision structure** lives in
CVXPY variables, not on element classes, until owner decides otherwise.

| Paper entity | Existing class | Gap / owner decision |
|--------------|----------------|----------------------|
| Wind `p^w_{k,t}` | `Ren_Source` + nodal `PGi_ren` in NL | CCP needs forecast `p^p`, error distribution params |
| BESS `p^c`, `p^d`, `y^c`, `y^d`, SoC | `Storage` | **Missing:** binary exclusivity; paper uses MI |
| BESS reactive `q`, ‖s^b‖ limit | `Storage.Q`, `S_max` on AC | Partially aligned |
| Electrolyser `p^e`, H₂ mass `M` | `Electrolyser`, `mass_H2` | Check min power (Eq. 15), daily quota (§5.2.4) |
| Export `p^exp` to countries | `Price_Zone` + ext grid gens | Paper uses **linear** revenue; pyflow often quadratic |
| HVDC losses `a_k`, `b_k` | `AC_DC_converter` | Verify affine loss model match (Eq. 10) |
| `ρ` (mono/bipolar DC) | `Polarity` on `Line_DC` | Map to paper `ρ` factor |
| Price / wind uncertainty | — | **Missing:** `γ`, quantile inputs, truncated-normal params |
| Edge sets `E_AC`, `E_DC` | Built from `Ybus` sparsity pattern | Runtime derived — no new class required |

**Proposed class-layer actions (owner review)**

| ID | Action | Notes |
|----|--------|-------|
| C3 | Uncertainty / CCP config (object or run kwargs) | Keeps `Grid` free of stale flags |
| C4 | **Do not** replace `Storage` with MI logic globally | Binaries only in SOCP BESS block |
| C5 | `grid.socp_run` flag (or sibling) | Results routing, like `window_opf_run` / `opf_run` |

### 4.2 Model (`convex_model.py`)

**Role:** CVXPY variables, SOC constraints, subsystem blocks. **Not** in NL or L builders.

#### 4.2.1 SOC kernel

```
convex_model.py
├── build_edge_sets(grid) → E_AC, E_DC from sparsity (k < m)
├── soc_lift_ac(...)     → h_k,t, w_km,t complex (Mario), SOC (55)
├── soc_lift_dc(...)     → h^dc, w^dc real, SOC (57)
├── linear_ac_balance(...)   → Eq. (54); conj(S)==flow (L11); bus roles (L10)
├── linear_dc_balance(...)   → Eq. (56); pol=pcn (L14)
├── thermal_limits_ac(...)   → Eqs. (40)–(43) **required** (L15)
├── thermal_limits_dc(...)   → DC rating limits (Mario + paper)
├── converter_acdc(...)      → Re(Ss)+Pdc+Ploss=0; Ploss=a+c·… (L13); ‖Ss‖≤Smax
└── voltage_bounds_on_h(...) → Eqs. (46)–(47)
```

| Aspect | NL model (`ACDC_OPF_NL_model`) | SOCP model (`convex_model`) |
|--------|----------------------------------|-----------------------------|
| Language | Pyomo | **CVXPY** |
| Voltage representation | `V_AC`, `theta_AC` (polar) | `h_k`, sparse complex `w_km` |
| AC balance | `cos/sin` of angle differences | Linear in `w`, `h` with Ybus; `conj(S)` |
| DC balance | Polar / simplified | Lift → linear in `w^dc`, `h^dc`; `pol=pcn` |
| Line limits | Nonlinear branch flow | **AC + DC** SOCP-reformulated (L15) |
| Converter AC branches | Explicit in NL model | **Ignored in v1** (L8 option B) |
| Relaxation gap | — | SOC inequality ≤ tightness vs NLP |

#### 4.2.2 Paper A extensions

| Block | Eqs. | v1 status |
|-------|------|-----------|
| VSC-HVDC interface | (10)–(12) | **In** — power+loss only; no conv AC Ybus (L8/L13) |
| **AC thermal limits** | (40)–(43) | **In** — mandatory (L15); missing in Mario script |
| DC thermal limits | Mario + paper | **In** |
| Export revenue objective | (37) | Available via `Ext_Gen` / L16; **default runner uses `Energy_cost` (L27)** |
| Wind / renewables | (26) + gamma | **In** — `gamma` param, start at 1 (L12); TS from `grid.Time_series` (L26) |
| AC gens / ext-grids | — | **In** — decision vars `PGi_gen`/`QGi_gen` (L24) |
| BESS MI | (17)–(25) | **Deferred** (L18); `T` indexing kept for later SoC |
| H₂ subsystem | (13)–(16) | **Deferred** |
| **CCP layer** | (70)–(72) | **Deferred** |

**Sparse only — GREEN LIGHT (L19).** No dense SOCP mode in the product plan.

#### 4.2.3 CVXPY / solver note

Paper A uses **CVXPY+MOSEK**. pyflow NLP OPF remains Pyomo. This stack solves through
CVXPY’s solver interface (`problem.solve(solver=…)`). Target solver package list inside
`[SOCP]` is **open** (U-S1).

### 4.3 Problem (`ACDC_convex.py`)

**Role:** assemble full problem from any `Grid` + time series + uncertainty options; call
CVXPY solve; export.

| Runner concern | Direction |
|----------------|-----------|
| Horizon | `T` + ordered `frame_ids` from `translate_pyf_socp` (L21/L23) |
| Input TS | From **`grid.Time_series`** only (L26): renewables, prices; loads TBD |
| Mode switch | Deterministic first; CCP later |
| Objective | Weighted `ObjComponent` (L27); default `Energy_cost`; Mario L16 via `Ext_Gen` |
| Output | `grid.socp_results`; element updates via `_export_to_grid` |

**Relationship to `window_nl_opf`:**

| Feature | `window_nl_opf` | `soc_window_optimisation` (v1) |
|---------|-----------------|--------------------------------|
| Coupling | Multi-hour SoC / H₂ links | Multi-period network (`T`); no BESS/H₂ yet |
| Physics | Polar NLP | Sparse SOCP + AC/DC thermals |
| BESS exclusivity | No (G6) | N/A until L18 lifted |
| Uncertainty | None | None in v1 |
| Solver | IPOPT (Pyomo) | CVXPY + conic solver |
| Single-period sibling | `optimal_pf` | `socp_optimise` (`T=1`) |

**Landed:** both runners share one `socp_model` (not a thin wrapper around independent per-frame solves).

### 4.4 Solution (`Results_class.py`, export, tests)

| Output | Paper reference | pyflow destination (TBD) |
|--------|-----------------|--------------------------|
| Total revenue | Tables 3–4 | `Results` summary method |
| Nodal `h_k` → voltage magnitude | Table 3 voltage check | `sqrt(h_k)` vs `V_AC` from NLP |
| Export powers | Fig. 8 | Time series export |
| BESS SoC / power | Fig. 9 | Storage results tables |
| H₂ production | Fig. 10 | Electrolyser results |
| CCP revenue sensitivity | Table 4 | Scenario comparison table |
| Solve stats (vars, constraints, time) | Table 2 | CVXPY / solver stats helper |

**Validation strategy:** deferred per L2. Prefer generic smoke tests (build + optional solve
on any small Grid) before paper-table reproduction. When Mario’s code/logs arrive, add
optional parity checks.

---

## 5. Subject-focused breakdown

### 5.1 Sparse SOCP relaxation (Paper A §3)

| Layer | What to implement | Where |
|-------|-------------------|-------|
| **Classes** | Nothing mandatory — `Ybus` sparsity from existing `analyse_grid` | `grid_analysis.py` read-only |
| **Model** | `E_AC`, `E_DC`; upper-triangular `w`; Hermitian conjugate in balance (Eq. 54); SOC on edges only (Eq. 55) | `convex_model.py` |
| **Problem** | Sparse edge sets only (L19); no `formulation='dense'` | `ACDC_convex.py` |
| **Solution** | Report variable/constraint counts; wall time | solver stats helper |

### 5.2 AC / DC thermal limits (Paper A Eqs. 40–43) — **v1 required**

| Layer | What to implement | Where |
|-------|-------------------|-------|
| **Classes** | Line ratings from existing `Line_AC` / `Line_DC` | Grid |
| **Model** | SOCP reformulation of AC apparent/active limits; DC rating as in Mario + paper | `convex_model.py` |
| **Problem** | Always on unless explicitly disabled (fail-hard default: on) | `ACDC_convex.py` |
| **Note** | Mario script has DC limits only — pyflow **must** add AC (L15) |

### 5.3 Converter AC↔DC (v1)

| Layer | What to implement | Where |
|-------|-------------------|-------|
| **Model** | `Re(Ss)+Pdc+Ploss=0`; `Ploss=a+c·…` (L13); `‖Ss‖≤Smax`; no conv AC Ybus (L8) | `convex_model.py` |
| **Classes** | `a_conv`, `c_rect` (or paper `a,c`), polarity `pcn` | Existing converter attrs |

### 5.4 MI-SOCP BESS (Paper A Eqs. 20–23) — **deferred (L18)**

| Layer | What to implement | Where |
|-------|-------------------|-------|
| **Model** | Binary `y^c`, `y^d`; SoC — **not in v1** | later `convex_model.py` |

### 5.5 Chance-constrained wind & price (Paper A §4) — **deferred**

| Layer | What to implement | Where |
|-------|-------------------|-------|
| **Model** | Deterministic equivalents (71)–(72) — **not in v1** | later |

### 5.6 Hydrogen subsystem (Paper A §2.1) — **deferred**

| Layer | What to implement | Where |
|-------|-------------------|-------|
| **Model** | Mass balance / quota — **not in v1** | later |

---

## 6. Phased roadmap

| Phase | Goal | Status (2026-07-29) | Exit criterion |
|-------|------|---------------------|----------------|
| **0** | `[SOCP]` extra; skeleton `convex_model` / `ACDC_convex` | **Done** | Import-guarded; L1–L28 reflected |
| **1** | Sparse SOC + AC balance + **AC thermals** + gens | **Scaffolded** (unvalidated) | Feasible / builds / solves small AC |
| **2** | DC + converters (L8/L13/L14) + DC thermals + `T` | **Scaffolded** (unvalidated) | AC/DC hub builds case-agnostically |
| **3** | `translate_pyf_socp` + ObjComponent + runners (L22–L27) | **Scaffolded** (unvalidated) | Single + window paths call same model |
| **4** | Smoke solve + scaling / solver defaults | **Next** | Completes within owner budget |
| **5** | CCP (optional later) | Deferred | Wind / price modes |
| **6** | BESS MI / H₂ (optional later; lifts L18) | Deferred (`T` ready) | Coupled assets |
| **7** | Docs, `Results`, CI | Deferred | `[SOCP]` documented; smoke tests |

Prefer Mario’s script (L7) for SOC / balance / converter patterns; **add AC limits from the paper** (L15) even though his script lacks them.

**Resume checklist (next session)**

1. Confirm `cvxpy` (+ solver) installed; `from pyflow_acdc import socp_optimise`.
2. Smoke: small Grid → `socp_optimise(..., build_only=True)` then solve.
3. Compare objective / voltages vs NLP on a known case (optional).
4. Tighten Paper A AC thermal expressions (U-A8) if needed.
5. Only then: BESS SoC across `t` / CCP.

---

## 7. Queries before / during implementation

Owner-locked items (**L1–L28**) are answered. Remaining opens below.

### 7.1 Architecture and API

| Q-ID | Question | Status |
|------|----------|--------|
| Q-1 | Paper A only? | **Locked L1** |
| Q-5 | Two-file split? | **Locked L4** |
| Q-6 | Public entry `socp_optimise`? | **Locked L5** (+ window L22) |
| Q-7 | Dedicated `translate_pyf_socp` vs fork `translate_pyf_opf`? | **Locked** — dedicated `translate_pyf_socp` (L23/L26) |
| Q-8 | Multi-period: standalone vs `window_*`? | **Locked L21/L22** — one `socp_model(…, T)`; `socp_optimise` + `soc_window_optimisation` |
| Q-9 | `w_{km}` complex vs Re/Im? | **Locked** — complex dict like Mario |

### 7.2 Solver and dependencies

| Q-ID | Question | Status |
|------|----------|--------|
| Q-10 | Solvers inside `[SOCP]` (MOSEK vs docs-only)? | **Open** |
| Q-11 | CI without commercial license? | **Open** |
| Q-12 | MIP gap (BESS) | **N/A until L18 lifted** |
| Q-13 | CVXPY solve path (not Pyomo) | **Locked L3** |

### 7.3 Formulation (mostly locked from Mario Q&A)

| Q-ID | Question | Status |
|------|----------|--------|
| Q-14 | Linear priced exports vs `Price_Zone` quadratic? | **L16 paper form**; default objective **L27 `Energy_cost`** |
| Q-15 | BESS MI exclusivity? | **Deferred L18** |
| Q-16 | H₂ daily quota? | **Deferred** |
| Q-17 | CCP quantiles? | **Deferred** |
| Q-19 | Paper parity tolerances | **Open** / deferred with L2 |

### 7.4 Data and reference

| Q-ID | Question | Status |
|------|----------|--------|
| Q-20 | Case-specific PEI fixtures | **Deferred L2** |
| Q-21 / Q-22 | Mario script | **Have** scop script (L7); still useful: `wpp_forecast.csv`, solve logs |
| Q-24 | CCP γ tests | **Deferred** |

### 7.5 Governance / product

| Q-ID | Question | Status |
|------|----------|--------|
| Q-2 | Complement NLP (not replace)? | **Open** (default: complement) |
| Q-25 | Sign-off on port | Mario / CITCEA |
| Q-26 | `Results` section name | **Open** |

### 7.6 Minimum to start Phase 0 / 1

**Scaffold done under L1–L28.** Next gate: Phase 4 smoke solve (cvxpy + solver).

**Still useful:** exact Paper A AC thermal expressions (40–43) mapped to pyflow rating fields (U-A8); `wpp_forecast` for smoke runs.

---

## 8. Uncertainties (Paper A / CVXPY only)

### 8.1 Architecture and scope

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-A3 | **Horizon wrapper** | **Resolved L21/L22** — one `socp_model` with `T`; `socp_optimise` + `soc_window_optimisation` |
| U-A4 | **Translate packing** | **Resolved** — dedicated `translate_pyf_socp` (L23/L26); does not fork `translate_pyf_opf` |
| U-A5 | **`w_{km}` representation** | **Locked** — complex (Mario) |
| U-A6 | **Export / gen objective** | **L16** paper revenue form available; **default = L27 `Energy_cost`** |
| U-A7 | **Licensing** — which solvers ship in `[SOCP]` vs docs-only | CI and install story |
| U-A8 | **AC thermal form** — exact Paper A (40–43) vs pyflow rating fields mapping | L15 requires AC limits; expression detail open |

### 8.2 Mathematical formulation

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-M1 | **SOC relaxation tightness** — gap vs polar NLP | When to warn users |
| U-M2 | **Paper A Eq. (3) DC balance** vs pyflow `Ybus_DC` | Sign of `ρ`, grounding |
| U-M3 | **Converter embedding** | **Locked L8** — v1 option B (power+loss only; no conv AC Ybus) |
| U-M4 | **Line limits (40–43)** — map paper form → pyflow `Line_AC` ratings | **Required** (L15); expression detail = U-A8 |
| U-M5 | **BESS Eq. (17) time units** — paper minute-scale vs `Storage.dt_hours` | SoC scaling |
| U-M6 | **H₂ daily quota** — hard vs soft / optional | §5.2.4 policy |
| U-M7 | **CCP truncated-normal quantiles** — analytic vs lookup | Precompute per `(k,t)` |
| U-M8 | **Spatial wind correlation** — node-wise CCP ignores correlation (Remark 1) | Document conservative bias? |

### 8.3 Units and grid mapping

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-U1 | **Physical vs pu** | **Locked L20** — model in grid pu |
| U-U2 | **`h_k` / `h^dc`** — pu² vs kV² | Eq. (47) bounds |
| U-U5 | **Export link modeling** | **Locked L16** — ext-grid negative injections |

### 8.4 Data and reproducibility

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-D3 | **CCP forecast errors** — σ = 10 %; truncation | Quantitative CCP |
| U-D5 | **Acceptance tolerances** if paper parity is pursued later | Optional exit gates |
| U-D6 | **Public CI without MOSEK** | Test design |
| U-G2 / U-G4 | **Reference logs / CVXPY code** from Mario | Avoid re-derivation (L7) |

### 8.5 Solver, numerics, and performance

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-S1 | **Primary conic MIP solver** for `[SOCP]` | Extra contents |
| U-S2 | **MI-SOCP scale** (large N × T) | Memory / time |
| U-S4 | **MIP gap** for BESS binaries | Optimality vs speed |
| U-S5 | **Warm-start from NLP** (`h ← V²`) | Heuristic only |

### 8.6 Class model and integration

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-C2 | **`grid.socp_run` vs reuse `opf_run`** | `Results.all()` routing |
| U-C3 | **Coexistence with G6** in NL — SOCP path stricter | User docs |
| U-C4 | **Dash / plotting** — new SOCP views or reuse NL | Later scope |
| U-G1 | **Formulation sign-off** — Mario on CVXPY port | Before claiming paper parity |

### 8.7 Suggested resolution order

1. ~~Scaffold Phase 0 with L1–L20 (`[SOCP]` may start as `["cvxpy"]` alone).~~ **Done.**
2. ~~Phase 1–3 scaffolding: sparse SOC + balance + converters + thermals + runners.~~ **In tree; unvalidated.**
3. Choose solvers for `[SOCP]` (U-S1, U-D6); smoke solve.
4. Later: CCP, BESS, H₂, option A `CONVEX_Ybus`.

---

## 9. File touch list

| File | Action / status |
|------|-----------------|
| **`pyflow_acdc/convex_model.py`** | **Landed** — `build_socp_data`, `socp_model`, subsystem vars/constraints |
| **`pyflow_acdc/ACDC_convex.py`** | **Landed** — `translate_pyf_socp`, `socp_optimise`, `soc_window_optimisation`, objective, export |
| `__init__.py` | **Landed** — guarded export of both runners + translate |
| `pyproject.toml` | **Landed** — `SOCP = ["cvxpy"]`; folded into `All` |
| `Classes.py` | Optional: CCP / run flags only as needed |
| `constants.py` | Optional: SOCP / CCP mode enums |
| `solver_utils.py` | Optional: CVXPY / conic capability probe |
| `Results_class.py` | Optional: `socp` report methods (`grid.socp_results` stub exists) |
| `docs/usage_socp.rst`, `docs/citing.rst` | User guide + citation |
| `pyflow_tests/...` | **Next** — generic smoke / build_only tests |

**Do not modify (v1)**

| File | Reason |
|------|--------|
| `ACDC_OPF_NL_model.py` | NLP stack remains canonical operational path |
| `ACDC_Static_TEP.py`, `ACDC_MultiPeriod_TEP.py` | Unrelated planning class |
| `window_opf.py` | Optional benchmark peer only |
| `pyomo_model_solve.py` | Wrong stack for CVXPY |
| `bess_sizing.py` (planned) | Sibling sizing problem |

---

## 10. References

### Primary

M. Useche-Arteaga et al., *Efficient AC/DC energy hubs operation using sparse SOCP
relaxation and chance-constrained optimization*, Sustainable Energy, Grids and Networks
**46**, 102217 (2026). https://doi.org/10.1016/j.segan.2026.102217

### Related pyflow / predecessor NLP paper

M. Useche-Arteaga et al., *Optimizing the operation of energy islands with predictive
nonlinear programming* (PEI case study), Wind Energy Sci. **11**, 349–372 (2026).
Implemented in [bess_integration_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_integration_plan.md) (`window_nl_opf`).

### Convex optimization background

A. Garces, *Mathematical Programming for Power Systems Operation* (Wiley, 2022) — SOC
lift background.

S. H. Low, Convex relaxation of optimal power flow, IEEE Trans. Control Netw. Syst. **1**(1)
(2014).

### pyflow_acdc siblings

[bess_integration_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_integration_plan.md) — operational NLP BESS + H₂.

[bess_sizing_ramp_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_sizing_ramp_plan.md) — plant-level BESS sizing (separate build).
