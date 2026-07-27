# BESS sizing and ramp-rate plan for pyflow_acdc

Living implementation plan for **optimal BESS sizing and operation under ramp-rate
limits and degradation**, based on the publication below.

This plan is **publication-grounded only**. It does **not** prescribe owner
implementation details, solver choices, or API names beyond what the paper and
existing pyflow_acdc architecture require. Owner code and business logic should
be waited for before coding.

**Primary reference**

M. Montalà Palau, M. Cheah Mañé, and O. Gomis-Bellmunt, *Techno-economic
optimization for BESS sizing and operation considering degradation and ramp rate
limit requirement*, J. Energy Storage **105**, 114631 (2025),
https://doi.org/10.1016/j.est.2024.114631.

**Related pyflow_acdc assets**

| Document | Path |
|----------|------|
| BESS operation (already shipped) | [plans/bess_integration_plan.md](bess_integration_plan.md) |
| Architecture map | [docs/architecture.md](../docs/architecture.md) |
| Storage user guide | [docs/usage_storage.rst](../docs/usage_storage.rst) |

---

## 0. Scope of this plan

| In scope (paper) | Out of scope (this plan) |
|------------------|--------------------------|
| Endogenous **BESS power** `P_BESS` and **capacity** `E_BESS` | Full AC/DC network expansion (lines, converters, `np_gen`) |
| **Ramp-rate limit** on grid export `P_grid[t]` | Mario / PEI energy-island coupled OPF (see `bess_integration_plan.md`) |
| **Cycling + calendar degradation** → lifetime → replacements | Myopic `ts_acdc_opf` operation (Phase 8, already shipped) |
| Representative high-resolution RES profile (1-min, reduced week) | Rainflow counting **inside** the optimizer (paper: post-processing only) |
| Technology / market / curtailment scenario switches | Implementing owner equations before owner review |

**Distinction from `bess_integration_plan.md`:** that plan covers **fixed-size BESS
operation** inside nonlinear AC/DC OPF (`window_nl_opf`, `ts_acdc_opf`). This plan
covers **sizing** in a new standalone module **`bess_sizing.py`** — `E_BESS` and
`P_BESS` are decision variables, with economics and degradation driving the optimum.
No changes to static TEP, MP TEP, or existing OPF builders.

---

## 1. Paper summary (what the methodology actually is)

### 1.1 Problem class

Single **renewable power plant + optional BESS** at the **point of interconnection
(POI)**. The renewable plant (`P_AC-n`, fixed) and retributive scheme are inputs.
The optimizer chooses:

- installed BESS **power** `P_BESS` [MW]
- installed BESS **energy** `E_BESS` [MWh]
- hourly/minute **operation**: charge, discharge, grid export, optional curtailment
- derived states: SoC, degradation, BESS lifetime, number of replacements

**Objective (Eq. 4):** maximize annual **profit**

```
Profit = Income − CCAP_B − COP_B − CCAP_RE − COP_RE
```

Renewable CAPEX/OPEX (`CCAP_RE`, `COP_RE`) are **fixed parameters** (not optimized).
BESS CAPEX/OPEX depend on the chosen size, replacements, and operation.

### 1.2 Time resolution and horizon

| Item | Paper value | Note |
|------|-------------|------|
| Native timestep | **1 minute** | Ramp rate is defined per minute |
| Full year | 525 600 minutes | Preprocessed for tractability |
| Case study horizon | **10 080 minutes** (1 representative week) | Critical-day selection per weekday |
| Terminal SoC | `SOC_T = SOC_ini` (Eq. 19) | Cyclic boundary |

### 1.3 Control variables **x** (Eq. 3.1)

| Symbol | Meaning |
|--------|---------|
| `E_BESS` | BESS energy capacity [MWh] |
| `P_BESS` | BESS power rating [MW] |
| `P_bat-ch[t]`, `P_bat-dch[t]` | Charge / discharge power each period |
| `P_grid[t]` | Active power exported to grid |

### 1.4 State / auxiliary variables **u** (Eq. 3.1)

Income `I`, annualized BESS CAPEX `CCAP_B`, BESS OPEX `COP_B`, unit count `n`,
BESS lifetime `L_BESS`, replacements `N_r`, annual degradation `E_deg`, curtailment
`P_cur[t]`, net battery flux `P_bat[t]`, SoC `SOC[t]`.

### 1.5 Ramp-rate constraint (Eq. 10) — subject ①

For `t ≥ 2`:

```
P_grid[t−1] − P_AC-n · δ_rrl ≤ P_grid[t] ≤ P_grid[t−1] + P_AC-n · δ_rrl
```

- `δ_rrl`: ramp-rate limit (case study: **5 %/min** of nameplate)
- `P_grid[1]` is **not** ramp-limited (only upper bound by `P_AC-n`, Eq. 11)
- This is a constraint on **POI export**, not on line flows or converter ramps

**Grid balance (Eq. 16):**

```
P_grid[t] = P_RE[t] + P_bat[t] + P_cur[t] · α_cur
```

`P_RE[t]` is exogenous RES potential; `α_cur = 0` disables curtailment.

### 1.6 BESS sizing constraints — subject ②

| Eq. | Constraint | Role |
|-----|------------|------|
| (12) | `−P_BESS ≤ P_bat[t] ≤ P_BESS` | Power rating |
| (13) | `SOC_min·E_BESS ≤ SOC[t] ≤ SOC_max·E_BESS` | Safe SoC band |
| (14) | `E_BESS = P_BESS · d_rate` | Optional: fixed discharge duration (commercial unit) |
| (15) | `P_BESS = n · P_B-u` | Optional: discrete commercial blocks |

Eqs. (14)–(15) can be **omitted** for free continuous P/E sizing (paper §3.4).

### 1.7 BESS operation (Eqs. 17–19)

```
P_bat[t] = P_bat-ch[t]·η_ch − P_bat-dch[t]/η_dch        (17)
SOC[1]   = SOC_ini·E_BESS + P_bat[1]/60                  (18, t=1)
SOC[t]   = SOC[t−1]·(1−τ_bat) + P_bat[t]/60   t≥2       (18)
SOC[T]   = SOC_ini·E_BESS                                (19)
```

Self-discharge `τ_bat` is a parameter. Timestep is **minutes** (`/60` converts
power to MWh increment).

### 1.8 Degradation model (§2.3) — sizing coupling

**Cycling** (Eqs. 1–2): Full Equivalent Cycles from energy throughput; annual
cycling degradation `E_cyc` scales with `N_FEC` and technology `N_FCTF`.

**Calendar** (Eq. 3): linear in average SoC: `E_cal = (A_cal·SOC_avg + B_cal)·E_BESS·365`.

**Total:** `E_deg = E_cyc + E_cal` (Fig. 2).

**Lifetime** (Eq. 21): `L_BESS = E_BESS·EEOL / (E_deg·α_deg + L_pr·(1−α_deg))`.

**Replacements** (Eq. 20): `N_r` from project lifetime `L_pr` and `L_BESS`, with
optional residual-value mode `α_RV`.

Toggle `α_deg = 0` skips degradation (faster, conservative on size — paper shows
**7.6×** oversize without degradation under pool prices).

**Rainflow** (ASTM E1049): validation / post-processing only — **not** in the NLP.

### 1.9 Economics (Eqs. 5–9)

| Term | Paper form |
|------|------------|
| `CCAP_B` | `(P_BESS·CC_B_p + E_BESS·CC_B_e)·N_r / L_pr` |
| `COP_B` | Throughput-based variable OPEX + fixed `E_BESS·CO_B_f` |
| `I` | `Σ_t P_grid[t]·C_t`, annualized from horizon length |

Technology tables (LFP vs LTO), market schemes (pool / VPPA), curtailment, and
residual-value flags are **scenario parameters** (Tables 4–5).

### 1.10 Key paper conclusions (inform validation, not design)

1. Degradation is **essential** for sizing — omission can oversize BESS dramatically.
2. Ramp-rate service is **power-dominated** (short discharge duration ≈ 3.25 min
   without degradation; LTO favourable when energy is under-used).
3. Curtailment as complementary strategy can **shrink** BESS but reduces operational
   flexibility.
4. Methodology is for the **development / sizing stage**; operation in the field
   needs EMS + forecasting (paper §6 limitation).

---

## 2. What pyflow_acdc already has

### 2.1 Shipped — operation layer (`bess_integration_plan.md`)

| Layer | Status | Relevance to this plan |
|-------|--------|------------------------|
| `Storage` class (`Classes.py`) | ✅ Fixed `E_max`, `P_*_max` | Parameter carrier; **not** sizing vars |
| `add_storage()` | ✅ | Sets fixed size before OPF |
| `ACDC_OPF_NL_model.py` `storage_*` | ✅ | Per-frame charge/discharge/SoC in **full AC/DC OPF** |
| `window_nl_opf` | ✅ | Coupled **hourly** horizon, hard SoC ini/final |
| `ts_acdc_opf` | ✅ | Myopic hourly, soft `soc_ref` |
| G11 in bess plan | ✅ Locked | **Operation only — no BESS CAPEX variables** |

### 2.2 Adjacent modules (not modified by this plan)

| Module | Role | Touch in this plan |
|--------|------|-------------------|
| `window_opf.py` | Fixed-size BESS on full AC/DC grid (hourly) | **Read-only** — optional post-sizing validation |
| `Time_series.py` | TS drivers | **Read-only** — may supply `P_RE` / price columns |
| `pyomo_model_solve.py` | Generic solve layer | **Reuse** — call from `bess_sizing.py` |

**Explicit non-touch:** `ACDC_Static_TEP.py`, `ACDC_MultiPeriod_TEP.py`, `TEP_variables`,
`investment_decisions`, and the `storage_*` block in `ACDC_OPF_NL_model.py` are **out of
scope**. Sizing is a separate build, not an extension of static or multi-period TEP.

### 2.3 Not shipped — this plan's subjects

- Endogenous `E_BESS` / `P_BESS` optimization
- POI ramp-rate limits on exported power
- Degradation-linked BESS lifetime and replacement cost in the **objective**
- Minute-resolution representative operational horizon for **sizing**
- Aggregated POI / plant-level balance (paper bypasses network physics)

---

## 3. Architecture decision: standalone `bess_sizing.py`

**Locked:** implement the Montalà et al. methodology as a **new, self-contained module**
`pyflow_acdc/bess_sizing.py`. Do **not** modify static TEP, MP TEP, or the existing NL
OPF / `window_opf` builders.

```
┌─────────────────────────────────────────────────────────────────────────┐
│  EXISTING (unchanged)              │  NEW: bess_sizing.py               │
│  window_nl_opf / ts_acdc_opf       │  Plant-level POI sizing NLP        │
│  ACDC_OPF_NL_model storage_*       │  E_BESS, P_BESS = decision vars  │
│  Fixed E_max, P_max from Storage   │  Ramp on P_grid[t] (Eq. 10)        │
│  Full AC/DC physics                │  Degradation economics (Eqs. 1–3)  │
│  Hourly operation                  │  Minute (or owner) timestep        │
└─────────────────────────────────────────────────────────────────────────┘
         │                                      │
         │  optimal E_max, P_nom (export)       │
         └──────────────────────────────────────┘
                    add_storage(grid, ...)  →  window_nl_opf (optional check)
```

### 3.1 What lives in `bess_sizing.py`

Single file (same pattern as `window_opf.py` owning its horizon assembly):

| Responsibility | Paper anchor |
|----------------|--------------|
| Input validation + scenario flags (`α_deg`, `α_cur`, `α_RV`, …) | Table 5 |
| Pyomo model: vars, Eqs. (4)–(21) | §3 |
| Solve via `pyomo_model_solve` | — |
| Export results (DataFrame / dict / optional `grid` write-back) | Table 6 |
| Optional post-solve rainflow on `SOC[t]` | Table 7 |

Public entry point (name TBD within file, e.g. `bess_sizing(...)`): one callable that
builds, solves, and returns sizing results.

### 3.2 What stays outside `bess_sizing.py`

| Layer | Policy |
|-------|--------|
| `Classes.py` | Minimal optional attrs for technology / plant config — only if needed for export |
| `add_storage()` | Called **after** sizing with optimal sizes; not part of the NLP |
| `window_nl_opf` | Optional **downstream** validation on a real grid — not part of v1 build |
| TEP / MP TEP | **No changes** |

### 3.3 Rationale

1. Paper problem = **plant commissioning sizing**, not grid expansion planning.
2. `bess_integration_plan.md` G11: operational OPF stays CAPEX-free.
3. Minute-resolution ramp + degradation economics do not fit TEP snapshot / hourly MS structure.
4. Isolated file → no regression risk on `ACDC_MultiPeriod_TEP.py` or NL storage blocks.
5. Clear hand-off: `bess_sizing()` → `add_storage()` → existing operation stack.

---

## 4. pyflow_acdc mapping: Classes → Model → Problem → Solution

### 4.1 Classes (`Classes.py`, `grid_modifications.py`, `constants.py`)

**Role:** hold **parameters** and **post-solve results**; do **not** assume
sizing variables live on the element class until owner decides.

| Paper parameter | Existing `Storage` attr | Gap / owner decision |
|-----------------|-------------------------|----------------------|
| `E_BESS` | `E_max` [MWh] | Today **fixed** input; sizing run should **write** optimum back |
| `P_BESS` | `P_charge_max`, `P_discharge_max`, `S_max`/`P_max` | Same |
| `η_ch`, `η_dch` | `eta_charge`, `eta_discharge` | ✅ |
| `SOC_min`, `SOC_max`, `SOC_ini` | `soc_min`, `soc_max`, `soc_initial` | ✅ |
| `τ_bat` | — | **Missing** — self-discharge |
| `N_FCTF`, `EEOL` | — | **Missing** — degradation technology |
| `A_cal`, `B_cal` | — | **Missing** — calendar degradation (Table 3: LFP/LMO/NMC/LTO) |
| `d_rate`, `P_B-u`, `n` | — | **Missing** — commercial unit coupling (Eqs. 14–15) |
| `δ_rrl` | — | **Missing** — likely on `Ren_Source` or plant config, not `Storage` |
| `P_AC-n` | `Ren_Source` / site nominal | Map POI nameplate |
| `α_cur`, `α_deg`, `α_RV` | — | Scenario flags (constants or run config) |
| `CC_B_p`, `CC_B_e`, `CO_B_v`, `CO_B_f` | — | **Missing** — BESS economics |
| `CC_RE`, `CO_RE`, `L_pr` | — | Plant-level, not on `Storage` |

**Proposed class-layer actions (for owner review)**

| ID | Action | Notes |
|----|--------|-------|
| C1 | **Technology profile** object or `Storage` optional attrs | LFP/LTO chemistry params from paper Table 3–4 |
| C2 | **Plant / POI config** on `Grid` or dedicated case object | `P_AC_n`, `delta_rrl`, `alpha_cur`, retributive scheme |
| C3 | **Do not** add `investment_decisions` to `Storage` | Sizing vars stay in `bess_sizing.py`, not TEP |
| C4 | **Sizing result record** | e.g. `grid.bess_sizing_run`, optimum `E_max`/`P_nom`/`n_rep`/`L_BESS` |
| C5 | `add_storage` unchanged as operational API | Sizing runner calls it **after** solve with optimal sizes |

### 4.2 Model (inside `bess_sizing.py` only)

**Role:** Pyomo variables, constraints, objective **expression** — all in
`bess_sizing.py`. **Not** in `ACDC_OPF_NL_model.py` or any TEP builder.

The paper model is **not** a specialization of `opf_create_nl_model_acdc`:

| Aspect | Current NL storage block | Paper model |
|--------|--------------------------|-------------|
| Network | AC/DC buses, lines, converters | Single POI |
| `P_grid` | Emerges from nodal balance | **Explicit variable** with ramp bounds |
| `P_RE[t]` | `Ren_Source` availability × rating | Exogenous profile vector |
| `E_BESS`, `P_BESS` | Fixed `Param` via bounds | **Decision variables** |
| Degradation | Commented future on `E_max` | Explicit `E_deg`, `L_BESS`, `N_r` |
| Timestep | `dt_hours` (1 h in window OPF) | **1 min** in publication |

**Model structure inside `bess_sizing.py`**

```
bess_sizing.py
├── build_bess_sizing_model(...)   # sets, params, vars, constraints, objective
│   ├── params: P_RE[t], C_t, P_AC_n, delta_rrl, L_pr, CC_*, CO_*, ...
│   ├── vars: E_BESS, P_BESS, P_bat_ch[t], P_bat_dch[t], P_grid[t], SOC[t], ...
│   ├── constraints: Eqs. (10)–(21) per scenario flags
│   └── objective: Eq. (4)
├── bess_sizing(...)               # public runner: validate → build → solve → export
└── export_bess_sizing_results(...) # DataFrame / dict / optional grid write-back
```

**Reuse from existing NL model (conceptual only — owner implements)**

- Sign convention: discharge − charge = net injection (aligned with `Storage.net_P_pu`)
- SoC bounds pattern (pu × `E_BESS` ↔ paper absolute MWh on `SOC`)
- Separate `P_charge` / `P_discharge` vars (same as G6 in bess plan)
- **Do not** reuse `storage_soc_balance` verbatim — paper Eq. (18) uses absolute
  MWh, self-discharge, and minute timestep

**Ramp rate (subject ①) — model placement**

| Constraint | Parent level | Index |
|------------|--------------|-------|
| `P_grid` ramp (Eq. 10) | Sizing model root | `t = 2…T` |
| `0 ≤ P_grid[t] ≤ P_AC-n` (Eq. 11) | Same | `t = 1…T` |
| POI balance (Eq. 16) | Same | `t = 1…T` |

Not in `window_soc_links` pattern — that parent links **hourly OPF blocks**, not
POI export ramps.

**BESS sizing (subject ②) — model placement**

| Constraint | Type |
|------------|------|
| `E_BESS`, `P_BESS` bounds | Variable bounds / commercial Eqs. (14)–(15) |
| `P_bat` limits (Eq. 12) | Couples operation to **sizing** vars |
| SoC limits (Eq. 13) | `SOC_min/max` × `E_BESS` |
| Degradation (Eqs. 1–3, 21) | Nonlinear auxiliary |
| Replacement cost (Eqs. 8, 20) | Nonlinear auxiliary |

### 4.3 Problem (runner in `bess_sizing.py`)

**Role:** assemble data, build model, invoke solver, validate inputs, export results.
All logic lives in `bess_sizing.py` — no changes to `window_opf.py` or TEP runners.

| Paper stage | Problem-layer responsibility |
|-------------|------------------------------|
| §4.1 PV profile | Load / build `P_RE[t]` from `Time_series` or external CSV (Solargis path in paper) |
| §4.1 Representative week | Owner preprocessing — reduce 525 600 → 10 080 rows |
| §4.3 Ramp limit | Set `delta_rrl` from plant config |
| §4.4 Markets | Attach `C_t` (pool / VPPA) |
| §4.5 Scenarios | Table 5 switches: `α_deg`, `α_cur`, `α_RV`, E-P constrained |
| §5 post | Optional rainflow on solved `SOC[t]` — **outside** optimizer |

**Relationship to existing runners**

| Runner | Relationship |
|--------|----------------|
| **`bess_sizing()`** | **This plan** — plant POI sizing |
| `window_nl_opf` | Downstream only — fixed BESS on real grid |
| `ts_acdc_opf` | Unrelated — myopic hourly operation |

**Input data dependencies**

- `Time_series` or aligned DataFrame: `P_RE` [MW], optional `price` [€/MWh]
- Plant metadata: `P_AC_n`, `delta_rrl`, `L_pr`, economics
- BESS technology row (LFP / LTO / …)

**Output (minimum)**

- Optimal `P_BESS`, `E_BESS`, `n`, `N_r`, `L_BESS`, profit breakdown
- Trajectories: `P_grid`, `P_bat-ch`, `P_bat-dch`, `SOC`, optional `P_cur`
- Flags: `grid.bess_sizing_run = True` (or owner equivalent)

### 4.4 Solution (`pyomo_model_solve.py`, `Results_class.py`, `Graph_Dash.py`)

**Role:** solve, report, visualize.

| Layer | Paper requirement | pyflow_acdc hook |
|-------|-------------------|------------------|
| Solver | Nonlinear (IPOPT-class) | `pyomo_model_solve` |
| Scaling | 10k+ minute variables | Owner: horizon reduction, scaling — paper uses representative week |
| Results tables | Table 6 style | New `Results` method or dedicated exporter |
| Rainflow | Table 7 | Post-solve script / `Results` appendix — **not** in model |
| Dash | Compare scenarios | Optional — extend family builder later; not Phase 1 |

---

## 5. Subject-focused breakdown

### 5.1 Ramp rate

| Layer | Content |
|-------|---------|
| **Classes** | `delta_rrl` on plant config; `P_AC_n` from site / `Ren_Source` aggregate |
| **Model** | Eq. (10) on `P_grid`; Eq. (11) export cap; Eq. (16) balance with `P_RE`, `P_bat`, optional `P_cur` |
| **Problem** | Minute (or sub-hour) time index; representative profile; scenario without curtailment (`α_cur=0`) isolates BESS-only ramp service |
| **Solution** | Report max ramp used vs limit; time series of `P_grid` vs raw `P_RE` (paper Figs. 6, 8) |

**Explicit non-goals for v1**

- DC line `ramp_max` (noted in `bess_integration_plan.md` Phase 4 future)
- Generator `ramp_agc` from MATPOWER
- Penalty-based soft ramp (paper uses **hard** constraint; Tahir [16] is different)

### 5.2 BESS sizing

| Layer | Content |
|-------|---------|
| **Classes** | Technology economics + degradation coefficients; sized `E_max`/`P_nom` written post-solve |
| **Model** | `E_BESS`, `P_BESS` as variables; optional Eqs. (14)–(15); degradation Eqs. (1)–(3), (20)–(21); profit Eq. (4) |
| **Problem** | Scenario matrix (Table 5); with/without degradation; with/without curtailment; LFP vs LTO |
| **Solution** | Table 6 metrics; hand-off to `add_storage` + `window_nl_opf` for grid-faithful check |

---

## 6. Phased plan (design-only — no code until owner lock)

### Phase 0 — Architecture lock

- [x] **Standalone `bess_sizing.py`** — no static / MP TEP changes (§3)
- [ ] Lock public entry point name (e.g. `bess_sizing(...)`)
- [ ] Lock unit system: paper uses MW, MWh, minutes; map to `Storage` pu + `S_base` on export only
- [ ] Lock horizon policy: full year vs representative week vs owner preprocessing

### Phase 1 — Paper fidelity (aggregated bus)

- [ ] Reproduce case study §4 (46.8 MW AC PV, 5 %/min, Table 4 parameters)
- [ ] Simulations 1–4 minimum (degradation on/off; ramp-focused VPPA)
- [ ] Compare Table 6 quantities within owner tolerances

### Phase 2 — Classes and data plumbing

- [ ] Technology profiles (LFP, LTO, …)
- [ ] Plant / POI configuration object
- [ ] `P_RE[t]` and `C_t` ingestion from `Time_series` or CSV
- [ ] Post-solve → `add_storage` bridge

### Phase 3 — Scenario engine

- [ ] Table 5 scenario flags (`α_deg`, `α_cur`, `α_RV`, E-P constrained, market scheme)
- [ ] Batch runner for technology / market comparisons

### Phase 4 — Post-processing and integration

- [ ] Rainflow SOC analysis (Table 7) — post-solve only
- [ ] `Results` / export / optional Dash
- [ ] Optional: feed sized BESS into `window_nl_opf` on PEI or PV case for network check

### Phase 5 — Documentation

- [ ] New `docs/usage_bess_sizing.rst` (or section in `usage_storage.rst`)
- [ ] Citation in `docs/citing.rst`
- [ ] Doc example + test (build-only or small horizon)

**Explicitly out of scope**

- Any edits to `ACDC_Static_TEP.py`, `ACDC_MultiPeriod_TEP.py`, `TEP_variables`
- Any edits to `storage_*` in `ACDC_OPF_NL_model.py` or `window_soc_links` in `window_opf.py`
- In-optimizer rainflow
- Full 525 600-min solve in CI

---

## 7. Queries before implementation

Explicit questions that need answers **before coding starts**. Each maps to one or more
uncertainty IDs in §8. Answers may come from pyflow owners, Montalà / CITCEA authors,
reference code, or a validation spike — not assumed here.

### 7.1 Scope and product

| Q-ID | Question | Blocks | Likely source | Uncertainty |
|------|----------|--------|---------------|-------------|
| Q-1 | Confirm **standalone `bess_sizing.py`** only — no TEP / NL OPF / `window_opf` edits? | Architecture | pyflow owner | U-A1 (implicit) |
| Q-2 | Is v1 **aggregated POI only** (no full AC/DC network), or is a mandatory **second-stage `window_nl_opf`** check required before sign-off? | Phase 1 vs 4 | pyflow owner | U-A2 |
| Q-3 | Validation on **paper PV case study** (46.8 MW, Iberian pool) vs **PEI / Mario data** — which is the primary acceptance target? | Test fixtures | Owner / authors | U-A3 |
| Q-4 | Should optimal sizes **auto-call `add_storage`**, or remain a report until the user applies them? | API | pyflow owner | U-A4 |
| Q-5 | New optional extra **`[BESS_SIZING]`**, or ship inside core **`[OPF]`**? | Packaging / CI | pyflow owner | U-A5 |
| Q-6 | Is there **author reference code** (FAIR project) for Eqs. (4)–(21) to port or diff against? | Phase 1 | Montalà / CITCEA | U-A6 |

### 7.2 Formulation and physics

| Q-ID | Question | Blocks | Likely source | Uncertainty |
|------|----------|--------|---------------|-------------|
| Q-7 | **`SOC_avg`** in calendar degradation (Eq. 3): arithmetic mean of `SOC[t]`, time-weighted, or fixed constant (Table 6 uses 0.651)? | Degradation block | Authors | U-M2 |
| Q-8 | **`N_r` replacements** (Eq. 20): strict integer / ceiling when `α_RV = 0`, or allow fractional as in Table 6 under `α_RV = 1`? | NLP vs MINLP | Authors | U-M3 |
| Q-9 | **Simultaneous charge/discharge**: allow overlap (paper separate vars, like G6) or enforce exclusivity? | Operation constraints | Owner / authors | U-M4 |
| Q-10 | Confirm **SoC sign convention** and **minute timestep** scaling in Eq. (18) vs pyflow `P_discharge − P_charge` | Dynamics | Authors + implementer | U-M5 |
| Q-11 | **`τ_bat` self-discharge**: exact conversion from %/month to per-minute factor? | Eq. (18) | Authors | U-M11 |
| Q-12 | **`P_grid[1]` ramp**: cold start vs continuing operation — boundary for multi-day extensions? | Ramp (Eq. 10) | Owner | U-M7 |
| Q-13 | **`α_deg = 0` mode**: is `N_r` fixed to 0, 1, or omitted from CAPEX (Eq. 8)? | Degradation toggle | Authors | U-M9 |

### 7.3 Units, data, and horizon

| Q-ID | Question | Blocks | Likely source | Uncertainty |
|------|----------|--------|---------------|-------------|
| Q-14 | Internal model units: **MW / MWh / minutes** (paper) with pu export only, or full pu inside the builder? | `add_storage` hand-off | pyflow owner | U-U1 |
| Q-15 | **`P_RE[t]` profile**: which public or licensed dataset substitutes for confidential Solargis 1-min data? | Phase 1 validation | Owner / data | U-D1 |
| Q-16 | **Representative-week selection** (“critical day per weekday”): exact algorithm or author-provided index list? | Horizon | Authors | U-D2 |
| Q-17 | Default operating horizon for v1: **10 080 min** (paper week), clustered medoids, or user-supplied only? | Model size | Owner | U-T1 |
| Q-18 | **Hourly pool prices** on 1-min grid: hold-forward, interpolate, or native 1-min series? | Income (Eq. 7) | Owner / data | U-U4 |
| Q-19 | What **% tolerance** on `P_BESS`, `E_BESS`, profit defines “Table 6 parity”? | Phase 1 exit | Authors / owner | U-D6 |

### 7.4 Solver and integration

| Q-ID | Question | Blocks | Likely source | Uncertainty |
|------|----------|--------|---------------|-------------|
| Q-20 | **Solver**: IPOPT only, or also Bonmin if integer `n` / `N_r` is enforced? | Solver choice | pyflow owner | U-S1 |
| Q-21 | **`delta_rrl` host**: `Grid`, plant config object, `Ren_Source`, or runner kwargs only? | Classes | Owner | U-C1 |
| Q-22 | **Technology library** in v1: LFP + LTO only (paper case study) or full Table 3 (LMO, NMC)? | Scenario engine | Owner | U-D5 |
| Q-23 | **`rainflow` post-processing**: required for v1 sign-off or Phase 4 optional? | Validation | Owner | U-M12, U-V1 |

### 7.5 Governance

| Q-ID | Question | Blocks | Likely source | Uncertainty |
|------|----------|--------|---------------|-------------|
| Q-24 | Who **signs off** on Eq. (18) `τ_bat` scaling and `SOC_avg` before Phase 1? | Formulation | Authors / owner | U-G2 |
| Q-25 | Will authors share **benchmark optimal values** (Table 6) or solver logs for regression tests? | CI / validation | Authors / CITCEA | U-G3 |

### 7.6 Minimum answers to start Phase 0 skeleton

At least tentatively resolve **Q-1, Q-2, Q-9, Q-10, Q-14, Q-17, Q-20, Q-24** before any
code is written. **Paper-faithful validation** additionally needs **Q-15, Q-16, Q-19,
Q-25**.

---

## 8. Uncertainties to resolve before start

Full catalog of open items. §7 distils the blocking subset into explicit queries. Items
below are **not yet decided** and may block or reshape implementation. They are recorded
here so work can proceed once answers exist — from owners, authors, validation runs, or
reference code. None of these assume the reader already knows the answer.

### 8.1 Architecture and product scope

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-A1 | **Public entry point name** inside `bess_sizing.py` (e.g. `bess_sizing` vs `solve_bess_sizing`) | `__init__.py` export |
| U-A2 | **v1 scope boundary** — aggregated POI only vs mandatory second-stage `window_nl_opf` refinement | Determines whether v1 is self-contained or always two-step |
| U-A3 | **Relationship to PEI / Mario / wind-island work** — separate PV case study vs reuse of `PEI_grid`, `_pei_bess_data.py` | Different data, physics, and validation baselines |
| U-A4 | **Whether sized output auto-wires into `add_storage`** or remains a standalone report until user confirms | API ergonomics and test design |
| U-A5 | **Optional dependency packaging** — core `[OPF]` only or new extra (e.g. `[BESS_SIZING]`) | Install surface and CI matrix |
| U-A6 | **Author reference implementation** — Montalà / CITCEA-UPC may have unpublished FAIR-project code; availability and reuse terms unknown | Could shortcut validation or constrain formulation choices |

### 8.2 Mathematical formulation (paper ambiguities)

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-M1 | **Eq. (17) typo** — paper prints `η_disch` in one place; nomenclature uses `η_dch` | Must pick one symbol for the model |
| U-M2 | **`SOC_avg` in Eq. (3)** — average over horizon as arithmetic mean of `SOC[t]`, time-weighted mean, or pre-specified constant (validation table uses 0.651) | Calendar degradation sensitivity |
| U-M3 | **`N_r` in Eq. (20)** — ceiling vs fractional when `α_RV = 0`; paper states integer domain but Table 6 shows non-integer `n` rep (e.g. 2.35) under `α_RV = 1` | MINLP vs NLP; solver choice |
| U-M4 | **Simultaneous charge and discharge** — paper uses separate `P_bat-ch`, `P_bat-dch` without exclusivity (same as Mario BESS plan G6) | Optimizer may use both; need policy on post-check vs constraint |
| U-M5 | **`P_bat[t]` sign in SoC update (Eq. 18)** — charging increases stored energy; confirm sign convention matches Eq. (17) and pyflow `P_discharge − P_charge` injection convention | SoC dynamics direction |
| U-M6 | **Curtailment `P_cur[t]` bounds** — only non-negative? capped by `P_RE[t]`? can export exceed `P_RE` when BESS discharges? | Feasibility of Eq. (16) |
| U-M7 | **`P_grid[t]` at `t = 1`** — no ramp from `t = 0`; what is the physical interpretation of the first interval (cold start vs continuing operation)? | Boundary condition for rolling / multi-day extensions |
| U-M8 | **Annualization factors** — Eqs. (2), (7), (8) scale partial horizons to one year via `8760·60/T`; confirm this is exact for profit comparison across different `T` | Objective scaling when horizon ≠ 10 080 |
| U-M9 | **Degradation toggle `α_deg = 0`** — paper sets `L_BESS = L_pr` with no replacements; confirm `N_r` fixed to 1 or 0 in that mode | CAPEX branch in Eq. (8) |
| U-M10 | **Eq. (14) `d_rate` units** — minutes of discharge at nominal power; LTO case 11.91 min vs LFP 60 min — confirm continuous vs catalog-snapped | Technology coupling |
| U-M11 | **Self-discharge `τ_bat`** — paper gives %/month; Eq. (18) applies per **minute** step — conversion formula not explicit | Must derive `τ_bat_per_min` consistently |
| U-M12 | **Rainflow validation scope** — Table 7 uses post-solve SOC; acceptance thresholds vs paper for calling implementation “faithful” undefined | Phase 1 exit criteria |

### 8.3 Units, per-unit, and pyflow mapping

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-U1 | **Internal units in the sizing model** — paper uses MW, MWh, minutes; existing `Storage` uses pu on `S_base` + `E_max` MWh | Export-to-`add_storage` conversion path |
| U-U2 | **`P_AC-n` vs sum of `Ren_Source` ratings** — single POI nameplate vs aggregated pyflow renewables | Ramp limit scaling (`δ_rrl · P_AC-n`) |
| U-U3 | **`P_RE[t]` source** — AC after PR (paper §4.1) vs raw `Time_series` ren columns vs DC-to-AC conversion | Profile magnitude and ramp stress |
| U-U4 | **Price `C_t` alignment** — hourly pool prices on 1-min grid (hold-forward? interpolate?) | Income Eq. (7) on minute index |
| U-U5 | **Currency** — Table 4 mixes $ and € labels; case study Iberian pool in €/MWh | Economics consistency in validation |
| U-U6 | **`CC_RE`, `CO_RE` in objective** — fixed offsets only; confirm they do not affect sizing optimum (should cancel in comparison) but affect reported profit | Table 6 profit column |
| U-U7 | **Mapping optimal `P_BESS` to `S_max` / `P_charge_max` / `P_discharge_max`** — symmetric vs asymmetric charge/discharge caps in `add_storage` | Hand-off fidelity |

### 8.4 Data and case study reproducibility

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-D1 | **Solargis 1-min data** — paper cites confidential data; no public CSV in repo | Cannot reproduce Table 6 exactly without substitute profile |
| U-D2 | **Representative-week algorithm** — “most critical day per weekday” selection rule not fully specified in equations | Different weeks → different `P_BESS`, `E_BESS` |
| U-D3 | **2022 Iberian pool price series** — which hours map to the 10 080-minute window | Income validation |
| U-D4 | **Technology cost tables** — LFP/LTO $/MW, $/MWh from literature [38], [30], [50]; whether to freeze paper values or allow updates | CAPEX/OPEX baseline |
| U-D5 | **`N_FCTF`, `EEOL`, `A_cal`, `B_cal` per chemistry** — Table 3 vs Table 4 cross-check; LMO/NMC in table but case study only LFP/LTO | Scope of bundled technology library |
| U-D6 | **Validation tolerances** — acceptable % deviation on `P_BESS`, `E_BESS`, profit for “paper parity” vs qualitative-only | Phase 1 exit gate |
| U-D7 | **Synthetic test profile for CI** — minimal public fixture when real Solargis data unavailable | CI without confidential inputs |

### 8.5 Horizon, time series, and clustering

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-T1 | **Default operating horizon** — full year (525 600), paper week (10 080), or `Time_series_clustering` medoids | Model size and ramp statistics |
| U-T2 | **Minute vs hourly internal timestep** — strict paper fidelity vs coarser step with adjusted `δ_rrl` | 10k+ variables per solve |
| U-T3 | **Reuse of `Time_series_clustering.py`** — whether representative periods from pyflow are methodologically equivalent to paper preprocessing | Could reduce U-D2 if policy aligned |
| U-T4 | **Terminal SoC Eq. (19)** — hard equality may cause infeasibility on short horizons or coarse timesteps | Relaxation or slack policy |
| U-T5 | **Leap years / DST** — if importing real timestamps | Row count ≠ 525 600 |

### 8.6 Solver, numerics, and performance

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-S1 | **Solver choice** — IPOPT for NLP; Bonmin/Couenne if integer `n` or `ceil(N_r)` enforced | `[OPF]` stack assumption |
| U-S2 | **Model convexity / local minima** — degradation and replacement terms are nonlinear; multiple scenario switches | Need multi-start or bounded search? |
| U-S3 | **Variable scaling** — `E_BESS`, `P_BESS` O(1–100 MW); `SOC` O(10 MWh); prices O(10–200) | IPOPT conditioning |
| U-S4 | **Build-only / solve time budget for CI** — 10 080-minute model may be too heavy for default pytest | CI strategy (tiny horizon, `build_only=True`) |
| U-S5 | **`obj_scaling` pattern** — sizing-specific vs reuse from other OPF drivers | Numerical stability |

### 8.7 Class model and configuration placement

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-C1 | **`delta_rrl` host object** — `Grid`, new `PlantPOI`, `Ren_Source`, or runner kwargs only | Discoverability and Dash/export |
| U-C2 | **Technology profile structure** — extend `Storage` vs separate `BESSTechnology` dataclass | Avoid polluting operational class |
| U-C3 | **`τ_bat`, degradation coeffs on `Storage`** — operational model ignores them today | Dead attrs vs sizing-only config |
| U-C4 | **Scenario flags `α_deg`, `α_cur`, `α_RV`** — `constants.py` enums vs runner kwargs vs case YAML | API consistency |
| U-C5 | **`grid.bess_sizing_run` flag** — new run flag alongside `window_opf_run` | Results routing in `Results.all()` |

### 8.8 Integration with existing operation stack

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-I1 | **Second-stage network check** — after POI sizing, is `window_nl_opf` required for sign-off or optional? | Documentation and examples |
| U-I2 | **Ramp at POI vs ramp on internal AC line** — hybrid grid may have multiple export points | Which bus gets Eq. (10) |
| U-I3 | **Coexistence with `ObjRule['SoC_deviation']`** — sizing uses hard terminal SoC; myopic TS uses soft ref | No conflict if separate, but docs must clarify |
| U-I4 | **Rolling / multi-year operation after sizing** — paper defers EMS; whether pyflow rolling window inherits sized BESS only | Out of v1 but affects class design |
| U-I5 | **Dash / visualization** — new sizing trajectories vs extend `Graph_Dash` family panels | Phase 4 optional scope |

### 8.9 Validation, testing, and documentation

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-V1 | **Minimum simulation set for v1** — all 13 paper runs vs subset (1–4, 3–4, 7–8) | Effort vs confidence |
| U-V2 | **LTO technology in v1** — required for parity with §5.6 or LFP-only first | Test matrix size |
| U-V3 | **Doc example data** — public synthetic PV curve acceptable for RTD literalinclude | Docs CI |
| U-V4 | **Citation placement** — new `usage_bess_sizing.rst` vs section in `usage_storage.rst` | Toctree structure |
| U-V5 | **Cross-link from `bess_integration_plan.md`** — single roadmap or sibling plans | Maintainer clarity |

### 8.10 Governance and ownership

| ID | Uncertainty | Why it matters |
|----|-------------|----------------|
| U-G1 | **Implementation owner** — CITCEA / paper authors vs pyflow maintainers | Review path for formulation |
| U-G2 | **Formulation sign-off** — who approves Eq. (18) `τ_bat` scaling and `SOC_avg` definition | Before Phase 1 coding |
| U-G3 | **Reference results** — will authors share benchmark optimal values or logs for Sim 2–4? | Validation without guessing tolerances |
| U-G4 | **Licensing of any shared FAIR-project scripts** | Legal gate if code is reused |

### 8.11 Suggested resolution order (non-binding)

See §7.6 for the query-level minimum. Full catalog: architecture skeleton after
**U-A2, U-M5, U-U1, U-T1, U-S1**; paper-faithful validation additionally **U-D1, U-D2,
U-D6, U-G3**.

---

## 9. File touch list

| File | Action |
|------|--------|
| **`pyflow_acdc/bess_sizing.py`** | **New** — model builder + runner + export (Eqs. 4–21) |
| `__init__.py` | Export public entry point from `bess_sizing` |
| `Classes.py` | Optional: technology / plant attrs for export only |
| `constants.py` | Optional: scenario enums, degradation defaults |
| `Time_series.py` | Optional: read-only helpers to build `P_RE[t]`, `C_t` inputs |
| `Results_class.py` | Optional: `bess_sizing()` report method |
| `grid_modifications.py` | Unchanged — `add_storage()` called after sizing |
| `docs/usage_*.rst`, `docs/citing.rst` | User guide + citation |
| `pyflow_tests/...` | Validation vs paper Table 6 |

**Do not modify**

| File | Reason |
|------|--------|
| `ACDC_Static_TEP.py` | Unrelated problem class |
| `ACDC_MultiPeriod_TEP.py` | Unrelated problem class |
| `ACDC_OPF_NL_model.py` (`storage_*`) | Operational OPF only |
| `window_opf.py` | Operational horizon only |

---

## 10. References

### Primary

M. Montalà Palau, M. Cheah Mañé, and O. Gomis-Bellmunt, *Techno-economic
optimization for BESS sizing and operation considering degradation and ramp rate
limit requirement*, J. Energy Storage **105**, 114631 (2025).
https://doi.org/10.1016/j.est.2024.114631

Local PDF: `mm_bess.pdf` (user-provided).

### Degradation validation source (paper Appendix)

T. Sayfutdinov et al., *Degradation and operation-aware framework for the optimal
siting, sizing, and technology selection of battery storage*, IEEE Trans. Sustain.
Energy **11**(4), 2130–2140 (2020). https://doi.org/10.1109/TSTE.2019.2950723

### pyflow_acdc (operation — separate concern)

M. Useche-Arteaga et al., *Optimizing the operation of energy islands…*, Wind
Energ. Sci. **11**, 349–372 (2026). Implemented in [bess_integration_plan.md](bess_integration_plan.md).
Hand-off: `bess_sizing()` → `add_storage()` → `window_nl_opf`.
