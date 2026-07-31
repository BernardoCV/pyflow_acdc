# Linear AC/DC hybrid OPF plan for pyflow_acdc

**Branch:** [`linear_DC`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/linear_DC)  
**Relation:** Sibling to [bess_integration_plan.md](bess_integration_plan.md) (NL + AC-only linear) and [convex_acdc_socp_plan.md](convex_acdc_socp_plan.md) (SOCP/CCP — out of scope here). This plan is **LP first** (optional convex QP later), not SOCP cones.

Living implementation plan for a full linear AC/DC hybrid OPF stack that **mirrors the nonlinear OPF modules** (snapshot, coupled window, rolling, myopic TS). Goal: **adapt existing NL formulations to LP** for computational efficiency — same pyflow_acdc surface and constraints, not a new physics set.

## Locked decisions

| ID | Topic | Decision |
|----|--------|----------|
| H0 | Formulation | **LP first**. Optional **convex QP later** only if needed (e.g. converter loss). **Solver-agnostic** (Pyomo); CI/docs default open-source **HiGHS**. No SOCP/rotated cones for AC power flow (SOCP plan). |
| H1 | Converter | **As rich as NL structure allows under LP**: keep topology modes, AC/DC P/Q link vars, S/P limits via **LP outer approximations** (box / diamond / piecewise), not exact ‖S‖₂. |
| H1b | Converter loss | **LP stage:** constant `a` and/or linear `|P|` aux (optional). **QP stage (later):** P-only epigraph `P_loss ≥ a + c_eff · P_c²` with `U` fixed (NL `(P²+Q²)/U²` → drop Q). MMC / `sqrt` deferred. |
| H2 | DC network | **Linearize NL `DC_constraints`**: replace `V_i(V_i−V_k)G` nodal sum and `PDC_from` / `PDC_to` with a fixed-operating-point linear map (use node `V_ini` as `V_ref`). Keep thermal bounds, DC gens/ren/storage/H₂ / converter injections with **same signs** as NL. |
| H3 | Module mirror | Full operational OPF surface: **snapshot + window + rolling + myopic TS**, AC and DC BESS/H₂ — only what NL already does. |
| H4 | Out of scope | SOCP/CCP, BESS sizing; features beyond current NL. **Linear hybrid TEP is in scope** (extend existing AC linear TEP with DC/conv when `TEP=True`, same flag pattern as NL). |
| H5 | Layout | **No new builder file.** Expand `L_models/AC_OPF_L_model.py` with `grid.ACmode` / `grid.DCmode` (and related) flags like `opf_create_nl_model_acdc`: single entry `opf_create_l_model_acdc` builds AC-only, DC+AC hybrid, or DC pieces as flagged. Keep `export_acdc_l_model_to_pyflow_acdc` in the same module, extended for DC/conv. |
| H6 | Scope rule | Linearize **only** what `NL_models/ACDC_OPF_NL_model.py` already models; no extra devices/constraints. |

```mermaid
flowchart TB
  subgraph drivers [Drivers mirror NL]
    optL[optimal_l_pf]
    winL[window_l_opf]
    rollL[rolling_window_l_opf]
    tsL[ts_acdc_l_opf or ts path in Time_series]
  end
  subgraph builders [L_models]
    lModel["AC_OPF_L_model\n(AC / DC / Conv via grid flags)"]
  end
  optL --> lModel
  winL --> lModel
  rollL --> lModel
  tsL --> lModel
```

## Current baseline (already shipped)

- AC-only LP: `opf_create_l_model_acdc` raises on `grid.DCmode` (`L_models/AC_OPF_L_model.py`) until hybrid flags are wired.
- AC window/rolling: `L_models/window_l_opf.py`.
- AC linear TEP already exists in the same L stack / `ACDC_L_TEP` paths.
- NL hybrid reference: `NL_models/ACDC_OPF_NL_model.py` (`DC_*`, `Converter_*`, DC BESS/H₂, `TEP` flag).

## Linearization principles (NL → LP, then optional QP)

Mirror NL **block structure and variable names** where practical so window parent links / export / `fx_conv` stay familiar.

**AC:** existing Bθ / known-P injection model in `AC_OPF_L_model` (gated by `grid.ACmode`).

**DC lines/nodes (H2 — explicit):** from NL `DC_constraints` / line equalities (gated by `grid.DCmode`):

- Nodal: `P_sum += pol * V_i * (V_i − V_k) * G * NumLines…` → linearize at `V_ref,i = V_ini` (e.g. `pol * V_ref,i * (V_i − V_k) * G * …`, or equivalent `G·V_ref` conductance form). Document the exact map in code comments / linearization table.
- Lines: NL `PDC_from = (V_f − V_t) * G * V_f * pol` and `PDC_to = (V_t − V_f) * G * V_t * pol` → same fixed-point linearization; keep `PDC_line_loss = PDC_from + PDC_to`.
- Keep `P_known_DC`, ren/gen/storage/H₂/converter/DCDC injections and signs identical to NL.

**Converters (LP first, when `ACmode` and `DCmode`):**

- Keep sets/modes (direct / filter / TF+filter) and vars: `P/Q_conv_s`, `P/Q_conv_c`, `P_conv_DC`, `P_conv_loss`, internal `U`/`θ` **or** reduced equivalents with voltages fixed to `V_ini` for linearity.
- **Loss:** LP-safe first (`a` and/or `b·|P|`); QP epigraph later (H1b).
- Apparent-power limits: LP outer approx (`|P|+|Q| ≤ α S_max` and/or separate `|P|,|Q|` caps)—document gap vs NL ‖S‖₂.
- `fx_conv` / `OPF_fx`: same PDC/PQ/PV fixes as NL on the linear vars.

**BESS/H₂:** AC as today (P-only); DC charge/discharge/SoC and H₂ mass as NL (`PGi_storage_DC`, `PGi_electrolyser_DC`); no DC Q; window parent SoC/H₂ links already side-aware in `NL_models/window_opf.py`.

**TEP:** when `TEP=True`, extend existing linear AC TEP parameters/variables with DC line / converter `np_*` as in NL `TEP_parameters` / `TEP_variables` — no separate hybrid TEP builder file for the snapshot OPF path.

## Phases

### Phase 0 — Plan + API contract

- [x] Living plan file `plans/linear_acdc_hybrid_plan.md` (this file).
- [x] Lock: LP first → optional QP later; H2 DC `V(V−V)G` / `PDC_from`/`to`; expand existing L builder (no new file); hybrid TEP in scope; NL-only physics.
- [x] Phase 1 snapshot LP: `opf_create_l_model_acdc` hybrid flags + DC linearization + thin converter + export + tests.
- [ ] Document public surface when coding starts: extend `optimal_l_pf` (no more hard AC-only when hybrid ready), `window_l_opf` / `rolling_window_l_opf`, new myopic `ts_acdc_l_opf` (or clearly named twin of `ts_acdc_opf`).
- [ ] Cross-link architecture / `api/L_models`; explicit “not SOCP” note.

### Phase 1 — Expand L model builder (snapshot, **LP**)

**File:** `L_models/AC_OPF_L_model.py` (same entry as today)

- [x] Drop hard `raise` on `grid.DCmode`; structure like NL with `ACmode` / `DCmode`.
- [x] **DC (H2):** linearized nodal `V(V−V)G` and `PDC_from` / `PDC_to` (and loss identity).
- [x] Thin converter: one link ``np·Ps + P_DC + np·(a + b·Ps) = 0`` (+ AC nodal aggregate).
- [x] Extend `export_acdc_l_model_to_pyflow_acdc` for DC nodes/lines/converters.
- [x] Wire `optimal_l_pf` / `opf_obj_l` (DC gen energy cost); hybrid `TEP=True` raises until later.
- [x] Tests: `test_hybrid_l_opf.py` + update former DC-reject test.

**Exit:** `optimal_l_pf` on AC-only and on a hybrid grid (`build_only`); converter + linearized DC flows export onto grid.

### Phase 2 — Converter richness + `fx_conv` (**still LP**)

- Flesh converter LP to match NL topology branches as far as allowed; document each approximation vs NL.
- Port `fx_conv` behavior for linear models.
- LP loss stand-in if not done in Phase 1 (`a` / `|P|`).
- Tests: fixed PDC/PQ modes; S-limit outer approx smoke; HiGHS LP preferred when available.

**Exit:** Hybrid snapshot with fx modes + documented LP linearization table (incl. DC H2 map).

### Phase 2b — Optional QP (later)

- Only if needed after LP stack works: H1b epigraph `P_loss ≥ a + c P²` (fixed U).
- Do not block window/TS on this.

### Phase 3 — Window + rolling hybrid

- `L_models/window_l_opf.py`: drop DCmode raise; build frames via `opf_create_l_model_acdc(..., window_block=True)` (hybrid via grid flags); reuse NL parent `window_soc_constraints` / `window_h2_constraints` / `export_window_opf_results`.
- Extend `_modify_l_window_parameters` for DC known-P / prices / converter-relevant mutable params.
- Rolling / `future_sight` unchanged at driver level.
- Tests: hybrid `build_only` window + rolling foresight half; PEI optional later (heavy).

**Exit:** `window_l_opf` / `rolling_window_l_opf` accept hybrid grids.

### Phase 4 — Myopic linear TS

- Add linear twin of `ts_acdc_opf` (carry SoC / `mass_H2_prev`, `empty_tank_cycle`, warm-start), using the same L builder per hour.
- Soft `SoC_deviation` remains **rejected** (quadratic) unless a later LP-safe surrogate is explicitly added—default: reject like snapshot.
- Tests: short hybrid TS carry; docs in `usage_window_opf` / `api/ts` / `api/L_models`.

**Exit:** Myopic linear hybrid TS path mirrored to NL.

### Phase 5 — Docs, changelog, polish

- `docs/api/L_models.rst`, `usage_window_opf.rst`, architecture note: hybrid LP vs NL vs SOCP.
- CHANGELOG; doc example on small hybrid case (`build_only`).
- Align hybrid TEP docs with existing AC linear TEP usage where applicable.

## Implementation order (dependencies)

```mermaid
flowchart LR
  P0[Phase0 plan_API] --> P1[Phase1 LP expand builder]
  P1 --> P2[Phase2 LP rich_conv_fx]
  P2 --> P2b[Phase2b optional QP]
  P1 --> P3[Phase3 window_rolling]
  P2 --> P3
  P1 --> P4[Phase4 myopic_TS]
  P3 --> P5[Phase5 docs]
  P4 --> P5
```

## Validation grids

- AC regression: `case39` / existing linear BESS-H₂ tests.
- Hybrid: `case39_acdc`, `Stagg5MATACDC` / `CigreB4_ACDC` as available; PEI only after Phase 3 if runtime allows.

## Explicit non-goals

- No changes to NL builders except shared export helpers if needed.
- No SOCP lifting / CCP / robust (convex plan).
- No new `ACDC_OPF_L_model.py` module — expand existing L builder only.
- No features beyond current NL operational OPF (plus linear hybrid TEP via existing `TEP` flag).
