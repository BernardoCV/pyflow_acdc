# Heat-pump flexible load plan for pyflow_acdc

**Repository:** In-repo links target the [`mario_integration`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration) branch ([`plans/`](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration/plans)).

Implementation plan for controllable **heat pumps** as AC flexible loads in the
nonlinear AC/DC OPF. Content was split out of an earlier draft that lived under
the BESS integration plan; this document is the sole plan for HP work.

This extension should reuse the same OPF integration style already used for
`Storage_*` and `Electrolyser`: explicit classes, nodal hooks, snapshot OPF
support, and time-coupled state updates in sequential / window modes.

**Primary reference**

M. Montalà-Palau, J. J. Markus, M. Kazemi, M. Cheah Mañé, C. Papadimitriou, and
O. Gomis-Bellmunt, *Enhancing Distribution System Resilience through Energy
Communities*, CIRED 2026 Brussels Workshop, Paper 1361, 2026.

- HP flexibility characterization: paper §3
- Planning-oriented HP + ESS resilience model: paper §4.1
- Operation-oriented FEL / reserve-envelope model: paper §4.2
- Local PDF: `citcea_extras_pyflow/heat_pumps_OPF/CIRED2026_Enhancing_Distribution_System_Resilience_through_Energy_Communities.pdf`
- Prototype: `citcea_extras_pyflow/heat_pumps_OPF/ILEC_full_code.py` (Montse)

**Related pyflow_acdc assets**

| Document | Link |
|----------|------|
| BESS / H₂ operation (separate) | [bess_integration_plan.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_integration_plan.md) |
| Architecture map | [docs/architecture.md](https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/docs/architecture.md) |
| Sibling plans | [plans/](https://github.com/CITCEA-UPC/pyflow_acdc/tree/mario_integration/plans) |

---

## 1. Role in the model

- Heat pumps are treated as **controllable electrical loads** connected at AC buses.
- They provide flexibility through deviations around a **baseline active-power profile** while respecting instantaneous power bounds and a cumulative energy state tied to thermal comfort.
- In the planning-oriented formulation, HP flexibility reduces the stationary ESS power required during contingencies.
- In the operation-oriented formulation, HP flexibility is scheduled economically while preserving a **state-dependent remaining reserve**.

---

## 2. Observed Montse pattern in `ILEC_full_code.py`

- The prototype keeps the original demand as a regular non-controllable `load`:
  - `pp.create_load(..., p_mw=P_load, q_mvar=Q_load, controllable=False)`
- For controllable HP demand, it adds a second controllable actuator named `"<load>_shedding"` implemented as `sgen`:
  - `pp.create_sgen(..., p_mw=0, q_mvar=0, controllable=True)`
- Its admissible active-power range is recomputed every step from:
  - device-level instantaneous shedding capability: `num_con_load * 1.76 kW`
  - cumulative energy state `LOAD_cap[name]`
  - time-varying comfort envelopes `e_min_load[t]`, `e_max_load[t]`
- After OPF, the cumulative state is updated explicitly:

```text
LOAD_cap_next = LOAD_cap + P_baseline_served - P_shedding
```

- Supplied controllable-load power is recovered as:

```text
P_hp_supplied = P_baseline_load - P_shedding
Q_hp_supplied = Q_baseline_load - Q_shedding
```

- This means the controllable part is modeled as a **baseline load plus a bounded flexibility actuator**, not as a single direct optimized load variable.

---

## 3. Proposed class and attributes

```python
class HeatPump:
    hpNumber
    connected = AcDcSide.AC
    # host: Node_AC
    # baseline / envelope / cumulative-state metadata
```

| Attribute | Meaning |
|-----------|---------|
| `P_ref` | Baseline active-power profile or current-frame reference power |
| `Q_ref` | Baseline reactive-power profile or current-frame reference power |
| `n_units` | Number of controllable HP units behind the aggregate load |
| `P_unit_max_MW` | Per-unit HP compressor power (Montse code uses `1.76 kW`) |
| `E_state` | Cumulative electrical-energy state (`LOAD_cap` / `e_d,t`) |
| `E_min`, `E_max` | Time-varying comfort / thermal bounds on `E_state` |
| `P_shed_max` | Optional explicit instantaneous shedding cap if precomputed |
| `P_nom` | Signed normalization power for FEL-style normalized dynamics |
| `dt_hours` | Time-step duration |
| `reserve_up`, `reserve_down` | Post-solve remaining reserve outputs |

---

## 4. Planning-oriented interpretation (paper Eq. 4–6 + Montse prototype)

- The implementation should follow Montse’s **baseline + flexibility-actuator** pattern, but expose it through a first-class `HeatPump` API in pyflow.
- Let `P_ref[d,t]`, `Q_ref[d,t]` be the baseline electrical demand and `P_shed[d,t]`, `Q_shed[d,t]` the flexibility actuator.
- The actually supplied controllable HP demand becomes:

```text
P_hp[d,t] = P_ref[d,t] - P_shed[d,t]
Q_hp[d,t] = Q_ref[d,t] - Q_shed[d,t]
```

- In Montse’s code, the actuator bounds are:

```text
P_shed_min[d,t] = max(P_ref[d,t] - n_units[d] * P_unit_max[d], E_state[d,t] + P_ref[d,t] - E_max[d,t])
P_shed_max[d,t] = min(P_ref[d,t], E_state[d,t] + P_ref[d,t] - E_min[d,t])
```

- with `E_state` in energy units consistent with the time step (Montse uses kWh-style accumulation).
- Reactive flexibility is bounded in a simple envelope around the baseline reactive demand:

```text
Q_shed_min[d,t] = min(0, -Q_ref[d,t])
Q_shed_max[d,t] = max(0, -Q_ref[d,t])
```

- The cumulative comfort state is then updated from the **served** controllable demand:

```text
E_state[d,t+1] = E_state[d,t] + P_hp[d,t] * dt
```

- As in the paper/prototype, this gives a fail-fast operational envelope rather than an inferred thermal model inside pyflow.

---

## 5. Operation-oriented FEL interpretation (paper Eq. 11–19)

- Use a normalized deviation state around the baseline:

```text
E_flex[d,t+1] = E_flex[d,t] + dt * ((P_hp[d,t] - P_ref[d,t]) / |P_nom[d]|)
```

- Enforce:

```text
P_low[d,t] <= P_hp[d,t] <= P_high[d,t]
E_min[d] <= E_flex[d,t] <= E_max[d]
E_flex[d,t0] = E_flex[d,tn] = 0
```

- After solving, derive actionable reserve exactly as a **post-processing output** from scheduled `P_hp` and `E_flex`, not as a new optimization variable:
  - instantaneous headroom from power bounds
  - comfort headroom converted to power-equivalent reserve
  - aggregate `R_up`, `R_down` over all HPs on the LEC / grid area

---

## 6. Implementation choice for pyflow_acdc

- For the **first implementation**, prefer the **planning-oriented Montse pattern** because it maps cleanly onto the existing pyflow architecture:
  - baseline demand already exists in nodal balances
  - the flexible part can be modeled as an explicit controllable HP actuator
  - the energy state can be updated in the same snapshot / time-coupled style already used for storage and H₂
- Keep the future FEL variant as a second step once the planning-oriented class is stable.

### Nodal balance

- In pyflow terms, the **net controllable HP demand** should still behave as a load at the connected AC node:

```text
P_var[node] -= P_hp[d,t]
Q_var[node] -= Q_hp[d,t]
```

- If implemented internally as baseline demand plus a flexibility actuator, the equivalent nodal contribution is:

```text
P_var[node] -= P_ref[d,t] - P_shed[d,t]
Q_var[node] -= Q_ref[d,t] - Q_shed[d,t]
```

- This keeps the sign convention aligned with existing `Load` / `Electrolyser` handling while matching Montse’s prototype.

### Implementation notes

- Add a dedicated `HeatPump` class instead of overloading `Load` or `Storage`.
- Prefer **explicit baseline + envelope inputs** from preprocessing / external thermal simulation, matching the PDF workflow and `ILEC_full_code.py`.
- Start with **AC-only** support.
- Keep reserve computation explicit in `Results_class.py` / reporting, rather than hiding it in OPF internals.
- Do not infer house-level thermal dynamics inside pyflow_acdc unless a later plan explicitly asks for that scope.
- Store the **cumulative state in physical energy units** for the planning-oriented variant, since that is how Montse’s implementation updates `LOAD_cap`.

---

## 7. File touch list

| File | Changes |
|------|---------|
| `Classes.py` | `HeatPump`, `Grid.heat_pumps`, `Node_AC.connected_heat_pumps` |
| `grid_modifications.py` | `add_heat_pump()` with baseline / flexibility envelope inputs |
| `ACDC_OPF_NL_model.py` | HP active/reactive load vars, flexibility-state constraints, AC nodal load hook |
| `window_opf.py`, `Time_series.py` | sequential / coupled `E_flex` / `E_state` updates and terminal neutrality rules |
| `Results_class.py` | HP schedule, comfort-state, and reserve-envelope reporting |
| `__init__.py` | Export new symbols |
| Docs | dedicated flexibility / heat-pump usage page with CIRED 2026 citation and Montse prototype mapping |
| Tests | reproduce Montse’s bound logic on a tiny one-bus / one-HP case before scaling to network cases |

---

## 8. Suggested work package

1. `Classes.py`: add `HeatPump`, `Grid.heat_pumps`, `Node_AC.connected_heat_pumps`.
2. `grid_modifications.py`: add `add_heat_pump(...)` with hard validation of baseline `P_ref` / `Q_ref`, `n_units`, and `E_min` / `E_max` envelopes.
3. `ACDC_OPF_NL_model.py`: add either explicit `P_heat_pump`, `Q_heat_pump` vars or an internal `P_shed`, `Q_shed` actuator representation with equivalent nodal hooks.
4. `Time_series.py` / `window_opf.py`: support sequential and coupled updates of the cumulative HP state.
5. `Results_class.py`: add HP schedules, cumulative energy state, and remaining-reserve summaries.
6. Tests: reproduce Montse’s bound logic on a tiny one-bus / one-HP case before scaling to network cases.
7. Docs: add a dedicated flexibility / heat-pump usage page with the CIRED 2026 citation and a note on the Montse prototype mapping.
