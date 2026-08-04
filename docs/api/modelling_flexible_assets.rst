Flexible assets modelling
=========================

Operation-oriented battery energy storage (BESS), green-hydrogen electrolysers,
and controllable heat pumps for hybrid AC/DC grids.

BESS and electrolyser formulations follow Useche-Arteaga et al. (2026)
[#useche2026]_ (§3.3–§3.4). Heat pumps follow the planning-oriented controllable
load model of Montalà-Palau et al. (2026) [#montala2026]_ (§4.1). See
:doc:`../citing`.

Elements attach with :func:`~pyflow_acdc.add_storage` /
:func:`~pyflow_acdc.add_electrolyser` /
:func:`~pyflow_acdc.add_heat_pump` (:doc:`grid_mod`). Coupled multi-hour
runs: :doc:`window` / :doc:`../usage_window_opf`. Myopic sequential hours:
:func:`~pyflow_acdc.ts_acdc_opf` (:doc:`ts`).

.. _Storage_modelling:

Battery energy storage system
-----------------------------

Energy islands can improve flexibility by integrating storage to manage wind
intermittency, grid constraints, and curtailment. Conventional BESS with power
converters regulate active and (on AC) reactive power. The energy storage
system follows the linear model of Pozo (2022) as used in [#useche2026]_
Eqs. (24)–(31).

Paper energy-balance form (SoE :math:`e_t`, charging / discharging
:math:`P_t^{c}`, :math:`P_t^{d}`, efficiencies :math:`\eta_c`, :math:`\eta_d`):

.. math::
    :label: eq:bess_soe

    \begin{align}
        e_{t} &= e_{t-1} + \eta_c P_{t}^{c} - \frac{1}{\eta_d} P_{t}^{d}
            & \qquad \forall t \in \mathcal{T} \\
        e_{t}^{\mathrm{ini}} &= E_{0}, \quad
        e_{t}^{\mathrm{fin}} = E_{f} & \\
        e^{\min} &\leq e_{t} \leq e^{\max}
            & \qquad \forall t \in \mathcal{T} \\
        0 &\leq P_{t}^{c} \leq P^{c,\max},\quad
        0 \leq P_{t}^{d} \leq P^{d,\max}
            & \qquad \forall t \in \mathcal{T}
    \end{align}

On AC buses the converter capability curve limits apparent power
(:math:`s_{t}^{b}` net injection, rating :math:`s_{b}^{\max}`):

.. math::
    :label: eq:bess_s_circle

    \begin{align}
        s_{t}^{b} &= \bigl(P_{t}^{d} - P_{t}^{c}\bigr) + j\, q_{t}^{b}
            & \qquad \forall t \in \mathcal{T} \\
        \|s_{t}^{b}\| &\leq s_{b}^{\max}
            & \qquad \forall t \in \mathcal{T}
    \end{align}

**pyflow-acdc implementation.** SoC is stored in **pu** (fraction of
:attr:`~pyflow_acdc.Storage.E_max` in MWh). Charge/discharge decision variables
are in **pu** of ``S_base``. The discrete SoC update implemented in
:mod:`~pyflow_acdc.NL_models.ACDC_OPF_NL_model` is:

.. math::
    :label: eq:bess_soc_pyflow

    \mathrm{SoC}_{t}
    =
    \mathrm{SoC}_{t-1}
    +
    \frac{\Delta t\, S_{\mathrm{base}}}{E_{\max}}
    \Bigl(
        \eta_c P_{t}^{c} - \frac{P_{t}^{d}}{\eta_d}
    \Bigr)

with bounds ``soc_min`` / ``soc_max``, optional window terminal ``soc_final``,
and AC S-circle / DC net-:math:`P` limits. **Sign convention:** net active
power injected into the bus is ``P_discharge - P_charge``.

Linear model
^^^^^^^^^^^^

In the linear OPF stack the same SoC update is used, with a net active-power
limit and no reactive storage power:

.. math::
    :label: eq:bess_l

    \begin{align}
        \mathrm{SoC}_{t}
        &=
        \mathrm{SoC}_{t-1}
        +
        \frac{\Delta t\, S_{\mathrm{base}}}{E_{\max}}
        \Bigl(
            \eta_c P_{t}^{c} - \frac{P_{t}^{d}}{\eta_d}
        \Bigr) \\
        |P_{t}^{d}-P_{t}^{c}| &\leq P^{\max}
    \end{align}

* :attr:`~pyflow_acdc.Node_AC.connected_storage` /
  :attr:`~pyflow_acdc.Node_DC.connected_storage` /
  :attr:`~pyflow_acdc.Grid.storage_elements`
* ``analyse_grid`` sets ``grid.ESS`` when storage is present
* Results: ``Results.ext_storage`` / ``Results.storage_window``

Class Reference: :class:`pyflow_acdc.Classes.Storage`

.. autoclass:: pyflow_acdc.Storage
   :no-members:

Add with :func:`~pyflow_acdc.add_storage`.

Example:

.. literalinclude:: ../../pyflow_tests/doc_examples/storage/01_add_storage.py
   :language: python
   :lines: 2-

.. _Electrolyser_modelling:

Green hydrogen / electrolyser
-----------------------------

Hydrogen production is inherently nonlinear (voltage–current density,
temperature, degradation). pyflow-acdc uses the **linear** production model of
[#useche2026]_ Eqs. (32)–(36): inventory :math:`M_t`, production
:math:`h_t`, electrolyser demand :math:`P_{t}^{e}`, slope / intercept
:math:`b_h`, :math:`c_h`:

.. math::
    :label: eq:h2_inventory

    \begin{align}
        M_{t} &= M_{t-1} + h_{t}
            & \qquad \forall t \in \mathcal{T} \\
        h_{t} &= b_{h}\, P_{t}^{e} + c_{h}
            & \qquad \forall t \in \mathcal{T} \\
        M_{t_{i}} &= M^{\mathrm{ini}},\quad
        M_{t_{f}} = M^{\mathrm{fin}} & \\
        0 &\leq M_{t} \leq \overline{M}
            & \qquad \forall t \in \mathcal{T}
    \end{align}

**pyflow-acdc implementation.** Inventory ``mass_H2`` is in **kg**. Active
power ``P_electrolyser`` is a **load** (subtracted from nodal injection).
Production each frame uses ``c_h`` every hour:

.. math::
    :label: eq:h2_pyflow

    h = b_{h}\, P_{e}\, S_{\mathrm{base}}\, \Delta t + c_{h},
    \qquad
    M_{t} = M_{t-1} + h

On **AC**, optional reactive compensation via ``Q_min_MVAR`` / ``Q_max_MVAR``
(generation convention). On **DC**, ``Q`` is fixed at zero.

``empty_tank_cycle`` (``None`` or ``N >= 1``) controls **out-of-opt** tank
resets between solves (not a Pyomo constraint) — see :doc:`../usage_window_opf`
and :func:`~pyflow_acdc.ts_acdc_opf`. Optional ``H2_mass_final`` is enforced in
coupled window / rolling OPF when set. Economics use ``h2_price`` with
``ObjRule['H2_sale']``.

Linear model
^^^^^^^^^^^^

In the linear OPF stack the same inventory update is used with electrolyser
active power only:

.. math::
    :label: eq:h2_l

    h = b_{h}\, P_{e}\, S_{\mathrm{base}}\, \Delta t + c_{h},
    \qquad
    M_{t} = M_{t-1} + h

* :attr:`~pyflow_acdc.Node_AC.connected_electrolyser` /
  :attr:`~pyflow_acdc.Node_DC.connected_electrolyser` /
  :attr:`~pyflow_acdc.Grid.electrolysers`
* ``analyse_grid`` sets ``grid.H2`` when electrolysers are present
* Results: ``Results.ext_electrolyser`` / ``Results.hydrogen_window``

Class Reference: :class:`pyflow_acdc.Classes.Electrolyser`

.. autoclass:: pyflow_acdc.Electrolyser
   :no-members:

Add with :func:`~pyflow_acdc.add_electrolyser`.

Example:

.. literalinclude:: ../../pyflow_tests/doc_examples/hydrogen/01_add_electrolyser.py
   :language: python
   :lines: 2-

.. _Heat_pump_modelling:

Controllable heat pumps
-----------------------

Local energy communities can provide flexibility through heat pumps (HPs)
operated as controllable electrical loads. In the planning-oriented model of
[#montala2026]_, HP active and reactive powers are decomposed into a baseline
(reference) demand and a flexibility activation, subject to converter-style
instantaneous bounds and cumulative energy (comfort) envelopes.

Paper form (controllable load :math:`d \in \mathcal{D}`, reference
:math:`p^{*}_{d,t}`, :math:`q^{*}_{d,t}`, served demand :math:`p_{d,t}`,
:math:`q_{d,t}`, cumulative electrical energy :math:`e_{d,t}` in kWh):

.. math::
    :label: eq:hp_bounds

    \begin{align}
        \underline{p}_{d,t} &\leq p_{d,t} \leq \overline{p}_{d,t}
            & \qquad \forall d \in \mathcal{D} \\
        \underline{q}_{d,t} &\leq q_{d,t} \leq \overline{q}_{d,t}
            & \qquad \forall d \in \mathcal{D} \\
        \frac{\underline{e}_{d,t} - e_{d,t}}{\Delta t}
            &\leq p_{d,t} \leq
            \frac{\overline{e}_{d,t} - e_{d,t}}{\Delta t}
            & \qquad \forall d \in \mathcal{D}
    \end{align}

Energy evolves sequentially as
:math:`e_{d,t+1} = e_{d,t} + p_{d,t}\,\Delta t` (with :math:`e_{d,0} = 0` in
the paper's planning setup). Admissible envelopes
:math:`(\underline{p},\overline{p})`, :math:`(\underline{q},\overline{q})`,
and :math:`(\underline{e},\overline{e})` encode thermal comfort / hot-water
limits derived offline from building simulations; pyflow-acdc does **not**
embed a thermal model.

**pyflow-acdc implementation.** Baseline electrical demand is stored as
``P_ref`` / ``Q_ref`` in **pu** on ``S_base``. Instantaneous flexibility is
bounded by the aggregate unit rating ``n_units * P_unit_max``:

.. math::
    :label: eq:hp_pyflow_p

    P_{\mathrm{ref}} - n_{\mathrm{units}}\, P_{\mathrm{unit}}^{\max}
    \leq
    P_{\mathrm{hp}}
    \leq
    P_{\mathrm{ref}}

Reactive demand follows the load sign convention used in the NL AC balance
(subtracted at the bus), with

.. math::
    :label: eq:hp_pyflow_q

    Q_{\mathrm{ref}} \leq Q_{\mathrm{hp}} \leq 0

Cumulative energy state ``E_state`` is in **kWh**. The discrete update in
:mod:`~pyflow_acdc.NL_models.ACDC_OPF_NL_model` / sequential OPF is:

.. math::
    :label: eq:hp_energy_pyflow

    E_{t}
    =
    E_{t-1}
    +
    P_{\mathrm{hp}}\, S_{\mathrm{base}}\, \Delta t

with bounds ``E_min`` / ``E_max``. Window OPF owns the energy chain across
frames (:func:`~pyflow_acdc.NL_models.window_opf.window_heat_pump_constraints`).

**Power flow.** Unlike BESS / H₂ PF setpoint series (``storage_P``, ``h2_P``,
… via :func:`~pyflow_acdc.update_grid_for_pf`), heat-pump baselines stay on
:func:`~pyflow_acdc.update_grid_data` (``hp_P_ref``, ``hp_Q_ref``,
``hp_E_min``, ``hp_E_max``). In AC power flow,
:meth:`~pyflow_acdc.Grid.update_pq_ac` treats ``P_ref`` / ``Q_ref`` as known
nodal loads (same sign convention as NL OPF).

* :attr:`~pyflow_acdc.Node_AC.connected_heat_pumps` /
  :attr:`~pyflow_acdc.Grid.heat_pumps`
* ``analyse_grid`` sets ``grid.HP`` when heat pumps are present
* Results: ``Results.ext_heat_pump`` / ``Results.heat_pump_window``

Class Reference: :class:`pyflow_acdc.Classes.HeatPump`

.. autoclass:: pyflow_acdc.HeatPump
   :no-members:

Add with :func:`~pyflow_acdc.add_heat_pump`. Snapshot inputs are scalars for
one-step OPF; for multi-hour studies attach time series to the heat-pump name
(``P_ref`` / ``Q_ref`` in pu, energy bounds in kWh):

.. code-block:: python

    import pyflow_acdc as pyf

    hp = pyf.add_heat_pump(
        grid,
        "bus1",
        P_ref_MW=0.08,
        Q_ref_MVAR=-0.02,
        n_units=2,
        P_unit_max_MW=1.76 / 1000,
        E_min_kWh=-5.0,
        E_max_kWh=5.0,
        E_state_initial_kWh=0.0,
    )
    pyf.add_TimeSeries(grid, df_p_ref, associated=hp.name, TS_type="hp_P_ref")
    pyf.add_TimeSeries(grid, df_q_ref, associated=hp.name, TS_type="hp_Q_ref")
    pyf.add_TimeSeries(grid, df_e_min, associated=hp.name, TS_type="hp_E_min")
    pyf.add_TimeSeries(grid, df_e_max, associated=hp.name, TS_type="hp_E_max")

**References**

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026

.. [#montala2026] M. Montalà-Palau, J. J. Markus, M. Kazemi, M. Cheah-Mañé,
   C. Papadimitriou, and O. Gomis-Bellmunt: *Enhancing Distribution System
   Resilience through Energy Communities*, CIRED 2026 Brussels Workshop,
   Paper 1361, 2026.
