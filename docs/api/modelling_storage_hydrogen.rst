Storage and hydrogen modelling
==============================

Operation-only battery energy storage (BESS) and green-hydrogen electrolysers
for hybrid AC/DC grids. The formulation follows Useche-Arteaga et al. (2026)
[#useche2026]_ (§3.3–§3.4); see :doc:`../citing`.

Elements attach with :func:`~pyflow_acdc.add_storage` /
:func:`~pyflow_acdc.add_electrolyser` (:doc:`grid_mod`). Coupled multi-hour
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
        e_{t}^{\mathrm{fin}} &= E_{f} \\
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
:mod:`~pyflow_acdc.ACDC_OPF_NL_model` is:

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
        M_{t_{f}} &= M^{\mathrm{fin}} \\
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

**References**

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
