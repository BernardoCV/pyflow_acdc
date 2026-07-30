Controllable Heat Pumps
=======================

Controllable heat pumps are implemented as an AC-only planning-oriented flexible
load model: baseline electrical demand plus a bounded flexibility actuator with a
cumulative energy state, following Montalà-Palau et al. (2026) [#montala2026]_.

Implemented: :class:`~pyflow_acdc.HeatPump`,
:func:`~pyflow_acdc.add_heat_pump`, NL OPF when ``grid.HP``,
:func:`~pyflow_acdc.ts_acdc_opf`, :func:`~pyflow_acdc.window_nl_opf`,
``Results.ext_heat_pump`` and ``Results.heat_pump_window``.

Related API: :doc:`api/heat_pump`.

Adding a controllable heat pump
-------------------------------

Heat pumps attach to AC buses. Snapshot inputs are scalars for one-step OPF:

.. code-block:: python

    pyf.add_heat_pump(
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

For multi-hour studies, attach time series to the heat-pump name to override
those scalars per frame (``P_ref`` / ``Q_ref`` in pu, energy bounds in kWh):

.. code-block:: python

    pyf.add_TimeSeries(grid, df_p_ref, associated=hp.name, TS_type="hp_P_ref")
    pyf.add_TimeSeries(grid, df_q_ref, associated=hp.name, TS_type="hp_Q_ref")
    pyf.add_TimeSeries(grid, df_e_min, associated=hp.name, TS_type="hp_E_min")
    pyf.add_TimeSeries(grid, df_e_max, associated=hp.name, TS_type="hp_E_max")

The implementation is fail-fast: pyflow_acdc does not infer a thermal model or
comfort envelope internally. You must provide the baseline and admissible
energy-state bounds.

Modelling note
--------------

At each frame the served heat-pump demand is bounded by:

.. code-block:: text

    P_ref - n_units * P_unit_max <= P_hp <= P_ref
    E_prev + P_hp * dt within [E_min, E_max]

Reactive demand follows the same load sign convention used elsewhere in
pyflow_acdc.

**References**

.. [#montala2026] M. Montalà-Palau, J. J. Markus, M. Kazemi, M. Cheah-Mañé,
   C. Papadimitriou, and O. Gomis-Bellmunt: *Enhancing Distribution System
   Resilience through Energy Communities*, CIRED 2026 Brussels Workshop,
   Paper 1361, 2026.
