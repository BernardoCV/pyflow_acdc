Battery energy storage (BESS)
=============================

Operation-only battery energy storage for hybrid AC/DC grids. The formulation
follows Useche-Arteaga et al. (2026) [#useche2026]_ (§3.3); see :doc:`citing`.

Implemented: :class:`~pyflow_acdc.Storage`,
:func:`~pyflow_acdc.add_storage`, NL OPF when ``grid.ESS``,
:func:`~pyflow_acdc.window_nl_opf`, and ``Results.ext_storage`` /
``Results.storage_window``. For long series, :func:`~pyflow_acdc.rolling_window_nl_opf`
chains windows (1-based ``start``/``end`` like ``ts_acdc_opf``) with SoC carry-over.

Related API: :doc:`api/storage`.

Adding a BESS
-------------

Storage elements attach to any **AC** or **DC** bus, using the same node-hook
pattern as generators and renewable sources
(:attr:`~pyflow_acdc.Node_AC.connected_storage`,
:attr:`~pyflow_acdc.Node_DC.connected_storage`,
:attr:`~pyflow_acdc.Grid.storage_elements`).

AC storage supports reactive power (``Q``) and an apparent-power rating
(``S_max``). DC storage has active power only (``P_max``).

SoC is stored in **pu** (fraction of :attr:`~pyflow_acdc.Storage.E_max`).
Physical energy capacity ``E_max`` is in **MWh** (reserved for future degradation
modelling).

**Sign convention:** net active power **injected into the bus** is
``P_discharge - P_charge`` (discharging counts as generation).

The example below is taken from ``pyflow_tests/doc_examples/storage/`` and
executed by ``test_docs_storage.py``.

.. literalinclude:: ../pyflow_tests/doc_examples/storage/01_add_storage.py
   :language: python
   :lines: 2-

Princess Elisabeth Energy Island
--------------------------------

For the Mario validation case, the offshore hub in :func:`~pyflow_acdc.cases.PEI_grid`
is bus **`PE_Island`** (220 kV), corresponding to node index ``0`` in the
reference script.

Roadmap
-------

+------------------+-----------------------------------------------------------+
| Item             | Status                                                    |
+==================+===========================================================+
| NL OPF model     | Done — SoC dynamics and S-circle in                       |
|                  | :mod:`pyflow_acdc.ACDC_OPF_NL_model`                      |
+------------------+-----------------------------------------------------------+
| ``window_nl_opf``| Done — coupled multi-hour nonlinear OPF                   |
+------------------+-----------------------------------------------------------+
| Results          | Done — ``ext_storage``, ``storage_window``                |
+------------------+-----------------------------------------------------------+
| PEI + Dash       | Done — see :doc:`usage_window_opf` (season-compare)       |
+------------------+-----------------------------------------------------------+
| ``ts_acdc_opf``  | Done — myopic SoC carry + soft ``soc_ref`` via            |
|                  | ``ObjRule['SoC_deviation']`` (quadratic)                   |
+------------------+-----------------------------------------------------------+

Full phase list and design decisions:
`plans/bess_integration_plan.md
<https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_integration_plan.md>`_.

Myopic TS + soft SoC reference
-------------------------------

Sequential ``ts_acdc_opf`` carries BESS SoC hour-to-hour.
A secondary objective ``SoC_deviation`` penalises ``(SoC - soc_ref)**2`` so the
battery is pulled back toward ``soc_ref`` (defaults to ``soc_initial``) instead of
emptying and staying empty under pure energy-cost myopia.

.. code-block:: python

    pyf.ts_acdc_opf(
        grid,
        ObjRule={"Energy_cost": 1, "SoC_deviation": 10},
        solver="ipopt",
    )

Coupled ``window_nl_opf`` / rolling keep hard ``soc_initial`` / ``soc_final``;
soft SoC is for the forward-only TS path.

H₂ in the same myopic run carries inventory (``empty_tank_cycle``) — see
:doc:`usage_hydrogen`.

**References**

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
