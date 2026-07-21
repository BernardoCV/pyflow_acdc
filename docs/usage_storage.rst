Battery energy storage (BESS)
=============================

Operation-only battery energy storage for hybrid AC/DC grids. The formulation
follows Useche-Arteaga et al. (2026) [#useche2026]_ (§3.3); see :doc:`citing`.

**Status:** Phase 1 is implemented (:class:`~pyflow_acdc.Storage_AC`,
:class:`~pyflow_acdc.Storage_DC`, :func:`~pyflow_acdc.add_storage`). Nonlinear OPF constraints, coupled
:func:`~pyflow_acdc.window_opf`, and results export are tracked in the
`BESS integration plan
<https://github.com/CITCEA-UPC/pyflow_acdc/blob/main/docs/plans/bess_integration_plan.md>`_.

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

SoC is stored in **pu** (fraction of :attr:`~pyflow_acdc.Storage_AC.E_max`).
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
| Planned          | Description                                               |
+==================+===========================================================+
| NL OPF model     | SoC dynamics and S-circle per storage element in          |
|                  | :mod:`pyflow_acdc.ACDC_OPF_NL_model`                      |
+------------------+-----------------------------------------------------------+
| ``window_opf``   | Coupled multi-hour nonlinear OPF (paper-faithful horizon)   |
+------------------+-----------------------------------------------------------+
| Results          | ``Results.ext_storage`` and window time-series tables     |
+------------------+-----------------------------------------------------------+
| ``ts_acdc_opf``  | Myopic sequential SoC (deferred)                          |
+------------------+-----------------------------------------------------------+

Full phase list and design decisions:
`docs/plans/bess_integration_plan.md
<https://github.com/CITCEA-UPC/pyflow_acdc/blob/main/docs/plans/bess_integration_plan.md>`_.

**References**

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
