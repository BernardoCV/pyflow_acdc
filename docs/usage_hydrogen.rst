Green hydrogen / electrolyser
=============================

Operation-only electrolyser and H₂ inventory for hybrid AC/DC grids. The
formulation follows Useche-Arteaga et al. (2026) [#useche2026]_ (§3.4); see
:doc:`citing`.

Implemented: :class:`~pyflow_acdc.Electrolyser`,
:func:`~pyflow_acdc.add_electrolyser`, NL OPF when ``grid.H2``,
:func:`~pyflow_acdc.window_nl_opf` H₂ inventory links, and
``Results.ext_electrolyser`` / ``Results.hydrogen_window``.

Related API: :doc:`api/hydrogen`. BESS (often co-optimized): :doc:`usage_storage`.

Adding an electrolyser
----------------------

Elements attach to any **AC** or **DC** bus
(:attr:`~pyflow_acdc.Node_AC.connected_electrolyser`,
:attr:`~pyflow_acdc.Node_DC.connected_electrolyser`,
:attr:`~pyflow_acdc.Grid.electrolysers`). ``analyse_grid`` sets
``grid.H2`` when the list is non-empty.

- Active power ``P_electrolyser`` is a **load** (subtracted from nodal injection).
- On **AC**, optional reactive compensation via ``Q_min_MVAR`` / ``Q_max_MVAR``
  (generation convention: positive ``Q`` injects vars).
- On **DC**, ``Q`` is fixed at zero.
- Inventory ``mass_H2`` is in **kg** (``H2_mass_max``, ``H2_mass_initial``,
  optional ``H2_mass_final`` for **window / rolling OPF only**).
- ``h2_price`` (EUR/kg, default ``0``), static or ``TSType.H2_PRICE`` series:
  with ``ObjRule={'H2_sale': 1, ...}`` a zero price contributes nothing.
  In myopic ``ts_acdc_opf``, use **sale only** (no tank); mass target and
  inventory belong to window/rolling.
- Rolling horizons: :func:`~pyflow_acdc.rolling_window_nl_opf` (1-based
  ``start``/``end`` like ``ts_acdc_opf``).

Linear production each hour:

.. math::

   h = b_h\, P_e\, S_{\mathrm{base}}\, \Delta t + c_h

with ``c_h`` applied **every** frame (paper and Mario script).

The example below is taken from ``pyflow_tests/doc_examples/hydrogen/`` and
executed by ``test_docs_hydrogen.py``.

.. literalinclude:: ../pyflow_tests/doc_examples/hydrogen/01_add_electrolyser.py
   :language: python
   :lines: 2-

Running OPF
-----------

- **Snapshot** ``optimal_pf``: one inventory step from ``H2_mass_initial``; no
  terminal ``H2_mass_final``.
- **Coupled** :func:`~pyflow_acdc.window_nl_opf`: parent chain across frames;
  ``H2_mass_final`` enforced on the last frame when set.
- **Myopic** :func:`~pyflow_acdc.ts_acdc_opf`: **direct sale only** — no H₂
  tank / inventory carry between hours, and no ``H2_mass_final`` (there is no
  foresight to meet a mass target). Economics use ``ObjRule['H2_sale']``;
  hourly production is limited by electrolyser ``P_min`` / ``P_max`` only.
  There is no cumulative sale cap over the series. Use window/rolling when a
  tank + terminal mass is required.
- Results: ``Results.ext_electrolyser()`` (snapshot) and
  ``Results.hydrogen_window()`` after a window solve.

.. code-block:: python

    pyf.ts_acdc_opf(
        grid,
        ObjRule={"Energy_cost": 1, "H2_sale": 1},
        solver="ipopt",
    )

Roadmap
-------

+------------------+-----------------------------------------------------------+
| Item             | Status                                                    |
+==================+===========================================================+
| NL OPF model     | Done — ``hydrogen_variables`` / ``hydrogen_constraints``  |
+------------------+-----------------------------------------------------------+
| ``window_nl_opf``| Done — parent H₂ inventory chain                          |
+------------------+-----------------------------------------------------------+
| Results          | Done — ``ext_electrolyser``, ``hydrogen_window``          |
+------------------+-----------------------------------------------------------+
| PEI + Dash       | Done — see :doc:`usage_window_opf` (season-compare)       |
+------------------+-----------------------------------------------------------+
| ``ts_acdc_opf``  | Direct sale via ``H2_sale`` (no tank / no mass target)    |
+------------------+-----------------------------------------------------------+

Full phase list and design decisions:
`plans/bess_integration_plan.md
<https://github.com/CITCEA-UPC/pyflow_acdc/blob/mario_integration/plans/bess_integration_plan.md>`_.

**References**

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
