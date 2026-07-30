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
Window / rolling / PEI: :doc:`usage_window_opf`.

Adding an electrolyser
----------------------

Elements attach to any **AC** or **DC** bus
(:attr:`~pyflow_acdc.Node_AC.connected_electrolyser`,
:attr:`~pyflow_acdc.Node_DC.connected_electrolyser`,
:attr:`~pyflow_acdc.Grid.electrolysers`). ``analyse_grid`` sets
``grid.H2`` when the list is non-empty.

- Active power ``P_electrolyser`` is a **load** (subtracted from nodal injection).
- On **AC**, optional reactive compensation via ``Q_min_MVAR`` / ``Q_max_MVAR``
  in **nonlinear** OPF (generation convention: positive ``Q`` injects vars).
- On **DC**, ``Q`` is fixed at zero.
- Inventory ``mass_H2`` is in **kg** (``H2_mass_max``, ``H2_mass_initial``,
  optional ``H2_mass_final`` for **window / rolling OPF** when a terminal mass
  target is set).
- ``empty_tank_cycle`` (``None`` or int ``N >= 1``) controls **out-of-opt**
  tank resets (between solves, not a Pyomo constraint):

  - **Myopic** ``ts_acdc_opf``: ``None`` → never empty (mass carries; production
    stops when ``H2_mass_max`` binds). ``N`` → empty after every ``N`` solved
    hours.
  - **Rolling** ``rolling_window_nl_opf``: ``None`` → empty at every commit
    window boundary. ``N`` → empty at the first commit end hour
    ``>= k·N`` (window boundary at or past each cycle multiple).
- ``h2_price`` (EUR/kg, default ``0``), static or ``TSType.H2_PRICE`` series:
  with ``ObjRule={'H2_sale': 1, ...}`` a zero price contributes nothing.
- Rolling horizons: :func:`~pyflow_acdc.rolling_window_nl_opf` (1-based
  ``start``/``end`` like ``ts_acdc_opf``).

Linear production each hour:

.. math::

   h = b_h\, P_e\, S_{\mathrm{base}}\, \Delta t + c_h

with ``c_h`` applied **every** frame (Useche-Arteaga et al. 2026).

The example below is taken from ``pyflow_tests/doc_examples/hydrogen/`` and
executed by ``test_docs_hydrogen.py``.

.. literalinclude:: ../pyflow_tests/doc_examples/hydrogen/01_add_electrolyser.py
   :language: python
   :lines: 2-

Running OPF
-----------

- **Snapshot** ``optimal_pf``: one inventory step from ``H2_mass_initial``; no
  terminal ``H2_mass_final``.
- **Linearised AC** :func:`~pyflow_acdc.optimal_l_pf`: P + mass; optional
  ``ObjRule['H2_sale']``.
- **Coupled** :func:`~pyflow_acdc.window_nl_opf`: parent chain across frames;
  ``H2_mass_final`` enforced on the last frame when set.
- **Myopic** :func:`~pyflow_acdc.ts_acdc_opf`: inventory carries hour-to-hour
  within ``H2_mass_max``. Economics use ``ObjRule['H2_sale']``. Tank empties
  follow ``empty_tank_cycle`` (``None`` = never; ``N`` = every ``N`` hours),
  applied between solves. ``H2_mass_final`` is not enforced in myopic OPF.
- **Rolling** :func:`~pyflow_acdc.rolling_window_nl_opf`: same inventory model
  inside each window; empties between windows per ``empty_tank_cycle``.
- Results: ``Results.ext_electrolyser()`` (snapshot) and
  ``Results.hydrogen_window()`` after a window solve.

.. code-block:: python

    pyf.ts_acdc_opf(
        grid,
        ObjRule={"Energy_cost": 1, "H2_sale": 1},
        solver="ipopt",
    )

**References**

.. [#useche2026] M. Useche-Arteaga, P. Gebraad, V. Lacerda, M. Cheah-Mane, and O. Gomis-Bellmunt: *Optimizing the operation of energy islands with predictive nonlinear programming -- a case study based on the Princess Elisabeth Energy Island*, Wind Energy Science, 11(2), 349--372, 2026, https://doi.org/10.5194/wes-11-349-2026
