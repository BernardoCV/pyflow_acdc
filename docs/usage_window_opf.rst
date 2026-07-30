Coupled window NL OPF
=====================

:func:`~pyflow_acdc.window_nl_opf` solves a **multi-hour nonlinear OPF** with
coupled storage SoC and (when present) H₂ inventory across frames.
:func:`~pyflow_acdc.rolling_window_nl_opf` chains successive windows with
SoC / H₂ carry-over between commits.

Add BESS / electrolysers first — see :doc:`api/modelling_storage_hydrogen`.
Interactive plots: :doc:`api/dash`.

Seasonal data for the Princess Elisabeth Island (PEI) case lives under
``examples/PEI_BESS/<Season>/`` (local checkout, else GitHub ``main``).
Build with case flags ``storage``, ``hydrogen``, and
``data='season_comparison'`` / ``'full'``::

    https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/PEI_BESS/

API
---

.. autofunction:: pyflow_acdc.window_nl_opf

.. autofunction:: pyflow_acdc.rolling_window_nl_opf

PEI season-compare + Dash
-------------------------

Solve each season as its own 24 h window, attach compare tables, then open the
season-compare dashboard (Overlay / Split layouts, Power and other families).

Requires ``ipopt`` and ``pip install "pyflow-acdc[OPF,Dash]"``.

The example lives in ``pyflow_tests/doc_examples/window_opf/``. Docs CI only
smoke-loads the seasonal CSVs (four IPOPT windows are too heavy for the suite).
Interactively prefer ``pyf.run_dash(grid)`` over ``create_season_compare_dash_app``.

.. literalinclude:: ../pyflow_tests/doc_examples/window_opf/01_pei_season_compare_dash.py
   :language: python
   :lines: 9-

Once running, open http://127.0.0.1:8050/ .

.. themed-figure:: pei_season_compare_dash
   :width: 100%
   :alt: PEI BESS H2 season-compare Dash

   Season-compare Dash: Power (source) and SoC across Spring–Winter (split layout).

Single-window solve
-------------------

For one seasonal (or concatenated) window without season-compare::

    grid, _ = pyf.cases["PEI_grid"](
        include_countries=["GB", "DK"],
        storage=True,
        hydrogen=True,
        data="season_comparison",
        seasons=("Autumn",),
    )
    pyf.window_nl_opf(
        grid,
        start=0,
        end=23,
        ObjRule={"Energy_cost": 1},
        solver="ipopt",
    )
    pyf.run_dash(grid)

Build-only PEI example (no solve):

.. literalinclude:: ../pyflow_tests/doc_examples/storage/02_window_nl_opf_pei.py
   :language: python
   :lines: 2-

Rolling window
--------------

Indexing uses 1-based ``start`` / ``end`` like ``ts_acdc_opf``. Tank empties
between windows follow each electrolyser's ``empty_tank_cycle``:

- ``None`` → empty at every commit window boundary
- ``N`` → empty at the first commit end hour ``>= k·N`` (boundary at or past
  each cycle multiple)

Coupled windows keep hard ``soc_initial`` / ``soc_final`` and optional
``H2_mass_final`` on the last frame when set.

.. code-block:: python

    pyf.rolling_window_nl_opf(
        grid,
        start=1,
        end=48,
        window_size=24,
        ObjRule={"Energy_cost": 1},
        solver="ipopt",
    )

Myopic TS (related)
-------------------

Sequential :func:`~pyflow_acdc.ts_acdc_opf` carries BESS SoC and H₂ inventory
hour-to-hour (no coupled horizon). Soft SoC reference via
``ObjRule['SoC_deviation']``; H₂ economics via ``ObjRule['H2_sale']``.
Myopic tank empties: ``empty_tank_cycle=None`` never empties; ``N`` empties
after every ``N`` solved hours. ``H2_mass_final`` is not enforced in myopic OPF.

.. code-block:: python

    pyf.ts_acdc_opf(
        grid,
        ObjRule={"Energy_cost": 1, "SoC_deviation": 10, "H2_sale": 1},
        solver="ipopt",
    )
