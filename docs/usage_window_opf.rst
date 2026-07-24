Coupled window NL OPF
=====================

:func:`~pyflow_acdc.window_nl_opf` solves a **multi-hour nonlinear OPF** with
coupled storage SoC and (when present) H₂ inventory across frames.

Add BESS / electrolysers first — see :doc:`usage_storage` and
:doc:`usage_hydrogen`. Interactive plots: :doc:`api/dash`.

Seasonal data for the Princess Elisabeth Island (PEI) case lives under
``examples/PEI_BESS/<Season>/`` (local checkout, else GitHub ``main``).
Build with case flags ``storage``, ``hydrogen``, and
``data='season_comparison'`` / ``'full'``::

    https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/PEI_BESS/

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

See also ``pyflow_tests/doc_examples/storage/02_window_nl_opf_pei.py`` (build-only).
