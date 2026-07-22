Coupled window NL OPF
=====================

:func:`~pyflow_acdc.window_nl_opf` solves a **multi-hour nonlinear OPF** with
coupled storage SoC and (when present) H₂ inventory across frames.

Add BESS / electrolysers first — see :doc:`usage_storage` and
:doc:`usage_hydrogen`. Interactive plots: :doc:`api/dash`.

Seasonal data for the Princess Elisabeth Island (PEI) case lives under
``examples/PEI_BESS/<Season>/``. Loaders in ``pyflow_tests._bess_h2_pei_data``
prefer a local checkout, otherwise fetch from GitHub ``main``::

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

Single-window solve
-------------------

For one seasonal (or concatenated) window without season-compare::

    from pyflow_tests._bess_h2_pei_data import (
        PEI_OBJ_RULE, WINDOW_START, build_pei_bess_h2_grid, window_end,
    )

    seasons = ("Autumn",)
    grid = build_pei_bess_h2_grid(seasons=seasons)
    pyf.window_nl_opf(
        grid,
        start=WINDOW_START,
        end=window_end(seasons),
        ObjRule=PEI_OBJ_RULE,
        solver="ipopt",
    )
    pyf.run_dash(grid)

See also ``pyflow_tests/doc_examples/storage/02_window_nl_opf_pei.py`` (build-only).
