Multi-period Transmission Expansion Planning (MP TEP and MP+MS TEP)
=====================================================================

Requires ``pip install pyflow-acdc[OPF]`` plus Bonmin/Ipopt (see :doc:`installation`).

This page covers **multi-period TEP (MP TEP)** and **multi-period multi-scenario
TEP (MP+MS TEP)** over an investment horizon. Static TEP and single-snapshot MS
TEP are in :doc:`usage_tep`. Grid fundamentals are in :doc:`usage`.

After grid setup (step 1) and planning CSVs (step 2), pick **one** driver for step 3:
:func:`~pyflow_acdc.multi_period_transmission_expansion` (MP TEP),
:func:`~pyflow_acdc.sequential_STEP`, or MP+MS via
:func:`~pyflow_acdc.multi_period_MS_TEP` / :func:`~pyflow_acdc.sequential_MS_STEP`.
Then export results (step 4).

Workflow
--------

1. Create the grid
^^^^^^^^^^^^^^^^^^

Grid creation for multi-period TEP includes topology, generators, renewables,
**and** marking which elements are expandable — in this order:

- nodes and branches
- :func:`~pyflow_acdc.add_gen` (with ``installation_cost``)
- :func:`~pyflow_acdc.add_RenSource` (with ``base_cost``)
- expandable table (line rows, then one row per generator and renewable) →
  :func:`~pyflow_acdc.expand_elements_from_pd`

For the bundled example:

.. code-block:: python

   import pyflow_acdc as pyf

   grid, res = pyf.cases["case24_MP"]()

When building a custom case, follow the same order as in :doc:`usage_tep` step 1
(or :doc:`usage` for topology and generators only).

2. Attach planning CSVs
^^^^^^^^^^^^^^^^^^^^^^^^^

Load CSVs in this order (same as ``case24_MP_TEP`` / ``case24_seq_STEP``):

**a) Investment and decommission series** — :func:`~pyflow_acdc.add_inv_series`
first (load multipliers, planned installations/decommissions, import expansion):

.. code-block:: python

   import importlib.util
   from pathlib import Path

   case_path = (
       Path(pyf.__file__).resolve().parent
       / "example_grids"
       / "TEP"
       / "case24_MP.py"
   )
   spec = importlib.util.spec_from_file_location("case24_MP", case_path)
   mod = importlib.util.module_from_spec(spec)
   spec.loader.exec_module(mod)

   inv_csv = mod._resolve_example_path("case24_MP_TEP_inv_series_10.csv")
   pyf.add_inv_series(grid, inv_csv)

Each column maps to ``investment_decisions`` on linked elements (``Load``,
``planned_decomision``, ``planned_installation``, etc.). CSVs ship under
``examples/Case24_MP/`` in a repo checkout.

**b) Generation-type mix limits** — :func:`~pyflow_acdc.add_gen_mix_limits`
second:

.. code-block:: python

   mix_csv = mod._resolve_example_path("case24_MP_TEP_gen_mix_limits.csv")
   pyf.add_gen_mix_limits(grid, mix_csv)

Each column is one period; row 0 is the generation type, row 1 the limit kind,
rows 2+ the numeric bounds.

3. Run
^^^^^^

Pick **one** of the following (same CSV setup from step 2 unless noted).

**MP TEP** — one-shot multi-period MINLP, all periods in one model:

.. literalinclude:: ../pyflow_tests/doc_examples/tep_mp/01_multi_period_tep_case24.py
   :language: python
   :lines: 2-

For production solves use ``solver="bonmin"`` and
``solver_options={"bonmin.algorithm": "B-Hyb", "bonmin.nlp_failure_behavior": "fathom"}``.
``ObjRule`` is ``{"Energy_cost": 1}``; ``n_years`` is ``10``. Check
``solver_stats["solution_found"]`` before exporting.

**Linear MP TEP** — AC-only MILP counterpart
(:func:`~pyflow_acdc.linear_multi_period_transmission_expansion`, default
``solver="gurobi"``). Same CSV setup; see :doc:`api/L_models`.

**Sequential STEP** — one static :func:`~pyflow_acdc.transmission_expansion`
per period (grid state carried forward):

.. code-block:: python

   run_results = pyf.sequential_STEP(
       grid=grid,
       inv_data=inv_csv,
       mix_data=mix_csv,
       n_years=mod.DEFAULT_N_YEARS,
       Hy=8760,
       discount_rate=0.02,
       ObjRule=mod.DEFAULT_OBJ_RULE,
       solver="bonmin",
       tee=False,
       obj_scaling=1e9,
       solver_options={
           "bonmin.nlp_failure_behavior": "fathom",
           "bonmin.algorithm": "B-BB",  # recommended for sequential STEP
       },
   )

**MP+MS TEP** — multi-period planning with clustered operating scenarios
(cases with time series). For the North Sea bundled case use
``pyf.cases["NS_MTDC_2025"](years_data="23,24", expandable="mp")`` — CSVs and
precomputed clusters are under ``examples/North_Sea_grid_data/``. Use
``expandable="step"`` for single-period MS TEP without MP investment series (see
:doc:`usage_tep`). Then call :func:`~pyflow_acdc.multi_period_MS_TEP` or
:func:`~pyflow_acdc.sequential_MS_STEP`:

**Cluster time series** — run :func:`~pyflow_acdc.cluster_TS` (or
:func:`~pyflow_acdc.run_clustering_analysis_and_plot` for exploratory work), or
reload a saved result via ``precomputed_clusters_path`` in ``clustering_options``
(see :func:`~pyflow_acdc.load_precomputed_clusters_to_grid`):

.. code-block:: python

   from pyflow_tests.test_constants import north_sea_ms_clustering_options

   grid, _ = pyf.cases["NS_MTDC_2025"](years_data="23,24", expandable="mp")

   clustering_options = north_sea_ms_clustering_options()
   # points to examples/North_Sea_grid_data/clusters_kmeans_medoids_k4.json

Pass ``clustering_options`` to :func:`~pyflow_acdc.multi_period_MS_TEP` or
:func:`~pyflow_acdc.sequential_MS_STEP``. For price-zone OPEX use
``ObjRule={"PZ_cost_of_generation": 1}`` (see :doc:`api/opf`). See
:doc:`api/clustering`.

.. literalinclude:: ../pyflow_tests/doc_examples/tep_mp/02_multi_period_multi_scenario_dynamic_tep.py
   :language: python
   :lines: 2-

4. Results
^^^^^^^^^^

After a feasible solve, the **grid is updated in place** — line multiplicities,
generator counts, and decommission flags reflect the expansion plan.

**Solver return values** (MP TEP):

- ``model``, ``model_results``, ``timing_info``, ``solver_stats``
- ``grid.OPF_obj`` — discounted operational cost components
- element ``np_gen`` / ``np_line`` — chosen build-out

**Export and post-processing** (typical after MP TEP):

.. code-block:: python

   if not solver_stats.get("solution_found", False):
       raise RuntimeError(solver_stats.get("solver_message", "no solution"))

   pyf.export_and_save_inv_period_svgs(grid, folder_name="case24_MP_TEP")
   res.pyomo_model_results(model, solver_stats=solver_stats, model_results=model_results)
   res.all(export_type="excel", file_name="case24_MP_TEP", export_location="case24_MP_TEP")

   pyf.run_opf_for_all_investment_periods(
       grid,
       ObjRule=mod.DEFAULT_OBJ_RULE,
       obj_scaling=1e9,
       export_excel=True,
       export_location="case24_MP_TEP",
   )

``run_opf_for_all_investment_periods`` re-solves OPF for each investment period
on the expanded grid and exports per-period operating results.

For **time-series OPF** on one built-out period (clustered or hourly), use
:func:`~pyflow_acdc.run_ts_opf_for_investment_period` — it applies the period
state, calls :func:`~pyflow_acdc.ts_acdc_opf`, and writes Excel via
:func:`~pyflow_acdc.results_ts_opf`.

After MP+MS TEP, export scenario tables with
:func:`~pyflow_acdc.export_TEP_multiScenario_results_to_excel` (deprecated alias
``export_TEP_TS_results_to_excel``):

.. code-block:: python

   pyf.export_TEP_multiScenario_results_to_excel(grid, "NS_MP_MS_results.xlsx")

**Sequential STEP** returns a ``run_results`` dict keyed by period. Set
``export_steps=True`` and ``export_dir=...`` for CSV summaries per step.

Example cases
-------------

* ``pyf.cases['case24_MP']()``
* ``pyf.cases['NS_MTDC_2025'](years_data="23,24", expandable="mp")`` (MP+MS;
  data in ``examples/North_Sea_grid_data/``)

See :doc:`usage` for the full catalogue.

See :doc:`api/tep_dynamic` and :doc:`api/sequential_step` for MP+MS signatures.

Related API pages
-----------------

- :doc:`api/tep_dynamic` — :func:`~pyflow_acdc.multi_period_transmission_expansion`,
  :func:`~pyflow_acdc.multi_period_MS_TEP`,
  :func:`~pyflow_acdc.export_and_save_inv_period_svgs`,
  :func:`~pyflow_acdc.run_opf_for_all_investment_periods`,
  :func:`~pyflow_acdc.run_ts_opf_for_investment_period`,
  :func:`~pyflow_acdc.export_TEP_multiScenario_results_to_excel`
- :doc:`api/sequential_step` — :func:`~pyflow_acdc.sequential_STEP`,
  :func:`~pyflow_acdc.sequential_MS_STEP`
- :doc:`api/clustering` — :func:`~pyflow_acdc.cluster_TS`,
  :func:`~pyflow_acdc.load_precomputed_clusters_to_grid`
