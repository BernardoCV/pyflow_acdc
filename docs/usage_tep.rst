Transmission Expansion Planning (TEP and MS TEP)
================================================

Requires ``pip install pyflow-acdc[OPF]`` plus Bonmin/Ipopt (see :doc:`installation`).

This page covers **static TEP** (one investment snapshot, one operating state)
and **multi-scenario TEP (MS TEP)** (one expansion plan, several clustered
operating scenarios). Multi-period planning is in :doc:`usage_mp_tep`. Grid
fundamentals are in :doc:`usage`.

After expandable grid setup (step 1), pick **one** driver for step 2:
:func:`~pyflow_acdc.transmission_expansion` (static TEP) or
:func:`~pyflow_acdc.multi_scenario_TEP` (MS TEP).

Workflow
--------

1. Prepare an expandable grid
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- nodes and branches
- :func:`~pyflow_acdc.add_gen` (with ``installation_cost``)
- :func:`~pyflow_acdc.add_RenSource` (with ``base_cost``)
- expandable table → :func:`~pyflow_acdc.expand_elements_from_pd`

  Candidate line types — :ref:`expandable <AC_expandable_branch>`,
  :ref:`reconducting <AC_reconducting_branch>`, and
  :ref:`conductor size selection (CSS) <AC_conductor_size_selection>` — are
  modelled in :doc:`api/modelling_ac`.

Bundled examples: ``pyf.cases["case118_TEP"]()`` (static TEP),
``pyf.cases["NS_MTDC_2025"](expandable="step")`` (MS TEP with time series).
Grid CSVs and precomputed clusters live under ``examples/North_Sea_grid_data/``
(see that folder's README for ``expandable="mp"`` vs ``"step"``).

For MS TEP on price-zone cases, set social cost in ``ObjRule`` with the key
``"PZ_cost_of_generation"`` (not ``"Price_Zones"``). See :doc:`api/opf` for all
objective component keys.

2. Run
^^^^^^

Pick **one** of the following.

**Static TEP** — one investment snapshot, one operational state:

.. literalinclude:: ../pyflow_tests/doc_examples/tep/01_running_one_state_transmission_expansion_planning.py
   :language: python
   :lines: 2-

The static TEP example uses ``solver="ipopt"`` for fast doc tests. For production
MINLP solves, prefer ``solver="bonmin"``.

**MS TEP** — one expansion plan, several clustered operating scenarios.
Attach time series with :func:`~pyflow_acdc.add_TimeSeries` (see :doc:`api/ts_mod`),
set ``clustering_options`` (see :doc:`api/clustering`), then call
:func:`~pyflow_acdc.multi_scenario_TEP`:

.. literalinclude:: ../pyflow_tests/doc_examples/tep/02_multi_scenario_tep.py
   :language: python
   :lines: 2-

The MS TEP example also uses ``solver="ipopt"`` and ``build_only=True`` in doc
tests; prefer ``solver="bonmin"`` for production solves with binary expansion
decisions, and omit ``build_only`` for a full solve.

**Inspecting TEP results** — after a static solve, read the expansion table
(:meth:`~pyflow_acdc.Results_class.Results.tep_n`) and the normalised objective
breakdown (:meth:`~pyflow_acdc.Results_class.Results.tep_norm`):

.. literalinclude:: ../pyflow_tests/doc_examples/tep/03_inspecting_tep_results.py
   :language: python
   :lines: 2-

**Reconductoring (REC)** — instead of building new parallel
circuits, an existing AC line can be *reconductored*: its conductor is replaced by
a higher-capacity one on the same right-of-way. Candidates must be marked **before**
calling the TEP driver.

Mark a batch of candidates from a table with
:func:`~pyflow_acdc.repurpose_element_from_pd`. The first column is the existing
line's ``Line_id``; the remaining columns describe the upgraded conductor and are
case-insensitive and optional (``r_new``, ``x_new``, ``g_new``, ``b_new``,
``MVA_rating_new``, ``Life_time``, ``base_cost``):

.. code-block:: python

   import pandas as pd
   import pyflow_acdc as pyf

   upgradable = pd.DataFrame([
       {'Line_id': '1-2', 'r_new': 0.00173, 'x_new': 0.00927, 'b_new': 0.3074,
        'MVA_rating_new': 300.0, 'base_cost': 0.9},
       # ... one row per reconductoring candidate ...
   ])
   pyf.repurpose_element_from_pd(grid, upgradable)

To convert a single line, call
``pyf.change_line_AC_to_reconducting(grid, line_name, r_new, x_new, g_new, b_new,
MVA_rating_new, Life_time, base_cost)``.

**Two admittances per element.** Each candidate is stored as a ``rec_Line_AC`` that
keeps **two** branch-admittance matrices for the same corridor:

- ``Ybus_branch`` — the *existing* conductor, built from the line's ``r, x, g, b``;
- ``Ybus_branch_new`` — the *reconductored* conductor, built once at setup from the
  ``*_new`` parameters.

A per-candidate binary flag ``rec_branch`` selects between them: ``create_Ybus_AC``
stamps ``Ybus_branch_new`` into the system matrix ``Ybus_AC_full`` when
``rec_branch`` is ``True`` and ``Ybus_branch`` otherwise, and the line's thermal
limit switches between ``MVA_rating`` and ``MVA_rating_new`` accordingly. In TEP
``rec_branch`` is a binary decision variable the solver optimises, and the objective
adds the reconductoring cost term (:math:`\Psi_{rec}`, see :doc:`api/tep`). Because
of these binaries the problem is a MINLP, so the example below uses ``bonmin``:

.. literalinclude:: ../pyflow_tests/doc_examples/tep/04_reconductoring_tep.py
   :language: python
   :lines: 2-

Linear TEP
----------

For faster studies and large sweeps, :func:`~pyflow_acdc.linear_transmission_expansion`
solves the MILP counterpart of static TEP (including reconductoring). It falls back
to ``build_only`` when no MIP solver is available. See :doc:`api/L_models` for the
full linear API (linear TEP and the linearised AC OPF counterpart
:func:`~pyflow_acdc.optimal_l_pf`).

.. literalinclude:: ../pyflow_tests/doc_examples/L_models/01_linear_transmission_expansion.py
   :language: python
   :lines: 2-

.. literalinclude:: ../pyflow_tests/doc_examples/L_models/02_linear_reconductoring.py
   :language: python
   :lines: 2-

Example cases
-------------

* ``pyf.cases['case118_TEP']()``
* ``pyf.cases['case39']()``
* ``pyf.cases['case39_acdc']()``
* ``pyf.cases['NS_MTDC_2025'](years_data="23,24", expandable="step")`` (MS TEP;
  data in ``examples/North_Sea_grid_data/``)

See :doc:`usage` for the full catalogue.

Related API
-----------

- :doc:`api/tep`
- :doc:`api/clustering`
