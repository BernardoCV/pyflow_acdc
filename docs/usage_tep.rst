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
