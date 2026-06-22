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
``pyf.cases["NS_MTDC"]()`` (MS TEP with time series).

2. Run
^^^^^^

Pick **one** of the following.

**Static TEP** — one investment snapshot, one operational state:

.. literalinclude:: ../pyflow_tests/doc_examples/tep/01_running_one_state_transmission_expansion_planning.py
   :language: python
   :lines: 2-

**MS TEP** — one expansion plan, several clustered operating scenarios.
Attach time series with :func:`~pyflow_acdc.add_TimeSeries` (see :doc:`api/ts_mod`),
set ``clustering_options`` (see :doc:`api/clustering`), then call
:func:`~pyflow_acdc.multi_scenario_TEP`:

.. literalinclude:: ../pyflow_tests/doc_examples/usage_tep/01_multi_scenario_tep.py
   :language: python
   :lines: 2-

The MS TEP example uses ``solver="ipopt"`` so doc tests finish quickly. For
production solves with binary expansion decisions, prefer ``solver="bonmin"``.

Example cases
-------------

* ``pyf.cases['case118_TEP']()``
* ``pyf.cases['case39']()``
* ``pyf.cases['case39_acdc']()``
* ``pyf.cases['NS_MTDC']()`` (MS TEP with time series)

See :doc:`usage` for the full catalogue.

Related API
-----------

- :doc:`api/tep`
- :doc:`api/clustering`
