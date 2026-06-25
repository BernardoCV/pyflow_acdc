Wind Array Sizing
=================

Requires ``[OPF]`` for Pyomo CSS (optional Gurobi) or ``[LINEAR_ARRAY]`` for
OR-Tools route MIP and CSS (see :doc:`installation`).

Workflow
--------

1. Build a candidate inter-array grid from a turbine graph with
   :func:`~pyflow_acdc.create_grid_from_turbine_graph` (see :doc:`api/grid`),
   or load a bundled case such as ``pyf.cases['alpha_ventus']()``.
2. Run :func:`~pyflow_acdc.sequential_CSS` — route MIP, then cable sizing on the
   fixed topology each iteration. Prefer ``enable_cable_types=False`` (sequential
   methodology) over a joint route+CSS MIP.

.. literalinclude:: ../pyflow_tests/doc_examples/wf_array/01_sequential_cable_sizing_css.py
   :language: python
   :lines: 2-

Example cases
-------------

* ``pyf.cases['alpha_ventus']()``
* ``pyf.cases['anholt']()``
* ``pyf.cases['barrow']()``
* ``pyf.cases['Borssele_3_and_4']()``
* ``pyf.cases['moray_east']()``
* ``pyf.cases['nordsee_one']()``
* ``pyf.cases['westermost_rough']()``

See :doc:`usage` for the full catalogue.

Related API
-----------

- :doc:`api/wf_array`
