Optimal Power Flow
==================

Requires ``pip install pyflow-acdc[OPF]`` plus Ipopt (see :doc:`installation`).

Start from a grid with generators (:func:`~pyflow_acdc.add_gen`; see :doc:`api/grid_mod`). Set ``ObjRule`` if needed (see :ref:`obj_functions`),
then call :func:`~pyflow_acdc.optimal_pf`.

.. _usage_opf:

Workflow
--------

To run OPF you need the optional packages:

- pyomo
- ipopt

**Quick example**

.. literalinclude:: ../pyflow_tests/doc_examples/usage/opf_quick.py
   :language: python
   :lines: 2-

Generators must be on the grid before running OPF.

**Detailed example**

Case 5 from the IEEE PES Power Grid Library [1]_.

.. literalinclude:: ../pyflow_tests/doc_examples/usage/opf_detailed.py
   :language: python
   :lines: 2-

Example cases
-------------

* ``pyf.cases['case118']()``
* ``pyf.cases['NS_MTDC']()`` — alternate North Sea MTDC topology; attach
  online TS from ``examples/North_Sea_grid_data/`` with
  :func:`~pyflow_acdc.add_TimeSeries`
* ``pyf.cases['NS_MTDC_2025'](years_data="23,24")`` — TEP-oriented North Sea
  case with bundled time-series load from the same CSV folder
* ``pyf.cases['pglib_opf_case5_pjm']()``
* ``pyf.cases['Stagg5MATACDC']()``

See :doc:`usage` for the full catalogue.

Related API
-----------

- :doc:`api/opf`

**References**

.. [1] https://github.com/power-grid-lib/pglib-opf
