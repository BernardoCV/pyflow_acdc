Usage Guide
===========

This package was designed to facilitate the management of node and branch data in Excel, allowing users to easily convert the data into CSV format for seamless import into Python for calculations. For a detailed guide on this process, I highly recommend referring to :ref:`csv_import`.

Alternatively, you can also construct your grid directly in Python. Below is the fundamental approach to creating a grid.

The Python blocks on this page are included from ``pyflow_tests/doc_examples/usage/`` and executed by ``test_docs_usage.py``.

Creating a Grid
---------------
This is the basic way to create a grid. This grid is the same as running MATACDC case5_stagg and case5_stagg_MTDC [1]_.

.. themed-figure:: stagg5
   :alt: Case 5 Stagg
   :align: center

   Case 5 Stagg Grid


.. literalinclude:: ../pyflow_tests/doc_examples/usage/creating_grid.py
   :language: python
   :lines: 2-

Adding Components
-----------------

Grids can also be built in the opposite order, creating the core grid first, then adding elements.

.. literalinclude:: ../pyflow_tests/doc_examples/usage/add_components.py
   :language: python
   :lines: 2-

Running a Power Flow
--------------------
Examples of running a power flow...

.. literalinclude:: ../pyflow_tests/doc_examples/usage/power_flow.py
   :language: python
   :lines: 2-

.. _usage_opf:

Running an Optimal Power Flow
-----------------------------
To run this, you need to have the OPF optional installed. This includes the following packages:

- pyomo
- ipopt


**Quick Example**

.. literalinclude:: ../pyflow_tests/doc_examples/usage/opf_quick.py
   :language: python
   :lines: 2-

It is important that for optimal power flow generators are added to the grid before running.


**Detailed Example**

Taking the Case 5 from the IEEE PES Power Grid Library [2]_.

.. literalinclude:: ../pyflow_tests/doc_examples/usage/opf_detailed.py
   :language: python
   :lines: 2-

Available test cases:
^^^^^^^^^^^^^^^^^^^^^^

Preloaded grids live under ``pyflow_acdc/example_grids/`` and are exposed as
``pyf.cases`` when you ``import pyflow_acdc as pyf``. Each factory returns a
``(grid, results)`` tuple, for example ``grid, res = pyf.cases['PEI_grid']()``.

**Power Flow** (``pyflow_acdc/example_grids/PF/``):

* ``pyf.cases['CigreB4_ACDC']()``
* ``pyf.cases['PEI_grid']()``

**Optimal Power Flow** (``pyflow_acdc/example_grids/OPF/``):

* ``pyf.cases['case118']()``
* ``pyf.cases['case1888rte']()``
* ``pyf.cases['case3120sp_acdc']()``
* ``pyf.cases['case_ACTIVSg2000']()``
* ``pyf.cases['NS_MTDC']()``
* ``pyf.cases['NS_SI']()``
* ``pyf.cases['NS_SII']()``
* ``pyf.cases['pglib_opf_case14_ieee']()``
* ``pyf.cases['pglib_opf_case24_ieee_rts']()``
* ``pyf.cases['pglib_opf_case300_ieee']()``
* ``pyf.cases['pglib_opf_case588_sdet_acdc']()``
* ``pyf.cases['pglib_opf_case5_pjm']()``
* ``pyf.cases['pglib_opf_hvdc_case67']()``
* ``pyf.cases['Stagg5MATACDC']()``

**Transmission Expansion Planning** (``pyflow_acdc/example_grids/TEP/``):

* ``pyf.cases['case118_TEP']()``
* ``pyf.cases['case118_TEP_DC']()``
* ``pyf.cases['case24_3zones_acdc']()``
* ``pyf.cases['case39']()``
* ``pyf.cases['case39_acdc']()``
* ``pyf.cases['case_118_TEP_benchmark']()``
* ``pyf.cases['case_ACTIVSg500']()``
* ``pyf.cases['Texas7k_20210804']()``

**Wind Array** (``pyflow_acdc/example_grids/Wind_Array/``):

* ``pyf.cases['anholt']()``
* ``pyf.cases['barrow']()``
* ``pyf.cases['Borssele_3_and_4']()``
* ``pyf.cases['nordsee_one']()``



    

**References**


.. [1] J. Beerten and R. Belmans, "MatACDC - an open source software tool for steady-state analysis and operation of HVDC grids," 11th IET International Conference on AC and DC Power Transmission, Birmingham, 2015, pp. 1-9, doi: 10.1049/cp.2015.0061. keywords: {Steady-state analysis;HVDC grids;AC/DC systems;power flow modelling},

.. [2] https://github.com/power-grid-lib/pglib-opf


