.. pyflow acdc documentation master file, created by
   sphinx-quickstart on Thu Feb 27 09:19:48 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PyFlow ACDC Documentation
=========================

Welcome to PyFlow ACDC's documentation!

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage


.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   api/modelling
   api/grid
   api/grid_mod
   api/pf
   api/opf
   api/plotting
   api/ts
   api/ts_clustering

Quick Start
-----------

Basic installation::

    pip install pyflow-acdc

Basic usage::

   import pyflow_acdc as pyf

   #Use pre saved grids to familiarize yourself with the package
   [grid,res]=pyf.PEI_grid()

   pyf.ACDC_sequential(grid,QLimit=False)

   
   res.All()
   
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`