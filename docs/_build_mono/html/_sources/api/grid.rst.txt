Grid Creation
==============    

This module provides functions for creating and manipulating power system grids.

functions are found in pyflow_acdc.grid_creator

Core Grid Class
---------------

.. autoclass:: pyflow_acdc.Grid
   :no-members:

Creating a Grid
^^^^^^^^^^^^^^^ 

.. code-block:: python

    import pyflow_acdc as pyf
    grid = pyf.Grid()

Create Grid From Data
^^^^^^^^^^^^^^^^^^^^^	

A more detailed description of the function can be found in the :doc:`csv_import` page.

.. autofunction:: pyflow_acdc.create_grid_from_data

**Example**

.. code-block:: python

    grid, results = pyf.create_grid_from_data(100, ac_nodes_df, ac_lines_df)

Create Grid From Matpower
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.create_grid_from_mat

**Example**

.. code-block:: python

    grid, results = pyf.create_grid_from_mat("case9.mat")

Create Grid From Turbine Graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.create_grid_from_turbine_graph

Create Grid From Pickle
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.create_grid_from_pickle

Extend Grid From Data  
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.extend_grid_from_data

**Example**

.. code-block:: python

    pyf.extend_grid_from_data(grid, new_ac_nodes_df)

Reset All Classes
^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.initialize_pyflowacdc

**Example**

.. code-block:: python

    pyf.initialize_pyflowacdc()

Change Base Power
^^^^^^^^^^^^^^^^^   

under development

.. autofunction:: pyflow_acdc.grid_creator.change_S_base

**Example**

.. code-block:: python

    pyf.change_S_base(grid, 100)

Create Sub Grid
^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.grid_creator.create_sub_grid

**Example**

.. code-block:: python

    subgrid, results = pyf.create_sub_grid(grid, Area_name="Zone1")