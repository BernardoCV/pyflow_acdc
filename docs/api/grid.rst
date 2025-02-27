Grid Creation
============

This module provides functions for creating and manipulating power system grids.

functions are found in pyflow_acdc.Grid_creator

Core Grid Class
--------------

Creating a Grid
^^^^^^^^^^^^^^ 

.. code-block:: python

    import pyflow_acdc as pyf
    grid = pyf.Grid()

Create Grid From Data
--------------------

.. py:function:: Create_grid_from_data(S_base, AC_node_data=None, AC_line_data=None, DC_node_data=None, DC_line_data=None, Converter_data=None, data_in='Real')

   Creates a new grid from pandas DataFrames containing component data.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``S_base``
        - float
        - Base power in MVA
      * - ``AC_node_data``
        - DataFrame
        - AC node pandas or geopandas 
      * - ``AC_line_data``
        - DataFrame
        - AC line pandas or geopandas 
      * - ``DC_node_data``
        - DataFrame
        - DC node pandas or geopandas 
      * - ``DC_line_data``
        - DataFrame
        - DC line pandas or geopandas 
      * - ``Converter_data``
        - DataFrame
        - Converter pandas or geopandas 
      * - ``data_in``
        - str
        - Input data format ('pu' or 'Ohm' if not assumed in Real values)
      * - Returns
        - grid, res
        - Grid and Results objects

   **Example**

   .. code-block:: python

       grid, results = pyf.Create_grid_from_data(100, ac_nodes_df, ac_lines_df)

Create Grid From Matpower
------------------------

.. py:function:: Create_grid_from_mat(matfile)

   Creates a grid from a MATPOWER case file. 

   Load your (.m) matpower case in matlab and save the variable as a .mat file.

   :param str matfile: Path to .mat file
   :return: Grid and Results objects
   :rtype: list[Grid, Results]

   **Example**

   .. code-block:: python

       grid, results = pyf.Create_grid_from_mat("case9.mat")

Extend Grid From Data  
--------------------

.. py:function:: Extend_grid_from_data(grid, AC_node_data=None, AC_line_data=None, DC_node_data=None, DC_line_data=None, Converter_data=None, data_in='Real')

   Extends an existing grid with additional components.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Existing grid to extend
      * - ``AC_node_data``
        - DataFrame
        - AC node specifications to add
      * - ``AC_line_data``
        - DataFrame
        - AC line specifications to add
      * - ``DC_node_data``
        - DataFrame
        - DC node specifications to add
      * - ``DC_line_data``
        - DataFrame
        - DC line specifications to add
      * - ``Converter_data``
        - DataFrame
        - Converter specifications to add
      * - ``data_in``
        - str
        - Input data format ('Real' or 'pu')
      * - Returns
        - Grid
        - Extended grid object

   **Example**

   .. code-block:: python

       pyf.Extend_grid_from_data(grid, new_ac_nodes_df)

Reset All Classes
----------------

.. py:function:: reset_all_class()

   Resets all component class counters.

   **Example**

   .. code-block:: python

       reset_all_class()

Change Base Power
----------------

under development

.. py:function:: change_S_base(grid, Sbase_new)

   Changes the power base of a grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``Sbase_new``
        - float
        - New base power in MVA
      * - Returns
        - Grid
        - Modified grid

   **Example**

   .. code-block:: python

       pyf.change_S_base(grid, 100)

Create Sub Grid
--------------

.. py:function:: create_sub_grid(grid, Area=None, Area_name=None, polygon_coords=None)

   Creates a sub-grid from a larger grid based on area or coordinates. At the moment only works unidirectionally, initial grid is useless after creation of sub-grid. Sub-grid is created as a new grid object. can be created from Area objects, area object names or polygon coordinates.
   
   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Original grid
      * - ``Area``
        - list of Area objects
        - Area object defining sub-grid
      * - ``Area_name``
        - list of str
        - Name of area for sub-grid
      * - ``polygon_coords``
        - polygon coordinates
        - Coordinates defining sub-grid boundary
      * - Returns
        - list[Grid, Results]
        - Sub-grid and Results objects

   **Example**

   .. code-block:: python

       subgrid, results = create_sub_grid(grid, Area_name="Zone1")