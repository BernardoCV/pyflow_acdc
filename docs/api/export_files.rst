Export Files Module
====================

This page has been pre-filled with the functions that are available in the Export_files module by AI, please check the code for more details.

This module provides functions for exporting grid data to various file formats.

functions are found in pyflow_acdc.Export_files

Grid Data Export
----------------

generate_loading_code
^^^^^^^^^^^^^^^^^^^^^

.. py:function:: generate_loading_code(grid, file_name)

   Generates Python code to recreate the grid.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid to export
        - Required
        - -
      * - ``file_name``
        - str
        - Function name
        - Required
        - -

   Generates code for:

   - Base power
   - AC/DC nodes and lines
   - Converters
   - Price zones
   - Generators
   - Renewable sources

   **Example**

   .. code-block:: python

       code = pyf.generate_loading_code(grid, "create_test_grid")

MATLAB Export
--------------

save_grid_to_matlab
^^^^^^^^^^^^^^^^^^^

.. py:function:: save_grid_to_matlab(grid, file_name, folder_name=None, dcpol=2)

   Exports grid to MATLAB format.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid to export
        - Required
        - -
      * - ``file_name``
        - str
        - Output filename
        - Required
        - -
      * - ``folder_name``
        - str
        - Output folder
        - None
        - -
      * - ``dcpol``
        - int
        - DC polarity
        - 2
        - -

   Exports:

   - Bus data
   - Branch data
   - Generator data
   - DC bus data
   - DC branch data
   - Converter data
   - Cost data

Data Dictionary Creation
------------------------

create_dictionaries
^^^^^^^^^^^^^^^^^^

.. py:function:: create_dictionaries(grid)

   Creates dictionaries of grid component data.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid to process
        - Required
        - -

   Creates dictionaries for:

   - AC/DC nodes
   - AC/DC lines
   - Converters
   - Price zones
   - Generators
   - Renewable sources
   - Base power

Grid Data Collection
---------------------

gather_grid_data
^^^^^^^^^^^^^^^^^

.. py:function:: gather_grid_data(grid)

   Collects all grid component data.

   Returns dictionaries containing:

   - AC node data
   - AC line data
   - DC node data
   - DC line data
   - Converter data
   - Generator data
   - Generator cost data

   Data is formatted for MATLAB export.
