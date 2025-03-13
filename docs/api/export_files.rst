Export Files Module
====================

This module provides functions for exporting grid data to various file formats.

functions are found in pyflow_acdc.Export_files

Grid Data Export
----------------

.. py:function:: save_grid_to_file(grid, file_name, folder_name=None):

   Exports grid data to a Python file. The file can be used to load the grid data into the model. Files in the example_grids folder are examples of the files that can be generated. Any file added to this folder will be automatically loaded when the pyflow_acdc package is imported.

MATLAB Export
--------------

save_grid_to_matlab
^^^^^^^^^^^^^^^^^^^

.. py:function:: save_grid_to_matlab(grid, file_name, folder_name=None, dcpol=2)

   Exports grid to MATLAB format. It is important to note, that for MATACDC format, only one polarity can be chosen for all DC grids.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid to export
        - Required
      * - ``file_name``
        - str
        - Output filename
        - Required
      * - ``folder_name``
        - str
        - Output folder
        - None
      * - ``dcpol``
        - int
        - DC polarity
        - 2

   
