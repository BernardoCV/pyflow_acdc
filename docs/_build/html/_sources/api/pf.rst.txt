Power Flow Module
=================

This page has been pre-filled with the functions that are available in the PF module by AI, please check the code for more details.

This module provides functions for AC, DC and AC/DC power flow analysis.

functions are found in pyflow_acdc.ACDC_PF

AC Power Flow
-------------

.. py:function:: AC_PowerFlow(grid, tol_lim=1e-8, maxIter=100)

   Performs AC power flow calculation.

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
        - Grid to analyze
        - Required
        - -
      * - ``tol_lim``
        - float
        - Convergence tolerance
        - 1e-8
        - -
      * - ``maxIter``
        - int
        - Maximum iterations
        - 100
        - -
      * - Returns
        - float
        - Computation time
        - -
        - seconds

   **Example**

   .. code-block:: python

       time = pyf.AC_PowerFlow(grid)

DC Power Flow
-------------

.. py:function:: DC_PowerFlow(grid, tol_lim=1e-8, maxIter=100)

   Performs DC power flow calculation.

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
        - Grid to analyze
        - Required
        - -
      * - ``tol_lim``
        - float
        - Convergence tolerance
        - 1e-8
        - -
      * - ``maxIter``
        - int
        - Maximum iterations
        - 100
        - -
      * - Returns
        - float
        - Computation time
        - -
        - seconds

   **Example**

   .. code-block:: python

       time = pyf.DC_PowerFlow(grid)

Sequential AC/DC Power Flow
--------------------------

.. py:function:: ACDC_sequential(grid, tol_lim=1e-8, maxIter=20, change_slack2Droop=False, QLimit=False)

   Performs sequential AC/DC power flow calculation.

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
        - Grid to analyze
        - Required
        - -
      * - ``tol_lim``
        - float
        - Convergence tolerance
        - 1e-8
        - -
      * - ``maxIter``
        - int
        - Maximum iterations
        - 20
        - -
      * - ``change_slack2Droop``
        - bool
        - Change slack to droop control
        - False
        - -
      * - ``QLimit``
        - bool
        - Enable converter Q limits
        - False
        - -
      * - Returns
        - float
        - Computation time
        - -
        - seconds

   The sequential solver performs the following steps:
   
   1. AC power flow
   2. Update converter power flows
   3. DC power flow
   4. Repeat until convergence

   **Example**

   .. code-block:: python

       time = pyf.ACDC_sequential(grid, QLimit=True)

   **Notes**

   - For grids with both AC and DC components, sequential solver is recommended
   - Converter Q limits can be enabled to respect reactive power constraints
   - Droop control can be enabled for DC voltage regulation

