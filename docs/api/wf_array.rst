Wind Farm Array Sizing Module
=============================

This module provides functions for wind farm array sizing based on [1]_.




Sequential Cable Sizing (CSS)
-----------------------------

.. autofunction:: pyflow_acdc.sequential_CSS

   Iteratively alternates between a path selection MIP and a linear/nonlinear OPF-based cable type selection to converge to an efficient array layout. Returns models, a summary of iterations, timing info, solver stats, and the best iteration index.

   .. list-table::
      :widths: 22 10 48 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid with candidate array lines and cable options
        - Required
      * - ``max_turbines_per_string``
        - int
        - Optional cap on per-string turbines (sets MIP flow bound)
        - None
      * - ``limit_crossings``
        - bool
        - Enforce one-active-per-crossing-group
        - True
      * - ``MIP_solver``
        - str
        - Solver for path MIP (e.g., ``glpk``/``gurobi``)
        - 'glpk'
      * - ``CSS_L_solver``
        - str
        - Solver for linear OPF step
        - 'gurobi'
      * - ``CSS_NL_solver``
        - str
        - Solver for nonlinear OPF step (if ``NL=True``)
        - 'bonmin'
      * - ``time_limit``
        - int
        - Solver time limit in seconds
        - 300
      * - ``NL``
        - bool
        - Use nonlinear OPF instead of linear in CSS
        - False

   **Example**

.. literalinclude:: ../../pyflow_tests/doc_examples/wf_array/01_sequential_cable_sizing_css.py
   :language: python
   :lines: 2-



MIP Path Selection (Array)
--------------------------

.. autofunction:: pyflow_acdc.MIP_path_graph

   Solves a master MIP to select array connection paths minimizing total cable length, with optional crossing constraints and Gurobi callback to record feasible solutions over time. Activates cable types on candidate lines upon success.

   .. list-table::
      :widths: 22 10 48 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid with candidate array lines
        - Required
      * - ``max_flow``
        - int
        - Per-line absolute flow bound (≈ turbines per string)
        - ``|nodes|-1``
      * - ``solver_name``
        - str
        - 'glpk' or 'gurobi' (callback supported with Gurobi)
        - 'glpk'
      * - ``crossings``
        - bool
        - Enforce one-active-per-crossing-group
        - False
      * - ``callback``
        - bool
        - Enable Gurobi MIPSOL callback to track (time, objective)
        - False

   **Returns**

   - ``flag`` (bool): feasible solution found
   - ``high_flow`` (int|None): maximum absolute line flow
   - ``model``: Pyomo model
   - ``feasible_solutions``: list of ``(time, objective)`` pairs (if callback)

   **Example**

.. literalinclude:: ../../pyflow_tests/doc_examples/wf_array/02_mip_path_selection_array.py
   :language: python
   :lines: 2-


Simplified CSS Workflow
-----------------------

.. autofunction:: pyflow_acdc.simple_CSS

   Runs a simplified sequential cable sizing workflow with reduced setup.

.. autofunction:: pyflow_acdc.simple_assign_cable_types

   Assigns cable types from an optimized model back into the grid.

Linear CSS Solvers
------------------

.. autofunction:: pyflow_acdc.optimal_l_css_gurobi

   Solves the linear CSS formulation with Gurobi.

.. autofunction:: pyflow_acdc.optimal_l_css_ortools

   Solves the linear CSS formulation with OR-Tools.

**References**
^^^^^^^^^^^^^^

.. [1] B.C. Valerio, P. Gebraad, M. Cheah-Mane, V. A. Lacerda and O. Gomis-Bellmunt,
       "Strategies for wind park inter array optimisation through Mixed Integer Linear Programming"