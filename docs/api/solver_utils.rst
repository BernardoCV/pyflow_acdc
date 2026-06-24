Solver Utilities
================

Check which Pyomo solvers and OR-Tools backends are **installed and available**
in the current environment (e.g. before choosing a solver for OPF or TEP).

Functions are found in ``pyflow_acdc.solver_utils``.

.. autofunction:: pyflow_acdc.check_available_solvers

.. autofunction:: pyflow_acdc.is_pyomo_solver_available

   Return ``True`` if a named Pyomo solver executable is installed (e.g. probe
   for ``'gurobi'`` before choosing array MIP/CSS defaults — see :doc:`wf_array`).
