Sparse SOCP
===========

Requires ``pip install pyflow-acdc[SOCP]``. By default,
:func:`~pyflow_acdc.socp_optimise` prefers ``MOSEK`` when it is installed, then
falls back to ``CLARABEL`` and ``SCS``.

This stack is the CVXPY sparse SOCP path for hybrid AC/DC grids. It mirrors the
public NL OPF surface with:

- :func:`~pyflow_acdc.socp_optimise` for single-period runs
- :func:`~pyflow_acdc.soc_window_optimisation` for explicit multiperiod windows
- :func:`~pyflow_acdc.translate_pyf_socp` for the prepared SOCP input object

For model details and current scope, see :doc:`api/socp` and
``plans/convex_acdc_socp_plan.md``.

.. _usage_socp:

Workflow
--------

Start from an analysed or analysable grid, then call the SOCP runner. The
quickest workspace check is ``build_only=True`` to confirm model assembly
without solving.

**Quick example**

.. literalinclude:: ../pyflow_tests/doc_examples/usage/socp_quick.py
   :language: python
   :lines: 1-

Solver notes
------------

- ``solver=None`` prefers ``MOSEK`` if available, then ``CLARABEL``, then
  ``SCS``.
- Use ``build_only=True`` when you only want model size / assembly stats.
- Use ``solver_opts`` to pass backend-specific options to CVXPY.

Window runs
-----------

Use :func:`~pyflow_acdc.soc_window_optimisation` when profiles live in
``grid.Time_series`` and you want a coupled ``T``-period network solve. Pass
``frame_ids`` to select a subset; otherwise the full horizon is used.

Related guides
--------------

- :doc:`usage_opf` for the nonlinear Pyomo/IPOPT OPF stack
- :doc:`usage_window_opf` for the nonlinear rolling/coupled window workflows
- :doc:`api/L_models` for the LP linearization path
