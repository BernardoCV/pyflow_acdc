Wind Farm Array Sizing Module
=============================

This module provides functions for wind farm array sizing based on [1]_ and
[2]_.

See :doc:`../usage_wf_array` for a sequential cable-sizing workflow.

Inter-array optimisation splits into two problems:

- **Route** — ``MIP_path_graph`` selects which candidate CT lines are built
  (spanning tree, string flow, substation limits). With
  ``enable_cable_types=True``, cable types are chosen in the same MIP.
- **CSS** — ``wind_farm_CSS`` / ``optimal_l_css_ortools`` pick cable types on a
  fixed topology (no routing). ``sequential_CSS`` alternates route MIP then CSS
  each iteration.

Recommended workflow
--------------------

For most studies, use **sequential** sizing: ``sequential_CSS`` with
``enable_cable_types=False`` on the route MIP (the default inside
``sequential_CSS``). Each iteration fixes the route, then runs CSS on that
topology, shrinking the cable catalogue until convergence.

Setting ``enable_cable_types=True`` on ``MIP_path_graph`` solves route and cable
types in one joint MIP. That can be useful for comparison, but the sequential
methodology is recommended for production workflows — see [2]_.

Solver and install matrix
-------------------------

.. list-table::
   :widths: 28 36 36
   :header-rows: 1

   * - Task
     - Recommended install
     - Solver / backend
   * - Route MIP (Pyomo)
     - ``pip install pyflow-acdc[OPF]`` (+ optional ``[Gurobi]``)
     - ``backend='pyomo'`` with ``gurobi`` or ``glpk`` (see :doc:`solver_utils`)
   * - Route MIP (OR-Tools)
     - ``pip install pyflow-acdc[LINEAR_ARRAY]``
     - ``backend='ortools'`` (CP-SAT)
   * - CSS (Pyomo linear)
     - ``pip install pyflow-acdc[OPF]`` (+ optional ``[Gurobi]``)
     - ``CSS_L_solver='gurobi'`` or other Pyomo LP/MIP solver →
       :func:`~pyflow_acdc.linear_transmission_expansion`
   * - CSS (OR-Tools)
     - ``pip install pyflow-acdc[LINEAR_ARRAY]``
     - ``CSS_L_solver='ortools'`` → :func:`~pyflow_acdc.optimal_l_css_ortools`
   * - CSS (nonlinear OPF)
     - ``pip install pyflow-acdc[OPF]`` + Bonmin/Ipopt
     - ``NL='OPF'`` → :func:`~pyflow_acdc.transmission_expansion`

**Recommended defaults** (probe availability with
:func:`~pyflow_acdc.is_pyomo_solver_available`):

- ``MIP_path_graph`` ``backend``: ``'pyomo'`` if Gurobi is available, else
  ``'ortools'``
- ``CSS_L_solver``: ``'gurobi'`` if available, else ``'ortools'``
- ``NL``: :attr:`~pyflow_acdc.constants.CssMode.PF` (linear CSS + post-solve
  power flow for losses), not ``CssMode.OPF``
- ``enable_cable_types``: ``False``; set ``True`` only for the unified route+CSS
  MIP

See :doc:`../installation` for ``[OPF]`` and ``[LINEAR_ARRAY]``.

CssMode.PF vs CssMode.OPF
---------------------------

When ``NL`` is not ``False``, ``sequential_CSS`` normalises it to a
:class:`~pyflow_acdc.constants.CssMode` value:

- **``CssMode.PF``** — linear CSS (Pyomo or OR-Tools), then
  :func:`~pyflow_acdc.power_flow` on the sized array to evaluate AC losses for
  the iteration objective.
- **``CssMode.OPF``** — nonlinear OPF inside CSS; losses come from the NL solver
  export directly.

``CssMode.PF`` is the recommended default: faster and sufficient for comparing
cable options once the route is fixed.

Internal helpers
----------------

``min_sub_connections`` and ``MIPConfig`` are used inside
:func:`~pyflow_acdc.sequential_CSS` and :func:`~pyflow_acdc.MIP_path_graph` to
enforce substation connection limits. They are not part of the public API.

Sequential Cable Sizing (CSS)
-----------------------------

.. autofunction:: pyflow_acdc.sequential_CSS

   Outer loop for array sizing: each iteration runs ``MIP_path_graph`` (route)
   then ``wind_farm_CSS`` (cable types on that route), shrinking the allowed cable
   catalogue until convergence or ``max_iter``. Returns models, iteration
   summary, timing info, solver stats, and the best iteration index.

   Key parameters beyond the table below:

   - ``backend`` — route MIP backend (``MIPBackend.PYOMO`` or
     ``MIPBackend.ORTOOLS``); prefer ``pyomo`` when Gurobi is available
   - ``NL`` — ``False`` (linear), ``CssMode.PF``, or ``CssMode.OPF``; prefer
     ``CssMode.PF``
   - ``sub_min_connections`` — internal; passed through to the route MIP

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
        - Solver for linear CSS step
        - 'glpk'
      * - ``CSS_NL_solver``
        - str
        - Solver for nonlinear OPF step (if ``NL='OPF'``)
        - 'bonmin'
      * - ``time_limit``
        - int
        - Solver time limit in seconds
        - 300
      * - ``NL``
        - bool/str
        - ``False``, ``CssMode.PF``, or ``CssMode.OPF``
        - False

   **Example**

.. literalinclude:: ../../pyflow_tests/doc_examples/wf_array/01_sequential_cable_sizing_css.py
   :language: python
   :lines: 2-



MIP Path Selection (Array)
--------------------------

.. autofunction:: pyflow_acdc.MIP_path_graph

   Solves the inter-array **route** MIP: minimum cable-length spanning forest
   over candidate CT lines, with optional crossing constraints and feasible-
   solution callbacks. Does not run OPF.

   - ``enable_cable_types=False`` (default): route only; marks lines active/
     inactive and may assign a rating via ``simple_assign_cable_types``.
   - ``enable_cable_types=True``: joint route + cable-type MIP (no separate
     CSS step required for sizing).

   Backends: ``pyomo`` (default, external MILP solver) or ``ortools`` (CP-SAT).

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
      * - ``enable_cable_types``
        - bool
        - Joint route + cable-type MIP vs route only
        - False
      * - ``solver_name``
        - str
        - Pyomo MILP solver (e.g. ``glpk``, ``gurobi``)
        - 'glpk'
      * - ``backend``
        - str
        - ``pyomo`` or ``ortools`` for the path MIP
        - 'pyomo'
      * - ``crossings``
        - bool
        - Enforce one-active-per-crossing-group
        - False
      * - ``callback``
        - bool
        - Record feasible solutions over time (solver-dependent)
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


Cable Size Selection (CSS)
--------------------------

.. autofunction:: pyflow_acdc.wind_farm_CSS

   Cable size selection on a **fixed** inter-array topology (``line.active_config``
   set beforehand, typically by ``MIP_path_graph``). Does not optimise routing.

   Dispatches to ``transmission_expansion`` (nonlinear, ``NL='OPF'``),
   ``optimal_l_css_ortools`` (``CSS_L_solver='ortools'``), or
   ``linear_transmission_expansion`` (other linear Pyomo solvers).

.. autofunction:: pyflow_acdc.simple_assign_cable_types

   Post-process a route MIP solution: assign the smallest sufficient cable rating
   per active line from turbine-string flow (used when ``enable_cable_types=False``).

Linear CSS Solver (OR-Tools)
----------------------------

.. autofunction:: pyflow_acdc.optimal_l_css_ortools

   OR-Tools ``linear_solver`` backend for CSS: one cable type per active CT line,
   Ybus DC balance, optional discounted generator OPEX. Called by ``wind_farm_CSS``
   when ``CSS_L_solver='ortools'``. Route selection is not part of this model.
   See also :doc:`L_opf` for the Pyomo linear model (which may include TEP/CT
   network-flow when ``TEP=True``).

**References**
^^^^^^^^^^^^^^

.. [1] Bernardo Castro Valerio et al 2026 J. Phys.: Conf. Ser. 3224 052005
       DOI 10.1088/1742-6596/3224/5/052005

.. [2] Castro Valerio, B., Gebraad, P. M. O., Cheah-Mane, M., A. Lacerda, V., 
      and Gomis-Bellmunt, O.: A multi-stage methodology for wind park inter-array 
      cabling: graph preparation, layout, and sizing, Wind Energ. Sci. Discuss. 
      [preprint], https://doi.org/10.5194/wes-2026-53, in review, 2026.
