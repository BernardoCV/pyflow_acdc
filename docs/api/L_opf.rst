AC 'dc linear' Optimal Power Flow Module
========================================

This module provides functions for AC 'dc linear' optimal power flow analysis [1]_.

functions are found in pyflow_acdc.AC_OPF_L_model

AC 'dc linear' Optimal Power Flow
---------------------------------

Running the OPF
^^^^^^^^^^^^^^^

This flow sets up and solves the AC 'dc linear' OPF. It creates the :ref:`model <L_model_creation>`, optionally adds TEP/REC/CT investment variables, and solves with a Pyomo solver. Results are then exported back to the `grid`.

When ``TEP=True``, the Pyomo linear model may include TEP/CT network-flow and
investment-linking constraints (see :func:`~pyflow_acdc.AC_OPF_L_model.TEP_variables`).
That is the Pyomo path used by :func:`~pyflow_acdc.linear_transmission_expansion`
and :func:`~pyflow_acdc.wind_farm_CSS`.

.. autofunction:: pyflow_acdc.optimal_l_pf

.. _L_model_creation:

Creating the Linear OPF model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.AC_OPF_L_model.opf_create_l_model_ac

**Variables**

The linear OPF includes variables for:

- AC node angles
- Generator active power
- Renewable generation via availability and curtailment factors
- AC line active power flows

**Constraints**

The model enforces constraints for:

- AC nodal active power balance (linearized)
- Generator aggregation at nodes
- Renewable injection aggregation at nodes
- AC branch linearized power flow equations
- Thermal limits (including linear big-M formulations for REC/CT states)
- Slack angle constraints
- Optional array network-flow conservation and investment-linking
- Optional investment bounds for generators and lines (if TEP)

TEP/REC/CT Parameters and Variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.AC_OPF_L_model.TEP_parameters

   Sets parameters for TEP/REC/CT decisions (e.g., base multiplicities, initial configs, limits).

.. autofunction:: pyflow_acdc.AC_OPF_L_model.TEP_variables

   Adds investment variables:
   - Generator multiplicities (optional integer bounded by capability)
   - AC expansion line multiplicities (integer)
   - Reconfiguration branch selection (binary)
   - Cable-type selection (binary per type and line)
   - Optional type-usage flags and array flow variables

Exporting Results
^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.AC_OPF_L_model.export_acdc_l_model_to_pyflow_acdc

   Exports Pyomo solution back to the `grid` (internal helper; not exported from
   ``pyflow_acdc`` top level — called by :func:`optimal_l_pf`):
   - Generator dispatch and renewable gamma
   - AC node angles and injections
   - AC line flows and losses (linearized, zero reactive)
   - TEP/REC/CT selections and flows (including optional array network-flow)
   - Optional post-processing for time-limit cases (oversizing analysis and fixes)

Solvers
^^^^^^^

The linear OPF can be solved by LP/MIP solvers in Pyomo.

Tested with:

- GLPK
- Gurobi

**Notes**

- If REC/CT/TEP or array flow variables are enabled, the problem becomes MIP. Prefer a MIP-capable solver (e.g., ``gurobi``).

**References**
^^^^^^^^^^^^^^

.. [1] B.C. Valerio, P. Gebraad, M. Cheah-Mane, V. A. Lacerda and O. Gomis-Bellmunt,
       "Strategies for wind park inter array optimisation through Mixed Integer Linear Programming"