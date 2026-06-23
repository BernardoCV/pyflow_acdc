Constants and enums
===================

Shared domain constants and string enums live in :mod:`pyflow_acdc.constants`.
They centralise magic values used across OPF, TEP, and array sizing.

Array sizing enums
------------------

.. autoclass:: pyflow_acdc.constants.CssMode
   :members:

   Cable-string-sizing mode for :func:`~pyflow_acdc.sequential_CSS` and
   :func:`~pyflow_acdc.simple_CSS` (``NL`` argument). ``False`` selects the
   linear model; ``CssMode.PF`` runs linear CSS then a post-solve
   :func:`~pyflow_acdc.power_flow` for losses; ``CssMode.OPF`` uses the
   nonlinear OPF path.

.. autoclass:: pyflow_acdc.constants.MIPBackend
   :members:

   Backend for the inter-array **route** MIP in :func:`~pyflow_acdc.MIP_path_graph`.

OPF objective components
------------------------

.. autoclass:: pyflow_acdc.constants.ObjComponent
   :members:

   Keys for :func:`~pyflow_acdc.optimal_pf` / TEP ``ObjRule`` weight dicts.

Economics and solver defaults
-----------------------------

.. py:data:: pyflow_acdc.constants.HOURS_PER_YEAR

   Hours per year (8760). Used in NPV and present-value calculations.

.. py:data:: pyflow_acdc.constants.DEFAULT_TIME_LIMIT

   Default solver time limit in seconds (300).

.. autofunction:: pyflow_acdc.constants.present_value_factor

.. autofunction:: pyflow_acdc.constants.default_obj_weights
