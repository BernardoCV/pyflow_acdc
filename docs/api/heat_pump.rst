Controllable heat pumps
=======================

:class:`~pyflow_acdc.HeatPump` and :func:`~pyflow_acdc.add_heat_pump` attach
planning-oriented flexible electrical loads to AC buses.

User guide: :doc:`../usage_heat_pump`.

HeatPump class
--------------

.. autoclass:: pyflow_acdc.HeatPump
   :members:
   :exclude-members: reset_class

Add heat pump
-------------

.. autofunction:: pyflow_acdc.add_heat_pump

Modelling note
--------------

The current model is AC-only and follows the planning-oriented formulation in
Montalà-Palau et al. (2026) [#montala2026]_:

- baseline active/reactive electrical demand
- bounded served demand between instantaneous and cumulative comfort limits
- cumulative energy-state carry in ``ts_acdc_opf`` and ``window_nl_opf``

See :doc:`../usage_heat_pump` for an overview.

.. [#montala2026] M. Montalà-Palau, J. J. Markus, M. Kazemi, M. Cheah-Mañé,
   C. Papadimitriou, and O. Gomis-Bellmunt: *Enhancing Distribution System
   Resilience through Energy Communities*, CIRED 2026 Brussels Workshop,
   Paper 1361, 2026.
