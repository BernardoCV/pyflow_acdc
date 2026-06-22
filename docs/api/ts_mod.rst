Time Series Modifications
=========================

Operational and investment time-series helpers in
:mod:`pyflow_acdc.grid_modifications`. Grid construction and zone definitions
are covered in :doc:`grid_mod`.

Renewable Source Zone
---------------------

Renewable-source zones group turbines or plants that share the same availability
time series (e.g. several identical wind turbines modelled as one zone).

.. autofunction:: pyflow_acdc.add_RenSource_zone

Assign Renewable to Zone
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.assign_RenToZone

Price Zone
----------

Price zones play a similar grouping role for load and price time series across
multiple buses. Price-zone-only series (``a_CG``, ``b_CG``, ``c_CG``,
``PGL_min``, ``PGL_max``) follow the market-based OPF formulation in [1]_.
See :ref:`price_zones` and :ref:`price_zone_assignments` in :doc:`grid_mod`.

Time Series Data
----------------

.. autofunction:: pyflow_acdc.add_TimeSeries

TimeSeries Object
-----------------

.. autoclass:: pyflow_acdc.TimeSeries
   :no-members:

Representative-period clustering for TEP is documented in :doc:`clustering`.
Workflow guides: :doc:`../usage_tep` (MS TEP), :doc:`../usage_mp_tep` (MP+MS TEP).

References
----------

.. [1] B. C. Valerio, V. A. Lacerda, M. Cheah-Mañe, P. Gebraad, and O. Gomis-Bellmunt,
       "Optimizing offshore wind integration through multi-terminal DC grids: a market-based
       OPF framework for the North Sea interconnectors," IET Conference Proceedings, vol. 2025,
       no. 6, pp. 150–155, 2025. doi: 10.1049/icp.2025.1198
