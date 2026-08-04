Grid Modifications
==================

Functions for modifying an existing :class:`~pyflow_acdc.Classes.Grid` — add
components and zones, attach time or investment series, convert line types, and
assign elements to zones or cable options.

Implemented in :mod:`pyflow_acdc.grid_modifications`. See also
:doc:`cable_database` (bundled cable types and source attributions),
:doc:`grid_analysis` (``cable_parameters``, ``analyse_grid``, …) and
:doc:`ts_mod` (time-series CSV layout and renewable zones).

Add Grid Components
-------------------

Add AC Node
^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_AC_node

Add AC Line
^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_line_AC

Line sizing
^^^^^^^^^^^

Add Cable Options
~~~~~~~~~~~~~~~~~

.. autofunction:: pyflow_acdc.add_cable_option

Add Line sizing
~~~~~~~~~~~~~~~

.. autofunction:: pyflow_acdc.add_line_sizing

Add DC Node
^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_DC_node

Add DC Line
^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_line_DC

Add AC/DC Converter
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_ACDC_converter

Add DC/DC Converter
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_DCDC_converter

Add Generator
^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_gen

Add DC Generator
^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_gen_DC

Add External Grid
^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_extgrid

Add Renewable Source
^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_RenSource

Add Battery Storage
^^^^^^^^^^^^^^^^^^^

See :doc:`modelling_storage_hydrogen` for the BESS model. Coupled runs:
:doc:`window` / :doc:`../usage_window_opf`.

.. autofunction:: pyflow_acdc.add_storage

Add Electrolyser
^^^^^^^^^^^^^^^^

See :doc:`modelling_storage_hydrogen` for the electrolyser model. Coupled runs:
:doc:`window` / :doc:`../usage_window_opf`.

.. autofunction:: pyflow_acdc.add_electrolyser

Bulk Add Generators
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_generators

.. _price_zones:

Zones
-----

Add Price Zone
^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_price_zone

Add MTDC Price Zone
^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_MTDC_price_zone

Add Offshore Price Zone
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_offshore_price_zone

Add Renewable Source Zone
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_RenSource_zone
   :no-index:

   Grouping of renewable sources for shared time series — see :doc:`ts_mod`.

Time Series and Investment Data
-------------------------------

Add Time Series
^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_TimeSeries
   :no-index:

   CSV layout and supported series types are documented in :doc:`ts_mod`.

Wire Time Series to Grid Elements
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.time_series_dict

   Registers a :class:`~pyflow_acdc.Time_series.TimeSeries` on the matching
   node, price zone, or renewable zone/source (sets ``TS_dict`` / availability
   keys by ``ts.type``). Called automatically by :func:`~pyflow_acdc.add_TimeSeries`;
   use directly when attaching custom :class:`~pyflow_acdc.Time_series.TimeSeries`
   objects without going through CSV import.

Add Investment Series
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_inv_series

Add Generator Mix Limits
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.add_gen_mix_limits

.. _price_zone_assignments:

Assignments
-----------

Assign Node to Price Zone
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.assign_nodeToPrice_Zone

Assign Converter to Price Zone
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.assign_ConvToPrice_Zone

Assign Line to Cable Option
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.assign_lineToCable_options

Assign Renewable to Zone
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.assign_RenToZone
   :no-index:

   See :doc:`ts_mod`.

Template and Import Helpers
---------------------------

.. autofunction:: pyflow_acdc.create_inv_csv_template

.. autofunction:: pyflow_acdc.create_gen_limit_csv_template

Cable import helpers are documented in :doc:`cable_database`.

Line Modifications
------------------

Change Line to Expandable
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.change_line_AC_to_expandable

Change Line to Reconducting
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.change_line_AC_to_reconducting

Change Line to Transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.change_line_AC_to_tap_transformer
