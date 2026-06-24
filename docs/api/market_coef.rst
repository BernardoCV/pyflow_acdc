Market coefficients
===================

Functions in ``pyflow_acdc.Market_Coeff`` build **price-zone quadratic cost curves**
for OPF and TEP (:ref:`Price_zone_modelling`). Two data sources are supported:

* **EPEX Spot** order books → hourly ``a``, ``b``, ``c`` coefficients via
  :func:`~pyflow_acdc.price_zone_coef_data`.
* **ENTSO-E Transparency** generation/load CSVs → normalized hourly profiles via
  :func:`~pyflow_acdc.clean_entsoe_data`.

Workflow (EPEX order book)
--------------------------

.. code-block:: python

   import pyflow_acdc as pyf

   # df: CSV with columns Date, Hour, Volume, Price, Sale/Purchase (see below)
   market_data, timing = pyf.price_zone_coef_data(df, start=1, end=8760)

   coef_table = pyf.price_zone_data_pd(market_data, save_csv="price_zone_coef")

   # Optional: inspect one hour
   pyf.plot_curves(market_data, hour=100, name="Belgium")

EPEX CSV format
^^^^^^^^^^^^^^^

The order-book reader expects an **EPEX Spot**-style export:

.. list-table::
   :header-rows: 1
   :widths: 18 12 50

   * - Column
     - Type
     - Description
   * - ``Date``
     - str
     - ``DD/MM/YYYY``
   * - ``Hour``
     - int or str
     - Clock hour; use ``3B`` style labels on DST fall-back nights
   * - ``Volume``
     - float
     - MW
   * - ``Price``
     - float
     - €/MWh
   * - ``C3``, ``C4``
     - any
     - Present in exports; ignored by the parser
   * - ``Sale/Purchase``
     - str
     - **Must be exactly this header.** Values ``Sell`` or ``Purchase`` (supply vs demand)

.. note::
   The ``Sale/Purchase`` column name is required: :func:`pandas.DataFrame.itertuples`
   exposes it as the seventh field (``row._6``). Renaming the column breaks parsing.

``start`` and ``end`` are **hour-of-year** bounds (1–8760, or 8784 in leap years),
not row indices in the CSV.

API reference
-------------

Price zone coefficients
^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.price_zone_coef_data

.. autofunction:: pyflow_acdc.price_zone_data_pd

Visualization
^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.plot_curves

ENTSO-E data cleaning
---------------------

Use this path when load and generation profiles come from the
`ENTSO-E Transparency Platform <https://transparency.entsoe.eu/>`_ rather than
EPEX order books.

Directory layout
^^^^^^^^^^^^^^^^

``key_list`` entries are subfolder names under ``path`` (e.g. bidding zones or
areas). For each key and each year in ``year_list``, place one generation and
one load file using ENTSO-E download names:

.. code-block:: text

   path
   |-- key0
   |   |-- AGGREGATED_GENERATION_PER_TYPE_GENERATION_{year_0-1}12312300-{year_0}12312300.csv
   |   |-- GUI_TOTAL_LOAD_DAYAHEAD_{year_0-1}12312300-{year_0}12312300.csv
   |   |-- AGGREGATED_GENERATION_PER_TYPE_GENERATION_{year_1-1}12312300-{year_1}12312300.csv
   |   |-- GUI_TOTAL_LOAD_DAYAHEAD_{year_1-1}12312300-{year_1}12312300.csv
   |   |-- ...
   |-- key1
   |   |-- AGGREGATED_GENERATION_PER_TYPE_GENERATION_{year_0-1}12312300-{year_0}12312300.csv
   |   |-- GUI_TOTAL_LOAD_DAYAHEAD_{year_0-1}12312300-{year_0}12312300.csv
   |   |-- ...
   |-- ...

``key0``, ``key1``, … are whatever strings you pass in ``key_list`` (for example
``"BE"``, ``"NL"``). ``{year_0}``, ``{year_1}``, … match ``year_list``.

Example::

   pyf.clean_entsoe_data(
       key_list=["BE", "NL"],
       year_list=[2022, 2023],
       path="/data/entsoe",
   )

.. autofunction:: pyflow_acdc.clean_entsoe_data

The workbook contains:

* **Maximum Values** — per-area annual maxima used for normalization.
* **One sheet per year** — hourly normalized profiles (generation types + load).

**References**

See :ref:`Price_zone_modelling` and the OPF bibliography in :doc:`opf`.
