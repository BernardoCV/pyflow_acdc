.. _cable_database:

Cable Database
==============

Bundled AC/DC cable types loaded from ``pyflow_acdc/Cable_database/*.yaml`` when using ``Cable_type=...`` in :func:`~pyflow_acdc.add_line_AC`, :func:`~pyflow_acdc.add_line_DC`, and related helpers.

This page lists **430** cable entries in **6** source groups (auto-generated from the YAML ``Reference`` field).

Import and Extend
-----------------

.. autofunction:: pyflow_acdc.import_orbit_cables

.. autofunction:: pyflow_acdc.expand_cable_database

.. contents::
   :local:
   :depth: 1

Summary
-------

.. warning::

   PyFlow-ACDC takes **no ownership** of the cable parameters bundled in ``pyflow_acdc/Cable_database/``. Data were obtained from the sources cited below and are included for **academic and testing purposes only**. Electrical parameters and especially ``Cost_per_km`` values do not represent commercial quotations or manufacturer warranties.

.. list-table::
   :header-rows: 1
   :widths: 60 10

   * - Source (``Reference`` field)
     - Cables
   * - Aluminium test cables
     - 4
   * - ABB\_extrapolated
     - 30
   * - CIGRE B4
     - 4
   * - ORBIT
     - 14
   * - XLPE Land Cable Systems User's Guide
     - 243
   * - XLPE Submarine Cable Systems Attachment to XLPE Land Cable Systems -
     - 135

.. _cable_ref_aluminium-test-cables:

Aluminium test cables
~~~~~~~~~~~~~~~~~~~~~

These cables are used for Moray East and West tests in [1]_ using cable data from XLPE Submarine Cable Systems Attachment to XLPE Land Cable Systems and Power capacity from the projects documentation.

.. [1] Castro Valerio, B., Gebraad, P. M. O., Cheah-Mane, M., A. Lacerda, V., and Gomis-Bellmunt, O.: A multi-stage methodology for wind park inter-array cabling: graph preparation, layout, and sizing, Wind Energ. Sci. Discuss. [preprint], https://doi.org/10.5194/wes-2026-53, in review, 2026

**Cable types (4):**

* ``MOF_240``
* ``MOF_300``
* ``MOF_630``
* ``MOF_800``

.. _cable_ref_abb-extrapolated:

ABB\_extrapolated
~~~~~~~~~~~~~~~~

**Source.** **ABB\_extrapolated**

**Cable types (30):**

* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_1000mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_1200mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_120mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_1400mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_150mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_1600mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_185mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_2000mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_240mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_300mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_400mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_500mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_630mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_800mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_flat_95mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_1000mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_1200mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_120mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_1400mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_150mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_1600mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_185mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_2000mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_240mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_300mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_400mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_500mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_630mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_800mm2``
* ``ABB_extrapolated_XLPE_Cu_33kV_ground_trefoil_95mm2``

.. _cable_ref_cigre-b4:

CIGRE B4
~~~~~~~~

**Source.** **CIGRE B4**

**Cable types (4):**

* ``CIGRE_B4_145kV``
* ``CIGRE_B4_200kV``
* ``CIGRE_B4_380kV``
* ``CIGRE_B4_400kV``

.. _cable_ref_orbit:

ORBIT
~~~~~

**Source.** `ORBIT <https://github.com/NLRWindSystems/ORBIT/tree/dev/library/cables>`__

**Cable types (14):**

* ``NREL_132kV_500mm2``
* ``NREL_220kV_1000mm2``
* ``NREL_220kV_500mm2``
* ``NREL_220kV_630mm2``
* ``NREL_275kV_1200mm2``
* ``NREL_275kV_1600mm2``
* ``NREL_275kV_1900mm2``
* ``NREL_320kV_2000mm2``
* ``NREL_33kV_400mm2``
* ``NREL_33kV_630mm2``
* ``NREL_400kV_2000mm2``
* ``NREL_525kV_2500mm2``
* ``NREL_66kV_185mm2``
* ``NREL_66kV_630mm2``

.. _cable_ref_xlpe-land-cable-systems-user-s-guide:

XLPE Land Cable Systems User's Guide
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source.** **XLPE Land Cable Systems User's Guide**

**Cable types (243):**

* ``ABB_XLPE_Cu_110kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_185mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_240mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_300mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_400mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_110kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_185mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_240mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_300mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_400mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_110kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_185mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_240mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_300mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_400mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_132kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_185mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_240mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_300mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_400mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_132kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_240mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_300mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_400mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_150kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_240mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_300mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_400mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_150kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_220kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_220kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_275kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_275kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_330kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_330kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_400kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_400kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_120mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_150mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_185mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_240mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_300mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_400mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_45kV_ground_flat_95mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_120mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_150mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_185mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_240mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_300mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_400mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_45kV_ground_trefoil_95mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_2500mm2``
* ``ABB_XLPE_Cu_500kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_2500mm2``
* ``ABB_XLPE_Cu_500kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_120mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_150mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_185mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_240mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_300mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_400mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_66kV_ground_flat_95mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_120mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_150mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_185mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_240mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_300mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_400mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_800mm2``
* ``ABB_XLPE_Cu_66kV_ground_trefoil_95mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_1000mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_1200mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_1400mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_150mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_1600mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_185mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_2000mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_240mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_300mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_400mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_500mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_630mm2``
* ``ABB_XLPE_Cu_70kV_ground_flat_800mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_1000mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_1200mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_1400mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_150mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_1600mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_185mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_2000mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_240mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_300mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_400mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_500mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_630mm2``
* ``ABB_XLPE_Cu_70kV_ground_trefoil_800mm2``

.. _cable_ref_xlpe-submarine-cable-systems-attachment-to-xlpe-land-cable-systems:

XLPE Submarine Cable Systems Attachment to XLPE Land Cable Systems -
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Source.** **XLPE Submarine Cable Systems Attachment to XLPE Land Cable Systems -**

**Cable types (135):**

* ``ABB_XLPE_Al_110kV_sub_1000mm2``
* ``ABB_XLPE_Al_110kV_sub_185mm2``
* ``ABB_XLPE_Al_110kV_sub_240mm2``
* ``ABB_XLPE_Al_110kV_sub_300mm2``
* ``ABB_XLPE_Al_110kV_sub_400mm2``
* ``ABB_XLPE_Al_110kV_sub_500mm2``
* ``ABB_XLPE_Al_110kV_sub_630mm2``
* ``ABB_XLPE_Al_110kV_sub_800mm2``
* ``ABB_XLPE_Al_132kV_sub_1000mm2``
* ``ABB_XLPE_Al_132kV_sub_185mm2``
* ``ABB_XLPE_Al_132kV_sub_240mm2``
* ``ABB_XLPE_Al_132kV_sub_300mm2``
* ``ABB_XLPE_Al_132kV_sub_400mm2``
* ``ABB_XLPE_Al_132kV_sub_500mm2``
* ``ABB_XLPE_Al_132kV_sub_630mm2``
* ``ABB_XLPE_Al_132kV_sub_800mm2``
* ``ABB_XLPE_Al_150kV_sub_1000mm2``
* ``ABB_XLPE_Al_150kV_sub_240mm2``
* ``ABB_XLPE_Al_150kV_sub_300mm2``
* ``ABB_XLPE_Al_150kV_sub_400mm2``
* ``ABB_XLPE_Al_150kV_sub_500mm2``
* ``ABB_XLPE_Al_150kV_sub_630mm2``
* ``ABB_XLPE_Al_150kV_sub_800mm2``
* ``ABB_XLPE_Al_220kV_sub_1000mm2``
* ``ABB_XLPE_Al_220kV_sub_500mm2``
* ``ABB_XLPE_Al_220kV_sub_630mm2``
* ``ABB_XLPE_Al_220kV_sub_800mm2``
* ``ABB_XLPE_Al_275kV_sub_1000mm2``
* ``ABB_XLPE_Al_275kV_sub_500mm2``
* ``ABB_XLPE_Al_275kV_sub_630mm2``
* ``ABB_XLPE_Al_275kV_sub_800mm2``
* ``ABB_XLPE_Al_30kV_sub_120mm2``
* ``ABB_XLPE_Al_30kV_sub_150mm2``
* ``ABB_XLPE_Al_30kV_sub_185mm2``
* ``ABB_XLPE_Al_30kV_sub_240mm2``
* ``ABB_XLPE_Al_30kV_sub_300mm2``
* ``ABB_XLPE_Al_30kV_sub_400mm2``
* ``ABB_XLPE_Al_30kV_sub_500mm2``
* ``ABB_XLPE_Al_30kV_sub_630mm2``
* ``ABB_XLPE_Al_30kV_sub_800mm2``
* ``ABB_XLPE_Al_30kV_sub_95mm2``
* ``ABB_XLPE_Al_45kV_sub_1000mm2``
* ``ABB_XLPE_Al_45kV_sub_120mm2``
* ``ABB_XLPE_Al_45kV_sub_150mm2``
* ``ABB_XLPE_Al_45kV_sub_185mm2``
* ``ABB_XLPE_Al_45kV_sub_240mm2``
* ``ABB_XLPE_Al_45kV_sub_300mm2``
* ``ABB_XLPE_Al_45kV_sub_400mm2``
* ``ABB_XLPE_Al_45kV_sub_500mm2``
* ``ABB_XLPE_Al_45kV_sub_630mm2``
* ``ABB_XLPE_Al_45kV_sub_800mm2``
* ``ABB_XLPE_Al_45kV_sub_95mm2``
* ``ABB_XLPE_Al_66kV_sub_1000mm2``
* ``ABB_XLPE_Al_66kV_sub_120mm2``
* ``ABB_XLPE_Al_66kV_sub_150mm2``
* ``ABB_XLPE_Al_66kV_sub_185mm2``
* ``ABB_XLPE_Al_66kV_sub_240mm2``
* ``ABB_XLPE_Al_66kV_sub_300mm2``
* ``ABB_XLPE_Al_66kV_sub_400mm2``
* ``ABB_XLPE_Al_66kV_sub_500mm2``
* ``ABB_XLPE_Al_66kV_sub_630mm2``
* ``ABB_XLPE_Al_66kV_sub_800mm2``
* ``ABB_XLPE_Al_66kV_sub_95mm2``
* ``ABB_XLPE_Cu_110kV_sub_1000mm2``
* ``ABB_XLPE_Cu_110kV_sub_185mm2``
* ``ABB_XLPE_Cu_110kV_sub_240mm2``
* ``ABB_XLPE_Cu_110kV_sub_300mm2``
* ``ABB_XLPE_Cu_110kV_sub_400mm2``
* ``ABB_XLPE_Cu_110kV_sub_500mm2``
* ``ABB_XLPE_Cu_110kV_sub_630mm2``
* ``ABB_XLPE_Cu_110kV_sub_800mm2``
* ``ABB_XLPE_Cu_132kV_sub_1000mm2``
* ``ABB_XLPE_Cu_132kV_sub_185mm2``
* ``ABB_XLPE_Cu_132kV_sub_240mm2``
* ``ABB_XLPE_Cu_132kV_sub_300mm2``
* ``ABB_XLPE_Cu_132kV_sub_400mm2``
* ``ABB_XLPE_Cu_132kV_sub_500mm2``
* ``ABB_XLPE_Cu_132kV_sub_630mm2``
* ``ABB_XLPE_Cu_132kV_sub_800mm2``
* ``ABB_XLPE_Cu_150kV_sub_1000mm2``
* ``ABB_XLPE_Cu_150kV_sub_240mm2``
* ``ABB_XLPE_Cu_150kV_sub_300mm2``
* ``ABB_XLPE_Cu_150kV_sub_400mm2``
* ``ABB_XLPE_Cu_150kV_sub_500mm2``
* ``ABB_XLPE_Cu_150kV_sub_630mm2``
* ``ABB_XLPE_Cu_150kV_sub_800mm2``
* ``ABB_XLPE_Cu_20kV_sub_120mm2``
* ``ABB_XLPE_Cu_20kV_sub_150mm2``
* ``ABB_XLPE_Cu_20kV_sub_185mm2``
* ``ABB_XLPE_Cu_20kV_sub_240mm2``
* ``ABB_XLPE_Cu_20kV_sub_300mm2``
* ``ABB_XLPE_Cu_20kV_sub_400mm2``
* ``ABB_XLPE_Cu_20kV_sub_500mm2``
* ``ABB_XLPE_Cu_20kV_sub_630mm2``
* ``ABB_XLPE_Cu_20kV_sub_95mm2``
* ``ABB_XLPE_Cu_220kV_sub_1000mm2``
* ``ABB_XLPE_Cu_220kV_sub_500mm2``
* ``ABB_XLPE_Cu_220kV_sub_630mm2``
* ``ABB_XLPE_Cu_220kV_sub_800mm2``
* ``ABB_XLPE_Cu_275kV_sub_1000mm2``
* ``ABB_XLPE_Cu_275kV_sub_500mm2``
* ``ABB_XLPE_Cu_275kV_sub_630mm2``
* ``ABB_XLPE_Cu_275kV_sub_800mm2``
* ``ABB_XLPE_Cu_33kV_sub_120mm2``
* ``ABB_XLPE_Cu_33kV_sub_150mm2``
* ``ABB_XLPE_Cu_33kV_sub_185mm2``
* ``ABB_XLPE_Cu_33kV_sub_240mm2``
* ``ABB_XLPE_Cu_33kV_sub_300mm2``
* ``ABB_XLPE_Cu_33kV_sub_400mm2``
* ``ABB_XLPE_Cu_33kV_sub_500mm2``
* ``ABB_XLPE_Cu_33kV_sub_630mm2``
* ``ABB_XLPE_Cu_33kV_sub_800mm2``
* ``ABB_XLPE_Cu_33kV_sub_95mm2``
* ``ABB_XLPE_Cu_45kV_sub_1000mm2``
* ``ABB_XLPE_Cu_45kV_sub_120mm2``
* ``ABB_XLPE_Cu_45kV_sub_150mm2``
* ``ABB_XLPE_Cu_45kV_sub_185mm2``
* ``ABB_XLPE_Cu_45kV_sub_240mm2``
* ``ABB_XLPE_Cu_45kV_sub_300mm2``
* ``ABB_XLPE_Cu_45kV_sub_400mm2``
* ``ABB_XLPE_Cu_45kV_sub_500mm2``
* ``ABB_XLPE_Cu_45kV_sub_630mm2``
* ``ABB_XLPE_Cu_45kV_sub_800mm2``
* ``ABB_XLPE_Cu_45kV_sub_95mm2``
* ``ABB_XLPE_Cu_66kV_sub_1000mm2``
* ``ABB_XLPE_Cu_66kV_sub_120mm2``
* ``ABB_XLPE_Cu_66kV_sub_150mm2``
* ``ABB_XLPE_Cu_66kV_sub_185mm2``
* ``ABB_XLPE_Cu_66kV_sub_240mm2``
* ``ABB_XLPE_Cu_66kV_sub_300mm2``
* ``ABB_XLPE_Cu_66kV_sub_400mm2``
* ``ABB_XLPE_Cu_66kV_sub_500mm2``
* ``ABB_XLPE_Cu_66kV_sub_630mm2``
* ``ABB_XLPE_Cu_66kV_sub_800mm2``
* ``ABB_XLPE_Cu_66kV_sub_95mm2``
