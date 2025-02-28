Optimal Power Flow Module
=========================

This page has been pre-filled with the functions that are available in the OPF module by AI, please check the code for more details.

This module provides functions for AC/DC hybrid optimal power flow analysis [1]_.

functions are found in pyflow_acdc.ACDC_OPF and pyflow_acdc.ACDC_OPF_model

AC/DC Hybrid Optimal Power Flow
------------------------------

.. py:function:: OPF_ACDC(grid, ObjRule=None, PV_set=False, OnlyGen=True, Price_Zones=False, TS=False)

   Performs AC/DC hybrid optimal power flow calculation.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``grid``
        - Grid
        - Grid to optimize
        - Required
        - -
      * - ``ObjRule``
        - dict
        - Objective function weights
        - None
        - -
      * - ``PV_set``
        - bool
        - Enable PV setpoint optimization
        - False
        - -
      * - ``OnlyGen``
        - bool
        - Only optimize generation
        - True
        - -
      * - ``Price_Zones``
        - bool
        - Enable price zone constraints
        - False
        - -
      * - ``TS``
        - bool
        - Time series optimization
        - False
        - -
      * - Returns
        - tuple
        - (model, results, timing)
        - -
        - -

   **Objective Function Weights**

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Weight
        - Description
      * - ``Ext_Gen``
        - External generation cost
      * - ``Energy_cost``
        - Energy cost
      * - ``Curtailment_Red``
        - Renewable curtailment reduction
      * - ``AC_losses``
        - AC transmission losses
      * - ``DC_losses``
        - DC transmission losses
      * - ``Converter_Losses``
        - Converter losses
      * - ``PZ_cost_of_generation``
        - Price zone generation cost
      * - ``Renewable_profit``
        - Renewable generation profit
      * - ``Gen_set_dev``
        - Generator setpoint deviation

   **Example**

   .. code-block:: python

       weights = {
           'Energy_cost': 1.0,
           'AC_losses': 0.1,
           'DC_losses': 0.1
       }
       model, results, timing = pyf.OPF_ACDC(grid, ObjRule=weights)

Model Components
---------------

Variables
^^^^^^^^^

The optimization model includes variables for:

- AC node voltages and angles
- DC node voltages 
- Generator active/reactive power
- Renewable generation and curtailment
- Line flows
- Converter power flows
- Price zone variables

Constraints
^^^^^^^^^^

The model enforces constraints for:

- AC power flow equations
- DC power flow equations
- Generator limits
- Line thermal limits
- Voltage limits
- Converter operation limits
- Price zone balancing

Results Processing
-----------------

.. py:function:: obtain_results_TSOPF(model, grid, current_range, idx, Price_Zones)

   Processes time series OPF results.

   Returns dictionaries containing:

   - Converter power flows (AC and DC side)
   - Generator dispatch
   - Line loadings
   - Price zone results
   - Renewable curtailment
   - System losses

   **Example**

   .. code-block:: python

       results = pyf.obtain_results_TSOPF(model, grid, 24, 0, Price_Zones=True)



References
----------

.. [1] B.C. Valerio, V. A. Lacerda, M. Cheah-Mane, P. Gebraad and O. Gomis-Bellmunt,
       "An optimal power flow tool for AC/DC systems, applied to the analysis of the
       North Sea Grid for offshore wind integration" in IEEE Transactions on Power
       Systems, doi: 10.1109/TPWRS.2023.3533889.