Market Coefficients Module
=========================

This page has been pre-filled with the functions that are available in the Market_Coeff module by AI, please check the code for more details.

This module provides functions for analyzing market data and generating cost coefficients for price zones.

functions are found in pyflow_acdc.Market_Coeff

Price Zone Coefficient Analysis
-----------------------------

price_zone_coef_data
^^^^^^^^^^^^^^^^^^

.. py:function:: price_zone_coef_data(df, start, end, increase_eq_price=50)

   Analyzes market data to generate price zone coefficients.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``df``
        - DataFrame
        - Market data
        - Required
        - -
      * - ``start``
        - int
        - Start hour
        - Required
        - hour
      * - ``end``
        - int
        - End hour
        - Required
        - hour
      * - ``increase_eq_price``
        - float
        - Price increase for equilibrium
        - 50
        - €/MWh

   Returns dictionary containing:

   - Supply/demand curves
   - Market equilibrium points
   - Cost coefficients
   - Timing information

Cost Generation Curves
--------------------

cost_generation_curve
^^^^^^^^^^^^^^^^^^

.. py:function:: cost_generation_curve(data, hour, increase_eq_price)

   Calculates cost and benefit curves for a specific hour.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``data``
        - dict
        - Market data
        - Required
        - -
      * - ``hour``
        - int
        - Hour to analyze
        - Required
        - hour
      * - ``increase_eq_price``
        - float
        - Price increase
        - Required
        - €/MWh

   Calculates:

   - Cost of generation curve
   - Benefit of consumption curve
   - Market equilibrium point
   - Operating limits (Pmin, Pmax)

Visualization
-----------

plot_curves
^^^^^^^^^

.. py:function:: plot_curves(data, hour, name=None)

   Creates visualization of market curves.

   .. list-table::
      :widths: 20 10 50 10 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
        - Units
      * - ``data``
        - dict
        - Market data
        - Required
        - -
      * - ``hour``
        - int
        - Hour to plot
        - Required
        - hour
      * - ``name``
        - str
        - Output filename
        - None
        - -

   Creates subplots showing:

   - Supply and demand curves
   - Cost of generation curve
   - Integrated supply and demand
   - Price curves

   **Example**

   .. code-block:: python

       pyf.plot_curves(market_data, hour=12, name='market_curves')

Coefficient Calculation
--------------------

calculate_cost_of_generation
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: calculate_cost_of_generation(supply_df, min_volume, eq_volume, max_volume, max_price, supply_interp, eq_price)

   Calculates generation cost coefficients.

   Returns quadratic cost function coefficients (a, b, c) where:
   
   Cost = a*P² + b*P + c

   - a: Quadratic coefficient (€/MW²h)
   - b: Linear coefficient (€/MWh)
   - c: Constant term (€/h)

   Also returns:

   - Volume ranges
   - Interpolation functions
   - Operating limits
