Market Coefficients Module
===========================

This module provides functions for analyzing supply and demand market data to prepare values for the price zone coefficients based on EPEX Spot data format. As well as cleaning data downloaded from ENTSO-E for load and generation trends.

functions are found in pyflow_acdc.Market_Coeff

Price Zone Coefficient Analysis
-------------------------------

Obtain price zone coefficients
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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

   Returns a list of dictionaries containing:

   - ``Date`` - Date of the data
   - ``Hour`` - Hour of the data
   - ``Sell`` - Supply curve data
   - ``Purchase`` - Demand curve data
   - ``Integrated_sets`` - Benefit of consumption and cost of generation from integral of supply and demand curves
   - ``Dem_data_points`` - Number of demand data points
   - ``Gen_data_points`` - Number of supply data points
   - ``max_gen`` - Maximum supply value
   - ``min_demand`` - Minimum demand value
   - ``Market_price`` - Market clearing price
   - ``Volume_eq`` - Market clearing volume
   - ``poly`` - Dictionary containing polynomial coefficients
   - ``prediction_BC`` - Predicted benefit of consumption
   - ``prediction_CG`` - Predicted cost of generation
   - ``Volume_0`` - Volume where first positive price is supplied

   And timing information with the following keys:

   - ``load data`` - Time taken to load data
   - ``avg process`` - Average time taken to process data
   - ``tot process`` - Total time taken to process data

Export data to csv
^^^^^^^^^^^^^^^^^^

.. py:function:: price_zone_data_pd(data,save_csv=None)

   Converts market data to pandas DataFrame and saves it to a csv file.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``data``
        - dict
        - Market data
        - Required
      * - ``save_csv``
        - str
        - Save csv file
        - None

Visualization
^^^^^^^^^^^^^^

Generates plots of market curves.

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

       pyf.plot_curves(market_data, hour=12, name='Belgium')

ENTSO-E Data Cleaning
---------------------

For the use of entsoe data cleaning, the data has to be structured as follows:

.. code-block:: text

    path
    |-- key0
    |   |-- AGGREGATED_GENERATION_PER_TYPE_GENERATION_{year_0-1}12312300-{year_0}12312300.csv
    |   |-- GUI_TOTAL_LOAD_DAYAHEAD_{year_0-1}12312300-{year_0}12312300.csv
    |   |-- AGGREGATED_GENERATION_PER_TYPE_GENERATION_{year_1-1}12312300-{year_1}12312300.csv
    |   |-- GUI_TOTAL_LOAD_DAYAHEAD_{year_1-1}12312300-{year_1}12312300.csv
    |   |-- ...
    |-- key1
    |   |-- ...
    |-- ...  

This is the name of the files when downloaded from ENTSO-E transparency platform.


.. py:function:: clean_entsoe_data(key_list, year_list, production_types=[], output_excel=None,path=None):
    
   Process generation and load data for multiple areas/years and save to Excel.

   .. list-table::
      :widths: 20 10 30 30
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``key_list``
        - list
        - List of keys
        - Required
      * - ``year_list``
        - list  
        - List of years
        - Required
      * - ``production_types``
        - list
        - Reduced list of production types
        - All
      * - ``output_excel``
        - str
        - Output excel file
        - output_data.xlsx
      * - ``path``
        - str
        - Path to save the excel file
        - Current working directory

   **Returns**

   Excel file containing the following sheets:

   * ``Maximum values`` - Contains maximum values for each column
   * ``year_0`` - Contains normalized data for the first year
   * ``year_n`` - Contains normalized data for subsequent years
