Transmission Expansion Planning Module
======================================

This module provides functions for transmission expansion planning (TEP) analysis of AC/DC hybrid power systems. [1]_

Functions are found in pyflow_acdc.ACDC_TEP

Transmission Expansion Planning
-------------------------------

This section creates an OPF :ref:`model <model_creation>`, chooses a state :ref:`objective function <obj_functions>`. Afterwards it will include transmission expansion planning in the model and :ref:`TEP objectives <TEP_obj_functions>`, finally :ref:`solves <model_solving>` the model.

Running one state transmission expansion planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: transmission_expansion(grid,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,ObjRule=None,solver='bonmin')

   Performs transmission expansion planning analysis.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid to analyze
        - Required
      * - ``NPV``
        - bool
        - Calculate net present value
        - True
      * - ``n_years``
        - int
        - Number of years for NPV calculation
        - 25
      * - ``Hy``
        - int
        - Hours per year
        - 8760
      * - ``discount_rate``
        - float
        - Discount rate for NPV
        - 0.02
      * - ``ObjRule``
        - dict
        - Objective function weights
        - None
      * - ``solver``
        - str
        - Solver to use
        - 'bonmin'
     

   **Returns**

   Returns a tuple containing:
   
   - Model object
   - Model results
   - Timing information
   - Solver statistics

   **Example**

   .. code-block:: python

       model, results, timing, stats = pyf.transmission_expansion(grid)

Running multiple scenario based transmission expansion planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: multi_scenario_TEP(grid,increase_Pmin=False,NPV=True,n_years=25,Hy=8760,discount_rate=0.02,clustering_options=None,ObjRule=None,solver='bonmin')

   Performs a multiple scenario based transmission expansion planning analysis. It utilizes the clustering module to cluster the time series data into different states.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid to analyze
        - Required
      * - ``increase_Pmin``
        - bool
        - Increase minimum power limit
        - False
      * - ``NPV``
        - bool
        - Calculate net present value
        - True
      * - ``n_years``
        - int
        - Number of years for NPV
        - 25
      * - ``Hy``
        - int
        - Hours per year
        - 8760
      * - ``discount_rate``
        - float
        - Discount rate for NPV
        - 0.02
      * - ``clustering_options``
        - dict
        - Time series clustering options
        - None
      * - ``ObjRule``
        - dict
        - Objective function weights
        - None
      * - ``solver``
        - str
        - Solver to use
        - 'bonmin'

   **Returns**

   Returns a tuple containing:
   
   - Model object
   - Model results
   - Timing information
   - Solver statistics
   - TEP time series results

   **Example**

   .. code-block:: python

       model, results, timing, stats, ts_results = pyf.transmission_expansion_TS(grid)


.. _TEP_obj_functions:

Transmission Expansion Planning objectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: TEP_obj(model,grid,NPV)

   Returns the objective function for the transmission expansion planning based on [1]_:

   .. list-table::
    :widths: 40 40
    :header-rows: 1

    * - Description
      - Formula
    * - AC expansion
      - :math:`\Psi_{exp}=\sum_{h \in \mathcal{E}_{ac}} \left[(n_h - n_{h,\text{b}}) \cdot \psi_h(L_h) \right]`
    * - AC reconducting
      - :math:`\Psi_{rec}=\sum_{u \in \mathcal{U}_{ac}} \left[\xi_u \cdot \psi_u(L_u) \right]`
    * - AC line selection
      - :math:`\Psi_{a}=\sum_{a \in \mathcal{E}_a} \sum_{n \in \mathcal{CT}} \left[ \xi_{a,n} \cdot \psi_n(L_a) \right]`
    * - DC expansion
      - :math:`\Psi_{dc}=\sum_{e \in \mathcal{E}_{dc}} \left[(n_e - n_{e,\text{b}}) \cdot \psi_e(L_e, p_e) \right]`
    * - Converter expansion
      - :math:`\Psi_{conv}=\sum_{cn \in \mathcal{E}_{cn}} \left[(n_{cn} - n_{cn,\text{b}}) \cdot \psi_{cn}(p_{cn}) \right]`
    * - General objective function
      - :math:`\Psi = \Psi_{exp}+\Psi_{rec}+\Psi_{a}+\Psi_{dc}+\Psi_{conv}`
    * - State objective function
      - :math:`\phi =` :ref:`OPF function <obj_functions>`
    * - Net present value
      - :math:`\min \left[\frac{1 - \left(1 + r\right)^{-y}}{r} \cdot H_y  \cdot \phi  + \Psi \right]`

Export Results
^^^^^^^^^^^^^^

.. py:function:: export_TEP_TS_results_to_excel(grid, export)

   Exports time series TEP results to Excel file.

   .. list-table::
      :widths: 20 10 50 10
      :header-rows: 1

      * - Parameter
        - Type
        - Description
        - Default
      * - ``grid``
        - Grid
        - Grid with results
        - Required
      * - ``export``
        - str
        - Export file path
        - Required

   **Example**

   .. code-block:: python

       pyf.export_TEP_TS_results_to_excel(grid, "results.xlsx")

**References**

.. [1] B. C. Valerio, M. Cheah-Mane, V. A. Lacerda, P. Gebraad and O. Gomis-Bellmunt,
       "Transmission expansion planning for hybrid AC/DC grids using a
       Mixed-Integer Non-linear Programming approach"
