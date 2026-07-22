Transmission Expansion Planning Module
======================================

This module provides functions for transmission expansion planning (TEP) analysis of AC/DC hybrid power systems. [1]_

See :doc:`../usage_tep` for static TEP and multi-scenario (MS) TEP workflows.

Functions are found in pyflow_acdc.ACDC_Static_TEP

Transmission Expansion Planning
-------------------------------

This section creates an OPF :ref:`model <model_creation>`, chooses a state :ref:`objective function <obj_functions>`. Afterwards it will include transmission expansion planning in the model and :ref:`TEP objectives <TEP_obj_functions>`, finally :ref:`solves <model_solving>` the model.

Running one state transmission expansion planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.transmission_expansion

   Example on ``case118_TEP`` (model build with **ipopt**; use **bonmin** and omit
   ``build_only`` for a full MINLP solve). See :doc:`../usage_tep` for grid setup
   and workflow details.

   .. literalinclude:: ../../pyflow_tests/doc_examples/tep/01_running_one_state_transmission_expansion_planning.py
      :language: python
      :lines: 2-

   **Returns**

   Returns a tuple containing:
   
   - Model object
   - Model results
   - Timing information
   - Solver statistics

Running multiple scenario based transmission expansion planning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.multi_scenario_TEP

   Performs a multiple scenario based transmission expansion planning analysis. It utilizes the clustering module to cluster the time series data into different states. See :doc:`clustering` and :doc:`../usage_tep`.

   Example on ``NS_MTDC_2025`` (model build with **ipopt** and ``build_only=True``; use
   **bonmin** and omit ``build_only`` for a full MINLP solve). See :doc:`../usage_tep`
   for grid setup, time series, and clustering workflow.

   .. literalinclude:: ../../pyflow_tests/doc_examples/tep/02_multi_scenario_tep.py
      :language: python
      :lines: 2-

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
        - Objective component weights (see :doc:`opf`)
        - None
      * - ``solver``
        - str
        - Solver to use
        - 'bonmin'
      * - ``obj_scaling``
        - float
        - Divide objective by this factor for numerical conditioning
        - 1.0
      * - ``build_only``
        - bool
        - Build model and return without solving or exporting
        - False

   **Returns**

   Returns a tuple containing:
   
   - Model object
   - Model results
   - Timing information
   - Solver statistics
   - TEP time series results

Sensitivity Utilities
^^^^^^^^^^^^^^^^^^^^^^

The linear (MILP) TEP driver :func:`~pyflow_acdc.linear_transmission_expansion`
is documented in :doc:`L_models`.

.. autofunction:: pyflow_acdc.alpha_pareto

   Computes Pareto-like trade-off points by sweeping alpha-style objective mixing.

.. autofunction:: pyflow_acdc.rate_sensitivity

   Runs discount-rate sensitivity for TEP objective outcomes.

.. autofunction:: pyflow_acdc.kappa_sensitivity

   Runs kappa-weight sensitivity for TEP objective outcomes.

.. autofunction:: pyflow_acdc.comprehensive_sensitivity_analysis

   Convenience wrapper to execute multiple TEP sensitivity studies.

Element Expansion Helpers
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.expand_elements_from_pd

   Applies expansion definitions from a pandas table.

.. autofunction:: pyflow_acdc.repurpose_element_from_pd

   Applies reconductoring/repurposing definitions from a pandas table.

.. autofunction:: pyflow_acdc.update_attributes

   Updates expansion-related attributes of a TEP-enabled element.

.. autofunction:: pyflow_acdc.expand_element

   Enables or updates one element for TEP investment modeling.


.. _TEP_obj_functions:

Transmission Expansion Planning objectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: pyflow_acdc.ACDC_Static_TEP.tep_obj

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

.. autofunction:: pyflow_acdc.export_TEP_multiScenario_results_to_excel

   Exports multi-scenario TEP time-series tables from ``grid.TEP_multiScenario_res``
   to an Excel workbook (expansion summary plus per-scenario operating tables).

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

.. function:: pyflow_acdc.export_TEP_TS_results_to_excel

   **Deprecated alias** for :func:`export_TEP_multiScenario_results_to_excel`.

**References**

.. [1] Castro Valerio, Bernardo and Cheah-Mane, Marc and Albernaz, Vinícius and Gebraad, Pieter 
       and Gomis-Bellmunt, Oriol, Transmission Expansion Planning for Hybrid Ac/Dc Grids Using a 
       Mixed-Integer Non-Linear Programming Approach. Available at SSRN: https://ssrn.com/abstract=5385596