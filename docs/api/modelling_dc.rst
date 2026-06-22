DC System Modelling
=================== 

.. _DC_node_modelling:

DC node
^^^^^^^

.. themed-figure:: dc_node_model
   :width: 250
   :alt: DC node model
   :align: center

   DC node equivalent circuit

The AC node is modeled using voltage :math:`U_d` where [1]_:

.. math::
    :label: eq:PdciSUM

    \begin{align}
        P_{net}^{dc} &= P_{flow}^{dc} \\
        P_{net}^{dc} &= P_{cn_d} + \sum \gamma_{rg_d}P_{rg_d} - P_{l_d} \\
        P_{flow}^{dc} &= U_d \sum_{\substack{f \in \mathcal{N}_{dc} \\ f \neq d}}  \left( (U_d-U_f) \cdot p_{e} \cdot \left(\frac{1}{R_{df}} \right) \right), \left\{ R_{df} \neq 0 \right\} \qquad \forall d \in \mathcal{N}_{dc}
    \end{align}

.. math::
    :label: eq:Udc

    U_{min} \leq U_{d} \leq U_{max} \qquad \forall d \in \mathcal{N}_{dc}


* :math:`U_d` is the voltage magnitude in pu
* :math:`P_{rg}` is the active power injection of renewable generation in pu
* :math:`P_{cn}` is the active power injection of converter in pu
* :math:`P_l` is the active power demand in pu

Class Reference: :class:`pyflow_acdc.Classes.Node_DC`

.. autoclass:: pyflow_acdc.Node_DC
   :no-members:

Example Usage:

.. literalinclude:: ../../pyflow_tests/doc_examples/modelling_dc/01_dc_node.py
   :language: python
   :lines: 2-



.. _DC_line_modelling:

DC line
^^^^^^^

.. themed-figure:: dc_line
   :width: 400
   :alt: DC line model
   :align: center

   DC line model

.. math::
    :label: eq:pol

    p_{e}=\begin{cases}
        1, &\text{for asymmetrical monopolar} \\
        2, &\text{for symmetrical monopolar or bipolar} \\
    \end{cases}

.. math::
    :label: eq:PfromDC

    \begin{align}
        P_{from,d}=&U_d(U_d-U_f) p_{e} \left(\frac{1}{R_{df}} \right) \\
        P_{to,f}=&U_f(U_f-U_d)p_{e} \left(\frac{1}{R_{df}} \right) \\
        -P_{e, rating} \leq& P_{to/from} \leq P_{e,rating} \qquad \forall e \in \mathcal{B}_{dc}
    \end{align}

Class Reference: :class:`pyflow_acdc.Classes.Line_DC`

.. autoclass:: pyflow_acdc.Line_DC
   :no-members:

Example Usage:

.. literalinclude:: ../../pyflow_tests/doc_examples/modelling_dc/02_dc_line.py
   :language: python
   :lines: 2-


DC line expansion
^^^^^^^^^^^^^^^^^

.. themed-figure:: dc_expbranch
   :width: 400
   :alt: DC expansion model
   :align: center

   DC expansion model

The expanded branch object is inside the :class:`Line_DC` class.

The expanded branch :math:`e` is modelled as:

.. math::
    :label: eq:PexpfromDC

    \begin{align}
        P_{from,d}=&U_d(U_d-U_f) p_{e} \left(\frac{n_e}{R_{df}} \right) \\
        P_{to,f}=&U_f(U_f-U_d)p_{e} \left(\frac{n_e}{R_{df}} \right) \\
        -P_{e, rating} \leq& P_{to/from} \leq P_{e,rating} \qquad \forall e \in \mathcal{E}_{dc}
    \end{align}

DCDC converter  
^^^^^^^^^^^^^^^



.. themed-figure:: dcdc_conv
   :width: 400
   :alt: DCDC converter model
   :align: center

   DCDC converter model
   
The DCDC converter is modelled very simply, with the set power injected into the ``toNode`` and the power drawn from the ``fromNode``.
To include losses there is a simple resistive model. For power flow calculations the power injected in either node is kept constant as the losses are pre calculated with the assumption of :math:`V_{to} = 1 pu`. 

.. math::
    :label: eq:DCDCLoss

    P_{from} + P_{to} + P_{loss} = 0 \\
    P_{loss} = \left(\frac{P_{to}}{V_{to}}\right)^2 \cdot R

.. math::
    :label: eq:DCDC_net

    P_{net}^{dc} += \sum P_{DCDC_{from}} + \sum P_{DCDC_{to}}
   

.. math::
    :label: eq:DCDC_case

    \text{DCDC converter}\qquad & 
        \begin{cases}
            P_{DCDC_{from}}, &  \text{if } d \text{ is } \textit{from node} \text{ of } DCDC \\
            P_{DCDC_{to}}, &  \text{if } d \text{ is } \textit{to node} \text{ of } DCDC
        \end{cases} 



Class Reference: :class:`pyflow_acdc.Classes.DCDC_converter`

Key attributes:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``fromNode``
     - Node_DC
     - The starting node
   * - ``toNode``
     - Node_DC
     - The ending node
   * - ``R``
     - float
     - Resistance of the converter
   * - ``P_set``
     - float
     - Power set point of the converter
   * - ``MW_rating``
     - float
     - Power rating of the converter
   * - ``name``
     - str
     - Name of the converter
   * - ``geometry``
     - str
     - Geometry of the converter

Example Usage:

.. literalinclude:: ../../pyflow_tests/doc_examples/modelling_dc/03_dcdc_converter.py
   :language: python
   :lines: 2-


Inputs can be given in MW or pu. For pu use the ``r`` and ``Pset`` parameters. For MW use the ``R_Ohm`` and ``P_MW`` parameters.


**References**

.. [1] B. C. Valerio, V. A. Lacerda, M. Cheah-Mane, P. Gebraad and O. Gomis-Bellmunt,
       "An optimal power flow tool for AC/DC systems, applied to the analysis of the
       North Sea Grid for offshore wind integration" in IEEE Transactions on Power
       Systems, doi: 10.1109/TPWRS.2023.3533889.

