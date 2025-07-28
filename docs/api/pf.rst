Power Flow Module
=================

This module provides functions for AC, DC and AC/DC power flow analysis.

functions are found in pyflow_acdc.ACDC_PF

Running the Power Flow
----------------------

For a simple power flow, the function :py:func:`Power_flow` can be used. This function will automatically detect the type of power flow to run (AC, DC or AC/DC) and will run the appropriate power flow.

.. py:function:: Power_flow(grid,tol_lim=1e-10, maxIter=100)

   Performs power flow calculation.



AC Power Flow
-------------

The AC power flow solution is based on [1]_, to solve the AC power flow Newton Raphson method is used. The equations involving :math:`P_{slack}` and :math:`Q_{slack}` of each separate AC grid ( :math:`\Gamma` = number of AC grids) are not included. In addition, the Q equations of PV nodes are also not included.

The number of unknown variables for :math:`V_i` and :math:`\theta_i` are (:math:`|\mathcal{N}_{ac}| - \Gamma - PV`) and (:math:`|\mathcal{N}_{ac}| - \Gamma` ) respectively. These unknown variables are arranged into vectors :math:`\boldsymbol{\theta}`, :math:`|V|`, and the composite vector :math:`\boldsymbol{x}`.

The active power equations at each node i are:

.. math::

  P_i = \sum P_{g_i} +  \sum \gamma_{rg_i}P_{rg_i} - P_{l_i} + \sum P_{cn_i}

.. math::

  P_i(x) = \sum_{k=1}^{\mathcal{N}_{ac}} |V_i||V_k|[G_{ik} \cos(\theta_i-\theta_k) + B_{ik} \sin(\theta_i-\theta_k)]

The reactive power equations at each node i are:

.. math::

  Q_i = \sum Q_{g_i} + \sum Q_{rg_i} -Q_{l_i}+\sum Q_{cn_{i}}

.. math::

  Q_i(x) = \sum_{k=1}^{\mathcal{N}_{ac}} |V_i||V_k|[G_{ik} \sin(\theta_i-\theta_k) - B_{ik} \cos(\theta_i-\theta_k)]

These equations are combined into the function :math:`\boldsymbol{f(x)} = 0`:

.. math::

  \boldsymbol{f(x)} = \begin{bmatrix}
  P_1(\boldsymbol{x})-P_1 \\
  \vdots \\
  P_{|\mathcal{N}_{ac}|-\Gamma}(\boldsymbol{x})-P_{|\mathcal{N}_{ac}|-\Gamma} \\
  Q_1(\boldsymbol{x})-Q_1 \\
  \vdots \\
  Q_{|\mathcal{N}_{ac}|-\Gamma-PV}(\boldsymbol{x})-Q_{|\mathcal{N}_{ac}|-\Gamma-PV}
  \end{bmatrix}

The Jacobian matrix :math:`\boldsymbol{J}` contains the first-order partial derivatives:

.. math::

  \boldsymbol{J} = \begin{bmatrix}
  \boldsymbol{J_{11}} & \boldsymbol{J_{12}} \\
  \boldsymbol{J_{21}} & \boldsymbol{J_{22}}
  \end{bmatrix}

Where:

.. math::

  [i,i] \text{ are the diagonal elements} \\
  [i,k] \text{ are the off-diagonal elements}


:math:`J_{11}` (size :math:`|\mathcal{N}_{ac}|-\Gamma \times |\mathcal{N}_{ac}|-\Gamma`) contains :math:`\partial P(x)/\partial \theta`:

.. math::

  \boldsymbol{J_{11}}[i,i] &= -Q_i-V_i^2 \cdot B_{ii} \\
  \boldsymbol{J_{11}}[i,k] &= V_i \cdot V_k \cdot (G_{ik} \cdot \sin(\theta_i-\theta_k)-B_{ik} \cdot \cos(\theta_i-\theta_k))

:math:`J_{12}` (size :math:`|\mathcal{N}_{ac}|-\Gamma \times |\mathcal{N}_{ac}|-\Gamma-PV`) contains :math:`\partial P(x)/\partial V`:

.. math::

  \boldsymbol{J_{12}}[i,i] &= \frac{P_i}{V_i}+ G_{ii} \cdot V_i \\
  \boldsymbol{J_{12}}[i,k] &= V_i \cdot (G_{ik} \cdot \cos(\theta_i-\theta_k)+B_{ik} \cdot \sin(\theta_i-\theta_k))

:math:`J_{21}` (size :math:`|\mathcal{N}_{ac}|-\Gamma-PV \times |\mathcal{N}_{ac}|-\Gamma`) contains :math:`\partial Q(x)/\partial \theta`:

.. math::

  \boldsymbol{J_{21}}[i,i] &= P_i-V_i^2 \cdot G_{ii} \\
  \boldsymbol{J_{21}}[i,k] &= -V_i \cdot V_k \cdot (G_{ik} \cdot \cos(\theta_i-\theta_k)+B_{ik} \cdot \sin(\theta_i-\theta_k))

:math:`J_{22}` (size :math:`|\mathcal{N}_{ac}|-\Gamma-PV \times |\mathcal{N}_{ac}|-\Gamma-PV`) contains :math:`\partial Q(x)/\partial V`:

.. math::

  \boldsymbol{J_{22}}[i,i] &= \frac{Q_i}{V_i}-B_{ii} \cdot V_i \\
  \boldsymbol{J_{22}}[i,k] &= V_i \cdot (G_{ik} \cdot \sin(\theta_i-\theta_k)-B_{ik} \cdot \cos(\theta_i-\theta_k))

The Newton-Raphson iteration is then:

.. math::

  \begin{bmatrix}
  \boldsymbol{J_{11}} & \boldsymbol{J_{12}} \\
  \boldsymbol{J_{21}} & \boldsymbol{J_{22}}
  \end{bmatrix}
  \begin{bmatrix}
  \boldsymbol{\Delta \theta} \\
  \boldsymbol{\Delta |V|}
  \end{bmatrix}
  =
  \begin{bmatrix}
  \boldsymbol{\Delta P(x)} \\
  \boldsymbol{\Delta Q(x)}
  \end{bmatrix}

Running the AC Power Flow
^^^^^^^^^^^^^^^^^^^^^^^^^	

.. py:function:: AC_PowerFlow(grid, tol_lim=1e-8, maxIter=100)

   Performs AC power flow calculation.

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
      * - ``tol_lim``
        - float
        - Convergence tolerance
        - 1e-8
      * - ``maxIter``
        - int
        - Maximum iterations
        - 100


   **Example**

   .. code-block:: python

       time = pyf.AC_PowerFlow(grid)

DC Power Flow
-------------

 It is important to note that 'DC power flow' here specifically refers to the flow in DC grids and not to the linearized power flow that is often used as a simplification of AC grids.

The DC power flow solution is based on [2]_, to solve the DC power flow Newton Raphson method is used. 


.. math::
    :label: eq:P_DC

    P_{cn_d} - P_{l_d} = U_d \sum_{\substack{f=1; f \neq d}}^{n_{dc}} \left( \left( U_d - U_f \right) \cdot p_e \left(\frac{1}{R_{df}} \right) \right), \left\{ R_{df} \neq 0 \right\}

Similar to the AC power flow, the DC power flow is solved by Newton-Raphson method by defining a vector :math:`\boldsymbol{y}`:

.. math::

    \boldsymbol{y} = \begin{bmatrix}
         U_{1} \\
         U_{2} \\
         \vdots \\
         U_{|\mathcal{N}_{dc}|-s_{DC}}
         \end{bmatrix}

.. math::

    P_d &= P_{cn_d} - P_{l_d} \\
    P_d(y) &= U_d \sum_{\substack{f=1; f \neq d}}^{n_{dc}} \left( \left( U_d - V_f \right) \cdot p_e \left(\frac{1}{R_{df}} \right) \right), \left\{ R_{df} \neq 0 \right\}

.. math::

    \boldsymbol{f(y)} = \begin{bmatrix}
         P_1(\boldsymbol{y})-P_1 \\
         \vdots \\
         P_{|\mathcal{N}_{dc}|-s_{DC}}(\boldsymbol{y})-P_{|\mathcal{N}_{dc}|-s_{DC}} \\
         \end{bmatrix} = 0

With :math:`J_{DC}`, the Jacobian matrix of :math:`\boldsymbol{f(y)}` considered as:
 


.. math::

    J_{DC} \cdot \frac{\Delta U_{DC}}{U_{DC}} &= \Delta P_{DC} \\
    J_{DC} &= U_{DC}\frac{\delta P_{DC}}{\delta U_{DC}}



:math:`J_{DC}` is a matrix of :math:`|\mathcal{N}_{dc}|-s_{DC}` (:math:`s_{DC}` is the number of DC slack buses)

.. math::

  [d,d] \text{ are the diagonal elements} \\
  [d,f] \text{ are the off-diagonal elements}

.. math::

    \boldsymbol{J_{DC}}{[d,d]} &= P_{DC_d}+ U_{d}^2 \cdot \sum^n_{f=1; d\neq f}(\frac{1}{R_{df}} \cdot p_{e}) , \left\{ R_{df} \neq 0 \right\} \\
    \boldsymbol{J_{DC}}{[d,f]} &= - p_{e}\cdot \frac{1}{R_{df}} \cdot U_d\cdot V_f , \left\{ R_{df} \neq 0 \right\}

In contrast to the AC Newton-Raphson, in the DC Newton-Raphson, the power target of the droop nodes changes each iteration.

.. math::

    P = -P_{l_d}+P_{conv_0}+ (1-U_d) \kappa

Where :math:`P_{conv_0}` is the target power in pu of the converter, :math:`U_i` is the voltage of the DC bus in pu and :math:`\kappa` the droop coefficient in :math:`P_{pu}/V_{pu}`. For the DC bus that are under droop control the Jacobian is also modified as follows:

.. math::

    \boldsymbol{J_{DC}}{[d,d]} = P_{DC_d}+ \kappa \cdot U_d+U_{d}^2 \cdot \sum^n_{f=1; f\neq d}(\frac{1}{R_{df}} \cdot p_{e}) , \left\{ R_{df} \neq 0 \right\}


Running the DC Power Flow
^^^^^^^^^^^^^^^^^^^^^^^^^	

.. py:function:: DC_PowerFlow(grid, tol_lim=1e-8, maxIter=100)

   Performs DC power flow calculation.

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
      * - ``tol_lim``
        - float
        - Convergence tolerance
        - 1e-8
      * - ``maxIter``
        - int
        - Maximum iterations
        - 100

   **Example**

   .. code-block:: python

       time = pyf.DC_PowerFlow(grid)

Sequential AC/DC Power Flow
---------------------------

Sequential AC/DC power flow is a method that solves the AC and DC power flows sequentially. It is a three-section process:

1. AC power flow: Solves the AC power flow equations for the AC grid.
2. Converter power flow: Solves the converter power flow equations.
3. DC power flow: Solves the DC power flow equations for the DC grid.

The sequential solver will compare the :math:`P_{conv}` of converters in the AC grid until convergence is reached.



.. figure:: ../images/Sequential_mod_dark.svg
   :alt: Sequential AC/DC Power Flow
   :align: center

   Sequential AC/DC Power Flow 



Running the Sequential AC/DC Power Flow
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. py:function:: ACDC_sequential(grid, tol_lim=1e-8, maxIter=20, change_slack2Droop=False, QLimit=False)

   Performs sequential AC/DC power flow calculation.

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
      * - ``tol_lim``
        - float
        - Convergence tolerance
        - 1e-8
      * - ``maxIter``
        - int
        - Maximum iterations
        - 20
      * - ``change_slack2Droop``
        - bool
        - Change slack to droop control
        - False
      * - ``QLimit``
        - bool
        - Enable converter Q limits
        - False


   **Example**

   .. code-block:: python

       time = pyf.ACDC_sequential(grid)

**References**

.. [1] Arthur R. Bergen. "Power systems analysis". eng. In: Power systems analysis. 2nd ed.
       Upper Saddle River, N.J: Prentice-Hall, 2000. Chap. 10, pp. 323–352. isbn: 0136919901.

.. [2] D. Van Hertem, O. Gomis-Bellmunt, and J. Liang. HVDC Grids: For Offshore and Super-
       grid of the Future. IEEE Press Series on Power and Energy Systems. Wiley, 2016. isbn:
       9781118859155. url: https://books.google.es/books?id=oPP8oAEACAAJ.