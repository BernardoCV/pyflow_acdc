Results
=======

The Results class provides methods for analyzing and displaying power flow results. Results are printed in the terminal with the use of prettytable.

Class Attributes
----------------

.. py:class:: Results(Grid, decimals=2, export=None)

    .. list-table::
       :widths: 20 80 20
       :header-rows: 1

       * - Attribute
         - Description     
         - default
       * - Grid
         - The power grid object containing the network data.
         - Required
       * - decimals
         - Number of decimal places to round results to.
         - 2
       * - export
         - Path to export results to CSV files.
         - None


Class Methods
-------------

Options
^^^^^^^


.. py:method:: options()

   Prints a list of all available results methods.

All
^^^

.. py:method:: All()

   Displays all available results for both AC and DC grids including:
   
   - :ref:`AC power flow <res_ac_powerflow>`
   - :ref:`AC voltages <res_ac_voltage>`
   - :ref:`AC line currents and power flows <res_ac_lines_current>`
   - :ref:`DC bus data <res_dc_bus>`
   - :ref:`DC line currents and power flows <res_dc_lines_current>`
   - :ref:`Converter data <res_converter>`
   - :ref:`Slack bus information <res_slack_all>`
   - :ref:`Power losses <res_power_loss>`

   If optimization module is used:

   - :ref:`External generator data <res_ext_gen>`
   - :ref:`Renewable energy sources <res_ext_ren>`
   - :ref:`Objective function <res_objective>`
   - :ref:`Price zone optimization <res_price_zone>`

   If transmission expansion module is used:

   - :ref:`Transmission expansion <res_TEP_N>`
   - :ref:`Normalized transmission expansion <res_TEP_norm>`

.. py:method:: All_AC()

   Displays results for AC grid only:
   
   - :ref:`AC power flow <res_ac_powerflow>`
   - :ref:`AC voltages <res_ac_voltage>`
   - :ref:`AC line currents and power flows <res_ac_lines_current>`
   - :ref:`AC slack bus info <res_ac_slack>`
   - :ref:`AC power losses <res_ac_power_loss>`

.. py:method:: All_DC()

   Displays results for DC grid only:
   
   - :ref:`DC bus data <res_dc_bus>`
   - :ref:`DC line currents and power flows <res_dc_lines_current>`
   - :ref:`DC slack bus info <res_dc_slack>`
   - :ref:`DC power losses <res_dc_power_loss>`

.. _res_slack_all:
.. py:method:: Slack_All()

   Displays slack bus information for both AC and DC grids.


AC Specific Results
^^^^^^^^^^^^^^^^^^^	
By running the following code, the results will be printed in the terminal. 

.. code-block:: python

   import pyflow_acdc as pyf

   grid,res = pyf.Stagg5MATACDC()
   pyf.ACDC_sequential(grid)
   res.All


.. _res_ac_powerflow:

.. py:method:: AC_Powerflow()

   Displays AC power flow, split into differnet asynchronous grids. Results include:
   
   - Power generation
   - Reactive power generation  
   - Power load
   - Reactive power load
   - Real power by converter (if hybrid grid)
   - Reactive power by converter (if hybrid grid)
   - Power injected
   - Reactive power injected

   Example output::
   
      Results AC power

      Grid AC 1
      +------+----------------+---------------------+-----------------+----------------------+-------------------------+-------------------------------+----------------------+---------------------------+
      | Node | Power Gen (MW) | Reactive Gen (MVAR) | Power Load (MW) | Reactive Load (MVAR) | Power converters DC(MW) | Reactive converters DC (MVAR) | Power injected  (MW) | Reactive injected  (MVAR) |
      +------+----------------+---------------------+-----------------+----------------------+-------------------------+-------------------------------+----------------------+---------------------------+
      |  1   |    133.619     |        84.328       |       0.0       |         0.0          |           0.0           |               0               |       133.619        |           84.328          |
      |  2   |      40.0      |       -32.844       |       20.0      |         10.0         |          -60.0          |             -40.0             |        -40.0         |          -82.844          |
      |  3   |      0.0       |         -0.0        |       45.0      |         15.0         |          20.774         |             7.131             |       -24.226        |           -7.869          |
      |  4   |      0.0       |          0          |       40.0      |         5.0          |           0.0           |               0               |        -40.0         |            -5.0           |
      |  5   |      0.0       |          0          |       60.0      |         10.0         |           35.0          |              5.0              |        -25.0         |            -5.0           |
      +------+----------------+---------------------+-----------------+----------------------+-------------------------+-------------------------------+----------------------+---------------------------+
      

.. _res_ac_voltage:
.. py:method:: AC_voltage()

   Displays AC voltage results including:
   
   - Voltage magnitude
   - Voltage angle

   Example output::

      Results AC bus voltage

      Grid AC 1
      +-----+--------------+---------------------+
      | Bus | Voltage (pu) | Voltage angle (deg) |
      +-----+--------------+---------------------+
      |  1  |     1.06     |         0.0         |
      |  2  |     1.0      |        -2.383       |
      |  3  |     1.0      |        -3.895       |
      |  4  |    0.996     |        -4.261       |
      |  5  |    0.991     |        -4.149       |
      +-----+--------------+---------------------+


.. _res_ac_lines_current:
.. py:method:: AC_lines_current()

   Displays AC line current results including:
   
   - Current magnitude
   - Line loading percentage
   - Line capacity
   - Line polarity

   Example output::

      Results AC Lines Currents
      Grid AC 1
      +------+----------+--------+-------------+-----------+-----------+----------------+
      | Line | From bus | To bus | i from (kA) | i to (kA) | Loading % | Capacity [MVA] |
      +------+----------+--------+-------------+-----------+-----------+----------------+
      |  1   |    1     |   2    |    0.192    |   0.198   |   81.019  |      150       |
      |  2   |    1     |   3    |    0.059    |   0.063   |   37.561  |      100       |
      |  3   |    2     |   3    |    0.024    |   0.022   |   14.637  |      100       |
      |  4   |    2     |   4    |     0.03    |   0.029   |   17.841  |      100       |
      |  5   |    2     |   5    |    0.042    |   0.042   |   25.395  |      100       |
      |  6   |    3     |   4    |    0.039    |    0.04   |   23.932  |      100       |
      |  7   |    4     |   5    |     0.0     |   0.008   |   4.648   |      100       |
      +------+----------+--------+-------------+-----------+-----------+----------------+

.. _res_ac_lines_power:
.. py:method:: AC_lines_power()

   Displays AC line power flow results including:
   
   - Power flow from sending end
   - Power flow to receiving end  
   - Power losses

   Example output::

      Results AC Lines power
      Grid AC 1
      +------+----------+--------+-------------+---------------+-----------+-----------+-----------------+---------------+
      | Line | From bus | To bus | P from (MW) | Q from (MVAR) | P to (MW) | Q to (MW) | Power loss (MW) | Q loss (MVAR) |
      +------+----------+--------+-------------+---------------+-----------+-----------+-----------------+---------------+
      |  1   |    1     |   2    |    98.365   |     71.369    |  -95.648  |   -69.59  |      2.717      |     1.779     |
      |  2   |    1     |   3    |    35.254   |     12.96     |  -34.192  |  -15.083  |      1.062      |     -2.123    |
      |  3   |    2     |   3    |    13.248   |     -6.223    |  -13.132  |   2.571   |      0.116      |     -3.652    |
      |  4   |    2     |   4    |    17.072   |     -5.181    |  -16.891  |    1.74   |      0.181      |     -3.441    |
      |  5   |    2     |   5    |    25.328   |     -1.85     |  -25.071  |   -0.352  |      0.257      |     -2.202    |
      |  6   |    3     |   4    |    23.098   |     4.643     |  -23.042  |   -6.465  |      0.057      |     -1.822    |
      |  7   |    4     |   5    |    -0.067   |     -0.275    |   0.071   |   -4.648  |      0.004      |     -4.922    |
      +------+----------+--------+-------------+---------------+-----------+-----------+-----------------+---------------+

.. _res_ac_slack:
.. py:method:: Slack_AC()

   Displays slack bus information for AC grid.

   Example output::

      Slack nodes
      +-----------+------------+
      |    Grid   | Slack node |
      +-----------+------------+
      | AC Grid 1 |     1      |
      +-----------+------------+


.. _res_ac_power_loss:
.. py:method:: Power_loss_AC()

   Displays power loss information for AC grid.   

   Example output::

      Power loss AC
      +------------+-----------------+
      |    Grid    | Power Loss (MW) |
      +------------+-----------------+
      | AC Grid 1  |      4.393      |
      | Total loss |      4.393      |
      +------------+-----------------+


DC Specific Results
^^^^^^^^^^^^^^^^^^^

.. _res_dc_bus:

.. py:method:: DC_bus()

   Displays DC bus results including:
   
   - Power generation
   - Power load
   - Converter power
   - Power injection
   - Voltage   

   Example output::

      Results DC

      Grid DC 1
      +------+----------------+-----------------+---------------------------+---------------------+--------------+
      | Node | Power Gen (MW) | Power Load (MW) | Power Converter ACDC (MW) | Power injected (MW) | Voltage (pu) |
      +------+----------------+-----------------+---------------------------+---------------------+--------------+
      |  1   |       0        |        0        |           58.652          |        58.652       |    1.008     |
      |  2   |       0        |        0        |           -21.92          |        -21.92       |     1.0      |
      |  3   |       0        |        0        |          -36.191          |       -36.191       |    0.998     |
      +------+----------------+-----------------+---------------------------+---------------------+--------------+

.. _res_dc_lines_current:
.. py:method:: DC_lines_current()

   Displays DC line current results including:
   
   - Current magnitude
   - Line loading percentage
   - Line capacity
   - Line polarity

   Example output::

      Results DC Lines current
      Grid DC 1
      +------+----------+--------+--------+-----------+---------------+------------------------------------+
      | Line | From bus | To bus | I (kA) | Loading % | Capacity [MW] | Polarity                           |
      +------+----------+--------+--------+-----------+---------------+------------------------------------+
      |  1   |    1     |   2    | -0.044 |   30.681  |      100      | Monopolar (symmetrically grounded) |
      |  2   |    2     |   3    | -0.012 |   8.519   |      100      | Monopolar (symmetrically grounded) |
      |  3   |    1     |   3    | -0.04  |   27.971  |      100      | Monopolar (symmetrically grounded) |
      +------+----------+--------+--------+-----------+---------------+------------------------------------+


.. _res_dc_lines_power:
.. py:method:: DC_lines_power()

   Displays DC line power flow results including:
   
   - Power flow from sending end
   - Power flow to receiving end  
   - Power losses

   Example output::

      Results DC Lines power
      Grid DC 1
      +------+----------+--------+-------------+-----------+-----------------+
      | Line | From bus | To bus | P from (MW) | P to (MW) | Power loss (MW) |
      +------+----------+--------+-------------+-----------+-----------------+
      |  1   |    1     |   2    |    30.681   |   -30.44  |      0.241      |
      |  2   |    2     |   3    |    8.519    |    -8.5   |      0.019      |
      |  3   |    1     |   3    |    27.971   |   -27.69  |      0.281      |
      +------+----------+--------+-------------+-----------+-----------------+


.. _res_dc_slack:
.. py:method:: Slack_DC() 

   Displays slack bus information for DC grid.

   Example output::

      Slack nodes
      +-----------+------------+
      |    Grid   | Slack node |
      +-----------+------------+
      | AC Grid 1 |     1      |
      | DC Grid 1 |     2      |
      +-----------+------------+


.. _res_dc_power_loss:
.. py:method:: Power_loss_DC()

   Displays power loss information for DC grid.

   Example output::

      Power loss DC
      +------------+-----------------+
      |    Grid    | Power Loss (MW) |
      +------------+-----------------+
      | DC Grid 1  |      0.541      |
      | Total loss |      0.541      |
      +------------+-----------------+

Optimization Results
^^^^^^^^^^^^^^^^^^^^^


.. _res_ext_gen:
.. py:method:: Ext_gen()

   Displays external generator results including:
   
   - Generator
   - Power (MW)
   - Price (€/MWh)
   - Loading (%)
   - Cost (€)

   Example output::

      External Generation optimization
      +-----------+------+------------+-----------------------+-------------------------+--------------------+-----------+---------+
      | Generator | Node | Power (MW) | Reactive power (MVAR) | Quadratic Price €/MWh^2 | Linear Price €/MWh | Loading % | Cost k€ |
      +-----------+------+------------+-----------------------+-------------------------+--------------------+-----------+---------+
      |     1     | 30.0 |  673.041   |         140.0         |           0.01          |        0.3         |   61.695  |   5.0   |
      |     2     | 31.0 |   646.0    |         300.0         |           0.01          |        0.3         |   100.0   |   4.0   |
      |     3     | 32.0 |  672.666   |        285.674        |           0.01          |        0.3         |   93.143  |   5.0   |
      +-----------+------+------------+-----------------------+-------------------------+--------------------+-----------+---------+

.. _res_ext_ren:
.. py:method:: Ext_REN()

   Displays renewable sources results including:
   
   - Base power (MW)
   - Curtailment (%)
   - Power injected (MW)
   - Reactive power injected (MVAR)

   Example output::

      Renewable energy sources
      +-------+-----------------+---------------+---------------------+--------------------------------+-------------+---------+-----------------------+
      |  Bus  | Base Power (MW) | Curtailment % | Power Injected (MW) | Reactive Power Injected (MVAR) | Price €/MWh | Cost k€ | Curtailment Cost [k€] |
      +-------+-----------------+---------------+---------------------+--------------------------------+-------------+---------+-----------------------+
      |   T1  |       9.5       |     0.152     |        9.486        |             -0.524             |      0      |    0    |           0           |
      |   T2  |       9.5       |     0.154     |        9.485        |             -0.527             |      0      |    0    |           0           |
      |   T3  |       9.5       |     0.156     |        9.485        |             -0.53              |      0      |    0    |           0           |
      |   T4  |       9.5       |     0.159     |        9.485        |             -0.535             |      0      |    0    |           0           |
      |   T5  |       9.5       |     1.095     |        9.396        |             1.402              |      0      |    0    |           0           |
      |   T6  |       9.5       |     1.091     |        9.396        |              1.4               |      0      |    0    |           0           |
      |   T7  |       9.5       |     1.088     |        9.397        |             1.398              |      0      |    0    |           0           |
      |   T8  |       9.5       |     1.085     |        9.397        |             1.396              |      0      |    0    |           0           |
      | Total |       76.0      |     0.623     |        75.527       |                                |             |    0    |           0           |
      +-------+-----------------+---------------+---------------------+--------------------------------+-------------+---------+-----------------------+

.. _res_objective:
.. py:method:: OBJ_res()

   Displays function value for all optimization functions.

   Example output::
      
      Objective function value
      +-----------------------+--------+------------------+------------------+
      |       Objective       | Weight |      Value       |  Weighted Value  |
      +-----------------------+--------+------------------+------------------+
      |        Ext_Gen        |  0.00  |    15 143.85     |       0.00       |
      |      Energy_cost      |  0.00  |    172 036.78    |       0.00       |
      |    Curtailment_Red    |  0.00  |     5 851.94     |       0.00       |
      |       AC_losses       |  0.00  |      47.55       |       0.00       |
      |       DC_losses       |  0.00  |      49.34       |       0.00       |
      |    Converter_Losses   |  0.00  |      156.38      |       0.00       |
      |     General_Losses    |  0.00  |      253.27      |       0.00       |
      | PZ_cost_of_generation |  1.00  | 2 266 461 478.61 | 2 266 461 478.61 |
      |    Renewable_profit   |  0.00  |       0.00       |       0.00       |
      |      Gen_set_dev      |  0.00  |     2 089.86     |       0.00       |
      +-----------------------+--------+------------------+------------------+

Price Zone optimization
^^^^^^^^^^^^^^^^^^^^^^^^

.. _res_price_zone:
.. py:method:: Price_zone()

   Displays price zone results

   Example output::

      Price_Zone
      +------------+--------------------------+-----------------+-----------+-------------+-------------+---------------+
      | Price_Zone | Renewable Generation(MW) | Generation (MW) | Load (MW) | Import (MW) | Export (MW) | Price (€/MWh) |
      +------------+--------------------------+-----------------+-----------+-------------+-------------+---------------+
      |     BE     |            0             |     2430.355    |  3028.613 |   598.258   |      0      |     18.45     |
      |     DE     |            0             |     7182.182    |  8215.099 |   1032.917  |      0      |      0.0      |
      |     DK     |            0             |     183.954     |  840.161  |   656.207   |      0      |      3.93     |
      |     GB     |            0             |     3880.814    |  5992.166 |   2111.353  |      0      |      24.8     |
      |     NL     |            0             |     1139.034    |  2380.084 |   1241.049  |      0      |     24.71     |
      |     NO     |            0             |     327.514     |  812.459  |   484.945   |      0      |      6.36     |
      |    MTDC    |            0             |        0        |     0     |      0      |      0      |       1       |
      |    o_BE    |          31.268          |       0.0       |    0.0    |      0      |    31.268   |     18.45     |
      |    o_DE    |         2050.71          |       0.0       |    0.0    |      0      |   2050.71   |      0.0      |
      |    o_DK    |         3082.808         |       0.0       |    0.0    |      0      |   3082.808  |      3.93     |
      |    o_NL    |         775.306          |       0.0       |    0.0    |      0      |   775.306   |     24.71     |
      |    o_NO    |         498.419          |       0.0       |    0.0    |      0      |   498.419   |      6.36     |
      +------------+--------------------------+-----------------+-----------+-------------+-------------+---------------+
      +------------+------------------+-------------------------+-----------------------+----------------------+-----------------+
      | Price_Zone | Social Cost [k€] | Renewable Gen Cost [k€] | Curtailment Cost [k€] | Generation Cost [k€] | Total Cost [k€] |
      +------------+------------------+-------------------------+-----------------------+----------------------+-----------------+
      |     BE     |     -38.712      |           0.0           |           0           |        44.845        |      6.133      |
      |     DE     |     -88.381      |           0.0           |           0           |         0.0          |     -88.381     |
      |     DK     |      -6.845      |           0.0           |           0           |        0.723         |      -6.123     |
      |     GB     |     -94.419      |           0.0           |           0           |        96.239        |      1.819      |
      |     NL     |     -99.751      |           0.0           |           0           |        28.148        |     -71.602     |
      |     NO     |      -3.909      |           0.0           |           0           |        2.082         |      -1.826     |
      |    MTDC    |       0.0        |           0.0           |           0           |         0.0          |       0.0       |
      |    o_BE    |       0.0        |          0.577          |           0           |         0.0          |      0.577      |
      |    o_DE    |       0.0        |           0.0           |           0           |         0.0          |       0.0       |
      |    o_DK    |       0.0        |          12.109         |           0           |         0.0          |      12.109     |
      |    o_NL    |       0.0        |          19.16          |           0           |         0.0          |      19.16      |
      |    o_NO    |       0.0        |          3.169          |           0           |         0.0          |      3.169      |
      |   Total    |     -332.017     |          35.015         |           0           |       172.037        |     -124.966    |
      +------------+------------------+-------------------------+-----------------------+----------------------+-----------------+

Treansmission expansion
^^^^^^^^^^^^^^^^^^^^^^^^

.. _res_TEP_N:
.. py:method:: TEP_N()

   Displays transmission expansion results 
   
   Example output::

      Transmission Expansion Problem
      +---------+---------+---------+-------------+---------+-----------------------------+--------------------+
      | Element |   Type  | Initial | Optimized N | Maximum | Optimized Power Rating [MW] | Expansion Cost [€] |
      +---------+---------+---------+-------------+---------+-----------------------------+--------------------+
      |   2-6   | AC Line |    0    |     1.0     |    5    |             120             |       30.00        |
      |   3-5   | AC Line |    1    |     3.0     |    5    |             360             |       40.00        |
      |   4-6   | AC Line |    0    |     2.0     |    5    |             240             |       60.00        |
      |  Total  |         |         |             |         |                             |       130.00       |
      +---------+---------+---------+-------------+---------+-----------------------------+--------------------+

.. _res_TEP_norm:
.. py:method:: TEP_norm()

   Displays NPV objective function value for transmission expansion results

   Example output::

      +-----------------------+--------+--------+----------------+----------------+
      |       Objective       | Weight | Value  | Weighted Value |      NPV       |
      +-----------------------+--------+--------+----------------+----------------+
      |        Ext_Gen        |  0.00  | 774.55 |      0.00      | 132 468 266.42 |
      |      Energy_cost      |  0.00  |  0.00  |      0.00      |      0.00      |
      |    Curtailment_Red    |  0.00  |  0.00  |      0.00      |      0.00      |
      |       AC_losses       |  0.00  | 14.55  |      0.00      |  2 488 902.61  |
      |       DC_losses       |  0.00  |  0.00  |      0.00      |      0.00      |
      |    Converter_Losses   |  0.00  |  0.00  |      0.00      |      0.00      |
      |     General_Losses    |  0.00  | 14.55  |      0.00      |  2 488 902.61  |
      | PZ_cost_of_generation |  0.00  |  0.00  |      0.00      |      0.00      |
      |    Renewable_profit   |  0.00  |  0.00  |      0.00      |      0.00      |
      |      Gen_set_dev      |  0.00  |  3.36  |      0.00      |   575 038.66   |
      +-----------------------+--------+--------+----------------+----------------+



Other Results
^^^^^^^^^^^^^^^

.. _res_converter:
.. py:method:: Converter()

   Displays converter results including:
   
   - AC and DC power
   - Reactive power
   - Power losses
   - Control modes
   - Loading

   Example output::

      AC DC Converters
      +-----------+---------+---------+-----------------+----------------------+-----------------+--------------+-----------------------+-----------------------+-----------------------------+
      | Converter | AC node | DC node | Power s AC (MW) | Reactive s AC (MVAR) | Power c AC (MW) | Power DC(MW) | Reactive power (MVAR) | Power loss IGBTs (MW) | Power loss AC elements (MW) |
      +-----------+---------+---------+-----------------+----------------------+-----------------+--------------+-----------------------+-----------------------+-----------------------------+
      |     1     |    2    |    1    |      -60.0      |        -40.0         |     -59.916     |    58.652    |        -32.129        |         1.264         |            0.084            |
      |     2     |    3    |    2    |      20.774     |        7.131         |      20.782     |    -21.92    |         -0.621        |         1.139         |            0.008            |
      |     3     |    5    |    3    |       35.0      |         5.0          |      35.02      |   -36.191    |         -0.269        |          1.17         |             0.02            |
      +-----------+---------+---------+-----------------+----------------------+-----------------+--------------+-----------------------+-----------------------+-----------------------------+
      +-----------+-----------------+-----------------+-----------+----------------+
      | Converter | AC control mode | DC control mode | Loading % | Capacity [MVA] |
      +-----------+-----------------+-----------------+-----------+----------------+
      |     1     |        PQ       |       PAC       |   60.093  |      120       |
      |     2     |        PV       |      Slack      |   18.303  |      120       |
      |     3     |        PQ       |       PAC       |   30.159  |      120       |
      +-----------+-----------------+-----------------+-----------+----------------+

.. _res_power_loss:
.. py:method:: Power_loss()

   Displays power loss information for both AC and DC grids.


   Example output::

      Power loss
      +------------------+-----------------+--------+
      |       Grid       | Power Loss (MW) | Load % |
      +------------------+-----------------+--------+
      |    AC Grid 1     |      4.393      | 32.739 |
      |    DC Grid 1     |      0.541      | 22.39  |
      | AC DC Converters |      3.685      |        |
      |    Total loss    |      8.619      |        |
      |                  |                 |        |
      |    Generation    |     173.619     |        |
      |    Efficiency    |      95.0%      |        |
      +------------------+-----------------+--------+
