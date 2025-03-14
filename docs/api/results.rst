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

