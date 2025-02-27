Grid Modifications
==================

This module provides functions for modifying existing grids by adding components and zones.

functions are found in pyflow_acdc.Class_editor

Add Grid Components
-------------------

Add AC Node
^^^^^^^^^^^^

.. py:function:: add_AC_node(grid, kV_base, node_type='PQ', Voltage_0=1.01, theta_0=0.01, Power_Gained=0, Reactive_Gained=0, Power_load=0, Reactive_load=0, name=None, Umin=0.9, Umax=1.1, Gs=0, Bs=0, x_coord=None, y_coord=None, geometry=None)

   Adds an AC node to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``kV_base``
        - float
        - Base voltage in kV
      * - ``node_type``
        - str
        - Node type ('PQ', 'PV', or 'Slack')
      * - ``Voltage_0``
        - float
        - Initial voltage magnitude in p.u.
      * - ``theta_0``
        - float
        - Initial voltage angle in radians
      * - ``Power_Gained``
        - float
        - Active power generation in p.u.
      * - ``Reactive_Gained``
        - float
        - Reactive power generation in p.u.
      * - ``Power_load``
        - float
        - Active power load in p.u.
      * - ``Reactive_load``
        - float
        - Reactive power load in p.u.
      * - ``name``
        - str
        - Node name
      * - ``Umin``
        - float
        - Minimum voltage magnitude in p.u.
      * - ``Umax``
        - float
        - Maximum voltage magnitude in p.u.
      * - ``Gs``
        - float
        - Shunt conductance in p.u.
      * - ``Bs``
        - float
        - Shunt susceptance in p.u.
      * - ``x_coord``
        - float
        - X coordinate for plotting
      * - ``y_coord``
        - float
        - Y coordinate for plotting
      * - ``geometry``
        - Geometry
        - Shapely geometry object
      * - Returns
        - Node_AC
        - Created AC node

   **Example**

   .. code-block:: python

       node = pyf.add_AC_node(grid, kV_base=400, name='bus1', node_type='PQ')


Add AC Line
^^^^^^^^^^^^

.. py:function:: add_line_AC(grid, fromNode, toNode, MVA_rating=None, r=0, x=0, b=0, g=0, R_Ohm_km=None, L_mH_km=None, C_uF_km=0, G_uS_km=0, A_rating=None, m=1, shift=0, name=None, tap_changer=False, Expandable=False, N_cables=1, Length_km=1, geometry=None, data_in='pu', Cable_type='Custom')

   Adds an AC line to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``fromNode``
        - Node_AC
        - Source node
      * - ``toNode``
        - Node_AC
        - Destination node
      * - ``MVA_rating``
        - float
        - Line rating in MVA
      * - ``r, x, b, g``
        - float
        - Line parameters in p.u.
      * - ``R_Ohm_km``
        - float
        - Resistance per km
      * - ``L_mH_km``
        - float
        - Inductance per km
      * - ``C_uF_km``
        - float
        - Capacitance per km
      * - ``G_uS_km``
        - float
        - Conductance per km
      * - ``A_rating``
        - float
        - Current rating in Amperes
      * - ``m``
        - float
        - Transformer ratio
      * - ``shift``
        - float
        - Phase shift angle
      * - ``tap_changer``
        - bool
        - If True, creates transformer
      * - ``Expandable``
        - bool
        - If True, creates expandable line
      * - ``N_cables``
        - int
        - Number of parallel cables
      * - ``Length_km``
        - float
        - Line length in km
      * - ``geometry``
        - Geometry
        - Shapely geometry object
      * - ``data_in``
        - str
        - Input format ('pu', 'Ohm', 'Real')
      * - ``Cable_type``
        - str
        - Cable specification name
      * - Returns
        - Line_AC
        - Created AC line

   **Example**

   .. code-block:: python

       line = pyf.add_line_AC(grid, node1, node2, R_Ohm_km=0.1, L_mH_km=0.4, Length_km=10)

Add DC Node
^^^^^^^^^^^^

.. py:function:: add_DC_node(grid, kV_base, node_type='P', Voltage_0=1.01, Power_Gained=0, Power_load=0, name=None, Umin=0.95, Umax=1.05, x_coord=None, y_coord=None, geometry=None)

   Adds a DC node to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``kV_base``
        - float
        - Base voltage in kV
      * - ``node_type``
        - str
        - Node type ('P', 'Slack', or 'Droop')
      * - ``Voltage_0``
        - float
        - Initial voltage magnitude in p.u.
      * - ``Power_Gained``
        - float
        - Power generation in p.u.
      * - ``Power_load``
        - float
        - Power load in p.u.
      * - ``name``
        - str
        - Node name
      * - ``Umin``
        - float
        - Minimum voltage magnitude in p.u.
      * - ``Umax``
        - float
        - Maximum voltage magnitude in p.u.
      * - ``x_coord``
        - float
        - X coordinate for plotting
      * - ``y_coord``
        - float
        - Y coordinate for plotting
      * - ``geometry``
        - Geometry
        - Shapely geometry object
      * - Returns
        - Node_DC
        - Created DC node

   **Example**

   .. code-block:: python

       node = pyf.add_DC_node(grid, kV_base=320, name='dc_bus1')


Add DC Line
^^^^^^^^^^^^

.. py:function:: add_line_DC(grid, fromNode, toNode, Resistance_pu=0.001, MW_rating=9999, Length_km=1, polarity='m', name=None, geometry=None, Cable_type='Custom')

   Adds a DC line to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``fromNode``
        - Node_DC
        - Source node
      * - ``toNode``
        - Node_DC
        - Destination node
      * - ``Resistance_pu``
        - float
        - Line resistance in p.u.
      * - ``MW_rating``
        - float
        - Power rating in MW
      * - ``Length_km``
        - float
        - Line length in km
      * - ``polarity``
        - str
        - 'm' for monopolar, 'b' for bipolar
      * - ``name``
        - str
        - Line name
      * - ``geometry``
        - Geometry
        - Shapely geometry object
      * - ``Cable_type``
        - str
        - Cable specification name
      * - Returns
        - Line_DC
        - Created DC line

   **Example**

   .. code-block:: python

       line = pyf.add_line_DC(grid, node1, node2, MW_rating=1000, polarity='b')

Add AC/DC Converter
^^^^^^^^^^^^^^^^^^^^

.. py:function:: add_ACDC_converter(grid, AC_node, DC_node, AC_type='PV', DC_type=None, P_AC_MW=0, Q_AC_MVA=0, P_DC_MW=0, Transformer_resistance=0, Transformer_reactance=0, Phase_Reactor_R=0, Phase_Reactor_X=0, Filter=0, Droop=0, kV_base=None, MVA_max=None, nConvP=1, polarity=1, lossa=1.103, lossb=0.887, losscrect=2.885, losscinv=4.371, Ucmin=0.85, Ucmax=1.2, name=None, geometry=None)

   Adds an AC/DC converter to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``AC_node``
        - Node_AC
        - AC side node
      * - ``DC_node``
        - Node_DC
        - DC side node
      * - ``AC_type``
        - str
        - AC control type ('PV', 'PQ')
      * - ``DC_type``
        - str
        - DC control type
      * - ``P_AC_MW``
        - float
        - AC active power setpoint
      * - ``Q_AC_MVA``
        - float
        - AC reactive power setpoint
      * - ``P_DC_MW``
        - float
        - DC power setpoint
      * - ``MVA_max``
        - float
        - Converter rating
      * - ``nConvP``
        - int
        - Number of parallel converters
      * - ``geometry``
        - Geometry
        - Shapely geometry object
      * - Returns
        - AC_DC_converter
        - Created converter

   **Example**

   .. code-block:: python

       conv = pyf.add_ACDC_converter(grid, ac_node, dc_node, MVA_max=1000)

Add Generator
^^^^^^^^^^^^^^

.. py:function:: add_gen(grid, node_name, gen_name=None, price_zone_link=False, lf=0, qf=0, MWmax=99999, MWmin=0, MVArmin=None, MVArmax=None, PsetMW=0, QsetMVA=0, Smax=None, fuel_type='Other', geometry=None)

   Adds a generator to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``node_name``
        - str
        - Name of node to connect to
      * - ``gen_name``
        - str
        - Generator name
      * - ``MWmax``
        - float
        - Maximum active power
      * - ``MWmin``
        - float
        - Minimum active power
      * - ``MVArmin``
        - float
        - Minimum reactive power
      * - ``MVArmax``
        - float
        - Maximum reactive power
      * - ``fuel_type``
        - str
        - Generator fuel type
      * - Returns
        - Gen_AC
        - Created generator

   **Example**

   .. code-block:: python

       gen = pyf.add_gen(grid, "bus1", MWmax=500, fuel_type="Natural Gas")

Add Renewable Source
^^^^^^^^^^^^^^^^^^^^

.. py:function:: add_RenSource(grid, node_name, base, ren_source_name=None, available=1, zone=None, price_zone=None, Offshore=False, MTDC=None, geometry=None, ren_type='Wind')

   Adds a renewable energy source to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``node_name``
        - str
        - Name of node to connect to
      * - ``base``
        - float
        - Base power in MW
      * - ``ren_type``
        - str
        - Type ('Wind', 'Solar')
      * - ``zone``
        - str
        - Renewable zone name
      * - ``price_zone``
        - str
        - Price zone name
      * - Returns
        - Ren_Source
        - Created renewable source

   **Example**

   .. code-block:: python

       source = pyf.add_RenSource(grid, "bus1", 100, ren_type="Wind")

Add Price Zone
^^^^^^^^^^^^^^	

.. py:function:: add_price_zone(grid, name, price, import_pu_L=1, export_pu_G=1, a=0, b=1, c=0, import_expand_pu=0)

   Adds a price zone to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``name``
        - str
        - Zone name
      * - ``price``
        - float
        - Base price
      * - ``import_pu_L``
        - float
        - Import limit p.u.
      * - ``export_pu_G``
        - float
        - Export limit p.u.
      * - Returns
        - Price_Zone
        - Created price zone

   **Example**

   .. code-block:: python

       zone = pyf.add_price_zone(grid, "Zone1", price=50)

Add Renewable Source Zone
^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: add_RenSource_zone(grid, name)

   Adds a renewable source zone to the grid.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``name``
        - str
        - Zone name
      * - Returns
        - Ren_source_zone
        - Created renewable zone

   **Example**

   .. code-block:: python

       zone = pyf.add_RenSource_zone(grid, "WindZone1")

Zone Assignments
----------------

Assign Node to Price Zone
^^^^^^^^^^^^^^^^^^^^^^^^^	

.. py:function:: assign_nodeToPrice_Zone(grid, node_name, ACDC, new_price_zone_name)

   Assigns a node to a price zone.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid containing node
      * - ``node_name``
        - str
        - Name of node to assign
      * - ``ACDC``
        - str
        - 'AC' or 'DC'
      * - ``new_price_zone_name``
        - str
        - Name of target price zone

   **Example**

   .. code-block:: python

       pyf.assign_nodeToPrice_Zone(grid, "bus1", "AC", "Zone1")

Assign Renewable to Zone
^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: assign_RenToZone(grid, ren_source_name, new_zone_name)

   Assigns a renewable source to a zone.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid containing source
      * - ``ren_source_name``
        - str
        - Name of renewable source
      * - ``new_zone_name``
        - str
        - Name of target zone

   **Example**

   .. code-block:: python

       pyf.assign_RenToZone(grid, "wind1", "WindZone1")

Line Modifications
------------------

Change Line to Expandable
^^^^^^^^^^^^^^^^^^^^^^^^    

.. py:function:: change_line_AC_to_expandable(grid, line_name)

   Converts an AC line to an expandable line.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid containing line
      * - ``line_name``
        - str
        - Name of line to convert

   **Example**

   .. code-block:: python

       pyf.change_line_AC_to_expandable(grid, "line1")

Change Line to Transformer
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. py:function:: change_line_AC_to_tap_transformer(grid, line_name)

   Converts an AC line to a tap-changing transformer.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid containing line
      * - ``line_name``
        - str
        - Name of line to convert

   **Example**

   .. code-block:: python

       pyf.change_line_AC_to_tap_transformer(grid, "line1")

Time Series
-----------

Add Time Series
^^^^^^^^^^^^^^^

.. py:function:: add_TimeSeries(grid, Time_Series_data, associated=None, TS_type=None, ignore=None)

   Adds time series data to grid components.

   .. list-table::
      :widths: 20 10 70
      :header-rows: 1

      * - Parameter
        - Type
        - Description
      * - ``grid``
        - Grid
        - Grid to modify
      * - ``Time_Series_data``
        - DataFrame
        - Time series data
      * - ``associated``
        - str
        - Component name
      * - ``TS_type``
        - str
        - Time series type
      * - ``ignore``
        - str
        - Pattern to ignore

   **Example**

   .. code-block:: python

       pyf.add_TimeSeries(grid, load_data, TS_type="Load")
