System Modelling
=================

This page presents the modelling of the different components of the grid.

The Classes are found in pyflow_acdc.Classes

AC System Modelling
-------------------

AC node modelling
^^^^^^^^^^^^^^^^^

.. figure:: ../images/AC_node_model.svg
   :width: 300
   :alt: AC node model
   :align: center

   AC node equivalent circuit with voltage source and impedance

The AC node is modeled using a complex voltage phasor :math:`V_i = V_i \angle \theta_i` where:

* :math:`V_i` is the voltage magnitude in pu
* :math:`\theta_i` is the voltage angle
* :math:`P_{rg}` is the active power injection of renewable generation in pu
* :math:`Q_{rg}` is the reactive power injection of renewable generation in pu
* :math:`P_{cn}` is the active power injection of converter in pu
* :math:`Q_{cn}` is the reactive power injection of converter in pu
* :math:`P_g` is the active power injection of generator in pu
* :math:`S_l` is the complex power demand in pu

Class Reference: :class:`pyflow_acdc.Classes.Node_AC`

Key Attributes:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``node_type``
     - str
     - Node type ('Slack' or 'PQ' or 'PV')
   * - ``Voltage_0``
     - float
     - Initial voltage magnitude in p.u.
   * - ``theta_0``
     - float
     - Initial voltage angle in degrees
   * - ``kV_base``
     - float
     - Base voltage in kV
   * - ``Power_Gained``
     - float
     - Active power injection in MW
   * - ``Reactive_Gained``
     - float
     - Reactive power injection in MVAr
   * - ``Power_load``
     - float
     - Active power demand in MW
   * - ``Reactive_load``
     - float
     - Reactive power demand in MVAr
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
     - x-coordinate preferably in decimal longitude
   * - ``y_coord``
     - float
     - y-coordinate preferably in decimal latitude

Example Usage:
    .. code-block:: python

        import pyflow_acdc as pyf
        # Create an AC node
        node = pyf.Node_AC('PQ', 1, 0,66, Power_Gained=0, Reactive_Gained=0, Power_load=100, Reactive_load=50, name='Bus1', Umin=0.9, Umax=1.1,Gs=0,Bs=0)
        


   
AC branch modelling
^^^^^^^^^^^^^^^^^

.. figure:: ../images/AC_line_pi.svg
   :width: 400
   :alt: AC line model
   :align: center

   AC line π-model

The AC branch is modeled with pi model from [1]_:


Class Reference: :class:`pyflow_acdc.Classes.Line_AC`

Key Attributes:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1  

   * - Attribute
     - Type
     - Description  
   * - ``fromNode``
     - :class:`Node_AC`
     - The starting node of the line
   * - ``toNode``
     - :class:`Node_AC`
     - The ending node of the line
   * - ``Resistance``
     - (float)
     - Resistance of the line in pu
   * - ``Reactance``
     - (float)
     - Reactance of the line in pu
   * - ``Conductance``
     - (float)
     - Conductance of the line in pu
   * - ``Susceptance``
     - (float)
     - Susceptance of the line in pu
   * - ``MVA_rating``
     - (float)
     - MVA rating of the line
   * - ``Length_km``
     - (float)
     - Length of the line in km
   * - ``m``
     - (float)
     - Number of conductors in the line
   * - ``shift``
     - (float)
     - Phase shift of the line in radians
   * - ``N_cables``
     - (int)
     - Number of cables in the line
   * - ``name``
     - (str)
     - Name of the line
   * - ``geometry``
     - (str)
     - Geometry of the line
   * - ``isTf``
     - (bool)
     - True if the line is a transformer, False otherwise
   * - ``S_base``
     - (float)
     - Base power of the line in MVA
   * - ``Cable_type``
     - (str)
     - Type of cable in the line

Example Usage:
    .. code-block:: python

        import pyflow_acdc as pyf
        # Create an AC node
        node1 = pyf.Node_AC('PQ', 1, 0,66, Power_Gained=0.5, name='Bus1')
        node2 = pyf.Node_AC('Slack', 1, 0,66,name='Bus2')

        # In pu
        line_1 = pyf.Line_AC(node1, node2, Resistance=0.01, Reactance=0.1, Conductance=0, Susceptance=0, MVA_rating=100, N_cables=1, name='Line1')
    
        # Or by cable type in database

        line_2 = pyf.Line_AC(node1, node2, S_base=100, Length_km=100, Cable_type='NREL_XLPE_630mm_66kV')

DC System Modelling
------------------- 

DC node modelling
^^^^^^^^^^^^^^^^^

.. figure:: ../images/DC_node_model.svg
   :width: 300
   :alt: DC node model
   :align: center

   DC node equivalent circuit

The AC node is modeled using voltage :math:`U_d` where:

* :math:`U_d` is the voltage magnitude in pu
* :math:`P_{rg}` is the active power injection of renewable generation in pu
* :math:`P_{cn}` is the active power injection of converter in pu
* :math:`P_l` is the active power demand in pu

Class Reference: :class:`pyflow_acdc.Classes.Node_DC`

Key Attributes:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``node_type``
     - str
     - Node type ('Slack' or 'P' or 'Droop' or 'PAC')
   * - ``Voltage_0``
     - float
     - Initial voltage magnitude in pu
   * - ``Power_Gained``
     - float
     - Active power injection in pu
   * - ``Power_load``
     - float
     - Active power demand in pu
   * - ``kV_base``
     - float
     - Base voltage in kV
   * - ``Umin``
     - float
     - Minimum voltage magnitude in p.u.
   * - ``Umax``
     - float
     - Maximum voltage magnitude in p.u.
   * - ``x_coord``
     - float
     - x-coordinate, preferably in longitude decimal format
   * - ``y_coord``
     - float
     - y-coordinate, preferably in latitude decimal format

Example Usage:
    .. code-block:: python

        import pyflow_acdc as pyf
        # Create an DC node
        node = pyf.Node_DC('P', 1, 0,0,525,name='Bus1')


DC line modelling
^^^^^^^^^^^^^^^^^

.. figure:: ../images/DC_line.svg
   :width: 400
   :alt: DC line model
   :align: center

   DC line model

Key Attributes:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``fromNode``
     - Node_DC
     - The starting node of the line
   * - ``toNode``
     - Node_DC
     - The ending node of the line
   * - ``Resistance``
     - float
     - Resistance of the line in pu
   * - ``MW_rating``
     - float
     - MW rating of the line
   * - ``km``
     - float
     - Length of the line in km
   * - ``polarity``
     - str
     - Polarity of the line ('m' or 'b' or 'sm')
   * - ``N_cables``
     - int
     - Number of parallelcables in the line
   * - ``Cable_type``
     - str
     - Type of cable in the line
   * - ``S_base``
     - float
     - Base power of the line in MVA



ACDC Converter Modelling
------------------------

.. figure:: ../images/Converter_model.svg
   :width: 400
   :alt: ACDC converter model
   :align: center

   ACDC converter equivalent circuit 

Class Reference: :class:`pyflow_acdc.Classes.AC_DC_converter`

Key Attributes:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Attribute
     - Type
     - Description
   * - ``AC_type``
     - str
     - Type of AC node ('Slack' or 'PV' or 'PQ')
   * - ``DC_type``
     - str
     - Type of DC node ('Slack' or 'P' or 'Droop' or 'PAC')
   * - ``AC_node``
     - Node_AC
     - AC node connected to the converter
   * - ``DC_node``
     - Node_DC
     - DC node connected to the converter
   * - ``P_AC``
     - float
     - Active power injection in AC node in pu
   * - ``Q_AC``
     - float
     - Reactive power injection in AC node in pu
   * - ``P_DC``
     - float
     - Active power injection in DC node in pu
   * - ``Transformer_resistance``
     - float
     - Transformer resistance in pu
   * - ``Transformer_reactance``
     - float
     - Transformer reactance in pu
   * - ``Phase_Reactor_R``
     - float
     - Phase reactor resistance in pu
   * - ``Phase_Reactor_X``
     - float
     - Phase reactor reactance in pu
   * - ``Filter``
     - float
     - Filter in pu
   * - ``Droop``
     - float
     - Droop in pu
   * - ``kV_base``
     - float
     - Base voltage in kV
   * - ``MVA_max``
     - float
     - Maximum MVA rating of the converter
   * - ``nConvP``
     - float
     - Number of parallel converters
   * - ``polarity``
     - int
     - Polarity of the converter (1 or -1)
   * - ``lossa``
     - float
     - No load loss factor for active power
   * - ``lossb``
     - float
     - Linear currentr loss factor
   * - ``losscrect``
     - float
     - Switching loss factor for rectifier
   * - ``losscinv``
     - float
     - Switching loss factor for inverter
   * - ``Ucmin``
     - float
     - Minimum voltage magnitude in pu
   * - ``Ucmax``
     - float
     - Maximum voltage magnitude in pu
   * - ``name``
     - str
     - Name of the converter

.. include:: ../refs.rst
