.. _csv_import:

CSV files for importing data
============================

In this page the column names for importing data from CSV files are described. It is important to note that column names are case sensitive. The CSV can then be used for the following functions:

.. autofunction:: pyflow_acdc.create_grid_from_data
   :noindex:

.. autofunction:: pyflow_acdc.extend_grid_from_data
   :noindex:


.. themed-figure:: stagg5
   :alt: Case 5 Stagg
   :align: center

   Case 5 Stagg Grid

The examples below are based on the Case 5 Stagg Grid in MATACDC [1]_.

Required CSV Files
-------------------

Select your data input format:

Data in per unit (pu)
^^^^^^^^^^^^^^^^^^^^^^

.. raw:: html

   <details>
   <summary><b>Data in per unit (pu)</b></summary>

.. list-table:: AC Nodes
   :widths: 20 20 20
   :header-rows: 1

   * - Variable
     - Column Name
     - Default Value
   * - Node ID
     - Node_id
     - Required
   * - Type
     - type
     - PQ
   * - Base Voltage (kV)
     - kV_base
     - Required
   * - Initial Voltage (pu)
     - Voltage_0
     - 1.01
   * - Initial Angle (rad)
     - theta_0
     - 0.01
   * - Generation Power (pu)
     - Power_Gained
     - 0
   * - Generation Reactive (pu)
     - Reactive_Gained
     - 0
   * - Load Power (pu)
     - Power_load
     - 0
   * - Load Reactive (pu)
     - Reactive_load
     - 0
   * - Min Voltage (pu)
     - Umin
     - 0.9
   * - Max Voltage (pu)
     - Umax
     - 1.1
   * - X Coordinate
     - x_coord
     - None
   * - Y Coordinate
     - y_coord
     - None
   * - Shunt Susceptance (pu)
     - Bs
     - 0
   * - Shunt Conductance (pu)
     - Gs
     - 0
   * - Geometry
     - geometry
     - None

.. list-table:: AC Branch
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Line ID
     - Line_id
     - Required
   * - From Node
     - fromNode
     - Required
   * - To Node
     - toNode
     - Required
   * - Resistance (pu)
     - r
     - 0.00001
   * - Reactance (pu)
     - x
     - 0.00001
   * - Conductance
     - g
     - 0
   * - Susceptance
     - b
     - 0
   * - MVA Rating
     - MVA_rating
     - S_base*1.05
   * - Length
     - Length_km
     - 1
   * - Transformer Ratio
     - m
     - 1
   * - Phase Shift
     - shift
     - 0
   * - Geometry
     - geometry
     - None

.. list-table:: DC Nodes
   :widths: 20 20 20
   :header-rows: 1

   * - Variable
     - Column Name
     - Default Value
   * - Node ID
     - Node_id  
     - Required 
   * - Node Type
     - type
     - P
   * - Base Voltage (kV)
     - kV_base
     - Required
   * - Initial Voltage (pu)
     - Voltage_0
     - 1.01
   * - Active Power Generation (pu)
     - Power_Gained
     - 0
   * - Active Power Load (pu)
     - Power_Load
     - 0
   * - Minimum Voltage (pu)
     - Umin
     - 0.95
   * - Maximum Voltage (pu)
     - Umax
     - 1.05
   * - X Coordinate
     - x_coord
     - None
   * - Y Coordinate
     - y_coord
     - None
   * - Geometry
     - geometry
     - None

.. list-table:: DC Branch
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Line ID
     - Line_id
     - Required
   * - From Node
     - fromNode
     - Required
   * - To Node
     - toNode
     - Required
   * - Resistance (pu)
     - r
     - 0.0001
   * - MVA Rating
     - MVA_rating
     - 99999
   * - Length (km)
     - Length_km
     - 1
   * - Mono/Bi-Polar (m/sm/b)
     - Mono_Bi_polar
     - m
   * - Number of parallel branches
     - N_cables
     - 1
   * - Geometry
     - geometry
     - None

.. list-table:: Converter
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value    
   * - Converter ID
     - Conv_id
     - Required
   * - AC node
     - AC_node
     - Required
   * - DC node
     - DC_node
     - Required
   * - AC Type
     - AC_type
     - Takes type from AC node
   * - DC Type
     - DC_type
     - Takes type from DC node
   * - Active Power AC setpoint (pu)
     - P_AC
     - 0
   * - Reactive Power AC setpoint (pu)
     - Q_AC
     - 0
   * - Active Power DC setpoint (pu)
     - P_DC
     - 0
   * - Transformer Resistance (pu)
     - T_r
     - 0
   * - Transformer Reactance (pu)
     - T_x
     - 0
   * - Phase Reactor Resistance (pu)
     - PR_r
     - 0
   * - Phase Reactor Reactance (pu)
     - PR_x
     - 0
   * - Filter Susceptance (pu)
     - Filter_b
     - 0
   * - Droop coefficient
     - Droop
     - 0
   * - AC Base Voltage (kV)
     - AC_kV_base
     - Takes from AC node
   * - MVA Rating
     - MVA_rating
     - 99999
   * - Minimum Voltage (pu)
     - Ucmin
     - 0.85
   * - Maximum Voltage (pu)
     - Ucmax
     - 1.2
   * - Number of converters
     - Nconverter
     - 1
   * - Polarity
     - pol
     - 1
   * - Geometry
     - geometry
     - None


Here are example CSV files from a 5-bus test system in per unit:

**AC Node Data (AC_node_data.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/AC_node_data.csv
   :language: text

**AC Line Data (AC_line_data.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/AC_line_data.csv
   :language: text

**DC Node Data (DC_node_data.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/DC_node_data.csv
   :language: text

**DC Line Data (DC_line_data.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/DC_line_data.csv
   :language: text

**Converter Data (Converter_data.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/Converter_data.csv
   :language: text

**Example Code**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/01_data_in_per_unit_pu.py
   :language: python
   :lines: 2-


.. raw:: html

   </details>

Data in Ohms
^^^^^^^^^^^^^^  

.. raw:: html
  
   <details>
   <summary><b>Data in Ohms</b></summary>

Data in Ohms affects AC and DC branch components, where the user specifies the absolute resistance. It is assumed that the user has taken into account length, parallel branches and so on. And the resistance, reactance, conductance and susceptance are given for the whole branch.

**AC Node Data (AC_node_data_Ohm.csv)**

.. list-table:: AC Nodes
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Node ID
     - Node_id
     - Required
   * - Node Type
     - type
     - PQ
   * - Base Voltage (kV)
     - kV_base
     - Required
   * - Initial Voltage (pu)
     - Voltage_0
     - 1.01
   * - Initial Angle (rad)
     - theta_0
     - 0.01
   * - Active Power Generation (MW)
     - Power_Gained
     - 0
   * - Reactive Power Generation (MVAR)
     - Reactive_Gained
     - 0
   * - Active Power Load (MW)
     - Power_load
     - 0
   * - Reactive Power Load (MVAR)
     - Reactive_load
     - 0
   * - Minimum Voltage (pu)
     - Umin
     - 0.9
   * - Maximum Voltage (pu)
     - Umax
     - 1.1
   * - X Coordinate
     - x_coord
     - None
   * - Y Coordinate
     - y_coord
     - None
   * - Shunt Susceptance
     - Bs
     - 0
   * - Shunt Conductance
     - Gs
     - 0
   * - Geometry
     - geometry
     - None


**AC Line Data (AC_line_data_Ohm.csv)**

.. list-table:: AC Branch
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Line ID
     - Line_id
     - Required
   * - From Node
     - fromNode
     - Required
   * - To Node
     - toNode
     - Required
   * - Resistance (Ω)
     - R
     - 0.0001
   * - Reactance (Ω)
     - X
     - 0.0001
   * - Conductance (S)
     - G
     - 0
   * - Susceptance (S)
     - B
     - 0
   * - MVA Rating
     - MVA_rating
     - 99999
   * - Length (km)
     - Length_km
     - 1
   * - Transformer Ratio
     - m
     - 1
   * - Phase Shift (rad)
     - shift
     - 0
   * - Geometry
     - geometry
     - None
            
**DC Node Data (DC_node_data_Ohm.csv)**

.. list-table:: DC Nodes
   :widths: 20 20 20
   :header-rows: 1

   * - Variable
     - Column Name
     - Default Value
   * - Node ID
     - Node_id  
     - Required 
   * - Node Type
     - type
     - P
   * - Base Voltage (kV)
     - kV_base
     - Required
   * - Initial Voltage (pu)
     - Voltage_0
     - 1.01
   * - Active Power Generation (MW)
     - Power_Gained
     - 0
   * - Active Power Load (MW)
     - Power_Load
     - 0
   * - Minimum Voltage (pu)
     - Umin
     - 0.95
   * - Maximum Voltage (pu)
     - Umax
     - 1.05
   * - X Coordinate
     - x_coord
     - None
   * - Y Coordinate
     - y_coord
     - None
   * - Geometry
     - geometry
     - None

**DC Line Data (DC_line_data_Ohm.csv)**

.. list-table:: DC Branch
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Line ID
     - Line_id
     - Required
   * - From Node
     - fromNode
     - Required
   * - To Node
     - toNode
     - Required
   * - Resistance (Ω)
     - R
     - 0.0095*km
   * - Length (km)
     - Length_km
     - 1
   * - Mono/Bi-Polar (m/sm/b)
     - Mono_Bi_polar
     - m
   * - Number of parallel branches
     - N_cables
     - 1
   * - Geometry
     - geometry
     - None

**Converter Data (Converter_data_Ohm.csv)**

.. list-table:: Converter
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value    
   * - Converter ID     
     - Conv_id
     - Required
   * - AC node
     - AC_node
     - Required
   * - DC node
     - DC_node
     - Required
   * - AC Type
     - AC_type
     - Takes type from AC node  
   * - DC Type
     - DC_type
     - Takes type from DC node
   * - Active Power AC setpoint (MW)
     - P_MW_AC
     - 0
   * - Reactive Power AC setpoint (MVAR)
     - Q_AC
     - 0
   * - Active Power DC setpoint (MW)
     - P_MW_DC
     - 0    
   * - Transformer Resistance (Ω)
     - T_R_Ohm
     - 0
   * - Transformer Reactance (mH)
     - T_X_mH
     - 0
   * - Phase Reactor Resistance (Ω)
     - PR_R_Ohm
     - 0
   * - Phase Reactor Reactance (mH)
     - PR_X_mH
     - 0
   * - Filter Susceptance (μS)
     - Filter_uF
     - 0
   * - Droop coefficient
     - Droop
     - 0
   * - AC Base Voltage (kV)
     - AC_kV_base
     - Takes from AC node
   * - MVA Rating
     - MVA_rating
     - 99999
   * - Minimum Voltage (pu)
     - Ucmin
     - 0.85
   * - Maximum Voltage (pu)
     - Ucmax
     - 1.2
   * - Number of converters
     - Nconverter
     - 1
   * - Polarity
     - pol
     - 1
   * - Geometry 
     - geometry
     - None

**Example CSV Files**

Here are example CSV files from a 5-bus test system using the data in Ohm values:

**AC Node Data (AC_node_data_Ohm.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/AC_node_data_Ohm.csv
   :language: text


**AC Line Data (AC_line_data_Ohm.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/AC_line_data_Ohm.csv
   :language: text


**DC Node Data (DC_node_data_Ohm.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/DC_node_data_Ohm.csv
   :language: text


**DC Line Data (DC_line_data_Ohm.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/DC_line_data_Ohm.csv
   :language: text


**Converter Data (Converter_data_Ohm.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/Converter_data_Ohm.csv
   :language: text

**Example Code**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/02_data_in_ohms.py
   :language: python
   :lines: 2-


.. raw:: html

   </details>


Data in Real values
^^^^^^^^^^^^^^^^^^^^

.. raw:: html 

   <details>
   <summary><b>Data in Real values</b></summary>

.. list-table:: AC Nodes
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Node ID
     - Node_id
     - Required
   * - Node Type
     - type
     - PQ
   * - Base Voltage (kV)
     - kV_base
     - Required
   * - Initial Voltage (pu)
     - Voltage_0
     - 1.01
   * - Initial Angle (rad)
     - theta_0
     - 0.01
   * - Active Power Generation (MW)
     - Power_Gained
     - 0
   * - Reactive Power Generation (MVAR)
     - Reactive_Gained
     - 0
   * - Active Power Load (MW)
     - Power_load
     - 0
   * - Reactive Power Load (MVAR)
     - Reactive_load
     - 0
   * - Minimum Voltage (pu)
     - Umin
     - 0.9
   * - Maximum Voltage (pu)
     - Umax
     - 1.1
   * - X Coordinate
     - x_coord
     - None
   * - Y Coordinate
     - y_coord
     - None
   * - Shunt Susceptance
     - Bs
     - 0
   * - Shunt Conductance
     - Gs
     - 0
   * - Geometry
     - geometry
     - None

.. list-table:: AC Branch
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Line ID
     - Line_id
     - Required
   * - From Node
     - fromNode
     - Required
   * - To Node
     - toNode
     - Required
   * - Resistance (Ω/km)
     - R_Ohm_km
     - Required
   * - Inductance (mH/km)
     - L_mH_km
     - Required
   * - Capacitance (μF/km)
     - C_uF_km
     - 0
   * - Conductance (μS/km)
     - G_uS_km
     - 0
   * - Current Rating (A)
     - A_rating
     - 9999
   * - Length
     - Length_km
     - 1
   * - Number of parallel branches
     - N_cables
     - 1
   * - Transformer Ratio
     - m
     - 1
   * - Phase Shift
     - shift
     - 0
   * - Geometry
     - geometry
     - None
.. list-table:: DC Nodes
   :widths: 20 20 20
   :header-rows: 1

   * - Variable
     - Column Name
     - Default Value
   * - Node ID
     - Node_id  
     - Required 
   * - Node Type
     - type
     - P
   * - Base Voltage
     - kV_base
     - Required
   * - Initial Voltage (pu)
     - Voltage_0
     - 1.01
   * - Active Power Generation (MW)
     - Power_Gained
     - 0
   * - Active Power Load (MW)
     - Power_Load
     - 0
   * - Minimum Voltage
     - Umin
     - 0.95
   * - Maximum Voltage
     - Umax
     - 1.05
   * - X Coordinate
     - x_coord
     - None
   * - Y Coordinate
     - y_coord
     - None
   * - Geometry
     - geometry
     - None


.. list-table:: DC Branch
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value
   * - Line ID
     - Line_id
     - Required
   * - From Node
     - fromNode
     - Required
   * - To Node
     - toNode
     - Required
   * - Resistance (Ω/km)    
     - R_Ohm_km
     - 0.0095
   * - Current Rating (A)
     - A_rating
     - 9999
   * - Number of parallel branches
     - N_cables
     - 1
   * - Length (km)
     - Length_km
     - 1
   * - Polarity (m/sm/b)
     - Mono_Bi_polar
     - m
   * - Geometry
     - geometry
     - None
     
.. list-table:: Converter
   :widths: 20 20 20
   :header-rows: 1

   * - Description
     - Column Name
     - Default Value    
   * - Converter ID     
     - Conv_id
     - Required
   * - AC node
     - AC_node
     - Required
   * - DC node
     - DC_node
     - Required
   * - AC Type
     - AC_type
     - Takes type from AC node  
   * - DC Type
     - DC_type
     - Takes type from DC node
   * - Active Power AC setpoint (MW)
     - P_MW_AC
     - 0
   * - Reactive Power AC setpoint (MVAR)
     - Q_MVA_AC
     - 0
   * - Active Power DC setpoint (MW)
     - P_MW_DC
     - 0    
   * - Transformer Resistance (Ω)
     - T_R_Ohm
     - 0
   * - Transformer Reactance (mH)
     - T_X_mH
     - 0
   * - Phase Reactor Resistance (Ω)
     - PR_R_Ohm
     - 0
   * - Phase Reactor Reactance (mH)
     - PR_X_mH
     - 0
   * - Filter Susceptance (μS)
     - Filter_uF
     - 0
   * - Droop coefficient
     - Droop
     - 0
   * - AC Base Voltage (kV)
     - AC_kV_base
     - Takes from AC node
   * - MVA Rating
     - MVA_rating
     - 99999
   * - Minimum Voltage (pu)
     - Ucmin
     - 0.85
   * - Maximum Voltage (pu)
     - Ucmax
     - 1.2
   * - Number of converters
     - Nconverter
     - 1
   * - Polarity
     - pol
     - 1
   * - Geometry 
     - geometry
     - None

Here are example CSV files from a 5-bus test system using the data in Real values:

**AC Node Data (AC_node_data_Real.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/AC_node_data_Real.csv
   :language: text

**AC Line Data (AC_line_data_Real.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/AC_line_data_Real.csv
   :language: text

**DC Node Data (DC_node_data_Real.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/DC_node_data_Real.csv
   :language: text


**DC Line Data (DC_line_data_Real.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/DC_line_data_Real.csv
   :language: text

**Converter Data (Converter_data_Real.csv)**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/data/Converter_data_Real.csv
   :language: text

**Example Code**

.. literalinclude:: ../../pyflow_tests/doc_examples/csv_import/03_data_in_real_values.py
   :language: python
   :lines: 2-


.. raw:: html

   </details>

**References**


.. [1] J. Beerten and R. Belmans, "MatACDC - an open source software tool for steady-state analysis and operation of HVDC grids," 11th IET International Conference on AC and DC Power Transmission, Birmingham, 2015, pp. 1-9, doi: 10.1049/cp.2015.0061. keywords: {Steady-state analysis;HVDC grids;AC/DC systems;power flow modelling},




