Usage Guide
===========

Basic Concepts
--------------
Explain core concepts here...

Creating a Grid
---------------
This is the basic way to create a grid. This grid is the same as running MATACDC case5_stagg and case5_stagg_MTDC [1]_.

.. figure:: /images/Stagg5MATACDC.svg
   :alt: Case 5 Stagg
   :align: center

   Case 5 Stagg Grid


.. code-block:: python

    import pyflow_acdc as pyf


    S_base = 100

    AC_node_1 = pyf.Node_AC(node_type='Slack', Voltage_0=1.06, theta_0=0, kV_base=345)
    AC_node_2 = pyf.Node_AC(node_type='PV', Voltage_0=1, theta_0=0.1, kV_base=345,Power_Gained=0.4,Power_load=0.2,Reactive_load=0.1)
    AC_node_3 = pyf.Node_AC(node_type='PQ', Voltage_0=1, theta_0=0.1, kV_base=345,Power_load=0.45,Reactive_load=0.15)
    AC_node_4 = pyf.Node_AC(node_type='PQ', Voltage_0=1, theta_0=0.1, kV_base=345,Power_load=0.4,Reactive_load=0.05)
    AC_node_5 = pyf.Node_AC(node_type='PQ', Voltage_0=1, theta_0=0.1, kV_base=345,Power_load=0.6,Reactive_load=0.1)

    AC_line_1 = pyf.Line_AC(AC_node_1, AC_node_2,Resistance=0.02,Reactance=0.06,Susceptance=0.06,MVA_rating=150)
    AC_line_2 = pyf.Line_AC(AC_node_1, AC_node_3,Resistance=0.08,Reactance=0.24,Susceptance=0.05,MVA_rating=100)
    AC_line_3 = pyf.Line_AC(AC_node_2, AC_node_3,Resistance=0.06,Reactance=0.18,Susceptance=0.04,MVA_rating=100)
    AC_line_4 = pyf.Line_AC(AC_node_2, AC_node_4,Resistance=0.06,Reactance=0.18,Susceptance=0.04,MVA_rating=100)
    AC_line_5 = pyf.Line_AC(AC_node_2, AC_node_5,Resistance=0.04,Reactance=0.12,Susceptance=0.03,MVA_rating=100)
    AC_line_6 = pyf.Line_AC(AC_node_3, AC_node_4,Resistance=0.01,Reactance=0.03,Susceptance=0.02,MVA_rating=100)   
    AC_line_7 = pyf.Line_AC(AC_node_4, AC_node_5,Resistance=0.08,Reactance=0.24,Susceptance=0.05,MVA_rating=100)



    DC_node_1 = pyf.Node_DC(node_type='P', Voltage_0=1,kV_base=345)
    DC_node_2 = pyf.Node_DC(node_type='Slack', Voltage_0=1,kV_base=345)
    DC_node_3 = pyf.Node_DC(node_type='P', Voltage_0=1,kV_base=345)

    DC_line_1 = pyf.Line_DC(DC_node_1, DC_node_2,Resistance=0.052,MW_rating=100,polarity='sm')
    DC_line_2 = pyf.Line_DC(DC_node_2, DC_node_3,Resistance=0.052,MW_rating=100,polarity='sm')
    DC_line_3 = pyf.Line_DC(DC_node_1, DC_node_3,Resistance=0.073,MW_rating=100,polarity='sm')

    Converter_1 = pyf.AC_DC_converter('PQ', 'PAC'  , AC_node_2, DC_node_1, P_AC=-0.6, Q_AC=-0.4, P_DC=0, Transformer_resistance=0.0015, Transformer_reactance=0.121, Phase_Reactor_R=0.0001, Phase_Reactor_X=0.16428, Filter=0.0887, Droop=0, kV_base=345, MVA_max=120)
    Converter_2 = pyf.AC_DC_converter('PV', 'Slack', AC_node_3, DC_node_2, Transformer_resistance=0.0015, Transformer_reactance=0.121, Phase_Reactor_R=0.0001, Phase_Reactor_X=0.16428, Filter=0.0887, Droop=0, kV_base=345, MVA_max=120)
    Converter_3 = pyf.AC_DC_converter('PQ', 'PAC'  , AC_node_5, DC_node_3, P_AC=0.35, Q_AC=0.05, Transformer_resistance=0.0015, Transformer_reactance=0.121, Phase_Reactor_R=0.0001, Phase_Reactor_X=0.16428, Filter=0.0887, Droop=0, kV_base=345, MVA_max=120)

    AC_nodes = [AC_node_1, AC_node_2, AC_node_3, AC_node_4, AC_node_5]
    DC_nodes = [DC_node_1, DC_node_2, DC_node_3]
    AC_lines = [AC_line_1, AC_line_2, AC_line_3, AC_line_4, AC_line_5, AC_line_6, AC_line_7]
    DC_lines = [DC_line_1, DC_line_2, DC_line_3]
    Converters = [Converter_1, Converter_2, Converter_3]


    grid = pyf.Grid(S_base,AC_nodes, AC_lines,Converters,DC_nodes, DC_lines)
    res= pyf.Results(grid,decimals=3)


    pyf.ACDC_sequential(grid)

    res.All()


Adding Components
---------------
Grids can also be built in the opposite order, creating the core grid first, then adding elements.

.. code-block:: python
    import pyflow_acdc as pyf

    grid = pyf.Grid(100)
    res = pyf.Results(grid)

    ac_node_1 = pyf.add_AC_node(grid,node_type='Slack', Voltage_0=1.06, theta_0=0, kV_base=345)
    ac_node_2 = pyf.add_AC_node(grid,node_type='PV', Voltage_0=1, theta_0=0.1, kV_base=345,Power_Gained=0.4,Power_load=0.2,Reactive_load=0.1)
    ac_node_3 = pyf.add_AC_node(grid,node_type='PQ', Voltage_0=1, theta_0=0.1, kV_base=345,Power_load=0.45,Reactive_load=0.15)
    ac_node_4 = pyf.add_AC_node(grid,node_type='PQ', Voltage_0=1, theta_0=0.1, kV_base=345,Power_load=0.4,Reactive_load=0.05)
    ac_node_5 = pyf.add_AC_node(grid,node_type='PQ', Voltage_0=1, theta_0=0.1, kV_base=345,Power_load=0.6,Reactive_load=0.1)

    ac_line_1 = pyf.add_line_AC(grid,ac_node_1,ac_node_2,r=0.02,x=0.06,b=0.06,MVA_rating=150)
    ac_line_2 = pyf.add_line_AC(grid,ac_node_1,ac_node_3,r=0.08,x=0.24,b=0.05,MVA_rating=100)
    ac_line_3 = pyf.add_line_AC(grid,ac_node_2,ac_node_3,r=0.06,x=0.18,b=0.04,MVA_rating=100)
    ac_line_4 = pyf.add_line_AC(grid,ac_node_2,ac_node_4,r=0.06,x=0.18,b=0.04,MVA_rating=100)
    ac_line_5 = pyf.add_line_AC(grid,ac_node_2,ac_node_5,r=0.04,x=0.12,b=0.03,MVA_rating=100)
    ac_line_6 = pyf.add_line_AC(grid,ac_node_3,ac_node_4,r=0.01,x=0.03,b=0.02,MVA_rating=100)
    ac_line_7 = pyf.add_line_AC(grid,ac_node_4,ac_node_5,r=0.08,x=0.24,b=0.05,MVA_rating=100)

    dc_node_1 = pyf.add_DC_node(grid,node_type='P', Voltage_0=1,kV_base=345)
    dc_node_2 = pyf.add_DC_node(grid,node_type='Slack', Voltage_0=1,kV_base=345)
    dc_node_3 = pyf.add_DC_node(grid,node_type='P', Voltage_0=1,kV_base=345)

    dc_line_1 = pyf.add_line_DC(grid,dc_node_1,dc_node_2,r=0.052,MW_rating=100,polarity='sm')
    dc_line_2 = pyf.add_line_DC(grid,dc_node_2,dc_node_3,r=0.052,MW_rating=100,polarity='sm')
    dc_line_3 = pyf.add_line_DC(grid,dc_node_1,dc_node_3,r=0.073,MW_rating=100,polarity='sm')


    converter_1 = pyf.add_ACDC_converter(grid,ac_node_2, dc_node_1,'PQ', 'PAC' , P_AC_MW=-60, Q_AC_MVA=-40, Transformer_resistance=0.0015, Transformer_reactance=0.121, Phase_Reactor_R=0.0001, Phase_Reactor_X=0.16428, Filter=0.0887, Droop=0, kV_base=345, MVA_max=120)
    converter_2 = pyf.add_ACDC_converter(grid,ac_node_3, dc_node_2,'PV', 'Slack', Transformer_resistance=0.0015, Transformer_reactance=0.121, Phase_Reactor_R=0.0001, Phase_Reactor_X=0.16428, Filter=0.0887, Droop=0, kV_base=345, MVA_max=120)
    converter_3 = pyf.add_ACDC_converter(grid,ac_node_5, dc_node_3,'PQ', 'PAC'  , P_AC_MW=35, Q_AC_MVA=5, Transformer_resistance=0.0015, Transformer_reactance=0.121, Phase_Reactor_R=0.0001, Phase_Reactor_X=0.16428, Filter=0.0887, Droop=0, kV_base=345, MVA_max=120)

    pyf.ACDC_sequential(grid)
    res.All()


Running a Power Flow
--------------------
Examples of running a power flow...

.. code-block:: python

    import pyflow_acdc as pyf

    [grid,res]=pyf.PEI_grid()

    pyf.ACDC_sequential(grid,QLimit=False)

    res.All()
    print ('------')
  


Running an Optimal Power Flow
-----------------------------
To run this, you need to have the OPF optional installed. This includes the following packages:
- pyomo
- ipopt


Examples of running an optimal power flow...

.. code-block:: python

    import pyflow_acdc as pyf
    obj = {'Energy_cost'  : 1}

    [grid,res]=pyf.case39_acdc()

    model, timing_info, [model_res,solver_stats] = pyf.OPF_ACDC(grid,ObjRule={'obj':{'w':1}})

    res.All()
    print ('------')


Available test cases:

For Power Flow:
- pyf.StaggSMATACDC()
- pyf.PEI_grid()

For Optimal Power Flow:

- pyf.case_ACTIVSg2000()
- pyf.case24_3zones_acdc()
- pyf.case39_acdc()
- pyf.case39()
- pyf.case118()
- pyf.NS_MTDC()
- pyf.NS_SII()
- pyf.pglib_opf_case14_ieee()
- pyf.pglib_opf_case300_ieee()
- pyf.pglib_opf_case588_sdet_acdc()
- pyf.StaggSMATACDC()





References
----------


.. [1] J. Beerten and R. Belmans, "MatACDC - an open source software tool for steady-state analysis and operation of HVDC grids," 11th IET International Conference on AC and DC Power Transmission, Birmingham, 2015, pp. 1-9, doi: 10.1049/cp.2015.0061. keywords: {Steady-state analysis;HVDC grids;AC/DC systems;power flow modelling},



