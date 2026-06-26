"""Post-creation grid modification.

Adds or mutates elements of an existing :class:`~pyflow_acdc.Classes.Grid`
(``add_*`` helpers, line-type conversions, price-zone assignment, and
time/investment-series wiring).

Owns: mutation of an already-constructed grid.
Does not own: initial grid construction (see ``grid_creator``).
"""

import pandas as pd
import numpy as np
import yaml
import re
import json
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from shapely.wkt import loads

from .Classes import (
    AC_DC_converter, Cable_options, DCDC_converter, Exp_Line_AC, Gen_AC,
    Gen_DC, Line_AC, Line_DC, MTDCPrice_Zone, Node_AC, Node_DC,
    OffshorePrice_Zone, Price_Zone, Ren_Source, Ren_source_zone,
    rec_Line_AC, Size_selection, TF_Line_AC, TimeSeries,
)
from .constants import (
    SQRT_3,
    MAX_RATING_PLACEHOLDER,
    DEFAULT_V_MIN_DC,
    DEFAULT_V_MAX_DC,
    NodeType,
    DEFAULT_GEN_TYPE,
    CableType,
    DataInput,
    Polarity,
    AcDcSide,
    PricingStrategy,
    TSType,
    TS_RENEWABLE_TYPES,
)
from .grid_analysis import (
    pol2cart,
    cart2pol,
    pol2cartz,
    cartz2pol,
    cable_parameters,
    converter_parameters,
    analyse_grid,
    grid_state,
    current_fuel_type_distribution,
)

from pathlib import Path

"""
"""

__all__ = [
    # Add grid Elements
    'add_AC_node',
    'add_DC_node',
    'add_line_AC',
    'add_line_DC',
    'add_ACDC_converter',
    'add_DCDC_converter',
    'add_gen',
    'add_gen_DC',
    'add_extgrid',
    'add_RenSource',
    'add_generators',
    'add_cable_option',
    'add_line_sizing',

    # Add Zones
    'add_RenSource_zone',
    'add_price_zone',
    'add_MTDC_price_zone',
    'add_offshore_price_zone',

    # Add Time Series
    'add_TimeSeries',
    'time_series_dict',

    #Add investment series
    'add_inv_series',
    'add_gen_mix_limits',
    'create_gen_limit_csv_template',
    'create_inv_csv_template',

    # Line Modifications
    'change_line_AC_to_expandable',
    'change_line_AC_to_reconducting',
    'change_line_AC_to_tap_transformer',

    # Zone Assignments
    'assign_RenToZone',
    'assign_nodeToPrice_Zone',
    'assign_ConvToPrice_Zone',
    'assign_lineToCable_options',

    # Parameter Calculations
    'cable_parameters',
    'converter_parameters',

    # Utility Functions
    'pol2cart',
    'cart2pol',
    'pol2cartz',
    'cartz2pol',

    # Analysis
    'analyse_grid',
    'grid_state',
    'import_orbit_cables',
    'expand_cable_database',
    'current_fuel_type_distribution'
]

"Add main components"

def add_AC_node(grid, kV_base,node_type='PQ',Voltage_0=1.01, theta_0=0.01, Power_Gained=0, Reactive_Gained=0, Power_load=0, Reactive_load=0, name=None, Umin=0.9, Umax=1.1,Gs= 0,Bs=0,x_coord=None,y_coord=None,geometry=None):
    """Append an AC bus to ``grid.nodes_AC``.

    Parameters
    ----------
    grid : Grid
        Grid to modify (mutated in place).
    kV_base : float
        Base voltage in kV.
    node_type : str, optional
        ``'PQ'``, ``'PV'``, or ``'Slack'``.
    Voltage_0 : float, optional
        Initial voltage magnitude in p.u.
    theta_0 : float, optional
        Initial voltage angle in rad.
    Power_Gained, Reactive_Gained : float, optional
        Initial generation setpoints in p.u.
    Power_load, Reactive_load : float, optional
        Initial load in p.u.
    name : str, optional
        Bus name; auto-generated if omitted.
    Umin, Umax : float, optional
        Voltage magnitude limits in p.u.
    Gs, Bs : float, optional
        Shunt conductance and susceptance in p.u.
    x_coord, y_coord : float, optional
        Plot coordinates; overridden when ``geometry`` is set.
    geometry : shapely.Geometry or str, optional
        Shapely geometry or WKT string; sets ``x_coord``/``y_coord`` from centroid.

    Returns
    -------
    Node_AC
        Created node.

    Examples
    --------
    >>> node = pyf.add_AC_node(grid, kV_base=400, name='bus1', node_type='PQ')
    """
    node = Node_AC( node_type, Voltage_0, theta_0,kV_base, Power_Gained, Reactive_Gained, Power_load, Reactive_load, name, Umin, Umax,Gs,Bs,x_coord,y_coord)
    if geometry is not None:
       if isinstance(geometry, str):
            geometry = loads(geometry)
       node.geometry = geometry
       node.x_coord = geometry.x
       node.y_coord = geometry.y

    grid.nodes_AC.append(node)

    return node

def add_DC_node(grid,kV_base,node_type='P', Voltage_0=1.01, Power_Gained=0, Power_load=0, name=None,Umin=DEFAULT_V_MIN_DC, Umax=DEFAULT_V_MAX_DC,x_coord=None,y_coord=None,geometry=None):
    """Append a DC bus to ``grid.nodes_DC``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    kV_base : float
        Base voltage in kV.
    node_type : str, optional
        ``'P'``, ``'Slack'``, or ``'Droop'``.
    Voltage_0 : float, optional
        Initial voltage in p.u.
    Power_Gained, Power_load : float, optional
        Generation and load in p.u.
    name : str, optional
        Bus name.
    Umin, Umax : float, optional
        Voltage limits in p.u. Defaults use ``DEFAULT_V_MIN_DC`` /
        ``DEFAULT_V_MAX_DC``.
    x_coord, y_coord : float, optional
        Plot coordinates.
    geometry : shapely.Geometry or str, optional
        Shapely geometry or WKT string.

    Returns
    -------
    Node_DC
        Created node.

    Examples
    --------
    >>> node = pyf.add_DC_node(grid, kV_base=525, name='dc_bus1')
    """
    node = Node_DC(node_type, kV_base, Voltage_0, Power_Gained, Power_load, name,Umin, Umax,x_coord,y_coord)
    grid.nodes_DC.append(node)
    if geometry is not None:
       if isinstance(geometry, str):
            geometry = loads(geometry)
       node.geometry = geometry
       node.x_coord = geometry.x
       node.y_coord = geometry.y


    return node

def add_line_AC(grid, fromNode, toNode,MVA_rating=None, r=0, x=0, b=0, g=0,R_Ohm_km=None,L_mH_km=None, C_uF_km=0, G_uS_km=0, A_rating=None ,m=1, shift=0, name=None,tap_changer=False,Expandable=False,N_cables=1,Length_km=1,geometry=None,data_in=DataInput.PU,Cable_type:str =CableType.CUSTOM,update_grid=True):
    """Append an AC branch (line, expandable line, or tap transformer).

    ``fromNode`` and ``toNode`` may be :class:`~pyflow_acdc.Classes.Node_AC`
    objects or bus name strings.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    fromNode, toNode : Node_AC or str
        End buses.
    MVA_rating : float, optional
        Thermal rating in MVA (required for ``data_in='pu'`` or ``'Ohm'``).
    r, x, g, b : float, optional
        Series/shunt parameters; meaning depends on ``data_in``.
    R_Ohm_km, L_mH_km, C_uF_km, G_uS_km, A_rating : float, optional
        Physical cable data when ``data_in='Real'`` (``L_mH_km`` alone forces Real).
    m, shift : float, optional
        Transformer ratio and phase shift (rad).
    tap_changer : bool, optional
        If True, store as :class:`~pyflow_acdc.Classes.TF_Line_AC`.
    Expandable : bool, optional
        If True, store as :class:`~pyflow_acdc.Classes.Exp_Line_AC`.
    N_cables : int, optional
        Number of parallel circuits.
    Length_km : float, optional
        Line length in km.
    name : str, optional
        Branch name.
    geometry : shapely.Geometry or str, optional
        Line geometry for plotting.
    data_in : str, optional
        ``'pu'``, ``'Ohm'``, or ``'Real'`` (see :class:`~pyflow_acdc.constants.DataInput`).
    Cable_type : str, optional
        Named cable from the bundled database, or ``'Custom'``.
    update_grid : bool, optional
        Rebuild Y-bus and NetworkX graph when True.

    Returns
    -------
    Line_AC, Exp_Line_AC, or TF_Line_AC
        Created branch object.

    Notes
    -----
    For ``data_in='pu'``, pass ``r``, ``x``, ``g``, ``b`` in p.u. and ``MVA_rating``.
    For ``data_in='Ohm'``, pass ``r``, ``x``, ``g``, ``b`` in ohms (converted via
    ``Z_base``). For ``data_in='Real'``, pass ``R_Ohm_km``, ``L_mH_km``, etc.
    For a database cable, pass ``Cable_type`` and ``Length_km`` only.

    Examples
    --------
    >>> line = pyf.add_line_AC(grid, 'bus1', 'bus2', r=0.029, x=0.0032,
    ...                        b=0.0022, Length_km=10, MVA_rating=50)
    >>> line_db = pyf.add_line_AC(grid, 'bus1', 'bus2',
    ...     Cable_type='NREL_XLPE_185mm_66kV', Length_km=10)
    """

    fromNode = _look_up_node(grid, fromNode, ac_or_dc="AC")
    toNode = _look_up_node(grid, toNode, ac_or_dc="AC")

    kV_base=toNode.kV_base
    if L_mH_km is not None:
        data_in = DataInput.REAL
    if data_in == DataInput.OHM:
        Z_base = kV_base**2/grid.S_base

        Resistance_pu = r / Z_base if r!=0 else 0.00001
        Reactance_pu  = x  / Z_base if x!=0  else 0.00001
        Conductance_pu = g*Z_base
        Susceptance_pu = b*Z_base
    elif data_in== DataInput.REAL and Cable_type == CableType.CUSTOM:
       [Resistance_pu, Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating] = cable_parameters(grid.S_base, R_Ohm_km, L_mH_km, C_uF_km, G_uS_km, A_rating, kV_base, Length_km,N_cables=N_cables)
    else:
        Resistance_pu = r if r!=0 else 0.00001
        Reactance_pu  = x if x!=0  else 0.00001
        Conductance_pu = g
        Susceptance_pu = b


    if tap_changer:
        line = TF_Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating, kV_base,m, shift, name)
        grid.lines_AC_tf.append(line)
        if update_grid:
            grid.update_graph_ac()
    elif Expandable:
        line = Exp_Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating,Length_km,m, shift,N_cables, name,S_base=grid.S_base,Cable_type=Cable_type)
        grid.lines_AC_exp.append(line)
        if update_grid:
            grid.update_graph_ac()

    else:
        line = Line_AC(fromNode, toNode, Resistance_pu,Reactance_pu, Conductance_pu, Susceptance_pu, MVA_rating,Length_km,m, shift,N_cables, name,S_base=grid.S_base,Cable_type=Cable_type)

        grid.lines_AC.append(line)
        if update_grid:
            grid.create_Ybus_AC()
            grid.update_graph_ac()

    if geometry is not None:
       if isinstance(geometry, str):
            geometry = loads(geometry)
       line.geometry = geometry

    return line

def change_line_AC_to_expandable(grid, line_name,update_grid=True):
    """Convert an existing AC line into an expandable (TEP) branch.

    Parameters
    ----------
    grid : Grid
        Grid containing the line.
    line_name : str
        Name of the line in ``grid.lines_AC``.
    update_grid : bool, optional
        Rebuild Y-bus and graph when True.

    Returns
    -------
    Exp_Line_AC
        New expandable line.

    Raises
    ------
    ValueError
        If ``line_name`` is not found in ``grid.lines_AC``.
    """
    l = None
    for line_to_process in grid.lines_AC:
        if line_name == line_to_process.name:
            l = line_to_process
            break

    if l is not None:
        grid.lines_AC.remove(l)
        l.remove()
        line_vars = {
            'fromNode': l.fromNode,
            'toNode': l.toNode,
            'r': l.r,
            'x': l.x,
            'g': l.g,
            'b': l.b,
            'MVA_rating': l.MVA_rating,
            'Length_km': l.Length_km,
            'm': l.m,
            'shift': l.shift,
            'N_cables': l.N_cables,
            'name': l.name,
            'geometry': l.geometry,
            'S_base': l.S_base,
            'Cable_type': l.Cable_type
        }
        expandable_line = Exp_Line_AC(**line_vars)
        grid.lines_AC_exp.append(expandable_line)
        if update_grid:
            grid.update_graph_ac()

    # Reassign line numbers to ensure continuity
    for i, line in enumerate(grid.lines_AC):
        line.lineNumber = i

    for i, line in enumerate(grid.lines_AC_exp):
        line.lineNumber = i
    if update_grid:
        grid.create_Ybus_AC()

    if l is None:
        raise ValueError(f"Line '{line_name}' not found in grid.lines_AC")
    return expandable_line

def change_line_AC_to_reconducting(grid, line_name, r_new,x_new,g_new,b_new,MVA_rating_new,Life_time,base_cost,update_grid=True):
    """Convert an AC line into a reconductoring candidate.

    The original branch is kept as the inactive base case inside the new
    :class:`~pyflow_acdc.Classes.rec_Line_AC`` (``rec_branch=False`` until
    reconductoring is selected in OPF/TEP).

    Parameters
    ----------
    grid : Grid
        Grid containing the line.
    line_name : str
        Name of the line in ``grid.lines_AC``.
    r_new, x_new, g_new, b_new : float
        Reconductored branch parameters in p.u.
    MVA_rating_new : float
        New thermal rating in MVA.
    Life_time : float
        Economic lifetime in years.
    base_cost : float
        Investment cost of reconductoring.
    update_grid : bool, optional
        Rebuild Y-bus and graph when True.

    Returns
    -------
    rec_Line_AC
        New reconductoring line object.

    Raises
    ------
    ValueError
        If ``line_name`` is not found.
    """
    l = None
    for line_to_process in grid.lines_AC:
        if line_name == line_to_process.name:
            l = line_to_process
            break

    if l is not None:
        grid.lines_AC.remove(l)
        l.remove()
        line_vars = {
            'fromNode': l.fromNode,
            'toNode': l.toNode,
            'r': l.r,
            'x': l.x,
            'g': l.g,
            'b': l.b,
            'MVA_rating': l.MVA_rating,
            'Length_km': l.Length_km,
            'm': l.m,
            'shift': l.shift,
            'N_cables': l.N_cables,
            'name': l.name,
            'geometry': l.geometry,
            'S_base': l.S_base,
            'Cable_type': l.Cable_type
        }
        rec_line = rec_Line_AC(r_new,x_new,g_new,b_new,MVA_rating_new,Life_time,base_cost,**line_vars)
        grid.lines_AC_rec.append(rec_line)
        if update_grid:
            grid.update_graph_ac()

    # Reassign line numbers to ensure continuity
    for i, line in enumerate(grid.lines_AC):
        line.lineNumber = i

    for i, line in enumerate(grid.lines_AC_rec):
        line.lineNumber = i
    if update_grid:
        grid.create_Ybus_AC()

    if l is None:
        raise ValueError(f"Line '{line_name}' not found in grid.lines_AC")
    return rec_line

def change_line_AC_to_tap_transformer(grid, line_name):
    """Convert an AC line into a tap-changing transformer branch.

    Parameters
    ----------
    grid : Grid
        Grid containing the line.
    line_name : str
        Name of the line in ``grid.lines_AC``.

    Returns
    -------
    TF_Line_AC
        New transformer object.

    Raises
    ------
    ValueError
        If ``line_name`` is not found.
    """
    l = None
    for line_to_process in grid.lines_AC:
        if line_name == line_to_process.name:
            l  = line_to_process
            break
    if l is not None:
            grid.lines_AC.remove(l)
            l.remove()
            trafo = TF_Line_AC(
                l.fromNode,
                l.toNode,
                l.r,
                l.x,
                l.g,
                l.b,
                l.MVA_rating,
                l.fromNode.kV_base,
                l.m,
                l.shift,
                l.name,
            )
            trafo.geometry = l.geometry
            grid.lines_AC_tf.append(trafo)
    else:
        raise ValueError(f"Line '{line_name}' not found in grid.lines_AC")
    # Reassign line numbers to ensure continuity in grid.lines_AC
    for i, line in enumerate(grid.lines_AC):
        line.lineNumber = i
    grid.create_Ybus_AC()
    return trafo

def add_line_sizing(grid, fromNode, toNode,cable_types=None, active_config: int = 0,Length_km=1.0,S_base=100,name=None,cable_option=None,update_grid=True,geometry=None):
    """Append a conductor-size-selection (array/CT) branch.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    fromNode, toNode : Node_AC or str
        End buses.
    cable_types : list, optional
        Ordered cable type names (smallest to largest capacity).
    active_config : int, optional
        Index into ``cable_types`` for the initial active rating.
    Length_km : float, optional
        Branch length in km.
    S_base : float, optional
        Power base in MVA used for per-unit parameters.
    name : str, optional
        Branch name.
    cable_option : str, optional
        Name of a :class:`~pyflow_acdc.Classes.Cable_options` entry; links the
        line via :func:`assign_lineToCable_options`.
    update_grid : bool, optional
        Rebuild Y-bus and graph when True.
    geometry : shapely.Geometry or str, optional
        Line geometry for plotting.

    Returns
    -------
    Size_selection
        Created sizing line in ``grid.lines_AC_ct``.
    """
    if cable_types is None:
        cable_types = []
    fromNode = _look_up_node(grid, fromNode, ac_or_dc="AC")
    toNode = _look_up_node(grid, toNode, ac_or_dc="AC")

    line = Size_selection(fromNode, toNode,cable_types, active_config,Length_km,S_base,name)
    grid.lines_AC_ct.append(line)
    if cable_option is not None:
        assign_lineToCable_options(grid,line.name,cable_option)
    if update_grid:
        grid.create_Ybus_AC()
        grid.update_graph_ac()
    if geometry is not None:
       if isinstance(geometry, str):
            geometry = loads(geometry)
       line.geometry = geometry
    return line

def add_line_DC(grid, fromNode, toNode, r=0.001, MW_rating=9999,Length_km=1,R_Ohm_km=None,A_rating=None,polarity=Polarity.MONOPOLAR, name=None,geometry=None,Cable_type:str =CableType.CUSTOM,data_in=DataInput.PU,update_grid=True):
    """Append a DC cable to ``grid.lines_DC``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    fromNode, toNode : Node_DC or str
        End buses.
    r : float, optional
        Resistance in p.u. when ``data_in='pu'``.
    MW_rating : float, optional
        Power rating in MW.
    Length_km : float, optional
        Cable length in km.
    R_Ohm_km : float, optional
        Resistance per km for ``data_in='Real'``.
    A_rating : float, optional
        Current rating in A (Real input).
    polarity : str or int, optional
        ``'m'`` monopolar, ``'sm'`` symmetric monopolar, or ``'b'`` bipolar.
    name : str, optional
        Branch name.
    geometry : shapely.Geometry or str, optional
        Line geometry.
    Cable_type : str, optional
        Named cable from database, or ``'Custom'``.
    data_in : str, optional
        ``'pu'``, ``'Ohm'``, or ``'Real'``.
    update_grid : bool, optional
        Rebuild DC Y-bus and graph when True.

    Returns
    -------
    Line_DC
        Created DC line.

    Examples
    --------
    >>> line = pyf.add_line_DC(grid, n1, n2, r=0.0000318, MW_rating=1000,
    ...                        polarity='b', Length_km=10)
    """

    fromNode = _look_up_node(grid, fromNode, ac_or_dc="DC")
    toNode = _look_up_node(grid, toNode, ac_or_dc="DC")

    kV_base=toNode.kV_base
    if data_in == DataInput.OHM:
        Z_base = kV_base**2/grid.S_base

        Resistance_pu = r / Z_base if r!=0 else 0.00001

    elif data_in== DataInput.REAL or R_Ohm_km is not None:
        if A_rating is None:
            A_rating = MW_rating*1000/kV_base
        [Resistance_pu, _, _, _, MW_rating] = cable_parameters(grid.S_base, R_Ohm_km, 0, 0, 0, A_rating, kV_base, Length_km,N_cables=1)
    else:
        Resistance_pu = r if r!=0 else 0.00001

    if isinstance(polarity, int):
        if polarity == 1:
            polarity = Polarity.MONOPOLAR
        elif polarity == 2:
            polarity = Polarity.BIPOLAR
        else:
            raise ValueError(f"Invalid polarity value: {polarity}. Must be 1 ('m'), 2 ('b'), 'm', or 'b'.")
    line = Line_DC(fromNode, toNode, Resistance_pu, MW_rating,Length_km, polarity, name,Cable_type=Cable_type)
    grid.lines_DC.append(line)

    if geometry is not None:
       if isinstance(geometry, str):
            geometry = loads(geometry)
       line.geometry = geometry
    if update_grid:
        grid.create_Ybus_DC()
        grid.update_graph_dc()
    return line

def add_ACDC_converter(grid,AC_node , DC_node , AC_type='PV', DC_type=None, P_AC_MW=0, Q_AC_MVA=0, P_DC_MW=0, Transformer_resistance=0, Transformer_reactance=0, Phase_Reactor_R=0, Phase_Reactor_X=0, Filter=0, Droop=0, kV_base=None, MVA_max= None,nConvP=1,polarity =1 ,lossa=1.103,lossb= 0.887,losscrect=2.885,losscinv=4.371,Arm_R=None,Ucmin= 0.85, Ucmax= 1.2, name=None,geometry=None):
    """Append an AC/DC converter to ``grid.Converters_ACDC``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    AC_node : Node_AC or str
        AC-side bus.
    DC_node : Node_DC or str
        DC-side bus.
    AC_type : str, optional
        AC control mode: ``'PV'``, ``'PQ'``, or ``'Slack'``.
    DC_type : str, optional
        DC control mode: ``'P'``, ``'Slack'``, or ``'Droop'``; defaults to DC
        node type.
    P_AC_MW, Q_AC_MVA, P_DC_MW : float, optional
        Power setpoints in MW / MVAr.
    Transformer_resistance, Transformer_reactance : float, optional
        Converter transformer impedance in p.u.
    Phase_Reactor_R, Phase_Reactor_X, Filter : float, optional
        Phase reactor and filter parameters in p.u.
    Droop : float, optional
        DC droop constant when ``DC_type='Droop'``.
    kV_base : float, optional
        AC base voltage in kV; defaults to AC node ``kV_base``.
    MVA_max : float, optional
        Converter rating in MVA.
    nConvP : int, optional
        Number of parallel converter units.
    polarity, lossa, lossb, losscrect, losscinv : float, optional
        MMC loss model coefficients.
    Arm_R : float, optional
        Arm resistance in ohms (converted internally).
    Ucmin, Ucmax : float, optional
        Submodule capacitor voltage limits in p.u.
    name : str, optional
        Converter name.
    geometry : shapely.Geometry or str, optional
        Plot geometry.

    Returns
    -------
    AC_DC_converter
        Created converter.

    Notes
    -----
    Power-flow control pairings (AC side vs DC side):

    - ``AC_type='Slack'`` with ``DC_type='P'``: no extra setpoints
    - ``AC_type='PQ'`` with ``DC_type='P'``: requires ``Q_AC_MVA``
    - ``AC_type='PV'`` with ``DC_type='P'``: requires ``P_AC_MW``
    - ``DC_type='Droop'`` with ``AC_type='PQ'`` or ``'PV'``: requires
      ``P_DC_MW`` and ``Droop``

    Examples
    --------
    >>> conv = pyf.add_ACDC_converter(grid, ac_node, dc_node, MVA_max=1000)
    """
    DC_node = _look_up_node(grid, DC_node, ac_or_dc="DC")
    AC_node = _look_up_node(grid, AC_node, ac_or_dc="AC")




    if MVA_max is None:
        MVA_max= grid.S_base*100
    if kV_base is None:
        kV_base = AC_node.kV_base
    if DC_type is None:
        DC_type = DC_node.type

    P_DC = P_DC_MW/grid.S_base
    P_AC = P_AC_MW/grid.S_base
    Q_AC = Q_AC_MVA/grid.S_base
    # if Filter !=0 and Phase_Reactor_R==0 and  Phase_Reactor_X!=0:
    #     print(f'Please fill out phase reactor values, converter {name} not added')
    #     return
    ra =0.001

    conv = AC_DC_converter(AC_type, DC_type, AC_node, DC_node, P_AC, Q_AC, P_DC, Transformer_resistance, Transformer_reactance, Phase_Reactor_R, Phase_Reactor_X, Filter, Droop, kV_base, MVA_max,nConvP,polarity ,lossa,lossb,losscrect,losscinv,Ucmin, Ucmax, ra, grid.S_base, name)
    if geometry is not None:
        if isinstance(geometry, str):
             geometry = loads(geometry)
        conv.geometry = geometry

    conv.basekA  = grid.S_base/(SQRT_3*conv.AC_kV_base)
    conv.basekA_DC = grid.S_base/(conv.DC_kV_base)
    if Arm_R is not None:
        conv.ra  = Arm_R*conv.basekA_DC**2/grid.S_base
    else:
        conv.ra = 0.001
    conv.a_conv  = conv.a_conv_og/grid.S_base
    conv.b_conv  = conv.b_conv_og*conv.basekA/grid.S_base
    conv.c_inver = conv.c_inver_og*conv.basekA**2/grid.S_base
    conv.c_rect  = conv.c_rect_og*conv.basekA**2/grid.S_base




    grid.Converters_ACDC.append(conv)
    return conv

def add_DCDC_converter(grid,fromNode , toNode ,P_MW=None,Pset=None,R_Ohm=None, r=0.0001, MW_rating=MAX_RATING_PLACEHOLDER,name=None,geometry=None):
    """Append a DC/DC converter between two DC buses.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    fromNode, toNode : Node_DC or str
        DC buses.
    P_MW : float, optional
        Power setpoint in MW (converted to p.u.).
    Pset : float, optional
        Power setpoint in p.u.; defaults from ``MW_rating`` if omitted.
    R_Ohm : float, optional
        Series resistance in ohms (converted to p.u.).
    r : float, optional
        Series resistance in p.u. when ``R_Ohm`` is not given.
    MW_rating : float, optional
        Converter rating in MW.
    name : str, optional
        Converter name.
    geometry : shapely.Geometry or str, optional
        Plot geometry.

    Returns
    -------
    DCDC_converter
        Created converter in ``grid.Converters_DCDC``.
    """
    fromNode = _look_up_node(grid, fromNode, ac_or_dc="DC")
    toNode = _look_up_node(grid, toNode, ac_or_dc="DC")

    if R_Ohm is not None:
        Z_base = toNode.kV_base**2/grid.S_base
        r = R_Ohm/Z_base
    if P_MW is not None:
        Pset = P_MW/grid.S_base
    if Pset is None:
        Pset = MW_rating/(2*grid.S_base)

    conv = DCDC_converter(fromNode , toNode , Pset, r, MW_rating,name,geometry)
    grid.Converters_DCDC.append(conv)
    return conv

"Zones"

def add_cable_option(grid, cable_types: list = None, name=None, cable_database=None):
    """Register an ordered list of cable types for array / CSS optimisation.

    Links multiple :class:`~pyflow_acdc.Classes.Size_selection` lines to one
    shared rating ladder so the optimiser limits how many distinct types are used
    (see ``grid.cab_types_allowed``).

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    cable_types : list, optional
        Cable names from the bundled database, smallest to largest capacity.
    name : str, optional
        Option name; auto-generated if omitted.
    cable_database : pandas.DataFrame, optional
        Custom cable database; defaults to the bundled YAML database.

    Returns
    -------
    Cable_options
        Created option in ``grid.Cable_options``.

    Examples
    --------
    >>> opt = pyf.add_cable_option(grid, [
    ...     'ABB_Cu_XLPE_95mm2_66kV',
    ...     'ABB_Cu_XLPE_120mm2_66kV',
    ...     'ABB_Cu_XLPE_150mm2_66kV',
    ... ])
    >>> grid.cab_types_allowed = 3
    """
    cable_option = Cable_options(cable_types, name, cable_database=cable_database)
    grid.Cable_options.append(cable_option)
    return cable_option


def add_RenSource_zone(grid,name):
    """Create a renewable-source zone for shared availability time series.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    name : str
        Zone name.

    Returns
    -------
    Ren_source_zone
        Created zone in ``grid.RenSource_zones``.

    See Also
    --------
    assign_RenToZone : assign sources to this zone.

    Examples
    --------
    >>> zone = pyf.add_RenSource_zone(grid, 'WindZone1')
    """
    RSZ = Ren_source_zone(name)
    grid.RenSource_zones.append(RSZ)
    grid.RenSource_zones_dic[name]=RSZ.ren_source_num

    return RSZ


def add_price_zone(grid,name,price,import_pu_L=1,export_pu_G=1,a=0,b=1,c=0,import_expand_pu=0,curvature_factor=1,positive_price_delta=None):
    """Append a market / load price zone.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    name : str
        Zone name (unique key in ``grid.Price_Zones_dic``).
    price : float
        Base energy price; also used as default linear cost coefficient ``b``
        when ``b==1``.
    import_pu_L, export_pu_G : float, optional
        Import/export limits relative to base load in p.u.
    a, b, c : float, optional
        Quadratic cost coefficients for OPF (``a*P^2 + b*P + c``).
    import_expand_pu : float, optional
        Additional import capacity from linked zones in p.u.
    curvature_factor : float, optional
        Price-curvature parameter for multi-period models.
    positive_price_delta : float, optional
        Minimum price increment for piecewise pricing.

    Returns
    -------
    Price_Zone
        Created zone.

    Examples
    --------
    >>> zone = pyf.add_price_zone(grid, 'Zone1', price=50)
    """
    if b==1:
        b= price

    M = Price_Zone(price,import_pu_L,export_pu_G,a,b,c,import_expand_pu,curvature_factor,grid.S_base,name,positive_price_delta=positive_price_delta)
    grid.Price_Zones.append(M)
    grid.Price_Zones_dic[name]=M.price_zone_num

    return M

def add_MTDC_price_zone(grid, name,  linked_price_zones=None,pricing_strategy=PricingStrategy.AVG.value):
    """Create an MTDC aggregate price zone linked to onshore zones.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    name : str
        MTDC zone name.
    linked_price_zones : list, optional
        Names or objects of linked :class:`~pyflow_acdc.Classes.Price_Zone`
        instances.
    pricing_strategy : str, optional
        Aggregation rule (default average; see
        :class:`~pyflow_acdc.constants.PricingStrategy`).

    Returns
    -------
    MTDCPrice_Zone
        Created zone in ``grid.Price_Zones``.
    """
    if linked_price_zones:
        resolved = []
        for pz in linked_price_zones:
            if isinstance(pz, str):
                pz_obj = next((M for M in grid.Price_Zones if M.name == pz), None)
                if pz_obj is None:
                    raise ValueError(f"Price zone '{pz}' not found for MTDC link.")
                resolved.append(pz_obj)
            else:
                resolved.append(pz)
        linked_price_zones = resolved

    mtdc_price_zone = MTDCPrice_Zone(name=name, linked_price_zones=linked_price_zones, pricing_strategy=pricing_strategy)
    grid.Price_Zones.append(mtdc_price_zone)

    return mtdc_price_zone


def add_offshore_price_zone(grid,main_price_zone,name):
    """Create an offshore price zone linked to a main onshore zone.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    main_price_zone : Price_Zone or str
        Onshore zone to link (name lookup if string).
    name : str
        Offshore zone name (often ``'o_<main>'``).

    Returns
    -------
    OffshorePrice_Zone
        Created zone in ``grid.Price_Zones``.
    """
    if isinstance(main_price_zone, str):
        main_price_zone = next((M for M in grid.Price_Zones if main_price_zone == M.name), None)

    oprice_zone = OffshorePrice_Zone(name=name, price=main_price_zone.price, main_price_zone=main_price_zone)
    grid.Price_Zones.append(oprice_zone)

    return oprice_zone

"Components for optimal power flow"

def add_generators(grid,Gen_csv,curtailment_allowed=1):
    """Bulk-add generators or renewable sources from a CSV or DataFrame.

    Rows with ``Fueltype`` in ``grid.renewable_types`` call
    :func:`add_RenSource`; others call :func:`add_gen`. Expected columns include
    ``Node``/``node``, ``MWmax``, optional ``MVARmin``/``MVARmax``, cost factors,
    ``Fueltype``, ``geometry``, ``Ren_zone``, and ``np``/``np_gen``/``np_rsgen``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    Gen_csv : str or pandas.DataFrame
        Generator table path or in-memory data.
    curtailment_allowed : float, optional
        Passed as ``min_gamma=1-curtailment_allowed`` for renewable rows.

    Returns
    -------
    None
    """
    if isinstance(Gen_csv, pd.DataFrame):
        Gen_data = Gen_csv
    else:
        Gen_data = pd.read_csv(Gen_csv)
    if 'Gen' in Gen_data.columns:
        Gen_data = Gen_data.set_index('Gen')


    for index, row in Gen_data.iterrows():
        var_name = Gen_data.at[index, 'Gen_name'] if 'Gen_name' in Gen_data.columns else index
        if 'Node' in Gen_data.columns:
            node_name = str(Gen_data.at[index, 'Node'])
        elif 'node' in Gen_data.columns:
            node_name = str(Gen_data.at[index, 'node'])
        else:
            raise ValueError(f"No 'Node' or 'node' column found in Gen_data for index {index}")

        MWmax = Gen_data.at[index, 'MWmax'] if 'MWmax' in Gen_data.columns else None
        MWmin = Gen_data.at[index, 'MWmin'] if 'MWmin' in Gen_data.columns else 0
        MVArmin = Gen_data.at[index, 'MVARmin'] if 'MVARmin' in Gen_data.columns else None
        MVArmax = Gen_data.at[index, 'MVARmax'] if 'MVARmax' in Gen_data.columns else None

        PsetMW = Gen_data.at[index, 'PsetMW']  if 'PsetMW'  in Gen_data.columns else 0
        QsetMVA= Gen_data.at[index, 'QsetMVA'] if 'QsetMVA' in Gen_data.columns else 0
        lf = Gen_data.at[index, 'Linear factor']    if 'Linear factor' in Gen_data.columns else 0
        qf = Gen_data.at[index, 'Quadratic factor'] if 'Quadratic factor' in Gen_data.columns else 0
        fc = Gen_data.at[index, 'Fixed cost'] if 'Fixed cost' in Gen_data.columns else 0
        geo  = Gen_data.at[index, 'geometry'] if 'geometry' in Gen_data.columns else None
        Ren_zone = Gen_data.at[index, 'Ren_zone'] if 'Ren_zone' in Gen_data.columns else None
        price_zone_link = False

        fuel_type = Gen_data.at[index, 'Fueltype']    if 'Fueltype' in Gen_data.columns else 'Other'
        np_value = Gen_data.at[index, 'np'] if 'np' in Gen_data.columns else 1
        if 'np_gen' in Gen_data.columns:
            np_value = Gen_data.at[index, 'np_gen']
        if 'np_rsgen' in Gen_data.columns:
            np_value = Gen_data.at[index, 'np_rsgen']
        if fuel_type.lower() in grid.renewable_types:
            add_RenSource(grid,node_name, MWmax,ren_source_name=var_name ,geometry=geo,ren_type=fuel_type,Qmin=MVArmin,Qmax=MVArmax,min_gamma=(1-curtailment_allowed),zone=Ren_zone,np_rsgen=np_value)
        else:
            if MVArmax is None:
                MVArmax = 9999
            if MVArmin is None:
                MVArmin = -9999
            add_gen(grid, node_name,var_name, price_zone_link,lf,qf,fc,MWmax,MWmin,MVArmin,MVArmax,PsetMW,QsetMVA,fuel_type=fuel_type,geometry=geo,np_gen=np_value)

def _look_up_node(grid, node, ac_or_dc="AC"):

    # Object case: ensure it's registered on the appropriate grid list
    if not isinstance(node, str):
        # Determine target list based on ac_or_dc and actual node type
        if ac_or_dc == "AC":
            nodes = grid.nodes_AC
        elif ac_or_dc == "DC":
            nodes = grid.nodes_DC
        elif ac_or_dc == "any":
            # Prefer existing membership; else infer from class.
            if node in grid.nodes_AC or node in grid.nodes_DC:
                return node
            if isinstance(node, Node_AC):
                nodes = grid.nodes_AC
            elif isinstance(node, Node_DC):
                nodes = grid.nodes_DC
            else:
                raise ValueError(f"Unsupported node type {node!r}")
        else:
            raise ValueError(f"Unsupported node type {ac_or_dc!r}")

        if node not in nodes:
            nodes.append(node)
        return node

    # String case: look up by name
    node_name = node
    if ac_or_dc == "AC":
        nodes = grid.nodes_AC
        found = next((n for n in nodes if n.name == node_name), None)
    elif ac_or_dc == "DC":
        nodes = grid.nodes_DC
        found = next((n for n in nodes if n.name == node_name), None)
    elif ac_or_dc == "any":
        found = next((n for n in grid.nodes_AC if n.name == node_name), None)
        if found is None:
            found = next((n for n in grid.nodes_DC if n.name == node_name), None)
    else:
        raise ValueError(f"Unsupported node type {ac_or_dc!r}")

    if found is None:
        raise ValueError(f"Node {node_name} does not exist in {ac_or_dc} grid")
    return found

def _look_up_price_zone(grid, price_zone):

    price_zones = getattr(grid, "Price_Zones", [])
    if not isinstance(price_zone, str):
        # Already an object: ensure it belongs to this grid
        if price_zone in price_zones:
            return price_zone
        else:
            raise ValueError(f"Price_Zone {price_zone} not found in grid")

    name = price_zone
    pz = next((pz for pz in price_zones if pz.name == name), None)
    if pz is None:
        raise ValueError(f"Price_Zone {name} not found.")
    return pz


def _look_up_converter(grid, conv):

    converters = getattr(grid, "Converters_ACDC", [])
    if not isinstance(conv, str):
        # Already an object: ensure it belongs to this grid
        if conv in converters:
            return conv
        else:
            raise ValueError(f"Converter {conv} not found in grid")

    conv_name = conv
    conv_obj = next((c for c in converters if c.name == conv_name), None)
    if conv_obj is None:
        raise ValueError(f"Converter {conv_name} not found.")
    return conv_obj

def add_gen(grid, node,gen_name=None, price_zone_link=False,lf=0,qf=0,fc=0,MWmax=MAX_RATING_PLACEHOLDER,MWmin=0,MVArmin=None,MVArmax=None,PsetMW=0,QsetMVA=0,Smax=None,fuel_type=DEFAULT_GEN_TYPE,geometry= None,installation_cost:float=0,np_gen:int=1):
    """Append an AC generator to ``grid.Generators``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    node : Node_AC or str
        Connection bus (name or object).
    gen_name : str, optional
        Generator name; defaults to ``'gen_<node>'``.
    price_zone_link : bool, optional
        If True, use node price zone marginal cost (``lf=node.price``).
    lf, qf, fc : float, optional
        Linear, quadratic, and fixed OPF cost coefficients.
    MWmax, MWmin : float, optional
        Active power limits in MW.
    MVArmin, MVArmax : float, optional
        Reactive limits in MVAr; default to ``±MWmax``.
    PsetMW, QsetMVA : float, optional
        Initial setpoints in MW / MVAr.
    Smax : float, optional
        Apparent power limit in MVA.
    fuel_type : str, optional
        Generation type label (must match ``grid.gen_ac_types``).
    geometry : shapely.Geometry or str, optional
        Plot geometry.
    installation_cost : float, optional
        Capital cost for TEP/CSS workflows.
    np_gen : int, optional
        Number of parallel units.

    Returns
    -------
    Gen_AC
        Created generator.

    Examples
    --------
    >>> gen = pyf.add_gen(grid, 'bus1', MWmax=500, fuel_type='Natural Gas')
    """

    if MVArmax is None:
        MVArmax=MWmax
    if MVArmin is None:
        MVArmin=-MVArmax
    if Smax is not None:
        Smax/=grid.S_base
    Max_pow_gen=MWmax/grid.S_base

    Max_pow_genR=MVArmax/grid.S_base
    Min_pow_genR=MVArmin/grid.S_base
    Min_pow_gen=MWmin/grid.S_base
    Pset=PsetMW/grid.S_base
    Qset=QsetMVA/grid.S_base

    node = _look_up_node(grid, node, ac_or_dc="AC")
    if gen_name is None:
        gen_name = f'gen_{node.name}'
    gen = Gen_AC(
        gen_name, node, Max_pow_gen, Min_pow_gen, Max_pow_genR, Min_pow_genR,
        qf, lf, fc, Pset, Qset, Smax,
        gen_type=fuel_type, installation_cost=installation_cost, S_base=grid.S_base, np_gen=np_gen
    )
    node.PGi = 0
    node.QGi = 0
    available_types = getattr(grid, 'gen_ac_types', ['other'])
    gen_type_lookup = {str(t).lower(): str(t) for t in available_types}
    normalized_fuel = str(fuel_type).lower()
    if normalized_fuel == 'gas':
        normalized_fuel = 'natural gas'
    gen.gen_type = gen_type_lookup.get(normalized_fuel, gen_type_lookup.get('other', 'other'))
    if geometry is not None:
        if isinstance(geometry, str):
            geometry = loads(geometry)
        gen.geometry= geometry
    gen.price_zone_link=price_zone_link

    if price_zone_link:

        gen.qf= 0
        gen.lf= node.price
    grid.Generators.append(gen)

    return gen

def add_gen_DC(grid, node,gen_name=None, price_zone_link=False,lf=0,qf=0,fc=0,MWmax=MAX_RATING_PLACEHOLDER,MWmin=0,PsetMW=0,fuel_type=DEFAULT_GEN_TYPE,geometry= None,installation_cost:float=0,np_gen:int=1):
    """Append a DC generator to ``grid.Generators_DC``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    node : Node_DC or str
        Connection bus.
    gen_name : str, optional
        Generator name.
    price_zone_link : bool, optional
        Link marginal cost to node price zone.
    lf, qf, fc : float, optional
        OPF cost coefficients.
    MWmax, MWmin : float, optional
        Active power limits in MW.
    PsetMW : float, optional
        Initial setpoint in MW.
    fuel_type : str, optional
        Generation type label.
    geometry : shapely.Geometry or str, optional
        Plot geometry.
    installation_cost : float, optional
        Capital cost.
    np_gen : int, optional
        Number of parallel units.

    Returns
    -------
    Gen_DC
        Created generator.
    """

    Max_pow_gen=MWmax/grid.S_base
    Min_pow_gen=MWmin/grid.S_base
    Pset=PsetMW/grid.S_base

    node = _look_up_node(grid, node, ac_or_dc="DC")
    if gen_name is None:
        gen_name = f'gen_DC_{node.name}'
    gen = Gen_DC(
        gen_name, node, Max_pow_gen, Min_pow_gen, qf, lf, fc, Pset,
        gen_type=fuel_type, installation_cost=installation_cost, S_base=grid.S_base, np_gen=np_gen
    )
    node.PGi = 0
    available_types = getattr(grid, 'gen_dc_types', ['other'])
    gen_type_lookup = {str(t).lower(): str(t) for t in available_types}
    normalized_fuel = str(fuel_type).lower()
    if normalized_fuel == 'gas':
        normalized_fuel = 'natural gas'
    gen.gen_type = gen_type_lookup.get(normalized_fuel, gen_type_lookup.get('other', 'other'))
    if geometry is not None:
        if isinstance(geometry, str):
            geometry = loads(geometry)
        gen.geometry= geometry
    gen.price_zone_link=price_zone_link

    if price_zone_link:

        gen.qf= 0
        gen.lf= node.price
    grid.Generators_DC.append(gen)

    return gen


def add_extgrid(grid, node, gen_name=None,price_zone_link=False,lf=0,qf=0,MVAmax=MAX_RATING_PLACEHOLDER,MWmax=None,MWmin=None,MVArmin=None,MVArmax=None,Allow_sell=True,P_load_MW=0):
    """Add an external-grid equivalent generator at an AC bus.

    Sets ``is_ext_grid=True``. If no slack bus exists, the connected node becomes
    ``'Slack'``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    node : Node_AC or str
        Connection bus.
    gen_name : str, optional
        Generator name; defaults to ``'extgrid_<node>'``.
    price_zone_link : bool, optional
        Link marginal cost to price zone.
    lf, qf : float, optional
        OPF cost coefficients.
    MVAmax : float, optional
        Apparent power rating in MVA (also sets ``MWmax`` when omitted).
    MWmax, MVArmin, MVArmax : float, optional
        Power limits in MW / MVAr.
    Allow_sell : bool, optional
        Allow negative export ( selling ) through the external grid.
    P_load_MW : float, optional
        Fixed load component modelled at the external grid in MW.

    Returns
    -------
    Gen_AC
        External-grid generator in ``grid.Generators``.
    """
    node = _look_up_node(grid, node, ac_or_dc="AC")
    if MWmax is None:
        MWmax=MVAmax
    if MWmin is None:
        MWmin=0
    if MVArmin is None:
        MVArmin=-MVAmax
    if MVArmax is None:
        MVArmax=MVAmax
    if gen_name is None:
        gen_name = f'extgrid_{node.name}'
    Max_pow_gen=MWmax/grid.S_base

    Max_pow_genR=MVArmax/grid.S_base
    Min_pow_genR=MVArmin/grid.S_base
    Min_pow_gen=MWmin/grid.S_base
    rating = MVAmax/ grid.S_base
    gen = Gen_AC(gen_name, node,Max_pow_gen,Min_pow_gen,Max_pow_genR,Min_pow_genR,qf,lf,S_rated=rating)
    gen.is_ext_grid = True
    gen.allow_sell = Allow_sell
    gen.p_load_base = P_load_MW / grid.S_base

    node.PGi = 0
    node.QGi = 0
    node.recalc_extgrid_load()
    if price_zone_link:
        # Keep aggregated price-zone load consistent after extgrid load is introduced.
        pz_name = getattr(node, "PZ", None)
        if pz_name:
            pz = _look_up_price_zone(grid, pz_name)
            if hasattr(pz, "recalc_PLi_base_and_total"):
                pz.recalc_PLi_base_and_total()

    gen.price_zone_link=price_zone_link
    if price_zone_link:
        gen.qf= 0
        gen.lf= node.price

    # Iterate over all AC nodes to see if any is already 'Slack'
    has_slack = any(n.type == NodeType.SLACK for n in grid.nodes_AC)
    if not has_slack:
        node.type = 'Slack'
    grid.Generators.append(gen)
    return gen

def add_RenSource(grid, node, base_MW, ren_source_name=None, available=1, zone=None, price_zone=None, Offshore=False, MTDC=None, geometry=None, ren_type='Wind', min_gamma=0, Qrel=0,Qmin=None,Qmax=None,np_rsgen: int = 1):
    """Append a renewable source to ``grid.RenSources``.

    Optionally assigns renewable zone, price zone, offshore, or MTDC pricing in
    the same call (delegates to :func:`assign_RenToZone` and
    :func:`assign_nodeToPrice_Zone`).

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    node : Node_AC, Node_DC, or str
        Connection bus.
    base_MW : float
        Nameplate active power in MW.
    ren_source_name : str, optional
        Source name; defaults to ``'rsgen_<node>'``.
    available : float, optional
        Availability factor in ``[0, 1]`` (``PRGi_available``).
    zone : str, optional
        Renewable zone name; triggers zone assignment.
    price_zone : str, optional
        Price zone name for market coupling.
    Offshore : bool, optional
        Create/link offshore price zone when True.
    MTDC : str, optional
        MTDC price zone name for offshore HVDC pricing.
    geometry : shapely.Geometry or str, optional
        Plot geometry.
    ren_type : str, optional
        Resource type (e.g. ``'Wind'``, ``'Solar'``).
    min_gamma : float, optional
        Minimum dispatch fraction (curtailment floor).
    Qrel : float, optional
        Reactive capability as fraction of ``base_MW`` when ``Qmin``/``Qmax``
        omitted (AC only).
    Qmin, Qmax : float, optional
        Reactive limits in MVAr (AC only).
    np_rsgen : int, optional
        Number of parallel units.

    Returns
    -------
    Ren_Source
        Created source.

    Examples
    --------
    >>> rs = pyf.add_RenSource(grid, 'bus1', 100, ren_type='Wind', np_rsgen=2)
    """

    node = _look_up_node(grid, node, ac_or_dc="any")

    # Set default ren_source_name if not provided
    if ren_source_name is None:
        ren_source_name = f'rsgen_{node.name}'

    # Create renewable source
    rensource = Ren_Source(ren_source_name, node, base_MW/grid.S_base,S_base=grid.S_base,np_rsgen=np_rsgen)
    rensource.PRGi_available = available
    rensource.rs_type = ren_type
    rensource.min_gamma = min_gamma


    # Determine connection type and set appropriate attributes
    if node in grid.nodes_AC:
        rensource.connected = AcDcSide.AC
        ACDC = AcDcSide.AC.value
        if Qmax is not None:
            rensource.Qmax = Qmax/grid.S_base
        else:
            rensource.Qmax = base_MW*Qrel/grid.S_base
        if Qmin is not None:
            rensource.Qmin = Qmin/grid.S_base
        else:
            rensource.Qmin = -base_MW*Qrel/grid.S_base
        grid.rs2node['AC'][rensource.rsNumber] = node.nodeNumber
    elif node in grid.nodes_DC:
        rensource.connected = AcDcSide.DC
        ACDC = AcDcSide.DC.value
        grid.rs2node['DC'][rensource.rsNumber] = node.nodeNumber
    else:
        raise ValueError(f'Node {node.name} is not in AC or DC nodes')

    # Handle geometry
    if geometry is not None:
        if isinstance(geometry, str):
            geometry = loads(geometry)
        rensource.geometry = geometry

    # Add to grid
    grid.RenSources.append(rensource)

    # Handle zone assignment
    if zone is not None:
        rensource.zone = zone
        assign_RenToZone(grid, ren_source_name, zone)

    # Handle price zone assignment
    if price_zone is not None:
        rensource.price_zone = price_zone
        if MTDC is not None:
            rensource.MTDC = MTDC
            main_price_zone = next((M for M in grid.Price_Zones if price_zone == M.name), None)
            if main_price_zone is not None:
                # Find or create the MTDC price_zone
                MTDC_price_zone = next((mdc for mdc in grid.Price_Zones if MTDC == mdc.name), None)

                if MTDC_price_zone is None:
                    # Create the MTDC price_zone using the MTDCPrice_Zone class
                    MTDC_price_zone = add_MTDC_price_zone(grid, MTDC)

            MTDC_price_zone.add_linked_price_zone(main_price_zone)
            main_price_zone.import_expand += base_MW / grid.S_base
            assign_nodeToPrice_Zone(grid, node.name,MTDC, ACDC)
            # Additional logic for MTDC can be placed here
        elif Offshore:
            rensource.Offshore = True
            # Create an offshore price_zone by appending 'o' to the main price_zone's name
            oprice_zone_name = f'o_{price_zone}'

            # Find the main price_zone
            main_price_zone = next((M for M in grid.Price_Zones if price_zone == M.name), None)

            if main_price_zone is not None:
                # Find or create the offshore price_zone
                oprice_zone = next((m for m in grid.Price_Zones if m.name == oprice_zone_name), None)

                if oprice_zone is None:
                    # Create the offshore price_zone using the OffshorePrice_Zone class
                    oprice_zone = add_offshore_price_zone(grid, main_price_zone, oprice_zone_name)

                # Assign the node to the offshore price_zone
                assign_nodeToPrice_Zone(grid, node.name, oprice_zone_name, ACDC)
                # Link the offshore price_zone to the main price_zone
                main_price_zone.link_price_zone(oprice_zone)
                # Expand the import capacity in the main price_zone
                main_price_zone.import_expand += base_MW / grid.S_base
        else:
            # Assign the node to the main price_zone
            assign_nodeToPrice_Zone(grid, node.name, price_zone, ACDC)

    return rensource

"Time series data "


def time_series_dict(grid, ts):
    typ = ts.type

    if typ == TSType.A_CG:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == TSType.B_CG:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == TSType.C_CG:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == TSType.PGL_MIN:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break
    elif typ == TSType.PGL_MAX:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break

    if typ == TSType.PRICE:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct price_zone
        for node in grid.nodes_AC + grid.nodes_DC:
            if ts.element_name == node.name:
                node.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct node

    elif typ == TSType.LOAD:
        for price_zone in grid.Price_Zones:
            if ts.element_name == price_zone.name:
                price_zone.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct price_zone
        for node in grid.nodes_AC + grid.nodes_DC:
            if ts.element_name == node.name:
                node.TS_dict[typ] = ts.TS_num
                break  # Stop after assigning to the correct node

    elif typ in TS_RENEWABLE_TYPES:
        for zone in grid.RenSource_zones:
            if ts.element_name == zone.name:
                zone.TS_dict['PRGi_available'] = ts.TS_num
                break  # Stop after assigning to the correct zone
        for rs in grid.RenSources:
            if ts.element_name == rs.name:
                rs.TS_dict['PRGi_available'] = ts.TS_num
                break  # Stop after assigning to the correct node


def add_inv_series(grid,inv_data,associated=None,inv_type=None,name=None):
    """Attach investment-period time series to grid elements from a CSV file.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    inv_data : str or pathlib.Path
        CSV path (``header=None``). Each column is one series; layout depends on
        ``associated`` / ``inv_type`` (see Notes).
    associated : str or Grid element, optional
        Target element name or object when type/name come from the CSV or args.
    inv_type : str, optional
        Investment key (e.g. ``'Load'``, ``'planned_installation'``).
    name : str, optional
        Override series label for the column.

    Returns
    -------
    None

    Notes
    -----
    Supported ``inv_type`` keys by element:

    - ``Price_Zone``: ``Load``, ``curvature_factor``, ``import_expand``
    - ``Node_AC`` / ``Node_DC``: ``Load`` (independent nodes only)
    - ``Gen_AC``, ``Ren_Source``, ``Exp_Line_AC``, ``Line_DC``, ``AC_DC_converter``:
      ``planned_installation``, ``planned_decommission``, ``max_inv``, ``np_dynamic``

    CSV layouts:

    - Both ``associated`` and ``inv_type`` given: column = period values only.
    - ``associated`` only: row 0 = ``inv_type``, rows 1+ = values.
    - ``inv_type`` only: row 0 = element name, rows 1+ = values.
    - Neither: row 0 = element, row 1 = ``inv_type``, row 2+ = values.

    Special case ``All``/``Load`` applies one load series to all eligible zones
    and independent nodes. All multi-period series in one import must share the
    same period count (scalar length-1 series are always allowed).
    """
    if not isinstance(inv_data, (str, Path)):
        raise TypeError("inv_data must be a CSV file path (str or Path)")

    inv = pd.read_csv(inv_data, header=None)

    if inv.empty:
        raise ValueError("inv_data is empty")

    known_types = {
        'Load', 'curvature_factor', 'import_expand',
        'planned_installation', 'planned_decommission',
        'max_inv', 'np_dynamic'
    }
    # Expected period length is inferred from the first imported series
    # with more than one value. Scalar series (len=1) are always allowed.
    expected_len = None

    def _series_name(col):
        return str(col) if name is None else name

    def _to_numeric(values, inv_name):
        values = pd.Series(values).dropna()
        data = pd.to_numeric(values, errors='coerce')
        if data.isna().any():
            raise ValueError(
                f"Investment series '{inv_name}' contains non-numeric period values"
            )
        return data.to_numpy(dtype=float)

    def _associated_name(value):
        return value.name if hasattr(value, 'name') else value

    def _find_investment_element(grid_obj, elem_name):
        name = str(elem_name)
        candidates = (
            list(grid_obj.Price_Zones)
            + list(grid_obj.nodes_AC)
            + list(grid_obj.nodes_DC)
            + list(grid_obj.Generators)
            + list(grid_obj.RenSources)
            + list(grid_obj.lines_AC_exp)
            + list(grid_obj.lines_DC)
            + list(grid_obj.Converters_ACDC)
        )
        return next((el for el in candidates if str(getattr(el, 'name', '')) == name), None)

    def _is_all_load_case(elem_name, elem_type):
        return (
            str(elem_name).strip().lower() == 'all'
            and str(elem_type).strip().lower() == 'load'
        )

    def _load_targets_for_all(grid_obj):
        targets = []
        # Apply to all price zones with load investment support.
        for price_zone in list(getattr(grid_obj, 'Price_Zones', [])):
            if hasattr(price_zone, 'investment_decisions') and 'Load' in price_zone.investment_decisions:
                targets.append(price_zone)

        # Apply to independent AC/DC nodes with load investment support.
        for node in list(getattr(grid_obj, 'nodes_AC', [])) + list(getattr(grid_obj, 'nodes_DC', [])):
            if not hasattr(node, 'investment_decisions') or 'Load' not in node.investment_decisions:
                continue
            if isinstance(node, (Node_AC, Node_DC)) and getattr(node, 'PLi_linked', False):
                continue
            targets.append(node)
        return targets

    def _is_ignore_token(value):
        return str(value).strip().lower() == 'ignore'

    for col in inv.columns:
        col_values = inv[col].reset_index(drop=True)
        inv_name = _series_name(col)
        if _is_ignore_token(inv_name):
            continue
        if len(col_values) > 0 and _is_ignore_token(col_values.iloc[0]):
            continue

        # Case 1: element and type explicitly provided in function call.
        # CSV column contains only period data.
        if associated is not None and inv_type is not None:
            element_name = _associated_name(associated)
            element_type = inv_type
            data = _to_numeric(col_values, inv_name)

        # Case 2: element is provided; type is taken from row 0.
        elif associated is not None:
            if len(col_values) < 2:
                raise ValueError(
                    f"Investment series '{inv_name}' needs at least 2 rows when "
                    "'associated' is provided and 'inv_type' is not."
                )
            element_name = _associated_name(associated)
            element_type = col_values.iloc[0]
            data = _to_numeric(col_values.iloc[1:], inv_name)

        # Case 3: inv_type is provided; element is taken from row 0.
        elif inv_type is not None:
            if len(col_values) < 2:
                raise ValueError(
                    f"Investment series '{inv_name}' needs at least 2 rows when "
                    "'inv_type' is provided and 'associated' is not."
                )
            element_name = col_values.iloc[0]
            element_type = inv_type
            data = _to_numeric(col_values.iloc[1:], inv_name)

        # Case 4: both element and inv_type are read from CSV rows 0 and 1.
        else:
            if len(col_values) < 3:
                raise ValueError(
                    f"Investment series '{inv_name}' needs at least 3 rows when "
                    "'associated' and 'inv_type' are not provided"
                )
            element_name = col_values.iloc[0]
            element_type = col_values.iloc[1]
            data = _to_numeric(col_values.iloc[2:], inv_name)

        if data.size == 0:
            raise ValueError(f"Investment series '{inv_name}' has no period values")

        # Keep period length consistent inside this imported file.
        # Use the first non-scalar series as reference; scalar series are always valid.
        data_len = len(data)
        if data_len > 1 and expected_len is None:
            expected_len = data_len
        elif expected_len is not None and data_len not in (1, expected_len):
            raise ValueError(
                f"Investment series '{inv_name}' has {data_len} periods, expected {expected_len} (or 1)."
            )

        element_type = str(element_type)
        if str(element_type) not in known_types:
            print(
                f"Warning: inv_type '{element_type}' is not in documented supported types."
            )

        if _is_all_load_case(element_name, element_type):
            load_targets = _load_targets_for_all(grid)
            if not load_targets:
                raise ValueError(
                    "Investment series 'All/Load' found no matching Price_Zones or independent nodes."
                )
            load_data = np.array(data, dtype=float).tolist()
            for target in load_targets:
                target.investment_decisions['Load'] = load_data.copy()
            continue

        element = _find_investment_element(grid, element_name)
        if element is None:
            raise ValueError(
                f"Investment series '{inv_name}' references unknown element '{element_name}'"
            )
        if not hasattr(element, 'investment_decisions'):
            raise ValueError(
                f"Element '{element_name}' has no investment_decisions dictionary"
            )
        if str(element_type) not in element.investment_decisions:
            valid_keys = ", ".join(element.investment_decisions.keys())
            raise ValueError(
                f"Element '{element_name}' does not support inv_type '{element_type}'. "
                f"Valid keys: {valid_keys}"
            )

        element.investment_decisions[str(element_type)] = np.array(data, dtype=float).tolist()

def add_gen_mix_limits(grid, mix_data):
    """Load per-period generation-type mix limits from CSV onto ``grid``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    mix_data : str or pathlib.Path
        CSV path (``header=None``). Each column: row 0 = generation type,
        rows 1+ = limit per investment period.

    Returns
    -------
    dict
        Mapping ``gen_type -> list`` of period limits stored on
        ``grid.generation_type_limits``.
    """
    if not isinstance(mix_data, (str, Path)):
        raise TypeError("mix_data must be a CSV file path (str or Path)")

    mix_df = pd.read_csv(mix_data, header=None)
    if mix_df.empty:
        raise ValueError("mix_data is empty")

    expected_len = None
    gen_mix_series = {}

    for col in mix_df.columns:
        col_values = mix_df[col].reset_index(drop=True)
        if len(col_values) < 2:
            raise ValueError(
                f"Generation mix column '{col}' needs at least 2 rows "
                "(gen_type + one data value)."
            )

        gen_type = str(col_values.iloc[0]).strip().lower()
        if not gen_type:
            raise ValueError(f"Generation mix column '{col}' has empty gen_type")

        values = pd.Series(col_values.iloc[1:]).dropna()
        data = pd.to_numeric(values, errors='coerce')
        if data.isna().any():
            raise ValueError(
                f"Generation mix column '{col}' ({gen_type}) contains non-numeric values"
            )
        if len(data) == 0:
            raise ValueError(
                f"Generation mix column '{col}' ({gen_type}) has no period values"
            )

        if expected_len is None:
            expected_len = len(data)
        elif len(data) != expected_len:
            raise ValueError(
                f"Generation mix type '{gen_type}' has {len(data)} periods, expected {expected_len}"
            )

        gen_mix_series[gen_type] = data.to_numpy(dtype=float).tolist()
        if gen_type not in grid.generation_types:
            grid.generation_types.append(gen_type)
        # Full per-period limits for MP model logic.
        grid.generation_type_limits[gen_type] = data.to_numpy(dtype=float).tolist()
        grid.current_generation_type_limits[gen_type] = float(data.iloc[0])

    return gen_mix_series


def create_gen_limit_csv_template(grid, file_path=None):
    """Write a generation-mix limit CSV template for active types on ``grid``.

    Parameters
    ----------
    grid : Grid
        Grid whose active AC generators and renewable sources define columns.
    file_path : str, optional
        Output path; defaults to ``'{grid.name}_gen_mix_limits.csv'``.

    Returns
    -------
    str
        Path to the written CSV (``header=None`` layout for
        :func:`add_gen_mix_limits`).

    Raises
    ------
    ValueError
        If no active generation types are found.
    """
    if file_path is None:
        file_path = f'{grid.name}_gen_mix_limits.csv'
    path = Path(file_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

    def _norm(gen_type):
        return str(gen_type).strip().lower() if gen_type is not None else None

    active_types = []
    seen_types = set()

    for gen in getattr(grid, 'Generators', []):
        gen_type = _norm(getattr(gen, 'gen_type', None))
        if not gen_type:
            continue
        units = float(getattr(gen, 'np_gen', 1.0))
        opf_active = bool(getattr(gen, 'np_gen_opf', False))
        if units <= 0 and not opf_active:
            continue
        if gen_type not in seen_types:
            seen_types.add(gen_type)
            active_types.append(gen_type)

    for ren_source in getattr(grid, 'RenSources', []):
        gen_type = _norm(getattr(ren_source, 'rs_type', None))
        if not gen_type:
            continue
        units = float(getattr(ren_source, 'np_rsgen', 1.0))
        opf_active = bool(getattr(ren_source, 'np_rsgen_opf', False))
        if units <= 0 and not opf_active:
            continue
        if gen_type not in seen_types:
            seen_types.add(gen_type)
            active_types.append(gen_type)

    if not active_types:
        raise ValueError("No active generation types were found in the provided grid.")

    # Read limits from the canonical container (can be scalar or per-period list).
    limit_series = {}
    scalar_limits_raw = getattr(grid, 'generation_type_limits', {})
    scalar_limits = {}
    if isinstance(scalar_limits_raw, dict):
        for k, v in scalar_limits_raw.items():
            kn = _norm(k)
            if not kn:
                continue
            if isinstance(v, (list, tuple, np.ndarray)):
                limit_series[kn] = list(v)
            else:
                scalar_limits[kn] = v

    max_periods = max((len(limit_series.get(gen_type, [])) for gen_type in active_types), default=0)

    if max_periods == 0:
        for element in list(getattr(grid, 'Generators', [])) + list(getattr(grid, 'RenSources', [])):
            inv_decisions = getattr(element, 'investment_decisions', {})
            if isinstance(inv_decisions, dict):
                for values in inv_decisions.values():
                    if values is not None:
                        max_periods = max(max_periods, len(values))
    if max_periods == 0:
        max_periods = 1

    columns = {}
    for col_idx, gen_type in enumerate(active_types):
        values = list(limit_series.get(gen_type, []))
        if not values:
            scalar_limit = scalar_limits.get(gen_type, 1.0)
            values = [float(scalar_limit)] * max_periods
        elif len(values) < max_periods:
            values = values + [values[-1]] * (max_periods - len(values))
        elif len(values) > max_periods:
            values = values[:max_periods]

        columns[col_idx] = [gen_type] + values

    template_df = pd.DataFrame(columns)
    template_df.to_csv(path, index=False, header=False)
    return str(path)

def create_inv_csv_template(grid, file_path=None, exclude=None):
    """Write an investment-series CSV template from current ``grid`` decisions.

    Parameters
    ----------
    grid : Grid
        Grid whose ``investment_decisions`` define template columns.
    file_path : str, optional
        Output path; defaults to ``'{grid.name}_inv_series.csv'``.
    exclude : list, optional
        ``inv_type`` keys to omit from the template.

    Returns
    -------
    str
        Path to the written CSV (``header=None`` layout for
        :func:`add_inv_series`).

    Raises
    ------
    ValueError
        If no elements with investment decisions are found.
    """
    if file_path is None:
        file_path = f'{grid.name}_inv_series.csv'
    path = Path(file_path)
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    exclude_keys = set(str(k) for k in (exclude or []))

    element_groups = [
        getattr(grid, 'Price_Zones', []),
        getattr(grid, 'nodes_AC', []),
        getattr(grid, 'nodes_DC', []),
        getattr(grid, 'Generators', []),
        getattr(grid, 'RenSources', []),
        getattr(grid, 'lines_AC_exp', []),
        getattr(grid, 'lines_DC', []),
        getattr(grid, 'Converters_ACDC', []),
    ]

    columns = {}
    col_idx = 0
    max_periods = 0
    for group in element_groups:
        for element in group:
            if not hasattr(element, 'investment_decisions'):
                continue
            # For nodes, only export independent load series (not price-zone-linked).
            if isinstance(element, (Node_AC, Node_DC)) and getattr(element, 'PLi_linked', True):
                continue
            # If element has an MP/TEP activation flag, only include active entries.
            if hasattr(element, 'np_line_opf') and not element.np_line_opf:
                continue
            if hasattr(element, 'np_gen_opf') and not element.np_gen_opf:
                continue
            if hasattr(element, 'np_rsgen_opf') and not element.np_rsgen_opf:
                continue
            if hasattr(element, 'np_conv_opf') and not element.np_conv_opf:
                continue
            element_name = getattr(element, 'name', None)
            if element_name is None:
                continue
            for inv_key, inv_values in element.investment_decisions.items():
                if str(inv_key) in exclude_keys:
                    continue
                values = list(inv_values) if inv_values is not None else []
                max_periods = max(max_periods, len(values))
                columns[col_idx] = [element_name, inv_key] + values
                col_idx += 1

    if not columns:
        raise ValueError(
            "No elements with investment_decisions were found in the provided grid."
        )

    # Pad columns to a common number of data rows so missing values remain visible.
    target_len = 2 + max_periods
    for col, values in columns.items():
        if len(values) < target_len:
            columns[col] = values + [""] * (target_len - len(values))

    template_df = pd.DataFrame(columns)
    template_df.to_csv(path, index=False, header=False)
    return str(path)

def add_TimeSeries(grid, Time_Series_data,associated=None,TS_type=None,name=None):
    """Import operational time-series columns onto ``grid.Time_series``.

    Each column becomes a :class:`~pyflow_acdc.Classes.TimeSeries` entry and is
    wired to the matching grid element via :func:`time_series_dict`. NaN values
    are replaced with zero. Sets ``grid.Time_series_ran = False``.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    Time_Series_data : pandas.DataFrame or array-like
        Time-series table; converted to a one-column DataFrame if array-like
        (requires ``name``).
    associated : str, optional
        Target element name when the type and/or name are fixed by arguments.
    TS_type : str, optional
        Series type label (see Notes).
    name : str, optional
        Column label when ``Time_Series_data`` is not a DataFrame.

    Returns
    -------
    None

    Notes
    -----
    **Accepted ``TS_type`` values** (object names must be unique across classes):

    - ``'Load'`` — load factor on :class:`~pyflow_acdc.Classes.Price_Zone`,
      :class:`~pyflow_acdc.Classes.Node_AC`, :class:`~pyflow_acdc.Classes.Node_DC`
      (updates ``PLi_factor``)
    - ``'price'`` — energy price on price zones or nodes
    - ``'WPP'``, ``'OWPP'``, ``'SF'``, ``'REN'``, ``'Solar'`` — renewable
      availability on :class:`~pyflow_acdc.Classes.Ren_source_zone` or
      :class:`~pyflow_acdc.Classes.Ren_Source`` (``PRGi_available``)
    - Price-zone-only cost/limit series: ``'a_CG'``, ``'b_CG'``, ``'c_CG'``,
      ``'PGL_min'``, ``'PGL_max'``

    **CSV column layout** (each column is one series; row 0 is the column header
    when read by pandas):

    1. ``associated`` and ``TS_type`` both given — rows 0+ are numeric samples.
    2. ``associated`` only — row 0 = ``TS_type``, rows 1+ = samples.
    3. ``TS_type`` only — row 0 = element name, rows 1+ = samples.
    4. Neither given — row 0 = element name, row 1 = ``TS_type``, row 2+ =
       samples.

    Examples
    --------
    >>> load = pd.DataFrame({'Load_n1': [0.95, 0.75, 0.84]})
    >>> pyf.add_TimeSeries(grid, load, associated='node1', TS_type='Load')
    """
    # Check if Time_Series_data is a numpy array and convert to pandas DataFrame if needed
    if not isinstance(Time_Series_data, pd.DataFrame):
        TS = pd.DataFrame(Time_Series_data, columns=[name])
    else:
        TS = Time_Series_data
    Time_series = {}
    # check if there are nan values in Time series and change to 0
    TS.fillna(0, inplace=True)

    for col in TS.columns:
        if associated is not None and TS_type is not None:
            element_name = associated
            element_type = TS_type
            data = TS.loc[0:, col].astype(float).to_numpy()
            name = col

        elif associated is not None:
            element_name = associated
            element_type = TS.at[0, col]
            data = TS.loc[1:, col].astype(float).to_numpy()
            name = col

        elif TS_type is not None:
            element_name = TS.at[0, col]
            element_type = TS_type
            data = TS.loc[1:, col].astype(float).to_numpy()
            name = col

        else:
            element_name = TS.at[0, col]
            element_type = TS.at[1, col]
            data = TS.loc[2:, col].astype(float).to_numpy()
            name = col


        Time_serie = TimeSeries(element_type, element_name, data,name)
        grid.Time_series.append(Time_serie)
        grid.Time_series_dic[name]=Time_serie.TS_num
        time_series_dict(grid, Time_serie)



    grid.Time_series_ran = False
    s = 1


def assign_RenToZone(grid,ren_source_name,new_zone_name):
    """Move a renewable source into ``new_zone_name``.

    Parameters
    ----------
    grid : Grid
        Grid containing the source and zones.
    ren_source_name : str
        :class:`~pyflow_acdc.Classes.Ren_Source` name.
    new_zone_name : str
        Target :class:`~pyflow_acdc.Classes.Ren_source_zone` name.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the zone or source is not found.

    Examples
    --------
    >>> pyf.assign_RenToZone(grid, 'wind1', 'WindZone1')
    >>> pyf.assign_RenToZone(grid, 'wind2', 'WindZone1')

    Or assign via :func:`add_RenSource` using ``zone='WindZone1'``.
    """
    new_zone = None
    old_zone = None
    ren_source_to_reassign = None

    for RenZone in grid.RenSource_zones:
        if RenZone.name == new_zone_name:
            new_zone = RenZone
            break
    if new_zone is None:
        raise ValueError(f"Zone {new_zone_name} not found.")

    # Remove node from its old price_zone
    for RenZone in grid.RenSource_zones:
        for ren_source in RenZone.RenSources:
            if ren_source.name == ren_source_name:
                old_zone = RenZone
                ren_source_to_reassign = ren_source
                break
        if old_zone:
            break

    if old_zone is not None:
        RenZone.ren_source = [ren_source for ren_source in old_zone.RenSources
                               if ren_source.name != ren_source_name]

    # If the node was not found in any Renewable zone, check grid.nodes_AC
    if ren_source_to_reassign is None:
        for ren_source in grid.RenSources:
            if ren_source.name == ren_source_name:
                ren_source_to_reassign = ren_source
                break

    if ren_source_to_reassign is None:
        raise ValueError(f"Renewable source {ren_source_name} not found.")
    ren_source_to_reassign.PGRi_linked = True
    ren_source_to_reassign.Ren_source_zone = new_zone.name
    # Add node to the new price_zone
    if ren_source_to_reassign not in new_zone.RenSources:
        new_zone.RenSources.append(ren_source_to_reassign)

"Assigning components to zones"

def assign_nodeToPrice_Zone(grid, node, new_price_zone_name, ACDC='AC', link_load=True):
    """Assign a bus to a price zone (removes it from the previous zone).

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    node : Node_AC, Node_DC, or str
        Bus to assign.
    new_price_zone_name : str
        Target :class:`~pyflow_acdc.Classes.Price_Zone` name.
    ACDC : str, optional
        ``'AC'``, ``'DC'``, or ``'any'`` for name lookup side.
    link_load : bool, optional
        Set ``node.PLi_linked`` on the bus.

    Returns
    -------
    None

    Examples
    --------
    >>> pyf.assign_nodeToPrice_Zone(grid, 'bus1', 'Zone1', ACDC='AC')
    """
    acdc_norm = ACDC if ACDC in ('AC', 'DC') else 'any'
    nodes_attr = 'nodes_DC' if acdc_norm == 'DC' else 'nodes_AC'
    node = _look_up_node(grid, node, ac_or_dc=acdc_norm)
    new_price_zone = _look_up_price_zone(grid, new_price_zone_name)

    # Remove node from its old price_zone
    old_price_zone = None
    for price_zone in grid.Price_Zones:
        nodes = getattr(price_zone, nodes_attr)
        if node in nodes:
            old_price_zone = price_zone
            break
    if old_price_zone is not None:
        old_nodes = getattr(old_price_zone, nodes_attr)
        setattr(old_price_zone, nodes_attr, [n for n in old_nodes if n is not node])
        if hasattr(old_price_zone, "recalc_PLi_base_and_total"):
            old_price_zone.recalc_PLi_base_and_total()

    # Add node to the new price_zone
    new_price_zone_nodes = getattr(new_price_zone, nodes_attr)
    if node not in new_price_zone_nodes:
        new_price_zone_nodes.append(node)
        node.PZ = new_price_zone.name
        node.price = new_price_zone.price
        node.PLi_linked = link_load
        if hasattr(new_price_zone, "recalc_PLi_base_and_total"):
            new_price_zone.recalc_PLi_base_and_total()

def assign_ConvToPrice_Zone(grid, conv, new_price_zone_name):
    """Assign an AC/DC converter to a price zone.

    Parameters
    ----------
    grid : Grid
        Grid to modify.
    conv : AC_DC_converter or str
        Converter object or name.
    new_price_zone_name : str
        Target price zone name.

    Returns
    -------
    None
    """
    new_price_zone = _look_up_price_zone(grid, new_price_zone_name)
    conv_obj = _look_up_converter(grid, conv)

    # Remove converter from its old price_zone, if any
    for price_zone in grid.Price_Zones:
        if conv_obj in price_zone.ConvACDC:
            price_zone.ConvACDC = [c for c in price_zone.ConvACDC if c is not conv_obj]
            break

    # Add converter to the new price_zone
    if conv_obj not in new_price_zone.ConvACDC:
        new_price_zone.ConvACDC.append(conv_obj)

def assign_lineToCable_options(grid, line, new_cable_option_name):
    """Link a sizing line to a :class:`~pyflow_acdc.Classes.Cable_options` entry.

    Removes the line from any previous option and copies ``cable_types`` from
    the target option onto the line.

    Parameters
    ----------
    grid : Grid
        Grid containing the line and options.
    line : Size_selection or str
        CT line object or line name in ``grid.lines_AC_ct``.
    new_cable_option_name : str
        Name of the target cable option.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If the line or cable option is not found.
    """
    new_cable_option = None
    old_cable_option = None

    # Normalize to a line object present in grid.lines_AC_ct
    lines_ct = getattr(grid, "lines_AC_ct", [])
    if isinstance(line, str):
        line_obj = next((l for l in lines_ct if l.name == line), None)
        if line_obj is None:
            raise ValueError(f"Line {line} not found.")
    else:
        if line in lines_ct:
            line_obj = line
        else:
            name = getattr(line, "name", None)
            line_obj = next((l for l in lines_ct if l.name == name), None) if name is not None else None
            if line_obj is None:
                raise ValueError(f"Line {name or line} not found in grid.")

    for cable_option in grid.Cable_options:
        if cable_option.name == new_cable_option_name:
            new_cable_option = cable_option
            break

    if new_cable_option is None:
        raise ValueError(f"Cable_option {new_cable_option_name} not found.")

    # Remove line from its old cable_option (if any), using identity
    for cable_option in grid.Cable_options:
        if line_obj in cable_option.lines:
            old_cable_option = cable_option
            cable_option.lines = [l for l in cable_option.lines if l is not line_obj]
            break

    # Add line to the new cable_option
    if line_obj not in new_cable_option.lines:
        new_cable_option.lines.append(line_obj)
        line_obj.cable_types = new_cable_option._cable_types





def expand_cable_database(data, format='yaml', save_yaml=False):
    """Merge new cable entries into the bundled AC/DC cable database.

    Use this for local or custom cable data (YAML dict, DataFrame, or CSV path).
    To fetch the upstream ORBIT library from GitHub, use
    :func:`import_orbit_cables` instead.

    Parameters
    ----------
    data : str, pathlib.Path, dict, or pandas.DataFrame
        New cable specs. For ``format='yaml'``: YAML file path or dict keyed by
        cable name. For ``format='pandas'``: DataFrame or CSV path (index =
        cable name).
    format : str, optional
        ``'yaml'`` or ``'pandas'``.
    save_yaml : bool, optional
        If True, write each cable as ``Cable_database/<name>.yaml``.

    Returns
    -------
    None

    Notes
    -----
    Expected fields per cable (units):

    - ``R_Ohm_km`` (ohm/km), ``L_mH_km`` (mH/km), ``C_uF_km`` (µF/km),
      ``G_uS_km`` (µS/km)
    - ``A_rating`` (A), ``Nominal_voltage_kV`` (kV), ``MVA_rating`` (MVA)
    - ``conductor_size`` (mm²), ``Type`` (``'AC'`` or ``'DC'``), ``Reference``

    See also :func:`import_orbit_cables` for bulk import from ORBIT libraries.
    """

    # Get the path to the Cable_database directory
    module_dir = Path(__file__).parent.parent
    cable_dir = module_dir / 'Cable_database'

    if format.lower() == 'yaml':
        if isinstance(data, (str, Path)):
            with open(data, 'r') as f:
                new_cables = yaml.safe_load(f)
        elif isinstance(data, dict):
            new_cables = data
        else:
            raise ValueError("For YAML format, data must be either a file path or dictionary")

    elif format.lower() == 'pandas':
        if isinstance(data, pd.DataFrame):
            new_cables = data.to_dict(orient='index')
        elif isinstance(data, (str, Path)):
            df = pd.read_csv(data)
            new_cables = df.to_dict(orient='index')
        else:
            raise ValueError("For pandas format, data must be either a DataFrame or file path")


    if save_yaml:
        # Save each cable type as a separate file
        for cable_name, cable_specs in new_cables.items():
            # Create a single-cable dictionary
            cable_data = {cable_name: cable_specs}

            # Create file path using cable name
            output_file = cable_dir / f"{cable_name}.yaml"

            # Save to YAML file
            with open(output_file, 'w') as f:
                yaml.dump(cable_data, f, sort_keys=False)

            print(f"Saved cable {cable_name} to {output_file}")

    # split ac and dc cables
    new_cables_ac = {}
    new_cables_dc = {}
    for key, value in new_cables.items():
        tval = str(value.get('Type', 'AC')).upper()
        if tval in ('HVAC', 'AC'):
            new_cables_ac[key] = value
        else:
            new_cables_dc[key] = value

    # Update the cable database
    if Line_DC._cable_database is None:
        Line_DC.load_cable_database()
    if Line_AC._cable_database is None:
        Line_AC.load_cable_database()
    # Add new cables to existing database
    Line_DC._cable_database = pd.concat([
        Line_DC._cable_database,
        pd.DataFrame.from_dict(new_cables_dc, orient='index')
    ])
    Line_AC._cable_database = pd.concat([
        Line_AC._cable_database,
        pd.DataFrame.from_dict(new_cables_ac, orient='index')
    ])


    print(f"Added {len(new_cables_ac)} new cables to AC and {len(new_cables_dc)} new cables to DC database")


def import_orbit_cables(
    column_map=None,
    default_type='AC',
    name_prefix='NREL',
    save_yaml=False,
    source_url='https://github.com/NLRWindSystems/ORBIT/tree/dev/library/cables',
):
    """Fetch ORBIT cable library data from GitHub and merge into the cable database.

    Fetches YAML/CSV files from ``source_url``, normalises ORBIT column names
    and units, then calls :func:`expand_cable_database`. For local dict,
    DataFrame, or CSV input already in pyflow schema, call
    :func:`expand_cable_database` directly.

    Parameters
    ----------
    column_map : dict, optional
        Maps pyflow_acdc field names to source columns (``name``, ``R_Ohm_km``,
        ``L_mH_km``, ``C_uF_km``, ``G_uS_km``, ``A_rating``,
        ``Nominal_voltage_kV``, ``conductor_size``, ``Type``, ``Cost_per_km``,
        ``Reference``).
    default_type : str, optional
        Fallback cable type when missing (``'AC'`` or ``'DC'``).
    name_prefix : str, optional
        Prefix for auto-generated cable names.
    save_yaml : bool, optional
        If True, persist imported entries as YAML files.
    source_url : str, optional
        ORBIT GitHub directory URL (``.../tree/<ref>/<path>``).

    Returns
    -------
    pandas.DataFrame
        Normalised cable schema indexed by cable name.
    """

    def _read_text(url):
        req = Request(url, headers={'User-Agent': 'pyflow-acdc'})
        with urlopen(req) as resp:
            return resp.read().decode('utf-8')

    def _github_dir_to_rows(url):
        parsed = urlparse(url)
        parts = [p for p in parsed.path.split('/') if p]
        if len(parts) < 5 or parts[2] not in ('tree', 'blob'):
            raise ValueError(
                'Expected GitHub URL like: https://github.com/<owner>/<repo>/tree/<ref>/<path>'
            )
        owner, repo = parts[0], parts[1]
        ref = parts[3]
        dir_path = '/'.join(parts[4:])
        api_url = f'https://api.github.com/repos/{owner}/{repo}/contents/{dir_path}?ref={ref}'
        listing = json.loads(_read_text(api_url))
        if isinstance(listing, dict):
            listing = [listing]

        rows = []
        for item in listing:
            if item.get('type') != 'file':
                continue
            name = str(item.get('name', ''))
            dl = item.get('download_url')
            if not dl:
                continue

            if name.lower().endswith(('.yaml', '.yml')):
                y = yaml.safe_load(_read_text(dl))
                if not isinstance(y, dict):
                    continue
                # Handle nested {CableName: {...}} and flat {field: value}
                if len(y) == 1 and isinstance(next(iter(y.values())), dict):
                    cable_name = next(iter(y.keys()))
                    spec = next(iter(y.values()))
                    spec = dict(spec)
                    spec.setdefault('name', cable_name)
                    rows.append(spec)
                else:
                    rows.append(dict(y))
            elif name.lower().endswith('.csv'):
                rows.extend(pd.read_csv(dl).to_dict(orient='records'))
        return rows

    def _slug(text):
        s = re.sub(r'[^A-Za-z0-9]+', '_', str(text)).strip('_')
        return s or 'Cable'

    def _pick_col(df, explicit, candidates, required=False):
        if explicit is not None:
            if explicit not in df.columns:
                raise KeyError(f"Mapped column '{explicit}' not found in source data")
            return explicit
        for c in candidates:
            if c in df.columns:
                return c
        if required:
            raise KeyError(f"Missing required source column. Tried: {candidates}")
        return None

    rows = _github_dir_to_rows(source_url)
    if not rows:
        raise ValueError(f'No cable files found at URL: {source_url}')
    src = pd.DataFrame(rows)
    cmap = column_map or {}

    c_name = _pick_col(src, cmap.get('name'),
                       ['name', 'cable_name', 'Cable Name', 'id', 'ID'])
    c_r = _pick_col(src, cmap.get('R_Ohm_km'),
                    ['R_Ohm_km', 'r_ohm_km', 'resistance_ohm_km', 'ac_resistance', 'dc_resistance', 'Resistance (ohm/km)'],
                    required=True)
    c_l = _pick_col(src, cmap.get('L_mH_km'),
                    ['L_mH_km', 'l_mh_km', 'inductance_mh_km', 'Inductance (mH/km)'])
    c_c = _pick_col(src, cmap.get('C_uF_km'),
                    ['C_uF_km', 'c_uf_km', 'capacitance_uf_km', 'capacitance', 'Capacitance (uF/km)', 'Capacitance (nF/km)'])
    c_g = _pick_col(src, cmap.get('G_uS_km'),
                    ['G_uS_km', 'g_us_km', 'conductance_us_km', 'Conductance (uS/km)'])
    c_a = _pick_col(src, cmap.get('A_rating'),
                    ['A_rating', 'ampacity_a', 'ampacity', 'current_rating_a', 'current_capacity', 'Current (A)'],
                    required=True)
    c_kv = _pick_col(src, cmap.get('Nominal_voltage_kV'),
                     ['Nominal_voltage_kV', 'voltage_kv', 'rated_voltage_kv', 'rated_voltage', 'Voltage (kV)'],
                     required=True)
    c_cs = _pick_col(src, cmap.get('conductor_size'),
                     ['conductor_size', 'cross_section_mm2', 'area_mm2', 'size_mm2'])
    c_type = _pick_col(src, cmap.get('Type'),
                       ['Type', 'type', 'current_type', 'cable_type'])
    c_cost = _pick_col(src, cmap.get('Cost_per_km'),
                       ['Cost_per_km', 'cost_per_km', 'cost_eur_per_km', 'Cost (per km)'])
    c_ref = _pick_col(src, cmap.get('Reference'),
                      ['Reference', 'reference', 'source'])

    out_rows = []
    skipped_rows = 0
    for i, row in src.iterrows():
        nm = row[c_name] if c_name is not None else f'{name_prefix}_{i+1}'
        typ = row[c_type] if c_type is not None and pd.notna(row[c_type]) else default_type
        typ = str(typ).upper()
        if typ.startswith('HVAC') or typ == 'AC':
            typ = 'AC'
        elif typ.startswith('HVDC') or typ == 'DC':
            typ = 'DC'
        if typ not in ('AC', 'DC'):
            typ = default_type

        if pd.isna(row[c_kv]) or pd.isna(row[c_a]) or pd.isna(row[c_r]):
            skipped_rows += 1
            continue

        kv = float(row[c_kv])
        a_rating = float(row[c_a])
        if not np.isfinite(kv) or not np.isfinite(a_rating) or kv <= 0 or a_rating <= 0:
            skipped_rows += 1
            continue

        c_val = float(row[c_c]) if c_c is not None and pd.notna(row[c_c]) else 0.0
        # NREL/ORBIT exports often use nF/km for capacitance.
        # Convert to pyflow's expected uF/km when values look like nF/km scale.
        if c_val > 50:
            c_val = c_val / 1000.0
        size_val = float(row[c_cs]) if c_cs is not None and pd.notna(row[c_cs]) else np.nan
        name = f"{name_prefix}_{_slug(nm)}"

        out_rows.append({
            'name': name,
            'R_Ohm_km': float(row[c_r]),
            'L_mH_km': float(row[c_l]) if c_l is not None and pd.notna(row[c_l]) else 0.0,
            'C_uF_km': c_val,
            'G_uS_km': float(row[c_g]) if c_g is not None and pd.notna(row[c_g]) else 0.0,
            'A_rating': a_rating,
            'Nominal_voltage_kV': kv,
            'MVA_rating': SQRT_3 * kv * a_rating / 1000.0,
            'conductor_size': size_val,
            'Type': typ,
            'Cost_per_km': float(row[c_cost]) if c_cost is not None and pd.notna(row[c_cost]) else 1.0,
            'Reference': row[c_ref] if c_ref is not None and pd.notna(row[c_ref]) else 'ORBIT',
        })

    if not out_rows:
        raise ValueError('No valid cable rows were found after parsing ORBIT data.')

    out_df = pd.DataFrame(out_rows).set_index('name')
    out_df = out_df[~out_df.index.duplicated(keep='first')]

    expand_cable_database(out_df, format='pandas', save_yaml=save_yaml)
    if skipped_rows:
        print(f"Skipped {skipped_rows} cable rows with missing/invalid key fields.")
    return out_df
