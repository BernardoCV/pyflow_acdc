"""8-turbine string array with cable options for CSS/TEP array-sizing tests.

Load with ``pyf.cases["array_sizing_pei"](gamma_limit=0.9)``.
"""
import pyflow_acdc as pyf


def array_sizing_pei(gamma_limit=0.9, power_rating=9.5, LCoE=79):
    S_base = 100

    grid, res = pyf.create_grid_from_data(S_base)

    SS = pyf.add_AC_node(grid, kV_base=66, node_type='Slack', name='SS')
    T1 = pyf.add_AC_node(grid, kV_base=66, name='T1')
    T2 = pyf.add_AC_node(grid, kV_base=66, name='T2')
    T3 = pyf.add_AC_node(grid, kV_base=66, name='T3')
    T4 = pyf.add_AC_node(grid, kV_base=66, name='T4')
    T5 = pyf.add_AC_node(grid, kV_base=66, name='T5')
    T6 = pyf.add_AC_node(grid, kV_base=66, name='T6')
    T7 = pyf.add_AC_node(grid, kV_base=66, name='T7')
    T8 = pyf.add_AC_node(grid, kV_base=66, name='T8')

    for node in ('T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8'):
        pyf.add_RenSource(grid, node, power_rating, min_gamma=gamma_limit, Qrel=0.3)

    pyf.add_extgrid(grid, 'SS', lf=LCoE)

    LT8_T7 = 1.4
    LT7_T6 = 1.4
    LT6_T5 = 1.4
    LT5_T4 = 1.1
    LT4_T3 = 1.2
    LT3_T2 = 1.5
    LT2_T1 = 1.5
    LT1_SS = 3.0

    cable_option = pyf.add_cable_option(grid, [
        'ABB_XLPE_Cu_66kV_sub_95mm2',
        'ABB_XLPE_Cu_66kV_sub_120mm2',
        'ABB_XLPE_Cu_66kV_sub_150mm2',
        'ABB_XLPE_Cu_66kV_sub_185mm2',
        'ABB_XLPE_Cu_66kV_sub_240mm2',
        'ABB_XLPE_Cu_66kV_sub_300mm2',
        'ABB_XLPE_Cu_66kV_sub_400mm2',
        'ABB_XLPE_Cu_66kV_sub_500mm2',
        'ABB_XLPE_Cu_66kV_sub_630mm2',
        'ABB_XLPE_Cu_66kV_sub_800mm2',
        'ABB_XLPE_Cu_66kV_sub_1000mm2',
    ], 'PEI')

    pyf.add_line_sizing(grid, T8, T7, cable_option=cable_option.name, active_config=0, Length_km=LT8_T7, name='T8_T7', update_grid=False)
    pyf.add_line_sizing(grid, T7, T6, cable_option=cable_option.name, active_config=0, Length_km=LT7_T6, name='T7_T6', update_grid=False)
    pyf.add_line_sizing(grid, T6, T5, cable_option=cable_option.name, active_config=0, Length_km=LT6_T5, name='T6_T5', update_grid=False)
    pyf.add_line_sizing(grid, T5, T4, cable_option=cable_option.name, active_config=1, Length_km=LT5_T4, name='T5_T4', update_grid=False)
    pyf.add_line_sizing(grid, T4, T3, cable_option=cable_option.name, active_config=3, Length_km=LT4_T3, name='T4_T3', update_grid=False)
    pyf.add_line_sizing(grid, T3, T2, cable_option=cable_option.name, active_config=5, Length_km=LT3_T2, name='T3_T2', update_grid=False)
    pyf.add_line_sizing(grid, T2, T1, cable_option=cable_option.name, active_config=6, Length_km=LT2_T1, name='T2_T1', update_grid=False)
    pyf.add_line_sizing(grid, T1, SS, cable_option=cable_option.name, active_config=8, Length_km=LT1_SS, name='T1_SS', update_grid=False)

    grid.create_Ybus_AC()
    grid.update_graph_ac()

    grid.name = 'array_sizing_pei'
    return grid, res
