# -*- coding: utf-8 -*-
"""
North Sea MTDC 2025 multi-period TEP example grid.

Data files live in ``examples/North_Sea_grid_data/`` (or GitHub raw URLs under the same
folder by default). Load with ``pyf.cases["NS_MTDC_2025"]()``.

Set ``expandable="mp"`` (default) for multi-period planning with investment
series CSVs, or ``expandable="step"`` for sequential / single-step MS TEP using
``Expandable_elements_step.csv``.
"""

from itertools import chain
from pathlib import Path

import pandas as pd
import pyflow_acdc as pyf
from shapely.geometry import LineString, Point

NORTH_SEA_GRID_DATA_GITHUB_BASE = (
    "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/North_Sea_grid_data/"
)


def _is_url(path):
    text = str(path)
    return text.startswith("http://") or text.startswith("https://")


def _north_sea_data_dir():
    data_dir = Path(__file__).resolve().parents[3] / "examples" / "North_Sea_grid_data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"North Sea grid data directory not found: {data_dir}. "
            "Expected examples/North_Sea_grid_data/ at the pyflow_acdc repository root."
        )
    return data_dir


def _resolve_example_path(filename, *, online=True):
    if _is_url(filename):
        return str(filename)
    if online:
        return NORTH_SEA_GRID_DATA_GITHUB_BASE + Path(filename).name
    path = _north_sea_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"North Sea grid data file not found: {path}")
    return str(path)


def NS_MTDC_2025(
    years_data='23,24',
    S_base=100,
    imp=1,
    export=1,
    per=1,
    AS=True,
    tee=False,
    expandable="mp",
    online=True,
):
    selected_years = [year.strip() for year in years_data.split(',')]

    AC_node_data = pd.read_csv(_resolve_example_path("NS_AC_node_data.csv", online=online))
    AC_line_data = pd.read_csv(_resolve_example_path("NS_AC_line_data.csv", online=online))
    DC_node_data = pd.read_csv(_resolve_example_path("NS_DC_node_data.csv", online=online))
    DC_line_data = pd.read_csv(_resolve_example_path("NS_DC_line_data.csv", online=online))
    Converter_ACDC_data = pd.read_csv(_resolve_example_path("NS_Converter_data.csv", online=online))

    extgrid_loads_mw = {
        'BE4': 1300,
        'BE7': 1300,
        'DE1': 2500,
        'DE3': 2500,
        'DE4': 2500,
        'DE6': 2500,
        'DK4': 700,
        'DK7': 340,
        'GB1': 500,
        'GB5': 1500,
        'GB21': 1000,
        'GB23': 1000,
        'GB25': 1000,
        'NL6': 900,
        'NL10': 900,
        'NO8': 500,
        'NO9': 500,
    }
    AC_node_data.loc[
        AC_node_data['Node_id'].isin(extgrid_loads_mw.keys()),
        'Power_load'
    ] = 0

    [grid, res] = pyf.create_grid_from_data(S_base, AC_node_data, AC_line_data, DC_node_data, DC_line_data, Converter_ACDC_data)
    # Enable MP-MS coupling constraint for standalone wind vs converter sizing.
    grid.enable_conv_wind_min_constraint = True
    grid.conv_wind_min_ratio = 0.9
    # Optional converter/DC-line sizing consistency band.
    grid.enable_conv_dcline_ratio_constraint = False #Experimental
    grid.conv_dcline_ratio_min = 0.9
    grid.conv_dcline_ratio_max = 1.1

    for node in chain(grid.nodes_AC, grid.nodes_DC):
        node.geometry = Point(node.x_coord, node.y_coord)

    geometry_map = {
        "dc_DE6_OFW_NO": [[10.159006,53.554383], [7.437744, 55.341642], [7.712402, 56.692442], [4.75708, 57.733971]],
        "dc_BE5_OFW_NO": [[3.597507, 51.113869], [2.647705, 54.162434], [3.02124, 55.366625], [4.75708, 57.733971]],
        "dc_TritonLink": [[2.493896, 51.387209], [3.284912, 52.422523], [3.218994, 54.895565], [5.152588, 56.237245], [7.316895, 56.46249]],
        "dc_OFW_DK_DE4": [[7.316895, 56.46249],[6.817017,55.416544],[9.488068, 53.550915]],
        "dc_OFW_GB4_OFW_DK": [[-0.032959, 55.534848], [4.921875, 56.273861], [7.316895, 56.46249]],
        "dc_BE2_OFW_NL": [[2.493896, 51.387209], [3.831482,52.497832], [3.878174, 52.716331]],
        "dc_OFW_GB5_DE3": [[-0.505371, 56.016808], [3.306885, 55.862982], [5.163574, 55.203953], [6.965332,54.470038],[8.398361,53.24755]],
        "dc_LionLink-NL":[[3.878174,52.716331],[3.88916,51.862924],[4.788666,51.698949]]
    }
    for line in chain(grid.lines_AC, grid.lines_DC):
        from_xy = (line.fromNode.x_coord, line.fromNode.y_coord)
        to_xy = (line.toNode.x_coord, line.toNode.y_coord)
        line.geometry = LineString(geometry_map[line.name]) if line.name in geometry_map else LineString([from_xy, to_xy])

    for conv in grid.Converters_ACDC:
        conv.geometry = LineString([(conv.Node_DC.x_coord, conv.Node_DC.y_coord), (conv.Node_AC.x_coord, conv.Node_AC.y_coord)])

    pyf.add_price_zone(grid, 'BE', 97.27, import_pu_L=imp, export_pu_G=export, positive_price_delta=50)
    pyf.add_price_zone(grid, 'DE', 95.67, import_pu_L=imp, export_pu_G=export, positive_price_delta=50)
    pyf.add_price_zone(grid, 'DK', 81.26, import_pu_L=imp, export_pu_G=export, positive_price_delta=50)
    pyf.add_price_zone(grid, 'GB', 108.23, import_pu_L=imp, export_pu_G=export, positive_price_delta=50)
    pyf.add_price_zone(grid, 'NL', 95.82, import_pu_L=imp, export_pu_G=export, positive_price_delta=50)
    pyf.add_price_zone(grid, 'NO', 79.44, import_pu_L=imp, export_pu_G=export, positive_price_delta=50)
 

    for index, _row in AC_node_data.iterrows():
        node_name = AC_node_data.at[index, 'Node_id']
        price_zone = AC_node_data.at[index, 'Market']
        pyf.assign_nodeToPrice_Zone(grid, node_name, price_zone, 'AC')

    pyf.add_extgrid(grid, 'BE4', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['BE4'])
    pyf.add_extgrid(grid, 'BE7', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['BE7'])
    pyf.add_extgrid(grid, 'DE1', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['DE1'])
    pyf.add_extgrid(grid, 'DE3', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['DE3'])
    pyf.add_extgrid(grid, 'DE4', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['DE4'])
    pyf.add_extgrid(grid, 'DE6', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['DE6'])
    pyf.add_extgrid(grid, 'DK4', MVAmax=1744, price_link=True, MVArmax=1744 / 3, MVArmin=-1744 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['DK4'])
    pyf.add_extgrid(grid, 'DK7', MVAmax=2330, price_link=True, MVArmax=2330 / 3, MVArmin=-2330 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['DK7'])
    pyf.add_extgrid(grid, 'GB1', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['GB1'])
    pyf.add_extgrid(grid, 'GB5', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['GB5'])
    pyf.add_extgrid(grid, 'GB21', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['GB21'])
    pyf.add_extgrid(grid, 'GB23', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['GB23'])
    pyf.add_extgrid(grid, 'GB25', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['GB25'])
    pyf.add_extgrid(grid, 'NL6', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['NL6'])
    pyf.add_extgrid(grid, 'NL10', MVAmax=4670, price_link=True, MVArmax=4670 / 3, MVArmin=-4670 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['NL10'])
    pyf.add_extgrid(grid, 'NO8', MVAmax=2173, price_link=True, MVArmax=2173 / 3, MVArmin=-2173 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['NO8'])
    pyf.add_extgrid(grid, 'NO9', MVAmax=2173, price_link=True, MVArmax=2173 / 3, MVArmin=-2173 / 3, Allow_sell=AS, P_load_MW=extgrid_loads_mw['NO9'])
    pyf.add_gen(grid, 'GB20', MWmax=545, MWmin=490.5, price_link=True)

    for z in ['BE', 'DE', 'DK', 'NL', 'NO', 'GB']:
        pyf.add_RenSource_zone(grid, z)
    pyf.add_RenSource(grid, 'BE2', 100, available=per, zone='BE', price_zone='BE', Offshore=True, np_rsgen=35)
    pyf.add_RenSource(grid, 'OFW_DEC', 100, available=per, zone='DE', price_zone='DE', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_DEC2', 100, available=per, zone='DE', price_zone='DE', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_DK', 100, available=per, zone='DK', price_zone='DK', Offshore=True, np_rsgen=35)
    pyf.add_RenSource(grid, 'OFW_NL', 100, available=per, zone='NL', price_zone='NL', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_NL2', 100, available=per, zone='NL', price_zone='NL', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_NL3', 100, available=per, zone='NL', price_zone='NL', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_NO', 100, available=per, zone='NO', price_zone='NO', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_GB1', 100, available=per, zone='GB', price_zone='GB', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_GB2', 100, available=per, zone='GB', price_zone='GB', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_GB3', 100, available=per, zone='GB', price_zone='GB', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_GB4', 100, available=per, zone='GB', price_zone='GB', Offshore=True, np_rsgen=20)
    pyf.add_RenSource(grid, 'OFW_GB5', 100, available=per, zone='GB', price_zone='GB', Offshore=True, np_rsgen=20)

    #for index, _row in Converter_ACDC_data.iterrows():
    #    conv_name = Converter_ACDC_data.at[index, 'Conv_id']
    #    price_zone = Converter_ACDC_data.at[index, 'Market']
    #    pyf.assign_ConvToPrice_Zone(grid, conv_name, price_zone)

    TS_MK_list = []
    TS_wl_list = []
    for year in selected_years:
        TS_MK_year = pd.read_csv(
            _resolve_example_path(f"NS_TS_marketPrices_data_sd20{year}.csv", online=online)
        )
        TS_wl_year = pd.read_csv(
            _resolve_example_path(f"NS_TS_WL_data20{year}.csv", online=online)
        )
        len_MK = len(TS_MK_year)
        if len(TS_wl_year) != len_MK:
            TS_wl_year = TS_wl_year.iloc[:len_MK]
        TS_MK_list.append(TS_MK_year)
        TS_wl_list.append(TS_wl_year)
        if tee:
            print(f'Year 20{year}: {len_MK - 2} time steps loaded')

    _TS_MK, TS_MK_all = pyf.Time_series.combine_TS(TS_MK_list, True)
    _TS_wl, TS_wl_all = pyf.Time_series.combine_TS(TS_wl_list, True)
    if tee:
        print(f'Combined time series: {len(TS_MK_all)} steps from {len(selected_years)} year(s)')

    pyf.add_TimeSeries(grid, TS_MK_all)
    pyf.add_TimeSeries(grid, TS_wl_all)
    if expandable:
        if expandable is True:
            expansion_mode = "mp"
        elif expandable in ("mp", "step"):
            expansion_mode = expandable
        else:
            raise ValueError(
                f"expandable must be False, True, 'mp', or 'step'; got {expandable!r}"
            )

        if expansion_mode == "step":
            exp_elements = pd.read_csv(
                _resolve_example_path("Expandable_elements_step.csv", online=online)
            )
            pyf.expand_elements_from_pd(grid, exp_elements)
        else:
            exp_elements = pd.read_csv(
                _resolve_example_path("Expandable_elements.csv", online=online)
            )
            pyf.expand_elements_from_pd(grid, exp_elements)

            pyf.add_inv_series(
                grid, _resolve_example_path("NS_exp_MP_planned_intalled.csv", online=online)
            )
            pyf.add_inv_series(
                grid, _resolve_example_path("NS_exp_MP_max_intall_per_period.csv", online=online)
            )
            pyf.add_inv_series(
                grid, _resolve_example_path("NS_exp_MP_price_zones.csv", online=online)
            )

            for line in grid.lines_DC:
                line.investment_decisions['lambda_capex'] = [0, -0.03, -0.08]
            for conv in grid.Converters_ACDC:
                conv.investment_decisions['lambda_capex'] = [0, -0.06, -0.15]
            for rs in grid.RenSources:
                rs.investment_decisions['lambda_capex'] = [0, -0.10, -0.25]

    return grid, res


if __name__ == "__main__":
    
    grid, _res = NS_MTDC_2025(tee=True)
    from importlib import import_module
    mp_module = import_module("pyflow_acdc.ACDC_MultiPeriod_TEP")
    # Keep all investment-decision vectors aligned across keys (Load/curvature/import_expand/etc).
    n_inv_periods = mp_module._fill_investment_decisions(grid)

    def _export_element_unit_costs_by_inv_period_csv(grid_obj, n_periods, out_csv_path):
        """
        Export per-investment-period *unit* CAPEX by element in wide format.

        Unit cost definition:
            UnitCost_i = _base_cost * (1 + lambda_capex[i])

        This is "what it would cost to install 1" in each investment period.
        """
        def _broadcast_to_periods(values):
            if isinstance(values, (list, tuple)):
                if len(values) == 1 and n_periods > 1:
                    return [float(values[0])] * int(n_periods)
                if len(values) != n_periods:
                    raise ValueError(f"lambda_capex length {len(values)} does not match n_periods={n_periods}")
                return [float(v) for v in values]
            return [float(values)] * int(n_periods)

        all_elements = (
            grid_obj.lines_AC
            + grid_obj.lines_AC_tf
            + grid_obj.lines_AC_rec
            + grid_obj.lines_AC_ct
            + grid_obj.lines_AC_exp
            + grid_obj.lines_DC
            + grid_obj.Converters_ACDC
            + grid_obj.Generators
            + grid_obj.RenSources
        )

        rows = []
        for element in all_elements:
            inv = getattr(element, "investment_decisions", None)
            if not isinstance(inv, dict) or "lambda_capex" not in inv:
                continue

            if not hasattr(element, "_base_cost"):
                raise ValueError(f"Element '{element.name}' missing _base_cost.")

            lambda_series = _broadcast_to_periods(inv["lambda_capex"])
            row = {"Element": str(element.name), "Type": type(element).__name__}
            for i in range(int(n_periods)):
                row[f"Cost_{i+1}"] = float(element._base_cost) * (1.0 + float(lambda_series[i]))
            rows.append(row)

        df = pd.DataFrame(rows)
        if df.empty:
            raise RuntimeError("No elements with investment_decisions['lambda_capex'] found; CSV would be empty.")
        df.to_csv(out_csv_path, index=False)
        return df

    print('\n[NS_MTDC_2025_setup __main__] Grid debug')
    print(f"nodes AC={len(grid.nodes_AC)}, nodes DC={len(grid.nodes_DC)}, zones={len(grid.Price_Zones)}")
    total_load_base = sum(node.PLi for node in grid.nodes_AC) + sum(node.PLi for node in grid.nodes_DC)
    print(f"total base system load (pu)={total_load_base:.6f}, MW={total_load_base * grid.S_base:.2f}")
    for pz in grid.Price_Zones:
        inv_load = pz.investment_decisions.get('Load', None)
        print(f"zone={pz.name} PLi_inv_factor={pz.PLi_inv_factor} inv_Load={inv_load}")

    costs_csv_path = _ns_mp_data_dir() / "NS_MTDC_2025_costs_by_investment_period.csv"
    costs_csv_path.parent.mkdir(parents=True, exist_ok=True)
    costs_df = _export_element_unit_costs_by_inv_period_csv(
        grid_obj=grid,
        n_periods=n_inv_periods,
        out_csv_path=str(costs_csv_path),
    )
    print(f"\n[NS_MTDC_2025_setup __main__] Wrote element costs CSV: {costs_csv_path}")
    print(costs_df.head(20))

    # Apply planned installations to current stocks for a quick visualization.
    for line in grid.lines_AC_exp:
        if hasattr(line, "planned_installation"):
            line.np_line = line.np_line_b + line.planned_installation
    for line in grid.lines_DC:
        if hasattr(line, "planned_installation"):
            line.np_line = line.np_line_b + line.planned_installation
    for conv in grid.Converters_ACDC:
        if hasattr(conv, "planned_installation"):
            conv.np_conv = conv.np_conv_b + conv.planned_installation
    for gen in grid.Generators:
        if hasattr(gen, "planned_installation"):
            gen.np_gen = gen.np_gen_b + gen.planned_installation
    for rs in grid.RenSources:
        if hasattr(rs, "planned_installation"):
            rs.np_rsgen = rs.np_rsgen_b + rs.planned_installation

    #pyf.save_network_svg(
    #    grid,
    #    name=str(Path(__file__).parent / "North_sea_MTDC" / "NS_MTDC_2025_np1"),
    #    square_ratio=False,
    #)
    # Sanity checks:
    # 1) DC geometry-based length (km) vs element.Length_km
    # 2) Expandability flags for all DC lines and converters
    # Note: geometry coords are provided as (lon, lat) degrees in `geometry_map` above.
    from math import radians, sin, cos, sqrt, atan2

    def _haversine_km(lon1, lat1, lon2, lat2, earth_radius_km=6371.0088):
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return earth_radius_km * c

    dc_missing_expand = []
    conv_missing_expand = []
    dc_geometry_missing = []

    print("\n[NS_MTDC_2025_setup __main__] DC geometry sanity (all dc lines)")
    for dc_line in grid.lines_DC:
        name = dc_line.name

        # Expandability checks (set by `pyf.expand_elements_from_pd` / `Expand_element`)
        if not getattr(dc_line, "np_line_opf", False):
            dc_missing_expand.append(name)

        if not hasattr(dc_line, "geometry") or dc_line.geometry is None:
            dc_geometry_missing.append(name)
            print(f"DC line={name} geometry missing (Length_km={getattr(dc_line, 'Length_km', None)})")
            continue

        coords = list(dc_line.geometry.coords)  # list of (lon, lat)
        length_km = 0.0
        for (lon1, lat1), (lon2, lat2) in zip(coords[:-1], coords[1:]):
            length_km += _haversine_km(lon1, lat1, lon2, lat2)

        element_length_km = getattr(dc_line, "Length_km", None)
        print(
            f"DC line={name} computed_km={length_km:.3f} element.Length_km={element_length_km} "
            f"(coords={len(coords)})"
        )

    print("\n[NS_MTDC_2025_setup __main__] Converter expandability sanity (all converters)")
    for conv in grid.Converters_ACDC:
        if not getattr(conv, "np_conv_opf", False):
            conv_missing_expand.append(conv.name)

    if dc_missing_expand:
        print(f"Missing DC expandability (np_line_opf=False) count={len(dc_missing_expand)}")
        print(f"Missing DC: {sorted(dc_missing_expand)}")
    else:
        print("All DC lines have np_line_opf=True")

    if conv_missing_expand:
        print(f"Missing converter expandability (np_conv_opf=False) count={len(conv_missing_expand)}")
        print(f"Missing converters: {sorted(conv_missing_expand)}")
    else:
        print("All converters have np_conv_opf=True")

    if dc_geometry_missing:
        print(f"DC lines missing geometry count={len(dc_geometry_missing)}")
        print(f"Missing DC geometry: {sorted(dc_geometry_missing)}")
    all_elements = (
        grid.lines_AC
        + grid.lines_AC_tf
        + grid.lines_AC_rec
        + grid.lines_AC_ct
        + grid.lines_AC_exp
        + grid.lines_DC
        + grid.Converters_ACDC
        + grid.Generators
        + grid.RenSources
    )

    lambda_period_max = 0
    for element in all_elements:
        inv = getattr(element, "investment_decisions", None)
        if isinstance(inv, dict):
            lambda_series = inv.get("lambda_capex")
            if isinstance(lambda_series, (list, tuple)) and len(lambda_series) > 0:
                lambda_period_max = max(lambda_period_max, len(lambda_series) - 1)

    for np_tag in ("np0", "np1", "npmax_inv0", "npmax_inv1", "npmax_inv2", "npmax"):
        if np_tag.startswith("npmax_inv"):
            lambda_period = int(np_tag[-1])
        elif np_tag in ("np0", "np1"):
            lambda_period = 0
        else:
            lambda_period = lambda_period_max

        for element in all_elements:
            inv = getattr(element, "investment_decisions", None)
            if isinstance(inv, dict):
                lambda_series = inv.get("lambda_capex")
                if isinstance(lambda_series, (list, tuple)) and len(lambda_series) > 0 and hasattr(element, "lambda_capex"):
                    idx = min(lambda_period, len(lambda_series) - 1)
                    element.lambda_capex = float(lambda_series[idx])

            if hasattr(element, "np_line") and getattr(element, "np_line_opf", False):
                if np_tag == "np0":
                    element.np_line = getattr(element, "np_line_b")
                elif np_tag == "np1":
                    element.np_line = element.investment_decisions["planned_installation"][0]
                elif np_tag.startswith("npmax_inv"):
                    i = int(np_tag[-1])
                    add_i = float(
                        element.investment_decisions["max_inv"][i]
                        + element.investment_decisions["planned_installation"][i]
                    )
                    if i == 0:
                        element.np_line = min(add_i, float(getattr(element, "np_line_max")))
                    else:
                        element.np_line = min(
                            float(getattr(element, "np_line")) + add_i,
                            float(getattr(element, "np_line_max")),
                        )
                else:
                    element.np_line = getattr(element, "np_line_max")

            if hasattr(element, "np_conv") and getattr(element, "np_conv_opf", False):
                if np_tag == "np0":
                    element.np_conv = getattr(element, "np_conv_b")
                elif np_tag == "np1":
                    element.np_conv = element.investment_decisions["planned_installation"][0]
                elif np_tag.startswith("npmax_inv"):
                    i = int(np_tag[-1])
                    add_i = float(
                        element.investment_decisions["max_inv"][i]
                        + element.investment_decisions["planned_installation"][i]
                    )
                    if i == 0:
                        element.np_conv = min(add_i, float(getattr(element, "np_conv_max")))
                    else:
                        element.np_conv = min(
                            float(getattr(element, "np_conv")) + add_i,
                            float(getattr(element, "np_conv_max")),
                        )
                else:
                    element.np_conv = getattr(element, "np_conv_max")

            if hasattr(element, "np_gen") and getattr(element, "np_gen_opf", False):
                if np_tag == "np0":
                    element.np_gen = getattr(element, "np_gen_b")
                elif np_tag == "np1":
                    element.np_gen = element.investment_decisions["planned_installation"][0]
                elif np_tag.startswith("npmax_inv"):
                    i = int(np_tag[-1])
                    add_i = float(
                        element.investment_decisions["max_inv"][i]
                        + element.investment_decisions["planned_installation"][i]
                    )
                    if i == 0:
                        element.np_gen = min(add_i, float(getattr(element, "np_gen_max")))
                    else:
                        element.np_gen = min(
                            float(getattr(element, "np_gen")) + add_i,
                            float(getattr(element, "np_gen_max")),
                        )
                else:
                    element.np_gen = getattr(element, "np_gen_max")

            if hasattr(element, "np_rsgen") and getattr(element, "np_rsgen_opf", False):
                if np_tag == "np0":
                    element.np_rsgen = getattr(element, "np_rsgen_b")
                elif np_tag == "np1":
                    element.np_rsgen = element.investment_decisions["planned_installation"][0]
                elif np_tag.startswith("npmax_inv"):
                    i = int(np_tag[-1])
                    add_i = float(
                        element.investment_decisions["max_inv"][i]
                        + element.investment_decisions["planned_installation"][i]
                    )
                    if i == 0:
                        element.np_rsgen = min(add_i, float(getattr(element, "np_rsgen_max")))
                    else:
                        element.np_rsgen = min(
                            float(getattr(element, "np_rsgen")) + add_i,
                            float(getattr(element, "np_rsgen_max")),
                        )
                else:
                    element.np_rsgen = getattr(element, "np_rsgen_max")

        pyf.save_network_svg(
            grid,
            name=str(_ns_mp_data_dir() / f"NS_MTDC_2025_{np_tag}_svg"),
            square_ratio=True,
            line_size_factor=0.5,
            scale_ac_nodes_with_rs=True,
            node_size_factor=.05,
            scale_dc_nodes_with_conv=True,
            dc_node_size_factor=0.5,
            draw_converters=False,
        )
        if np_tag in ("np1", "npmax") or np_tag.startswith("npmax_inv"):
            pyf.plot_folium(
                grid,
                name=str(_ns_mp_data_dir() / f"NS_MTDC_2025_{np_tag}"),
            )
    
    print('\n[NS_MTDC_2025_setup __main__] Post-install state debug')
    total_load_post = sum(node.PLi for node in grid.nodes_AC) + sum(node.PLi for node in grid.nodes_DC)
    print(f"total load after planned-install update (pu)={total_load_post:.6f}, MW={total_load_post * grid.S_base:.2f}")