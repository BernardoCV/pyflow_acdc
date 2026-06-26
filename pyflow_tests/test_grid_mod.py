# -*- coding: utf-8 -*-
"""``grid_creator`` tests: sub-grid, turbine-graph build, and extend-from-data."""

import copy
import gzip
import pickle
from pathlib import Path

import pandas as pd
import pyflow_acdc as pyf
from pyflow_acdc.grid_creator import create_sub_grid

ARRAY_BUNDLE_PATH = Path(__file__).resolve().parent / "alpha_ventus_flat.pkl.gz"
_STAGG5_DATA = Path(__file__).resolve().parents[1] / "examples" / "Stagg5MATACDC"


def _load_array_bundle(path=ARRAY_BUNDLE_PATH):
    if not path.is_file():
        raise FileNotFoundError(
            f"Array graph bundle not found: {path} "
            "(run Graph_Creation/saving_farms.py to generate alpha_ventus_flat.pkl.gz)"
        )
    with gzip.open(path, "rb") as f:
        bundle = pickle.load(f)
    if not isinstance(bundle, (tuple, list)) or len(bundle) != 3:
        raise TypeError(
            f"Expected (array_graph, Data, cable_types) tuple, got {type(bundle).__name__}"
        )
    return bundle


def _bundle_to_extend_tables(array_graph, Data):
    """Build AC node and line DataFrames from an array graph bundle for ``extend_grid_from_data``."""
    sub_key = (
        "offshore_substation"
        if "offshore_substation" in Data
        else "transformer_station"
    )
    node_rows = []
    for i, attrs in array_graph.nodes(data=True):
        point_type = attrs.get("point_type")
        if point_type == "access_point":
            continue
        if point_type == "turbine":
            src = Data["turbine"].loc[attrs["original_idx"]]
        elif point_type == "substation":
            src = Data[sub_key].loc[attrs["original_idx"]]
        else:
            continue
        node_rows.append({
            "Node_id": str(i),
            "geometry": src["geometry"],
            "kV_base": float(src["kV_rating"]),
        })
    nodes = pd.DataFrame(node_rows)

    line_rows = []
    for line_id, (u, v, attrs) in enumerate(array_graph.edges(data=True)):
        u_type = array_graph.nodes[u].get("point_type")
        v_type = array_graph.nodes[v].get("point_type")
        if u_type == "access_point" or v_type == "access_point":
            continue
        line_rows.append({
            "Line_id": f"AC{line_id}",
            "fromNode": str(u),
            "toNode": str(v),
            "R_Ohm_km": 0.1,
            "L_mH_km": 0.4,
            "C_uF_km": 0.2,
            "A_rating": 500.0,
            "Length_km": attrs["length"] / 1000.0,
            "geometry": attrs.get("geometry"),
        })
    lines = pd.DataFrame(line_rows)
    return nodes, lines


def test_create_sub_grid_ns_mtdc_be():
    grid, _ = pyf.cases["NS_MTDC"]()
    n_ac_full = len(grid.nodes_AC)
    n_dc_full = len(grid.nodes_DC)
    assert n_ac_full > 0
    assert n_dc_full > 0

    grid_copy = copy.deepcopy(grid)
    subgrid, res = create_sub_grid(grid_copy, Area_name="BE")

    assert res is not None
    assert len(subgrid.nodes_AC) > 0
    assert len(subgrid.nodes_AC) < n_ac_full
    assert len(subgrid.nodes_DC) <= n_dc_full
    assert any(pz.name == "BE" for pz in subgrid.Price_Zones)
    assert any(node.PZ == "BE" for node in subgrid.nodes_AC)


def test_create_grid_from_turbine_graph_alpha_ventus_bundle():
    array_graph, Data, cable_types = _load_array_bundle()
    n_turbines = len(Data["turbine"])
    n_edges = array_graph.number_of_edges()
    assert n_turbines > 0
    assert n_edges > 0
    assert len(cable_types) > 0

    grid, res = pyf.create_grid_from_turbine_graph(
        array_graph,
        Data,
        cable_types=cable_types,
        name="alpha_ventus",
    )

    assert res is not None
    assert grid.Array_opf is True
    assert len(grid.nodes_AC) > 0
    assert len(grid.lines_AC_ct) > 0
    assert len(grid.RenSources) == n_turbines
    assert len(grid.Cable_options) == 1
    assert grid.crossing_groups is not None


def test_extend_grid_from_data_stagg5_partial_csv():
    """Extend an empty grid with Stagg5MATACDC CSV tables (nodes first, then rest)."""
    grid, _ = pyf.create_grid_from_data(
        100,
        AC_node_data=str(_STAGG5_DATA / "MATACDC_AC_node_data.csv"),
        data_in="pu",
    )
    n_ac_nodes = len(grid.nodes_AC)
    assert n_ac_nodes == 5
    assert len(grid.lines_AC) == 0

    pyf.extend_grid_from_data(
        grid,
        AC_line_data=str(_STAGG5_DATA / "MATACDC_AC_line_data.csv"),
        DC_node_data=str(_STAGG5_DATA / "MATACDC_DC_node_data.csv"),
        DC_line_data=str(_STAGG5_DATA / "MATACDC_DC_line_data.csv"),
        Converter_data=str(_STAGG5_DATA / "MATACDC_Converter_data.csv"),
        data_in="pu",
    )

    assert len(grid.nodes_AC) == n_ac_nodes
    assert len(grid.lines_AC) == 7
    assert len(grid.nodes_DC) == 3
    assert len(grid.lines_DC) == 3
    assert len(grid.Converters_ACDC) == 3


def test_extend_grid_from_data_alpha_ventus_bundle():
    """Extend empty grid with Real-valued node/line tables built from array bundle."""
    array_graph, Data, _ = _load_array_bundle()
    nodes, lines = _bundle_to_extend_tables(array_graph, Data)

    grid, _ = pyf.create_grid_from_data(100)
    assert len(grid.nodes_AC) == 0
    assert len(grid.lines_AC) == 0

    pyf.extend_grid_from_data(
        grid,
        AC_node_data=nodes,
        AC_line_data=lines,
        data_in="Real",
    )

    assert len(grid.nodes_AC) == len(nodes)
    assert len(grid.lines_AC) == len(lines)
    assert len(grid.nodes_AC) > len(Data["turbine"])
    assert grid.Graph_AC is not None


def run_test():
    test_create_sub_grid_ns_mtdc_be()
    test_create_grid_from_turbine_graph_alpha_ventus_bundle()
    test_extend_grid_from_data_stagg5_partial_csv()
    test_extend_grid_from_data_alpha_ventus_bundle()
    print("✓ grid_mod tests passed")


if __name__ == "__main__":
    run_test()
