from pathlib import Path

import pytest
import pyflow_acdc as pyf
from pyflow_acdc.Graph_and_plot import update_hovertexts
from pyflow_acdc.windfarm_loader import load_case_grid_and_geo
from pyflow_tests.test_constants import CABLE_TYPES_OFF66, MORAY_EAST_CABLE_DECISIONS


def _get_plot_context(grid):
    polygon = getattr(grid, "dev_polygon", None)
    exclusion_zones = getattr(grid, "exclusion_zones", None)
    export_cables = getattr(grid, "export_cables", None)

    if isinstance(polygon, list):
        try:
            from shapely.ops import unary_union
            polygon = unary_union(polygon) if polygon else None
        except Exception:
            polygon = polygon[0] if polygon else None

    if isinstance(exclusion_zones, list):
        try:
            from shapely.ops import unary_union
            exclusion_zones = unary_union(exclusion_zones) if exclusion_zones else None
        except Exception:
            exclusion_zones = exclusion_zones[0] if exclusion_zones else None

    if polygon is not None and exclusion_zones is not None:
        try:
            polygon = polygon.difference(exclusion_zones)
        except Exception:
            pass
    return polygon, export_cables


def _assign_manual_cables(grid):
    # Same cable assignment approach used in Graph_Creation/WES_ComparisonNL_L.py.
    grid.cab_types_allowed = len(CABLE_TYPES_OFF66)
    cable_index = {name: idx for idx, name in enumerate(CABLE_TYPES_OFF66)}
    for line in grid.lines_AC_ct:
        line.cable_types = list(CABLE_TYPES_OFF66)
        line.active_config = cable_index.get(MORAY_EAST_CABLE_DECISIONS.get(str(line.name), ""), -1)


def _load_moray_grid():
    return load_case_grid_and_geo("moray_east", source_tag="gebco")


def test_moray_east_assign_and_plot_outputs(tmp_path):
    pytest.importorskip("folium")
    pytest.importorskip("branca")
    pytest.importorskip("svgwrite")

    grid, _res = _load_moray_grid()
    _assign_manual_cables(grid)

    assert any(line.active_config >= 0 for line in grid.lines_AC_ct), "No cable type was assigned."

    final_polygon, export_cables = _get_plot_context(grid)
    svg_prefix = tmp_path / "moray_east_manual_network"
    html_3d_path = tmp_path / "moray_east_manual_3d.html"
    folium_prefix = tmp_path / "moray_east_manual_map"

    pyf.save_network_svg(
        grid,
        name=str(svg_prefix),
        width=1000,
        height=1000,
        journal=True,
        legend=False,
        square_ratio=True,
        poly=final_polygon,
        linestrings=export_cables,
    )
    pyf.plot_3D(grid, show=False, save_path=str(html_3d_path))
    pyf.plot_folium(
        grid,
        name=str(folium_prefix),
        show=False,
        polygon=final_polygon,
        linestrings=export_cables,
        clustering=False,
    )

    assert Path(f"{svg_prefix}.svg").exists(), "SVG export was not created."
    assert html_3d_path.exists(), "3D HTML export was not created."
    assert Path(f"{folium_prefix}.html").exists(), "Folium HTML export was not created."


def test_ns_mtdc_generates_hovertexts():
    grid, _res = pyf.cases['NS_MTDC']()

    # Exercise hovertext generation paths on a representative built-in grid.
    update_hovertexts(grid, "data")
    update_hovertexts(grid, "inPu")
    update_hovertexts(grid, "Real")

    assert len(grid.nodes_AC) > 0, "NS_MTDC did not provide AC nodes."
    assert hasattr(grid.nodes_AC[0], "hover_text") and grid.nodes_AC[0].hover_text

    if getattr(grid, "lines_AC", []):
        assert hasattr(grid.lines_AC[0], "hover_text") and grid.lines_AC[0].hover_text
    if getattr(grid, "lines_DC", []):
        assert hasattr(grid.lines_DC[0], "hover_text") and grid.lines_DC[0].hover_text
    if getattr(grid, "Converters_ACDC", []):
        assert hasattr(grid.Converters_ACDC[0], "hover_text") and grid.Converters_ACDC[0].hover_text


def run_test():
    """Run plotting tests from the legacy run_tests.py harness."""
    exit_code = pytest.main([__file__, "-q"])
    if exit_code == 0:
        print("tests_plot passed")
    else:
        print("tests_plot failed")


if __name__ == "__main__":
    run_test()