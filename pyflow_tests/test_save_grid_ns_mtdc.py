# -*- coding: utf-8 -*-
"""Round-trip test for ``save_grid_to_file`` on the NS_MTDC case."""

import enum
import importlib.util

import pytest
import pyflow_acdc as pyf
from pyflow_acdc.Export_files import create_dictionaries


def _load_saved_grid(py_path, factory_name):
    spec = importlib.util.spec_from_file_location(factory_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, factory_name)
    return factory()


def _normalize_export_value(value):
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, dict):
        return {k: _normalize_export_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_export_value(v) for v in value]
    if hasattr(value, "name") and hasattr(value, "price"):
        return value.name
    return value


def _sort_export_rows(rows, key):
    return sorted(rows, key=lambda row: row[key])


def _export_snapshot(grid):
    data = create_dictionaries(grid)
    return {
        "S_base": data["S_base"],
        "nodes_AC": _sort_export_rows(data["nodes_AC"], "Node_id"),
        "lines_AC": _sort_export_rows(data["lines_AC"], "Line_id"),
        "nodes_DC": _sort_export_rows(data["nodes_DC"], "Node_id"),
        "lines_DC": _sort_export_rows(data["lines_DC"], "Line_id"),
        "Converters_ACDC": _sort_export_rows(data["Converters_ACDC"], "Conv_id"),
        "Price_Zone": _sort_export_rows(data["Price_Zone"], "name"),
        "RenSource_zone": _sort_export_rows(data["RenSource_zone"], "name"),
        "Generators": _sort_export_rows(data["Generators"], "name"),
        "RenSources": _sort_export_rows(data["RenSources"], "ren_source_name"),
    }


def _assert_row_equal(row_a, row_b):
    assert row_a.keys() == row_b.keys()
    for key in row_a:
        val_a = _normalize_export_value(row_a[key])
        val_b = _normalize_export_value(row_b[key])
        if isinstance(val_a, (float, int)) and isinstance(val_b, (float, int)):
            assert float(val_a) == pytest.approx(float(val_b), rel=0, abs=1e-9)
        else:
            assert val_a == val_b


def _assert_grid_export_equivalent(grid_a, grid_b):
    snap_a = _export_snapshot(grid_a)
    snap_b = _export_snapshot(grid_b)
    assert snap_a.keys() == snap_b.keys()
    for section in snap_a:
        if section == "S_base":
            assert snap_a[section] == snap_b[section]
            continue
        assert len(snap_a[section]) == len(snap_b[section])
        for row_a, row_b in zip(snap_a[section], snap_b[section]):
            _assert_row_equal(row_a, row_b)


def test_save_and_reload_ns_mtdc_grid(tmp_path):
    """Export NS_MTDC to a .py loader and reload equivalent grid data."""
    grid_orig, _res_orig = pyf.cases["NS_MTDC"]()
    file_name = "ns_mtdc_export"

    pyf.save_grid_to_file(grid_orig, file_name, folder_name=str(tmp_path))
    saved_py = tmp_path / f"{file_name}.py"
    assert saved_py.is_file()

    grid_reload, res_reload = _load_saved_grid(saved_py, file_name)

    assert grid_reload.name == file_name
    assert res_reload is not None
    _assert_grid_export_equivalent(grid_orig, grid_reload)


def run_test():
    """Script entrypoint using a temporary directory."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="pyflow_ns_mtdc_save_") as tmpdir:
        test_save_and_reload_ns_mtdc_grid(Path(tmpdir))
    print("✓ NS_MTDC save_grid_to_file round-trip passed")


if __name__ == "__main__":
    run_test()
