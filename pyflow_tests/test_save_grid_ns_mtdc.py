# -*- coding: utf-8 -*-
"""Round-trip test for ``save_grid_to_file`` on the NS_MTDC case."""

import importlib.util

import pyflow_acdc as pyf


def _load_saved_grid(py_path, factory_name):
    spec = importlib.util.spec_from_file_location(factory_name, py_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    factory = getattr(module, factory_name)
    return factory()


def test_save_and_reload_ns_mtdc_grid(tmp_path):
    """Export NS_MTDC to a .py loader and reload it."""
    grid_orig, _res_orig = pyf.cases["NS_MTDC"]()
    file_name = "ns_mtdc_export"

    pyf.save_grid_to_file(grid_orig, file_name, folder_name=str(tmp_path))
    saved_py = tmp_path / f"{file_name}.py"
    assert saved_py.is_file()

    grid_reload, res_reload = _load_saved_grid(saved_py, file_name)

    assert grid_reload.name == file_name
    assert grid_reload.S_base == grid_orig.S_base
    assert grid_reload.nn_AC == grid_orig.nn_AC
    assert grid_reload.nn_DC == grid_orig.nn_DC
    assert grid_reload.nconv == grid_orig.nconv
    assert len(grid_reload.Price_Zones) == len(grid_orig.Price_Zones)
    assert len(grid_reload.RenSources) == len(grid_orig.RenSources)
    assert res_reload is not None

    pyf.acdc_sequential(grid_reload, maxIter=200)
    assert grid_reload.ACmode or grid_reload.DCmode


def run_test():
    """Script entrypoint using a temporary directory."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="pyflow_ns_mtdc_save_") as tmpdir:
        test_save_and_reload_ns_mtdc_grid(Path(tmpdir))
    print("✓ NS_MTDC save_grid_to_file round-trip passed")


if __name__ == "__main__":
    run_test()
