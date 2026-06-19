# -*- coding: utf-8 -*-
"""
Smoke tests for example_grids factory functions (PF / OPF / TEP / Wind_Array).

Each case module under ``pyflow_acdc/example_grids/`` is loaded and its primary
factory is called once. This mirrors ``pyf.cases[...]`` without invoking helper
functions that are also exported into that namespace (e.g. ``resolve_local_path``).
"""

import importlib.util
import inspect
from pathlib import Path

import pytest
import pyflow_acdc as pyf

EXAMPLE_GRIDS_DIR = Path(pyf.__file__).resolve().parent / "example_grids"
CASE_SUBDIRS = ("PF", "OPF", "TEP", "Wind_Array")

# Lightweight kwargs for factories that accept TEP / expansion / TS options.
CASE_KWARGS = {
    "NS_MTDC_2025": {"years_data": "24", "expandable": False},
    "case39": {"TEP": False},
    "case39_acdc": {"TEP": False},
    "case24_3zones_acdc": {"TEP": False},
    "case118_TEP": {"exp": "None"},
    "case118_TEP_DC": {"exp": "None"},
    "case_ACTIVSg500": {"TEP": False},
    "Texas7k_20210804": {"TEP": False},
}

# Very large grids: excluded from the default parametrized run (~1 min each).
SKIP_CASES = {"Texas7k_20210804"}


def _discover_case_files():
    case_files = []
    for subdir in CASE_SUBDIRS:
        folder = EXAMPLE_GRIDS_DIR / subdir
        if not folder.is_dir():
            continue
        case_files.extend(sorted(folder.glob("*.py")))
    return case_files


CASE_FILES = _discover_case_files()
CASE_FILES_DEFAULT = [f for f in CASE_FILES if f.stem not in SKIP_CASES]


def _load_module_from_path(file_path):
    module_name = f"_example_grid_{file_path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pick_factory(module, stem):
    candidates = [
        fn
        for _, fn in inspect.getmembers(module, inspect.isfunction)
        if not fn.__name__.startswith("_") and fn.__module__ == module.__name__
    ]
    if not candidates:
        raise AssertionError(f"No public factory function found in {module.__name__}")

    for fn in candidates:
        if fn.__name__ == stem:
            return fn
    return candidates[0]


def _assert_grid_result(case_name, result):
    assert isinstance(result, (list, tuple)), f"{case_name}: expected (grid, res) tuple"
    assert len(result) >= 2, f"{case_name}: expected at least grid and results"
    grid, res = result[0], result[1]
    assert grid is not None, f"{case_name}: grid is None"
    assert res is not None, f"{case_name}: results is None"
    assert hasattr(grid, "nodes_AC"), f"{case_name}: grid missing nodes_AC"
    assert hasattr(grid, "nodes_DC"), f"{case_name}: grid missing nodes_DC"
    assert len(grid.nodes_AC) > 0 or len(grid.nodes_DC) > 0, (
        f"{case_name}: grid has no AC or DC nodes"
    )


def _run_case_factory(case_file):
    case_name = case_file.stem
    module = _load_module_from_path(case_file)
    factory = _pick_factory(module, case_name)
    kwargs = CASE_KWARGS.get(case_name, {})
    result = factory(**kwargs)
    _assert_grid_result(case_name, result)


@pytest.mark.parametrize("case_file", CASE_FILES_DEFAULT, ids=lambda p: p.stem)
def test_example_grid_factory_loads_grid_and_results(case_file):
    """Load each example grid module and verify it returns grid/results."""
    _run_case_factory(case_file)


@pytest.mark.parametrize("case_file", CASE_FILES, ids=lambda p: p.stem)
def test_example_grid_is_registered_in_pyf_cases(case_file):
    """Primary factory name should exist in pyf.cases when names match."""
    case_name = case_file.stem
    module = _load_module_from_path(case_file)
    factory = _pick_factory(module, case_name)
    if factory.__name__ in pyf.cases:
        assert pyf.cases[factory.__name__] is factory


@pytest.mark.slow
@pytest.mark.parametrize("case_name", sorted(SKIP_CASES))
def test_slow_example_grid_factory_loads(case_name):
    """Optional smoke for very large cases (run with ``pytest -m slow``)."""
    case_file = next(f for f in CASE_FILES if f.stem == case_name)
    _run_case_factory(case_file)


def run_test():
    """Run example_grids smoke test from script entrypoint."""
    exit_code = pytest.main([__file__, "-q", "-m", "not slow"])
    if exit_code == 0:
        print("✓ Example grids smoke test passed")
    else:
        print("✗ Example grids smoke test failed")


if __name__ == "__main__":
    run_test()
