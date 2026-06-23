"""Shared dependency and solver checks for script-style case tests.

``require_*()`` — for pytest ``test_*()`` functions; calls :func:`pytest.skip`.
``*_missing_for_run_test()`` — for ``run_test()`` script entrypoints used by
``run_tests.py``; prints a skip line and returns ``True`` so the runner exits
quietly without an exception.

Each optional dependency uses a private ``_*_available()`` helper shared by both
entry points.
"""

import os

import pytest

import pyflow_acdc as pyf


def _missing_for_run_test(available_fn, skip_message):
    if available_fn():
        return False
    print(skip_message)
    return True


def _require(available_fn, skip_message):
    if not available_fn():
        pytest.skip(skip_message)


def _pyomo_available():
    try:
        __import__("pyomo")
    except Exception:
        return False
    return True


def pyomo_missing_for_run_test():
    """Return True when pyomo is absent (``run_test`` should return early)."""
    return _missing_for_run_test(_pyomo_available, "Skipped: pyomo is not installed")


def require_pyomo():
    _require(_pyomo_available, "pyomo is not installed")


def _folium_available():
    try:
        __import__("folium")
    except Exception:
        return False
    return True


def folium_missing_for_run_test():
    return _missing_for_run_test(_folium_available, "Skipped: folium is not installed")


def require_folium():
    _require(_folium_available, "folium is not installed")


def _dash_available():
    try:
        __import__("dash")
    except Exception:
        return False
    return True


def dash_missing_for_run_test():
    return _missing_for_run_test(_dash_available, "Skipped: dash is not installed")


def require_dash():
    _require(_dash_available, "dash is not installed")


def _dill_available():
    try:
        __import__("dill")
    except Exception:
        return False
    return True


def dill_missing_for_run_test():
    return _missing_for_run_test(_dill_available, "Skipped: dill is not installed")


def require_dill():
    _require(_dill_available, "dill is not installed")


def _ortools_available():
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except Exception:
        return False
    return True


def ortools_missing_for_run_test():
    """Return True when OR-Tools CP-SAT is absent (``run_test`` should return early)."""
    return _missing_for_run_test(_ortools_available, "Skipped: OR-Tools is not installed")


def require_ortools():
    _require(_ortools_available, "OR-Tools is not installed")


def tep_solver():
    """TEP solver for tests.

    Default is Ipopt (fast NLP). For full MINLP expansion solves, run with
    ``PYFLOW_TEP_SOLVER=bonmin`` when Bonmin is installed.
    """
    forced = os.environ.get("PYFLOW_TEP_SOLVER", "").strip().lower()
    if forced:
        if pyf.is_pyomo_solver_available(forced):
            return forced
        return forced
    if pyf.is_pyomo_solver_available("ipopt"):
        return "ipopt"
    if pyf.is_pyomo_solver_available("bonmin"):
        return "bonmin"
    return "ipopt"


def mip_solvers():
    """Return ``(MIP_solver, CSS_L_solver)`` for :func:`sequential_CSS`."""
    if pyf.is_pyomo_solver_available("gurobi"):
        return "gurobi", "gurobi"
    print("Gurobi is not available; falling back to glpk for MIP and CSS.")
    return "glpk", "glpk"


def lopf_solver():
    if pyf.is_pyomo_solver_available("gurobi"):
        return "gurobi"
    return "glpk"
