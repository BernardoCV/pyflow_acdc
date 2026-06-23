"""Shared dependency and solver checks for script-style case tests."""

import os

import pytest

import pyflow_acdc as pyf


def _missing_for_run_test(import_fn, skip_message):
    try:
        import_fn()
    except Exception:
        print(skip_message)
        return True
    return False


def pyomo_missing_for_run_test():
    """Return True when pyomo is absent (``run_test`` should return early)."""
    return _missing_for_run_test(
        lambda: __import__("pyomo"),
        "Skipped: pyomo is not installed",
    )


def require_pyomo():
    pytest.importorskip("pyomo")


def folium_missing_for_run_test():
    return _missing_for_run_test(
        lambda: __import__("folium"),
        "Skipped: folium is not installed",
    )


def require_folium():
    pytest.importorskip("folium")


def dash_missing_for_run_test():
    return _missing_for_run_test(
        lambda: __import__("dash"),
        "Skipped: dash is not installed",
    )


def require_dash():
    pytest.importorskip("dash")


def dill_missing_for_run_test():
    return _missing_for_run_test(
        lambda: __import__("dill"),
        "Skipped: dill is not installed",
    )


def require_dill():
    pytest.importorskip("dill")


def ortools_missing_for_run_test():
    try:
        from ortools.sat.python import cp_model  # noqa: F401
    except Exception:
        print("Skipped: OR-Tools is not installed")
        return True
    return False


def require_ortools():
    pytest.importorskip("ortools")


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
