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
from pyflow_acdc.constants import ORTOOLS_LINEAR_SOLVERS

PYOMO_MIP_CSS_SOLVERS = ("gurobi", "glpk")


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


def pyomo_mip_css_solver_available():
    """Return True when a Pyomo MIP/CSS-L solver (Gurobi or GLPK) is available."""
    if not _pyomo_available():
        return False
    return any(pyf.is_pyomo_solver_available(name) for name in PYOMO_MIP_CSS_SOLVERS)


def require_pyomo_mip_css_solvers():
    """Skip unless Pyomo and at least one MIP/CSS-L solver are available."""
    require_pyomo()
    if not pyomo_mip_css_solver_available():
        pytest.skip(
            "no Pyomo MIP/CSS-L solver available (need gurobi or glpk)"
        )


def pyomo_mip_css_solvers_missing_for_run_test():
    """Return True when Pyomo MIP/CSS-L solvers are absent (for ``run_test``)."""
    if not _pyomo_available():
        return False
    if pyomo_mip_css_solver_available():
        return False
    print("No Pyomo MIP/CSS-L solver available (need gurobi or glpk)")
    return True


def _ortools_lp_solver_available():
    if not _ortools_available():
        return False
    try:
        from ortools.linear_solver import pywraplp
    except Exception:
        return False
    for name in ORTOOLS_LINEAR_SOLVERS:
        if pywraplp.Solver.CreateSolver(name) is not None:
            return True
    return False


def ortools_array_stack_available():
    """Return True when OR-Tools CP-SAT (MIP) and a linear CSS-L solver are available."""
    return _ortools_available() and _ortools_lp_solver_available()


def require_ortools_array_stack():
    """Skip unless the full OR-Tools array stack (MIP + CSS-L) can solve."""
    require_ortools()
    if not _ortools_lp_solver_available():
        pytest.skip("no OR-Tools linear CSS-L solver available")


def ortools_array_stack_missing_for_run_test():
    """Return True when the OR-Tools array solve stack is absent (for ``run_test``)."""
    if not _ortools_available():
        return False
    if ortools_array_stack_available():
        return False
    print("No OR-Tools linear CSS-L solver available")
    return True


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
    if pyf.is_pyomo_solver_available("glpk"):
        print("Gurobi is not available; falling back to glpk for MIP and CSS.")
        return "glpk", "glpk"
    raise RuntimeError(
        "no Pyomo MIP/CSS-L solver available (need gurobi or glpk)"
    )


def lopf_solver():
    if pyf.is_pyomo_solver_available("gurobi"):
        return "gurobi"
    return "glpk"
