"""Shared pytest fixtures for the pyflow_acdc test suite.

Auto-discovered by pytest for every test in this directory.
"""
import pytest

import pyflow_acdc as pyf


@pytest.fixture(autouse=True)
def reset_pyflow_state():
    """Reset all element class counters/registries before each test.

    The element classes (``Node_AC``, ``Line_AC``, ``AC_DC_converter``,
    ``Price_Zone``, ``TimeSeries``, ...) keep class-level numbering and name
    registries. Without a reset these leak between tests running in the same
    pytest process, so we restore a clean slate before every test.
    """
    pyf.initialize_pyflowacdc()
    yield


@pytest.fixture
def quick_fake_solve():
    """Return the fake-solve context manager for solver-free quick runs.

    Usage::

        def test_x(quick_fake_solve):
            with quick_fake_solve(opf=True, tep=True):
                ...
    """
    from pyflow_tests._quick_fake_solve import quick_fake_solve_context

    return quick_fake_solve_context
