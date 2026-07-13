# -*- coding: utf-8 -*-
"""Parser tests for :mod:`pyflow_acdc.pyomo_model_solve` using committed log fixtures."""

from pathlib import Path

import pytest

from pyflow_acdc.pyomo_model_solve import (
    _parse_bonmin_log,
    _parse_highs_log,
    _parse_ipopt_log,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "solver_logs"


@pytest.mark.parametrize(
    "fixture_name,parser,kwargs,expect_feasible,expect_all,expect_bounds",
    [
        (
            "ipopt_optimal.log",
            _parse_ipopt_log,
            {},
            1,
            None,
            None,
        ),
        (
            "highs_mip_sample.log",
            _parse_highs_log,
            {},
            2,
            2,
            2,
        ),
        (
            "bonmin_bb_sample.log",
            _parse_bonmin_log,
            {"bonmin_algorithm": "B-BB"},
            1,
            2,
            1,
        ),
    ],
)
def test_parse_solver_log_fixture(
    fixture_name, parser, kwargs, expect_feasible, expect_all, expect_bounds
):
    log_path = FIXTURES / fixture_name
    assert log_path.is_file(), f"Missing fixture: {log_path}"
    result = parser(str(log_path), **kwargs)
    if parser is _parse_ipopt_log:
        events = result
        assert len(events) >= expect_feasible
        assert events[-1][2] is True
        return
    feasible, all_solutions, bound_solutions = result
    assert len(feasible) >= expect_feasible
    if expect_all is not None:
        assert len(all_solutions) >= expect_all
    if expect_bounds is not None:
        assert len(bound_solutions) >= expect_bounds


def test_parse_bonmin_log_missing_file_returns_empty():
    feasible, all_solutions, bound_solutions = _parse_bonmin_log(
        str(FIXTURES / "does_not_exist.log")
    )
    assert feasible == []
    assert all_solutions == []
    assert bound_solutions == []


def run_test():
    pytest.main([__file__, "-q"])


if __name__ == "__main__":
    run_test()
