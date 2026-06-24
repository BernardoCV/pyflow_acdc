# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/ts/."""

from pyflow_tests.run_doc_examples import run_doc_examples
from pyflow_tests._test_solver_deps import (
    ipopt_missing_for_run_test,
    pyomo_missing_for_run_test,
    require_ipopt,
    require_pyomo,
)

FOLDER = "ts"


def test_docs_ts():
    require_pyomo()
    require_ipopt()
    run_test()


def run_test():
    if pyomo_missing_for_run_test():
        return
    if ipopt_missing_for_run_test():
        return
    run_doc_examples(FOLDER)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
