# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/plotting/."""


from pyflow_tests.run_doc_examples import run_doc_examples
from pyflow_tests._test_solver_deps import (
    ipopt_missing_for_run_test,
    require_ipopt,
)

FOLDER = "plotting"


def test_docs_plotting():
    require_ipopt()
    run_test()


def run_test():
    if ipopt_missing_for_run_test():
        return
    run_doc_examples(FOLDER, skip_on_import_error=True)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
