# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/tep/."""

from pyflow_tests.run_doc_examples import run_doc_examples

FOLDER = "tep"


def test_docs_tep():
    run_test()


def run_test():
    try:
        import pyomo  # noqa: F401
    except ImportError:
        print("pyomo is not installed...")
        return

    import pyflow_acdc as pyf

    if not pyf.is_pyomo_solver_available("bonmin"):
        print("Skipped: Bonmin solver not available")
        return

    run_doc_examples(FOLDER)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
