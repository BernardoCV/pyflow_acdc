# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/tep_pymoo/."""

from pyflow_tests.run_doc_examples import run_doc_examples

FOLDER = "tep_pymoo"


def test_docs_tep_pymoo():
    run_test()


def run_test():
    try:
        import pymoo  # noqa: F401
        import pyomo  # noqa: F401
    except ImportError:
        print("pymoo/pyomo is not installed...")
        return
    run_doc_examples(FOLDER)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
