# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/ts/."""

from pyflow_tests.run_doc_examples import run_doc_examples

FOLDER = "ts"


def test_docs_ts():
    run_test()


def run_test():
    try:
        import pyomo  # noqa: F401
    except ImportError:
        print("pyomo is not installed...")
        return
    run_doc_examples(FOLDER)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
