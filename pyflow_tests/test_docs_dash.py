# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/dash/."""

from pyflow_tests.run_doc_examples import run_doc_examples

FOLDER = "dash"


def test_docs_dash():
    run_test()


def run_test():
    try:
        import dash  # noqa: F401
    except ImportError:
        print("dash is not installed...")
        return
    run_doc_examples(FOLDER)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
