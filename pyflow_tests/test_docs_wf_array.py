# -*- coding: utf-8 -*-
"""Run all examples in doc_examples/wf_array/."""

from pyflow_tests.run_doc_examples import run_doc_examples

FOLDER = "wf_array"


def test_docs_wf_array():
    run_test()


def run_test():
    try:
        import ortools  # noqa: F401
    except ImportError:
        print("ortools is not installed...")
        return
    run_doc_examples(FOLDER)
    print(f"✓ {FOLDER} doc examples passed")


if __name__ == "__main__":
    run_test()
