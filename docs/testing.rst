Testing
=======

Install the test extra from a local checkout::

    pip install -e ".[tests]"

This adds ``pytest`` and ``pytest-cov``. Some cases also need optional extras
(``OPF``, ``ORTOOLS_ARRAY``, ``Dash``, and so on); missing dependencies are
reported as **Skipped**, not failed.

Test runner
-----------

The project test suite is driven by ``pyflow_tests/run_tests.py``, exposed as a
console script after editable install::

    pyflow-acdc-test                 # full suite
    pyflow-acdc-test --quick         # fast subset (basic functionality)
    pyflow-acdc-test --docs          # documentation literalinclude examples
    pyflow-acdc-test --opf           # solver-dependent OPF tests
    pyflow-acdc-test --tep           # transmission-expansion tests
    pyflow-acdc-test --show-output   # stream each case's output

Equivalent module invocation::

    python -m pyflow_tests.run_tests --quick

Case lists (``DOCS_CASES``, ``QUICK_CASES``, ``OPF_CASES``, ``TEP_CASES``,
``ALL_CASES``) live in ``pyflow_tests/test_constants.py``.

Before opening a pull request, run at least ``--quick``. If you changed documentation
examples, run ``--docs``. If you changed OPF or TEP code, also run ``--opf`` or ``--tep``
as appropriate.

Pytest
------

Several modules under ``pyflow_tests/`` are collected by ``pytest`` directly
(for example ``test_example_grids_smoke.py``, ``test_model_build_only.py``).
From the repository root::

    pytest pyflow_tests/ -q

Coverage::

    pytest pyflow_tests/ --cov=pyflow_acdc --cov-report=term-missing -q

CI uploads coverage to Codecov on each push/PR to ``main`` (see the ``coverage``
job in ``.github/workflows/pr-tests.yml``; requires the ``CODECOV_TOKEN`` repository
secret).

The shipped snapshot lives in ``TEST_COVERAGE.md`` at the repository root (overall %,
per-module miss counts, missing lines). Pull requests that change tests must update
that file and the **Test coverage** badge in ``README.md`` (see ``CONTRIBUTING.md``).
The Codecov badge in ``README.md`` updates automatically from CI uploads.

TEP tests use Ipopt by default (via ``tep_solver()``). For a full MINLP solve on
one file, opt in to Bonmin::

    PYFLOW_TEP_SOLVER=bonmin pytest pyflow_tests/test_case24_TEP.py -q

Override with any installed Pyomo solver::

    PYFLOW_TEP_SOLVER=ipopt pytest pyflow_tests/ -q

.. note::

   Ipopt is NLP-only. Tests that run a real expansion solve with binary
   investment variables (for example ``test_case24_TEP``) may behave differently or
   fail without Bonmin. Use ``PYFLOW_TEP_SOLVER=bonmin`` when you want a proper
   MINLP solve on a single test. Build-only tests (MP+MS,
   ``test_model_build_only``, and similar) are unaffected.

To run a single file::

    pytest pyflow_tests/test_plot.py -q

Documentation examples
----------------------

Examples live in ``pyflow_tests/doc_examples/``, grouped in subfolders that mirror
the docs. Most folders have a matching ``test_docs_<folder>.py`` that runs every
``.py`` file inside it; edit the ``doc_examples`` file only and docs/tests stay in sync.

Included in ``--docs`` (``DOCS_CASES``):

* ``index/``, ``usage/``, ``tep/`` (static and MS TEP), ``tep_mp/`` (MP TEP and
  MP+MS; **ipopt** full solve, then ``res.all()`` in each example)
* ``csv_import/``, ``modelling_ac/``, ``modelling_dc/``, ``results/``, ``ts/``,
  ``clustering/``, ``plotting/``, ``tep_pymoo/``, ``wf_array/``, ``dash/``

``test_docs_tep.py`` runs ``tep/`` (static :func:`~pyflow_acdc.transmission_expansion`
and MS :func:`~pyflow_acdc.multi_scenario_TEP` on **ipopt**).
``test_docs_tep_mp.py`` runs ``tep_mp/`` (MP and MP+MS examples).

Not run by ``--docs`` (no runner, or long / optional solvers):

* ``L_opf/``, ``windfarm_loader/`` — no ``test_docs_*.py`` yet

Adding a case
-------------

New scripted cases should expose a top-level ``run_test()`` function and be listed
in ``ALL_CASES`` (and ``DOCS_CASES``, ``QUICK_CASES``, ``OPF_CASES``, or ``TEP_CASES``
when relevant) in ``pyflow_tests/test_constants.py``.

For a new doc example folder, add ``doc_examples/<folder>/`` plus
``test_docs_<folder>.py`` calling ``run_doc_examples("<folder>")``, then append
``test_docs_<folder>.py`` to ``DOCS_CASES``.
