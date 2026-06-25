# Contributing to pyflow-acdc

Thanks for your interest in contributing! This project is experimental and
under active development — issue reports and pull requests are very welcome.

## Development setup

```bash
git clone https://github.com/CITCEA-UPC/pyflow_acdc.git
cd pyflow_acdc
pip install -e .
```

Requires **Python 3.10+**. Install the optional extras you need (see the
README for the full list), e.g.:

```bash
pip install -e ".[OPF]"            # pyomo for optimal power flow
pip install -e ".[LINEAR_ARRAY]"  # ortools + highspy for array MIP/CSS
pip install -e ".[tests]"          # pytest + pytest-cov
pip install -e ".[All]"        # everything except gurobipy (needs a license)
```

Solvers such as `ipopt` / `bonmin` are not on PyPI and must come from
conda-forge — see the README "Installation" section.

## Running the tests

The test runner is exposed as a console script and as a module:

```bash
pyflow-acdc-test                 # full suite
pyflow-acdc-test --quick         # fast subset (basic functionality)
pyflow-acdc-test --opf           # solver-dependent OPF tests
pyflow-acdc-test --tep           # transmission-expansion tests
pyflow-acdc-test --show-output   # stream each case's output

# equivalently:
python -m pyflow_tests.run_tests --quick
```

Tests that require an unavailable optional dependency are reported as
**Skipped**, not failed. Please make sure `--quick` passes before opening a
PR, and run the relevant `--opf` / `--tep` subset if you touched those areas.

When adding a new test case, add the file to the appropriate list
(`ALL_CASES`, `QUICK_CASES`, `OPF_CASES`, `TEP_CASES`, `DOCS_CASES`) in
`pyflow_tests/test_constants.py`, and expose a top-level `run_test()` function in
the case module.

## Test coverage

CI uploads pytest coverage to [Codecov](https://codecov.io/gh/CITCEA-UPC/pyflow_acdc)
on each push/PR to `main` (`.github/workflows/pr-tests.yml`, `coverage` job).
The repository must have a `CODECOV_TOKEN` Actions secret.

The package also maintains a coverage snapshot in [`TEST_COVERAGE.md`](TEST_COVERAGE.md)
(overall %, per-module miss counts, missing lines, and public API functions
without dedicated tests).

**Pull requests that add, remove, or materially change tests must update
`TEST_COVERAGE.md`** — at minimum the header stats and any module sections
affected by your change. Update the **Test coverage** badge percentage in
[`README.md`](README.md) to match the overall % in the snapshot header.
Regenerate with:

```bash
pytest pyflow_tests/ --cov=pyflow_acdc --cov-report=term-missing
```

Copy updated totals and `Missing` line ranges from the report into the matching
sections. Update the **Functions not covered** table and any **function / block**
rows whose status changed (mark **Tested**, **Partial**, or **Not tested**).
If you add tests for previously untested public API, remove or adjust the entry
in **Functions not covered**.

## Coding conventions

- **New code uses `snake_case`** for functions and variables. Several existing
  public names are mixed-case for backward compatibility; do not rename them
  without a deprecation alias.
- **Centralise closed-vocabulary strings** in `pyflow_acdc/constants.py` as
  `(str, Enum)` members rather than scattering string literals. Use the enum
  member in comparisons (typo-safe) and `.value` for stored attributes / dict
  keys (preserves serialization and display).
- **Prefer fail-fast over silent fallbacks**: raise clear errors on invalid
  input instead of masking problems.
- **Keep changes simple and local**; avoid introducing new helper/abstraction
  layers unless reuse clearly justifies it.
- **Optional dependencies** must be guarded with `try/except ImportError` and a
  feature flag, matching the pattern in `__init__.py`.
- **Public API**: each module defines `__all__`. If you add a public symbol,
  add it to the module's `__all__` (and to `__init__.py` if it should be part
  of the top-level `pyflow_acdc` namespace).
- **Docstrings** use the NumPy style (the docs build with
  `sphinx.ext.napoleon`): one-line purpose, then `Parameters` / `Returns` /
  `Raises` / side-effects as relevant.

See `docs/architecture.md` for module responsibilities and the dependency
layering.

## Documentation

```bash
cd docs
pip install -r requirements.txt
make html      # or make.bat html on Windows
```

Output is written to `docs/_build/html/index.html`.

## Pull requests

1. Create a feature branch off `main`.
2. Keep PRs focused; describe the motivation ("why") in the description.
3. Ensure tests pass and add/adjust tests for behavioural changes.
4. Update [`TEST_COVERAGE.md`](TEST_COVERAGE.md) and the README coverage badge when tests change (see above).
5. To merge into `main`, contact the repository owner.
