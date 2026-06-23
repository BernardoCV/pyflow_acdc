import pyflow_acdc as pyf

from pyflow_tests._test_solver_deps import pyomo_missing_for_run_test, require_pyomo, tep_solver


def case24_TEP():
    grid, res = pyf.cases['case24_TEP']()
    model, model_results, timing_info, solver_stats = pyf.transmission_expansion(
        grid, NPV=True, solver=tep_solver()
    )
    res.tep_n()
    res.tep_norm()
    print(timing_info)
    model.obj.display()


def test_case24_tep():
    require_pyomo()
    case24_TEP()


def run_test():
    """Test case24 transmission expansion planning."""
    if pyomo_missing_for_run_test():
        return
    case24_TEP()


if __name__ == "__main__":
    run_test()
