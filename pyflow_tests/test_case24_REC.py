import pyflow_acdc as pyf

from pyflow_tests._test_solver_deps import pyomo_missing_for_run_test, require_pyomo, tep_solver


def case24_REC():
    grid, res = pyf.cases['case24_REC']()
    model, model_results, timing_info, solver_stats = pyf.transmission_expansion(
        grid, NPV=True, solver=tep_solver()
    )
    res.tep_n()
    res.tep_norm()
    print(timing_info)
    model.obj.display()


def test_case24_rec():
    require_pyomo()
    case24_REC()


def run_test():
    """Test case24 renewable energy curtailment."""
    if pyomo_missing_for_run_test():
        return
    case24_REC()


if __name__ == "__main__":
    run_test()
