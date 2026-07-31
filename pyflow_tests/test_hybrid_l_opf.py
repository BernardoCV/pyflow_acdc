import pyflow_acdc as pyf
from pyflow_tests._test_solver_deps import lopf_solver, require_pyomo


def test_case39_acdc_lopf_build_only():
    """Hybrid linear OPF builds and exports without solving."""
    require_pyomo()
    grid, _ = pyf.cases["case39_acdc"]()
    model, model_res, timing_info, solver_stats = pyf.optimal_l_pf(
        grid, ObjRule={"Energy_cost": 1}, build_only=True
    )
    assert grid.DCmode and grid.ACmode
    assert model is not None
    assert hasattr(model, "V_DC")
    assert hasattr(model, "PDC_from")
    assert hasattr(model, "P_conv_s_AC")
    assert hasattr(model, "Conv_Ps_PDC_constraint")
    assert grid.OPF_run
    assert len(grid.V_DC) == grid.nn_DC
    assert len(grid.Converters_ACDC) == 10


def test_case39_ac_lopf_regression_build_only():
    """AC-only linear OPF still builds after hybrid expansion."""
    require_pyomo()
    grid, _ = pyf.cases["case39"]()
    model, _, _, _ = pyf.optimal_l_pf(
        grid, ObjRule={"Energy_cost": 1}, build_only=True
    )
    assert not grid.DCmode
    assert hasattr(model, "theta_AC")
    assert not hasattr(model, "V_DC")


def test_case39_acdc_lopf_optional_solve():
    """Optional LP solve on hybrid grid when a linear solver is available."""
    require_pyomo()
    solver = lopf_solver()
    grid, _ = pyf.cases["case39_acdc"]()
    model, model_res, timing_info, solver_stats = pyf.optimal_l_pf(
        grid, ObjRule={"Energy_cost": 1}, solver=solver
    )
    assert grid.OPF_run
    assert model.obj is not None
