import pyflow_acdc as pyf
from pyflow_acdc.constants import ConverterOpfFxType
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
    assert not hasattr(model, "Q_conv_s_AC")
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


def test_case39_acdc_lopf_fx_pdc_build_only():
    """``fx_conv`` PDC fix is applied on the linear hybrid model."""
    require_pyomo()
    grid, _ = pyf.cases["case39_acdc"]()
    conv = next(c for c in grid.Converters_ACDC if c.np_conv > 0)
    conv.OPF_fx = True
    conv.OPF_fx_type = ConverterOpfFxType.PDC
    conv.P_DC = -0.1
    model, _, _, _ = pyf.optimal_l_pf(
        grid, ObjRule={"Energy_cost": 1}, build_only=True
    )
    assert hasattr(model, "Conv_fx_pdc")
    assert hasattr(model, "Conv_fx_pac")
    assert hasattr(model, "Conv_fx_qac")
    # Constraint is indexed; Skip entries are inactive / None body.
    assert conv.ConvNumber in model.Conv_fx_pdc


def test_case39_acdc_lopf_fx_pq_build_only():
    """``fx_conv`` PQ fixes P; Q fix skipped on linear (no ``Q_conv_s_AC``)."""
    require_pyomo()
    grid, _ = pyf.cases["case39_acdc"]()
    conv = next(c for c in grid.Converters_ACDC if c.np_conv > 0)
    conv.OPF_fx = True
    conv.OPF_fx_type = ConverterOpfFxType.PQ
    conv.P_AC = 0.05
    conv.Q_AC = 0.0
    model, _, _, _ = pyf.optimal_l_pf(
        grid, ObjRule={"Energy_cost": 1}, build_only=True
    )
    assert hasattr(model, "Conv_fx_pac")
    assert hasattr(model, "Conv_fx_qac")
    assert conv.ConvNumber in model.Conv_fx_pac
    assert not hasattr(model, "Q_conv_s_AC")
