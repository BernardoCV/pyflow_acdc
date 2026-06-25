# -*- coding: utf-8 -*-
"""Sequential CSS on alpha_ventus (linear and PF loss post-processing)."""
import time

import pyflow_acdc as pyf
import pyomo.environ as pyo

from pyflow_acdc.constants import CssMode
from pyflow_tests._test_solver_deps import (
    mip_solvers,
    pyomo_mip_css_solvers_missing_for_run_test,
    pyomo_missing_for_run_test,
    require_pyomo,
    require_pyomo_mip_css_solvers,
    tep_solver,
)

ARRAY_CASE = "alpha_ventus"
CT = 3
TL = 60
TEE = False
FS = False
FLH = 8760
WACC = 0.02
L_OPEX = True


def _format_result(nl_label, result):
    (
        i,
        total_time,
        edges,
        substations,
        turbines,
        obj_value,
        path_time,
        css_time,
        crossing,
        cable_length,
    ) = result
    return (
        f"{ARRAY_CASE} [{nl_label}] - iterations {i}, total time {total_time}, "
        f"edges {edges}, subsations {substations}, turbines {turbines}, "
        f"obj_value {obj_value}, path_time {path_time}, css_time {css_time}, "
        f"crossing {crossing}, cable_length {cable_length}"
    )


def run_case(mip_solver=None, css_l_solver=None, *, nl=False):
    if mip_solver is None or css_l_solver is None:
        mip_solver, css_l_solver = mip_solvers()

    start_time = time.perf_counter()
    grid, res = pyf.cases[ARRAY_CASE](cab_types_allowed=CT)

    model, summary_results, timing_info, solver_stats, best_i = pyf.sequential_CSS(
        grid,
        NPV=True,
        n_years=25,
        Hy=FLH,
        discount_rate=WACC,
        L_OPEX=L_OPEX,
        MIP_solver=mip_solver,
        CSS_L_solver=css_l_solver,
        CSS_NL_solver=tep_solver(),
        max_iter=None,
        time_limit=TL,
        NL=nl,
        tee=TEE,
        fs=FS,
    )
    i = len(summary_results["iteration"])
    obj_value = pyo.value(model[1].obj)
    cable_length = summary_results["cable_length"][best_i]
    path_time = timing_info["Paths"]
    css_time = timing_info["CSS"]
    crossing = len(getattr(grid, "crossing_groups", []))
    edges = len(getattr(grid, "lines_AC_ct", []))
    turbines = len(getattr(grid, "RenSources", []))
    substations = sum(
        1 for n in getattr(grid, "nodes_AC", []) if getattr(n, "type", None) == "Slack"
    )
    total_time = time.perf_counter() - start_time

    return (
        i,
        total_time,
        edges,
        substations,
        turbines,
        obj_value,
        path_time,
        css_time,
        crossing,
        cable_length,
    )


def test_sequential_array_alpha_ventus_linear():
    require_pyomo()
    require_pyomo_mip_css_solvers()
    mip_solver, css_solver = mip_solvers()
    result = run_case(mip_solver, css_solver, nl=False)
    print(_format_result("linear", result))


def test_sequential_array_alpha_ventus_pf():
    require_pyomo()
    require_pyomo_mip_css_solvers()
    mip_solver, css_solver = mip_solvers()
    result = run_case(mip_solver, css_solver, nl=CssMode.PF)
    print(_format_result("PF", result))


def run_test():
    if pyomo_missing_for_run_test():
        return
    if pyomo_mip_css_solvers_missing_for_run_test():
        return
    mip_solver, css_solver = mip_solvers()
    for nl, label in ((False, "linear"), (CssMode.PF, "PF")):
        result = run_case(mip_solver, css_solver, nl=nl)
        print(_format_result(label, result))

if __name__ == "__main__":
    run_test()
