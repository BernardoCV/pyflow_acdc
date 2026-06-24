# -*- coding: utf-8 -*-
"""Sequential CSS on alpha_ventus (Gurobi MIP with GLPK fallback)."""
import time

import pyflow_acdc as pyf
import pyomo.environ as pyo

from pyflow_tests._test_solver_deps import (
    dill_missing_for_run_test,
    mip_solvers,
    pyomo_mip_css_solvers_missing_for_run_test,
    pyomo_missing_for_run_test,
    require_dill,
    require_pyomo,
    require_pyomo_mip_css_solvers,
    tep_solver,
)

ARRAY_CASE = "alpha_ventus"
CT = 3
TL = 60
NL = False
TEE = False
FS = False
FLH = 8760
WACC = 0.02
L_OPEX = True


def run_case(mip_solver=None, css_l_solver=None):
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
        NL=NL,
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


def test_sequential_array_alpha_ventus():
    require_pyomo()
    require_dill()
    require_pyomo_mip_css_solvers()
    mip_solver, css_solver = mip_solvers()
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
    ) = run_case(mip_solver, css_solver)
    print(
        f"{ARRAY_CASE}- iterations {i}, total time {total_time}, edges {edges}, "
        f"subsations {substations}, turbines {turbines}, obj_value {obj_value}, "
        f"path_time {path_time}, css_time {css_time}, crossing {crossing}, "
        f"cable_length {cable_length}"
    )


def run_test():
    if dill_missing_for_run_test():
        return
    if pyomo_missing_for_run_test():
        return
    if pyomo_mip_css_solvers_missing_for_run_test():
        return
    mip_solver, css_solver = mip_solvers()
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
    ) = run_case(mip_solver, css_solver)
    print(
        f"{ARRAY_CASE}- iterations {i}, total time {total_time}, edges {edges}, "
        f"subsations {substations}, turbines {turbines}, obj_value {obj_value}, "
        f"path_time {path_time}, css_time {css_time}, crossing {crossing}, "
        f"cable_length {cable_length}"
    )


if __name__ == "__main__":
    run_test()
