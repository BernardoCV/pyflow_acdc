# -*- coding: utf-8 -*-
"""Sequential CSS on westermost_rough using OR-Tools for MIP/CSS-L."""
import time

import pyflow_acdc as pyf
import pyomo.environ as pyo
import pytest

from pyflow_acdc.windfarm_loader import load_case_grid_and_geo
from pyflow_tests._test_solver_deps import (
    ortools_missing_for_run_test,
    pyomo_missing_for_run_test,
    require_ortools,
    require_pyomo,
    tep_solver,
)

CASES = {'westermost_rough'}

CT = 3
MIP_SOLVER = 'ortools'
TL = 300
NL = False
TEE = False
FS = False
OBJ = {'Energy_cost': 1}
FLH = 8760
WACC = 0.02


def run_case(case, mip_solver='ortools'):
    start_time = time.perf_counter()

    grid, res = load_case_grid_and_geo(case)
    grid.cab_types_allowed = CT

    model, summary_results, timing_info, solver_stats, best_i = pyf.sequential_CSS(
        grid,
        NPV=True,
        n_years=25,
        Hy=FLH,
        discount_rate=WACC,
        ObjRule=OBJ,
        MIP_solver=mip_solver,
        CSS_L_solver='ortools',
        CSS_NL_solver=tep_solver(),
        max_iter=None,
        time_limit=TL,
        NL=NL,
        tee=TEE,
        fs=FS,
    )

    i = len(summary_results['iteration'])
    obj_value = pyo.value(model[1].obj)
    cable_length = summary_results['cable_length'][best_i]
    path_time = timing_info['Paths']
    css_time = timing_info['CSS']
    crossing = len(getattr(grid, 'crossing_groups', []))
    edges = len(getattr(grid, 'lines_AC_ct', []))
    turbines = len(getattr(grid, 'RenSources', []))
    substations = sum(
        1 for n in getattr(grid, 'nodes_AC', []) if getattr(n, 'type', None) == 'Slack'
    )
    total_time = time.perf_counter() - start_time

    return (
        i, total_time, edges, substations, turbines, obj_value,
        path_time, css_time, summary_results, crossing, cable_length,
    )


@pytest.mark.slow
def test_sequential_array_ortools_westermost_rough():
    require_ortools()
    require_pyomo()
    case = 'westermost_rough'
    (
        i, total_time, edges, substations, turbines, obj_value,
        path_time, css_time, summary_results, crossing, cable_length,
    ) = run_case(case, MIP_SOLVER)
    print(
        f'{case}- iterations {i}, total time {total_time}, edges {edges}, '
        f'subsations {substations}, turbines {turbines}, obj_value {obj_value}, '
        f'path_time {path_time}, css_time {css_time}, crossing {crossing}, '
        f'cable_length {cable_length}'
    )


def run_test():
    if ortools_missing_for_run_test():
        return
    if pyomo_missing_for_run_test():
        return
    for case in CASES:
        (
            i, total_time, edges, substations, turbines, obj_value,
            path_time, css_time, summary_results, crossing, cable_length,
        ) = run_case(case, MIP_SOLVER)
        print(
            f'{case}- iterations {i}, total time {total_time}, edges {edges}, '
            f'subsations {substations}, turbines {turbines}, obj_value {obj_value}, '
            f'path_time {path_time}, css_time {css_time}, crossing {crossing}, '
            f'cable_length {cable_length}'
        )


if __name__ == "__main__":
    run_test()
