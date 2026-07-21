# -*- coding: utf-8 -*-
"""Coupled window NL OPF with BESS (frame blocks + parent SoC links)."""

import time

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from .ACDC_OPF import (
    calculate_objective,
    fx_conv,
    obj_w_rule,
    opf_obj,
    translate_pyf_opf,
)
from .ACDC_OPF_NL_model import export_acdc_nl_model_to_pyflow_acdc, opf_create_nl_model_acdc
from .constants import AcDcSide
from .grid_analysis import analyse_grid
from .pyomo_model_solve import build_only_solver_stats, pyomo_model_solve
from .Time_series import _modify_parameters, update_grid_data

__all__ = ['window_nl_opf']


def _sum_frame_objectives(model, frames):
    total = 0
    for t in frames:
        total += model.frame_model[t].obj.expr
        model.frame_model[t].obj.deactivate()
    return total


def _soc_delta_expr(st, block, s, ac=True):
    scale = st.dt_hours * st.S_base / st.E_max
    if ac:
        return scale * (
            st.eta_charge * block.P_storage_charge[s]
            - block.P_storage_discharge[s] / st.eta_discharge
        )
    return scale * (
        st.eta_charge * block.P_storage_charge_DC[s]
        - block.P_storage_discharge_DC[s] / st.eta_discharge
    )


def window_soc_constraints(model, grid, storage_info, frames):
    """Parent-level SoC chain across ``frame_model`` blocks."""
    ordered = list(frames)
    _, _, storage_ac_by_number, storage_dc_by_number = storage_info

    model.window_soc_constraint = pyo.ConstraintList()
    # Future (P4-1b): may link actual energy [MWh] or SoC×E_max_eff for degradation.

    for i, t in enumerate(ordered):
        block = model.frame_model[t]
        for s, st in storage_ac_by_number.items():
            delta = _soc_delta_expr(st, block, s, ac=True)
            if i == 0:
                model.window_soc_constraint.add(block.SoC[s] == st.soc_initial + delta)
            else:
                t_prev = ordered[i - 1]
                model.window_soc_constraint.add(
                    block.SoC[s] == model.frame_model[t_prev].SoC[s] + delta)

        for s, st in storage_dc_by_number.items():
            delta = _soc_delta_expr(st, block, s, ac=False)
            if i == 0:
                model.window_soc_constraint.add(block.SoC_DC[s] == st.soc_initial + delta)
            else:
                t_prev = ordered[i - 1]
                model.window_soc_constraint.add(
                    block.SoC_DC[s] == model.frame_model[t_prev].SoC_DC[s] + delta)

    if ordered:
        t_last = ordered[-1]
        for s, st in storage_ac_by_number.items():
            if st.soc_final is not None:
                model.window_soc_constraint.add(
                    model.frame_model[t_last].SoC[s] == st.soc_final)
        for s, st in storage_dc_by_number.items():
            if st.soc_final is not None:
                model.window_soc_constraint.add(
                    model.frame_model[t_last].SoC_DC[s] == st.soc_final)


def _create_frame_blocks(
    model,
    grid,
    frames,
    pv_set,
    price_zones,
    limit_flow_rate,
    weights_def,
    only_gen,
):
    base_model = pyo.ConcreteModel()
    opf_create_nl_model_acdc(
        base_model,
        grid,
        PV_set=pv_set,
        Price_Zones=price_zones,
        limit_flow_rate=limit_flow_rate,
        window_block=True,
    )

    opf_data = translate_pyf_opf(grid, Price_Zones=price_zones)
    storage_info = opf_data['storage_info']

    for t in frames:
        base_copy = base_model.clone()
        model.frame_model[t].transfer_attributes_from(base_copy)
        for ts in grid.Time_series:
            update_grid_data(grid, ts, t, price_zone_restrictions=price_zones)
        _modify_parameters(grid, model.frame_model[t], price_zones)
        obj_rule = opf_obj(model.frame_model[t], grid, weights_def, only_gen)
        model.frame_model[t].obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
        if grid.nn_DC != 0 and any(conv.OPF_fx for conv in grid.Converters_ACDC):
            fx_conv(model.frame_model[t], grid)

    return storage_info


def export_window_opf_results(model, grid, frames):
    """Build per-frame storage trajectories from solved frame blocks."""
    rows_soc = []
    rows_pc = []
    rows_pd = []
    rows_q = []

    for t in frames:
        block = model.frame_model[t]
        row_soc = {'frame': t}
        row_pc = {'frame': t}
        row_pd = {'frame': t}
        row_q = {'frame': t}
        for storage in grid.storage_elements:
            name = storage.name
            if storage.connected == AcDcSide.AC:
                s = storage.storageNumber
                row_soc[name] = np.float64(pyo.value(block.SoC[s]))
                row_pc[name] = np.float64(pyo.value(block.P_storage_charge[s])) * grid.S_base
                row_pd[name] = np.float64(pyo.value(block.P_storage_discharge[s])) * grid.S_base
                row_q[name] = np.float64(pyo.value(block.Q_storage[s])) * grid.S_base
            else:
                s = storage.storageNumber_DC
                row_soc[name] = np.float64(pyo.value(block.SoC_DC[s]))
                row_pc[name] = np.float64(pyo.value(block.P_storage_charge_DC[s])) * grid.S_base
                row_pd[name] = np.float64(pyo.value(block.P_storage_discharge_DC[s])) * grid.S_base
                row_q[name] = np.nan
        rows_soc.append(row_soc)
        rows_pc.append(row_pc)
        rows_pd.append(row_pd)
        rows_q.append(row_q)

    summary_rows = []
    for storage in grid.storage_elements:
        pc_col = [row[storage.name] for row in rows_pc]
        pd_col = [row[storage.name] for row in rows_pd]
        energy_in = sum(pc_col) * storage.dt_hours
        energy_out = sum(pd_col) * storage.dt_hours
        rt_eff = energy_out / energy_in if energy_in > 0 else np.nan
        summary_rows.append({
            'Name': storage.name,
            'Side': storage.connected.value,
            'Energy charged (MWh)': energy_in,
            'Energy discharged (MWh)': energy_out,
            'Round-trip efficiency': rt_eff,
        })

    return {
        'storage_soc': pd.DataFrame(rows_soc),
        'storage_P_charge': pd.DataFrame(rows_pc),
        'storage_P_discharge': pd.DataFrame(rows_pd),
        'storage_Q': pd.DataFrame(rows_q),
        'storage_summary': pd.DataFrame(summary_rows),
        'total_objective': np.float64(pyo.value(model.obj)),
    }


def window_nl_opf(
    grid,
    start=0,
    end=23,
    ObjRule=None,
    PV_set=False,
    OnlyGen=True,
    limit_flow_rate=True,
    solver='ipopt',
    tee=False,
    callback=False,
    obj_scaling=1.0,
    build_only=False,
):
    """Build and solve a coupled NL OPF over frames ``start…end`` (0-based, inclusive).

    Each frame is a snapshot NL block (``window_block=True``); parent
    ``window_soc_constraint`` links SoC across frames. Requires ``grid.ESS`` and
    ``grid.Time_series``. Step duration is ``Storage_* .dt_hours`` (default 1 h);
    frame indices index ``Time_series.data[t]``.

    Parameters
    ----------
    grid : Grid
        Network with storage elements (mutated in place).
    start, end : int, optional
        Inclusive 0-based frame indices aligned with ``Time_series.data[t]``.
    ObjRule : dict or None, optional
        Objective-component weights.
    PV_set : bool, optional
        Fix PV-bus setpoints.
    OnlyGen : bool, optional
        Restrict objective to generator costs.
    limit_flow_rate : bool, optional
        Enforce line flow limits.
    solver : str, optional
        Pyomo solver name.
    tee : bool, optional
        Stream solver output.
    callback : bool, optional
        Enable solver progress callback.
    obj_scaling : float, optional
        Divide objective by this factor.
    build_only : bool, optional
        Build without solving.

    Returns
    -------
    tuple
        ``(model, model_res, timing_info, solver_stats)``.
    """
    grid.reset_run_flags()
    analyse_grid(grid)

    if not grid.ESS:
        raise ValueError("window_nl_opf requires at least one storage element (grid.ESS)")

    if not grid.Time_series:
        raise ValueError("window_nl_opf requires grid.Time_series")

    if start < 0:
        raise ValueError(f"start must be >= 0, got {start}")
    if end < start:
        raise ValueError(f"end ({end}) must be >= start ({start})")

    ts_len = len(grid.Time_series[0].data)
    if end >= ts_len:
        raise ValueError(
            f"end={end} out of range for Time_series length {ts_len} (0-based)")

    weights_def, price_zones = obj_w_rule(grid, ObjRule, OnlyGen)
    frames = list(range(start, end + 1))

    model = pyo.ConcreteModel()
    model.name = f"Window NL OPF [{start}…{end}]"
    model.frames = pyo.Set(initialize=frames)
    model.frame_model = pyo.Block(model.frames)

    t1 = time.perf_counter()
    storage_info = _create_frame_blocks(
        model,
        grid,
        frames,
        PV_set,
        price_zones,
        limit_flow_rate,
        weights_def,
        OnlyGen,
    )
    window_soc_constraints(model, grid, storage_info, frames)

    obj_total = _sum_frame_objectives(model, frames)
    if obj_scaling != 1.0:
        obj_total = obj_total / obj_scaling
    model.obj = pyo.Objective(rule=obj_total, sense=pyo.minimize)
    model.obj_scaling = obj_scaling

    t2 = time.perf_counter()
    t_modelcreate = t2 - t1

    if build_only:
        model_res, solver_stats = build_only_solver_stats(solver, model)
        export_results = True
    else:
        model_res, solver_stats = pyomo_model_solve(
            model, grid, solver, tee, callback=callback)
        export_results = (
            model_res is not None and solver_stats.get('solution_found') is not False)

    t3 = time.perf_counter()
    if export_results:
        grid.window_opf_results = export_window_opf_results(model, grid, frames)
        last_frame = frames[-1]
        for ts in grid.Time_series:
            update_grid_data(grid, ts, last_frame, price_zone_restrictions=price_zones)
        _modify_parameters(grid, model.frame_model[last_frame], price_zones)
        export_acdc_nl_model_to_pyflow_acdc(
            model.frame_model[last_frame], grid, price_zones)

    for obj in weights_def:
        weights_def[obj]['v'] = calculate_objective(grid, obj, OnlyGen)

    t4 = time.perf_counter()

    grid.window_opf_run = True
    grid.OPF_run = True
    grid.OPF_obj = weights_def

    timing_info = {
        'create': t_modelcreate,
        'solve': solver_stats['time'] if solver_stats.get('time') is not None else t4 - t3,
        'export': t4 - t3,
    }
    return model, model_res, timing_info, solver_stats
