"""
pyflow_acdc SOCP optimisation entry point.

Public API
----------
socp_optimise(grid, ...)   – mirror of optimal_pf for the CVXPY SOCP stack.

Design rules (locked):
  L2  – case-agnostic: works on any pyflow Grid
  L4  – files: convex_model.py (build) + this file (run)
  L5  – entry point: socp_optimise (British spelling)
  L16 – objective: min Σ Re(S_export)·price  (≡ max revenue)
  L20 – grid pu throughout; € only at objective scaling
"""

import time
import warnings
from types import SimpleNamespace

import numpy as np

from .constants import ObjComponent, TSType, TS_RENEWABLE_TYPES, default_obj_weights
from .grid_analysis import analyse_grid

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cvxpy is required for the SOCP stack.  "
        "Install it with: pip install pyflow_acdc[SOCP]"
    ) from exc

from .convex_model import build_socp_data, socp_model

__all__ = ["socp_optimise", "soc_window_optimisation", "translate_pyf_socp"]


def _rs_node_number(grid, rs):
    """Resolve RenSource host AC node number from ``rs.Node`` name."""
    nodes = getattr(grid, 'nodes_AC_dict', None)
    if nodes is None:
        nodes = {n.name: n for n in grid.nodes_AC}
    node = nodes.get(rs.Node)
    if node is None:
        raise ValueError(f"RenSource {rs.name!r} Node={rs.Node!r} not found in grid.nodes_AC")
    return node.nodeNumber


def _ts_type(ts):
    """Normalize TimeSeries.type to a comparable value."""
    typ = ts.type
    return typ.value if hasattr(typ, 'value') else typ


# ---------------------------------------------------------------------------
# Translate  Grid  →  SOCP inputs
# ---------------------------------------------------------------------------

def translate_pyf_socp(grid, gamma=1.0, frame_ids=None, P_ext_bounds=None):
    """Prepare SOCP data from *grid* and ``grid.Time_series``.

    Elements do not hold TS arrays; they only link by name. Profiles are read
    from ``grid.Time_series`` (same pattern as window OPF / ``update_grid_data``).

    Parameters
    ----------
    grid : Grid
        Analysed pyflow_acdc grid.
    gamma : float
        Renewable curtailment parameter (L12).  ``1.0`` = no curtailment.
    frame_ids : sequence or None
        Absolute indices into each ``TimeSeries.data``.  If ``None`` and the
        grid has time series, uses the full horizon ``0…len-1``.  If ``None``
        and there are no time series, uses a single static frame.
    P_ext_bounds : dict or None
        Optional ``{node: (P_min, P_max)}`` overrides for ext-grid generators.

    Returns
    -------
    SimpleNamespace
        Static SOCP data plus ``T``, ``frame_ids``, ``P_ren``, ``prices``,
        and ``P_ext_bounds``.
    """
    socp_data = build_socp_data(grid)

    series = list(grid.Time_series)
    if frame_ids is None:
        if series:
            T = len(series[0].data)
            frame_ids = list(range(T))
        else:
            T = 1
            frame_ids = [0]
    else:
        frame_ids = list(frame_ids)
        T = len(frame_ids)
        if T < 1:
            raise ValueError("frame_ids must contain at least one frame")

    if series:
        lengths = {len(ts.data) for ts in series}
        if len(lengths) != 1:
            raise ValueError(f"All Time_series must share the same length; got {sorted(lengths)}")
        ts_len = lengths.pop()
        for abs_t in frame_ids:
            if abs_t < 0 or abs_t >= ts_len:
                raise ValueError(f"frame_id={abs_t} out of range for Time_series length {ts_len}")

    nodes_ac = {n.name: n for n in grid.nodes_AC}
    grid.nodes_AC_dict = nodes_ac
    rs_by_name = {rs.name: rs for rs in grid.RenSources}
    zones_by_name = {z.name: z for z in getattr(grid, 'RenSource_zones', []) or []}
    pz_by_name = {pz.name: pz for pz in getattr(grid, 'Price_Zones', []) or []}

    # ---- renewable injections: node → array(T) ----------------------------
    # Start from current/static availability, then overwrite from TS profiles.
    avail = {rs.name: np.full(T, float(rs.PRGi_available), dtype=float) for rs in grid.RenSources}

    for ts in series:
        typ = _ts_type(ts)
        if typ not in TS_RENEWABLE_TYPES:
            continue
        data = np.asarray(ts.data, dtype=float)
        profile = np.array([data[abs_t] for abs_t in frame_ids], dtype=float)

        rs = rs_by_name.get(ts.element_name)
        if rs is not None:
            avail[rs.name] = profile
            continue
        zone = zones_by_name.get(ts.element_name)
        if zone is not None:
            for linked in zone.RenSources:
                if linked.PGRi_linked:
                    avail[linked.name] = profile

    P_ren_out = {}
    for rs in grid.RenSources:
        n = _rs_node_number(grid, rs)
        scale = rs.PGi_ren_base * getattr(rs, 'np_rsgen', 1) * gamma
        inj = avail[rs.name] * scale
        if n in P_ren_out:
            P_ren_out[n] = P_ren_out[n] + inj
        else:
            P_ren_out[n] = inj

    # ---- prices: node → array(T) ------------------------------------------
    prices_out = {}
    # Static fallback from current price-zone / node prices
    for n in grid.nodes_AC:
        prices_out[n.nodeNumber] = np.full(T, float(getattr(n, 'price', 0.0)), dtype=float)

    for ts in series:
        typ = _ts_type(ts)
        if typ != TSType.PRICE:
            continue
        data = np.asarray(ts.data, dtype=float)
        profile = np.array([data[abs_t] for abs_t in frame_ids], dtype=float)

        pz = pz_by_name.get(ts.element_name)
        if pz is not None:
            for n_ac in getattr(pz, 'nodes_AC', []) or []:
                node_num = n_ac.nodeNumber if hasattr(n_ac, 'nodeNumber') else nodes_ac[n_ac].nodeNumber
                prices_out[node_num] = profile.copy()

        node = nodes_ac.get(ts.element_name)
        if node is not None:
            prices_out[node.nodeNumber] = profile.copy()

    return SimpleNamespace(
        **vars(socp_data),
        T=T,
        frame_ids=frame_ids,
        P_ren=P_ren_out,
        prices=prices_out,
        P_ext_bounds=P_ext_bounds or {},
    )


# ---------------------------------------------------------------------------
# Build objective
# ---------------------------------------------------------------------------

def _default_socp_obj_weights():
    """Return SOCP default objective weights matching current v1 behavior."""
    weights = default_obj_weights()
    weights[ObjComponent.ENERGY_COST.value]['w'] = 1
    return weights


def _sum_ac_losses_expr(ac, d):
    """Return total AC active losses across all sparse AC edges."""
    terms = []
    for t in range(d.T):
        for (k, m) in d.E_AC:
            w_km_t = ac.w_AC[(k, m)][t]
            s_km = cp.conj(d.Ybus_AC[k, m]) * (ac.h_AC[k, t] - w_km_t)
            s_mk = cp.conj(d.Ybus_AC[m, k]) * (ac.h_AC[m, t] - cp.conj(w_km_t))
            terms.append(cp.real(s_km + s_mk))
    return sum(terms) if terms else 0


def _sum_dc_losses_expr(dc, d):
    """Return total DC active losses across all sparse DC edges."""
    if dc is None:
        return 0

    terms = []
    for t in range(d.T):
        for (k, m) in d.E_DC:
            w_km_t = dc.w_DC[(k, m)][t]
            p_km = (dc.h_DC[k, t] - w_km_t) * d.Ybus_DC[k, m]
            p_mk = (dc.h_DC[m, t] - w_km_t) * d.Ybus_DC[m, k]
            terms.append(p_km + p_mk)
    return sum(terms) if terms else 0


def _build_objective(variables, d, grid, weights_def=None):
    """Construct the weighted SOCP objective using NLP OPF component names.

    Supported v1 components:
    - ``Energy_cost``       : OPF-like generator energy cost
    - ``Ext_Gen``           : total external-grid active power
    - ``AC_losses``         : AC branch active losses
    - ``DC_losses``         : DC branch active losses
    - ``Converter_Losses``  : converter losses

    Any active unsupported component raises ``NotImplementedError``.
    """
    if weights_def is None:
        weights_def = getattr(grid, 'OPF_obj', None) or _default_socp_obj_weights()

    ac = variables.ac
    gen = variables.gen
    dc = variables.dc
    conv = variables.conv
    terms = []
    np_den_eps = 1e-3

    for component, cfg in weights_def.items():
        weight = cfg.get('w', 0)
        if weight == 0:
            continue

        if component == ObjComponent.ENERGY_COST.value:
            if gen is None:
                continue
            for gi, gd in enumerate(d.gen_data_AC):
                for t in range(d.T):
                    p_mw = gen.PGi_gen[gi, t] * d.Sbase
                    terms.append(
                        weight * (
                            (p_mw ** 2) * gd['qf'] / (gd['np_gen'] + np_den_eps)
                            + p_mw * gd['lf']
                            + gd['np_gen'] * gd['fc']
                        )
                    )
        elif component == ObjComponent.EXT_GEN.value:
            if gen is None:
                continue
            for gi, gd in enumerate(d.gen_data_AC):
                if gd['is_ext_grid']:
                    terms.append(weight * cp.sum(gen.PGi_gen[gi, :]) * d.Sbase)
        elif component == ObjComponent.AC_LOSSES.value:
            terms.append(weight * _sum_ac_losses_expr(ac, d) * grid.LCoE * d.Sbase)
        elif component == ObjComponent.DC_LOSSES.value:
            terms.append(weight * _sum_dc_losses_expr(dc, d) * grid.LCoE * d.Sbase)
        elif component == ObjComponent.CONVERTER_LOSSES.value:
            if conv is None:
                continue
            terms.append(weight * cp.sum(conv.Ploss) * grid.LCoE * d.Sbase)
        else:
            raise NotImplementedError(
                f"SOCP objective component '{component}' is not implemented."
            )

    if not terms:
        warnings.warn(
            "socp_optimise: objective is zero — no active supported components.",
            stacklevel=3,
        )
        return cp.Minimize(0)

    return cp.Minimize(sum(terms))


# ---------------------------------------------------------------------------
# Export solution back to Grid
# ---------------------------------------------------------------------------

def _export_to_grid(grid, variables, socp_data):
    """Write SOCP solution values back onto Grid elements.

    Only scalar (T=1) or the first time step is used for static grid attrs;
    full time-series are stored on ``grid.socp_results``.
    """
    ac   = variables.ac
    gen  = variables.gen
    dc   = variables.dc
    conv = variables.conv

    # Voltage magnitudes → AC nodes
    if ac.h_AC.value is not None:
        for n in grid.nodes_AC:
            n.V_AC = float(np.sqrt(np.clip(ac.h_AC.value[n.nodeNumber, 0], 0, None)))

    # Voltage magnitudes → DC nodes
    if dc is not None and dc.h_DC.value is not None:
        for n in grid.nodes_DC:
            n.V_DC = float(np.sqrt(np.clip(dc.h_DC.value[n.nodeNumber, 0], 0, None)))

    # Generator results
    if gen is not None and gen.PGi_gen.value is not None:
        for gi, gd in enumerate(socp_data.gen_data_AC):
            g_obj = grid.Generators[gd['idx']]
            g_obj.PGen = float(gen.PGi_gen.value[gi, 0])
            g_obj.QGen = float(gen.QGi_gen.value[gi, 0])

    # Converter results
    if conv is not None and conv.Ss.value is not None:
        for ci, cd in enumerate(socp_data.conv_data):
            c_obj = grid.Converters_ACDC[ci]
            c_obj.P_AC   = float(np.real(conv.Ss.value[ci, 0]))
            c_obj.Q_AC   = float(np.imag(conv.Ss.value[ci, 0]))
            c_obj.P_loss = float(conv.Ploss.value[ci, 0]) if conv.Ploss.value is not None else 0.0
            if dc is not None and dc.P_DC.value is not None:
                c_obj.P_DC = float(dc.P_DC.value[cd['nDC'], 0])

    # Store full time-series for post-processing
    grid.socp_results = SimpleNamespace(
        h_AC  = ac.h_AC.value,
        w_AC  = {e: v.value for e, v in ac.w_AC.items()},
        PGi_gen = gen.PGi_gen.value if gen is not None else None,
        QGi_gen = gen.QGi_gen.value if gen is not None else None,
        h_DC  = dc.h_DC.value  if dc   is not None else None,
        P_DC  = dc.P_DC.value  if dc   is not None else None,
        Ss    = conv.Ss.value  if conv is not None else None,
        Ploss = conv.Ploss.value if conv is not None else None,
        T     = socp_data.T,
        frame_ids = socp_data.frame_ids,
    )

    grid.socp_run = True


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def socp_optimise(
    grid,
    gamma=1.0,
    frame_id=0,
    P_ext_bounds=None,
    weights_def=None,
    solver=None,
    solver_opts=None,
    build_only=False,
    verbose=False,
):
    """Build and solve the sparse AC/DC SOCP for *grid* (single period).

    Parameters
    ----------
    grid : Grid
        pyflow_acdc grid (will be analysed if not already).
    gamma : float
        Renewable curtailment parameter (L12).  ``1.0`` = full dispatch.
    frame_id : int
        Absolute index into ``grid.Time_series`` data when series are present.
        Ignored for static (no-TS) grids.
    P_ext_bounds : dict or None
        ``{node_AC: (P_min_pu, P_max_pu)}`` for export buses.
    weights_def : dict or None
        Objective-component weights using the same public keys as NLP OPF
        (for example ``Energy_cost`` or ``AC_losses``). If ``None``, uses
        ``grid.OPF_obj`` when available, otherwise defaults to
        ``{'Energy_cost': {'w': 1}}``.
    solver : str or None
        CVXPY solver name, e.g. ``'MOSEK'``, ``'GUROBI'``, ``'CLARABEL'``.
        If ``None``, CVXPY picks automatically.
    solver_opts : dict or None
        Keyword options forwarded to ``problem.solve()``.
    build_only : bool
        If ``True``, build the problem but do not solve.  Useful for
        inspecting model size or CI without a commercial solver.
    verbose : bool
        Stream solver output.

    Returns
    -------
    tuple
        ``(problem, variables, timing_info, solver_stats)``.
    """
    analyse_grid(grid)

    t0 = time.perf_counter()

    socp_data = translate_pyf_socp(
        grid,
        gamma=gamma,
        frame_ids=[frame_id],
        P_ext_bounds=P_ext_bounds,
    )

    t1 = time.perf_counter()

    constraints, variables = socp_model(grid, socp_data)

    objective = _build_objective(
        variables,
        socp_data,
        grid,
        weights_def=weights_def,
    )
    problem   = cp.Problem(objective, constraints)

    t2 = time.perf_counter()

    solver_stats = {
        'status':   None,
        'value':    None,
        'time':     0.0,
        'n_vars':   problem.variables().__len__(),
        'n_constr': len(problem.constraints),
    }

    if build_only:
        timing_info = {'translate': t1 - t0, 'build': t2 - t1, 'solve': 0.0}
        return problem, variables, timing_info, solver_stats

    solve_kwargs = {'verbose': verbose}
    if solver is not None:
        solve_kwargs['solver'] = solver
    if solver_opts:
        solve_kwargs.update(solver_opts)

    t3 = time.perf_counter()
    problem.solve(**solve_kwargs)
    t4 = time.perf_counter()

    solver_stats['status'] = problem.status
    solver_stats['value']  = problem.value
    solver_stats['time']   = t4 - t3

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        warnings.warn(
            f"socp_optimise: solver returned status '{problem.status}'.",
            stacklevel=2,
        )
    else:
        _export_to_grid(grid, variables, socp_data)

    timing_info = {
        'translate': t1 - t0,
        'build':     t2 - t1,
        'solve':     t4 - t3,
    }

    return problem, variables, timing_info, solver_stats


def soc_window_optimisation(
    grid,
    gamma=1.0,
    frame_ids=None,
    P_ext_bounds=None,
    weights_def=None,
    solver=None,
    solver_opts=None,
    build_only=False,
    verbose=False,
):
    """Build and solve the multiperiod/window SOCP for *grid*.

    Profiles come from ``grid.Time_series``.  If ``frame_ids`` is ``None``,
    the full TS horizon is used.
    """
    analyse_grid(grid)

    t0 = time.perf_counter()
    socp_data = translate_pyf_socp(
        grid,
        gamma=gamma,
        frame_ids=frame_ids,
        P_ext_bounds=P_ext_bounds,
    )
    t1 = time.perf_counter()

    constraints, variables = socp_model(grid, socp_data)
    objective = _build_objective(
        variables,
        socp_data,
        grid,
        weights_def=weights_def,
    )
    problem = cp.Problem(objective, constraints)
    t2 = time.perf_counter()

    solver_stats = {
        'status': None,
        'value': None,
        'time': 0.0,
        'n_vars': problem.variables().__len__(),
        'n_constr': len(problem.constraints),
    }

    if build_only:
        timing_info = {'translate': t1 - t0, 'build': t2 - t1, 'solve': 0.0}
        return problem, variables, timing_info, solver_stats

    solve_kwargs = {'verbose': verbose}
    if solver is not None:
        solve_kwargs['solver'] = solver
    if solver_opts:
        solve_kwargs.update(solver_opts)

    t3 = time.perf_counter()
    problem.solve(**solve_kwargs)
    t4 = time.perf_counter()

    solver_stats['status'] = problem.status
    solver_stats['value'] = problem.value
    solver_stats['time'] = t4 - t3

    if problem.status not in ('optimal', 'optimal_inaccurate'):
        warnings.warn(
            f"soc_window_optimisation: solver returned status '{problem.status}'.",
            stacklevel=2,
        )
    else:
        _export_to_grid(grid, variables, socp_data)

    timing_info = {
        'translate': t1 - t0,
        'build': t2 - t1,
        'solve': t4 - t3,
    }

    return problem, variables, timing_info, solver_stats
