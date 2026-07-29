"""
Sparse SOCP model builder for pyflow_acdc.

Constructs the CVXPY variables, SOC constraints, nodal balances, thermal
limits, and converter coupling for an AC/DC grid SOCP relaxation following
Useche-Arteaga et al. (SEGAN 2026).

Structure mirrors ACDC_OPF_NL_model.py:
  build_socp_data        – extract topology/data from Grid → SimpleNamespace
  socp_model             – top-level orchestrator on prepared SOCP data
    generator_variables  – PGi_gen, QGi_gen
    ac_variables         – h_AC, w_AC
    ac_constraints       – voltage bounds, SOC lifts, nodal balance, thermals
    dc_variables         – h_DC, w_DC, P_DC
    dc_constraints       – voltage bounds, SOC lifts, nodal balance, thermals
    converter_variables  – Ss, Ploss
    converter_constraints– power balance, loss model, rating

Design rules (locked):
  L3  – CVXPY only (no Pyomo)
  L8  – Converter AC-side RL branches ignored in Ybus
  L11 – Balance uses conj(S) == flow_k
  L13 – Converter loss: Ploss = a + c * |Ss|^2
  L14 – DC polarity: pol = cn_pol
  L15 – AC and DC thermal limits are mandatory
  L19 – Sparse edge sets only; no dense mode
  L20 – All quantities in grid pu
"""

import numpy as np
from types import SimpleNamespace

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "cvxpy is required for the SOCP stack.  "
        "Install it with: pip install pyflow_acdc[SOCP]"
    ) from exc

__all__ = [
    "build_socp_data",
    "socp_model",
    "generator_variables",
    "generator_constraints",
    "ac_variables",
    "ac_constraints",
    "dc_variables",
    "dc_constraints",
    "converter_variables",
    "converter_constraints",
]


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------

def build_socp_data(grid):
    """Extract all SOCP-relevant data from *grid* into a SimpleNamespace.

    Parameters
    ----------
    grid : Grid
        An analysed pyflow_acdc grid (``analyse_grid`` must have been called).

    Returns
    -------
    SimpleNamespace
        All power / admittance values are in grid pu.
    """
    Sbase = grid.S_base

    # ------------------------------------------------------------------ AC --
    N_AC = grid.nn_AC
    Ybus_AC = np.array(grid.Ybus_AC, dtype=complex)  # (N_AC, N_AC)

    E_AC = set()
    for k in range(N_AC):
        for m in range(k + 1, N_AC):
            if abs(Ybus_AC[k, m]) > 1e-12:
                E_AC.add((k, m))

    V_min_AC = np.array([n.Umin if hasattr(n, 'Umin') and n.Umin is not None else 0.9
                         for n in grid.nodes_AC])
    V_max_AC = np.array([n.Umax if hasattr(n, 'Umax') and n.Umax is not None else 1.1
                         for n in grid.nodes_AC])

    ac_slack = [n.nodeNumber for n in grid.nodes_AC
                if str(n.type).lower() in ('slack', 'nodetype.slack')]

    ac_line_limits = {}
    for line in grid.Lines_AC:
        k = line.fromNode.nodeNumber
        m = line.toNode.nodeNumber
        rating_pu = line.MVA_rating / Sbase if line.MVA_rating and line.MVA_rating > 0 else None
        edge = (min(k, m), max(k, m))
        if rating_pu is not None:
            if edge not in ac_line_limits or rating_pu < ac_line_limits[edge]:
                ac_line_limits[edge] = rating_pu

    # ------------------------------------------------------------------ DC --
    N_DC = grid.nn_DC
    Ybus_DC = np.array(grid.Ybus_DC, dtype=float)  # (N_DC, N_DC)

    E_DC = set()
    for k in range(N_DC):
        for m in range(k + 1, N_DC):
            if abs(Ybus_DC[k, m]) > 1e-12:
                E_DC.add((k, m))

    V_min_DC = np.array([n.Umin if hasattr(n, 'Umin') and n.Umin is not None else 0.9
                         for n in grid.nodes_DC])
    V_max_DC = np.array([n.Umax if hasattr(n, 'Umax') and n.Umax is not None else 1.1
                         for n in grid.nodes_DC])

    dc_slack = [n.nodeNumber for n in grid.nodes_DC
                if str(n.type).lower() in ('slack', 'nodetype.slack')]

    dc_line_limits = {}
    for line in grid.Lines_DC:
        k = line.fromNode.nodeNumber
        m = line.toNode.nodeNumber
        rating_pu = line.MW_rating / Sbase if line.MW_rating and line.MW_rating > 0 else None
        edge = (min(k, m), max(k, m))
        if rating_pu is not None:
            if edge not in dc_line_limits or rating_pu < dc_line_limits[edge]:
                dc_line_limits[edge] = rating_pu * getattr(line, 'np_line', 1)

    # -------------------------------------------------------------- converters
    conv_data = []
    for conv in grid.Converters_ACDC:
        np_c = conv.np_conv if hasattr(conv, 'np_conv') and conv.np_conv > 0 else 1
        conv_data.append({
            'idx':  conv.ConvNumber,
            'nAC':  conv.Node_AC.nodeNumber,
            'nDC':  conv.Node_DC.nodeNumber,
            'pol':  abs(conv.cn_pol) if hasattr(conv, 'cn_pol') else 1,
            'a':    conv.a_conv,
            'c':    conv.c_rect,
            'Smax': conv.MVA_max * np_c / Sbase,
        })

    conv_ac_nodes = {cd['nAC'] for cd in conv_data}

    # --------------------------------------------------------- bus role maps
    gen_data_AC = []
    ren_nodes_AC = {}

    for gen in grid.Generators:
        np_gen = getattr(gen, 'np_gen', 1)
        if getattr(gen, 'is_ext_grid', False):
            if not getattr(gen, 'allow_sell', True):
                p_min = 0.0
            else:
                p_min = -(gen.Max_pow_gen * np_gen - gen.p_load_eff)
        else:
            p_min = gen.Min_pow_gen * np_gen
        p_max = gen.Max_pow_gen * np_gen
        q_min = gen.Min_pow_genR * np_gen
        q_max = gen.Max_pow_genR * np_gen
        p_ini = min(max(gen.Pset * np_gen, p_min), p_max)
        q_ini = min(max(gen.Qset * np_gen, q_min), q_max)
        gen_data_AC.append({
            'idx': gen.genNumber,
            'node': gen._node.nodeNumber,
            'is_ext_grid': getattr(gen, 'is_ext_grid', False),
            'p_min': p_min,
            'p_max': p_max,
            'q_min': q_min,
            'q_max': q_max,
            'p_ini': p_ini,
            'q_ini': q_ini,
            'qf': gen.qf,
            'lf': gen.lf,
            'fc': gen.fc,
            'np_gen': np_gen,
            'max_s': gen.Max_S * np_gen if gen.Max_S is not None else None,
        })

    nodes_ac = {n.name: n for n in grid.nodes_AC}
    for rs in grid.RenSources:
        node = nodes_ac.get(rs.Node)
        if node is None:
            raise ValueError(f"RenSource {rs.name!r} Node={rs.Node!r} not found in grid.nodes_AC")
        ren_nodes_AC[node.nodeNumber] = (
            rs.PGi_ren_base * rs.PRGi_available * getattr(rs, 'np_rsgen', 1),
            0.0,
        )  # Q=0 (L12)

    return SimpleNamespace(
        N_AC=N_AC, Ybus_AC=Ybus_AC, E_AC=E_AC,
        V_min_AC=V_min_AC, V_max_AC=V_max_AC, ac_slack=ac_slack,
        ac_line_limits=ac_line_limits,
        N_DC=N_DC, Ybus_DC=Ybus_DC, E_DC=E_DC,
        V_min_DC=V_min_DC, V_max_DC=V_max_DC, dc_slack=dc_slack,
        dc_line_limits=dc_line_limits,
        conv_data=conv_data, conv_ac_nodes=conv_ac_nodes,
        gen_data_AC=gen_data_AC, ren_nodes_AC=ren_nodes_AC,
        Sbase=Sbase,
    )


# ---------------------------------------------------------------------------
# Generator variables
# ---------------------------------------------------------------------------

def generator_variables(d, T):
    """Declare AC generator decision variables."""
    n_gen = len(d.gen_data_AC)
    p_ini = np.array([g['p_ini'] for g in d.gen_data_AC], dtype=float)
    q_ini = np.array([g['q_ini'] for g in d.gen_data_AC], dtype=float)

    PGi_gen = cp.Variable((n_gen, T), name="PGi_gen")
    QGi_gen = cp.Variable((n_gen, T), name="QGi_gen")

    return SimpleNamespace(PGi_gen=PGi_gen, QGi_gen=QGi_gen, p_ini=p_ini, q_ini=q_ini)


# ---------------------------------------------------------------------------
# Generator constraints
# ---------------------------------------------------------------------------

def generator_constraints(d, gen_vars):
    """Build AC generator bounds and capability constraints."""
    constrs = []
    T = d.T
    for gi, gd in enumerate(d.gen_data_AC):
        p_min = gd['p_min']
        p_max = gd['p_max']
        if gd['is_ext_grid'] and gd['node'] in d.P_ext_bounds:
            p_min, p_max = d.P_ext_bounds[gd['node']]
        constrs += [
            gen_vars.PGi_gen[gi, :] >= p_min,
            gen_vars.PGi_gen[gi, :] <= p_max,
            gen_vars.QGi_gen[gi, :] >= gd['q_min'],
            gen_vars.QGi_gen[gi, :] <= gd['q_max'],
        ]
        if gd['max_s'] is not None:
            for t in range(T):
                constrs += [
                    cp.norm(cp.vstack([gen_vars.PGi_gen[gi, t], gen_vars.QGi_gen[gi, t]])) <= gd['max_s']
                ]
    return constrs


# ---------------------------------------------------------------------------
# AC variables
# ---------------------------------------------------------------------------

def ac_variables(d, T):
    """Declare AC SOCP variables.

    Returns
    -------
    SimpleNamespace
        ``h_AC`` – voltage magnitude squared (N_AC × T)
        ``w_AC`` – sparse cross-product dict (k,m) → (T,) complex
    """
    h_AC = cp.Variable((d.N_AC, T), nonneg=True, name="h_AC")
    w_AC = {e: cp.Variable(T, complex=True, name=f"w_AC_{e[0]}_{e[1]}")
            for e in d.E_AC}

    return SimpleNamespace(h_AC=h_AC, w_AC=w_AC)


# ---------------------------------------------------------------------------
# AC constraints
# ---------------------------------------------------------------------------

def ac_constraints(d, grid, ac_vars, gen_vars, conv_vars):
    """Build AC constraints: voltage bounds, SOC lifts, nodal balance, thermals.

    Parameters
    ----------
    d : SimpleNamespace
        Output of :func:`build_socp_data`.
    grid : Grid
        Live grid (used for ACmode flag).
    ac_vars : SimpleNamespace
        Output of :func:`ac_variables`.
    conv_vars : SimpleNamespace or None
        Output of :func:`converter_variables`, or ``None`` if no converters.
    Returns
    -------
    list
        CVXPY constraints.
    """
    h_AC  = ac_vars.h_AC
    w_AC  = ac_vars.w_AC
    Ss    = conv_vars.Ss if conv_vars is not None else None
    T = d.T

    constrs = []

    for t in range(T):
        # voltage magnitude bounds
        constrs += [h_AC[:, t] >= d.V_min_AC ** 2,
                    h_AC[:, t] <= d.V_max_AC ** 2]

        # slack bus fixed at 1 pu²
        for k in d.ac_slack:
            constrs += [h_AC[k, t] == 1.0]

        # SOC lifts on sparse edges (L19)
        for (k, m) in d.E_AC:
            w_km = w_AC[(k, m)][t]
            constrs += [
                cp.SOC(
                    h_AC[k, t] + h_AC[m, t],
                    cp.vstack([
                        2 * cp.real(w_km),
                        2 * cp.imag(w_km),
                        h_AC[k, t] - h_AC[m, t],
                    ])
                )
            ]

        # nodal power balance: conj(S_k) == flow_k (L11)
        for k in range(d.N_AC):
            flow_k = d.Ybus_AC[k, k] * h_AC[k, t]
            for m in range(d.N_AC):
                if m == k or abs(d.Ybus_AC[k, m]) < 1e-12:
                    continue
                w_km_t = w_AC[(k, m)][t] if (k, m) in w_AC else cp.conj(w_AC[(m, k)][t])
                flow_k = flow_k + d.Ybus_AC[k, m] * w_km_t

            S_k = _ac_node_injection(k, t, d, gen_vars, Ss)
            constrs += [flow_k == 0] if S_k is None else [cp.conj(S_k) == flow_k]

        # thermal limits (L15)
        for (k, m), rating_pu in d.ac_line_limits.items():
            w_km_t = w_AC[(k, m)][t] if (k, m) in w_AC else cp.conj(w_AC[(m, k)][t])
            S_km_from = cp.conj(d.Ybus_AC[k, m]) * (h_AC[k, t] - w_km_t)
            S_km_to   = cp.conj(d.Ybus_AC[m, k]) * (h_AC[m, t] - cp.conj(w_km_t))
            constrs += [cp.norm(cp.vstack([cp.real(S_km_from), cp.imag(S_km_from)])) <= rating_pu,
                        cp.norm(cp.vstack([cp.real(S_km_to),   cp.imag(S_km_to)]))   <= rating_pu]

    return constrs


# ---------------------------------------------------------------------------
# DC variables
# ---------------------------------------------------------------------------

def dc_variables(d, T):
    """Declare DC SOCP variables.

    Returns
    -------
    SimpleNamespace
        ``h_DC`` – voltage squared (N_DC × T)
        ``w_DC`` – sparse cross-product dict (k,m) → (T,) real
        ``P_DC``  – nodal active injection (N_DC × T)
    """
    h_DC = cp.Variable((d.N_DC, T), nonneg=True, name="h_DC")
    w_DC = {e: cp.Variable(T, nonneg=True, name=f"w_DC_{e[0]}_{e[1]}")
            for e in d.E_DC}
    P_DC = cp.Variable((d.N_DC, T), name="P_DC")

    return SimpleNamespace(h_DC=h_DC, w_DC=w_DC, P_DC=P_DC)


# ---------------------------------------------------------------------------
# DC constraints
# ---------------------------------------------------------------------------

def dc_constraints(d, dc_vars):
    """Build DC constraints: voltage bounds, SOC lifts, nodal balance, thermals.

    Returns
    -------
    list
        CVXPY constraints.
    """
    h_DC = dc_vars.h_DC
    w_DC = dc_vars.w_DC
    P_DC = dc_vars.P_DC
    T = d.T

    constrs = []

    for t in range(T):
        # voltage magnitude bounds
        constrs += [h_DC[:, t] >= d.V_min_DC ** 2,
                    h_DC[:, t] <= d.V_max_DC ** 2]

        # slack bus fixed at 1 pu²
        for k in d.dc_slack:
            constrs += [h_DC[k, t] == 1.0]

        # SOC lifts on sparse edges (L19)
        for (k, m) in d.E_DC:
            w_km = w_DC[(k, m)][t]
            constrs += [
                cp.SOC(
                    h_DC[k, t] + h_DC[m, t],
                    cp.vstack([
                        2 * w_km,
                        h_DC[k, t] - h_DC[m, t],
                    ])
                )
            ]

        # nodal balance: P_DC[k] = pol * flow_k (L14)
        for cd in d.conv_data:
            k = cd['nDC']
            flow_k = d.Ybus_DC[k, k] * h_DC[k, t]
            for m in range(d.N_DC):
                if m == k or abs(d.Ybus_DC[k, m]) < 1e-12:
                    continue
                w_km_t = w_DC[(k, m)][t] if (k, m) in w_DC else w_DC[(m, k)][t]
                flow_k = flow_k + d.Ybus_DC[k, m] * w_km_t
            constrs += [P_DC[k, t] == cd['pol'] * flow_k]

        # thermal limits (L15)
        for (k, m), rating_pu in d.dc_line_limits.items():
            w_km_t = w_DC[(k, m)][t] if (k, m) in w_DC else w_DC[(m, k)][t]
            P_km = (h_DC[k, t] - w_km_t) * d.Ybus_DC[k, m]
            P_mk = (h_DC[m, t] - w_km_t) * d.Ybus_DC[m, k]
            constrs += [cp.abs(P_km) <= rating_pu,
                        cp.abs(P_mk) <= rating_pu]

    return constrs


# ---------------------------------------------------------------------------
# Converter variables
# ---------------------------------------------------------------------------

def converter_variables(d, T):
    """Declare VSC converter variables.

    Returns
    -------
    SimpleNamespace
        ``Ss``    – AC-side apparent power (n_conv × T) complex
        ``Ploss`` – converter losses       (n_conv × T) real
    """
    n_conv = len(d.conv_data)
    Ss    = cp.Variable((n_conv, T), complex=True, name="Ss")
    Ploss = cp.Variable((n_conv, T), name="Ploss")

    return SimpleNamespace(Ss=Ss, Ploss=Ploss)


# ---------------------------------------------------------------------------
# Converter constraints
# ---------------------------------------------------------------------------

def converter_constraints(d, conv_vars, dc_vars):
    """Build converter constraints: power balance, loss model, rating.

    Returns
    -------
    list
        CVXPY constraints.
    """
    Ss    = conv_vars.Ss
    Ploss = conv_vars.Ploss
    P_DC  = dc_vars.P_DC
    T = d.T

    constrs = []

    for t in range(T):
        for ci, cd in enumerate(d.conv_data):
            # power balance: Re(Ss) + P_DC + Ploss = 0 (L13)
            constrs += [cp.real(Ss[ci, t]) + P_DC[cd['nDC'], t] + Ploss[ci, t] == 0]

            # loss model: Ploss = a + c * |Ss|²  (h_AC = 1 pu denominator, v1; L13)
            constrs += [
                Ploss[ci, t] == cd['a'] + cd['c'] * (
                    cp.real(Ss[ci, t]) ** 2 + cp.imag(Ss[ci, t]) ** 2)
            ]

            # apparent power rating
            constrs += [cp.norm(Ss[ci, t]) <= cd['Smax']]

    return constrs


# ---------------------------------------------------------------------------
# Top-level model builder
# ---------------------------------------------------------------------------

def socp_model(grid, d):
    """Build all CVXPY SOCP variables and constraints for prepared SOCP data.

    Parameters
    ----------
    grid : Grid
        pyflow_acdc grid (``analyse_grid`` must have been called).
    d : SimpleNamespace
        Prepared SOCP data object with static grid data plus translated
        multiperiod inputs (``T``, ``frame_ids``, ``P_ren``, ``prices``,
        ``P_ext_bounds``).

    Returns
    -------
    tuple
        ``(constraints, variables)`` where *variables* is a SimpleNamespace
        holding all CVXPY variable blocks.
    """
    T = d.T

    # -- variables -----------------------------------------------------------
    gen_vars  = generator_variables(d, T) if grid.ACmode and d.gen_data_AC else None
    ac_vars   = ac_variables(d, T)
    dc_vars   = dc_variables(d, T)   if grid.DCmode             else None
    conv_vars = converter_variables(d, T) if grid.ACmode and grid.DCmode and d.conv_data else None

    # -- constraints ---------------------------------------------------------
    constrs  = generator_constraints(d, gen_vars) if gen_vars is not None else []
    constrs += ac_constraints(d, grid, ac_vars, gen_vars, conv_vars)
    if grid.DCmode:
        constrs += dc_constraints(d, dc_vars)
    if grid.ACmode and grid.DCmode and d.conv_data:
        constrs += converter_constraints(d, conv_vars, dc_vars)

    variables = SimpleNamespace(
        gen=gen_vars,
        ac=ac_vars,
        dc=dc_vars,
        conv=conv_vars,
    )

    return constrs, variables


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _ac_node_injection(k, t, d, gen_vars, Ss):
    """Return the net complex injection at AC node *k*, time *t*.

    Returns ``None`` for zero-injection nodes.
    """
    parts = []

    for gi, gd in enumerate(d.gen_data_AC):
        if gd['node'] == k:
            parts.append(gen_vars.PGi_gen[gi, t] + 1j * gen_vars.QGi_gen[gi, t])

    if k in d.ren_nodes_AC:
        p_t = float(d.P_ren[k][t]) if k in d.P_ren else d.ren_nodes_AC[k][0]
        parts.append(p_t + 1j * d.ren_nodes_AC[k][1])

    if k in d.conv_ac_nodes:
        for ci, cd in enumerate(d.conv_data):
            if cd['nAC'] == k:
                parts.append(Ss[ci, t])

    if not parts:
        return None

    result = parts[0]
    for p in parts[1:]:
        result = result + p
    return result
