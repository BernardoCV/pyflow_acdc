"""Docs: api\\tep_pymoo.rst — Transmission Expansion (pymoo)"""
import pyflow_acdc as pyf

grid, res = pyf.cases["case39_acdc"](
    TEP=True, exp="All", N_b_dc=0, N_b_ac=0, N_i=0, N_max=5, Increase=1.5
)
obj = {"Energy_cost": 1}
model, model_results, timing_info, solver_stats = pyf.transmission_expansion_pymoo(
    grid,
    NPV=True,
    ObjRule=obj,
    solver="GA",
    n_gen=2,
    tee=False,
)
