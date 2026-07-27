# -*- coding: utf-8 -*-
"""Study jobs callable without Qt (unit-tested)."""

from __future__ import annotations

from pyflow_acdc.ACDC_PF import power_flow
from pyflow_acdc.Classes import Grid
from pyflow_acdc.Results_class import Results


def run_power_flow_job(grid: Grid) -> Results:
    """Run PF and build Results tables (no terminal output)."""
    power_flow(grid)
    results = Results(grid, save_res=False)
    results.all(print_table=False)
    return results
