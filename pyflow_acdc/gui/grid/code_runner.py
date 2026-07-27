# -*- coding: utf-8 -*-
"""Run pasted case-style code that assigns ``grid``."""

from __future__ import annotations

import traceback

import pandas as pd

import pyflow_acdc as pyf
from pyflow_acdc.Classes import Grid
from pyflow_acdc.constants import (
    ConverterDCType,
    DataInput,
    NodeType,
    Polarity,
)


def run_grid_code(source: str, existing_grid: Grid | None = None) -> Grid:
    """Execute user code; require a ``Grid`` assigned to ``grid``."""
    if not source or not source.strip():
        raise ValueError("Paste code that builds a grid and assigns it to 'grid'.")

    namespace = {
        "pyf": pyf,
        "pyflow_acdc": pyf,
        "pd": pd,
        "pandas": pd,
        "grid": existing_grid,
        "NodeType": NodeType,
        "ConverterDCType": ConverterDCType,
        "Polarity": Polarity,
        "DataInput": DataInput,
    }
    try:
        exec(source, namespace)  # noqa: S102 — trusted local GUI paste
    except Exception as exc:
        raise RuntimeError(f"Code failed:\n{traceback.format_exc()}") from exc

    grid = namespace.get("grid")
    if grid is None or not isinstance(grid, Grid):
        raise ValueError(
            "Code must assign a pyflow_acdc Grid to the variable 'grid'."
        )
    return grid
