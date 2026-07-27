# -*- coding: utf-8 -*-
"""Element counts for the Grid sidebar inspector."""

from __future__ import annotations

from pyflow_acdc.Classes import Grid


def grid_element_counts(grid: Grid | None) -> list[tuple[str, int]]:
    """Return (label, count) pairs for the main Grid collections."""
    if grid is None:
        return []
    return [
        ("AC nodes", len(grid.nodes_AC)),
        ("DC nodes", len(grid.nodes_DC)),
        ("AC lines", len(grid.lines_AC)),
        ("DC lines", len(grid.lines_DC)),
        ("AC/DC converters", len(grid.Converters_ACDC)),
        ("DC/DC converters", len(grid.Converters_DCDC)),
        ("Generators (AC)", len(grid.Generators)),
        ("Generators (DC)", len(grid.Generators_DC)),
        ("Renewable sources", len(grid.RenSources)),
        ("Storage", len(grid.storage_elements)),
        ("Electrolysers", len(grid.electrolysers)),
        ("Price zones", len(grid.Price_Zones)),
        ("Time series", len(grid.Time_series)),
    ]


def format_grid_inventory(grid: Grid | None) -> str:
    if grid is None:
        return "No grid loaded."
    name = getattr(grid, "name", None) or "(unnamed)"
    lines = [f"Name: {name}", f"S_base: {grid.S_base}", ""]
    for label, count in grid_element_counts(grid):
        lines.append(f"{label}: {count}")
    return "\n".join(lines)
