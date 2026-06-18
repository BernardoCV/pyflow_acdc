"""Docs: api/csv_import.rst — Data in per unit (pu)"""
from pathlib import Path

import pyflow_acdc as pyf

_DATA = Path(__file__).resolve().parent / "data"

grid, results = pyf.create_grid_from_data(
    S_base=100,
    AC_node_data=str(_DATA / "AC_node_data.csv"),
    AC_line_data=str(_DATA / "AC_line_data.csv"),
    DC_node_data=str(_DATA / "DC_node_data.csv"),
    DC_line_data=str(_DATA / "DC_line_data.csv"),
    Converter_data=str(_DATA / "Converter_data.csv"),
    data_in="pu",
)