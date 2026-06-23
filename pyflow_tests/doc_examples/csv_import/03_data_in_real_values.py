"""Docs: api/csv_import.rst — Data in Real values"""
from pathlib import Path

import pyflow_acdc as pyf

_DATA = Path(__file__).resolve().parents[3] / "examples" / "Stagg5MATACDC"

grid, results = pyf.create_grid_from_data(
    S_base=100,
    AC_node_data=str(_DATA / "MATACDC_AC_node_data_Real.csv"),
    AC_line_data=str(_DATA / "MATACDC_AC_line_data_Real.csv"),
    DC_node_data=str(_DATA / "MATACDC_DC_node_data_Real.csv"),
    DC_line_data=str(_DATA / "MATACDC_DC_line_data_Real.csv"),
    Converter_data=str(_DATA / "MATACDC_Converter_data_Real.csv"),
    data_in="Real",
)
