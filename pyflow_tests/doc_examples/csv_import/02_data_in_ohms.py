"""Docs: api/csv_import.rst — Data in Ohms"""
from pathlib import Path

import pyflow_acdc as pyf

_DATA = Path(__file__).resolve().parent / "data"


grid, results = pyf.create_grid_from_data(
    S_base=100,
    AC_node_data=str(_DATA / "AC_node_data_Ohm.csv")    ,
    AC_line_data=str(_DATA / "AC_line_data_Ohm.csv"),
    DC_node_data=str(_DATA / "DC_node_data_Ohm.csv"),
    DC_line_data=str(_DATA / "DC_line_data_Ohm.csv"),
    Converter_data=str(_DATA / "Converter_data_Ohm.csv"),
    data_in="Ohm",
)
