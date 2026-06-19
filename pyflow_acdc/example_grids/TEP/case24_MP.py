# -*- coding: utf-8 -*-
"""
IEEE RTS-96 case24 multi-period TEP example grid.

Investment CSVs live in ``examples/Case24_MP/`` (or GitHub raw URLs under the same
folder). Load the grid with ``pyf.cases["case24_MP"]()``.
"""

import os
from pathlib import Path

import pandas as pd
import pyflow_acdc as pyf

DEFAULT_OBJ_RULE = {"Energy_cost": 1}
DEFAULT_N_YEARS = 10

CASE24_MP_GITHUB_BASE = (
    "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/Case24_MP/"
)


def _is_url(path):
    text = str(path)
    return text.startswith("http://") or text.startswith("https://")


def _case24_mp_data_dir():
    data_dir = Path(__file__).resolve().parents[3] / "examples" / "Case24_MP"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Case24_MP example data directory not found: {data_dir}. "
            "Expected examples/Case24_MP/ at the pyflow_acdc repository root."
        )
    return data_dir


def resolve_example_path(filename, *, online=False):
    if _is_url(filename):
        return str(filename)
    if online:
        return CASE24_MP_GITHUB_BASE + Path(filename).name
    path = _case24_mp_data_dir() / filename
    if not path.exists():
        raise FileNotFoundError(f"Case24_MP example file not found: {path}")
    return str(path)


def example_data_url(filename):
    """GitHub raw URL for a Case24_MP CSV filename."""
    return CASE24_MP_GITHUB_BASE + Path(filename).name


def case24_MP():
    S_base=100

    nodes_AC_data = [
    {'Node_id': '1', 'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.08, 'Reactive_load': 0.22, 'x_coord': 26.194, 'y_coord': 0.869},
    {'Node_id': '2', 'type': 'PV', 'Voltage_0': 1.035, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 0.97, 'Reactive_load': 0.2, 'x_coord': 52.652, 'y_coord': 0.869},
    {'Node_id': '3', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.8, 'Reactive_load': 0.37, 'x_coord': 18.256, 'y_coord': 28.65},
    {'Node_id': '4', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 0.74, 'Reactive_load': 0.15, 'x_coord': 36.777, 'y_coord': 10.999},
    {'Node_id': '5', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 0.71, 'Reactive_load': 0.14, 'x_coord': 50.271, 'y_coord': 10.999},
    {'Node_id': '6', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.36, 'Reactive_load': 0.28, 'x_coord': 76.906, 'y_coord': 29.092},
    {'Node_id': '7', 'type': 'PV', 'Voltage_0': 1.025, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.25, 'Reactive_load': 0.25, 'x_coord': 69.321, 'y_coord': 0.869},
    {'Node_id': '8', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.71, 'Reactive_load': 0.35, 'x_coord': 76.906, 'y_coord': 10.999},
    {'Node_id': '9', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.75, 'Reactive_load': 0.36, 'x_coord': 40.847, 'y_coord': 28.65},
    {'Node_id': '10', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 1.95, 'Reactive_load': 0.4, 'x_coord': 62.808, 'y_coord': 28.65},
    {'Node_id': '11', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 40.847, 'y_coord': 39.035},
    {'Node_id': '12', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 62.808, 'y_coord': 39.035},
    {'Node_id': '13', 'type': 'Slack', 'Voltage_0': 1.02, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 2.65, 'Reactive_load': 0.54, 'x_coord': 69.857, 'y_coord': 58.555},
    {'Node_id': '14', 'type': 'PV', 'Voltage_0': 0.98, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 1.94, 'Reactive_load': 0.39, 'x_coord': 34.497, 'y_coord': 59.519},
    {'Node_id': '15', 'type': 'PV', 'Voltage_0': 1.014, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 3.17, 'Reactive_load': 0.64, 'x_coord': 15.027, 'y_coord': 59.076},
    {'Node_id': '16', 'type': 'PV', 'Voltage_0': 1.017, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 1.0, 'Reactive_load': 0.2, 'x_coord': 15.027, 'y_coord': 68.791},
    {'Node_id': '17', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 3.81, 'y_coord': 87.209},
    {'Node_id': '18', 'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 3.33, 'Reactive_load': 0.68, 'x_coord': 15.027, 'y_coord': 91.75},
    {'Node_id': '19', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 1.81, 'Reactive_load': 0.37, 'x_coord': 34.497, 'y_coord': 68.791},
    {'Node_id': '20', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 1.28, 'Reactive_load': 0.26, 'x_coord': 51.126, 'y_coord': 68.791},
    {'Node_id': '21', 'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 34.497, 'y_coord': 91.75},
    {'Node_id': '22', 'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 62.808, 'y_coord': 91.75},
    {'Node_id': '23', 'type': 'PV', 'Voltage_0': 1.05, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 62.484, 'y_coord': 78.201},
    {'Node_id': '24', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 18.256, 'y_coord': 39.035},

    {'Node_id': '25', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 1.56, 'Reactive_load': 0.0, 'x_coord': 3.81, 'y_coord': 0.869},    
    {'Node_id': '26', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 2.34, 'Reactive_load': 0.0, 'x_coord': 3.81, 'y_coord': 55},
    {'Node_id': '27', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 3.64, 'Reactive_load': 0.0, 'x_coord': 78.5, 'y_coord': 82},
    {'Node_id': '28', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 55.5, 'y_coord': 61.75},
    {'Node_id': '29', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 3.81, 'y_coord': 27.75},
    {'Node_id': '30', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 18.256, 'y_coord': 0.869},
    {'Node_id': '31', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 230.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 73, 'y_coord': 91.75},
    {'Node_id': '32', 'type': 'PQ', 'Voltage_0': 1.01, 'theta_0': 0.0, 'kV_base': 138.0, 'Power_load': 0.0, 'Reactive_load': 0.0, 'x_coord': 69.25, 'y_coord': 78.201}


    ]
    nodes_AC = pd.DataFrame(nodes_AC_data)


    lines_AC_data_base = [
            {'Line_id': 'B_1-2', 'fromNode': '1', 'toNode': '2', 'Length_km': 3.0, 'r': 0.0026, 'x': 0.0139,  'b': 0.4611, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_1-3', 'fromNode': '1', 'toNode': '3', 'Length_km': 55.0, 'r': 0.0546, 'x': 0.2112,  'b': 0.0572, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_1-5', 'fromNode': '1', 'toNode': '5', 'Length_km': 55.0, 'r': 0.0218, 'x': 0.0845,  'b': 0.0229, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_2-4', 'fromNode': '2', 'toNode': '4', 'Length_km': 22.0, 'r': 0.0328, 'x': 0.1267,  'b': 0.0343, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_2-6', 'fromNode': '2', 'toNode': '6', 'Length_km': 50.0, 'r': 0.0497, 'x': 0.192,  'b': 0.052, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_3-9', 'fromNode': '3', 'toNode': '9', 'Length_km': 31.0, 'r': 0.0308, 'x': 0.119,  'b': 0.0322, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_3-24', 'fromNode': '3', 'toNode': '24', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03},
            {'Line_id': 'B_4-9', 'fromNode': '4', 'toNode': '9', 'Length_km': 27.0, 'r': 0.0268, 'x': 0.1037,  'b': 0.0281, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_5-10', 'fromNode': '5', 'toNode': '10', 'Length_km': 23.0, 'r': 0.0228, 'x': 0.0883,  'b': 0.0239, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_6-10', 'fromNode': '6', 'toNode': '10', 'Length_km': 16.0, 'r': 0.0139, 'x': 0.0605,  'b': 2.459, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_7-8', 'fromNode': '7', 'toNode': '8', 'Length_km': 16.0, 'r': 0.0159, 'x': 0.0614,  'b': 0.0166, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_8-9', 'fromNode': '8', 'toNode': '9', 'Length_km': 43.0, 'r': 0.0427, 'x': 0.1651,  'b': 0.0447, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_8-10', 'fromNode': '8', 'toNode': '10', 'Length_km': 43.0, 'r': 0.0427, 'x': 0.1651,  'b': 0.0447, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
            {'Line_id': 'B_9-11', 'fromNode': '9', 'toNode': '11', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03},
            {'Line_id': 'B_9-12', 'fromNode': '9', 'toNode': '12', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.03},
            {'Line_id': 'B_10-11', 'fromNode': '10', 'toNode': '11', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
            {'Line_id': 'B_10-12', 'fromNode': '10', 'toNode': '12', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
            {'Line_id': 'B_11-13', 'fromNode': '11', 'toNode': '13', 'Length_km': 33.0, 'r': 0.0061, 'x': 0.0476,  'b': 0.0999, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_11-14', 'fromNode': '11', 'toNode': '14', 'Length_km': 29.0, 'r': 0.0054, 'x': 0.0418,  'b': 0.0879, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_12-13', 'fromNode': '12', 'toNode': '13', 'Length_km': 33.0, 'r': 0.0061, 'x': 0.0476,  'b': 0.0999, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_12-23', 'fromNode': '12', 'toNode': '23', 'Length_km': 67.0, 'r': 0.0124, 'x': 0.0966,  'b': 0.203, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_13-23', 'fromNode': '13', 'toNode': '23', 'Length_km': 60.0, 'r': 0.0111, 'x': 0.0865,  'b': 0.1818, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_14-16', 'fromNode': '14', 'toNode': '16', 'Length_km': 27.0, 'r': 0.005, 'x': 0.0389,  'b': 0.0818, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_15-16', 'fromNode': '15', 'toNode': '16', 'Length_km': 12.0, 'r': 0.0022, 'x': 0.0173,  'b': 0.0364, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_15-21', 'fromNode': '15', 'toNode': '21', 'Length_km': 34.0, 'r': 0.0063, 'x': 0.049,  'b': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            # {'Line_id': 'B_15-21', 'fromNode': '15', 'toNode': '21', 'Length_km': 34.0, 'r': 0.0063, 'x': 0.049,  'b': 0.103, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_15-24', 'fromNode': '15', 'toNode': '24', 'Length_km': 36.0, 'r': 0.0067, 'x': 0.0519,  'b': 0.1091, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_16-17', 'fromNode': '16', 'toNode': '17', 'Length_km': 18.0, 'r': 0.0033, 'x': 0.0259,  'b': 0.0545, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_16-19', 'fromNode': '16', 'toNode': '19', 'Length_km': 16.0, 'r': 0.003, 'x': 0.0231,  'b': 0.0485, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_17-18', 'fromNode': '17', 'toNode': '18', 'Length_km': 10.0, 'r': 0.0018, 'x': 0.0144,  'b': 0.0303, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_17-22', 'fromNode': '17', 'toNode': '22', 'Length_km': 73.0, 'r': 0.0135, 'x': 0.1053,  'b': 0.2212, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_18-21', 'fromNode': '18', 'toNode': '21', 'Length_km': 18.0, 'r': 0.0033, 'x': 0.0259,  'b': 0.0545, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            # {'Line_id': 'B_18-21', 'fromNode': '18', 'toNode': '21', 'Length_km': 18.0, 'r': 0.0033, 'x': 0.0259,  'b': 0.0545, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_19-20', 'fromNode': '19', 'toNode': '20', 'Length_km': 27.5, 'r': 0.0051, 'x': 0.0396,  'b': 0.0833, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            # {'Line_id': 'B_19-20', 'fromNode': '19', 'toNode': '20', 'Length_km': 27.5, 'r': 0.0051, 'x': 0.0396,  'b': 0.0833, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_20-23', 'fromNode': '20', 'toNode': '23', 'Length_km': 15.0, 'r': 0.0028, 'x': 0.0216,  'b': 0.0455, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            # {'Line_id': 'B_20-23', 'fromNode': '20', 'toNode': '23', 'Length_km': 15.0, 'r': 0.0028, 'x': 0.0216,  'b': 0.0455, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
            {'Line_id': 'B_21-22', 'fromNode': '21', 'toNode': '22', 'Length_km': 47.0, 'r': 0.0087, 'x': 0.0678,  'b': 0.1424, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1}
        ]


    lines_AC_data_additional = [
        # Previously existing additional lines (kept).
        {'Line_id': 'A_1-8', 'fromNode': '1', 'toNode': '8', 'Length_km': 33.1, 'r': 0.0328, 'x': 0.1344,  'b': 0.0, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_2-8', 'fromNode': '2', 'toNode': '8', 'Length_km': 33.1, 'r': 0.0328, 'x': 0.1267,  'b': 0.0, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_6-7', 'fromNode': '6', 'toNode': '7', 'Length_km': 50.2, 'r': 0.0497, 'x': 0.1920,  'b': 0.0, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_13-14', 'fromNode': '13', 'toNode': '14', 'Length_km': 30.8, 'r': 0.0057, 'x': 0.0447,  'b': 0.0, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_14-23', 'fromNode': '14', 'toNode': '23', 'Length_km': 43.2, 'r': 0.0080, 'x': 0.0620,  'b': 0.0, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_16-23', 'fromNode': '16', 'toNode': '23', 'Length_km': 56.7, 'r': 0.0105, 'x': 0.0822,  'b': 0.0, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_19-23', 'fromNode': '19', 'toNode': '23', 'Length_km': 42.1, 'r': 0.0078, 'x': 0.0606,  'b': 0.0, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},

        # Candidate lines from paper table (r/b placeholders).
        {'Line_id': 'A_1-4', 'fromNode': '1', 'toNode': '4', 'Length_km': 3.9, 'r': 0.0039, 'x': 0.015,  'b': 0.0157, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_1-30', 'fromNode': '1', 'toNode': '30', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
        {'Line_id': 'A_2-7', 'fromNode': '2', 'toNode': '7', 'Length_km': 5.5, 'r': 0.0054, 'x': 0.021,  'b': 0.0222, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_3-29', 'fromNode': '3', 'toNode': '29', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
        {'Line_id': 'A_3-30', 'fromNode': '3', 'toNode': '30', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
        {'Line_id': 'A_7-10', 'fromNode': '7', 'toNode': '10', 'Length_km': 5.2, 'r': 0.0052, 'x': 0.020,  'b': 0.0209, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_9-10', 'fromNode': '9', 'toNode': '10', 'Length_km': 4.2, 'r': 0.0042, 'x': 0.016,  'b': 0.0169, 'MVA_rating': 175.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_11-15', 'fromNode': '11', 'toNode': '15', 'Length_km': 15.3, 'r': 0.0028, 'x': 0.022,  'b': 0.0464, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_11-24', 'fromNode': '11', 'toNode': '24', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_11-28', 'fromNode': '11', 'toNode': '28', 'Length_km': 16.6, 'r': 0.0031, 'x': 0.024,  'b': 0.0503, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_13-28', 'fromNode': '13', 'toNode': '28', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_14-19', 'fromNode': '14', 'toNode': '19', 'Length_km': 11.8, 'r': 0.0022, 'x': 0.017,  'b': 0.0358, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_15-26', 'fromNode': '15', 'toNode': '26', 'Length_km': 11.8, 'r': 0.0022, 'x': 0.017,  'b': 0.0358, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_19-21', 'fromNode': '19', 'toNode': '21', 'Length_km': 9.7, 'r': 0.0018, 'x': 0.014,  'b': 0.0294, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_20-22', 'fromNode': '20', 'toNode': '22', 'Length_km': 9.7, 'r': 0.0018, 'x': 0.014,  'b': 0.0294, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_20-28', 'fromNode': '20', 'toNode': '28', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_22-31', 'fromNode': '22', 'toNode': '31', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_23-31', 'fromNode': '23', 'toNode': '31', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_32-23', 'fromNode': '32', 'toNode': '23', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
        {'Line_id': 'A_24-26', 'fromNode': '24', 'toNode': '26', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_25-29', 'fromNode': '25', 'toNode': '29', 'Length_km': 11.8, 'r': 0.0022, 'x': 0.017,  'b': 0.0358, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_25-30', 'fromNode': '25', 'toNode': '30', 'Length_km': 7.6, 'r': 0.0014, 'x': 0.011,  'b': 0.0230, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_26-29', 'fromNode': '26', 'toNode': '29', 'Length_km': 9.7, 'r': 0.0018, 'x': 0.014,  'b': 0.0294, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
        {'Line_id': 'A_27-31', 'fromNode': '27', 'toNode': '31', 'Length_km': 0.0, 'r': 0.0023, 'x': 0.0839,  'b': 0.0, 'MVA_rating': 400.0, 'kV_base': 230.0, 'm': 1.015},
        {'Line_id': 'A_27-32', 'fromNode': '27', 'toNode': '32', 'Length_km': 9.7, 'r': 0.0018, 'x': 0.014,  'b': 0.0294, 'MVA_rating': 500.0, 'kV_base': 138.0, 'm': 1},
        {'Line_id': 'A_29-30', 'fromNode': '29', 'toNode': '30', 'Length_km': 11.8, 'r': 0.0022, 'x': 0.017,  'b': 0.0358, 'MVA_rating': 500.0, 'kV_base': 230.0, 'm': 1},
    ]


    lines_AC = pd.DataFrame(lines_AC_data_base+lines_AC_data_additional)

    expandable_base = [
        {'Expandable elements': 'B_1-2',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':   3 * 10**6},
        {'Expandable elements': 'B_1-3',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  55 * 10**6},
        {'Expandable elements': 'B_1-5',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  22 * 10**6},
        {'Expandable elements': 'B_2-4',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  33 * 10**6},
        {'Expandable elements': 'B_2-6',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'B_3-9',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  31 * 10**6},
        {'Expandable elements': 'B_3-24', 'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'B_4-9',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  27 * 10**6},
        {'Expandable elements': 'B_5-10', 'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  23 * 10**6},
        {'Expandable elements': 'B_6-10', 'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  16 * 10**6},
        {'Expandable elements': 'B_7-8',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  16 * 10**6},
        {'Expandable elements': 'B_8-9',  'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  43 * 10**6},
        {'Expandable elements': 'B_8-10', 'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  43 * 10**6},
        {'Expandable elements': 'B_9-11', 'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'B_9-12', 'N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'B_10-11','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'B_10-12','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'B_11-13','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  66 * 10**6},
        {'Expandable elements': 'B_11-14','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  58 * 10**6},
        {'Expandable elements': 'B_12-13','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  66 * 10**6},
        {'Expandable elements': 'B_12-23','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 134 * 10**6},
        {'Expandable elements': 'B_13-23','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 120 * 10**6},
        {'Expandable elements': 'B_14-16','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  54 * 10**6},
        {'Expandable elements': 'B_15-16','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  24 * 10**6},
        {'Expandable elements': 'B_15-21','N_b': 2, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  68 * 10**6},
        {'Expandable elements': 'B_15-24','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  72 * 10**6},
        {'Expandable elements': 'B_16-17','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  36 * 10**6},
        {'Expandable elements': 'B_16-19','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  32 * 10**6},
        {'Expandable elements': 'B_17-18','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  20 * 10**6},
        {'Expandable elements': 'B_17-22','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 146 * 10**6},
        {'Expandable elements': 'B_18-21','N_b': 2, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  36 * 10**6},
        {'Expandable elements': 'B_19-20','N_b': 2, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  55 * 10**6},
        {'Expandable elements': 'B_20-23','N_b': 2, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  30 * 10**6},
        {'Expandable elements': 'B_21-22','N_b': 1, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  94 * 10**6},
    ]

    new_expandable = [
        {'Expandable elements': 'A_1-8',  'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  35 * 10**6},
        {'Expandable elements': 'A_2-8',  'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  33 * 10**6},
        {'Expandable elements': 'A_6-7',  'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  50 * 10**6},
        {'Expandable elements': 'A_13-14','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  62 * 10**6},
        {'Expandable elements': 'A_14-23','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  86 * 10**6},
        {'Expandable elements': 'A_16-23','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 114 * 10**6},
        {'Expandable elements': 'A_19-23','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  84 * 10**6}]
    new_expandable_2 =[
        {'Expandable elements': 'A_1-4',  'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  3.9 * 10**6},
        {'Expandable elements': 'A_1-30', 'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 50.0 * 10**6},
        {'Expandable elements': 'A_2-7',  'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  5.5 * 10**6},
        {'Expandable elements': 'A_3-29', 'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 50.0 * 10**6},
        {'Expandable elements': 'A_3-30', 'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  8.76 * 10**6},
        {'Expandable elements': 'A_7-10', 'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  5.2 * 10**6},
        {'Expandable elements': 'A_9-10', 'N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost':  4.2 * 10**6},
        {'Expandable elements': 'A_11-15','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 30.6 * 10**6},
        {'Expandable elements': 'A_11-24','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_11-28','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 33.2 * 10**6},
        {'Expandable elements': 'A_13-28','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_14-19','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 23.6 * 10**6},
        {'Expandable elements': 'A_15-26','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 23.6 * 10**6},
        {'Expandable elements': 'A_19-21','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 19.4 * 10**6},
        {'Expandable elements': 'A_20-22','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 19.4 * 10**6},
        {'Expandable elements': 'A_20-28','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_22-31','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_23-31','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_32-23','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 50.0 * 10**6},
        {'Expandable elements': 'A_24-26','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_25-29','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 23.6 * 10**6},
        {'Expandable elements': 'A_25-30','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 15.2 * 10**6},
        {'Expandable elements': 'A_26-29','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 19.4 * 10**6},
        {'Expandable elements': 'A_27-31','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 50.0 * 10**6},
        {'Expandable elements': 'A_27-32','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 19.4 * 10**6},
        {'Expandable elements': 'A_29-30','N_b': 0, 'N_max': 10, 'n_inv_max': 3, 'Life_time': 50, 'base_cost': 23.6 * 10**6},

        
    ]

    expandable_data = expandable_base + new_expandable+new_expandable_2

        
    # expandable_data = []




    grid,res = pyf.create_grid_from_data(S_base,nodes_AC,lines_AC,data_in='pu')



    def _fuel_from_mwmax(mwmax):
        mw = int(round(float(mwmax)))
        fuel_map = {
            0: 'shunt reactor',
            12: 'natural gas',
            20: 'natural gas',
            50: 'hydro',
            76: 'hard coal',
            100: 'natural gas',
            155: 'hard coal',
            197: 'natural gas',
            350: 'hard coal',
            400: 'nuclear',
        }
        if mw not in fuel_map:
            raise ValueError(f"Unsupported MWmax '{mwmax}' (rounded={mw}) in _fuel_from_mwmax.")
        return fuel_map[mw]

    def _cost_from_fuel(fuel_type, MWmax, MVArmax=None, MVArmin=None):
        if fuel_type is None:
            raise ValueError("fuel_type is required in _cost_from_fuel.")
        if MWmax is None and MVArmax is None and MVArmin is None:
            raise ValueError("At least one of MWmax/MVArmax/MVArmin is required in _cost_from_fuel.")

        normalized_type = str(fuel_type).strip().lower()
        cost_map = {
            'natural gas': 2271.0*1000,  # table: Natural Gas
            'hard coal': 6603.5*1000,  # table: Coal
            'hydro': 12149.5*1000,     # table: Hydropower
            'nuclear': 6028.0*1000,    # table: Nuclear
            'onshore wind': 1750.05*1000,    # table: onshore wind
            'offshore wind': 6887.5*1000,    # table: offshore wind
            'shunt reactor': 35226 ,    # table: Reactor
        }
        if normalized_type not in cost_map:
            raise ValueError(f"Unsupported fuel_type '{fuel_type}' in _cost_from_fuel.")
        if normalized_type == 'shunt reactor':
            # For shunt devices, size by reactive capability when available.
            qmax = float(MVArmax) if MVArmax is not None else 0.0
            qmin = float(MVArmin) if MVArmin is not None else 0.0
            reactive_size = max(abs(qmax), abs(qmin))
            if reactive_size <= 0 and MWmax is not None:
                reactive_size = abs(float(MWmax))
            return cost_map[normalized_type] * reactive_size
        return cost_map[normalized_type] * float(MWmax)


    TECH_LIFE_TIME_YEARS = {
        'natural gas': 30,
        'hard coal': 30,
        'hydro': 100,
        'nuclear': 60,
        'onshore wind': 30,
        'offshore wind': 30,
        'shunt reactor': 30,
    }

    INV_MAX_TYPE = {
        'natural gas': 3,
        'hard coal': 2,
        'hydro': 2,
        'nuclear': 1,
        'onshore wind': 5,
        'offshore wind': 5,
        'shunt reactor': 5,
    }

    def _invest_max_from_type(gen_type):
        return INV_MAX_TYPE[gen_type]

    def _tech_life_time_years(tech_name, default=30):
        normalized = str(tech_name).strip().lower() if tech_name is not None else ''
        return TECH_LIFE_TIME_YEARS.get(normalized, default)

    def add_gen_by_mwmax(grid, node, gen_name, **kwargs):
        if 'fuel_type' not in kwargs:
            if 'MWmax' not in kwargs:
                raise ValueError("add_gen_by_mwmax requires 'MWmax' when fuel_type is not provided.")
            kwargs['fuel_type'] = _fuel_from_mwmax(kwargs['MWmax'])
        if 'installation_cost' not in kwargs:
            if 'MWmax' not in kwargs:
                raise ValueError("add_gen_by_mwmax requires 'MWmax' when installation_cost is not provided.")
            kwargs['installation_cost'] = _cost_from_fuel(
                kwargs['fuel_type'],
                kwargs['MWmax'],
                kwargs.get('MVArmax'),
                kwargs.get('MVArmin'),
            )
        return pyf.add_gen(grid, node, gen_name, **kwargs)

    def add_additional_gen(grid, node, gen_name, **kwargs):
        fuel_type = kwargs.get('fuel_type')
        ren_type = kwargs.get('ren_type')
        if ren_type is None and fuel_type is None:
            raise ValueError("add_additional_gen requires either 'ren_type' or 'fuel_type'.")
        gen_type = ren_type if ren_type is not None else fuel_type
        size_mw = kwargs.get('MWmax', kwargs.get('base_MW', None))
        if size_mw is None:
            raise ValueError("add_additional_gen requires 'MWmax' or 'base_MW'.")
        installation_cost = kwargs.pop(
            'installation_cost',
            _cost_from_fuel(gen_type, size_mw, kwargs.get('MVArmax'), kwargs.get('MVArmin'))
        )
        renewable_types_l = {str(t).lower() for t in grid.renewable_types}
        if str(gen_type).lower() in renewable_types_l:
            base_mw = kwargs.pop('base_MW', kwargs.pop('MWmax', None))
            if base_mw is None:
                raise ValueError("Renewable generator requires 'base_MW' (or 'MWmax')")
            kwargs.pop('fuel_type', None)
            kwargs.setdefault('ren_type', gen_type)
            ren = pyf.add_RenSource(grid, node, base_mw, ren_source_name=gen_name, **kwargs)
            ren.base_cost = installation_cost
            return ren

        kwargs.pop('ren_type', None)
        kwargs.setdefault('fuel_type', gen_type)
        kwargs.setdefault('installation_cost', installation_cost)
        return pyf.add_gen(grid, node, gen_name, **kwargs)

    add_gen_by_mwmax(grid, '1', 'gen1', np_gen=2, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '1', 'gen2', np_gen=1, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '1', 'gen3', np_gen=2, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '1', 'gen4', np_gen=1, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '2', 'gen5', np_gen=2, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '2', 'gen6', np_gen=1, fc=400.6849,lf=130.0, qf=0.0, MWmax=20.0, MWmin=16.0, MVArmax=10.0, MVArmin=0.0, PsetMW=10.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '2', 'gen7', np_gen=2, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '2', 'gen8', np_gen=1, fc=212.3076,lf=16.0811, qf=0.014142, MWmax=76.0, MWmin=15.2, MVArmax=30.0, MVArmin=-25.0, PsetMW=76.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '7', 'gen9', np_gen=3, fc=781.521,lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '7', 'gen10', np_gen=1, fc=781.521,lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '7', 'gen11', np_gen=1, fc=781.521,lf=43.6615, qf=0.052672, MWmax=100.0, MWmin=25.0, MVArmax=60.0, MVArmin=0.0, PsetMW=80.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '13', 'gen12', np_gen=3, fc=832.7575,lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '13', 'gen13', np_gen=1, fc=832.7575,lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '13', 'gen14', np_gen=1, fc=832.7575,lf=48.5804, qf=0.00717, MWmax=197.0, MWmin=69.0, MVArmax=80.0, MVArmin=0.0, PsetMW=95.1, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '14', 'gen15', np_gen=1, fc=0.0,lf=0.0, qf=0.0, MWmax=0.0, MWmin=0.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=0.0, QsetMVA=35.3)
    add_gen_by_mwmax(grid, '15', 'gen16', np_gen=5, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '15', 'gen17', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '15', 'gen18', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '15', 'gen19', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '15', 'gen20', np_gen=1, fc=86.3852,lf=56.564, qf=0.328412, MWmax=12.0, MWmin=2.4, MVArmax=6.0, MVArmin=0.0, PsetMW=12.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '15', 'gen21', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '16', 'gen22', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '18', 'gen23', np_gen=1, fc=395.3749,lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '21', 'gen24', np_gen=1, fc=395.3749,lf=4.4231, qf=0.000213, MWmax=400.0, MWmin=100.0, MVArmax=200.0, MVArmin=-50.0, PsetMW=400.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '22', 'gen25', np_gen=6, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '22', 'gen26', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '22', 'gen27', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '22', 'gen28', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '22', 'gen29', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '22', 'gen30', np_gen=1, fc=0.001,lf=0.001, qf=0.0, MWmax=50.0, MWmin=10.0, MVArmax=16.0, MVArmin=-10.0, PsetMW=50.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '23', 'gen31', np_gen=2, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    #add_gen_by_mwmax(grid, '23', 'gen32', np_gen=1, fc=382.2391,lf=12.3883, qf=0.008342, MWmax=155.0, MWmin=54.29999999999999, MVArmax=80.0, MVArmin=-50.0, PsetMW=155.0, QsetMVA=0.0)
    add_gen_by_mwmax(grid, '23', 'gen33', np_gen=1, fc=665.1094,lf=11.8495, qf=0.004895, MWmax=350.0, MWmin=140.0, MVArmax=150.0, MVArmin=-25.0, PsetMW=350.0, QsetMVA=0.0)
        

    #(grid, node, base, ren_source_name=None, available=1, zone=None, price_zone=None, Offshore=False, MTDC=None, geometry=None, ren_type='Wind', min_gamma=0, Qrel=0,Qmin=None,Qmax=None):
        
    add_additional_gen(grid, '1', 'ren_gen1', base_MW=400.0,  ren_type='offshore wind')
    add_additional_gen(grid, '7', 'ren_gen2', base_MW=300.0,  ren_type='offshore wind')
    add_additional_gen(grid, '17', 'ren_gen3', base_MW=200.0,  ren_type='onshore wind')
    add_additional_gen(grid, '19', 'ren_gen4', base_MW=200.0,  ren_type='onshore wind')
    add_additional_gen(grid, '27', 'ren_gen5', base_MW=200.0,  ren_type='onshore wind')    
    add_additional_gen(grid, '29', 'ren_gen6', base_MW=200.0,  ren_type='offshore wind')    

    
    gen_tracking_data = []
    for gen in grid.Generators:
        expandable_data.append({
            'Expandable elements': gen.name,
            'N_b': gen.np_gen,
            'N_max': _invest_max_from_type(gen.gen_type) * 5,
            'Life_time': _tech_life_time_years(getattr(gen, 'gen_type', None)),
            'base_cost': gen.base_cost,
            'n_inv_max': _invest_max_from_type(gen.gen_type)
        })
        gen_tracking_data.append({
            'gen_name': gen.name,
            'gen_type': gen.gen_type,
            'gen_np_gen': gen.np_gen,
            'gen_np_gen_max': _invest_max_from_type(gen.gen_type) * 5,
            'gen_life_time': _tech_life_time_years(getattr(gen, 'gen_type', None)),
            'gen_base_cost': gen.base_cost,
            'gen_n_inv_max': _invest_max_from_type(gen.gen_type)
        })
        
    for ren in grid.RenSources:
        expandable_data.append({
            'Expandable elements': ren.name,
            'N_b': 0,
            'N_max': _invest_max_from_type(ren.rs_type) * 5,
            'Life_time': _tech_life_time_years(getattr(ren, 'rs_type', None)),
            'base_cost': ren.base_cost,
            'n_inv_max': _invest_max_from_type(ren.rs_type)
        })
        gen_tracking_data.append({
            'gen_name': ren.name,
            'gen_type': ren.rs_type,
            'gen_np_gen': 0,
            'gen_np_gen_max': _invest_max_from_type(ren.rs_type) * 5,
            'gen_life_time': _tech_life_time_years(getattr(ren, 'rs_type', None)),
            'gen_base_cost': ren.base_cost,
            'gen_n_inv_max': _invest_max_from_type(ren.rs_type)
        })

    
    

    expandable_data = pd.DataFrame(expandable_data)

    pyf.expand_elements_from_pd(grid,expandable_data)
    
    return grid,res


def resolve_local_path(base_file, maybe_relative_path):
    del base_file
    if _is_url(maybe_relative_path) or os.path.isabs(maybe_relative_path):
        return maybe_relative_path
    return resolve_example_path(maybe_relative_path)


def read_inv_series_csv(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[0] < 3:
        raise ValueError(f"Investment CSV '{csv_path}' has insufficient rows.")
    return df


def get_column_indices_by_type(df, type_name):
    return [idx for idx, t in enumerate(df.iloc[1, :].tolist()) if str(t).strip() == type_name]


def get_last_load_multiplier(df):
    data = df.iloc[2:, get_column_indices_by_type(df, "Load")].apply(pd.to_numeric, errors="coerce")
    if data.empty:
        raise ValueError("No numeric load rows found in investment CSV.")
    last_row = data.iloc[-1].dropna()
    if last_row.empty:
        raise ValueError("Last load row is empty/NaN in investment CSV.")
    return float(last_row.mean())


def apply_uniform_load_multiplier(grid, load_multiplier):
    for node in grid.nodes_AC:
        node.PLi_factor = float(load_multiplier)
    for node in getattr(grid, "nodes_DC", []):
        node.PLi_factor = float(load_multiplier)