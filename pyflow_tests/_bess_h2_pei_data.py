# -*- coding: utf-8 -*-
"""PEI BESS + H₂ validation helpers (Useche-Arteaga et al. 2026, Phase 6).

Wind CSV row order matches ``PEI_grid()`` ren-source order (PE I → II → III).
"""

from pathlib import Path

import pandas as pd
import pyflow_acdc as pyf

# PEI native base; Table 1 BESS expressed in physical MW/MWh (0.33 pu @ 3500 MVA).

BESS_P_NOM_MW = 0.33 * 3500.0
BESS_E_MAX_MWH = 3500.0
BESS_ETA_C = 0.85
BESS_ETA_D = 0.90
BESS_SOC_MIN = 0.1
BESS_SOC_MAX = 1.0
BESS_SOC_INITIAL = 0.5
BESS_SOC_FINAL = 0.5

H2_P_MAX_MW = 150.0
H2_P_MIN_MW = 22.5
H2_B_H = 16.0585
H2_C_H = 8.2195
H2_NE_MWH_PER_KG = 58e-3
H2_HOURS = 24

TURBINE_RATED_MW = 22.0
HUB_NODE = "PE_Island"
WINDOW_START = 0
WINDOW_END = 23

# Export buses (hourly market prices from Mario CSVs, EUR/MWh)
EXPORT_PRICE_NODES = {
    "BE_ON": "BE_Price.csv",
    "Na_AC_UK": "GB_Price.csv",
    "Tr_AC_DK": "DK_Price.csv",
}

PEI_OBJ_RULE = {"Energy_cost": 1}

PEI_DATA_DIR = Path(__file__).resolve().parents[2] / "mario_implementation" / "18414805"


def _pei_data_path(filename):
    path = PEI_DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"PEI data not found at {path}.")
    return path


def h2_mass_max_kg():
    return H2_P_MAX_MW * H2_HOURS / H2_NE_MWH_PER_KG


def h2_mass_final_kg():
    return 0.7 * H2_P_MAX_MW * H2_HOURS / H2_NE_MWH_PER_KG


def load_pei_power_matrix():
    """Return (160, 24) turbine power forecast in MW."""
    matrix = pd.read_csv(_pei_data_path("power_matrix.csv"), header=None).to_numpy(
        dtype=float
    )
    if matrix.shape != (160, 24):
        raise ValueError(f"power_matrix.csv must be 160×24 (got {matrix.shape})")
    return matrix


def load_pei_export_prices():
    """Return dict of 24 h export prices [EUR/MWh] keyed by export bus name."""
    prices = {}
    for node_name, csv_name in EXPORT_PRICE_NODES.items():
        series = pd.read_csv(_pei_data_path(csv_name))["Price"].to_numpy(dtype=float)
        if series.shape != (24,):
            raise ValueError(
                f"{csv_name} must contain 24 hourly prices (got {series.shape})"
            )
        prices[node_name] = series
    return prices


def _add_named_time_series(grid, hour_data, ts_type):
    names = list(hour_data.columns)
    header = pd.DataFrame([{name: name for name in names}])
    pyf.add_TimeSeries(
        grid,
        pd.concat([header, hour_data.reset_index(drop=True)], ignore_index=True),
        TS_type=ts_type,
    )


def attach_pei_export_prices(grid):
    """Attach hourly ``price`` time series at BE_ON, Na_AC_UK, and Tr_AC_DK."""
    ac_names = {node.name for node in grid.nodes_AC}
    export_nodes = [name for name in EXPORT_PRICE_NODES if name in ac_names]
    if not export_nodes:
        raise ValueError("No PEI export buses found on grid; expected at least BE_ON")

    prices = load_pei_export_prices()
    _add_named_time_series(
        grid,
        pd.DataFrame({node: prices[node] for node in export_nodes}),
        "price",
    )


def attach_pei_wind_time_series(grid):
    """Attach 24 h WPP availability series to all PEI ren sources."""
    power_mw = load_pei_power_matrix()
    names = [rs.name for rs in grid.RenSources]
    if len(names) != power_mw.shape[0]:
        raise ValueError(f"Expected {power_mw.shape[0]} ren sources, got {len(names)}")

    availability = power_mw / TURBINE_RATED_MW
    _add_named_time_series(grid, pd.DataFrame(availability.T, columns=names), "WPP")


def build_pei_bess_h2_grid(
    *,
    include_countries=None,
    attach_wind=True,
    attach_export_prices=True,
):
    """PEI grid with Table 1 BESS, electrolyser, and optional time series."""
    if include_countries is None:
        include_countries = ["UK", "DK"]
    grid, _ = pyf.cases["PEI_grid"](include_countries=include_countries)
    

    pyf.add_storage(
        grid,
        HUB_NODE,
        E_max_MWh=BESS_E_MAX_MWH,
        P_charge_MW=BESS_P_NOM_MW,
        P_discharge_MW=BESS_P_NOM_MW,
        eta_charge=BESS_ETA_C,
        eta_discharge=BESS_ETA_D,
        soc_min=BESS_SOC_MIN,
        soc_max=BESS_SOC_MAX,
        soc_initial=BESS_SOC_INITIAL,
        soc_final=BESS_SOC_FINAL,
    )
    pyf.add_electrolyser(
        grid,
        HUB_NODE,
        P_max_MW=H2_P_MAX_MW,
        P_min_MW=H2_P_MIN_MW,
        b_h=H2_B_H,
        c_h=H2_C_H,
        H2_mass_max_kg=h2_mass_max_kg(),
        H2_mass_initial_kg=0.0,
        H2_mass_final_kg=h2_mass_final_kg(),
    )

    if attach_export_prices:
        attach_pei_export_prices(grid)
    if attach_wind:
        attach_pei_wind_time_series(grid)

    return grid
