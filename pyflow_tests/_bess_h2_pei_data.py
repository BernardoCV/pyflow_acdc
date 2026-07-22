# -*- coding: utf-8 -*-
"""PEI BESS + H₂ validation helpers (Useche-Arteaga et al. 2026, Phase 6).

Wind CSV row order matches ``PEI_grid()`` ren-source order (PE I → II → III).
Seasonal 24 h windows live under ``examples/PEI_BESS/<Season>/``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pyflow_acdc as pyf

# Table 1 BESS in physical MW/MWh (0.33 pu @ 3500 MVA).
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
HOURS_PER_SEASON = 24

TURBINE_RATED_MW = 22.0
HUB_NODE = "PE_Island"
WINDOW_START = 0

# Seasonal folders under examples/PEI_BESS (default = Autumn = former root CSVs).
PEI_SEASONS = ("Spring", "Summer", "Autumn", "Winter")
DEFAULT_SEASONS = ("Autumn",)

# Export buses (hourly market prices, EUR/MWh)
EXPORT_PRICE_NODES = {
    "BE_ON": "BE_Price.csv",
    "Na_AC_UK": "GB_Price.csv",
    "Tr_AC_DK": "DK_Price.csv",
}

PEI_OBJ_RULE = {"Energy_cost": 1}

PEI_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "examples" / "PEI_BESS"
)


def normalize_pei_seasons(seasons=None):
    """Return a validated tuple of season folder names."""
    if seasons is None:
        seasons = DEFAULT_SEASONS
    if isinstance(seasons, str):
        seasons = (seasons,)
    out = tuple(seasons)
    if not out:
        raise ValueError("seasons must contain at least one season")
    unknown = [s for s in out if s not in PEI_SEASONS]
    if unknown:
        raise ValueError(
            f"Unknown PEI season(s) {unknown}; expected one of {list(PEI_SEASONS)}"
        )
    return out


def pei_hours(seasons=None):
    """Total hours for the selected seasonal windows."""
    return HOURS_PER_SEASON * len(normalize_pei_seasons(seasons))


def window_end(seasons=None):
    """Inclusive last frame index for ``window_nl_opf``."""
    return pei_hours(seasons) - 1


# Back-compat for single-season default (Autumn).
WINDOW_END = window_end(DEFAULT_SEASONS)
H2_HOURS = pei_hours(DEFAULT_SEASONS)


def _pei_season_path(season, filename):
    path = PEI_DATA_DIR / season / filename
    if not path.is_file():
        raise FileNotFoundError(f"PEI data not found at {path}.")
    return path


def h2_mass_max_kg(seasons=None):
    return H2_P_MAX_MW * pei_hours(seasons) / H2_NE_MWH_PER_KG


def h2_mass_final_kg(seasons=None):
    return 0.7 * H2_P_MAX_MW * pei_hours(seasons) / H2_NE_MWH_PER_KG


def load_pei_power_matrix(seasons=None):
    """Return ``(160, 24*n)`` turbine power forecast in MW for selected seasons."""
    seasons = normalize_pei_seasons(seasons)
    blocks = []
    for season in seasons:
        matrix = pd.read_csv(
            _pei_season_path(season, "power_matrix.csv"), header=None
        ).to_numpy(dtype=float)
        if matrix.shape != (160, HOURS_PER_SEASON):
            raise ValueError(
                f"{season}/power_matrix.csv must be 160×{HOURS_PER_SEASON} "
                f"(got {matrix.shape})"
            )
        blocks.append(matrix)
    return np.hstack(blocks)


def load_pei_export_prices(seasons=None):
    """Return dict of export prices [EUR/MWh] keyed by export bus name."""
    seasons = normalize_pei_seasons(seasons)
    n_hours = pei_hours(seasons)
    prices = {}
    for node_name, csv_name in EXPORT_PRICE_NODES.items():
        parts = []
        for season in seasons:
            series = pd.read_csv(_pei_season_path(season, csv_name))[
                "Price"
            ].to_numpy(dtype=float)
            if series.shape != (HOURS_PER_SEASON,):
                raise ValueError(
                    f"{season}/{csv_name} must contain {HOURS_PER_SEASON} "
                    f"hourly prices (got {series.shape})"
                )
            parts.append(series)
        prices[node_name] = np.concatenate(parts)
        if prices[node_name].shape != (n_hours,):
            raise ValueError(
                f"Concatenated {csv_name} length mismatch "
                f"(got {prices[node_name].shape}, expected {(n_hours,)})"
            )
    return prices


def _add_named_time_series(grid, hour_data, ts_type):
    names = list(hour_data.columns)
    header = pd.DataFrame([{name: name for name in names}])
    pyf.add_TimeSeries(
        grid,
        pd.concat([header, hour_data.reset_index(drop=True)], ignore_index=True),
        TS_type=ts_type,
    )


def attach_pei_export_prices(grid, seasons=None):
    """Attach hourly ``price`` time series at BE_ON, Na_AC_UK, and Tr_AC_DK."""
    ac_names = {node.name for node in grid.nodes_AC}
    export_nodes = [name for name in EXPORT_PRICE_NODES if name in ac_names]
    if not export_nodes:
        raise ValueError("No PEI export buses found on grid; expected at least BE_ON")

    prices = load_pei_export_prices(seasons=seasons)
    _add_named_time_series(
        grid,
        pd.DataFrame({node: prices[node] for node in export_nodes}),
        "price",
    )


def attach_pei_wind_time_series(grid, seasons=None):
    """Attach WPP availability series (concatenated seasonal windows) to ren sources."""
    power_mw = load_pei_power_matrix(seasons=seasons)
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
    seasons=None,
):
    """PEI grid with Table 1 BESS, electrolyser, and optional seasonal time series.

    Parameters
    ----------
    seasons : str or sequence of str, optional
        Seasonal windows to concatenate when attaching TS. Each name must be one
        of ``Spring``, ``Summer``, ``Autumn``, ``Winter``. Default ``("Autumn",)``.
        Pass e.g. ``("Spring", "Summer", "Autumn", "Winter")`` for all four.
    """
    seasons = normalize_pei_seasons(seasons)
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
        H2_mass_max_kg=h2_mass_max_kg(seasons),
        H2_mass_initial_kg=0.0,
        H2_mass_final_kg=h2_mass_final_kg(seasons),
    )

    if attach_export_prices:
        attach_pei_export_prices(grid, seasons=seasons)
    if attach_wind:
        attach_pei_wind_time_series(grid, seasons=seasons)

    return grid
