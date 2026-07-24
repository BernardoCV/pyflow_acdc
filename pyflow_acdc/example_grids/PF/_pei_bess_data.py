# -*- coding: utf-8 -*-
"""PEI BESS / H₂ / time-series data for :func:`PEI_grid` (Useche-Arteaga et al. 2026).

Seasonal 24 h windows: ``examples/PEI_BESS/<Season>/``.
Long series: ``examples/PEI_BESS/Full_data/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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
H2_MASS_MAX_KG = 60000.0
H2_MASS_FINAL_KG = 40000.0
# Seasonal CSV length and default rolling-window horizon.
WINDOW_HOURS = 24

TURBINE_RATED_MW = 22.0
HUB_NODE = "PE_Island"

PEI_SEASONS = ("Spring", "Summer", "Autumn", "Winter")
DEFAULT_SEASONS = ("Autumn",)

EXPORT_NODE_TO_ZONE = {
    "BE_ON": "Belgium",
    "Na_AC_GB": "Great Britain",
    "Tr_AC_DK": "Denmark",
}
EXPORT_PRICE_ZONES = {
    "Belgium": "BE_Price.csv",
    "Great Britain": "GB_Price.csv",
    "Denmark": "DK_Price.csv",
}
EXPORT_PRICE_NODES = {
    node: EXPORT_PRICE_ZONES[zone]
    for node, zone in EXPORT_NODE_TO_ZONE.items()
}

PEI_OBJ_RULE = {"Energy_cost": 1}

_REPO_ROOT = Path(__file__).resolve().parents[3]
PEI_DATA_DIR = _REPO_ROOT / "examples" / "PEI_BESS"
PEI_FULL_DATA_DIR = PEI_DATA_DIR / "Full_data"
PEI_BESS_GITHUB_BASE = (
    "https://raw.githubusercontent.com/CITCEA-UPC/pyflow_acdc/main/examples/PEI_BESS/"
)

FULL_MARKET_CSV = (
    "BE_GB_DK_market_2021-12-01_to_2023-11-29_UTC.csv"
)
FULL_WIND_CSV = "Turbine_power.csv"
FULL_ZONE_B_CG = {
    "Belgium": "BE_b_CG",
    "Great Britain": "GB_b_CG",
    "Denmark": "DK_b_CG",
}

DATA_SEASON_COMPARISON = "season_comparison"
DATA_FULL = "full"
PEI_DATA_MODES = (DATA_SEASON_COMPARISON, DATA_FULL)


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


def _pei_season_source(season, filename):
    path = PEI_DATA_DIR / season / filename
    if path.is_file():
        return path
    return f"{PEI_BESS_GITHUB_BASE}{season}/{filename}"


def load_pei_power_matrix(seasons=None):
    """Return ``(160, 24*n)`` turbine power forecast in MW for selected seasons."""
    seasons = normalize_pei_seasons(seasons)
    blocks = []
    for season in seasons:
        matrix = pd.read_csv(
            _pei_season_source(season, "power_matrix.csv"), header=None
        ).to_numpy(dtype=float)
        if matrix.shape != (160, WINDOW_HOURS):
            raise ValueError(
                f"{season}/power_matrix.csv must be 160×{WINDOW_HOURS} "
                f"(got {matrix.shape})"
            )
        blocks.append(matrix)
    return np.hstack(blocks)


def load_pei_export_prices(seasons=None, *, by_zone=True):
    """Return dict of export prices [EUR/MWh] for seasonal windows."""
    seasons = normalize_pei_seasons(seasons)
    n_hours = WINDOW_HOURS * len(seasons)
    prices = {}
    for zone_name, csv_name in EXPORT_PRICE_ZONES.items():
        parts = []
        for season in seasons:
            series = pd.read_csv(_pei_season_source(season, csv_name))[
                "Price"
            ].to_numpy(dtype=float)
            if series.shape != (WINDOW_HOURS,):
                raise ValueError(
                    f"{season}/{csv_name} must contain {WINDOW_HOURS} "
                    f"hourly prices (got {series.shape})"
                )
            parts.append(series)
        prices[zone_name] = np.concatenate(parts)
        if prices[zone_name].shape != (n_hours,):
            raise ValueError(
                f"Concatenated {csv_name} length mismatch "
                f"(got {prices[zone_name].shape}, expected {(n_hours,)})"
            )
    if by_zone:
        return prices
    return {
        node: prices[zone]
        for node, zone in EXPORT_NODE_TO_ZONE.items()
        if zone in prices
    }


def _add_named_time_series(grid, hour_data, ts_type):
    import pyflow_acdc as pyf

    names = list(hour_data.columns)
    header = pd.DataFrame([{name: name for name in names}])
    pyf.add_TimeSeries(
        grid,
        pd.concat([header, hour_data.reset_index(drop=True)], ignore_index=True),
        TS_type=ts_type,
    )


def attach_pei_export_prices(grid, seasons=None):
    """Attach seasonal ``b_CG`` series to PEI price zones."""
    zone_names = {pz.name for pz in grid.Price_Zones}
    export_zones = [name for name in EXPORT_PRICE_ZONES if name in zone_names]
    if not export_zones:
        raise ValueError(
            "No PEI price zones found on grid; expected at least Belgium"
        )
    prices = load_pei_export_prices(seasons=seasons, by_zone=True)
    _add_named_time_series(
        grid,
        pd.DataFrame({zone: prices[zone] for zone in export_zones}),
        "b_CG",
    )


def attach_pei_wind_time_series(grid, seasons=None):
    """Attach seasonal WPP availability to ren sources."""
    power_mw = load_pei_power_matrix(seasons=seasons)
    names = [rs.name for rs in grid.RenSources]
    if len(names) != power_mw.shape[0]:
        raise ValueError(f"Expected {power_mw.shape[0]} ren sources, got {len(names)}")
    availability = power_mw / TURBINE_RATED_MW
    _add_named_time_series(grid, pd.DataFrame(availability.T, columns=names), "WPP")


def _parse_utc(series: pd.Series) -> pd.DatetimeIndex:
    return pd.to_datetime(
        series.astype(str).str.replace(" UTC", "", regex=False),
        utc=True,
    )


def load_pei_full_frames(ts_start=None, ts_end=None):
    """Load Full_data market + wind; optional UTC ``[ts_start, ts_end)`` slice.

    Returns
    -------
    timestamps : ndarray
    b_cg : DataFrame
    wind_avail : DataFrame
    """
    market_path = PEI_FULL_DATA_DIR / FULL_MARKET_CSV
    wind_path = PEI_FULL_DATA_DIR / FULL_WIND_CSV
    if not market_path.is_file():
        raise FileNotFoundError(market_path)
    if not wind_path.is_file():
        raise FileNotFoundError(wind_path)

    market = pd.read_csv(market_path)
    m_ts = _parse_utc(market.iloc[2:, 0])
    m_data = market.iloc[2:].copy()
    m_data.index = m_ts
    m_data = m_data.drop(columns=[market.columns[0]])

    wind = pd.read_csv(wind_path)
    w_ts = _parse_utc(wind["timestamp"])
    w_mw = wind.drop(columns=["timestamp"])
    w_mw.index = w_ts

    if ts_start is not None or ts_end is not None:
        start = pd.Timestamp(ts_start, tz="UTC") if ts_start is not None else m_ts.min()
        end = pd.Timestamp(ts_end, tz="UTC") if ts_end is not None else (
            m_ts.max() + pd.Timedelta(hours=1)
        )
        mask = (m_ts >= start) & (m_ts < end)
        if not mask.any():
            raise ValueError(f"No Full_data rows in [{start}, {end})")
        m_ts = m_ts[mask]
        m_data = m_data.loc[m_ts]
    else:
        m_data = m_data.loc[m_ts]

    w_mw = w_mw.reindex(m_ts)
    if w_mw.isna().any().any():
        raise ValueError("Wind series has gaps vs market timestamps in selected range")

    b_cg = pd.DataFrame(
        {
            zone: pd.to_numeric(m_data[col], errors="raise")
            for zone, col in FULL_ZONE_B_CG.items()
        }
    )
    avail = w_mw.astype(float) / TURBINE_RATED_MW
    return m_ts.to_numpy(), b_cg, avail


def attach_pei_full_time_series(grid, ts_start=None, ts_end=None):
    """Attach Full_data ``b_CG`` + WPP and set ``grid.ts_timestamps``."""
    timestamps, b_cg, wind_avail = load_pei_full_frames(ts_start, ts_end)
    zone_names = {pz.name for pz in grid.Price_Zones}
    b_cg = b_cg[[z for z in b_cg.columns if z in zone_names]]
    if b_cg.empty:
        raise ValueError("No matching price zones on grid for Full_data b_CG")
    _add_named_time_series(grid, b_cg, "b_CG")

    rs_names = [rs.name for rs in grid.RenSources]
    missing = [n for n in rs_names if n not in wind_avail.columns]
    if missing:
        raise ValueError(f"Wind CSV missing ren sources: {missing[:5]}…")
    _add_named_time_series(grid, wind_avail.loc[:, rs_names], "WPP")

    grid.ts_timestamps = timestamps
    if len(grid.Time_series[0].data) != len(timestamps):
        raise ValueError(
            f"ts_timestamps length {len(timestamps)} != "
            f"Time_series length {len(grid.Time_series[0].data)}"
        )
    return timestamps
