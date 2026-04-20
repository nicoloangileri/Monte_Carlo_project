"""
data_loader.py — Mediterranean Quantitative Finance Society (MQFS)
===================================================================
Hybrid data ingestion for Sicilian agrometeorological time series.

Priority order
--------------
1.  Explicit csv_path argument (if provided and valid)
2.  data/weather_data.csv          ← real data (SIAS / ERA5 / NOAA)
3.  data/sicily_reference_synthetic.csv  ← calibrated synthetic (ships with repo)
4.  On-the-fly synthetic generator (last resort fallback)

 DATA TRANSPARENCY
     data/sicily_reference_synthetic.csv contains 12,418 SYNTHETIC daily
     observations (1990-01-01 → 2023-12-31). Parameters are calibrated on
     published SIAS Palermo-Boccadifalco climate normals (ISPRA/SIAS, 2021)
     but the values are NOT real measurements from a weather station.
     Run `python src/python/era5_fetcher.py --real` for genuine ERA5 data.

HOW TO SUPPLY REAL DATA
-----------------------
Option A — ERA5 (recommended):
    python src/python/era5_fetcher.py --real
    Requires a free Copernicus CDS account and ~/.cdsapirc key file.
    Downloads ~150 MB of ERA5 reanalysis for Sicily (1990–2023).

Option B — SIAS Sicilia:
    http://www.sias.regione.sicilia.it/ → Dati Storici → Palermo-Boccadifalco
    Export daily CSV and save as data/weather_data.csv.

Option C — NOAA ISD:
    https://www.ncei.noaa.gov/products/land-based-station
    Download station USAF 164010 (Palermo airport), export daily CSV.

Required CSV format:
    date,temperature_c,rainfall_mm
    1990-01-01,12.3,5.1
    1990-01-02,11.8,0.0

Author : MQFS Research Division
License: MIT
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date", "temperature_c", "rainfall_mm"}
CSV_DTYPES       = {"temperature_c": float, "rainfall_mm": float}

# Checked in priority order when no explicit path is given
_AUTO_PATHS = [
    "data/weather_data.csv",
    "data/sicily_reference_synthetic.csv",
]


# ---------------------------------------------------------------------------
# Synthetic generator configuration
# ---------------------------------------------------------------------------

@dataclass
class SyntheticWeatherConfig:
    """
    Parameters for the on-the-fly synthetic generator.
    Calibrated on SIAS Palermo-Boccadifalco 1990–2020 climate normals.
    ⚠  Produces SYNTHETIC data — not real measurements.
    """
    start_date: str = "2000-01-01"
    end_date:   str = "2023-12-31"

    # Temperature — Sicily annual cycle (°C)
    # Source: ISPRA/SIAS — Atlante Climatico della Sicilia (2021)
    temp_annual_mean:  float = 18.4   # Palermo long-run mean (°C)
    temp_seasonal_amp: float = 9.5    # half-amplitude; peak mid-July (day 196)
    temp_ar1_coef:     float = 0.74   # day-to-day persistence
    temp_noise_scale:  float = 2.7    # residual std-dev (°C)

    # Rainfall — Mediterranean dry-summer regime (mm/day)
    rain_annual_mean:  float = 2.16   # ≈ 790 mm/yr (Palermo coastal)
    rain_seasonal_amp: float = 1.9    # strong winter/summer contrast
    rain_ar1_coef:     float = 0.46
    rain_noise_scale:  float = 3.6

    # Student-t degrees of freedom for innovations
    t_dof_temp: float = 6.0
    t_dof_rain: float = 4.0   # fatter tails: drought spells cluster

    random_seed: int = 42


def _seasonal_temperature(doy: np.ndarray,
                           cfg: SyntheticWeatherConfig) -> np.ndarray:
    """
    Cosine seasonal cycle: maximum at mid-July (day 196).
    cos(0) = 1 → peak when phase = 0 → phase = 0 when doy = 196.
    Palermo hottest day: ~July 15 (day 196). Coldest: ~January 15 (day 15).
    """
    phase = 2.0 * np.pi * (doy - 196) / 365.25
    return cfg.temp_annual_mean + cfg.temp_seasonal_amp * np.cos(phase)


def _seasonal_rainfall(doy: np.ndarray,
                        cfg: SyntheticWeatherConfig) -> np.ndarray:
    """
    Cosine seasonal cycle: maximum near winter solstice (day 355).
    Mediterranean regime: wet Nov–Mar, dry Jun–Aug.
    """
    phase = 2.0 * np.pi * (doy - 355) / 365.25
    return np.maximum(
        cfg.rain_annual_mean + cfg.rain_seasonal_amp * np.cos(phase), 0.05
    )


def generate_synthetic_weather(
        cfg: Optional[SyntheticWeatherConfig] = None) -> pd.DataFrame:
    """
    Generate a synthetic daily weather series.
    ⚠  NOT real data. Used only as last-resort fallback.

    Returns pd.DataFrame with columns [date, temperature_c, rainfall_mm].
    """
    if cfg is None:
        cfg = SyntheticWeatherConfig()

    rng   = np.random.default_rng(cfg.random_seed)
    dates = pd.date_range(start=cfg.start_date, end=cfg.end_date, freq="D")
    n     = len(dates)
    doy   = dates.day_of_year.to_numpy(dtype=float)

    # Temperature: AR(1) around seasonal mean with Student-t innovations
    temp_seas = _seasonal_temperature(doy, cfg)
    innov_t   = student_t.rvs(df=cfg.t_dof_temp, size=n,
                               random_state=int(rng.integers(1_000_000_000)))
    innov_t  *= cfg.temp_noise_scale / np.std(innov_t)
    resid_t   = np.zeros(n)
    for i in range(1, n):
        resid_t[i] = cfg.temp_ar1_coef * resid_t[i - 1] + innov_t[i]
    temperature = temp_seas + resid_t

    # Rainfall: AR(1) with Student-t shocks, hard-clipped to ≥ 0
    rain_seas = _seasonal_rainfall(doy, cfg)
    innov_r   = student_t.rvs(df=cfg.t_dof_rain, size=n,
                               random_state=int(rng.integers(1_000_000_000)))
    innov_r  *= cfg.rain_noise_scale / np.std(innov_r)
    resid_r   = np.zeros(n)
    for i in range(1, n):
        resid_r[i] = cfg.rain_ar1_coef * resid_r[i - 1] + innov_r[i]
    rainfall = np.maximum(rain_seas + resid_r, 0.0)

    df = pd.DataFrame({
        "date":          dates,
        "temperature_c": temperature,
        "rainfall_mm":   rainfall,
    })
    log.warning(
        "⚠  ON-THE-FLY SYNTHETIC data: %d observations (%s → %s). "
        "NOT real measurements. Run era5_fetcher.py for real data.",
        n, cfg.start_date, cfg.end_date,
    )
    return df


# ---------------------------------------------------------------------------
# CSV loader (shared by real data and reference synthetic)
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    """
    Load, validate, and clean a weather CSV.

    Raises ValueError if required columns are missing or if
    more than 5 consecutive NaN days cannot be interpolated.
    """
    df = pd.read_csv(path, parse_dates=["date"], dtype=CSV_DTYPES)
    df.columns = df.columns.str.strip().str.lower()

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    df = df.sort_values("date").reset_index(drop=True)
    df["rainfall_mm"] = df["rainfall_mm"].clip(lower=0.0)

    # Interpolate gaps ≤ 5 days; fill any boundary NaNs
    df[["temperature_c", "rainfall_mm"]] = (
        df[["temperature_c", "rainfall_mm"]]
        .interpolate(method="linear", limit=5)
        .bfill()
        .ffill()
    )

    n_nan = df[["temperature_c", "rainfall_mm"]].isna().sum().sum()
    if n_nan > 0:
        raise ValueError(
            f"{path.name}: {n_nan} unresolvable NaN values. "
            "Check for consecutive gaps longer than 5 days."
        )
    return df


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def load_weather_data(
        csv_path: Optional[str | Path] = None,
) -> tuple[pd.DataFrame, str]:
    """
    Load weather data. Returns (DataFrame, source_label).

    source_label is one of:
        'real'                  — verified real data (weather_data.csv)
        'reference_synthetic'   — MQFS calibrated synthetic (ships with repo)
        'on_the_fly_synthetic'  — generated at runtime (last resort)

    Parameters
    ----------
    csv_path : explicit path to override auto-detection (optional)
    """
    candidates: list[tuple[str, Path]] = []
    if csv_path is not None:
        candidates.append(("explicit", Path(csv_path)))
    candidates += [("auto", Path(p)) for p in _AUTO_PATHS]

    for _, path in candidates:
        if not path.exists():
            continue
        try:
            df    = _load_csv(path)
            is_real = "synthetic" not in path.name.lower()
            label = "real" if is_real else "reference_synthetic"
            tag   = "✓ REAL DATA" if is_real else "⚠ REFERENCE SYNTHETIC"
            log.info(
                "%s: %d observations (%s → %s) — %s",
                tag, len(df),
                df["date"].iloc[0].date(),
                df["date"].iloc[-1].date(),
                path.name,
            )
            return df, label
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)

    # Last resort
    log.warning(
        "No valid CSV found. Falling back to ON-THE-FLY SYNTHETIC data. "
        "Run `python src/python/era5_fetcher.py` for real data."
    )
    return generate_synthetic_weather(), "on_the_fly_synthetic"


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, source = load_weather_data()
    print(f"\nSource  : {source}")
    print(f"Shape   : {df.shape}")
    print(df.head(5).to_string(index=False))
    print(df.describe().round(3))
