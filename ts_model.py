"""
ts_model.py — Mediterranean Quantitative Finance Society (MQFS)
================================================================
Time-series analysis pipeline for Sicilian agrometeorological data.

Pipeline
--------
1. STL decomposition    — trend / seasonal / residual isolation (robust=True)
2. ADF stationarity     — validates modelling assumptions
3. SARIMA(1,0,0)        — AR(1) on log-annual rainfall → OU parameters
4. GARCH(1,1) MLE       — Student-t log-likelihood via L-BFGS-B (3 restarts)
5. JSON export          — typed parameter contract for the C++ MC engine

GARCH estimation method
-----------------------
Maximum Likelihood Estimation with Student-t innovations.
Unconstrained reparameterisation ensures:
    ω > 0         via  ω = exp(p₀)
    α ∈ (0, 0.30) via  α = 0.30 · σ(p₁)
    β ∈ (0, 0.97−α) via β = (0.97−α) · σ(p₂)
    ν > 4         via  ν = exp(p₃) + 4

Three L-BFGS-B starts prevent local optima.
Moment-matching fallback is retained for degenerate samples only.

Author : MQFS Research Division
License: MIT
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.special import expit, gammaln
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter dataclasses — serialisable to JSON → C++ engine
# ---------------------------------------------------------------------------

@dataclass
class GarchParams:
    omega: float   # long-run variance intercept  ω > 0
    alpha: float   # ARCH coefficient             α ∈ (0, 0.30)
    beta:  float   # GARCH coefficient            β ∈ (0, 0.97−α)
    nu:    float   # Student-t degrees of freedom ν > 4

    @property
    def persistence(self) -> float:
        """α + β — must be < 1 for covariance stationarity."""
        return self.alpha + self.beta


@dataclass
class FittedParams:
    """
    All parameters exported to the Monte Carlo pricing engine.
    Each field maps 1-to-1 to a field in ModelParams (C++ struct).
    """
    temp_mu:               float   # °C/year trend slope (OLS on STL trend)
    temp_sigma:            float   # temperature residual std-dev (°C)
    rain_mu:               float   # log-space OU long-run mean
    rain_kappa:            float   # OU mean-reversion speed
    rain_sigma:            float   # log-space diffusion coefficient
    garch:                 GarchParams
    drought_threshold_mm:  float   # growing-season mean rainfall R* (mm/day)
    historical_strike_cdi: float   # 65th-pct annual CDI → P(trigger) ≈ 35%
    rho_temp_rain:         float   # temperature–rainfall shock correlation
    n_years:               int     # years of historical data used
    dt:                    float = 1.0 / 365.25


# ---------------------------------------------------------------------------
# ADF stationarity test
# ---------------------------------------------------------------------------

def check_stationarity(series: np.ndarray, name: str,
                        alpha: float = 0.05) -> bool:
    """Augmented Dickey-Fuller test. Logs result; returns True if stationary."""
    adf_stat, p_value, *_ = adfuller(series, autolag="AIC")
    is_stationary = p_value < alpha
    log.info(
        "[ADF] %-30s  stat=%8.4f  p=%.4f  → %s",
        name, adf_stat, p_value,
        "STATIONARY ✓" if is_stationary else "NON-STATIONARY ✗",
    )
    return is_stationary


# ---------------------------------------------------------------------------
# STL decomposition
# ---------------------------------------------------------------------------

def run_stl_decomposition(
        df: pd.DataFrame, period: int = 365) -> Dict[str, pd.Series]:
    """
    Robust STL decomposition on daily temperature.
    Returns dict with keys: 'trend', 'seasonal', 'residual'.
    """
    series = pd.Series(
        df["temperature_c"].values,
        index=pd.DatetimeIndex(df["date"]),
        dtype=float,
    )
    res = STL(series, period=period, robust=True).fit()
    return {
        "trend":    res.trend,
        "seasonal": res.seasonal,
        "residual": res.resid,
    }


def compute_annual_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily data to annual summaries for SARIMA fitting."""
    tmp = df.copy()
    tmp["year"] = pd.DatetimeIndex(tmp["date"]).year
    return (
        tmp.groupby("year")
        .agg(
            temp_mean  = ("temperature_c", "mean"),
            rain_total = ("rainfall_mm",   "sum"),
            rain_mean  = ("rainfall_mm",   "mean"),
        )
        .reset_index()
    )


# ---------------------------------------------------------------------------
# GARCH(1,1) variance recursion
# ---------------------------------------------------------------------------

def _garch_variance(omega: float, alpha: float,
                    beta: float, eps2: np.ndarray) -> np.ndarray:
    """
    GARCH(1,1) conditional variance sequence.
    Initialised at unconditional variance E[h] = ω / (1−α−β).
    """
    n           = len(eps2)
    persistence = alpha + beta
    h           = np.empty(n)
    h[0]        = omega / max(1.0 - persistence, 1e-6)
    for t in range(1, n):
        h[t] = omega + alpha * eps2[t - 1] + beta * h[t - 1]
    return np.maximum(h, 1e-12)


# ---------------------------------------------------------------------------
# GARCH MLE with Student-t innovations
# ---------------------------------------------------------------------------

def fit_garch_mle(residuals: np.ndarray) -> GarchParams:
    """
    Estimate GARCH(1,1) parameters by Maximum Likelihood.

    Student-t log-likelihood:
        L = Σ_t [ C(ν) − ½ log h_t − (ν+1)/2 · log(1 + ε_t²/(h_t(ν−2))) ]

    Reparameterisation guarantees constraint satisfaction without box bounds:
        ω = exp(p₀)
        α = 0.30 · σ(p₁)
        β = (0.97 − α) · σ(p₂)
        ν = exp(p₃) + 4

    Three starting points are tried; the global best is returned.
    Falls back to moment-matching if all optimisation attempts fail.
    """
    s2 = float(np.var(residuals))
    if s2 < 1e-12:
        log.warning("[GARCH MLE] Near-zero variance — using moment-matching.")
        return _fit_garch_moments(residuals)

    eps2 = residuals ** 2
    n    = len(residuals)

    def _neg_loglik(raw: np.ndarray) -> float:
        omega = np.exp(raw[0])
        alpha = 0.30 * float(expit(raw[1]))
        beta  = (0.97 - alpha) * float(expit(raw[2]))
        nu    = np.exp(raw[3]) + 4.0

        h     = _garch_variance(omega, alpha, beta, eps2)
        const = (gammaln(0.5 * (nu + 1))
                 - gammaln(0.5 * nu)
                 - 0.5 * np.log(np.pi * (nu - 2.0)))
        ll = (n * const
              - 0.5 * np.sum(np.log(h))
              - 0.5 * (nu + 1.0)
              * np.sum(np.log(1.0 + eps2 / (h * (nu - 2.0)))))
        return -float(ll)

    starts = [
        [np.log(s2 * 0.05),  0.0,  0.0, np.log(2.0)],
        [np.log(s2 * 0.01), -1.0,  1.5, np.log(2.0)],
        [np.log(s2 * 0.10), -2.0,  2.0, np.log(0.5)],
    ]

    best_result, best_nll = None, np.inf
    for x0 in starts:
        try:
            res = minimize(
                _neg_loglik, x0,
                method="L-BFGS-B",
                options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
            )
            if res.fun < best_nll:
                best_nll, best_result = res.fun, res
        except Exception:
            continue

    if best_result is None:
        log.warning("[GARCH MLE] All attempts failed — using moment-matching.")
        return _fit_garch_moments(residuals)

    raw   = best_result.x
    omega = float(np.exp(raw[0]))
    alpha = float(0.30 * expit(raw[1]))
    beta  = float((0.97 - alpha) * expit(raw[2]))
    nu    = float(np.exp(raw[3]) + 4.0)

    params = GarchParams(omega=omega, alpha=alpha, beta=beta, nu=nu)
    log.info(
        "[GARCH MLE] ω=%.2e  α=%.4f  β=%.4f  α+β=%.4f  ν=%.2f  "
        "nll=%.2f  converged=%s",
        omega, alpha, beta, params.persistence, nu,
        best_nll, best_result.success,
    )
    return params


def _fit_garch_moments(residuals: np.ndarray) -> GarchParams:
    """Private fallback: moment-matching estimate of GARCH(1,1)."""
    s2 = float(np.var(residuals))
    if s2 < 1e-12:
        return GarchParams(omega=1e-5, alpha=0.05, beta=0.90, nu=8.0)

    kurtosis    = max(float(np.mean(residuals ** 4) / s2 ** 2), 1.5)
    persistence = min(0.97, 1.0 - 1.0 / kurtosis)
    alpha       = float(np.clip(
        (kurtosis - 3.0) / max(kurtosis, 3.001) * (1.0 - persistence),
        0.03, 0.15,
    ))
    beta        = float(np.clip(persistence - alpha, 0.0, 0.94))
    omega       = float(s2 * (1.0 - alpha - beta))
    nu          = float(np.clip(6.0 / max(kurtosis - 3.0, 0.01) + 4.0, 4.5, 30.0))

    params = GarchParams(omega=omega, alpha=alpha, beta=beta, nu=nu)
    log.info(
        "[GARCH moments] ω=%.2e  α=%.4f  β=%.4f  α+β=%.4f  ν=%.2f",
        omega, alpha, beta, params.persistence, nu,
    )
    return params


# ---------------------------------------------------------------------------
# SARIMA(1,0,0) on log-annual rainfall → OU parameters
# ---------------------------------------------------------------------------

def fit_sarima_rainfall(annual: pd.DataFrame) -> Tuple[float, float, float]:
    """
    Fit ARIMA(1,0,0) on log-annual rainfall totals.

    Maps AR(1) parameters to Ornstein-Uhlenbeck notation:
        X_t = c + φ · X_{t-1} + ε_t
        μ = c / (1−φ),  κ = −log(φ),  σ = sqrt(σ²_ε)

    Uses result.sigma2 (stable across statsmodels versions).
    Returns (mu_log, kappa, sigma_log).
    """
    log_rain = np.log(annual["rain_total"].values)
    check_stationarity(log_rain, "log(annual_rain)")

    result = SARIMAX(log_rain, order=(1, 0, 0), trend="c").fit(
        disp=False, method="lbfgs"
    )
    phi   = float(np.clip(result.params[1], 0.01, 0.99))
    c     = float(result.params[0])
    mu    = c / (1.0 - phi)
    kappa = float(-np.log(phi))
    sigma = float(np.sqrt(max(result.sigma2, 1e-8)))

    log.info("[SARIMA] log-rain: μ=%.4f  κ=%.4f  σ=%.4f", mu, kappa, sigma)
    return mu, kappa, sigma


# ---------------------------------------------------------------------------
# Master fitting function
# ---------------------------------------------------------------------------

def fit_all(
        df: pd.DataFrame,
        output_path: str = "data/fitted_params.json",
        stl_components: Optional[Dict[str, pd.Series]] = None,
) -> Tuple[FittedParams, Dict[str, pd.Series]]:
    """
    Run the full fitting pipeline and export parameters to JSON.

    Returns (FittedParams, stl_components_dict).
    STL is computed here if not provided, avoiding redundant computation
    when called from run_pipeline.py.
    """
    log.info("=== MQFS Time-Series Fitting Pipeline ===")

    annual = compute_annual_aggregates(df)
    log.info("Annual aggregates: %d years.", len(annual))

    if stl_components is None:
        stl_components = run_stl_decomposition(df)

    temp_resid = stl_components["residual"].dropna().values
    temp_trend = stl_components["trend"].dropna().values

    slope, *_ = stats.linregress(np.arange(len(temp_trend)), temp_trend)
    temp_sigma = float(np.std(temp_resid))
    check_stationarity(temp_resid, "temp_residuals")
    log.info("[Temp] trend=%.5f °C/day (%.4f °C/yr)  σ=%.4f",
             slope, slope * 365.25, temp_sigma)

    rain_mu, rain_kappa, rain_sigma = fit_sarima_rainfall(annual)

    # De-seasonalised log-daily rainfall — O(n) vectorised groupby
    rain_log  = np.log(df["rainfall_mm"].clip(lower=0.01).values)
    doy       = df["date"].dt.dayofyear
    seas_mean = (
        pd.Series(rain_log, index=df.index)
        .groupby(doy)
        .transform("mean")
        .values
    )
    rain_resid   = rain_log - seas_mean
    garch_params = fit_garch_mle(rain_resid)

    # ── Drought threshold: growing-season mean rainfall ───────────────────
    # Using the growing-season mean as R* is both actuarially meaningful
    # (it represents the minimum daily input for crop maintenance) and
    # ensures the CDI distribution is well-spread for pricing.
    gs_mask     = df["date"].dt.month.isin([5, 6, 7, 8])
    drought_thr = float(df.loc[gs_mask, "rainfall_mm"].mean())
    log.info("[Drought] Growing-season mean R* = %.4f mm/day", drought_thr)

    # ── Historical strike: 65th-pct annual CDI → P(trigger) ≈ 35% ────────
    # Robust implementation: explicit copy + separate year column avoids
    # index alignment issues after groupby on filtered subsets.
    gs_df        = df[gs_mask].copy()
    gs_df["year"]= gs_df["date"].dt.year
    cdi_annual   = gs_df.groupby("year").apply(
        lambda x: float(np.maximum(0.0, drought_thr - x["rainfall_mm"]).mean())
    )
    hist_strike  = float(np.percentile(cdi_annual.values, 65))
    p_trigger    = float((cdi_annual > hist_strike).mean())
    log.info(
        "[Strike] 65th-pct CDI K = %.4f mm/day  (historical P(trigger) = %.1f%%)",
        hist_strike, p_trigger * 100,
    )

    # ── Shock correlation ─────────────────────────────────────────────────
    min_len   = min(len(temp_resid), len(rain_resid))
    rho, pval = stats.pearsonr(temp_resid[-min_len:], rain_resid[-min_len:])
    log.info("[Correlation] ρ(temp, rain) = %.4f  p = %.4f", rho, pval)

    n_years = int((df["date"].iloc[-1] - df["date"].iloc[0]).days / 365.25)

    params = FittedParams(
        temp_mu               = float(slope * 365.25),
        temp_sigma            = temp_sigma,
        rain_mu               = rain_mu,
        rain_kappa            = rain_kappa,
        rain_sigma            = rain_sigma,
        garch                 = garch_params,
        drought_threshold_mm  = drought_thr,
        historical_strike_cdi = hist_strike,
        rho_temp_rain         = float(rho),
        n_years               = n_years,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(asdict(params), f, indent=2)
    log.info("Parameters exported → %s", out.resolve())

    return params, stl_components


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_weather_data

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    df, source = load_weather_data()
    log.info("Data source: %s", source)
    fitted, _ = fit_all(df)
    print(json.dumps(asdict(fitted), indent=2))
