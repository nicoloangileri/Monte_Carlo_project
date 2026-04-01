"""
visualise.py — Mediterranean Quantitative Finance Society (MQFS)
=================================================================
Publication-ready figures (300 dpi, dark-background style).

Figures generated
-----------------
1. fan_chart.png        — MC rainfall paths with percentile confidence bands
2. ts_diagnostics.png   — 4-panel: temperature, rainfall, annual totals, Q-Q
3. cdi_distribution.png — CDI histogram + Wang risk-neutral density overlay

Author : MQFS Research Division
License: MIT
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde, probplot, norm

log    = logging.getLogger(__name__)
FIGDIR = Path("outputs")
FIGDIR.mkdir(exist_ok=True)
DPI = 300

_PAL = {
    "background":   "0D1117",
    "surface":      "161B22",
    "border":       "30363D",
    "text":         "E6EDF3",
    "muted":        "8B949E",
    "accent_blue":  "58A6FF",
    "accent_gold":  "D29922",
    "accent_red":   "F85149",
    "accent_green": "3FB950",
}


def _h(key: str) -> str:
    return "#" + _PAL[key]


def _apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor":  _h("background"),
        "axes.facecolor":    _h("surface"),
        "axes.edgecolor":    _h("border"),
        "axes.labelcolor":   _h("text"),
        "axes.titlecolor":   _h("text"),
        "xtick.color":       _h("muted"),
        "ytick.color":       _h("muted"),
        "text.color":        _h("text"),
        "grid.color":        _h("border"),
        "grid.linewidth":    0.5,
        "grid.linestyle":    "--",
        "legend.facecolor":  _h("surface"),
        "legend.edgecolor":  _h("border"),
        "legend.labelcolor": _h("text"),
        "font.family":       "monospace",
        "font.size":         9,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "figure.dpi":        DPI,
    })


def _load_pricing_result(pricing_json: str) -> Optional[dict]:
    """Load pricing result JSON if available; return None otherwise."""
    p = Path(pricing_json)
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


# ===========================================================================
# Figure 1 — Monte Carlo Fan Chart
# ===========================================================================

def plot_fan_chart(
        mc_paths_csv: str = "data/mc_paths.csv",
        contract_days: int = 120,
        drought_threshold_mm: float = 1.823,
        data_source: str = "synthetic",
        pricing_json: str = "data/pricing_result.json",
        save_path: Optional[str] = None,
) -> None:
    """
    Fan chart of simulated daily rainfall paths.
    Percentile bands: 10/25/50/75/90.
    Uses pivot_table for O(n) loading — no iterrows().
    Drought threshold auto-read from pricing_result.json when available.
    """
    _apply_style()

    # Override threshold from pricing result if available
    pr = _load_pricing_result(pricing_json)
    if pr is not None and "drought_threshold_mm" in pr:
        drought_threshold_mm = pr["drought_threshold_mm"]

    csv = Path(mc_paths_csv)
    if csv.exists():
        df     = pd.read_csv(csv)
        pivot  = df.pivot_table(
            index="path_id", columns="step",
            values="rainfall_mm", aggfunc="first",
        )
        matrix = pivot.values
        n_paths, n_steps = matrix.shape
        log.info("Fan chart: %d paths × %d steps.", n_paths, n_steps)
    else:
        log.warning("mc_paths.csv absent — generating illustrative chart.")
        n_paths, n_steps = 2_000, contract_days
        rng    = np.random.default_rng(42)
        x      = np.full(n_paths, drought_threshold_mm)
        matrix = np.zeros((n_paths, n_steps))
        for t in range(n_steps):
            x = np.maximum(
                x + 0.04 * (drought_threshold_mm - x)
                + 0.6 * rng.standard_normal(n_paths), 0.0,
            )
            matrix[:, t] = x

    days  = np.arange(n_steps)
    bands = np.percentile(matrix, [10, 25, 50, 75, 90], axis=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(_h("background"))

    ax.fill_between(days, bands[0], bands[4],
                    color="#1F4E79", alpha=0.22, label="10th–90th percentile")
    ax.fill_between(days, bands[1], bands[3],
                    color="#2E75B6", alpha=0.40, label="25th–75th percentile")
    ax.plot(days, bands[2],
            color=_h("accent_blue"), linewidth=2.0, label="Median", zorder=5)
    ax.axhline(
        drought_threshold_mm,
        color=_h("accent_red"), linestyle="--", linewidth=1.5,
        label=f"Drought threshold  R* = {drought_threshold_mm:.3f} mm/day",
    )
    ax.fill_between(days, 0, drought_threshold_mm,
                    color=_h("accent_red"), alpha=0.06, zorder=0)
    ax.annotate(
        "Drought risk zone",
        xy=(n_steps * 0.82, drought_threshold_mm * 0.42),
        color=_h("accent_red"), fontsize=8, style="italic",
    )

    ax.set_xlabel("Day of Growing Season  (May → Aug)")
    ax.set_ylabel("Daily Rainfall (mm)")
    ax.set_title(
        "Monte Carlo Fan Chart — Simulated Daily Rainfall\n"
        "Drought Micro-Insurance Model  |  MQFS",
        fontweight="bold",
    )
    ax.legend(loc="upper right", framealpha=0.8, fontsize=8)
    ax.grid(True, axis="y")
    ax.set_xlim(0, n_steps - 1)
    ax.set_ylim(bottom=0)

    src_label = (
        "✓ Real ERA5 data — Copernicus C3S / ECMWF (Hersbach et al., 2020)"
        if data_source == "real"
        else "⚠ Reference synthetic data — calibrated on SIAS Palermo climatology, NOT real measurements"
    )
    ax.text(0.01, 0.02, f"MQFS  |  {src_label}",
            transform=ax.transAxes, color=_h("muted"), fontsize=6)

    plt.tight_layout()
    out = save_path or str(FIGDIR / "fan_chart.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=_h("background"))
    plt.close()
    log.info("Fan chart → %s", out)


# ===========================================================================
# Figure 2 — Time-Series Diagnostics (4-panel)
# ===========================================================================

def plot_ts_diagnostics(
        df: pd.DataFrame,
        stl_components: Optional[dict] = None,
        data_source: str = "synthetic",
        save_path: Optional[str] = None,
) -> None:
    """4-panel: temperature, monthly rain, annual rain totals, Q-Q."""
    _apply_style()

    dates = pd.DatetimeIndex(df["date"])
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor(_h("background"))

    src_tag = (
        "Real ERA5 data — Copernicus C3S / ECMWF"
        if data_source == "real"
        else "⚠ Reference synthetic — calibrated on SIAS Palermo, NOT real measurements"
    )
    fig.suptitle(
        f"Weather Data Diagnostics  |  MQFS\n{src_tag}",
        fontsize=13, fontweight="bold", color=_h("text"), y=1.02,
    )

    # (a) Daily temperature with 365-day rolling mean
    ax = axes[0, 0]
    ax.plot(dates, df["temperature_c"],
            color=_h("accent_gold"), alpha=0.4, linewidth=0.5, label="Daily")
    ax.plot(dates, df["temperature_c"].rolling(365, center=True).mean(),
            color=_h("accent_red"), linewidth=1.5, label="365-day mean")
    ax.set_title("(a) Daily Temperature (°C)")
    ax.set_ylabel("°C")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y")

    # (b) Monthly rainfall totals
    ax = axes[0, 1]
    monthly = df.set_index("date")["rainfall_mm"].resample("ME").sum()
    ax.bar(monthly.index, monthly.values,
           color=_h("accent_blue"), alpha=0.75, width=20)
    ax.set_title("(b) Monthly Rainfall Total (mm)")
    ax.set_ylabel("mm")
    ax.grid(True, axis="y")

    # (c) Annual totals with 5-year rolling trend
    ax = axes[1, 0]
    tmp = df.copy()
    tmp["year"] = pd.DatetimeIndex(tmp["date"]).year
    yearly = tmp.groupby("year")["rainfall_mm"].sum()
    ax.bar(yearly.index, yearly.values, color=_h("accent_green"), alpha=0.80)
    trend = pd.Series(yearly.values, index=yearly.index).rolling(5, center=True).mean()
    ax.plot(yearly.index, trend.values,
            color=_h("accent_red"), linewidth=2, label="5-year trend")
    ax.axhline(yearly.mean(), color=_h("muted"), linestyle=":",
               label=f"Mean {yearly.mean():.0f} mm/yr")
    ax.set_title("(c) Annual Rainfall (mm)")
    ax.set_ylabel("mm")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y")

    # (d) Q-Q plot of temperature residuals vs Normal
    ax = axes[1, 1]
    if stl_components and "residual" in stl_components:
        resid = stl_components["residual"].dropna().values
    else:
        roll  = pd.Series(df["temperature_c"].values).rolling(365, center=True).mean()
        resid = df["temperature_c"].values - roll.bfill().ffill().values

    (osm, osr), (slope, intercept, r) = probplot(resid, dist="norm")
    ax.scatter(osm, osr, s=2, color=_h("accent_blue"), alpha=0.5, label="Residuals")
    line_x = np.array([osm[0], osm[-1]])
    ax.plot(line_x, slope * line_x + intercept,
            color=_h("accent_red"), linewidth=1.5, label="Normal reference")
    ax.set_title(f"(d) Q-Q: Temperature Residuals  (R²={r**2:.3f})")
    ax.set_xlabel("Theoretical Quantiles")
    ax.set_ylabel("Sample Quantiles")
    ax.legend(fontsize=7)
    ax.grid(True)

    plt.tight_layout()
    out = save_path or str(FIGDIR / "ts_diagnostics.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=_h("background"))
    plt.close()
    log.info("Diagnostics → %s", out)


# ===========================================================================
# Figure 3 — CDI Distribution + Risk-Neutral Density
# ===========================================================================

def plot_cdi_distribution(
        cdi_csv: str = "data/cdi_distribution.csv",
        strike: Optional[float] = None,       # auto-read from pricing JSON
        wang_lambda: float = 0.25,
        notional: float = 5_000.0,
        pricing_json: str = "data/pricing_result.json",
        save_path: Optional[str] = None,
) -> None:
    """
    Two-panel risk analysis figure.
    Left:  CDI histogram with physical and Wang-distorted densities.
    Right: Payoff distribution conditional on trigger.

    Strike is auto-read from pricing_result.json when available,
    ensuring the chart always reflects the data-calibrated value.
    """
    _apply_style()

    # Read strike and pricing summary from JSON
    pr = _load_pricing_result(pricing_json)
    price_str = ""
    if pr is not None:
        price_str = (
            f"Fair Value: €{pr['price_eur']:,.2f}  "
            f"|  P(trigger): {pr['probability_trigger']:.1%}"
        )
        if strike is None:
            strike = pr.get("strike_cdi", 1.0162)

    if strike is None:
        strike = 1.0162   # absolute fallback = historical 65th-pct CDI

    if Path(cdi_csv).exists():
        cdis = pd.read_csv(cdi_csv)["cdi"].values
    else:
        log.warning("CDI CSV absent — using illustrative data.")
        cdis = np.random.default_rng(99).gamma(shape=1.8, scale=0.58, size=50_000)

    payoffs = notional * np.maximum(0.0, cdis - strike)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(_h("background"))
    fig.suptitle(
        f"Drought Micro-Insurance — Risk Analysis  |  MQFS\n{price_str}",
        fontsize=12, fontweight="bold", color=_h("text"),
    )

    # ── Left: CDI histogram + densities ──────────────────────────────────
    ax    = axes[0]
    x_rng = np.linspace(max(cdis.min(), 0.0), np.percentile(cdis, 99.5), 400)

    ax.hist(cdis, bins=80, density=True,
            color=_h("accent_blue"), alpha=0.38, label="CDI histogram")

    kde_p = gaussian_kde(cdis, bw_method=0.12)
    ax.plot(x_rng, kde_p(x_rng),
            color=_h("accent_blue"), linewidth=2, label="Physical density ℙ")

    # Wang-distorted risk-neutral density via CDF distortion
    cdf_p = np.clip(
        np.cumsum(kde_p(x_rng)) * (x_rng[1] - x_rng[0]), 1e-8, 1.0 - 1e-8,
    )
    cdf_q = 1.0 - norm.cdf(norm.ppf(1.0 - cdf_p) + wang_lambda)
    pdf_q = np.maximum(np.gradient(cdf_q, x_rng), 0.0)
    ax.plot(x_rng, pdf_q,
            color=_h("accent_gold"), linewidth=2, linestyle="--",
            label=f"Risk-neutral density ℚ  (λ={wang_lambda})")

    y_top = kde_p(x_rng).max() * 1.15
    ax.axvline(strike, color=_h("accent_red"), linewidth=1.5, linestyle="--",
               label=f"Strike  K = {strike:.4f} mm/day")
    ax.fill_betweenx([0, y_top], strike, x_rng[-1],
                     color=_h("accent_red"), alpha=0.07)
    ax.set_ylim(0, y_top)
    ax.set_xlabel("Cumulative Drought Index  (mm/day)")
    ax.set_ylabel("Density")
    ax.set_title("CDI — Physical  vs  Risk-Neutral Measure")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y")

    # ── Right: Payoff distribution (conditional on trigger) ───────────────
    ax      = axes[1]
    nonzero = payoffs[payoffs > 0]
    p_trig  = len(nonzero) / max(len(payoffs), 1)

    if len(nonzero) > 100:
        ax.hist(nonzero, bins=60, density=True,
                color=_h("accent_green"), alpha=0.55, label="Payoff | triggered")
        kp = gaussian_kde(nonzero, bw_method=0.2)
        xp = np.linspace(nonzero.min(), nonzero.max(), 400)
        ax.plot(xp, kp(xp), color=_h("accent_green"), linewidth=2)

    ax.axvline(
        np.mean(payoffs), color=_h("accent_gold"),
        linewidth=1.5, linestyle=":",
        label=f"E[payoff] = €{np.mean(payoffs):,.0f}",
    )
    ax.set_xlabel("Derivative Payoff (€)")
    ax.set_ylabel("Density  (conditional on trigger)")
    ax.set_title(f"Payoff Distribution   [P(trigger) = {p_trig:.1%}]")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y")
    ax.xaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}")
    )

    plt.tight_layout()
    out = save_path or str(FIGDIR / "cdi_distribution.png")
    plt.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=_h("background"))
    plt.close()
    log.info("CDI distribution → %s", out)


# ===========================================================================
# Master entry-point
# ===========================================================================

def generate_all_figures(
        df: pd.DataFrame,
        stl_components: Optional[dict] = None,
        data_source: str = "synthetic",
        drought_threshold_mm: float = 1.823,
) -> None:
    """Generate and save all three publication figures."""
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    log.info("=== MQFS — Generating Publication Figures ===")
    plot_fan_chart(
        drought_threshold_mm=drought_threshold_mm,
        data_source=data_source,
    )
    plot_ts_diagnostics(df, stl_components=stl_components, data_source=data_source)
    plot_cdi_distribution()
    log.info("All figures → %s/", FIGDIR)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_weather_data
    df, source = load_weather_data()
    generate_all_figures(df, data_source=source)
