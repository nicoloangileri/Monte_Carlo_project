"""
tests/test_pipeline.py — Mediterranean Quantitative Finance Society (MQFS)
===========================================================================
Comprehensive unit and integration test suite for the MQFS pipeline.

Coverage
--------
TestDataLoader      — ingestion, validation, synthetic generator
TestSeasonality     — CRITICAL regression tests for seasonal direction
TestTsModel         — STL, SARIMA, annual aggregates, fit_all integration
TestGarchMLE        — MLE convergence, parameter bounds, fallback path
TestNumerical       — edge cases, long series, near-zero variance

Run
---
    pytest tests/ -v
    pytest tests/ -v --cov=src/python --cov-report=term-missing

Author : MQFS Research Division
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make src/python importable from any working directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "python"))

from data_loader import (
    SyntheticWeatherConfig,
    _load_csv,
    generate_synthetic_weather,
    load_weather_data,
)
from ts_model import (
    GarchParams,
    _fit_garch_moments,
    check_stationarity,
    compute_annual_aggregates,
    fit_all,
    fit_garch_mle,
    run_stl_decomposition,
)


# ===========================================================================
# Module-scoped fixtures (expensive operations run once)
# ===========================================================================

@pytest.fixture(scope="module")
def sample_df() -> pd.DataFrame:
    """24-year synthetic DataFrame."""
    return generate_synthetic_weather()


@pytest.fixture(scope="module")
def fitted_result(sample_df):
    """Full fitting pipeline result — runs once per test session."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        params, stl = fit_all(sample_df, output_path=tmp)
        yield params, stl, tmp
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@pytest.fixture(scope="module")
def garch_residuals(sample_df) -> np.ndarray:
    """De-seasonalised log-daily rainfall residuals for GARCH tests."""
    rain_log  = np.log(sample_df["rainfall_mm"].clip(lower=0.01).values)
    doy       = sample_df["date"].dt.dayofyear
    seas_mean = (
        pd.Series(rain_log, index=sample_df.index)
        .groupby(doy)
        .transform("mean")
        .values
    )
    return rain_log - seas_mean


# ===========================================================================
# TestDataLoader
# ===========================================================================

class TestDataLoader:

    def test_returns_correct_columns(self, sample_df):
        assert set(sample_df.columns) == {"date", "temperature_c", "rainfall_mm"}

    def test_no_negative_rainfall(self, sample_df):
        assert (sample_df["rainfall_mm"] >= 0.0).all(), \
            "Rainfall must be non-negative everywhere"

    def test_temperature_physical_range(self, sample_df):
        assert sample_df["temperature_c"].between(-10.0, 50.0).all(), \
            "Temperature outside physical range [−10, 50]°C for Sicily"

    def test_no_nan_values(self, sample_df):
        assert not sample_df[["temperature_c", "rainfall_mm"]].isna().any().any()

    def test_leap_year_length(self):
        cfg = SyntheticWeatherConfig(start_date="2000-01-01", end_date="2000-12-31")
        df  = generate_synthetic_weather(cfg)
        assert len(df) == 366, "2000 is a leap year — expect 366 rows"

    def test_non_leap_year_length(self):
        cfg = SyntheticWeatherConfig(start_date="2001-01-01", end_date="2001-12-31")
        df  = generate_synthetic_weather(cfg)
        assert len(df) == 365

    def test_reproducibility(self):
        df1 = generate_synthetic_weather()
        df2 = generate_synthetic_weather()
        pd.testing.assert_frame_equal(df1, df2, check_exact=True)

    def test_date_column_dtype(self, sample_df):
        assert pd.api.types.is_datetime64_any_dtype(sample_df["date"])

    def test_load_weather_data_missing_path(self):
        df, source = load_weather_data("nonexistent_xyz_12345.csv")
        assert source in ("reference_synthetic", "on_the_fly_synthetic")
        assert len(df) >= 365

    def test_load_csv_rejects_missing_columns(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text("date,temp\n2000-01-01,12.0\n")
        with pytest.raises(ValueError, match="missing required columns"):
            _load_csv(bad)

    def test_load_csv_clips_negative_rainfall(self, tmp_path):
        csv = tmp_path / "neg.csv"
        csv.write_text(
            "date,temperature_c,rainfall_mm\n"
            "2000-01-01,12.0,-5.0\n"
            "2000-01-02,13.0,2.0\n"
        )
        df = _load_csv(csv)
        assert (df["rainfall_mm"] >= 0).all()

    def test_load_csv_sorts_by_date(self, tmp_path):
        csv = tmp_path / "unsorted.csv"
        csv.write_text(
            "date,temperature_c,rainfall_mm\n"
            "2000-01-03,14.0,1.0\n"
            "2000-01-01,12.0,0.0\n"
            "2000-01-02,13.0,2.0\n"
        )
        df = _load_csv(csv)
        assert list(df["date"].dt.day) == [1, 2, 3]


# ===========================================================================
# TestSeasonality — CRITICAL regression suite
# ===========================================================================

class TestSeasonality:
    """
    These tests specifically guard against the inverted-seasonality bug
    (cos phase offset doy-15 instead of doy-196 for temperature peak)
    that was found and fixed during code review.
    Failure here means something fundamental has been broken.
    """

    def test_hottest_month_is_july_or_august(self, sample_df):
        df2     = sample_df.copy()
        df2["month"] = pd.DatetimeIndex(df2["date"]).month
        hottest = df2.groupby("month")["temperature_c"].mean().idxmax()
        assert hottest in [7, 8], (
            f"REGRESSION: hottest month = {hottest}, expected 7 (July) or 8. "
            "Check _seasonal_temperature uses phase offset (doy - 196)."
        )

    def test_coldest_month_is_january_or_february(self, sample_df):
        df2     = sample_df.copy()
        df2["month"] = pd.DatetimeIndex(df2["date"]).month
        coldest = df2.groupby("month")["temperature_c"].mean().idxmin()
        assert coldest in [1, 2], (
            f"REGRESSION: coldest month = {coldest}, expected 1 or 2."
        )

    def test_summer_mean_exceeds_winter_mean(self, sample_df):
        df2 = sample_df.copy()
        df2["month"] = pd.DatetimeIndex(df2["date"]).month
        summer = df2[df2["month"].isin([6, 7, 8])]["temperature_c"].mean()
        winter = df2[df2["month"].isin([12, 1, 2])]["temperature_c"].mean()
        assert summer > winter, (
            f"REGRESSION: summer mean ({summer:.2f}°C) ≤ winter ({winter:.2f}°C)"
        )

    def test_temperature_peak_near_day_196(self):
        """Annual maximum should fall in July (days 182–212)."""
        cfg = SyntheticWeatherConfig(start_date="2010-01-01", end_date="2010-12-31")
        df  = generate_synthetic_weather(cfg)
        df["doy"] = pd.DatetimeIndex(df["date"]).day_of_year
        monthly = df.copy()
        monthly["month"] = df["date"].dt.month
        monthly_means = monthly.groupby("month")["temperature_c"].mean()
        # July (month 7) should be hottest
        assert monthly_means.idxmax() in [7, 8]

    def test_wettest_month_is_nov_through_feb(self, sample_df):
        df2     = sample_df.copy()
        df2["month"] = pd.DatetimeIndex(df2["date"]).month
        wettest = df2.groupby("month")["rainfall_mm"].mean().idxmax()
        assert wettest in [11, 12, 1, 2], (
            f"Mediterranean regime error: wettest month = {wettest}, "
            "expected Nov–Feb."
        )

    def test_driest_month_is_summer(self, sample_df):
        df2     = sample_df.copy()
        df2["month"] = pd.DatetimeIndex(df2["date"]).month
        driest  = df2.groupby("month")["rainfall_mm"].mean().idxmin()
        assert driest in [6, 7, 8], (
            f"Mediterranean regime error: driest month = {driest}, "
            "expected Jun–Aug."
        )


# ===========================================================================
# TestTsModel
# ===========================================================================

class TestTsModel:

    def test_stl_returns_three_components(self, sample_df):
        stl = run_stl_decomposition(sample_df)
        assert set(stl.keys()) == {"trend", "seasonal", "residual"}

    def test_stl_residuals_same_length_as_input(self, sample_df):
        stl = run_stl_decomposition(sample_df)
        assert len(stl["residual"]) == len(sample_df)

    def test_stationarity_returns_bool(self, sample_df):
        stl    = run_stl_decomposition(sample_df)
        resid  = stl["residual"].dropna().values
        result = check_stationarity(resid, "test_series")
        assert isinstance(result, bool)

    def test_annual_aggregates_year_count(self, sample_df):
        annual = compute_annual_aggregates(sample_df)
        n_years = (
            pd.DatetimeIndex(sample_df["date"]).year.max()
            - pd.DatetimeIndex(sample_df["date"]).year.min()
            + 1
        )
        assert len(annual) == n_years

    def test_annual_aggregates_non_negative_rain(self, sample_df):
        annual = compute_annual_aggregates(sample_df)
        assert (annual["rain_total"] >= 0).all()

    def test_fit_all_returns_tuple(self, fitted_result):
        params, stl, _ = fitted_result
        assert stl is not None
        assert "residual" in stl

    def test_fit_all_json_contains_required_keys(self, fitted_result):
        _, _, tmp = fitted_result
        with open(tmp) as f:
            data = json.load(f)
        required = {
            "temp_mu", "temp_sigma", "rain_mu", "rain_kappa", "rain_sigma",
            "garch", "drought_threshold_mm", "historical_strike_cdi",
            "rho_temp_rain", "n_years",
        }
        for key in required:
            assert key in data, f"Missing key in fitted_params.json: {key}"

    def test_fit_all_drought_threshold_positive(self, fitted_result):
        params, _, _ = fitted_result
        assert params.drought_threshold_mm > 0.0

    def test_fit_all_strike_positive(self, fitted_result):
        params, _, _ = fitted_result
        assert params.historical_strike_cdi > 0.0

    def test_fit_all_strike_less_than_threshold(self, fitted_result):
        """Strike CDI must be less than threshold (otherwise P(trigger)→100%)."""
        params, _, _ = fitted_result
        assert params.historical_strike_cdi < params.drought_threshold_mm, (
            f"Strike {params.historical_strike_cdi:.4f} ≥ "
            f"threshold {params.drought_threshold_mm:.4f}"
        )

    def test_fit_all_rho_in_unit_interval(self, fitted_result):
        params, _, _ = fitted_result
        assert -1.0 < params.rho_temp_rain < 1.0

    def test_fit_all_n_years_positive(self, fitted_result):
        params, _, _ = fitted_result
        assert params.n_years > 0


# ===========================================================================
# TestGarchMLE
# ===========================================================================

class TestGarchMLE:

    def test_mle_returns_garch_params_type(self, garch_residuals):
        assert isinstance(fit_garch_mle(garch_residuals), GarchParams)

    def test_mle_omega_positive(self, garch_residuals):
        assert fit_garch_mle(garch_residuals).omega > 0.0

    def test_mle_alpha_in_range(self, garch_residuals):
        alpha = fit_garch_mle(garch_residuals).alpha
        assert 0.0 < alpha < 0.30, f"α = {alpha:.4f} out of (0, 0.30)"

    def test_mle_beta_positive(self, garch_residuals):
        assert fit_garch_mle(garch_residuals).beta > 0.0

    def test_mle_covariance_stationary(self, garch_residuals):
        """α + β < 1 is required for covariance stationarity."""
        p = fit_garch_mle(garch_residuals)
        assert p.persistence < 1.0, (
            f"GARCH non-stationary: α+β = {p.persistence:.6f} ≥ 1"
        )

    def test_mle_nu_above_four(self, garch_residuals):
        """ν > 4 required for finite kurtosis of Student-t distribution."""
        assert fit_garch_mle(garch_residuals).nu > 4.0

    def test_mle_better_than_moment_matching(self, garch_residuals):
        """
        MLE log-likelihood should be ≥ that of moment-matching initialisation.
        This confirms the optimiser actually improves on the starting point.
        """
        from scipy.special import gammaln
        from ts_model import _garch_variance

        def loglik(p: GarchParams) -> float:
            eps2  = garch_residuals ** 2
            h     = _garch_variance(p.omega, p.alpha, p.beta, eps2)
            n     = len(eps2)
            nu    = p.nu
            const = (gammaln(0.5*(nu+1)) - gammaln(0.5*nu)
                     - 0.5*np.log(np.pi*(nu-2)))
            return float(n * const
                         - 0.5 * np.sum(np.log(h))
                         - 0.5 * (nu+1) * np.sum(np.log(1 + eps2/(h*(nu-2)))))

        p_mle  = fit_garch_mle(garch_residuals)
        p_mm   = _fit_garch_moments(garch_residuals)
        ll_mle = loglik(p_mle)
        ll_mm  = loglik(p_mm)
        assert ll_mle >= ll_mm - 1.0, (
            f"MLE log-lik ({ll_mle:.2f}) worse than moment-matching ({ll_mm:.2f})"
        )

    def test_moment_matching_fallback_stationary(self):
        rng       = np.random.default_rng(0)
        residuals = rng.standard_normal(500)
        p         = _fit_garch_moments(residuals)
        assert p.persistence < 1.0
        assert p.omega > 0
        assert p.nu > 4.0

    def test_mle_near_zero_variance_no_exception(self):
        """Near-zero variance must not raise — should fall back gracefully."""
        tiny = np.full(300, 1e-15)
        p    = fit_garch_mle(tiny)
        assert p.persistence < 1.0


# ===========================================================================
# TestNumerical
# ===========================================================================

class TestNumerical:

    def test_long_series_no_nan(self):
        cfg = SyntheticWeatherConfig(start_date="1970-01-01", end_date="2023-12-31")
        df  = generate_synthetic_weather(cfg)
        assert not df[["temperature_c", "rainfall_mm"]].isna().any().any()

    def test_long_series_no_negative_rain(self):
        cfg = SyntheticWeatherConfig(start_date="1970-01-01", end_date="2023-12-31")
        df  = generate_synthetic_weather(cfg)
        assert (df["rainfall_mm"] >= 0.0).all()

    def test_stl_residuals_stationary_long_series(self):
        cfg = SyntheticWeatherConfig(start_date="1990-01-01", end_date="2023-12-31")
        df  = generate_synthetic_weather(cfg)
        stl = run_stl_decomposition(df)
        assert check_stationarity(stl["residual"].dropna().values, "long_series")

    def test_different_seeds_produce_different_data(self):
        cfg1 = SyntheticWeatherConfig(random_seed=1)
        cfg2 = SyntheticWeatherConfig(random_seed=2)
        df1  = generate_synthetic_weather(cfg1)
        df2  = generate_synthetic_weather(cfg2)
        assert not np.allclose(
            df1["rainfall_mm"].values, df2["rainfall_mm"].values
        )
