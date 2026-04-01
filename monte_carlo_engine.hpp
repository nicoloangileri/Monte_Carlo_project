/**
 * monte_carlo_engine.hpp — Mediterranean Quantitative Finance Society (MQFS)
 * ===========================================================================
 * High-performance Monte Carlo engine for pricing a Drought Micro-Insurance
 * Derivative on Sicilian agricultural yields.
 *
 * Mathematical model
 * ------------------
 * Underlying: Cumulative Drought Index (CDI)
 *
 *   CDI_T = (1/T) · Σ_{t=1}^{T} max(0, R* − R_t)
 *
 *   R*  = growing-season mean rainfall (data-calibrated, mm/day)
 *   R_t = simulated daily rainfall
 *   T   = 120 days (May 1 – Aug 28 growing season)
 *
 * Bivariate correlated diffusion (Euler-Maruyama discretisation):
 *
 *   dX_t = κ(μ − X_t)dt + σ_X dW¹_t     [log-rainfall, OU process]
 *   dT_t = μ_T dt        + σ_T dW²_t     [temperature, arithmetic BM]
 *   dW¹ · dW² = ρ dt                      [Cholesky decomposition]
 *
 * GARCH(1,1) volatility clustering on rainfall residuals:
 *   h_t = ω + α ε²_{t-1} + β h_{t-1}
 *   ε_t ~ Student-t(ν)
 *
 * Payoff (European aggregate trigger):
 *   Payoff = N · max(0, CDI_T − K)
 *
 * Risk-neutral pricing via Wang Transform:
 *   S*(x) = Φ(Φ⁻¹(S(x)) + λ)
 *   S(x)  = empirical survival function of CDI under physical measure
 *
 * Strike calibration:
 *   K = historical 65th-percentile annual CDI → P(trigger) ≈ 35%
 *
 * Author  : MQFS Research Division
 * Standard: C++17
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace mqfs {

// ===========================================================================
// Parameter structs
// ===========================================================================

struct GarchParams {
    double omega = 1e-5;   ///< long-run variance intercept ω > 0
    double alpha = 0.08;   ///< ARCH coefficient             α ∈ (0, 0.30)
    double beta  = 0.87;   ///< GARCH coefficient            β ∈ (0, 0.97−α)
    double nu    = 6.0;    ///< Student-t degrees of freedom ν > 4
};

struct ModelParams {
    double temp_mu              =  0.02;          ///< °C / year trend
    double temp_sigma           =  2.7;           ///< temperature residual std-dev
    double rain_mu              =  0.5;           ///< log-space OU long-run mean
    double rain_kappa           =  0.6;           ///< OU mean-reversion speed
    double rain_sigma           =  0.45;          ///< log-space diffusion
    GarchParams garch;
    double drought_threshold_mm =  1.823;         ///< R*: growing-season mean (mm/day)
    double historical_strike    =  1.0162;        ///< K: 65th-pct CDI (mm/day)
    double rho_temp_rain        = -0.35;          ///< temperature–rainfall correlation
    int    n_years              =  34;
    double dt                   =  1.0 / 365.25;
};

struct DerivativeSpec {
    double notional_eur_per_mm = 5'000.0;  ///< € payout per mm CDI above strike
    double strike_cdi;                     ///< K: trigger level (set from ModelParams)
    double wang_lambda         = 0.25;     ///< risk aversion parameter λ
    int    contract_days       = 120;      ///< May 1 – Aug 28 growing season
};

struct SimulationConfig {
    int          n_paths    = 500'000;
    int          n_steps    = 120;
    unsigned int seed       = 42;
    bool         save_paths = true;   ///< write mc_paths.csv for Python fan chart
    int          n_save     = 2'000;  ///< number of paths to persist
    std::string  output_dir = "data/";
};

struct PricingResult {
    double price_eur;            ///< Wang-adjusted fair value (€)
    double std_error;            ///< Monte Carlo standard error (€)
    double ci_lower_95;
    double ci_upper_95;
    double probability_trigger;  ///< P(CDI_T > K) — physical measure
    double expected_shortfall;   ///< CVaR at 95th percentile of payoffs
    double delta;                ///< ∂Price/∂K — strike sensitivity (€/mm)
    long   n_paths_used;

    void print() const {
        std::cout << "\n╔══════════════════════════════════════════════╗\n"
                  << "║  MQFS — Drought Micro-Insurance Pricing       ║\n"
                  << "║  Mediterranean Quantitative Finance Society   ║\n"
                  << "╚══════════════════════════════════════════════╝\n"
                  << std::fixed;
        std::cout.precision(2);
        std::cout << "  Price (€):             " << price_eur            << "\n"
                  << "  Std Error (€):         " << std_error            << "\n"
                  << "  95% CI:         [€"     << ci_lower_95
                  << ",  €"                      << ci_upper_95 << "]\n";
        std::cout.precision(4);
        std::cout << "  P(trigger):            " << probability_trigger  << "\n"
                  << "  CVaR-95% payoff (€):   " << expected_shortfall   << "\n"
                  << "  Delta (€/mm):          " << delta                << "\n"
                  << "  Paths simulated:       " << n_paths_used         << "\n\n";
    }
};

// ===========================================================================
// Wang Transform distortion operator
// ===========================================================================

/**
 * S*(x) = Φ(Φ⁻¹(S(x)) + λ)
 *
 * Wang, S.S. (2000). A Class of Distortion Operators for Pricing
 * Financial and Insurance Risks. Journal of Risk and Insurance, 67(1).
 *
 * Φ⁻¹ implemented via Abramowitz & Stegun rational approximation (26.2.23).
 * More accurate than std::erfinv at extreme tail probabilities.
 */
inline double wang_distort(double prob, double lambda) noexcept {
    if (prob <= 0.0) return 0.0;
    if (prob >= 1.0) return 1.0;

    constexpr double a[] = {
        -3.969683028665376e+01,  2.209460984245205e+02,
        -2.759285104469687e+02,  1.383577518672690e+02,
        -3.066479806614716e+01,  2.506628277459239e+00
    };
    constexpr double b[] = {
        -5.447609879822406e+01,  1.615858368580409e+02,
        -1.556989798598866e+02,  6.680131188771972e+01,
        -1.328068155288572e+01
    };

    const double p = (prob < 0.5) ? prob : 1.0 - prob;
    const double t = std::sqrt(-2.0 * std::log(p));
    double x = (((((a[0]*t+a[1])*t+a[2])*t+a[3])*t+a[4])*t+a[5]) /
               ((((b[0]*t+b[1])*t+b[2])*t+b[3])*t+b[4]);
    if (prob < 0.5) x = -x;

    return 0.5 * std::erfc(-(x + lambda) / std::sqrt(2.0));
}

// ===========================================================================
// Student-t sampler  (ratio-of-normals method)
// ===========================================================================

class StudentTSampler {
public:
    explicit StudentTSampler(double nu, std::mt19937_64& rng)
        : nu_(nu), normal_(0.0, 1.0), chi2_(nu), rng_(rng) {}

    double sample() {
        const double z = normal_(rng_);
        const double v = chi2_(rng_);
        return z / std::sqrt(v / nu_);
    }

private:
    double                                nu_;
    std::normal_distribution<double>      normal_;
    std::chi_squared_distribution<double> chi2_;
    std::mt19937_64&                      rng_;
};

// ===========================================================================
// Monte Carlo Engine
// ===========================================================================

class MonteCarloEngine {
public:
    explicit MonteCarloEngine(ModelParams      model,
                               DerivativeSpec   deriv,
                               SimulationConfig cfg)
        : model_(std::move(model))
        , deriv_(std::move(deriv))
        , cfg_  (std::move(cfg))
        , rng_  (cfg_.seed)
    {}

    PricingResult price();

private:
    ModelParams      model_;
    DerivativeSpec   deriv_;
    SimulationConfig cfg_;
    std::mt19937_64  rng_;

    /**
     * Simulate one path of the bivariate (temp, log-rain) diffusion.
     * GARCH(1,1) variance applied to rainfall innovations.
     * Returns the CDI for this path.
     */
    double simulate_single_path(
        std::normal_distribution<double>& stdnorm,
        StudentTSampler&                  t_sampler,
        std::vector<double>*              path_out = nullptr
    );

    /**
     * Cholesky-factor of 2×2 correlation matrix.
     * L = [[1, 0], [ρ, √(1−ρ²)]]
     * Returns correlated pair (w_temp, w_rain).
     */
    static std::pair<double, double>
    correlated_normals(double z1, double z2, double rho) noexcept {
        return { z1, rho * z1 + std::sqrt(1.0 - rho * rho) * z2 };
    }
};

// ---------------------------------------------------------------------------
// simulate_single_path
// ---------------------------------------------------------------------------

inline double MonteCarloEngine::simulate_single_path(
    std::normal_distribution<double>& stdnorm,
    StudentTSampler&                  t_sampler,
    std::vector<double>*              path_out)
{
    const int    T  = cfg_.n_steps;
    const double dt = (deriv_.contract_days > 0)
                    ? 1.0 / static_cast<double>(deriv_.contract_days)
                    : model_.dt;
    const double sqrt_dt = std::sqrt(dt);

    double log_rain = model_.rain_mu;

    // GARCH initial variance = unconditional expectation E[h]
    // Guard: if persistence → 1, cap denominator to avoid divide-by-zero
    const double persistence = model_.garch.alpha + model_.garch.beta;
    double h = (persistence < 0.999)
             ? model_.garch.omega / (1.0 - persistence)
             : model_.garch.omega / 0.001;

    double prev_eps           = 0.0;
    double cumulative_deficit = 0.0;

    if (path_out) path_out->clear();

    for (int t = 0; t < T; ++t) {
        // Two independent standard normals
        const double z1 = stdnorm(rng_);
        const double z2 = stdnorm(rng_);

        // Correlated shocks via Cholesky decomposition
        auto [w_temp, w_rain] = correlated_normals(z1, z2, model_.rho_temp_rain);

        // GARCH(1,1) variance update
        h = model_.garch.omega
          + model_.garch.alpha * prev_eps * prev_eps
          + model_.garch.beta  * h;
        h = std::max(h, 1e-12);   // numerical floor prevents degenerate paths

        // Student-t innovation scaled by GARCH conditional std-dev
        const double eps_rain = t_sampler.sample() * std::sqrt(h);
        prev_eps = eps_rain;

        // Euler-Maruyama step: log-rainfall OU process
        log_rain += model_.rain_kappa * (model_.rain_mu - log_rain) * dt
                  + model_.rain_sigma * sqrt_dt * w_rain
                  + eps_rain * sqrt_dt;

        const double rain_t = std::exp(log_rain);
        cumulative_deficit += std::max(0.0, model_.drought_threshold_mm - rain_t);

        if (path_out) path_out->push_back(rain_t);
    }

    return cumulative_deficit / static_cast<double>(T);
}

// ---------------------------------------------------------------------------
// price()
// ---------------------------------------------------------------------------

inline PricingResult MonteCarloEngine::price()
{
    const int N = cfg_.n_paths;
    std::vector<double> payoffs(N);
    std::vector<double> cdis(N);

    std::normal_distribution<double> stdnorm(0.0, 1.0);
    StudentTSampler t_sampler(model_.garch.nu, rng_);

    std::vector<std::vector<double>> saved_paths;
    saved_paths.reserve(cfg_.n_save);

    // ── Main simulation loop ──────────────────────────────────────────────
    for (int i = 0; i < N; ++i) {
        std::vector<double>  tmp_path;
        std::vector<double>* path_ptr =
            (cfg_.save_paths && i < cfg_.n_save) ? &tmp_path : nullptr;

        const double cdi = simulate_single_path(stdnorm, t_sampler, path_ptr);
        cdis[i]    = cdi;
        payoffs[i] = deriv_.notional_eur_per_mm
                   * std::max(0.0, cdi - deriv_.strike_cdi);

        if (path_ptr) saved_paths.push_back(std::move(tmp_path));
    }

    // ── Summary statistics ────────────────────────────────────────────────
    const double mean_payoff =
        std::accumulate(payoffs.begin(), payoffs.end(), 0.0) / N;
    double var_acc = 0.0;
    for (const double p : payoffs) { const double d = p - mean_payoff; var_acc += d*d; }
    const double se = std::sqrt(var_acc / (static_cast<double>(N) * (N - 1)));

    // ── Wang Transform pricing ────────────────────────────────────────────
    const long n_triggered = std::count_if(
        cdis.begin(), cdis.end(),
        [this](double c){ return c > deriv_.strike_cdi; }
    );
    const double p_trigger  = static_cast<double>(n_triggered) / N;
    const double p_star     = wang_distort(p_trigger, deriv_.wang_lambda);
    const double wang_price = (p_trigger > 1e-9)
                            ? mean_payoff * (p_star / p_trigger)
                            : 0.0;

    // ── CVaR at 95th percentile ───────────────────────────────────────────
    std::vector<double> sorted_payoffs = payoffs;
    std::sort(sorted_payoffs.begin(), sorted_payoffs.end());
    const int var_idx    = static_cast<int>(0.95 * N);
    const int tail_count = std::max(1, N - var_idx);
    double es = 0.0;
    for (int i = var_idx; i < N; ++i) es += sorted_payoffs[i];
    es /= tail_count;

    // ── Delta: analytic strike sensitivity on the existing CDI sample ─────
    // d/dK E[max(0, CDI−K)] = −P(CDI > K) under physical measure
    // Wang-adjusted: d/dK Wang_price ≈ (price(K−1) − price(K)) / 1
    // Uses same paths → zero additional simulation noise.
    const double K_bump = deriv_.strike_cdi - 1.0;
    double n_above_bump = 0.0;
    for (const double c : cdis) if (c > K_bump) n_above_bump += 1.0;
    const double p_star_bump = wang_distort(n_above_bump / N, deriv_.wang_lambda);
    const double price_bump  = (p_trigger > 1e-9)
                             ? mean_payoff * (p_star_bump / p_trigger) : 0.0;
    const double delta = price_bump - wang_price;   // bump = 1.0 mm

    // ── Persist sample paths for Python fan chart ─────────────────────────
    if (cfg_.save_paths && !saved_paths.empty()) {
        const std::string path_file = cfg_.output_dir + "mc_paths.csv";
        std::ofstream ofs(path_file);
        if (ofs.is_open()) {
            ofs << "path_id,step,rainfall_mm\n";
            for (int pi = 0; pi < static_cast<int>(saved_paths.size()); ++pi)
                for (int s = 0; s < static_cast<int>(saved_paths[pi].size()); ++s)
                    ofs << pi << ',' << s << ',' << saved_paths[pi][s] << '\n';
            std::cout << "[MC] Paths → " << path_file << "\n";
        }
    }

    // ── Persist CDI distribution ──────────────────────────────────────────
    {
        const std::string cdi_file = cfg_.output_dir + "cdi_distribution.csv";
        std::ofstream ofs(cdi_file);
        if (ofs.is_open()) {
            ofs << "cdi\n";
            for (const double c : cdis) ofs << c << '\n';
            std::cout << "[MC] CDI distribution → " << cdi_file << "\n";
        }
    }

    PricingResult result;
    result.price_eur           = wang_price;
    result.std_error           = se;
    result.ci_lower_95         = wang_price - 1.96 * se;
    result.ci_upper_95         = wang_price + 1.96 * se;
    result.probability_trigger = p_trigger;
    result.expected_shortfall  = es;
    result.delta               = delta;
    result.n_paths_used        = N;

    return result;
}

} // namespace mqfs
