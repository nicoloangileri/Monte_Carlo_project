/**
 * main.cpp — Mediterranean Quantitative Finance Society (MQFS)
 * =============================================================
 * Entry point for the Drought Micro-Insurance Monte Carlo Pricing Engine.
 *
 * JSON parsing: nlohmann/json v3.11.3 (MIT licence, single-header).
 *
 * INSTALL nlohmann/json
 * ----------------------
 * Option A — Homebrew (recommended):
 *     brew install nlohmann-json
 *
 * Option B — Vendored single header (no Homebrew required):
 *     mkdir -p src/cpp/vendor
 *     curl -L https://raw.githubusercontent.com/nlohmann/json/v3.11.3/\
 *     single_include/nlohmann/json.hpp -o src/cpp/vendor/json.hpp
 *
 * COMPILATION (Apple Silicon, macOS)
 * -----------------------------------
 * With Homebrew:
 *     clang++ -std=c++17 -O3 -march=native \
 *             -I$(brew --prefix nlohmann-json)/include \
 *             -o mc_engine src/cpp/main.cpp
 *
 * With vendored header:
 *     clang++ -std=c++17 -O3 -march=native \
 *             -I src/cpp \
 *             -o mc_engine src/cpp/main.cpp
 *
 * RUN
 * ---
 *     ./mc_engine                                    # uses defaults
 *     ./mc_engine data/fitted_params.json 500000
 *
 * Author  : MQFS Research Division
 * Standard: C++17
 */

#include "monte_carlo_engine.hpp"

// nlohmann/json — Homebrew first, vendored header second
#if __has_include(<nlohmann/json.hpp>)
  #include <nlohmann/json.hpp>
#elif __has_include("vendor/json.hpp")
  #include "vendor/json.hpp"
#else
  #error "nlohmann/json not found. See INSTALL section in main.cpp."
#endif

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Load ModelParams from the JSON file produced by ts_model.py
// ---------------------------------------------------------------------------

static mqfs::ModelParams load_params(const std::string& path)
{
    std::ifstream f(path);
    if (!f.is_open()) {
        std::cerr << "[WARN] Cannot open '" << path
                  << "'. Using built-in default parameters.\n";
        return mqfs::ModelParams{};
    }

    json j;
    try {
        f >> j;
    } catch (const json::parse_error& e) {
        std::cerr << "[ERROR] JSON parse error in '" << path
                  << "': " << e.what()
                  << "\n        Using default parameters.\n";
        return mqfs::ModelParams{};
    }

    mqfs::ModelParams p;
    p.temp_mu              = j.value("temp_mu",              0.02);
    p.temp_sigma           = j.value("temp_sigma",           2.7);
    p.rain_mu              = j.value("rain_mu",              0.5);
    p.rain_kappa           = j.value("rain_kappa",           0.6);
    p.rain_sigma           = j.value("rain_sigma",           0.45);
    p.drought_threshold_mm = j.value("drought_threshold_mm", 1.823);
    p.historical_strike    = j.value("historical_strike_cdi",1.0162);
    p.rho_temp_rain        = j.value("rho_temp_rain",       -0.35);
    p.n_years              = j.value("n_years",              34);
    p.dt                   = j.value("dt",                   1.0 / 365.25);

    // GARCH sub-object (serialised nested by Python dataclasses.asdict)
    if (j.contains("garch") && j["garch"].is_object()) {
        const auto& g = j["garch"];
        p.garch.omega = g.value("omega", 1e-5);
        p.garch.alpha = g.value("alpha", 0.08);
        p.garch.beta  = g.value("beta",  0.87);
        p.garch.nu    = g.value("nu",    6.0);
    }

    // Guard: GARCH stationarity (α+β must be < 1)
    if (p.garch.alpha + p.garch.beta >= 1.0) {
        std::cerr << "[WARN] GARCH non-stationary (α+β="
                  << p.garch.alpha + p.garch.beta
                  << "). Scaling down to 0.97.\n";
        const double scale = 0.97 / (p.garch.alpha + p.garch.beta);
        p.garch.alpha *= scale;
        p.garch.beta  *= scale;
    }

    std::cout << "[Config] Parameters loaded from: " << path        << "\n"
              << "  R* (drought threshold) = " << p.drought_threshold_mm << " mm/day\n"
              << "  K  (strike CDI)        = " << p.historical_strike    << " mm/day\n"
              << "  GARCH α+β              = " << p.garch.alpha + p.garch.beta << "\n"
              << "  ρ(temp, rain)          = " << p.rho_temp_rain         << "\n";
    return p;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[])
{
    const std::string params_path = (argc > 1) ? argv[1] : "data/fitted_params.json";
    const int         n_paths     = (argc > 2) ? std::atoi(argv[2]) : 500'000;

    std::cout << 
              << "Paths requested: " << n_paths << "\n\n";

    // ── Load fitted model parameters ─────────────────────────────────────
    mqfs::ModelParams model = load_params(params_path);

    // ── Derivative specification ──────────────────────────────────────────
    // Strike is data-calibrated (65th-pct historical CDI → P(trigger)≈35%)
    mqfs::DerivativeSpec deriv;
    deriv.notional_eur_per_mm = 5'000.0;
    deriv.strike_cdi          = model.historical_strike;
    deriv.wang_lambda         = 0.25;
    deriv.contract_days       = 120;

    // ── Simulation configuration ──────────────────────────────────────────
    mqfs::SimulationConfig cfg;
    cfg.n_paths    = n_paths;
    cfg.n_steps    = deriv.contract_days;
    cfg.seed       = 42;
    cfg.save_paths = true;
    cfg.n_save     = 2'000;
    cfg.output_dir = "data/";

    // ── Run pricing ────────────────────────────────────────────────────────
    const auto t0 = std::chrono::steady_clock::now();
    mqfs::MonteCarloEngine    engine(model, deriv, cfg);
    const mqfs::PricingResult result = engine.price();
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    result.print();
    std::cout << "Elapsed : " << elapsed << " s\n"
              << "Throughput: "
              << static_cast<long>(n_paths / elapsed) << " paths/s\n";

    // ── Persist pricing result via nlohmann/json ──────────────────────────
    std::ofstream out("data/pricing_result.json");
    if (out.is_open()) {
        json r;
        r["price_eur"]            = result.price_eur;
        r["std_error"]            = result.std_error;
        r["ci_lower_95"]          = result.ci_lower_95;
        r["ci_upper_95"]          = result.ci_upper_95;
        r["probability_trigger"]  = result.probability_trigger;
        r["expected_shortfall"]   = result.expected_shortfall;
        r["delta"]                = result.delta;
        r["n_paths"]              = result.n_paths_used;
        r["strike_cdi"]           = deriv.strike_cdi;
        r["drought_threshold_mm"] = model.drought_threshold_mm;
        out << r.dump(2) << "\n";
        std::cout << "[MC] Pricing result → data/pricing_result.json\n";
    }

    return 0;
}
