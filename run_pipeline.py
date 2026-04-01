"""
run_pipeline.py — Mediterranean Quantitative Finance Society (MQFS)
====================================================================
Master orchestration script. Runs the full pipeline in sequence.

  Step 1 — Data ingestion         (data_loader.py)
  Step 2 — Time-series fitting    (ts_model.py)
  Step 3 — Monte Carlo pricing    (C++ engine, compiled on first run)
  Step 4 — Publication figures    (visualise.py)
  Step 5 — Pricing summary        (stdout)

Usage
-----
  python run_pipeline.py                            # auto-detect data, 500k paths
  python run_pipeline.py --csv data/weather.csv    # explicit real data path
  python run_pipeline.py --fetch-era5              # download ERA5 first
  python run_pipeline.py --fast                    # 50k paths (test mode)
  python run_pipeline.py --no-cpp                  # Python-only, skip C++

Author : MQFS Research Division
License: MIT
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SRC_PY = Path(__file__).parent / "src" / "python"
sys.path.insert(0, str(SRC_PY))

from data_loader import load_weather_data
from ts_model    import fit_all
from visualise   import generate_all_figures

CPP_BINARY = "./mc_engine"
CPP_SRC    = Path("src/cpp/main.cpp")


# ---------------------------------------------------------------------------
# C++ build helpers
# ---------------------------------------------------------------------------

def _find_nlohmann_include() -> list[str]:
    """
    Return compiler flags for nlohmann/json.
    Tries (in order): Homebrew, vendored header, pkg-config.
    Returns empty list if not found (compilation will fail with a clear error).
    """
    # Option 1: Homebrew
    brew = shutil.which("brew")
    if brew:
        result = subprocess.run(
            [brew, "--prefix", "nlohmann-json"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            prefix = result.stdout.strip()
            inc    = Path(prefix) / "include"
            if inc.exists():
                return [f"-I{inc}"]

    # Option 2: Vendored single header
    if Path("src/cpp/vendor/json.hpp").exists():
        return ["-Isrc/cpp"]

    log.error(
        "nlohmann/json not found.\n"
        "  Option A: brew install nlohmann-json\n"
        "  Option B: mkdir -p src/cpp/vendor && "
        "curl -L https://raw.githubusercontent.com/nlohmann/json/"
        "v3.11.3/single_include/nlohmann/json.hpp "
        "-o src/cpp/vendor/json.hpp"
    )
    return []


def compile_cpp(binary: str = CPP_BINARY) -> bool:
    """Compile the C++ engine optimised for Apple Silicon M-series."""
    include_flags = _find_nlohmann_include()
    if not include_flags:
        return False

    cmd = [
        "clang++", "-std=c++17", "-O3", "-march=native",
        *include_flags,
        "-o", binary, str(CPP_SRC),
    ]
    log.info("Compiling: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error("Compilation failed:\n%s", result.stderr)
        return False
    log.info("Compiled successfully → %s", binary)
    return True


def run_cpp_engine(
        binary:      str = CPP_BINARY,
        params_json: str = "data/fitted_params.json",
        n_paths:     int = 500_000,
) -> bool:
    """Execute the compiled pricing engine."""
    if not Path(binary).exists():
        log.error(
            "Binary '%s' not found. Compilation may have failed. "
            "Try manually: clang++ -std=c++17 -O3 -march=native "
            "-I$(brew --prefix nlohmann-json)/include "
            "-o mc_engine src/cpp/main.cpp",
            binary,
        )
        return False
    result = subprocess.run([binary, params_json, str(n_paths)])
    if result.returncode != 0:
        log.error("MC engine exited with code %d.", result.returncode)
        return False
    return True


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(pricing_json: str = "data/pricing_result.json") -> None:
    """Pretty-print the final derivative pricing summary to stdout."""
    if not Path(pricing_json).exists():
        log.warning("No pricing result at %s — C++ step may have been skipped.", pricing_json)
        return
    with open(pricing_json) as f:
        r = json.load(f)

    w = 64
    print("\n" + "═" * w)
    print("  DROUGHT MICRO-INSURANCE — PRICING SUMMARY")
    print("  Mediterranean Quantitative Finance Society (MQFS)")
    print("═" * w)
    print(f"  Drought threshold (R*):          {r.get('drought_threshold_mm', '—'):.4f} mm/day")
    print(f"  Strike (K):                      {r.get('strike_cdi', '—'):.4f} mm/day")
    print(f"  Fair Value (Wang-adjusted):     €{r['price_eur']:>12,.2f}")
    print(f"  Monte Carlo Std Error:          €{r['std_error']:>12,.2f}")
    print(f"  95% CI:            [€{r['ci_lower_95']:,.2f},   €{r['ci_upper_95']:,.2f}]")
    print(f"  P(Trigger):                      {r['probability_trigger']:>11.1%}")
    print(f"  CVaR-95% Payoff:                €{r['expected_shortfall']:>12,.2f}")
    print(f"  Delta (€/mm):                   €{r['delta']:>12,.2f}")
    print(f"  Simulation paths:                {r['n_paths']:>12,}")
    print("═" * w + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MQFS — Climate-Adjusted Drought Insurance Pricing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_pipeline.py --fast\n"
            "  python run_pipeline.py --csv data/weather_data.csv\n"
            "  python run_pipeline.py --fetch-era5\n"
        ),
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to real weather CSV. Auto-detected if omitted.",
    )
    parser.add_argument(
        "--fetch-era5", action="store_true",
        help="Download real ERA5 data before running (requires ~/.cdsapirc).",
    )
    parser.add_argument(
        "--paths", type=int, default=500_000,
        help="Monte Carlo paths (default: 500,000).",
    )
    parser.add_argument(
        "--no-cpp", action="store_true",
        help="Skip C++ compilation and engine execution.",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Quick test run: 50,000 paths.",
    )
    args = parser.parse_args()

    if args.fast:
        args.paths = 50_000
        log.info("Fast mode: %d paths.", args.paths)

    Path("data").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    # ── Optional: fetch real ERA5 data ────────────────────────────────────
    if args.fetch_era5:
        log.info("━━━ Fetching real ERA5 data for Sicily ━━━")
        from era5_fetcher import download_era5
        download_era5()

    # ── Step 1: Data ingestion ────────────────────────────────────────────
    log.info("━━━ Step 1 / 4: Data Ingestion ━━━")
    df, data_source = load_weather_data(args.csv)
    log.info("Data source: %s  (%d observations)", data_source, len(df))

    # ── Step 2: Time-series fitting ───────────────────────────────────────
    # fit_all returns (FittedParams, stl_components) — STL runs only once
    log.info("━━━ Step 2 / 4: Time-Series Fitting ━━━")
    fitted, stl = fit_all(df, output_path="data/fitted_params.json")

    # ── Step 3: Monte Carlo pricing ───────────────────────────────────────
    if not args.no_cpp:
        log.info("━━━ Step 3 / 4: Monte Carlo Pricing (C++) ━━━")
        if compile_cpp():
            run_cpp_engine(n_paths=args.paths)
        else:
            log.warning(
                "C++ engine unavailable. "
                "Figures will use illustrative CDI data."
            )
    else:
        log.info("Skipping C++ step (--no-cpp).")

    # ── Step 4: Figures ───────────────────────────────────────────────────
    log.info("━━━ Step 4 / 4: Generating Figures ━━━")
    generate_all_figures(
        df,
        stl_components        = stl,
        data_source           = data_source,
        drought_threshold_mm  = fitted.drought_threshold_mm,
    )

    print_summary()
    log.info("Pipeline complete. Figures → outputs/")


if __name__ == "__main__":
    main()
