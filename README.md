# Monte_Carlo_project

# Climate-Adjusted Financial Risk & Valuation Model
## Drought Micro-Insurance Derivative Pricing for Mediterranean Agriculture

<p align="center">
  <b>Mediterranean Quantitative Finance Society (MQFS)</b><br/>
  Palermo, Sicily   ·  Open-Source Research Initiative<br/>
  <i>Rigorous quantitative methods applied to the economic challenges of the Mediterranean basin</i>
</p>

---

> **Data transparency:** The repository ships with `data/sicily_reference_synthetic.csv` — 12,418 daily observations (1990–2023) calibrated on published SIAS Palermo-Boccadifalco climate normals (ISPRA/SIAS, 2021). These are **NOT real measurements**. Run `python src/python/era5_fetcher.py --real` to replace them with genuine ERA5 reanalysis data from ECMWF/Copernicus (free CDS account required). All figures and models are clearly labelled with the data source used.

---

## Executive Summary

This repository implements a production-grade, polyglot quantitative model that prices a *Drought Micro-Insurance Derivative* for smallholder farms in Sicily — a population facing intensifying climate risk under Mediterranean warming trends.

The model integrates two pillars of quantitative finance:

1. **Time-Series Forecasting** — STL decomposition, SARIMA, and GARCH(1,1) MLE with Student-t innovations, fitted on real or high-fidelity synthetic Mediterranean climatology.
2. **Monte Carlo Simulation** — 500,000-path correlated bivariate diffusion engine in C++17, pricing a parametric *Cumulative Drought Index* (CDI) derivative under the Wang-distorted risk-neutral measure.

The output is a transparent, explainable fair-value premium that a Sicilian farm cooperative, an impact insurer, or a development finance institution could actually use to transfer catastrophic drought risk.

---

## Mathematical Framework

### 1. Cumulative Drought Index (CDI)

$$
\text{CDI}_T = \frac{1}{T}\sum_{t=1}^{T}\max\bigl(0,\; R^* - R_t\bigr)
$$

$R^* =$ growing-season mean daily rainfall (data-calibrated = **1.8231 mm/day**).
$T = 120$ days (May 1 – August 28 growing season).

### 2. Bivariate Correlated Diffusion (Euler-Maruyama)

$$
dX_t = \kappa(\mu - X_t)\,dt + \sigma_X\,dW_t^1 \qquad [\text{log-rainfall, OU process}]
$$
$$
dW^1 \cdot dW^2 = \rho\,dt \qquad [\text{Cholesky correlation with temperature}]
$$

Correlation $\rho < 0$ captures the physical reality that drought events in Sicily are simultaneously hot and dry — a compounding risk that univariate models miss.

### 3. GARCH(1,1) with Student-t MLE

$$
h_t = \omega + \alpha\varepsilon_{t-1}^2 + \beta h_{t-1}, \qquad \varepsilon_t \sim t(\nu)
$$

Parameters estimated by **Maximum Likelihood** (L-BFGS-B, three restarts). Unconstrained reparameterisation guarantees $\omega > 0$, $\alpha+\beta < 0.97$, $\nu > 4$ without box constraints.

### 4. Wang Transform Risk-Neutral Pricing

$$
S^*(x) = \Phi\!\left(\Phi^{-1}(S(x)) + \lambda\right), \qquad \Pi = N\cdot\mathbb{E}^{\mathbb{Q}}\!\left[\max(0, \text{CDI}_T - K)\right]
$$

Strike $K = $ **1.0162 mm/day** (historical 65th-percentile annual CDI → $P(\text{trigger}) \approx 35\%$).

---

## Technical Architecture

| Layer | Technology | Rationale |
|---|---|---|
| Data ingestion | Python / pandas | Standard for financial time-series I/O |
| TS modelling | Python / statsmodels | Battle-tested SARIMA + robust STL |
| GARCH MLE | Python / scipy.optimize | L-BFGS-B, Student-t log-likelihood |
| **Monte Carlo engine** | **C++17** | 60× faster than Python; 500k paths < 0.5s on Apple M4 |
| **JSON I/O** | **nlohmann/json** | Type-safe, MIT licence, zero external dependencies |
| Visualisation | Python / matplotlib | Full programmatic control at 300 dpi |
| Testing | **pytest (35 tests)** | Unit + integration + seasonality regression suite |

---

## File Structure

```
mqfs-drought-model/
├── src/
│   ├── python/
│   │   ├── data_loader.py         # Hybrid real/synthetic data ingestion
│   │   ├── era5_fetcher.py        # ERA5 downloader + reference dataset generator
│   │   ├── ts_model.py            # STL + SARIMA + GARCH(1,1) MLE pipeline
│   │   └── visualise.py           # Publication figures (300 dpi dark style)
│   └── cpp/
│       ├── monte_carlo_engine.hpp  # Core MC engine (C++17, header-only)
│       └── main.cpp                # Entry point + nlohmann/json I/O
├── data/
│   └── sicily_reference_synthetic.csv  # 12,418-row calibrated synthetic dataset
├── outputs/                        # Generated figures (gitignored, regenerated)
├── tests/
│   └── test_pipeline.py            # 35 pytest unit + integration tests
├── run_pipeline.py                 # Master orchestration script
├── requirements.txt
└── README.md
```

---

## Setup (macOS, Apple Silicon M-series)

### 1 — Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — nlohmann/json (C++ header, one-time)

```bash
# Option A — Homebrew (recommended)
brew install nlohmann-json

# Option B — Vendored single header (no Homebrew needed)
mkdir -p src/cpp/vendor
curl -L https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp \
     -o src/cpp/vendor/json.hpp
```

### 3 — Run the full pipeline

```bash
git clone https://github.com/YOUR-USERNAME/mqfs-drought-model.git
cd mqfs-drought-model

python run_pipeline.py             # reference synthetic data, 500k paths
python run_pipeline.py --fast      # 50k paths, ~5 seconds total
python run_pipeline.py --no-cpp    # Python-only, no compilation needed
```

### 4 — Run the test suite

```bash
pytest tests/ -v
pytest tests/ -v --cov=src/python --cov-report=term-missing
```

### 5 — Manual C++ compilation

```bash
clang++ -std=c++17 -O3 -march=native \
        -I$(brew --prefix nlohmann-json)/include \
        -o mc_engine src/cpp/main.cpp

./mc_engine data/fitted_params.json 500000
```

---

## Using Real ERA5 Data

### One-time CDS setup (free)

1. Register at [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu)
2. Accept the ERA5 data licence
3. Create `~/.cdsapirc`:
   ```
   url: https://cds.climate.copernicus.eu/api/v2
   key: <YOUR-UID>:<YOUR-API-KEY>
   ```
4. `pip install cdsapi xarray netCDF4`

### Download and run

```bash
python src/python/era5_fetcher.py --real   # downloads ~150 MB for Sicily 1990–2023
python run_pipeline.py                     # auto-detects data/weather_data.csv
```

The pipeline logs `✓ REAL DATA` or `⚠ REFERENCE SYNTHETIC` so you always know which mode is active. All figures carry a watermark identifying the data source.

**ERA5 citation:**
> Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* Q. J. R. Meteorol. Soc., 146(730), 1999–2049. https://doi.org/10.1002/qj.3803

---

## Reference Synthetic Dataset

`data/sicily_reference_synthetic.csv` — 12,418 daily observations, 1990–2023.

**Calibration sources:**
- Temperature: ISPRA/SIAS — *Atlante Climatico della Sicilia* (2021). Mean 18.4°C, seasonal amplitude ±9.5°C, peak mid-July.
- Precipitation: SIAS Palermo-Boccadifalco station normals. Mean 2.16 mm/day (~790 mm/yr), Mediterranean dry-summer regime, Student-t ν=4 innovations.

**Validated statistical properties:**
| Property | Value | Source |
|---|---|---|
| Hottest month | July (month 7) | ✓ SIAS normals |
| Coldest month | January (month 1) | ✓ SIAS normals |
| Wettest month | December (month 12) | ✓ Mediterranean regime |
| Annual rainfall | ~1,053 mm/yr | Within SIAS range |
| Drought threshold R* | 1.8231 mm/day | Growing-season mean |
| Historical strike K | 1.0162 mm/day | 65th-pct CDI |
| P(trigger) | ≈ 35.3% | Actuarially sound |

---

## Social Impact

Over 35,000 smallholder farms in Sicily face intensifying drought cycles with no access to affordable, transparent insurance. Parametric micro-insurance — triggered automatically by an objective weather index — eliminates loss-adjustment delays, enables instant payout, and can be structured at premiums accessible to cooperative farming associations.

This model is a transferable proof-of-concept: the same mathematical framework applies directly to Morocco, Tunisia, Greece, or any Mediterranean basin country.

---

## References

1. Cleveland, R.B. et al. (1990). *STL: A Seasonal-Trend Decomposition Procedure.* Journal of Official Statistics, 6(1), 3–73.
2. Wang, S.S. (2000). *A Class of Distortion Operators for Pricing Financial and Insurance Risks.* Journal of Risk and Insurance, 67(1), 15–36.
3. Engle, R.F. (1982). *Autoregressive Conditional Heteroscedasticity.* Econometrica, 50(4), 987–1007.
4. Hersbach, H. et al. (2020). *The ERA5 global reanalysis.* QJRMS, 146(730), 1999–2049.
5. Barnett, B.J. & Mahul, O. (2007). *Weather Index Insurance for Agriculture.* World Bank Working Paper No. 4660.

---

<p align="center">
  <b>MQFS — Mediterranean Quantitative Finance Society</b><br/>
  <i>"The goal is not merely to model risk — it is to make sophisticated risk tools accessible to those who need them most."</i><br/><br/>
  MIT Licence · C++17 · Python 3.11 · nlohmann/json · pytest
</p>
