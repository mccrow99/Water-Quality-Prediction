# Water Quality Prediction

An end-to-end machine learning project for forecasting water quality conditions at Iowa EPA monitoring stations. It combines water quality measurements, climate records, streamflow, soil, and agricultural data into a single 315-column modeling table, then serves predictions for **thirteen** targets through an interactive Dash dashboard with map-based spatial interpolation.

This repository is maintained as a completed project snapshot, with pre-trained artifacts and processed datasets included for reproducibility. It is organized so the project can be reviewed or run without rebuilding the full pipeline first.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Data Sources](#data-sources)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the App](#how-to-run-the-app)
- [Dashboard Features](#dashboard-features)
- [Data Pipeline](#data-pipeline)
- [Feature Contract](#feature-contract)
- [Models](#models)
- [Model Performance](#model-performance)
- [How to Retrain the Models](#how-to-retrain-the-models)
- [Testing](#testing)
- [Deployment](#deployment)
- [Tech Stack](#tech-stack)

---

## Project Overview

This project enables predictive water quality modeling and analysis across Iowa using a comprehensive, multi-source dataset. The core application answers: **given the location of an EPA monitoring station and a date, what water quality should we expect there?**

The repository integrates data spanning water quality, climate, streamflow, soil, land use, agriculture, and regulatory sources into one modeling table (`data/final/epa-full.csv`, 48,251 rows × 315 columns before EDA, 318 after). See [`DATA.md`](DATA.md) for the full data dictionary and [`MERGE.md`](MERGE.md) for exactly how each source is joined.

Four scikit-learn model families (Linear Regression, Random Forest, Gradient Boosting, Neural Network) are trained for each of **thirteen** water quality targets:

| Target Variable | Unit |
|---|---|
| Water Temperature | °C |
| Dissolved Oxygen | mg/L |
| pH | pH |
| Nitrate | mg/L as N |
| Nitrite | mg/L as N |
| Nitrate + Nitrite | mg/L as N |
| Total Phosphorus | mg/L as P |
| Specific Conductance | µS/cm |
| Total Dissolved Solids | mg/L |
| Total Suspended Solids | mg/L |
| Turbidity | NTU |
| E. coli | MPN/100mL |
| WQI (composite Water Quality Index) | index, 0 = best, 100 = worst |

Predictions are delivered through a locally runnable Dash app. The user selects a target, a model family, and a date; the app runs inference across every monitoring station that has the necessary predictors and renders a spatially interpolated map of predicted values across Iowa, along with per-station hover detail and held-out performance context.

Pre-trained model files (52 total — 13 targets × 4 families) are included in the repository so the dashboard works immediately without retraining.

---

## Data Sources

This project assembles a multi-modal dataset covering water quality, climate, streamflow, soil, land use, agriculture, NPDES regulatory compliance, and census data for Iowa.

> 📑 **See [`DATA.md`](DATA.md) for the full data dictionary** — every column in every dataset described, with source links and a raw-data size table (rows, columns, MB).
>
> 📑 **See [`MERGE.md`](MERGE.md) for the merge plan** — how every cleaned table is joined into the terminal modeling table, with keys, grain, and verified row/column counts at every stage.

At a glance:

- **Water Quality**: ~971K EPA WQX observations across 1,666 monitoring stations, pivoted to one row per station-day
- **Climate**: ISU/IEM daily station records and PRISM gridded daily climate, joined per station-day
- **Streamflow**: USGS daily discharge at ~700 gauges, matched to the nearest gauge per station
- **Soil**: SSURGO map-unit properties (Ksat, available water capacity) via spatial join
- **Land Use**: HUC-12-level cropland fractions (corn, soybean, developed, forest) from the USDA Cropland Data Layer
- **Agriculture**: County-level nutrient loading (N/P from fertilizer and manure) and chemical spending from USDA NASS / USGS
- **Regulatory**: NPDES facility density and ATTAINS impairment context near each station
- **Demographics**: County population from the Census Bureau

**Geographic coverage**: Iowa statewide. **Temporal coverage**: water quality and climate records span multiple decades; the terminal modeling table (`epa-full.csv`) holds 48,251 station-day observation rows.

---

## Repository Structure

```
.
├── app.py                                  # Dash dashboard (main entry point)
├── test_app.py                             # Smoke tests for the dashboard
├── build_station_table.py                  # Precomputes data/stations.csv for deployment
├── requirements.txt                        # Full dev environment (notebooks + pipeline)
├── run-requirements.txt                    # Runtime-only subset, used by the deployed app
├── DATA.md                                 # Full data dictionary
├── MERGE.md                                # Merge plan: keys, stages, verified shapes
│
├── data/
│   ├── tabular/
│   │   ├── 01_raw/<domain>/                # Raw downloaded inputs, by domain
│   │   └── 02_clean/<domain>/               # Cleaned outputs, by domain
│   ├── spatial/
│   │   ├── 01_raw/                         # Raw shapefiles/rasters (CDL, SSURGO, NHDPlus)
│   │   └── 02_clean/                       # Tabular crosswalks from spatial joins
│   ├── 03a_merge_primary/                  # P1–P7 primary merges (per-source, station/county grain)
│   ├── 03b_merge_secondary/                # S1–S2 secondary merges
│   ├── stations.csv                        # One row per station (1,345 × 318) — what the app reads
│   ├── final/
│   │   └── epa-full.csv                    # Terminal modeling table (48,251 × 318), gitignored
│   ├── images/water-images/                # Water quality classification image samples
│   └── text/raw/                           # City-level water summary narratives
│
└── src/
    ├── 01_download/                        # API/portal download notebooks, one per source
    ├── 02_clean/
    │   ├── tabular/<domain>/                # Cleaning notebooks, one per raw table
    │   └── spatial/<domain>/                # Spatial-join crosswalk notebooks
    ├── 03_merge/                           # P1–P7, S1–S2, T1 merge notebooks (see MERGE.md)
    ├── 04_eda/
    │   ├── univariate-analysis.ipynb        # Read-only EDA notebooks
    │   ├── bivariate-analysis.ipynb
    │   ├── multivariate-analysis.ipynb
    │   ├── wqi-calculation.ipynb            # Appends WQI, WQI_n_groups, WQI_weight_coverage
    │   ├── eda-summary.md                   # Consolidated EDA findings
    │   └── outputs/                        # ~35 CSV/PNG diagnostic artifacts
    └── 05_modeling/
        ├── linear_regression/
        │   ├── multiple_linear_regression.ipynb
        │   └── lr_<target>.pkl              # 13 files
        ├── random_forest/
        │   ├── random_forest.ipynb
        │   └── rf_<target>.pkl              # 13 files
        ├── gradient_boosting/
        │   ├── gradient_boosting.ipynb
        │   └── gb_<target>.pkl              # 13 files
        ├── neural_network/
        │   ├── neural_network.ipynb
        │   └── nn_<target>.pkl              # 13 files
        ├── model_metrics.csv                # Held-out test metrics for all 52 models
        └── model_outcomes.md                # Human-readable performance summary
```

---

## Setup and Installation

**Requirements:** Python 3.9 or higher.

### 1. Clone the repository

```bash
git clone https://github.com/NullPranaya/Water-Quality-Prediction.git
cd Water-Quality-Prediction
```

### 2. Create a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs everything needed: Dash, Plotly, pandas, NumPy, SciPy, scikit-learn, GeoPandas, and the full Jupyter environment.

---

## How to Run the App

With your virtual environment activated and dependencies installed, run the app from the project root:

```bash
python3 app.py
```

Then open your browser and go to:

```
http://127.0.0.1:8050
```

The app loads all 52 pre-trained `.pkl` models from `src/05_modeling/<family>/` at startup, along with the station table (`data/stations.csv`, falling back to `data/final/epa-full.csv`) and `src/05_modeling/model_metrics.csv` — no retraining needed. Any target/model combination whose file is missing, or whose stored feature list disagrees with the app's, is disabled in the UI rather than crashing the app.

---

## Dashboard Features

The dashboard is organized into a left control panel and a right map panel.

**Left Panel — Controls:**
- **Target Variable** — choose one of thirteen water quality parameters (twelve measured parameters plus the composite WQI)
- **Prediction Model** — switch between Linear Regression, Random Forest, and Gradient Boosting
- **Prediction Date** — pick any date, or use the quick-select buttons (Today, In 1 week, In 2 weeks, In 1 month, In 6 months, In 1 year)
- **Run Prediction** — runs inference across every monitoring station and a held-out performance summary for the selected target/model

**Right Panel — Map:**
- Displays Iowa EPA monitoring stations on a scoped U.S. map
- After running a prediction, shows a color-coded interpolated surface (`scipy.griddata`, cubic with linear fallback) across the station network, anchored to the 2nd–98th percentile of station predictions to avoid spline overshoot distorting the display
- Station markers overlay the interpolated grid with per-station predicted values on hover, reverse-geocoded to the nearest major Iowa city
- Summary panel shows Min, Max, Mean, and Std Dev for the current prediction, plus the model's held-out R², error rate, and its margin over a same-station persistence baseline

**Color scales by target** (`app.py::TARGET_COLORSCALES`): e.g. Water Temperature — RdYlBu (reversed); pH — RdYlGn; Dissolved Oxygen — Blues; the nitrogen/turbidity/solids family — Yellow-Orange-Red/Brown ramps; WQI — RdYlGn reversed (green = good/low, red = bad/high, since WQI runs 0=best to 100=worst).

---

## Data Pipeline

The project is a linear, five-stage pipeline. Pre-processed artifacts are committed at every stage so the app and notebooks work without re-running anything upstream.

```
src/01_download/   → data/tabular/01_raw/         API/portal download notebooks
src/02_clean/      → data/tabular/02_clean/       Cleaning notebooks, one per raw table
                     data/spatial/02_clean/       Spatial-join crosswalks (e.g. station → HUC-12)
src/03_merge/      → data/03a_merge_primary/      P1–P7: per-source merges onto station/county grain
                     data/03b_merge_secondary/    S1–S2: station-day + station-year context
                     data/final/     T1: epa-full.csv, the terminal modeling table
src/04_eda/        → epa-full.csv (in place)      3 read-only EDA notebooks + WQI calculation
src/05_modeling/   → src/05_modeling/<family>/*.pkl   3 training notebooks, 13 targets each
                     model_metrics.csv
app.py             ← epa-full.csv + <family>/*.pkl + model_metrics.csv
```

Full per-stage detail — every notebook, its output file, keys, and verified row/column counts — is in [`MERGE.md`](MERGE.md). Stage-1/2 source detail is in [`DATA.md`](DATA.md).

`src/04_eda/wqi-calculation.ipynb` appends three columns to `epa-full.csv` in place: `WQI` (0–100, 0 = best), `WQI_n_groups`, and `WQI_weight_coverage` — present on 56.6% of rows (943 stations). These are not part of the model feature set; `WQI` is itself the thirteenth prediction target.

---

## Feature Contract

Every model is trained on the same **29 columns**, in a fixed order, shared across `app.py` and all four training notebooks: 25 station-level base features read straight from `epa-full.csv`, plus 4 temporal features derived from the observation/prediction date.

**Base features (25):** location (`LatitudeMeasure`, `LongitudeMeasure`), distance to the nearest climate station and streamflow gauge, PRISM climate (`prism_tmax_c`, `prism_tmin_c`, `prism_ppt_mm`, `prism_tdmean_c`), ISU daily weather (wind, humidity, snow, feels-like temps), streamflow discharge, soil (`ksat_mean`, `awc_mean`), land cover fractions (`pct_corn`, `pct_soybean`, `pct_developed`, `pct_forest`), and nutrient loading (N/P from fertilizer and manure).

**Temporal features (4):** `doy`, `doy_sin`, `doy_cos` (cyclical day-of-year encoding), `obs_year`.

`pct_row_crops` exists in `epa-full.csv` but is deliberately excluded — it equals `pct_corn + pct_soybean` exactly on every row and was a pure rank-deficiency problem (condition number 1.3e15 with it in, 27.8 without), carrying no information.

Each `.pkl` stores its own `feature_cols`, checked against the app's feature list at load time; a mismatch disables that model in the UI rather than risking a silently misaligned prediction.

---

## Models

### Model files

Each `.pkl` in `src/05_modeling/<family>/` holds a plain dict, not a bare estimator:

```python
{"pipeline": Pipeline, "target_transform": "none" | "log10",
 "log_offset": float | None, "smearing_factor": float,
 "feature_cols": [...29 names...]}
```

`pipeline` is a fitted scikit-learn `Pipeline`:

1. `SimpleImputer(strategy="median")` — models impute missing predictors internally, so the app passes raw feature values straight through. The neural network uses `add_indicator=True`, emitting extra missingness flags; that expansion happens *inside* the pipeline and does not change the 29-column input contract
2. `StandardScaler()` — linear regression and neural network only (the tree families do not need it)
3. Estimator — `LinearRegression`, `RandomForestRegressor`, `HistGradientBoostingRegressor`, or a `VotingRegressor` averaging three `MLPRegressor`s with seeds 42/43/44 (the gradient boosting pipeline uses early stopping; the neural network searches its epoch count per target)

Every component is stock scikit-learn on purpose: no bespoke class has to be importable at unpickle time. The random forests are additionally **size-capped** at 100 trees with `min_samples_leaf=10` so the family fits in git and in the deploy host's memory — see [Deployment](#deployment).

### Target transform

39 of the 52 models regress on the raw target scale; **13 are fitted on `log10(y + c)`** and their predictions are back-transformed (with Duan's smearing correction) before display. Which (target, family) pairs take the log is decided by a cross-validated bake-off in each training notebook — comparing MAE in the target's own units, with guardrails against targets that are mostly zero (Nitrate, Nitrite) or can be negative (Water Temperature) — not asserted by hand. The outcome: `Turbidity` takes the log in all four families; `E. coli` and `Total Suspended Solids` take it for Linear Regression, Random Forest and Gradient Boosting but not the Neural Network; `Total Phosphorus` takes it for Random Forest, Gradient Boosting and the Neural Network but not Linear Regression; everything else, including WQI, stays raw. Per family that is 3 log models for LR, 4 for RF, 4 for GB and 2 for NN. See [`src/05_modeling/model_outcomes.md`](src/05_modeling/model_outcomes.md) for the full selection methodology.

### Train/test split

All four notebooks use `GroupShuffleSplit(test_size=0.2, random_state=42)` grouped on `MonitoringLocationIdentifier` — 20% of the *stations* that measured a target are held out whole, so every reported score answers "how well does this predict at a station the model has never seen?" This replaced a row-level split under which 99%+ of test rows shared a station with training data, which let latitude/longitude alone drive apparent accuracy. See [`src/04_eda/eda-summary.md`](src/04_eda/eda-summary.md) and [`src/05_modeling/model_outcomes.md`](src/05_modeling/model_outcomes.md) for the before/after comparison.

Each model's R² is also reported alongside a same-station **persistence baseline** ("this station's next value equals its previous value") on the same held-out rows — the memorization bar. Three targets (Specific Conductance, Total Dissolved Solids, Total Phosphorus) do not clear it in any family.

---

## Model Performance

Held-out test-set R² by target and model family (raw scale; **bold** = the model was fitted on `log10` and R² is shown on the log scale, where it is the meaningful number). Full metrics — RMSE, MAE, error rate, persistence baselines, station/row counts — are in [`src/05_modeling/model_metrics.csv`](src/05_modeling/model_metrics.csv) and summarized in [`src/05_modeling/model_outcomes.md`](src/05_modeling/model_outcomes.md).

| Target | Linear Regression | Random Forest | Gradient Boosting | Neural Network |
|---|--:|--:|--:|--:|
| Water Temperature | 0.895 | 0.938 | 0.940 | 0.921 |
| Specific Conductance | 0.220 | 0.546 | 0.262 | 0.020 |
| Total Dissolved Solids | 0.421 | 0.533 | 0.432 | 0.425 |
| Nitrate + Nitrite | 0.218 | 0.442 | 0.515 | 0.326 |
| Dissolved Oxygen | 0.388 | 0.476 | 0.467 | 0.452 |
| Nitrate | 0.172 | 0.455 | 0.393 | 0.238 |
| pH | 0.171 | 0.375 | 0.349 | 0.162 |
| WQI | 0.076 | 0.288 | 0.278 | 0.106 |
| E. coli | **0.204** | **0.346** | **0.358** | 0.026 |
| Total Suspended Solids | **0.151** | **0.347** | **0.338** | 0.065 |
| Turbidity (log) | **0.062** | **0.296** | **0.283** | **0.225** |
| Total Phosphorus | 0.023 | **0.178** | **0.189** | **0.099** |
| Nitrite | -0.003 | 0.011 | -0.145 | -0.186 |

Scored the way the table shows them — log scale where the model was log-fitted,
raw otherwise — **Random Forest is best on 9 of the 13 targets and Gradient
Boosting on the other 4**; neither Linear Regression nor the Neural Network is
best on any. Mean across targets: RF 0.40, GB 0.36, LR 0.23, NN 0.22. The
neural network is beaten by a tree ensemble on every single target, and is kept
as a measured baseline rather than a recommendation — it collapses on the
targets where trees exploit sharp splits on latitude/longitude (Specific
Conductance 0.020 against RF's 0.546).

Water Temperature is the best-predicted target across all four families
(R² 0.90–0.94). E. coli, Total Phosphorus, Nitrite, and Turbidity remain hard —
sparse/skewed measurements and station-level heterogeneity limit generalization
to unseen stations, which the grouped split surfaces rather than hides.

The random forests are deliberately **size-capped** (100 trees,
`min_samples_leaf=10`). Grown out they scored a mean 0.018 R² higher but came to
1.8 GB, which cannot be pushed to GitHub or loaded on the deployment host; see
[`src/05_modeling/model_outcomes.md`](src/05_modeling/model_outcomes.md),
"Model size and the deployment ceiling".

---

## How to Retrain the Models

Each notebook trains all 13 targets, overwrites its family's 13 `.pkl` files, and rewrites only its family's rows of `model_metrics.csv` (the other three families' rows are untouched), so the notebooks may be run in any order:

```bash
source venv/bin/activate

jupyter nbconvert --to notebook --execute --inplace src/05_modeling/linear_regression/multiple_linear_regression.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/random_forest/random_forest.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/gradient_boosting/gradient_boosting.ipynb
jupyter nbconvert --to notebook --execute --inplace src/05_modeling/neural_network/neural_network.ipynb
```

The first three notebooks run in well under a minute; the neural network takes about two minutes, since it fits three MLPs per target and searches the epoch count. After retraining, regenerate `src/05_modeling/model_outcomes.md` by hand from the new `model_metrics.csv`, and re-run `python3 build_station_table.py` if `epa-full.csv` changed.

To re-run an upstream cleaning or merge notebook non-interactively (e.g. after a data refresh):

```bash
jupyter nbconvert --to notebook --execute src/02_clean/tabular/water-quality/epa-wq-clean.ipynb
```

---

## Testing

```bash
python -m unittest test_app.py
```

`test_app.py` is the whole suite: it smoke-tests the full Dash callback path (importing `app` loads all 52 `.pkl` files and the station table), checks the 29-column feature contract, and runs a live prediction for the first available target/model combination. It requires the `.pkl` files, `model_metrics.csv`, and a station source (`data/stations.csv`, or `epa-full.csv` as fallback) to be present — all of which are committed, so a fresh clone can run the suite.

---

## Deployment

The dashboard is deployed as a web service on Render. Everything it needs is
committed, so the host neither trains a model nor runs the merge pipeline:

| Setting | Value |
|---|---|
| Build command | `pip install -r run-requirements.txt` |
| Start command | `gunicorn app:server --workers 1 --threads 4` |

**Use `run-requirements.txt`, not `requirements.txt`.** The former is the
serving subset — 12 pinned packages, a 385 MB install. The latter is the full
dev environment (861 MB) and additionally carries matplotlib/seaborn,
geopandas/shapely/pyproj/pyogrio and jupyterlab, none of which `app.py` imports.
If you add an import to `app.py`, add it to `run-requirements.txt` too: the dev
virtualenv will hide the omission and the deploy will fail on boot.

**Keep gunicorn at one worker.** Startup resident memory is ~470 MB — roughly
180 MB of libraries plus 290 MB of models — against 512 MB on the free and
Starter instances. Each additional worker holds its own copy of the models and
will exhaust the instance.

**The app reads `data/stations.csv`, not `epa-full.csv`.** The full modeling
table is 81 MB and gitignored, so it never reaches the host; `app.py` only ever
needed its one-row-per-station projection (1,345 rows, 2.7 MB), which
`build_station_table.py` precomputes and which is committed. Regenerate and
commit it whenever `epa-full.csv` changes — notably after
`src/04_eda/wqi-calculation.ipynb` rewrites the file:

```bash
python3 build_station_table.py
```

If the file is missing, the app falls back to recomputing the identical frame
from `epa-full.csv`, so a stale station table costs startup time rather than
correctness.

Two constraints are load-bearing and easy to break by accident: do not regrow
the random forests past their leaf floor (see
[`src/05_modeling/model_outcomes.md`](src/05_modeling/model_outcomes.md)), and
do not make `app.py` depend on any file under `data/` other than `stations.csv`.

---

## Tech Stack

Runtime (what the deployed app imports — see `run-requirements.txt`):

| Category | Libraries |
|---|---|
| Dashboard | Dash 4.x, Plotly 6.x |
| Data processing | pandas, NumPy |
| Machine learning | scikit-learn (Linear Regression, Random Forest, HistGradientBoosting, MLPRegressor + VotingRegressor, Pipeline, SimpleImputer, StandardScaler) |
| Spatial interpolation | SciPy (`griddata` — cubic with linear fallback) |
| Server | gunicorn |
| Language | Python 3.9+ |

Pipeline and notebooks only (in `requirements.txt`, not installed on the host):

| Category | Libraries |
|---|---|
| Geospatial | GeoPandas, Shapely, pyproj, pyogrio, Folium |
| Notebooks | JupyterLab, IPython |
| Visualization | Matplotlib, Seaborn |

---

## Status

The full pipeline — download through modeling — is built and reproducible from committed artifacts. All 52 models (13 targets × 4 families) are trained and stored in the repository, and the dashboard launches locally with a single command for interactive prediction and visualization across Iowa's water monitoring network. See `CLAUDE.md` for the detailed technical contract between the app and the models (feature order, target transforms, metrics schema) if you're modifying the pipeline.
