# Water Quality Prediction

An end-to-end machine learning project for estimating water quality conditions at Iowa EPA monitoring stations. It combines water quality measurements, climate records, streamflow, soil, and agricultural data into a single 315-column modeling table, then serves predictions for **thirteen** targets through an interactive Dash dashboard with a station map.

This repository is a **research prototype**. The data pipeline, exploratory analysis and model evaluation are complete, and the trained models and station table are committed so the dashboard runs from a fresh clone. The rest of `data/` (~8 GB, including the 81 MB modeling table) is **not** in git, so re-running any notebook needs a copy of it — see [`NEXT_STEPS.md`](NEXT_STEPS.md#step-0--get-set-up). **The dashboard is not yet ready to inform real decisions**: its maps are less accurate than the reported test scores suggest, for reasons measured below. Read [Current Status and Limitations](#current-status-and-limitations) before using any prediction, and [`NEXT_STEPS.md`](NEXT_STEPS.md) before continuing the work.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Current Status and Limitations](#current-status-and-limitations)
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
- [Next Steps](#next-steps)

---

## Project Overview

This project enables predictive water quality modeling and analysis across Iowa using a comprehensive, multi-source dataset. The core application is built to answer: **given the location of an EPA monitoring station and a date, what water quality should we expect there?** It does not yet answer that reliably — see [Current Status and Limitations](#current-status-and-limitations).

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

Predictions are delivered through a locally runnable Dash app. The user selects a target, a model family, and a date; the app runs inference at all 1,345 monitoring stations and colours each station on a map of Iowa (an interpolated surface between stations is available but off by default), along with per-station hover detail and held-out performance context.

Pre-trained model files (52 total — 13 targets × 4 families) are included in the repository so the dashboard works immediately without retraining.

---

## Current Status and Limitations

### At a glance

| Component | State | Useful today for | Not useful for |
|---|---|---|---|
| **Data pipeline** (`src/01`–`03`) | Complete; grain, units and plausibility verified | A clean, documented station-day modeling table for Iowa, 2015-01-02 → 2025-12-25 | — |
| **EDA** (`src/04_eda/`) | Thorough, but model-diagnostic | Data scientists judging whether and how this table can be modeled | Anyone asking about Iowa's water itself — where, when and how badly standards are exceeded, or whether conditions are improving. Parts of `eda-summary.md` also predate fixes it recommended (see below) |
| **Model evaluation** (`src/05_modeling/`) | Rigorous | Knowing how well each target can be predicted at a station the model never saw, against a no-feature persistence baseline | Knowing how accurate the dashboard's maps are — that was never measured, and it is worse |
| **Dashboard** (`app.py`) | Working prototype | Demonstrating the pipeline; comparing model families in the table at the bottom of the page | Deciding anything about a real water body |

The strongest part of the project is its evaluation discipline: whole stations are held out, every score sits beside a persistence baseline, the target transform is chosen by cross-validation rather than by hand, and the trade-off between model size and accuracy is documented. The water temperature model is genuinely good (R² 0.94) when it is given the weather recorded on the sample date.

### Why the dashboard should not yet inform decisions

Measured by calling the app's own prediction path (`app.py::_cached_station_predictions`) against `data/stations.csv` and `data/final/epa-full.csv`. How to reproduce these numbers is in [`NEXT_STEPS.md`](NEXT_STEPS.md#appendix-reproducing-the-readme-figures).

**1. The test scores do not describe what the map shows.** Every model was scored using the weather actually recorded on each sample's date. The app instead gives each station the weather from its **last visit** (`app.py::build_feature_matrix`) and changes only the day-of-year and year features. The median station's last visit was **September 2022**, and **87%** of last visits fell in April–October, so the models are fed summer weather in every season. The effect on the project's best model, water temperature:

| Statewide median, °C | Jan | Apr | Jul | Oct |
|---|--:|--:|--:|--:|
| Observed (all samples in that month) | 0.6 | 10.5 | 24.5 | 14.4 |
| Dashboard prediction (Gradient Boosting, 2026) | 11.6 | 15.5 | 20.8 | 16.4 |

The January map is about **11 °C too warm**, and the seasonal swing is flattened by more than half. Dissolved oxygen shows the same flattening (January: 12.2 mg/L observed, 9.8 predicted). This is red flag 4 in `src/04_eda/eda-summary.md`, and it is still open.

**2. Future dates are not forecasts.** The training data ends in December 2025.
- The **tree models** (Random Forest, Gradient Boosting) cannot extrapolate the year, so from 2026 onward the year has no effect — a July 2030 map is identical to a July 2025 map at every station.
- **Linear Regression and the Neural Network** instead extend the year trend they fitted past the end of the data. Between July 2025 and July 2030 the neural network's median Nitrate + Nitrite prediction rises from 8.4 to 18.1 mg/L, and linear regression's median Turbidity falls 44% — trends nothing in the data supports.
- The "In 6 months" and "In 1 year" shortcuts, and the "Recent trend" panel, therefore show a seasonal curve built on stale weather, not a trend or a forecast.

**3. Most targets are drawn at stations of a type the model never saw.** Every target is predicted at all 1,345 stations, which include 190 wells (groundwater), 318 lakes and 158 wetlands. Station type is not a model input, so a model cannot tell a well from a stream. For example:

| Target | Stations that ever measured it | Shown on the map at, among others |
|---|--:|---|
| Nitrate | 278 | 315 lakes and 158 wetlands that never measured nitrate |
| E. coli | 434 | 186 wells and 158 wetlands that never measured E. coli |
| Specific Conductance | 483 | 494 river/stream sites that never measured it (training was mostly lakes and wells) |

The "unseen stations" test only held out stations that measured the target, so it says nothing about these.

**4. Where real data exists, the map mostly repeats it — without showing it.** At stations with five or more samples, the dashboard's predictions rank-correlate **0.60–0.91** with that station's own historical median (partly because most of those stations were in training). For Specific Conductance, Total Dissolved Solids and Total Phosphorus, simply repeating the station's last reading beats every model. Yet the hover panel shows no observed value, last sample date or sample count, so the better answer already in the data is hidden.

**5. Point predictions carry confident labels and no uncertainty.** Each station gets a label such as "Low" or "High Concern" from a single predicted value, but typical errors are as large as the thresholds those labels use: the best E. coli model's mean absolute error is about **1,490 MPN/100 mL** against a 235 MPN/100 mL recreational threshold, and the best nitrate model's is **2.5 mg/L** against bands at 1, 3 and 10. Nothing on screen says how often a label would be wrong. WQI is an index defined by this project; its bands are quartiles of the observed data, not a regulatory standard.

Underlying all five: **the project has no named user or decision.** "What should we expect at this station on this date" is a question, not a decision. "Should this beach post an advisory this weekend?" or "Will nitrate at this river site exceed 10 mg/L next month?" are decisions, and each needs a different target, horizon, set of locations and success metric. Choosing one is step 1 of [`NEXT_STEPS.md`](NEXT_STEPS.md).

### Documentation known to be out of date

Kept for the record of how the project evolved, but not a description of the current state:

- **`src/04_eda/eda-summary.md`** — the red-flags table (§2) has no status column, so the station-leaking split (flag 1) and the redundant `pct_row_crops` column (flag 5) still read as open "critical" issues; both are fixed. §3.1 quotes the pre-fix, leaky R² scores (Gradient Boosting mean 0.547); §4.1 refers to 36 `.pkl` files and three training notebooks (now 52 and four); §5 calls the grouped re-fit "the open question" — it has been done, and its results are in `model_outcomes.md`.
- **`src/05_modeling/model_outcomes.md`** — says `app.py` loads only three model families (it loads all four); the WQI section's text quotes Random Forest R² 0.337 / margin +0.168 where its own table shows 0.288 / +0.121; the "Against the memorization baseline" table and several takeaways still use the pre-size-cap random forest scores (e.g. Specific Conductance 0.655, now 0.546).

`src/05_modeling/model_metrics.csv` is current and is the source of truth for every score.

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

**Geographic coverage**: Iowa statewide. **Temporal coverage**: the terminal modeling table (`epa-full.csv`) holds 48,251 station-day observation rows at 1,345 stations, from 2015-01-02 to 2025-12-25.

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
├── NEXT_STEPS.md                           # Prioritised plan for continuing the project
│
├── data/                                   # gitignored except stations.csv (~8 GB locally)
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
    │   ├── block_permutation_importance.py  # Which predictor blocks the trained models rely on
    │   ├── eda-summary.md                   # Consolidated EDA findings (partly out of date — see Current Status)
    │   ├── plots/                           # Key EDA figures (PNG), gitignored — regenerate from the notebooks
    │   └── outputs/                        # CSV/PNG diagnostic tables, gitignored — regenerate from the notebooks
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

> Read [Current Status and Limitations](#current-status-and-limitations) first: the features below work as described, but the values they display are not yet reliable enough to act on.

The dashboard has a left control column, a centre map, a right information rail, and a model comparison table beneath them.

**Left — Controls:**
- **What to measure** — one of thirteen water quality parameters (twelve measured parameters plus the composite WQI)
- **Which model** — Linear Regression, Random Forest, Gradient Boosting or Neural Network (defaults to Gradient Boosting)
- **When** — any date from 2000 to 2030, or a quick-select button (Today, In 1 week, In 2 weeks, In 1 month, In 6 months, In 1 year). Only the day-of-year and year features change with the date; every other predictor is carried forward from the station's last visit
- **Run prediction** — runs inference at all 1,345 monitoring stations
- **Map display** — toggles for the interpolated surface (**off** by default) and for city, river and lake labels (on by default)

**Centre — Map:**
- Each station is coloured by its predicted value, on a scale anchored to the 2nd–98th percentile of station predictions
- With the interpolated surface on, a cubic spline (`scipy.griddata`, linear fallback) fills the gaps between stations. It is drawn between the model's outputs, not predicted, and is least trustworthy where stations are sparse

**Right — Information rail:**
- **Point detail** — the hovered station's predicted value, a qualitative label ("Low", "High Concern", …), provider, matched climate station and the nearest major city
- **Recent trend** — a calendar heatmap of the statewide average prediction for the 14 weeks up to the chosen date. Because weather is held fixed, this is a seasonal curve rather than an observed trend
- **Statewide spread** — low, median, high and average of the station predictions
- **Model accuracy** — the selected model's held-out R² (log-scale R² for log-fitted models), RMSE, MAE, error rate, and its margin over the repeat-last-value persistence baseline

**Below — Compare every model:** held-out scores for all 13 targets × 4 families from `model_metrics.csv`. Clicking a row loads that target and model into the controls.

**Color scales by target** (`app.py::TARGET_COLORSCALES`): e.g. Water Temperature — RdYlBu (reversed); pH — RdYlGn; Dissolved Oxygen — Blues; the nitrogen/turbidity/solids family — Yellow-Orange-Red/Brown ramps; WQI — RdYlGn reversed (green = good/low, red = bad/high, since WQI runs 0=best to 100=worst).

---

## Data Pipeline

The project is a linear, five-stage pipeline. Its intermediate outputs under `data/` are gitignored; only the final station table (`data/stations.csv`) and the trained models are committed, which is enough to run the app but not the notebooks.

```
src/01_download/   → data/tabular/01_raw/         API/portal download notebooks
src/02_clean/      → data/tabular/02_clean/       Cleaning notebooks, one per raw table
                     data/spatial/02_clean/       Spatial-join crosswalks (e.g. station → HUC-12)
src/03_merge/      → data/03a_merge_primary/      P1–P7: per-source merges onto station/county grain
                     data/03b_merge_secondary/    S1–S2: station-day + station-year context
                     data/final/     T1: epa-full.csv, the terminal modeling table
src/04_eda/        → epa-full.csv (in place)      3 read-only EDA notebooks + WQI calculation
src/05_modeling/   → src/05_modeling/<family>/*.pkl   4 training notebooks, 13 targets each
                     model_metrics.csv
build_station_table.py → data/stations.csv        one row per station, collapsed from epa-full.csv
app.py             ← data/stations.csv + <family>/*.pkl + model_metrics.csv
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

> **These are not the dashboard's accuracy.** Each score below was measured on held-out stations using the weather actually recorded on each sample's date. The dashboard supplies weather from each station's last visit instead, so its maps are less accurate — for water temperature, about 11 °C too warm in January. See [Current Status and Limitations](#current-status-and-limitations).

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

## Next Steps

The pipeline is built through modeling, all 52 models are trained and committed, and the dashboard launches locally with a single command. What remains is turning a well-evaluated prototype into something a person could make a decision with. [`NEXT_STEPS.md`](NEXT_STEPS.md) lays out the course of action in order:

0. **Get set up** — obtain the gitignored `data/` files and fix the out-of-date documentation
1. **Choose one user and one decision** — this determines the target, locations, time horizon and success metric
2. **Measure the dashboard's real accuracy** — backtest the app's own feature-building code on held-out observations
3. **Feed the models weather for the chosen date** — and stop implying that future dates are forecasts
4. **Only show predictions the model can support** — separate groundwater from surface water; hide unmeasured station types
5. **Show observed data beside predictions** — last value, date and sample count
6. **Replace labels with probabilities and ranges** — e.g. "chance above 235 MPN/100 mL"
7. **Then improve the models** — previous-observation features, a two-part model for the zero-heavy nitrogen targets, fewer redundant predictors
