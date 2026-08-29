"""
Water Quality Prediction Dashboard
====================================
Dash/Plotly app for predicting water quality across Iowa monitoring
stations using pre-trained scikit-learn models loaded from disk.

Expected files on the server (set paths in CONFIGURATION below):
  DATA_FILE_PATH          – data/03c_merge_tertiary/epa-full.csv
  MODEL_DIR               – src/05_modeling/, with one sub-folder per model
                            family, each holding one .pkl per target:
                              linear_regression/lr_<target>.pkl
                              random_forest/rf_<target>.pkl
                              gradient_boosting/gb_<target>.pkl
                              neural_network/nn_<target>.pkl
                            e.g. random_forest/rf_specific_conductance.pkl

Thirteen targets are supported (water temperature, dissolved oxygen, pH,
nitrate, nitrite, nitrate + nitrite, total phosphorus, specific conductance,
total dissolved solids, total suspended solids, turbidity, E. coli, and the
composite WQI), each in four model flavours — 52 pkl files in total.

Each .pkl holds a dict: {"pipeline", "target_transform", "log_offset",
"smearing_factor", "feature_cols"}. The pipeline's feature order must match
FEATURE_COLS exactly — it is checked at load time. The pipelines impute
missing predictors internally, so the app may pass NaNs straight through.

Some models were fitted on log10(y + c) and predict on that scale rather than
in the target's own units. Which ones is decided per (target, family) by a
cross-validated bake-off in the training notebooks, not by target name, so it
travels inside the .pkl; _to_raw_scale applies the inverse.

If a model file is missing the app still starts — that target/model
combination is simply disabled in the UI.
"""

import os
import pickle
import re
import warnings
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import dash
from dash import ALL, dcc, html, Input, Output, State, callback_context, no_update
import plotly.graph_objects as go

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# CONFIGURATION  ← edit these paths to match your server layout
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE_PATH = BASE_DIR / "data/03c_merge_tertiary/epa-full.csv"
MODEL_DIR = BASE_DIR / "src/05_modeling"        # folder containing the model sub-folders
METRICS_PATH = BASE_DIR / "src/05_modeling/model_metrics.csv"

# Column names in the CSV
DATE_COL = "ActivityStartDateTime"
LAT_COL  = "LatitudeMeasure"
LON_COL  = "LongitudeMeasure"

# Station-level predictors read straight from the CSV. These are the same
# 26 base features the notebooks in src/05_modeling/ trained on.
BASE_FEATURE_COLS = [
    # Location
    "LatitudeMeasure", "LongitudeMeasure",
    "distance_to_climate_station_km", "distance_to_streamflow_gauge_km",
    # PRISM climate normals at the observation
    "prism_tmax_c", "prism_tmin_c", "prism_ppt_mm", "prism_tdmean_c",
    # ISU station weather
    "isu_avg_wind_speed_kts", "isu_avg_rh", "isu_snow_in", "isu_snowd_in",
    "isu_max_feel_c", "isu_min_feel_c",
    # Hydrology
    "streamflow_discharge_cfs",
    # Soil
    "ksat_mean", "awc_mean",
    # Land cover. `pct_row_crops` is deliberately absent: it equals
    # pct_corn + pct_soybean exactly on every row, so including it made the
    # design matrix singular without adding a fact (eda-summary.md §4.1).
    "pct_corn", "pct_soybean", "pct_developed", "pct_forest",
    # Nutrient loading context
    "npfert__n__total_kg", "npfert__p__total_kg",
    "npmanure__total__n_kg", "npmanure__total__p_kg",
]

# Temporal predictors derived from the chosen prediction date at inference time.
TEMPORAL_FEATURE_COLS = ["doy", "doy_sin", "doy_cos", "obs_year"]

# Columns the models were trained on — ORDER MUST MATCH TRAINING.
FEATURE_COLS = BASE_FEATURE_COLS + TEMPORAL_FEATURE_COLS

# Target variable display name → CSV column name
TARGET_COLS = {
    "Water Temperature":      "Temperature, water_value",
    "Dissolved Oxygen":       "Dissolved oxygen (DO)_value",
    "pH":                     "pH_value",
    "Nitrate":                "Nitrate_value",
    "Nitrite":                "Nitrite_value",
    "Nitrate + Nitrite":      "Nitrate + Nitrite_value",
    "Total Phosphorus":       "Total Phosphorus, mixed forms_value",
    "Specific Conductance":   "Specific conductance_value",
    "Total Dissolved Solids": "Total dissolved solids_value",
    "Total Suspended Solids": "Total suspended solids_value",
    "Turbidity":              "Turbidity_value",
    "E. coli":                "Escherichia coli_value",
    # Composite index from src/04_eda/wqi-calculation.ipynb — 0 = best,
    # 100 = worst. Present on 56.6% of rows (943 stations).
    "WQI":                    "WQI",
}

# Model type display name → filename prefix. Iteration order is the display
# order everywhere — the model dropdown and the comparison table both walk this
# dict — so it runs simplest to most complex rather than alphabetically.
MODEL_PREFIXES = {
    "Linear Regression": "lr",
    "Random Forest":     "rf",
    "Gradient Boosting": "gb",
    "Neural Network":    "nn",
}
MODEL_SUBDIRS = {
    "Linear Regression": "linear_regression",
    "Random Forest":     "random_forest",
    "Gradient Boosting": "gradient_boosting",
    "Neural Network":    "neural_network",
}


def _stem(label: str) -> str:
    """'Nitrate + Nitrite' → 'nitrate_nitrite', 'E. coli' → 'e_coli' — matches the notebooks."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


# Target display name → safe filename stem (used to build .pkl paths)
TARGET_STEMS = {label: _stem(label) for label in TARGET_COLS}

TARGET_UNITS = {
    "Water Temperature":      "°C",
    "Dissolved Oxygen":       "mg/L",
    "pH":                     "pH",
    "Nitrate":                "mg/L as N",
    "Nitrite":                "mg/L as N",
    "Nitrate + Nitrite":      "mg/L as N",
    "Total Phosphorus":       "mg/L as P",
    "Specific Conductance":   "µS/cm",
    "Total Dissolved Solids": "mg/L",
    "Total Suspended Solids": "mg/L",
    "Turbidity":              "NTU",
    "E. coli":                "MPN/100mL",
    "WQI":                    "index",
}

TARGET_COLORSCALES = {
    "Water Temperature":      "RdYlBu_r",
    "Dissolved Oxygen":       "Blues",
    "pH":                     "RdYlGn",
    "Nitrate":                "YlOrRd",
    "Nitrite":                "YlOrRd",
    "Nitrate + Nitrite":      "YlOrRd",
    "Total Phosphorus":       "PuRd",
    "Specific Conductance":   "Viridis",
    "Total Dissolved Solids": "Cividis",
    "Total Suspended Solids": "YlOrBr",
    "Turbidity":              "YlOrBr",
    "E. coli":                "Reds",
    # WQI runs 0 = best to 100 = worst, so the ramp must NOT be reversed:
    # green at the low end, red at the high end.
    "WQI":                    "RdYlGn_r",
}

MODEL_DESCRIPTIONS = {
    "Gradient Boosting": "Highest-accuracy tree ensemble with stronger seasonal and nonlinear pattern capture.",
    "Random Forest": "Robust ensemble model with richer nonlinear behavior and stable predictions across stations.",
    "Linear Regression": "Fast baseline model with simpler, interpretable behavior.",
    "Neural Network": "Averaged multilayer perceptrons — smooth nonlinear fits, generally between the linear and tree models.",
}

TARGET_SHORT_NOTES = {
    "Water Temperature": "Use this to inspect seasonal warming and cooling patterns across stations.",
    "pH": "Use this to compare acidity and alkalinity patterns across Iowa waterways.",
    "Dissolved Oxygen": "Use this to spot areas where oxygen availability may be stronger or weaker.",
    "Nitrate": "Use this to inspect likely nutrient concentration hotspots across the network.",
    "Nitrite": "Use this to inspect nitrite levels, an intermediate nitrogen form, across the network.",
    "Nitrate + Nitrite": "Use this to inspect combined oxidized-nitrogen loading across the network.",
    "Total Phosphorus": "Use this to spot phosphorus enrichment that can drive algal growth.",
    "Specific Conductance": "Use this as a proxy for dissolved-ion content across waterways.",
    "Total Dissolved Solids": "Use this to compare overall dissolved mineral content.",
    "Total Suspended Solids": "Use this to inspect sediment and particulate load patterns.",
    "Turbidity": "Use this to compare water clarity across the network.",
    "E. coli": "Use this to inspect likely bacterial-contamination hotspots.",
    "WQI": "Use this as a single roll-up of overall degradation — 0 is best, 100 is worst.",
}


def _target_assessment(target: Optional[str], value: Optional[Union[float, int]]) -> Dict[str, str]:
    """
    Return a short qualitative interpretation for a predicted value.
    These labels are UI guidance only and should not be treated as a
    regulatory determination.
    """
    if target is None or value is None or pd.isna(value):
        return {
            "label": "Unknown",
            "detail": "No interpretation available.",
            "color": TEXT_MID,
        }

    val = float(value)

    if target == "pH":
        if val < 6.5:
            return {
                "label": "Acidic",
                "detail": "Below the common neutral-to-healthy range for many freshwater systems.",
                "color": DANGER,
            }
        if val <= 8.5:
            return {
                "label": "Balanced",
                "detail": "Within a common freshwater range and generally suitable for aquatic life.",
                "color": SUCCESS,
            }
        return {
            "label": "Basic",
            "detail": "Above the common freshwater range and more alkaline than neutral.",
            "color": "#b45309",
        }

    if target == "Water Temperature":
        if val < 5:
            return {
                "label": "Very Cold",
                "detail": "Cold-water conditions; fine for some species but stressful for others.",
                "color": "#1d4ed8",
            }
        if val <= 20:
            return {
                "label": "Moderate",
                "detail": "A generally favorable temperature range for many freshwater ecosystems.",
                "color": SUCCESS,
            }
        if val <= 28:
            return {
                "label": "Warm",
                "detail": "Warmer water can begin reducing oxygen availability and increase stress.",
                "color": "#d97706",
            }
        return {
            "label": "Very Warm",
            "detail": "Potentially stressful for aquatic life, especially when oxygen is limited.",
            "color": DANGER,
        }

    if target == "Dissolved Oxygen":
        if val < 5:
            return {
                "label": "Low Oxygen",
                "detail": "Below 5 mg/L and often stressful for aquatic life.",
                "color": DANGER,
            }
        if val < 7:
            return {
                "label": "Fair",
                "detail": "Above the minimum stress threshold, but not yet in the healthier range.",
                "color": "#d97706",
            }
        if val <= 19:
            return {
                "label": "Healthy",
                "detail": "Within the 7-19 mg/L range, which is generally healthy for freshwater habitats.",
                "color": SUCCESS,
            }
        return {
            "label": "Very High",
            "detail": "Above 19 mg/L; oxygen is abundant, though extremely high values can reflect unusual conditions.",
            "color": "#0891b2",
        }

    if target == "Nitrate":
        if val < 1:
            return {
                "label": "Low",
                "detail": "Low nitrate concentration and generally not a nutrient concern.",
                "color": SUCCESS,
            }
        if val < 3:
            return {
                "label": "Moderate",
                "detail": "Some nutrient presence, but not unusually elevated.",
                "color": "#65a30d",
            }
        if val < 10:
            return {
                "label": "Elevated",
                "detail": "Higher nitrate levels that may indicate runoff or nutrient loading.",
                "color": "#d97706",
            }
        return {
            "label": "High Concern",
            "detail": "Strongly elevated nitrate and a clearer water-quality concern.",
            "color": DANGER,
        }

    # ── Additional targets ────────────────────────────────────────
    # Simple, band-based guidance drawn from commonly cited freshwater and
    # drinking/recreational-water reference points. UI guidance only.
    if target in ("Nitrite", "Nitrate + Nitrite"):
        # Drinking-water MCL: nitrite 1 mg/L-N, nitrate+nitrite 10 mg/L-N.
        hi = 1.0 if target == "Nitrite" else 10.0
        if val < 0.3 * hi:
            return {"label": "Low", "detail": "Low oxidized-nitrogen concentration.", "color": SUCCESS}
        if val < hi:
            return {"label": "Elevated", "detail": "Detectable nitrogen loading, below the drinking-water limit.", "color": "#d97706"}
        return {"label": "High Concern", "detail": "Above the drinking-water reference limit.", "color": DANGER}

    if target == "Total Phosphorus":
        if val < 0.05:
            return {"label": "Low", "detail": "Below common stream targets; limited algal-growth risk.", "color": SUCCESS}
        if val < 0.1:
            return {"label": "Moderate", "detail": "Near typical stream targets (~0.075 mg/L).", "color": "#65a30d"}
        if val < 0.3:
            return {"label": "Elevated", "detail": "Phosphorus enrichment that can promote algal growth.", "color": "#d97706"}
        return {"label": "High Concern", "detail": "Strongly enriched; elevated eutrophication risk.", "color": DANGER}

    if target == "Specific Conductance":
        if val < 500:
            return {"label": "Low", "detail": "Low dissolved-ion content.", "color": SUCCESS}
        if val < 1500:
            return {"label": "Moderate", "detail": "Typical range for many Midwestern streams.", "color": "#65a30d"}
        return {"label": "Elevated", "detail": "High ionic content, often reflecting runoff or discharge.", "color": "#d97706"}

    if target == "Total Dissolved Solids":
        if val < 500:
            return {"label": "Good", "detail": "Within the secondary drinking-water guideline (500 mg/L).", "color": SUCCESS}
        if val < 1000:
            return {"label": "Moderate", "detail": "Above the aesthetic guideline but common in surface water.", "color": "#d97706"}
        return {"label": "Elevated", "detail": "High dissolved-solids content.", "color": DANGER}

    if target == "Total Suspended Solids":
        if val < 25:
            return {"label": "Clear", "detail": "Low suspended-sediment load.", "color": SUCCESS}
        if val < 80:
            return {"label": "Moderate", "detail": "Noticeable sediment load.", "color": "#d97706"}
        return {"label": "Turbid", "detail": "High sediment load that can stress aquatic habitat.", "color": DANGER}

    if target == "Turbidity":
        if val < 5:
            return {"label": "Clear", "detail": "Clear water with low particulate scattering.", "color": SUCCESS}
        if val < 25:
            return {"label": "Moderate", "detail": "Some cloudiness from suspended particles.", "color": "#d97706"}
        return {"label": "Very Turbid", "detail": "Cloudy water, often after runoff or disturbance.", "color": DANGER}

    if target == "E. coli":
        # EPA recreational water: ~126 MPN/100mL geomean, ~235 single-sample.
        if val < 126:
            return {"label": "Low", "detail": "Below the recreational-water geomean reference (126 MPN/100mL).", "color": SUCCESS}
        if val < 235:
            return {"label": "Elevated", "detail": "Between the geomean and single-sample recreational thresholds.", "color": "#d97706"}
        return {"label": "High Concern", "detail": "Above the single-sample recreational threshold (235 MPN/100mL).", "color": DANGER}

    if target == "WQI":
        # Note the direction: this index runs 0 = best to 100 = worst, the
        # opposite of most "quality score" scales. Bands follow the quartiles of
        # the observed distribution (median 43.3, IQR 31.5-57.5) rather than any
        # published standard — this index is defined in
        # src/04_eda/wqi-calculation.ipynb, not by a regulator.
        if val < 31.5:
            return {"label": "Least Degraded", "detail": "In the cleanest quarter of observed samples (index below 31.5).", "color": SUCCESS}
        if val < 57.5:
            return {"label": "Typical", "detail": "Within the middle half of observed samples (index 31.5-57.5).", "color": "#d97706"}
        return {"label": "Most Degraded", "detail": "In the most degraded quarter of observed samples (index above 57.5).", "color": DANGER}

    return {
        "label": "Measured",
        "detail": "Predicted value available.",
        "color": TEXT_MID,
    }

# Major Iowa cities for reverse-geocoding interpolated hover points
IOWA_CITIES = [
    ("Des Moines",     41.5868, -93.6250),
    ("Cedar Rapids",   41.9779, -91.6656),
    ("Davenport",      41.5236, -90.5776),
    ("Sioux City",     42.4999, -96.4003),
    ("Iowa City",      41.6611, -91.5302),
    ("Waterloo",       42.4928, -92.3426),
    ("Council Bluffs", 41.2619, -95.8608),
    ("Ames",           42.0308, -93.6319),
    ("Dubuque",        42.5006, -90.6646),
    ("Ankeny",         41.7321, -93.6030),
    ("West Des Moines",41.5772, -93.7113),
    ("Cedar Falls",    42.5349, -92.4452),
    ("Marion",         42.0341, -91.5974),
    ("Bettendorf",     41.5244, -90.5121),
    ("Urbandale",      41.6265, -93.7122),
    ("Mason City",     43.1536, -93.2010),
    ("Ottumwa",        41.0200, -92.4113),
    ("Marshalltown",   42.0494, -92.9080),
    ("Clinton",        41.8444, -90.1887),
    ("Burlington",     40.8073, -91.1128),
    ("Fort Dodge",     42.4975, -94.1680),
    ("Muscatine",      41.4245, -91.0432),
    ("Coralville",     41.6727, -91.5802),
    ("Waukee",         41.6113, -93.8888),
    ("North Liberty",  41.7494, -91.6052),
    ("Oskaloosa",      41.2961, -92.6457),
    ("Storm Lake",     42.6411, -95.2097),
    ("Carroll",        42.0661, -94.8672),
    ("Fairfield",      41.0086, -91.9657),
    ("Spencer",        43.1414, -95.1441),
]


def _nearest_city(lat: float, lon: float) -> str:
    """Return the name of the closest Iowa city to the given coordinates."""
    best_name, best_dist = "Iowa", float("inf")
    for name, clat, clon in IOWA_CITIES:
        dist = (lat - clat) ** 2 + (lon - clon) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _nearest_station_name(lat: float, lon: float) -> str:
    """Return the MonitoringLocationName of the closest station."""
    dists = (STATIONS[LAT_COL] - lat) ** 2 + (STATIONS[LON_COL] - lon) ** 2
    idx = dists.idxmin()
    name = STATIONS.loc[idx, "MonitoringLocationName"]
    return str(name) if pd.notna(name) else "Unknown"


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# Previously: models were trained here at startup (slow, heavy).
# Now:        we deserialise pre-trained .pkl files from disk
#             (fast, lightweight — no training data required at runtime).
#
# Key differences vs. the training approach:
#   • No sklearn fit() call at all — just pickle.load()
#   • No dependency on target columns in the CSV for training
#   • Models load in milliseconds instead of seconds
#   • FEATURE_COLS order is contractual: the pkl was trained with this
#     exact column order, so we must replicate it faithfully at inference
# ─────────────────────────────────────────────────────────────
def load_models() -> tuple:
    """
    Walk MODEL_DIR/<family>/ looking for files named {prefix}_{stem}.pkl.

    Each .pkl holds a dict written by the training notebooks:

        {"pipeline": Pipeline, "target_transform": "none" | "log10",
         "log_offset": float | None, "smearing_factor": float,
         "feature_cols": [...]}

    Whether a model was fitted on the raw target or on log10(y + c) is decided
    per (target, family) by a cross-validated bake-off in the notebooks, so it
    cannot be inferred from the target name — it travels inside the artifact
    instead, which is what stops a log10 prediction from ever being rendered as
    mg/L because a metrics file went stale.

    Returns (models, transforms):
      models[target][family]     -> the fitted pipeline
      transforms[(target, family)] -> (offset, smearing) or None if raw-scale

    Missing files are skipped with a warning; the app still starts.
    """
    models = {target: {} for target in TARGET_COLS}
    transforms: Dict[tuple, Optional[Tuple[float, float]]] = {}

    for target_label, stem in TARGET_STEMS.items():
        for model_label, prefix in MODEL_PREFIXES.items():
            pkl_path = MODEL_DIR / MODEL_SUBDIRS[model_label] / f"{prefix}_{stem}.pkl"
            if not pkl_path.exists():
                print(f"[WARNING] Model not found: {pkl_path} — "
                      f"'{target_label} / {model_label}' will be unavailable.")
                continue

            with open(pkl_path, "rb") as f:
                artifact = pickle.load(f)

            # Pre-transform pkl files held a bare Pipeline. Those were always
            # raw-scale, so treating them as such stays correct.
            if not isinstance(artifact, dict):
                models[target_label][model_label] = artifact
                transforms[(target_label, model_label)] = None
                print(f"[INFO] Loaded model (legacy bare pipeline): {pkl_path}")
                continue

            saved_cols = artifact.get("feature_cols")
            if saved_cols is not None and list(saved_cols) != FEATURE_COLS:
                print(f"[WARNING] {pkl_path} was trained on a different feature "
                      f"set ({len(saved_cols)} cols) than app.py expects "
                      f"({len(FEATURE_COLS)}) — '{target_label} / {model_label}' "
                      "disabled to avoid silently mismatched predictions.")
                continue

            models[target_label][model_label] = artifact["pipeline"]
            if artifact.get("target_transform") == "log10":
                transforms[(target_label, model_label)] = (
                    float(artifact["log_offset"]),
                    float(artifact["smearing_factor"]),
                )
            else:
                transforms[(target_label, model_label)] = None
            print(f"[INFO] Loaded model: {pkl_path}")

    loaded = sum(len(v) for v in models.values())
    n_log = sum(1 for v in transforms.values() if v is not None)
    print(f"[INFO] {loaded} model(s) loaded from '{MODEL_DIR}/' "
          f"({n_log} fitted on log10(y + c))")
    return models, transforms


def load_model_metrics() -> pd.DataFrame:
    """Load saved training metrics for display in the dashboard."""
    metrics_path = METRICS_PATH
    if not metrics_path.exists():
        print(f"[WARNING] Metrics file not found: {metrics_path}")
        return pd.DataFrame()

    metrics = pd.read_csv(metrics_path)
    print(f"[INFO] Loaded model metrics from '{metrics_path}'")
    return metrics


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# The CSV is still needed at runtime — but only for station
# locations and feature values, not for training targets.
# ─────────────────────────────────────────────────────────────
def load_station_data() -> pd.DataFrame:
    """
    Load the CSV and return one representative row per monitoring station.
    We keep the most-recent observation per station so that lat/lon and
    feature values are up to date without duplicating stations on the map.
    """
    df = pd.read_csv(DATA_FILE_PATH, parse_dates=[DATE_COL])
    df = df.sort_values(DATE_COL)

    # Deduplicate: one row per station (latest observation)
    stations = (
        df.groupby("MonitoringLocationIdentifier", sort=False)
          .last()
          .reset_index()
    )
    print(f"[INFO] Loaded {len(stations)} unique monitoring stations "
          f"from '{DATA_FILE_PATH}'")
    return stations


MODELS, MODEL_TRANSFORMS = load_models()
STATIONS = load_station_data()
MODEL_METRICS = load_model_metrics()


# ─────────────────────────────────────────────────────────────
# TARGET BACK-TRANSFORM
# Some models were fitted on log10(y + c) and so predict on that
# scale. Which ones is not a property of the target — the
# notebooks choose per (target, family) via a cross-validated
# bake-off, and e.g. total phosphorus takes the log for the two
# tree families but not for linear regression.
#
# The parameters ride inside the .pkl, so this cannot fall out
# of sync with a stale model_metrics.csv. The inverse itself
# lives here rather than in the pickle because baking it in
# would need a custom estimator class importable at unpickle
# time — the fragility that got model_feature_engineering.py
# deleted. A dict of a Pipeline and two floats needs no such
# class.
# ─────────────────────────────────────────────────────────────
def _to_raw_scale(target: str, model_type: str, y_pred: np.ndarray) -> np.ndarray:
    """Convert a model's output into the target's own units.

    A no-op for a raw-scale model. For a log-fitted one this undoes
    log10(y + c) and applies Duan's smearing factor — without it the
    back-transform is biased low, because E[y] is not 10 ** E[log10 y].
    Clipped at zero: every log-fitted target here is a concentration.
    """
    params = MODEL_TRANSFORMS.get((target, model_type))
    if params is None:
        return y_pred
    offset, smearing = params
    return np.clip((10.0 ** y_pred) * smearing - offset, 0.0, None)


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# Mirrors exactly what was done at training time.
# The only NEW input here is the user-chosen prediction date;
# all other features come from the station's most-recent record.
# ─────────────────────────────────────────────────────────────
def _add_temporal_features(base: pd.DataFrame, day_of_year: int, year: int) -> pd.DataFrame:
    """
    Append the four date-derived predictors to a base feature frame and return
    the columns in the exact FEATURE_COLS training order.

    The cyclical doy_sin/doy_cos encoding mirrors the notebooks so that day 365
    sits next to day 1. NaNs in the base columns are left in place — every
    fitted pipeline imputes them internally with the training-set medians.
    """
    X = base.copy()
    X["doy"] = day_of_year
    radians = 2.0 * np.pi * day_of_year / 365.25
    X["doy_sin"] = np.sin(radians)
    X["doy_cos"] = np.cos(radians)
    X["obs_year"] = year
    return X[FEATURE_COLS]   # enforce column order contract


def build_feature_matrix(pred_date: date) -> pd.DataFrame:
    """
    Construct the inference feature matrix for every station.

    For a future prediction date we know:
      doy / doy_sin / doy_cos / obs_year — derived directly from pred_date
      All station/climate/soil/land-cover columns — taken from the station's
        historical record (best available proxy when a forecast is not provided)

    Returns a DataFrame with columns in FEATURE_COLS order, one row per station.
    """
    return _add_temporal_features(
        STATIONS[BASE_FEATURE_COLS],
        pred_date.timetuple().tm_yday,
        pred_date.year,
    )


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPER
# ─────────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _cached_station_predictions(target: str, model_type: str, iso_date: str) -> tuple:
    """
    Raw-scale predictions for every station, memoised on (target, family, date).

    The cache exists so that redrawing the same prediction — which the
    interpolation toggle does on every click — costs nothing. Returns a tuple
    because lru_cache needs a hashable value; the caller re-wraps it.
    """
    X   = build_feature_matrix(date.fromisoformat(iso_date))
    mdl = MODELS[target][model_type]       # already-fitted pipeline from pkl

    raw = mdl.predict(X.to_numpy())        # inference only — no fit() call
    return tuple(_to_raw_scale(target, model_type, raw).tolist())


def predict_at_stations(target: str, model_type: str, pred_date: date) -> pd.DataFrame:
    """
    Run the loaded model for all stations at pred_date.
    Returns STATIONS with an added 'predicted' column.
    """
    result = STATIONS.copy()
    result["predicted"] = np.asarray(
        _cached_station_predictions(target, model_type, pred_date.isoformat()),
        dtype=float,
    )
    return result


# ─────────────────────────────────────────────────────────────
# SPATIAL INTERPOLATION
# ─────────────────────────────────────────────────────────────
def interpolate_to_grid(df: pd.DataFrame, resolution: int = 120) -> tuple:
    """
    Interpolate scattered station predictions onto a regular lat/lon grid.
    Uses cubic spline (smooth) with linear fallback at convex-hull edges.
    Returns (lon_grid, lat_grid, value_grid).
    """
    # Bounding box derived from actual station extents + small margin
    lat_min = df[LAT_COL].min() - 0.5
    lat_max = df[LAT_COL].max() + 0.5
    lon_min = df[LON_COL].min() - 0.5
    lon_max = df[LON_COL].max() + 0.5

    lons = np.linspace(lon_min, lon_max, resolution)
    lats = np.linspace(lat_min, lat_max, resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    points = df[[LON_COL, LAT_COL]].values
    values = df["predicted"].values

    grid        = griddata(points, values, (lon_grid, lat_grid), method="cubic")
    grid_linear = griddata(points, values, (lon_grid, lat_grid), method="linear")
    grid        = np.where(np.isnan(grid), grid_linear, grid)  # fill edge NaNs

    return lon_grid, lat_grid, grid


# ─────────────────────────────────────────────────────────────
# UI STYLE CONSTANTS
# ─────────────────────────────────────────────────────────────
NAVY         = "#0b1929"          # header background
ACCENT       = "#2563eb"          # primary interactive blue
ACCENT_DIM   = "#1e50c0"          # hover / pressed
BORDER       = "#e4eaf2"
TEXT_DARK    = "#0f172a"
TEXT_MID     = "#4b5a6e"
TEXT_LIGHT   = "#8fa3b8"
BG_WHITE     = "#ffffff"
BG_PAGE      = "#f0f4fa"          # subtle blue-tinted page
SUCCESS      = "#15803d"
DANGER       = "#dc2626"
FONT         = "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

CARD_STYLE = {
    "background": BG_WHITE,
    "borderRadius": "14px",
    "border": "none",
    "boxShadow": "0 1px 4px rgba(11,25,41,0.08), 0 0 0 1px rgba(11,25,41,0.04)",
    "padding": "22px",
    "marginBottom": "0",
}

SIDECARD_STYLE = {
    "padding": "18px 20px",
}

LABEL_STYLE = {
    "fontFamily": FONT,
    "fontSize": "10px",
    "fontWeight": "700",
    "letterSpacing": "0.07em",
    "textTransform": "uppercase",
    "color": TEXT_LIGHT,
    "marginBottom": "8px",
    "display": "block",
}


def _triggered_id():
    """
    callback_context.triggered_id, or None when there is no request context.

    The callbacks are called directly by test_app.py, where Dash raises rather
    than reporting "nothing triggered this".
    """
    try:
        return callback_context.triggered_id
    except Exception:
        return None


def _panel_heading(text: str, hint: Optional[str] = None) -> html.Div:
    """Small uppercase heading used at the top of every right-rail panel."""
    children = [html.Span(text, className="panel-heading-text")]
    if hint:
        children.append(html.Span(hint, className="panel-heading-hint"))
    return html.Div(className="panel-heading", children=children)

# Outer style for the target/model dcc.Dropdowns. The inner control is a
# React-Select widget, so its border, height and menu are styled in
# assets/dashboard.css (.Select-*) rather than here.
DROPDOWN_STYLE = {
    "fontFamily": FONT,
    "fontSize": "13px",
    "color": TEXT_DARK,
}

DIVIDER_STYLE = {
    "height": "1px",
    "background": BORDER,
    "margin": "0",
}

STREAK_WEEKS = 14
STREAK_LEVELS = [
    "#ebedf0",
    "#d9f3ea",
    "#9adac2",
    "#49b88d",
    "#1f6f55",
]


# ─────────────────────────────────────────────────────────────
# FIGURE HELPERS
# ─────────────────────────────────────────────────────────────

# Neighboring-state label positions (within Iowa's viewport)
_NEIGHBOR_LABELS = [
    ("MINNESOTA",   44.3, -94.2),
    ("WISCONSIN",   43.4, -90.4),
    ("ILLINOIS",    41.0, -90.1),
    ("MISSOURI",    39.7, -92.8),
    ("NEBRASKA",    41.6, -96.7),
    ("S. DAKOTA",   43.9, -97.1),
]
_IOWA_LABEL = ("IOWA", 42.1, -93.5)

# Major Iowa city markers for map reference. Plotly's `scope="usa"` basemap
# draws no place names at all, so without these the map is an outline: a user
# looking at a dot has no way to say where in Iowa it is.
_IOWA_CITY_MARKERS = [
    ("Des Moines",      41.5868, -93.6250),
    ("Cedar Rapids",    41.9779, -91.6656),
    ("Davenport",       41.5236, -90.5776),
    ("Sioux City",      42.4999, -96.4003),
    ("Iowa City",       41.6611, -91.5302),
    ("Waterloo",        42.4928, -92.3426),
    ("Council Bluffs",  41.2619, -95.8608),
    ("Ames",            42.0308, -93.6319),
    ("Dubuque",         42.5006, -90.6646),
    ("Mason City",      43.1536, -93.2010),
    ("Fort Dodge",      42.4975, -94.1680),
    ("Ottumwa",         41.0200, -92.4113),
]

# River name labels, hand-placed on a point the named river actually runs
# through — Natural Earth (which is what `showrivers` draws) carries the
# geometry but no labels, and Plotly cannot label it for us.
#
# Each position is anchored between two known river towns: e.g. the Mississippi
# label sits between Clinton (41.84, -90.19) and Dubuque (42.50, -90.66), the
# Missouri label between Sioux City (42.50, -96.40) and Council Bluffs
# (41.26, -95.86). These are labels of convenience, not surveyed placements.
_IOWA_RIVER_LABELS = [
    ("Mississippi R.", 42.15, -90.38),
    ("Missouri R.",    41.90, -96.13),
    ("Des Moines R.",  42.22, -93.98),
    ("Cedar R.",       42.25, -92.03),
    ("Iowa R.",        41.76, -91.85),
    ("Big Sioux R.",   43.20, -96.53),
]

# Named lakes and reservoirs, largest/most recognisable first. The bigger ones
# also render as polygons from `showlakes`; these supply the names.
_IOWA_LAKE_LABELS = [
    ("Iowa Great Lakes", 43.42, -95.14),
    # Nudged onto the water west of the town of Clear Lake, which also keeps
    # the label clear of Mason City's.
    ("Clear Lake",       43.13, -93.47),
    ("Storm Lake",       42.63, -95.20),
    # Saylorville and Coralville reservoirs are deliberately absent: both sit
    # close enough to Des Moines and Iowa City that their labels collide with
    # the city they are next to, and the city is the better landmark.
    ("Lake Red Rock",    41.38, -92.98),
    ("Rathbun Lake",     40.86, -92.88),
]


# A white halo, so a label stays readable where it lands on top of a dense
# cluster of station dots. `textfont.shadow` takes a CSS text-shadow value and
# needs plotly >= 5.23 / plotly.js >= 2.35; this repo is on 6.7.0.
_LABEL_HALO = ("1px 1px 2px #ffffff, -1px -1px 2px #ffffff, "
               "1px -1px 2px #ffffff, -1px 1px 2px #ffffff")


def _add_map_labels(fig: go.Figure, show_landmarks: bool = True) -> None:
    """
    Add the state labels, and — when `show_landmarks` — city, river and lake
    reference labels.

    Every trace added here is `hoverinfo="skip"` and `showlegend=False`, so
    landmarks can never capture a hover meant for a station or shift the
    legend. They are appended after the data traces so their text renders on
    top and stays legible.
    """
    # The big centred "IOWA" watermark sits at (42.1, -93.5), which is all but
    # on top of Ames — so it is only drawn when the city labels are not. With
    # landmarks on, the named cities identify the state far better anyway.
    if not show_landmarks:
        fig.add_trace(go.Scattergeo(
            lat=[_IOWA_LABEL[1]],
            lon=[_IOWA_LABEL[2]],
            mode="text",
            text=[_IOWA_LABEL[0]],
            textfont=dict(size=14, color=ACCENT, family=FONT, shadow=_LABEL_HALO),
            showlegend=False,
            hoverinfo="skip",
        ))
    fig.add_trace(go.Scattergeo(
        lat=[r[1] for r in _NEIGHBOR_LABELS],
        lon=[r[2] for r in _NEIGHBOR_LABELS],
        mode="text",
        text=[r[0] for r in _NEIGHBOR_LABELS],
        textfont=dict(size=9, color="#9fb8cc", family=FONT, shadow=_LABEL_HALO),
        showlegend=False,
        hoverinfo="skip",
    ))

    if not show_landmarks:
        return

    # Rivers — italic-blue text sitting on the drawn river lines.
    fig.add_trace(go.Scattergeo(
        lat=[r[1] for r in _IOWA_RIVER_LABELS],
        lon=[r[2] for r in _IOWA_RIVER_LABELS],
        mode="text",
        text=[r[0] for r in _IOWA_RIVER_LABELS],
        textfont=dict(size=8.5, color="#2d6d94", family=FONT, style="italic",
                      shadow=_LABEL_HALO),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Lakes — a small filled dot plus its name.
    fig.add_trace(go.Scattergeo(
        lat=[r[1] for r in _IOWA_LAKE_LABELS],
        lon=[r[2] for r in _IOWA_LAKE_LABELS],
        mode="markers+text",
        text=[r[0] for r in _IOWA_LAKE_LABELS],
        textposition="bottom center",
        textfont=dict(size=8.5, color="#2d6d94", family=FONT, shadow=_LABEL_HALO),
        marker=dict(size=5, color="#6aa5cc", symbol="circle",
                    line=dict(color="white", width=0.8)),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Cities — a dark square, deliberately a different shape from the circular
    # station markers so the two are never confused.
    fig.add_trace(go.Scattergeo(
        lat=[c[1] for c in _IOWA_CITY_MARKERS],
        lon=[c[2] for c in _IOWA_CITY_MARKERS],
        mode="markers+text",
        text=[c[0] for c in _IOWA_CITY_MARKERS],
        textposition="top center",
        textfont=dict(size=9.5, color="#1d2b3a", family=FONT, weight=600,
                      shadow=_LABEL_HALO),
        marker=dict(size=5.5, color="#33465c", symbol="square",
                    line=dict(color="white", width=0.9)),
        showlegend=False,
        hoverinfo="skip",
    ))


def empty_map_figure(show_landmarks: bool = True) -> go.Figure:
    """Base map with station dots — shown before first prediction."""
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=STATIONS[LAT_COL],
        lon=STATIONS[LON_COL],
        mode="markers",
        customdata=np.column_stack([
            STATIONS["MonitoringLocationName"].fillna("Unknown station"),
            STATIONS["MonitoringLocationIdentifier"].fillna("Unknown ID"),
            STATIONS["ProviderName"].fillna("Unknown provider"),
            STATIONS["climate_station_name"].fillna("Unknown climate station"),
            STATIONS["distance_to_climate_station_km"].fillna(0.0),
        ]),
        marker=dict(size=6, color=ACCENT, opacity=0.55, line=dict(width=0)),
        name="Monitoring stations",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Station ID: %{customdata[1]}<br>"
            "Provider: %{customdata[2]}<br>"
            "Climate station: %{customdata[3]}<br>"
            "Distance to climate station: %{customdata[4]:.1f} km<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<extra></extra>"
        ),
    ))
    _add_map_labels(fig, show_landmarks)
    _apply_geo_layout(fig)
    return fig


def _apply_geo_layout(fig: go.Figure, height: int = 560) -> None:
    """Apply consistent geo + paper layout to a figure in-place."""
    fig.update_layout(
        geo=dict(
            scope="usa",
            projection_type="albers usa",
            # 1:50m Natural Earth rather than the 1:110m default. At one
            # state's extent the coarse set reduces Iowa's rivers to a couple
            # of stubs; 50m resolves the Mississippi, Missouri, Des Moines,
            # Cedar and Iowa rivers and the larger lakes. It is the finest
            # Plotly offers for geo traces.
            resolution=50,
            showland=True,    landcolor="#e8edf5",
            showlakes=True,   lakecolor="#bcd6ee",
            showrivers=True,  rivercolor="#8bb8d8", riverwidth=1.1,
            showcoastlines=True, coastlinecolor="#7a9ab5",
            showsubunits=True,   subunitcolor="#8bafc8",
            subunitwidth=1.5,
            bgcolor="#dde7f2",
            center=dict(lat=42.0, lon=-93.5),
            lataxis_range=[39.0, 45.0],
            lonaxis_range=[-97.5, -89.5],
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=BG_WHITE,
        plot_bgcolor=BG_WHITE,
        font=dict(family=FONT, size=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=0.02,
            xanchor="left", x=0.02,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=BORDER, borderwidth=1,
            font=dict(size=11),
        ),
        height=height,
    )


def _stat_tile(label: str, value: str, unit: str = "") -> html.Div:
    return html.Div(
        className="stat-tile",
        children=[
            html.Div(
                className="stat-tile-value",
                children=[
                    html.Span(value),
                    html.Span(unit, className="stat-tile-unit") if unit else None,
                ],
            ),
            html.Div(label, className="stat-tile-label"),
        ],
    )


def _info_row(label: str, value: str, value_color: str = TEXT_DARK,
              hint: Optional[str] = None) -> html.Div:
    """One label/value line. `hint` becomes the native tooltip on hover."""
    return html.Div(
        className="info-row",
        title=hint,
        children=[
            html.Span(label, className="info-row-label"),
            html.Span(value, className="info-row-value", style={"color": value_color}),
        ],
    )


def _fmt_metric(value: Optional[Union[float, int]], suffix: str = "", digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _fmt_num(value: Optional[Union[float, int]]) -> str:
    """
    Format an error metric with a magnitude-appropriate number of decimals, so
    a 0.471 mg/L phosphorus RMSE and a 9,763 MPN/100mL E. coli RMSE both read
    cleanly in the same column.
    """
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    magnitude = abs(value)
    if magnitude >= 1000:
        return f"{value:,.0f}"
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_signed(value: Optional[Union[float, int]], digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "—"
    value = float(value)
    return f"{'+' if value >= 0 else '−'}{abs(value):.{digits}f}"


def _get_metric_row(target: Optional[str], model_type: Optional[str]) -> Optional[pd.Series]:
    if not target or not model_type or MODEL_METRICS.empty:
        return None

    match = MODEL_METRICS[
        (MODEL_METRICS["target"] == target) &
        (MODEL_METRICS["model"] == model_type)
    ]
    if match.empty:
        return None
    return match.iloc[0]


def _performance_panel(target: Optional[str], model_type: Optional[str]) -> html.Div:
    metric_row = _get_metric_row(target, model_type)

    if metric_row is None:
        return html.Div(
            style=SIDECARD_STYLE,
            children=[
                _panel_heading("Model accuracy"),
                html.Div(
                    "Pick a measurement and model to see how it scored on held-out stations.",
                    className="panel-empty",
                ),
            ],
        )

    r2_val     = float(metric_row.get("r2")   or 0)
    rmse       = float(metric_row.get("rmse") or 0)
    error_rate = metric_row.get("error_rate")
    unit       = TARGET_UNITS.get(target, "")

    rows = [
        _panel_heading("Model accuracy", f"{model_type}"),
        _info_row("Model score (R²)", _fmt_metric(r2_val),
                  hint="Share of held-out variance explained. 1.0 is perfect, 0.0 matches "
                       "predicting the mean, and negative is worse than the mean."),
    ]

    # For a log-fitted target the raw-scale R² above is dominated by the same
    # extreme tail the log transform exists to de-emphasise, so it understates
    # the model on its own. Show the score on the scale it was fitted and is
    # read on — for E. coli that is also the scale the EPA criterion uses.
    r2_log = metric_row.get("r2_log")
    is_log_fitted = r2_log is not None and not pd.isna(r2_log)
    if is_log_fitted:
        rows.append(_info_row(
            "Model score (R², log scale)", _fmt_metric(float(r2_log)),
            hint="This model was fitted on log10(y + c); this is the score on the "
                 "scale it was actually fitted and should be read on.",
        ))

    rows.append(_info_row("Avg error (RMSE)", f"{_fmt_num(rmse)} {unit}",
                          hint="Root mean squared error, in the target's own units."))
    mae = metric_row.get("mae")
    if mae is not None and not pd.isna(mae):
        rows.append(_info_row("Median-ish error (MAE)", f"{_fmt_num(float(mae))} {unit}",
                              hint="Mean absolute error, in the target's own units."))
    if error_rate is not None and not pd.isna(error_rate):
        rows.append(_info_row(
            "Typical error rate", f"{float(error_rate):.0f}%",
            hint="sMAPE. It tracks a target's zero fraction more than model quality — "
                 "prefer MAE/RMSE on the zero-inflated targets.",
        ))

    # The memorization bar: "repeat this station's previous value" scored on the
    # same held-out rows. A model that does not clear it is recalling the site
    # rather than predicting the water.
    #
    # For a log-fitted target, compare on the log scale. On raw units these
    # targets' extreme tails make persistence catastrophically bad (R² down to
    # −0.99, worse than predicting the mean), so a raw-scale margin of +1.10
    # flatters the model rather than testing it.
    if is_log_fitted:
        persistence = metric_row.get("persistence_r2_log")
        margin      = metric_row.get("model_minus_persistence_log")
        baseline_label = "Repeat-last-value baseline (R², log scale)"
    else:
        persistence = metric_row.get("persistence_r2")
        margin      = metric_row.get("model_minus_persistence")
        baseline_label = "Repeat-last-value baseline (R²)"
    baseline_hint = ("\"This station's next value equals its previous value\" — a model with "
                     "no features at all, scored on the same held-out rows. It is the "
                     "memorization bar the model has to clear.")
    if persistence is not None and not pd.isna(persistence):
        rows.append(_info_row(baseline_label, _fmt_metric(float(persistence)),
                              hint=baseline_hint))
    if margin is not None and not pd.isna(margin):
        margin = float(margin)
        rows.append(_info_row(
            "Beats the baseline by",
            f"{_fmt_signed(margin)} R²",
            SUCCESS if margin >= 0 else DANGER,
            hint=baseline_hint,
        ))

    stations = metric_row.get("test_stations")
    if stations is not None and not pd.isna(stations):
        note = (f"Scored on {int(stations):,} monitoring stations the model "
                "never saw during training.")
        if is_log_fitted:
            note += (" This target is right-skewed enough that this model is fitted on "
                     "log10 — read the log-scale score; the raw-scale one is set by "
                     "a handful of extreme readings.")
        rows.append(html.Div(note, className="panel-footnote"))

    return html.Div(style=SIDECARD_STYLE, children=rows)


def _hover_panel_default() -> html.Div:
    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            _panel_heading("Point detail"),
            html.Div(
                "Hover any station on the map to see its predicted value, "
                "interpretation and source metadata.",
                className="panel-empty",
            ),
        ],
    )


def _summary_panel_default() -> html.Div:
    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            _panel_heading("Statewide spread"),
            html.Div("Run a prediction to see the low / average / high across all "
                     "monitoring stations.", className="panel-empty"),
        ],
    )


def _streak_panel_default() -> html.Div:
    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            _panel_heading("Recent trend"),
            html.Div(
                f"Run a prediction to unlock the {STREAK_WEEKS}-week statewide-average heatmap.",
                className="panel-empty",
            ),
        ],
    )


def _available_targets() -> list:
    """
    Return dropdown options for all targets.
    Disabled if no pkl was found for that target.
    """
    options = []
    for label in TARGET_COLS:
        has_model = bool(MODELS.get(label))
        options.append({
            "label": label if has_model else f"{label} (model unavailable)",
            "value": label,
            "disabled": not has_model,
        })
    return options


def _available_model_types(target) -> list:
    """Return model type options, disabling any not loaded for the given target."""
    options = []
    for mt in MODEL_PREFIXES:
        available = bool(target and MODELS.get(target, {}).get(mt))
        options.append({
            "label": mt if (not target or available) else f"{mt} (unavailable)",
            "value": mt,
            "disabled": (target is not None and not available),
        })
    return options


def _build_feature_matrix_for_doy(day_of_year: int, year: int) -> pd.DataFrame:
    """
    Construct the inference matrix for a specific day-of-year and year.
    Only the temporal signal changes; station-level features stay fixed.
    """
    return _add_temporal_features(STATIONS[BASE_FEATURE_COLS], day_of_year, year)


@lru_cache(maxsize=4096)
def _statewide_mean_prediction(target: str, model_type: str, day_of_year: int, year: int) -> float:
    """Cache statewide mean predictions for streak rendering."""
    X = _build_feature_matrix_for_doy(day_of_year, year)
    preds = MODELS[target][model_type].predict(X.to_numpy())
    return float(np.mean(_to_raw_scale(target, model_type, preds)))


def _streak_level(value: float, low: float, high: float) -> int:
    if high <= low:
        return 2

    scaled = (value - low) / (high - low)
    if scaled < 0.2:
        return 0
    if scaled < 0.4:
        return 1
    if scaled < 0.6:
        return 2
    if scaled < 0.8:
        return 3
    return 4


def _streak_panel(target: str, model_type: str, pred_date: date) -> html.Div:
    """Render a GitHub-style recent prediction heatmap."""
    days_until_saturday = (5 - pred_date.weekday()) % 7
    grid_end = pred_date + timedelta(days=days_until_saturday)
    grid_start = grid_end - timedelta(days=(STREAK_WEEKS * 7) - 1)

    all_dates = [grid_start + timedelta(days=offset) for offset in range(STREAK_WEEKS * 7)]
    historic_dates = [d for d in all_dates if d <= pred_date]
    values = [
        _statewide_mean_prediction(target, model_type, d.timetuple().tm_yday, d.year)
        for d in historic_dates
    ]

    window_low = float(min(values)) if values else 0.0
    window_high = float(max(values)) if values else 1.0
    current_value = values[-1] if values else None
    current_assessment = _target_assessment(target, current_value)
    window_mean = float(np.mean(values)) if values else None
    peak_value = float(max(values)) if values else None
    unit = TARGET_UNITS.get(target, "")

    weeks = [all_dates[index:index + 7] for index in range(0, len(all_dates), 7)]
    month_labels = []
    previous_month = None
    for week in weeks:
        month = week[0].strftime("%b")
        month_labels.append(month if month != previous_month else "")
        previous_month = month

    weekday_labels = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
    week_columns = []
    for week in weeks:
        cells = []
        for cell_date in week:
            if cell_date > pred_date:
                level = 0
                background = "#f8fafc"
                border = "1px solid rgba(15, 23, 42, 0.05)"
                title = f"{cell_date.strftime('%b %d, %Y')}: upcoming"
            else:
                value = _statewide_mean_prediction(target, model_type, cell_date.timetuple().tm_yday, cell_date.year)
                level = _streak_level(value, window_low, window_high)
                background = STREAK_LEVELS[level]
                border = "1px solid rgba(15, 23, 42, 0.06)"
                title = f"{cell_date.strftime('%b %d, %Y')}: {value:.2f} {unit}"

            cells.append(
                html.Div(
                    className="streak-cell",
                    title=title,
                    style={"background": background, "border": border},
                )
            )

        week_columns.append(html.Div(className="streak-week", children=cells))

    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            _panel_heading("Recent trend", "statewide avg"),
            html.Div(
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "flex-start", "gap": "12px", "marginBottom": "12px"},
                children=[
                    html.Div(
                        children=[
                            html.Div(
                                style={"display": "flex", "alignItems": "baseline", "gap": "6px"},
                                children=[
                                    html.Span(f"{current_value:.1f}" if current_value is not None else "—", style={"fontSize": "30px", "fontWeight": "700", "color": TEXT_DARK, "lineHeight": "1", "letterSpacing": "-0.02em"}),
                                    html.Span(unit, style={"fontSize": "13px", "color": TEXT_LIGHT}),
                                ],
                            ),
                            html.Div(
                                current_assessment["label"],
                                className="assessment-pill",
                                style={"borderColor": current_assessment["color"], "color": current_assessment["color"], "marginTop": "9px"},
                            ),
                        ],
                    ),
                    html.Div(
                        style={"minWidth": "104px"},
                        children=[
                            _info_row(f"{STREAK_WEEKS}-wk avg", _fmt_metric(window_mean, f" {unit}", digits=1)),
                            _info_row("Peak", _fmt_metric(peak_value, f" {unit}", digits=1)),
                            _info_row("Low", _fmt_metric(window_low if values else None, f" {unit}", digits=1)),
                        ],
                    ),
                ],
            ),
            html.Div(
                f"{STREAK_WEEKS} weeks of statewide average predictions ending "
                f"{pred_date.strftime('%b %d, %Y')}. Shading is scaled within this window only.",
                className="panel-footnote",
                style={"marginTop": "0", "marginBottom": "12px"},
            ),
            html.Div(className="streak-months", children=[
                html.Div(className="streak-month-spacer"),
                html.Div(className="streak-month-labels", children=[
                    html.Div(label, className="streak-month-label")
                    for label in month_labels
                ]),
            ]),
            html.Div(
                className="streak-chart",
                children=[
                    html.Div(
                        className="streak-weekdays",
                        children=[html.Div(day, className="streak-weekday") for day in weekday_labels],
                    ),
                    html.Div(className="streak-heatmap", children=week_columns),
                ],
            ),
            html.Div(
                className="streak-legend",
                children=[
                    html.Span("Lower", className="streak-legend-label"),
                    *[
                        html.Div(
                            className="streak-legend-chip",
                            style={"background": color},
                        )
                        for color in STREAK_LEVELS
                    ],
                    html.Span("Higher", className="streak-legend-label"),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────
# CROSS-MODEL COMPARISON TABLE
#
# The same view as src/05_modeling/model_metrics.png, rendered live from
# model_metrics.csv: every target × every family, side by side, so a user can
# pick a model on evidence instead of by trying all four.
#
# Two presentation rules carried over from that figure:
#   • a log-fitted (target, family) is shown on the LOG scale — r2_log and
#     model_minus_persistence_log — flagged with †, because the raw-scale R² of
#     a log fit is set by a handful of extreme readings.
#   • RMSE / MAE / error rate are always in the target's own units, for all 52
#     models, so the thirteen targets stay comparable down a column.
# ─────────────────────────────────────────────────────────────
COMPARISON_COLUMNS = [
    ("Measurement", "cmp-col-target",
     "The water-quality variable being predicted."),
    ("Model",       "cmp-col-model",
     "Model family. Click any row to load that target/model pair into the controls above."),
    ("R²",          "cmp-col-r2",
     "Held-out R² — 1.0 perfect, 0.0 no better than the mean, negative worse. "
     "† rows are on the log scale."),
    ("RMSE",        "cmp-col-num",
     "Root mean squared error, in the target's own units."),
    ("MAE",         "cmp-col-num",
     "Mean absolute error, in the target's own units."),
    ("Error rate",  "cmp-col-num",
     "sMAPE (%). Tracks a target's zero fraction more than model quality — "
     "prefer MAE/RMSE on the zero-inflated targets."),
    ("vs. baseline","cmp-col-num",
     "R² minus the persistence baseline ('this station's next value equals its "
     "previous value'). Negative means the model does not beat repeating the last reading."),
    ("Test sites",  "cmp-col-num",
     "Monitoring stations held out whole from training and scored on."),
]


def _display_r2(row: pd.Series) -> Tuple[Optional[float], bool]:
    """
    Return (R² to display, was_log_fitted).

    For a log-fitted model the log-scale score is the honest one, so that is
    what the table shows — flagged, because it is not on the same scale as the
    raw-fit rows beside it.
    """
    r2_log = row.get("r2_log")
    if r2_log is not None and not pd.isna(r2_log):
        return float(r2_log), True
    r2 = row.get("r2")
    return (None if r2 is None or pd.isna(r2) else float(r2)), False


def _display_margin(row: pd.Series, is_log: bool) -> Optional[float]:
    key = "model_minus_persistence_log" if is_log else "model_minus_persistence"
    value = row.get(key)
    return None if value is None or pd.isna(value) else float(value)


def _r2_bar(value: Optional[float]) -> html.Div:
    """
    Numeric R² plus a proportional bar — the fast visual scan down a target's
    four rows. Length and shade both encode the score, so the ordering reads at
    a glance even where the four families are close together.
    """
    if value is None:
        return html.Div("—", className="cmp-r2-empty")

    fraction = max(0.0, min(1.0, value))
    # Blend #cfe0fb → #1d4ed8 so darker always means better.
    low, high = (207, 224, 251), (29, 78, 216)
    fill = tuple(int(low[i] + (high[i] - low[i]) * fraction) for i in range(3))
    return html.Div(
        className="cmp-r2",
        children=[
            html.Span(
                f"{value:.3f}",
                className="cmp-r2-value",
                style={"color": DANGER if value < 0 else TEXT_DARK},
            ),
            html.Div(
                className="cmp-r2-track",
                children=html.Div(
                    className="cmp-r2-fill",
                    style={
                        "width": f"{max(fraction * 100.0, 1.5):.1f}%",
                        "background": f"rgb{fill}",
                    },
                ),
            ),
        ],
    )


def _best_model_for(target: str) -> Optional[str]:
    """Family with the highest displayed R² for a target, or None."""
    if MODEL_METRICS.empty:
        return None
    rows = MODEL_METRICS[MODEL_METRICS["target"] == target]
    scored = {}
    for _, row in rows.iterrows():
        value, _ = _display_r2(row)
        if value is not None and row["model"] in MODEL_PREFIXES:
            scored[row["model"]] = value
    return max(scored, key=scored.get) if scored else None


def _comparison_table(selected_target: Optional[str] = None,
                      selected_model: Optional[str] = None,
                      scope: str = "all") -> html.Div:
    """
    Build the full target × family metrics table.

    `scope` is "all" or "selected" (only the currently chosen measurement).
    The currently selected pair is highlighted, and every row is clickable —
    clicking loads that pair into the dropdowns above.
    """
    if MODEL_METRICS.empty:
        return html.Div(
            "src/05_modeling/model_metrics.csv was not found, so the cross-model "
            "comparison is unavailable. Predictions are unaffected — the app reads "
            "everything it needs for inference out of the .pkl files.",
            className="cmp-unavailable",
        )

    present = set(MODEL_METRICS["target"])
    targets = [t for t in TARGET_COLS if t in present]
    if scope == "selected" and selected_target in targets:
        targets = [selected_target]

    body_rows = []
    for group_index, target in enumerate(targets):
        subset = MODEL_METRICS[MODEL_METRICS["target"] == target]
        by_model = {
            row["model"]: row
            for _, row in subset.iterrows()
            if row["model"] in MODEL_PREFIXES
        }
        best = _best_model_for(target)
        unit = TARGET_UNITS.get(target, "")

        family_rows = [m for m in MODEL_PREFIXES if m in by_model]
        for row_index, model_type in enumerate(family_rows):
            row = by_model[model_type]
            r2_value, is_log = _display_r2(row)
            margin = _display_margin(row, is_log)
            is_selected = (target == selected_target and model_type == selected_model)
            is_loaded = bool(MODELS.get(target, {}).get(model_type))

            classes = ["cmp-row", "cmp-group-a" if group_index % 2 == 0 else "cmp-group-b"]
            if row_index == 0:
                classes.append("cmp-group-start")
            if is_selected:
                classes.append("cmp-selected")
            if not is_loaded:
                classes.append("cmp-unloaded")

            body_rows.append(html.Tr(
                id={"type": "cmp-row", "target": target, "model": model_type},
                n_clicks=0,
                className=" ".join(classes),
                title=(f"Load {target} / {model_type}"
                       if is_loaded else
                       f"{target} / {model_type} — model file not loaded"),
                children=[
                    html.Td(
                        className="cmp-col-target",
                        children=[
                            html.Div(target, className="cmp-target-name"),
                            html.Div(unit, className="cmp-target-unit"),
                        ] if row_index == 0 else "",
                    ),
                    html.Td(
                        className="cmp-col-model",
                        children=[
                            html.Span("●", className="cmp-best-dot",
                                      title="Best R² for this measurement")
                            if model_type == best else
                            html.Span("", className="cmp-best-dot-spacer"),
                            html.Span(model_type),
                            html.Span(" †", className="cmp-flag",
                                      title="Fitted on log10(y + c); R² and margin are on the log scale.")
                            if is_log else None,
                            html.Span(" (not loaded)", className="cmp-missing")
                            if not is_loaded else None,
                        ],
                    ),
                    html.Td(_r2_bar(r2_value), className="cmp-col-r2"),
                    html.Td(_fmt_num(row.get("rmse")), className="cmp-col-num"),
                    html.Td(_fmt_num(row.get("mae")), className="cmp-col-num"),
                    html.Td(
                        "—" if pd.isna(row.get("error_rate")) else f"{float(row['error_rate']):.1f}",
                        className="cmp-col-num",
                    ),
                    html.Td(
                        _fmt_signed(margin) + ("†" if is_log and margin is not None else ""),
                        className="cmp-col-num",
                        style={"color": TEXT_LIGHT if margin is None
                               else (SUCCESS if margin >= 0 else DANGER)},
                    ),
                    html.Td(
                        "—" if pd.isna(row.get("test_stations")) else f"{int(row['test_stations']):,}",
                        className="cmp-col-num",
                    ),
                ],
            ))

    return html.Div(
        className="cmp-scroll",
        children=html.Table(
            className="cmp-table",
            children=[
                html.Thead(html.Tr([
                    html.Th(label, className=css_class, title=hint)
                    for label, css_class, hint in COMPARISON_COLUMNS
                ])),
                html.Tbody(body_rows),
            ],
        ),
    )


def _comparison_summary(target: Optional[str], model_type: Optional[str]) -> html.Div:
    """One-line 'here is the best model for what you picked' readout."""
    if not target or MODEL_METRICS.empty:
        return html.Div(
            "Pick a measurement above and this line will name its best-scoring model.",
            className="cmp-summary cmp-summary-idle",
        )

    best = _best_model_for(target)
    if best is None:
        return html.Div(f"No scores recorded for {target}.", className="cmp-summary cmp-summary-idle")

    best_row = _get_metric_row(target, best)
    best_r2, best_log = _display_r2(best_row)
    suffix = " (log scale)" if best_log else ""

    if model_type == best:
        verdict = html.Span(
            [html.B(model_type), " is the best-scoring model for ", html.B(target),
             f" — R² {best_r2:.3f}{suffix}."],
        )
        tone = "cmp-summary-good"
    else:
        current_row = _get_metric_row(target, model_type)
        current_r2, current_log = _display_r2(current_row) if current_row is not None else (None, False)
        if current_r2 is None:
            verdict = html.Span([html.B(best), " scores highest for ", html.B(target),
                                 f" — R² {best_r2:.3f}{suffix}."])
        else:
            verdict = html.Span([
                html.B(best), " scores highest for ", html.B(target),
                f" — R² {best_r2:.3f}{suffix}, versus {current_r2:.3f}"
                f"{' (log scale)' if current_log else ''} for the selected {model_type}.",
            ])
        tone = "cmp-summary-note"

    return html.Div(className=f"cmp-summary {tone}", children=[verdict])


COMPARISON_FOOTNOTES = [
    "† Fitted on log10(y + c). The R² and margin shown for those rows are on the log scale "
    "(r2_log / model_minus_persistence_log) — the like-for-like comparison for those fits, "
    "and not on the same scale as the untransformed rows beside them.",
    "RMSE, MAE and error rate are always in the target's own units, so the thirteen "
    "measurements stay comparable down a column.",
    "Every score is on stations held out whole from training (GroupShuffleSplit on "
    "MonitoringLocationIdentifier, 20% of stations), so it answers \"how well does this "
    "predict at a site the model has never seen?\"",
    "vs. baseline compares against persistence — \"this station's next value equals its "
    "previous value\", scored on the same paired rows. A negative margin means the model "
    "does not beat simply repeating the last reading.",
    "Error rate is sMAPE; it floors near 168% for Nitrite's 84% zeros. It reflects a "
    "target's zero fraction more than model quality.",
]


# ─────────────────────────────────────────────────────────────
# APP LAYOUT
# ─────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    title="Water Quality Predictor",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server

TODAY = date.today()
DATE_SHORTCUTS = [
    ("Today",       TODAY),
    ("In 1 week",   TODAY + timedelta(weeks=1)),
    ("In 2 weeks",  TODAY + timedelta(weeks=2)),
    ("In 1 month",  TODAY + timedelta(days=30)),
    ("In 6 months", TODAY + timedelta(days=182)),
    ("In 1 year",   TODAY + timedelta(days=365)),
]

_n_loaded = sum(len(v) for v in MODELS.values())

app.layout = html.Div(
    className="app-shell",
    style={"fontFamily": FONT, "backgroundColor": BG_PAGE, "minHeight": "100vh"},
    children=[

        dcc.Store(id="interp-store", data=False),
        dcc.Store(id="landmark-store", data=True),

        # ── Header ─────────────────────────────────────
        html.Div(
            className="app-header",
            children=[
                html.Div(
                    className="app-header-inner",
                    children=[
                        html.Div(
                            children=[
                                html.H1("Iowa Water Quality Predictor", className="app-title"),
                                html.Div(
                                    "Predicted water-quality conditions at every EPA monitoring "
                                    "station in Iowa, for any date you choose.",
                                    className="app-subtitle",
                                ),
                            ],
                        ),
                        html.Div(
                            className="header-chips",
                            children=[
                                html.Div(className="header-chip", children=[
                                    html.Span(f"{len(STATIONS):,}", className="header-chip-value"),
                                    html.Span("stations", className="header-chip-label"),
                                ]),
                                html.Div(className="header-chip", children=[
                                    html.Span(f"{len(TARGET_COLS)}", className="header-chip-value"),
                                    html.Span("measurements", className="header-chip-label"),
                                ]),
                                html.Div(className="header-chip", children=[
                                    html.Span(f"{_n_loaded}", className="header-chip-value"),
                                    html.Span("models loaded", className="header-chip-label"),
                                ]),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # ── Main content ────────────────────────────────
        html.Div(
            className="main-content",
            style={"maxWidth": "1320px", "margin": "0 auto", "padding": "0 24px 40px"},
            children=[

                    html.Div(
                        className="dashboard-grid",
                        style={"display": "grid", "gap": "22px", "alignItems": "start"},
                        children=[

                        html.Div(
                            className="left-rail",
                            style={"display": "grid", "gap": "18px"},
                            children=[

                            html.Div(style={**CARD_STYLE, "display": "flex", "flexDirection": "column", "gap": "18px"}, children=[

                                # Target variable
                                html.Div(children=[
                                    html.Span("1 · What to measure", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="target-dropdown",
                                        options=_available_targets(),
                                        value=None,
                                        placeholder="Pick a measurement…",
                                        clearable=False,
                                        searchable=False,
                                        style=DROPDOWN_STYLE,
                                    ),
                                    html.Div(id="target-helper", className="control-helper"),
                                ]),

                                html.Div(className="control-divider"),

                                # Model type
                                html.Div(children=[
                                    html.Span("2 · Which model", style=LABEL_STYLE),
                                    dcc.Dropdown(
                                        id="model-dropdown",
                                        options=_available_model_types(None),
                                        value="Gradient Boosting",
                                        clearable=False,
                                        searchable=False,
                                        style=DROPDOWN_STYLE,
                                    ),
                                    html.Div(id="model-helper", className="control-helper"),
                                ]),

                                html.Div(className="control-divider"),

                                # Date input + quick-fill buttons
                                html.Div(children=[
                                    html.Span("3 · When", style=LABEL_STYLE),
                                    dcc.DatePickerSingle(
                                        id="date-picker",
                                        min_date_allowed=date(2000, 1, 1),
                                        max_date_allowed=date(2030, 12, 31),
                                        placeholder="Pick a date…",
                                        display_format="MMM D, YYYY",
                                        style={"width": "100%", "marginBottom": "9px"},
                                    ),
                                    html.Div(
                                        className="shortcut-grid",
                                        children=[
                                            html.Button(
                                                label,
                                                id=f"btn-{label.lower().replace(' ', '-')}",
                                                n_clicks=0,
                                                className="shortcut-btn",
                                            )
                                            for label, _ in DATE_SHORTCUTS
                                        ],
                                    ),
                                ]),

                                html.Button("Run prediction", id="predict-btn", n_clicks=0,
                                            className="primary-btn"),
                            ]),

                            # ── Map display options ──────────────────
                            html.Div(style={**CARD_STYLE, "padding": "18px 20px"}, children=[
                                html.Span("Map display", style=LABEL_STYLE),
                                html.Button(
                                    id="interp-toggle",
                                    n_clicks=0,
                                    className="toggle-btn",
                                    children=[
                                        html.Span(className="toggle-track", children=html.Span(className="toggle-knob")),
                                        html.Span("Interpolated surface", className="toggle-label"),
                                        html.Span("Off", id="interp-state", className="toggle-state"),
                                    ],
                                ),
                                html.Div(
                                    "Off shows the station predictions alone. On fills the gaps "
                                    "between stations with a cubic-spline estimate — smooth, but "
                                    "invented data wherever the network is sparse. The colour scale "
                                    "is the same either way.",
                                    className="control-helper",
                                ),

                                html.Div(className="control-divider", style={"margin": "16px 0"}),

                                html.Button(
                                    id="landmark-toggle",
                                    n_clicks=0,
                                    className="toggle-btn toggle-btn-on",
                                    children=[
                                        html.Span(className="toggle-track", children=html.Span(className="toggle-knob")),
                                        html.Span("Landmarks", className="toggle-label"),
                                        html.Span("On", id="landmark-state", className="toggle-state"),
                                    ],
                                ),
                                html.Div(
                                    "Names 12 major cities (dark squares), 6 rivers and 5 lakes "
                                    "for orientation. Turn it off if the labels crowd the stations "
                                    "you are reading.",
                                    className="control-helper",
                                ),
                                html.Div(
                                    className="map-key",
                                    children=[
                                        html.Span(className="map-key-item", children=[
                                            html.Span(className="map-key-swatch map-key-city"),
                                            "City",
                                        ]),
                                        html.Span(className="map-key-item", children=[
                                            html.Span(className="map-key-swatch map-key-lake"),
                                            "Lake",
                                        ]),
                                        html.Span(className="map-key-item", children=[
                                            html.Span(className="map-key-swatch map-key-station"),
                                            "Station",
                                        ]),
                                    ],
                                ),
                            ]),
                            ],
                        ),

                        html.Div(
                            className="center-stage",
                            style={"display": "grid", "gap": "8px"},
                            children=[
                            html.Div(
                                className="map-header",
                                children=[
                                    html.Div(id="map-title", className="map-title", children="Monitoring stations"),
                                    html.Div(id="map-subtitle", className="map-subtitle",
                                             children="Pick a measurement and date, then run a prediction"),
                                ],
                            ),
                            html.Div(id="status-msg"),
                            html.Div(
                                style={**CARD_STYLE, "padding": "6px"},
                                children=[
                                    dcc.Graph(
                                        id="usa-map",
                                        figure=empty_map_figure(),
                                        config={
                                            "displayModeBar": "hover",
                                            "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                                            "displaylogo": False,
                                        },
                                    ),
                                ],
                            ),
                            html.Div(
                                id="map-footnote",
                                className="map-footnote",
                                children="Station features are carried forward from each site's most "
                                         "recent observation; only the date-derived predictors change "
                                         "with the date you pick.",
                            ),
                        ],
                        ),

                        html.Div(
                            className="right-rail",
                            style={"display": "grid", "gap": "0", "alignContent": "start"},
                            children=[
                                html.Div(
                                    style={**CARD_STYLE, "padding": "0", "overflow": "hidden"},
                                    children=[
                                        html.Div(id="hover-panel",       children=_hover_panel_default()),
                                        html.Div(style=DIVIDER_STYLE),
                                        html.Div(id="streak-panel",      children=_streak_panel_default()),
                                        html.Div(style=DIVIDER_STYLE),
                                        html.Div(id="stats-panel",       children=_summary_panel_default()),
                                        html.Div(style=DIVIDER_STYLE),
                                        html.Div(id="performance-panel", children=_performance_panel(None, None)),
                                    ],
                                ),
                            ],
                        ),

                    ],
                ),

                # ── Cross-model comparison ─────────────────────
                html.Div(
                    style={**CARD_STYLE, "marginTop": "24px", "padding": "24px"},
                    children=[
                        html.Div(
                            className="cmp-header",
                            children=[
                                html.Div(children=[
                                    html.H2("Compare every model", className="cmp-title"),
                                    html.Div(
                                        "Held-out test scores for all "
                                        f"{len(TARGET_COLS)} measurements × {len(MODEL_PREFIXES)} "
                                        "model families, from src/05_modeling/model_metrics.csv. "
                                        "Click any row to load that pair into the controls above.",
                                        className="cmp-subtitle",
                                    ),
                                ]),
                                dcc.RadioItems(
                                    id="cmp-scope",
                                    options=[
                                        {"label": "All measurements", "value": "all"},
                                        {"label": "Selected only", "value": "selected"},
                                    ],
                                    value="all",
                                    className="cmp-scope",
                                    inputClassName="cmp-scope-input",
                                    labelClassName="cmp-scope-label",
                                ),
                            ],
                        ),
                        html.Div(id="cmp-summary", children=_comparison_summary(None, None)),
                        html.Div(id="cmp-table", children=_comparison_table(None, None, "all")),
                        html.Div(
                            className="cmp-legend",
                            children=[
                                html.Span([html.Span("●", className="cmp-best-dot"),
                                           " best R² for that measurement"], className="cmp-legend-item"),
                                html.Span("† fitted on log10 — scores shown on the log scale",
                                          className="cmp-legend-item"),
                                html.Span("darker bar = higher R²", className="cmp-legend-item"),
                            ],
                        ),
                        html.Div(
                            className="cmp-footnotes",
                            children=[html.Div(note, className="cmp-footnote")
                                      for note in COMPARISON_FOOTNOTES],
                        ),
                    ],
                ),

            ],
        ),
    ],
)


# ─────────────────────────────────────────────────────────────
# CALLBACK: Quick-fill date buttons
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("date-picker", "date"),
    [Input(f"btn-{label.lower().replace(' ', '-')}", "n_clicks")
     for label, _ in DATE_SHORTCUTS],
    prevent_initial_call=True,
)
def fill_date(*_):
    """Set the date picker to the matching shortcut date."""
    ctx = callback_context
    if not ctx.triggered:
        return no_update
    btn_id = ctx.triggered[0]["prop_id"].split(".")[0]
    for label, d in DATE_SHORTCUTS:
        if btn_id == f"btn-{label.lower().replace(' ', '-')}":
            return d.isoformat()
    return no_update


# ─────────────────────────────────────────────────────────────
# CALLBACK: Update model-dropdown options when target changes
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("model-dropdown", "options"),
    Output("model-dropdown", "value"),
    Input("target-dropdown", "value"),
    State("model-dropdown",  "value"),
    prevent_initial_call=True,
)
def update_model_options(target, current_model):
    """Grey out model types that don't have a loaded pkl for the chosen target."""
    options   = _available_model_types(target)
    available = [o["value"] for o in options if not o.get("disabled")]
    new_value = current_model if current_model in available else (
        available[0] if available else no_update
    )
    return options, new_value


@app.callback(
    Output("model-helper", "children"),
    Input("model-dropdown", "value"),
)
def update_model_helper(model_type):
    if not model_type:
        return ""
    return MODEL_DESCRIPTIONS.get(model_type, "")


@app.callback(
    Output("target-helper", "children"),
    Input("target-dropdown", "value"),
)
def update_target_helper(target):
    """Surface the measurement's units and what it is useful for."""
    if not target:
        return "Thirteen water-quality variables are available, including the composite WQI."
    unit = TARGET_UNITS.get(target, "")
    note = TARGET_SHORT_NOTES.get(target, "")
    return html.Span([html.B(f"{unit} · "), note] if unit else note)


# ─────────────────────────────────────────────────────────────
# CALLBACK: Interpolated-surface toggle
# Off by default — the station predictions are the model's actual output;
# the interpolated grid is a cubic spline drawn between them, which is
# useful for reading regional pattern but is not data.
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("interp-store",  "data"),
    Output("interp-state",  "children"),
    Output("interp-toggle", "className"),
    Input("interp-toggle",  "n_clicks"),
    State("interp-store",   "data"),
    prevent_initial_call=True,
)
def toggle_interpolation(_n_clicks, enabled):
    enabled = not bool(enabled)
    return (
        enabled,
        "On" if enabled else "Off",
        "toggle-btn toggle-btn-on" if enabled else "toggle-btn",
    )


# ─────────────────────────────────────────────────────────────
# CALLBACK: Landmark toggle
# On by default: the scope="usa" basemap carries no place names, so without
# these labels a station dot cannot be located by eye.
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("landmark-store",  "data"),
    Output("landmark-state",  "children"),
    Output("landmark-toggle", "className"),
    Input("landmark-toggle",  "n_clicks"),
    State("landmark-store",   "data"),
    prevent_initial_call=True,
)
def toggle_landmarks(_n_clicks, enabled):
    enabled = not bool(enabled)
    return (
        enabled,
        "On" if enabled else "Off",
        "toggle-btn toggle-btn-on" if enabled else "toggle-btn",
    )


# ─────────────────────────────────────────────────────────────
# CALLBACK: Run prediction & update map
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("usa-map",     "figure"),
    Output("status-msg",  "children"),
    Output("map-title",   "children"),
    Output("map-subtitle", "children"),
    Output("streak-panel", "children"),
    Output("stats-panel", "children"),
    Output("performance-panel", "children"),
    Input("predict-btn",    "n_clicks"),
    Input("interp-store",   "data"),
    Input("landmark-store", "data"),
    State("target-dropdown", "value"),
    State("model-dropdown",  "value"),
    State("date-picker",  "date"),
    prevent_initial_call=True,
)
def run_prediction(n_clicks, show_interpolation, show_landmarks, target, model_type, selected_date):
    """
    Validate → load pre-trained model → build features → predict → interpolate → render.

    Compared to the training-based version:
      - No Pipeline.fit() anywhere — we call .predict() on the loaded pkl directly
      - Feature matrix is built from station CSV rows + the user-chosen date only
      - Model unavailability is caught here (not just at startup) in case
        a pkl was deleted while the app was running

    Also fires on the two display toggles so the map redraws without a second
    click; the predictions themselves come out of _cached_station_predictions,
    so that redraw costs no inference.
    """
    ready = bool(target and selected_date and MODELS.get(target, {}).get(model_type))

    # A display toggle before anything has been predicted should restyle the
    # base map, not scold the user for not having filled the form in yet.
    if _triggered_id() in ("interp-store", "landmark-store") and not ready:
        return (empty_map_figure(bool(show_landmarks)),) + (no_update,) * 6

    # ── Input validation ─────────────────────────────
    errors = []
    if not target:
        errors.append("a target variable")
    if not selected_date:
        errors.append("a prediction date")
    if target and not MODELS.get(target, {}).get(model_type):
        errors.append(f"a loaded model for '{target} / {model_type}'")

    if errors:
        msg = html.Div(
            f"Please pick {' and '.join(errors)} first.",
            className="status-msg status-msg-error",
        )
        return (
            empty_map_figure(bool(show_landmarks)),
            msg,
            no_update,
            no_update,
            _streak_panel_default(),
            no_update,
            _performance_panel(target, model_type),
        )

    # ── Predict ──────────────────────────────────────
    pred_date    = date.fromisoformat(selected_date)
    station_df   = predict_at_stations(target, model_type, pred_date)
    show_surface = bool(show_interpolation)

    colorscale = TARGET_COLORSCALES.get(target, "Viridis")
    unit       = TARGET_UNITS.get(target, "")

    # Base the color range on actual station predictions, not the interpolated
    # grid.  Cubic spline interpolation can overshoot wildly in sparse areas
    # (Runge phenomenon), producing physically impossible values like -300,000°C.
    # Anchoring vmin/vmax to the station data and clipping the grid keeps the
    # colorscale meaningful and the heatmap within realistic bounds.
    preds      = station_df["predicted"]
    vmin       = float(np.percentile(preds, 2))
    vmax       = float(np.percentile(preds, 98))

    fig = go.Figure()

    # Interpolated background grid (smooth coverage between stations). Off by
    # default: it is a spline drawn between the model's actual outputs, not a
    # prediction in its own right. The colour scale lives on the station trace
    # below so the legend reads identically whether or not this layer is drawn.
    if show_surface:
        lon_grid, lat_grid, val_grid = interpolate_to_grid(station_df)
        val_grid  = np.clip(val_grid, vmin, vmax)
        flat_lons = lon_grid.ravel()
        flat_lats = lat_grid.ravel()
        flat_vals = val_grid.ravel()
        idx       = np.where(~np.isnan(flat_vals))[0]
        if len(idx) > 4000:                                # downsample for performance
            idx = np.random.default_rng(0).choice(idx, 4000, replace=False)

        fig.add_trace(go.Scattergeo(
            lat=flat_lats[idx],
            lon=flat_lons[idx],
            mode="markers",
            customdata=np.column_stack([
                np.full(len(idx), "Interpolated surface"),
                np.full(len(idx), target),
                [
                    _target_assessment(target, value)["label"]
                    for value in flat_vals[idx]
                ],
                [
                    _target_assessment(target, value)["detail"]
                    for value in flat_vals[idx]
                ],
            ]),
            marker=dict(
                size=9,
                color=flat_vals[idx],
                colorscale=colorscale,
                cmin=vmin, cmax=vmax,
                opacity=0.72,
                showscale=False,
                line=dict(width=0),
            ),
            name="Interpolated grid",
            hovertemplate=(
                f"<b>Interpolated {target}</b><br>"
                f"Estimated value: %{{marker.color:.2f}} {unit}<br>"
                "Interpretation: %{customdata[2]}<br>"
                "%{customdata[3]}<br>"
                "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<extra></extra>"
            ),
        ))

    # Actual station markers on top
    station_customdata = np.column_stack([
        station_df["MonitoringLocationName"].fillna("Unknown station"),
        station_df["MonitoringLocationIdentifier"].fillna("Unknown ID"),
        station_df["ProviderName"].fillna("Unknown provider"),
        station_df["climate_station_name"].fillna("Unknown climate station"),
        station_df["distance_to_climate_station_km"].fillna(0.0),
        station_df["predicted"].round(4),
        [
            _target_assessment(target, value)["label"]
            for value in station_df["predicted"]
        ],
        [
            _target_assessment(target, value)["detail"]
            for value in station_df["predicted"]
        ],
    ])
    fig.add_trace(go.Scattergeo(
        lat=station_df[LAT_COL],
        lon=station_df[LON_COL],
        mode="markers",
        customdata=station_customdata,
        marker=dict(
            # Slightly larger when they are the only layer on the map.
            size=10 if show_surface else 12,
            symbol="circle",
            color=station_df["predicted"],
            colorscale=colorscale,
            cmin=vmin, cmax=vmax,
            # The scale rides on this trace, not on the interpolated one, so it
            # is present and identical whether or not the surface is drawn.
            showscale=True,
            colorbar=dict(
                title=dict(text=f"<b>{target}</b><br>{unit}", font=dict(size=12)),
                thickness=14, len=0.72, x=1.01,
                tickfont=dict(size=10),
                outlinewidth=0,
            ),
            line=dict(color="white", width=1.2),
        ),
        name="Monitoring stations",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Station ID: %{customdata[1]}<br>"
            "Provider: %{customdata[2]}<br>"
            "Climate station: %{customdata[3]}<br>"
            "Distance to climate station: %{customdata[4]:.1f} km<br>"
            f"Predicted {target}: %{{customdata[5]:.2f}} {unit}<br>"
            "Interpretation: %{customdata[6]}<br>"
            "%{customdata[7]}<br>"
            "Coordinates: %{lat:.4f}°N, %{lon:.4f}°W<extra></extra>"
        ),
    ))

    _add_map_labels(fig, bool(show_landmarks))
    _apply_geo_layout(fig)

    title = html.Span([
        html.Span(target, className="map-title-target"),
        html.Span(f" · {unit}", className="map-title-unit"),
    ])
    subtitle = html.Span([
        html.Span(pred_date.strftime("%b %d, %Y"), className="map-subtitle-strong"),
        f"  ·  {model_type}  ·  {len(station_df):,} stations",
        html.Span("  ·  + interpolated surface" if show_surface else "  ·  stations only",
                  className="map-subtitle-mode"),
    ])
    status = ""

    # ── Summary stats ─────────────────────────────────
    preds = station_df["predicted"]
    stats = html.Div(
        style=SIDECARD_STYLE,
        children=[
            _panel_heading("Statewide spread", f"{len(station_df):,} sites"),
            html.Div(
                className="stat-grid",
                children=[
                    _stat_tile("Low",    _fmt_num(preds.min()),    unit),
                    _stat_tile("Median", _fmt_num(preds.median()), unit),
                    _stat_tile("High",   _fmt_num(preds.max()),    unit),
                ],
            ),
            html.Div(
                style={"marginTop": "10px"},
                children=[
                    _info_row("Average", f"{_fmt_num(preds.mean())} {unit}"),
                    _info_row("Colour range shown", f"{_fmt_num(vmin)} – {_fmt_num(vmax)} {unit}",
                              hint="The map's colour scale is anchored to the 2nd–98th percentile "
                                   "of station predictions so outliers do not flatten the ramp."),
                ],
            ),
        ],
    )

    return fig, status, title, subtitle, _streak_panel(target, model_type, pred_date), stats, _performance_panel(target, model_type)


@app.callback(
    Output("hover-panel", "children"),
    Input("usa-map", "hoverData"),
    State("target-dropdown", "value"),
)
def update_hover_panel(hover_data, target):
    if not hover_data or "points" not in hover_data or not hover_data["points"]:
        return _hover_panel_default()

    point = hover_data["points"][0]
    lat = point.get("lat")
    lon = point.get("lon")
    unit = TARGET_UNITS.get(target, "")
    custom = point.get("customdata")

    # Identify the trace by the shape of its customdata rather than by curve
    # index: the interpolated layer is optional, so the station trace is not
    # always curve 1.
    is_station = bool(custom) and len(custom) >= 8

    if is_station:
        station_name, station_id, provider, climate_station, distance_km, predicted, assessment_label, assessment_detail = custom
        pred_val = float(predicted)
        assessment = _target_assessment(target, pred_val)
        city = _nearest_city(float(lat), float(lon)) if lat is not None and lon is not None else ""
        return html.Div(
            style=SIDECARD_STYLE,
            children=[
                _panel_heading("Point detail", "monitoring station"),
                html.Div(f"Near {city}", className="hover-place"),
                html.Div(station_name, className="hover-station"),
                html.Div(
                    className="hover-value",
                    children=[
                        html.Span(f"{pred_val:.1f}", style={"color": ACCENT}),
                        html.Span(unit, className="hover-value-unit"),
                    ],
                ),
                html.Div(
                    assessment_label,
                    className="assessment-pill",
                    style={"borderColor": assessment["color"], "color": assessment["color"]},
                ),
                html.Div(assessment_detail, className="hover-detail"),
                html.Div(
                    style={"marginTop": "12px"},
                    children=[
                        _info_row("Station ID", str(station_id)),
                        _info_row("Provider", str(provider)),
                        _info_row("Climate station", str(climate_station)),
                        _info_row("Distance", f"{float(distance_km):.1f} km",
                                  hint="Distance from this station to the weather station "
                                       "supplying its climate predictors."),
                        _info_row("Coordinates", f"{float(lat):.4f}°N, {abs(float(lon)):.4f}°W"),
                    ],
                ),
            ],
        )

    value = point.get("marker.color")
    city = _nearest_city(float(lat), float(lon)) if lat is not None and lon is not None else "Iowa"
    numeric_value = float(value) if value is not None and target else None
    assessment = _target_assessment(target, numeric_value)

    return html.Div(
        style=SIDECARD_STYLE,
        children=[
            _panel_heading("Point detail", "interpolated"),
            html.Div(f"Near {city}", className="hover-place"),
            html.Div("Not a monitoring station — a spline estimate between them.",
                     className="hover-station"),
            html.Div(
                className="hover-value",
                children=[
                    html.Span(f"{numeric_value:.1f}" if numeric_value is not None else "—",
                              style={"color": TEXT_MID}),
                    html.Span(unit, className="hover-value-unit"),
                ],
            ),
            html.Div(
                assessment["label"],
                className="assessment-pill",
                style={"borderColor": assessment["color"], "color": assessment["color"]},
            ),
            html.Div(assessment["detail"], className="hover-detail"),
            html.Div(
                style={"marginTop": "12px"},
                children=[
                    _info_row("Coordinates",
                              f"{float(lat):.4f}°N, {abs(float(lon)):.4f}°W"
                              if lat is not None and lon is not None else "—"),
                ],
            ),
        ],
    )


# ─────────────────────────────────────────────────────────────
# CALLBACK: Cross-model comparison table
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("cmp-table",   "children"),
    Output("cmp-summary", "children"),
    Input("target-dropdown", "value"),
    Input("model-dropdown",  "value"),
    Input("cmp-scope",       "value"),
)
def update_comparison(target, model_type, scope):
    """Re-render the table so the current pair is highlighted and the scope filter applies."""
    return (
        _comparison_table(target, model_type, scope or "all"),
        _comparison_summary(target, model_type),
    )


# ─────────────────────────────────────────────────────────────
# CALLBACK: Click a comparison row to load that target/model pair
#
# Writes both dropdowns at once. Dash applies every output of a callback
# before firing downstream ones, so update_model_options — which listens on
# the target dropdown — sees the new model in its State and leaves it alone.
# ─────────────────────────────────────────────────────────────
@app.callback(
    Output("target-dropdown", "value", allow_duplicate=True),
    Output("model-dropdown",  "value", allow_duplicate=True),
    Input({"type": "cmp-row", "target": ALL, "model": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def select_from_comparison(_clicks):
    triggered = callback_context.triggered
    # Re-rendering the table mounts fresh rows at n_clicks=0, which fires this
    # callback; only a real click carries a positive count.
    if not triggered or not triggered[0].get("value"):
        return no_update, no_update

    row_id = callback_context.triggered_id
    target = row_id["target"]
    model_type = row_id["model"]
    if not MODELS.get(target, {}).get(model_type):
        return no_update, no_update
    return target, model_type


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
