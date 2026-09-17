"""Out-of-sample permutation importance for the fitted RF and GB models.

This is a post-hoc companion to the three EDA notebooks, not one of them. The
notebooks characterise the *input space* (correlation, VIF, mutual information);
this script asks the complementary question — of the signal the deployed models
actually found, which predictors carry it — and it asks it on the held-out
stations, so the answer is not station recall.

Two things it does differently from a stock `permutation_importance` call:

* **Correlated features are permuted as a block, under one shared permutation.**
  Five of the six thermal columns are the same fact (`prism_tmin_c` ↔
  `isu_min_feel_c` at rho = 0.955) and the four nutrient columns span about two
  dimensions. Permuted one at a time they all look unimportant, because the
  model reads the survivors instead. Permuting the block together, with the
  same row order applied to every column in it, destroys the block's relation
  to the target while preserving the relations *within* it.
* **Scoring happens on the scale each model was fitted on** — log10(y + c) for
  a log-fitted (target, family), raw otherwise — so the drop is measured
  against the objective the model was trained to optimise.

The split is rebuilt to match the training notebooks exactly
(`GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)` on
`MonitoringLocationIdentifier`), which is checked by the base R2 it prints:
those reproduce `src/05_modeling/model_metrics.csv` to within the subsampling
tolerance. Nothing is re-fitted — the committed .pkl files are read as-is.

Usage (from the repo root, venv active):

    python3 src/04_eda/block_permutation_importance.py

Writes `src/04_eda/outputs/pi_permutation_importance.csv`. Findings are
consolidated in `eda-summary.md` section 1.12.
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "data" / "final" / "epa-full.csv"
OUT_PATH = REPO_ROOT / "src" / "04_eda" / "outputs" / "pi_permutation_importance.csv"

# Mirrors the training notebooks. Changing either of these without changing
# them there invalidates the comparison.
RANDOM_STATE = 42
TEST_SIZE = 0.2
MIN_SAMPLES = 100
GROUP_COL = "MonitoringLocationIdentifier"

# Cap the scored rows so the run stays a few minutes; 5,000 rows is far more
# than enough to resolve a drop against the ~0.001 repeat-to-repeat spread.
MAX_TEST_ROWS = 5_000
N_REPEATS = 5

BASE_FEATURE_COLS = [
    "LatitudeMeasure", "LongitudeMeasure", "distance_to_climate_station_km",
    "distance_to_streamflow_gauge_km", "prism_tmax_c", "prism_tmin_c", "prism_ppt_mm",
    "prism_tdmean_c", "isu_avg_wind_speed_kts", "isu_avg_rh", "isu_snow_in", "isu_snowd_in",
    "isu_max_feel_c", "isu_min_feel_c", "streamflow_discharge_cfs", "ksat_mean", "awc_mean",
    "pct_corn", "pct_soybean", "pct_developed", "pct_forest",
    "npfert__n__total_kg", "npfert__p__total_kg", "npmanure__total__n_kg", "npmanure__total__p_kg",
]
TEMPORAL_FEATURE_COLS = ["doy", "doy_sin", "doy_cos", "obs_year"]
FEATURE_COLS = BASE_FEATURE_COLS + TEMPORAL_FEATURE_COLS

# Blocks follow the |rho| >= 0.65 clustering in `mv_feature_clusters.csv`, with
# the six-member thermal cluster split so the calendar half (`doy_*`) can be
# credited separately from the temperature half.
BLOCKS = {
    "location (lat/lon)": ["LatitudeMeasure", "LongitudeMeasure"],
    "network geometry": ["distance_to_climate_station_km", "distance_to_streamflow_gauge_km"],
    "air temperature": ["prism_tmax_c", "prism_tmin_c", "prism_tdmean_c",
                        "isu_max_feel_c", "isu_min_feel_c"],
    "precip + humidity": ["prism_ppt_mm", "isu_avg_rh"],
    "wind": ["isu_avg_wind_speed_kts"],
    "snow": ["isu_snow_in", "isu_snowd_in"],
    "streamflow": ["streamflow_discharge_cfs"],
    "soil": ["ksat_mean", "awc_mean"],
    "land cover": ["pct_corn", "pct_soybean", "pct_developed", "pct_forest"],
    "nutrient budget": ["npfert__n__total_kg", "npfert__p__total_kg",
                        "npmanure__total__n_kg", "npmanure__total__p_kg"],
    "season (day-of-year)": ["doy", "doy_sin", "doy_cos"],
    "year": ["obs_year"],
}

# Column and plausible range per target, copied from the training notebooks.
TARGETS = {
    "Water Temperature": ("Temperature, water_value", (-5.0, 45.0)),
    "Dissolved Oxygen": ("Dissolved oxygen (DO)_value", (0.0, 30.0)),
    "pH": ("pH_value", (0.0, 14.0)),
    "Nitrate": ("Nitrate_value", (0.0, 100.0)),
    "Nitrite": ("Nitrite_value", (0.0, 20.0)),
    "Nitrate + Nitrite": ("Nitrate + Nitrite_value", (0.0, 100.0)),
    "Total Phosphorus": ("Total Phosphorus, mixed forms_value", (0.0, 25.0)),
    "Specific Conductance": ("Specific conductance_value", (0.0, 10000.0)),
    "Total Dissolved Solids": ("Total dissolved solids_value", (0.0, 10000.0)),
    "Total Suspended Solids": ("Total suspended solids_value", (0.0, 10000.0)),
    "Turbidity": ("Turbidity_value", (0.0, 5000.0)),
    "E. coli": ("Escherichia coli_value", (0.0, 1_000_000.0)),
    "WQI": ("WQI", (0.0, 100.0)),
}

FAMILIES = [
    ("Random Forest", "random_forest", "rf"),
    ("Gradient Boosting", "gradient_boosting", "gb"),
]


def target_stem(label: str) -> str:
    """'Nitrate + Nitrite' -> 'nitrate_nitrite', 'E. coli' -> 'e_coli'."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


def load_dataset(path: Path) -> pd.DataFrame:
    """Read the terminal table and derive the four temporal predictors."""
    df = pd.read_csv(path, low_memory=False)
    ts = pd.to_datetime(df["ActivityStartDateTime"], errors="coerce")
    doy = ts.dt.dayofyear
    radians = 2.0 * np.pi * doy / 365.25
    temporal = pd.DataFrame({
        "doy": doy,
        "obs_year": ts.dt.year,
        "doy_sin": np.sin(radians),
        "doy_cos": np.cos(radians),
    }, index=df.index)
    return pd.concat([df, temporal], axis=1)


def held_out(df: pd.DataFrame, column: str, valid_range: tuple[float, float],
             rng: np.random.Generator):
    """Rebuild one target's test stations, exactly as the training notebooks do."""
    low, high = valid_range
    d = df[df[column].notna()]
    d = d[(d[column] >= low) & (d[column] <= high)].dropna(subset=[GROUP_COL])
    if len(d) < MIN_SAMPLES:
        return None
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    _, test_idx = next(splitter.split(d, groups=d[GROUP_COL].to_numpy()))
    X = d.iloc[test_idx][FEATURE_COLS].to_numpy(float)
    y = d.iloc[test_idx][column].to_numpy(float)
    if len(X) > MAX_TEST_ROWS:
        keep = rng.choice(len(X), MAX_TEST_ROWS, replace=False)
        X, y = X[keep], y[keep]
    return X, y


def permutation_drop(pipeline, X: np.ndarray, y_fit: np.ndarray, base: float,
                     columns: list[str], seed: int) -> tuple[float, float]:
    """Mean and SD of the R2 lost when `columns` are shuffled together."""
    rng = np.random.default_rng(seed)
    positions = [FEATURE_COLS.index(c) for c in columns]
    deltas = []
    for _ in range(N_REPEATS):
        shuffled = X.copy()
        order = rng.permutation(len(shuffled))       # one order for the whole block,
        shuffled[:, positions] = shuffled[order][:, positions]  # so within-block ties survive
        deltas.append(base - r2_score(y_fit, pipeline.predict(shuffled)))
    return float(np.mean(deltas)), float(np.std(deltas))


def main() -> None:
    data = load_dataset(DATA_PATH)
    print(f"Rows: {len(data):,}")

    rng = np.random.default_rng(0)
    records = []

    for family, subdir, prefix in FAMILIES:
        for label, (column, valid_range) in TARGETS.items():
            path = REPO_ROOT / "src" / "05_modeling" / subdir / f"{prefix}_{target_stem(label)}.pkl"
            if not path.exists():
                print(f"  skip {family} / {label}: no .pkl")
                continue
            with open(path, "rb") as handle:
                artifact = pickle.load(handle)
            assert artifact["feature_cols"] == FEATURE_COLS, f"stale feature contract in {path.name}"

            split = held_out(data, column, valid_range, rng)
            if split is None:
                continue
            X, y = split

            pipeline = artifact["pipeline"]
            offset = artifact.get("log_offset")
            y_fit = np.log10(y + offset) if offset is not None else y
            base = float(r2_score(y_fit, pipeline.predict(X)))

            for i, feature in enumerate(FEATURE_COLS):
                mean, sd = permutation_drop(pipeline, X, y_fit, base, [feature], 1000 + i)
                records.append(dict(family=family, target=label, kind="single", name=feature,
                                    base_r2=base, delta_r2=mean, delta_sd=sd, n_test=len(X)))
            for j, (block, columns) in enumerate(BLOCKS.items()):
                mean, sd = permutation_drop(pipeline, X, y_fit, base, columns, 2000 + j)
                records.append(dict(family=family, target=label, kind="block", name=block,
                                    base_r2=base, delta_r2=mean, delta_sd=sd, n_test=len(X)))

            print(f"{family:18s} {label:24s} base R2 = {base:6.3f}  (n = {len(X):,})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(OUT_PATH, index=False)
    print(f"Wrote {OUT_PATH.relative_to(REPO_ROOT)} ({len(records):,} rows)")


if __name__ == "__main__":
    main()
