# Next Steps

A plan for whoever continues this project: what to do, in what order, and how to
tell when each step is finished. Read
[Current Status and Limitations](README.md#current-status-and-limitations) in the
README first — this file assumes you know why the work below is needed.

## The short version

The data pipeline and the model evaluation are solid. The dashboard is not yet
decision-ready, for five measured reasons: it feeds the models **stale weather**,
it presents **future dates as forecasts**, it predicts at **station types the
models never saw**, it **hides the observed data**, and it shows **confident
labels with no uncertainty**. A better model fixes none of these, so model
improvements come last.

| Step | Goal | Depends on | Changes `FEATURE_COLS`? |
|---|---|---|---|
| [0](#step-0--get-set-up) | Get set up: data, environment, up-to-date docs | — | No |
| [1](#step-1--choose-one-user-and-one-decision) | Choose one user and one decision | 0 | No |
| [2](#step-2--measure-the-dashboards-real-accuracy) | Measure the dashboard's real accuracy | 0 | No |
| [3](#step-3--feed-the-models-weather-for-the-chosen-date) | Feed the models weather for the chosen date | 2 | No — the values change, not the columns |
| [4](#step-4--only-show-predictions-the-model-can-support) | Only show predictions the model can support | 1 | Only if station type becomes a feature |
| [5](#step-5--show-observed-data-beside-predictions) | Show observed data beside predictions | 0 | No |
| [6](#step-6--replace-labels-with-probabilities-and-ranges) | Replace labels with probabilities and ranges | 1, 2 | No — the `.pkl` dict gains a key |
| [7](#step-7--improve-the-models) | Improve the models | 1–6 | Usually |

Steps 1 and 2 can run in parallel, and step 5 can be done at any point. Step 1 is
a conversation rather than code, but it sets the scope of steps 4, 6 and 7 —
skip it and you end up tuning 52 models for no particular purpose.

---

## Step 0 — Get set up

### Environment

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt     # full dev environment: notebooks, geospatial, plotting
python -m unittest test_app.py      # passes on a fresh clone
python3 app.py                      # then open http://127.0.0.1:8050
```

### Data

Git tracks only `data/stations.csv`. Everything else under `data/` (~8 GB) is
gitignored. The dashboard and the tests run without it; nothing else does.

| To… | You need | Size |
|---|---|--:|
| Run the dashboard or the tests | nothing extra | — |
| Run the EDA notebooks, retrain models, or run the step 2 backtest | `data/final/epa-full.csv` | 81 MB |
| Build per-date weather (step 3) or rainfall totals (step 7) | `data/tabular/02_clean/climate/prism-iowa-climate-clean.csv`<br>`data/tabular/02_clean/climate/isu-climate-clean.csv`<br>`data/tabular/02_clean/streamflow/usgs-iowa-discharge-clean.csv` | 239 MB<br>46 MB<br>16 MB |
| Rebuild everything from raw | all raw inputs (`DATA.md` lists them) | ~7.1 GB |

Ask the previous maintainers for a copy first. Rebuilding from raw is only partly
scripted: `src/01_download/` has notebooks for CDL cropland, ISU climate, ECHO
facilities, Iowa NRS BMPs, PRISM, SSURGO soil and USGS streamflow, but **none for
the Water Quality Portal measurements, USDA NASS agriculture or Census tables** —
the core measurements included. `DATA.md` documents where those came from; if you
rebuild, write the missing download notebooks as you go.

After any change to `epa-full.csv`, run `python3 build_station_table.py` and
commit the new `data/stations.csv` — it is the only data file the deployed app
reads.

### Bring the documentation up to date

Do this first. It takes an hour or two, and the stale parts actively mislead
newcomers.

**`src/04_eda/eda-summary.md`**

- Add a **Status** column to the §2 red-flags table. Statuses verified in
  September 2026:

  | # | Red flag | Status |
  |---|---|---|
  | 1 | Random split leaks stations | **Fixed** — all four notebooks split on station |
  | 2 | Skill comes from station identity | **Measured, still open** — location is the second-largest predictor block (§1.12) |
  | 3 | Persistence baseline nearly matches the models | **Addressed** — published beside every score; SC, TDS and TP still lose to it |
  | 4 | Dashboard serves weather from the wrong date | **Open** — step 3 |
  | 5 | `pct_row_crops` makes the design matrix singular | **Fixed** — dropped from `FEATURE_COLS` |
  | 6 | `error_rate` (sMAPE) tracks zero fraction | **Mitigated** — kept, with caveats in the app and docs |
  | 7 | Nitrite modeled as continuous | **Open** — step 7 |
  | 8 | Median imputation fills gaps from a different subpopulation | **Partial** — only the neural network adds missing-value indicators |
  | 9 | Map interpolates past the correlation distance | **Mitigated** — surface off by default; no distance mask (step 4) |
  | 11 | Nutrient block is four columns of about two facts | **Open** — step 7 |
  | 12 | E. coli tail mixes real counts with an artefact | **Open** — still only a `valid_range` of (0, 1e6) |
  | 13 | Nutrient-budget features up to 8 years stale | **Open** — `np_years_stale` is not a feature |
  | 14 | Specific Conductance and TDS are one target | **Open** — both still modeled |

  Flags 10 and 15–18 describe the data rather than a defect; mark them
  "informational" or check them one by one.
- §3.1 quotes the leaky-split scores (Gradient Boosting mean R² 0.547, "9 of
  12"). Label it historical and point to `model_outcomes.md`.
- §4.1 says "36 `.pkl` files" and three training notebooks — now 52 and four.
- §5 calls the grouped re-fit "the open question". It has been done; say so.

**`src/05_modeling/model_outcomes.md`**

- "`app.py` currently loads only the first three families (39 models)" — it
  loads all four (52).
- The WQI section's text quotes Random Forest R² 0.337 and margin +0.168; its own
  table, and `model_metrics.csv`, say 0.288 and +0.121.
- The "Against the memorization baseline" table and several Takeaways use
  pre-size-cap random forest scores — e.g. Specific Conductance 0.655 (now 0.546)
  and Water Temperature 0.947 (now 0.938). Regenerate those tables from
  `model_metrics.csv` rather than patching numbers by hand.

**Done when:** someone reading only the docs would reach the same conclusions as
the README's Current Status section.

---

## Step 1 — Choose one user and one decision

**Why first.** The dashboard offers 13 targets × 4 model families × any date ×
every station. That is a demo. A tool answers one question for one kind of
person, and that question fixes everything downstream: the target, the
locations, the time horizon, the threshold, and which metric counts as success.
R² is almost never that metric.

**Candidates the existing data can support:**

| Use case | Who decides | Target | Where | Horizon | Success looks like | Already in hand | Biggest gap |
|---|---|---|---|---|---|---|---|
| Recreational advisories | Beach and park managers | E. coli | Lakes and river access points (159 lakes, 434 stations with E. coli data) | Days, May–Sept | Most days above 235 MPN/100 mL flagged, few false alarms | Best log-scale R² 0.36; rainfall + humidity is its top predictor block | Needs actual recent rainfall (step 3), not seasonal averages |
| Nitrate exceedance | Drinking-water utilities, nutrient-reduction programmes | Nitrate or Nitrate + Nitrite | River and stream sites | Weeks to a season | A well-calibrated probability of exceeding 10 mg/L | Strong agricultural signal; streamflow drives change within a site | Only 278–367 stations measured it; most signal is *between* sites, not over time |
| Where to sample next | State monitoring programme | Start with one | Unmonitored HUC-12 watersheds | None — typical conditions | Correct ranking and honest ranges at held-out stations | Exactly what the station-grouped split tests | App predicts only at existing stations; features must be computed for new places |
| Thermal stress | Fisheries and habitat managers | Water Temperature | Streams | Days | Error in °C; probability above a species threshold | R² 0.94, a real physical mechanism | Same-day air temperature (step 3); a threshold chosen with a domain expert |

**What to do.**

1. Talk to at least one potential user — someone at the Iowa DNR, a county
   conservation board or a water utility. For a research deliverable, pick the
   use case and justify it in writing.
2. Before committing, run a short descriptive analysis of the **observed** data
   for that target: how often, where and when the threshold is exceeded, broken
   down by station type, month and year. The current EDA does not do this, and it
   tells you whether the decision is worth modeling — a threshold that is never
   crossed needs no model.
3. Write a half-page **use-case statement** near the top of the README: user,
   decision, target, locations, horizon, threshold, and the error the user can
   tolerate.
4. Scope the dashboard to it. Hide out-of-scope targets and families in the UI;
   keep them in the notebooks and `model_metrics.csv` for the record.

**Done when:** the use-case statement exists, and steps 4, 6 and 7 have a named
target and threshold to aim at.

---

## Step 2 — Measure the dashboard's real accuracy

**Why.** Every score in `model_metrics.csv` was computed with the weather
recorded on each sample's date. The dashboard supplies weather from each
station's last visit instead. What that costs has only been spot-checked
(January water temperature 11 °C too warm), never measured. Until it is, no
later fix can be verified.

**Build `src/05_modeling/serving_backtest.py`.** Model it on
`src/04_eda/block_permutation_importance.py`, whose `held_out()` already rebuilds
each target's held-out stations and reproduces `model_metrics.csv`. For each
target and family:

1. Rebuild the held-out rows: `GroupShuffleSplit(n_splits=1, test_size=0.2,
   random_state=42)` grouped on `MonitoringLocationIdentifier`, with the same
   `valid_range` filters as the notebooks.
2. Build each row's features **the way the app does**: the station's 25 base
   features from `data/stations.csv`, plus the four date features from the
   sample date via `app._add_temporal_features`. Reusing the app's own functions
   keeps the backtest from drifting away from what users see.
3. Predict, back-transform with `app._to_raw_scale`, and score R² and MAE — plus
   precision and recall at the step 1 threshold once it exists.
4. Write `src/05_modeling/serving_metrics.csv`, with test-time and served scores
   side by side.

If time allows, score two variants. **As served** uses the station's last visit
exactly as the app does, even when that visit came after the sample. **As of**
uses the last visit *before* the sample date — what a real forecast would have
had available. The gap between the test score and the as-of score is the true
cost of not knowing the day's weather.

**Surface the result.** Put the served score beside the held-out score in the
dashboard's Model accuracy panel and in `model_outcomes.md`.

**Add a seasonal sanity test** to `test_app.py`: the statewide median January
water temperature prediction should be below 5 °C. It fails today (11.6 °C), so
mark it `@unittest.expectedFailure` until step 3 lands; the fix then shows up as
an "unexpected success".

**Done when:** every model has a served score beside its test score, and the
dashboard shows the served one.

---

## Step 3 — Feed the models weather for the chosen date

**The problem, precisely.** Eleven of the 25 base features change from day to
day at a station: the four `prism_*` columns, the six `isu_*` columns and
`streamflow_discharge_cfs`. `app.py::build_feature_matrix` freezes all eleven at
the station's last visit — median September 2022, and in April–October for 87%
of stations. Only `doy`, `doy_sin`, `doy_cos` and `obs_year` move with the date.

### First fix: seasonal averages per station

For each station and each of the eleven columns, compute the typical value on
every day of the year, and use that at inference.

| Columns | Source | Join to the station via |
|---|---|---|
| `prism_*` | `prism-iowa-climate-clean.csv` — daily, keyed on `station_id` + `date`, 2015–2025, 1,666 stations | `station_id` = `MonitoringLocationIdentifier` |
| `isu_*` | `isu-climate-clean.csv` — daily, keyed on `station` + `day` | `climate_station` in `stations.csv` (e.g. `MUT`) |
| `streamflow_discharge_cfs` | `usgs-iowa-discharge-clean.csv` — daily, keyed on `site_no` + `date` | `streamflow_site_no` in `stations.csv` — stored as a number, so zero-pad it to 8 digits (`5465500` → `05465500`) |

- **Use every day of the record**, not only the days the station was sampled.
  Sampling days are summer-heavy (45% of samples fall in June–August); that is
  the bias being fixed.
- **A smooth fit is compact and denoises**: mean plus two or three annual
  sine/cosine terms per column. For precipitation and snow, which are mostly
  zero and spiky, use a ±15-day moving average by day of year instead — or drop
  snow and wind in step 7 (they contribute nothing, EDA §1.12).
- **Storage without breaking deployment.** `app.py` may read nothing under
  `data/` except `stations.csv` (a deliberate rule; see `CLAUDE.md`). Either
  store the fitted coefficients as extra columns in `stations.csv`, computed by
  `build_station_table.py` (~55 columns, negligible size), or commit a separate
  small file and change the rule on purpose — updating `CLAUDE.md`, the README's
  Deployment section and `build_station_table.py` together.
- **Label it for what it is**: *typical conditions for this station at this time
  of year*, not the weather on that date. Say so in the map footnote.

### Fix the year at the same time

- **Clamp `obs_year` to 2025** (the last training year) in
  `app.py::_add_temporal_features`. Tree models already behave this way, but
  linear regression and the neural network extend trends past the data — the
  neural network's median Nitrate + Nitrite prediction doubles from 8.4 to
  18.1 mg/L between 2025 and 2030. This is a one-line change; ship it before the
  rest of this step.
- Later, test whether `obs_year` earns its place: drop it in the notebooks and
  compare grouped-split scores.
- Remove the "In 6 months" and "In 1 year" shortcuts, or relabel them "typical
  for this time of year". Rename the "Recent trend" panel to "Seasonal pattern"
  unless it is fed real recent data.

### Later, if the step 1 decision needs it

- **Actual recent weather**, fetched from the sources already used: the Iowa
  Environmental Mesonet (`mesonet.agron.iastate.edu`, used by
  `src/01_download/climate-download.ipynb`) and PRISM (used by
  `prism-climate-download.ipynb`). This needs network access from the host, a
  cache, and any new client library added to `run-requirements.txt`.
- **Forecast weather** for roughly the next week from the National Weather
  Service, falling back to seasonal averages beyond it.

The E. coli use case in particular needs real recent rainfall — seasonal
averages cannot capture a storm.

**Done when:** the step 2 backtest shows served scores closer to test scores; the
January statewide median water temperature prediction is within a few degrees of
the observed 0.6 °C (remove `expectedFailure` from the step 2 test); and no
family's July 2030 map differs from its July 2025 map.

---

## Step 4 — Only show predictions the model can support

**The problem.** Every target is drawn at all 1,345 stations — 559 River/Stream,
318 Lake, 190 Well, 158 Wetland Undifferentiated, 68 Stream and a handful of
smaller types — but station type is not a model input. So E. coli appears at 186
wells that never measured it and nitrate at 315 lakes that never did. Wells
sample groundwater, a different medium from the surface water most targets were
measured in.

**What to do.**

1. **Group the station types**: stream (River/Stream, Stream, River/Stream
   Perennial, Stream: Ditch, …), lake, wetland, groundwater (Well), other.
2. **Add a serving rule** — no retraining needed: show a target at a station
   only if that station's type group had at least *N* training stations for the
   target (start at N = 20). Put the number of hidden stations in the map
   subtitle, so nothing disappears silently.
3. **Decide on groundwater.** Most likely exclude wells from surface-water
   targets (E. coli, turbidity, suspended solids) or model them separately.
4. **Test station type as a feature.** Add the type group as one-hot columns,
   retrain all four notebooks, and compare grouped-split scores and permutation
   importance. `eda-summary.md` §4.3 advises against this, but that advice was
   written when the split leaked stations; with whole stations held out, a new
   site's type is known when you predict for it, so it is not leakage. The
   remaining risk is that type stands in for *which programme ran the site* —
   check whether its importance holds up within a single organization.
5. **Fix or remove the interpolated surface.** Mask it beyond the distance over
   which each target stays spatially correlated
   (`src/04_eda/outputs/bv_spatial_autocorrelation.csv`, EDA §4.5), and never
   interpolate across station types.

**Done when:** no target is shown at a station type it was never measured at,
the map says how many stations are hidden, and a test checks it (e.g. no wells on
the E. coli map).

---

## Step 5 — Show observed data beside predictions

**The problem.** At stations with history, the dashboard's predictions mostly
restate it (Spearman 0.60–0.91 against each station's own median), and for
Specific Conductance, TDS and Total Phosphorus the station's last reading beats
every model. The hover panel shows none of that data.

**What to do.**

1. **Extend `build_station_table.py`** to add, per station and per target: last
   observed value, *that target's own* last sample date, sample count, median,
   and 10th/90th percentiles (~80 small columns). Don't reuse the row's
   `ActivityStartDateTime` — `groupby.last()` stitches columns together from
   different visits, so it is not the date of any one target's value.
2. **In the Point detail panel**, show that summary beside the prediction,
   including how long ago the site was last sampled.
3. **Where the selected model loses to persistence** (`model_minus_persistence`
   < 0 in `model_metrics.csv`), lead with the observed value and show the
   prediction second.
4. Optionally trim `stations.csv` to the columns the app uses; it currently
   carries all 318.

**Done when:** hovering any station shows its observed history for the selected
target, or says it has none.

---

## Step 6 — Replace labels with probabilities and ranges

**The problem.** The "Low" / "High Concern" labels come from a single predicted
value, yet typical errors are as large as the thresholds — the best E. coli
model's mean absolute error is about 1,490 MPN/100 mL, against a 235 threshold.
A decision needs "how likely is it over the line?", not a point estimate.

### Recommended: ranges from out-of-fold errors

This is split-conformal prediction. It needs only stock scikit-learn and plain
numbers, so it respects the rule that pickles hold no custom classes.

1. **Collect out-of-fold residuals** in each training notebook, from `GroupKFold`
   over training stations. The transform bake-off already runs this
   cross-validation, so reuse it. Keep log-fitted models' residuals on the log
   scale.
2. **Store a residual percentile grid** (1st–99th) in the `.pkl` dict as a new
   key beside `log_offset` and `smearing_factor`. Keep `app.py::load_models`
   tolerant of artifacts without it, as it already is for bare pipelines, and
   document the key in `CLAUDE.md`.
3. **At inference**, an 80% range is the prediction plus the 10th and 90th
   residual percentiles (back-transformed for log models), and P(value >
   threshold) is the share of the residual grid above (threshold − prediction).
4. **Watch the width.** Raw-scale errors grow with the level of the target. If
   ranges are too narrow at high values, bin residuals by predicted level or
   work on the log scale.

The alternative, `HistGradientBoostingRegressor(loss="quantile")` for the 10th
and 90th percentiles, is more flexible but adds two pickles per target, against a
startup footprint already at ~470 MB of the host's 512 MB.

### Evaluate on held-out stations

- **Coverage**: an 80% range should contain about 80% of held-out observations.
- **At the step 1 threshold**: Brier score, a reliability plot, and precision and
  recall at a chosen probability cut-off.
- **Against two simple baselines**: the station's own historical exceedance rate
  (where it has history), and the statewide exceedance rate for that month.

### In the app

Replace the label with the probability and the range — e.g. "62% chance above
235 MPN/100 mL; likely 80–900". Move every threshold, with its source, into one
table in `app.py`; today they are scattered through `_target_assessment`.

Adding columns to `model_metrics.csv` is safe, but each notebook's publish step
drops all rows when it sees a schema it doesn't recognise, so **run all four
notebooks** afterwards.

**Done when:** held-out coverage is within a few points of nominal, the
exceedance probability beats both baselines on Brier score, and the dashboard
shows probabilities instead of labels.

---

## Step 7 — Improve the models

Only now: a better score on the wrong inputs still draws a wrong map. Prioritise
by the step 1 use case. Section numbers refer to `src/04_eda/eda-summary.md`.

| Change | Why | Watch out for |
|---|---|---|
| **Previous observation as a feature** (`y_prev`, `days_since_prev`) | Persistence beats every model on SC, TDS and TP (§3.3) | At serving time the gap since the last visit is often years, while training revisit gaps have medians of 1–33 days. Use it only where a site was sampled recently; needs step 5's per-target last value |
| **Two-part model for Nitrite and Nitrate** — P(detected), then amount if detected | 84% and 38% zeros, most likely non-detects (§4.4) | Decide what a zero means first. Store both pipelines in the `.pkl` dict and combine them in `app.py` |
| **Rainfall totals** over the previous 3, 7 and 30 days | Runoff follows accumulated rain; rainfall + humidity is E. coli's top block (§1.12, §4.3) | Needs the daily PRISM file from step 3 at both training and serving time |
| **Fewer redundant predictors** — `prism_tmax_c` + `prism_tmin_c` from the six-column temperature cluster, one N and one P nutrient column, no snow or wind | Collinearity up to VIF 61; snow and wind contribute nothing (§1.7, §1.12, §4.1) | Changes `FEATURE_COLS` |
| **Missing-value indicators for the tree families** | Where soil or discharge data is missing, the target differs systematically (§4.4) | Changes the pipeline, not `FEATURE_COLS` |
| **Treat Specific Conductance and TDS as one target** | Correlation 0.984 (§1.8) | Decide which to keep in the UI |
| **A written E. coli censoring rule** | The (0, 1e6) range silently keeps a 610,000 reading but drops 2,420,000 (§4.4) | Apply the same rule to the observed summaries in step 5 |
| **Filter WQI rows by `WQI_weight_coverage`** | About a tenth of WQI's variance is how many parameters were measured (`model_outcomes.md`) | Fewer training rows |
| **Multi-task neural network** | Lets sparse targets borrow from well-sampled ones (`model_outcomes.md`) | Needs a tool beyond `MLPRegressor`, without a custom class in the pickle |

For every change: rerun all four notebooks, check the persistence margin, rerun
the step 2 backtest, and regenerate `model_outcomes.md` from `model_metrics.csv`.

---

## Rules that keep the project working

From `CLAUDE.md`. Breaking one of these usually fails silently, or only on deploy.

- **Feature order is a contract.** `FEATURE_COLS` must match across `app.py`, the
  four training notebooks and `src/04_eda/block_permutation_importance.py`. After
  changing it, retrain all four families — the app disables any model whose
  stored `feature_cols` disagree.
- **Run all four notebooks after a metrics schema change.** The publish step
  drops rows it does not recognise.
- **Never tune on the test set.** Transforms, epochs, thresholds and interval
  widths are chosen by grouped cross-validation on training stations only.
- **Keep the random forests capped** (`n_estimators=100, min_samples_leaf=10`)
  and gunicorn at **one worker** — startup uses ~470 MB of the host's 512 MB.
- **`app.py` reads only `data/stations.csv` from `data/`.** Rebuild it with
  `build_station_table.py` whenever `epa-full.csv` changes, and commit it.
- **Pickles are plain dicts of stock scikit-learn objects** — no custom classes.
- **A new import in `app.py` goes into `run-requirements.txt`.** The dev
  environment hides the omission until deploy.
- **Run `python -m unittest test_app.py` before committing.**

---

## Open questions to settle early

- Who is the user, and what decision do they make? (Step 1)
- Should the tool cover only existing monitoring stations, or unmonitored places
  too?
- Is groundwater (wells) in scope at all?
- May the deployed app call external services for live data, or must it stay
  self-contained?
- How much error can the decision tolerate — and does the step 2 backtest say
  that is achievable?

---

## Appendix: reproducing the README figures

The figures in the README's Current Status section were measured on 2026-09-13
against the committed models. Save the script below anywhere, then run it from
the repo root with the virtual environment active and `data/final/epa-full.csv`
present:

```bash
PYTHONPATH=. python path/to/check_serving.py
```

```python
# check_serving.py — what the dashboard serves, compared with the observed data.
from datetime import date
import numpy as np, pandas as pd
import app  # loads the 52 models and data/stations.csv

S = app.STATIONS
last = pd.to_datetime(S[app.DATE_COL])
print("median last visit:", last.median().date(),
      "| share Apr-Oct:", round(last.dt.month.between(4, 10).mean(), 3))

full = pd.read_csv("data/final/epa-full.csv", low_memory=False)
full["month"] = pd.to_datetime(full[app.DATE_COL]).dt.month

def served(target, family, d):  # exactly what the map shows
    return np.asarray(app._cached_station_predictions(target, family, d.isoformat()))

# 1. Seasonal cycle: observed vs served
for target, family in [("Water Temperature", "Gradient Boosting"), ("Dissolved Oxygen", "Random Forest")]:
    obs = full.groupby("month")[app.TARGET_COLS[target]].median()
    for m in (1, 4, 7, 10):
        print(f"{target} month {m:2d}: observed {obs[m]:5.1f}  served {np.median(served(target, family, date(2026, m, 15))):5.1f}")

# 2. Year: 2030 vs 2025, same day of year
for target, family in [("Water Temperature", "Gradient Boosting"), ("Nitrate + Nitrite", "Neural Network"), ("Turbidity", "Linear Regression")]:
    a, b = served(target, family, date(2025, 7, 15)), served(target, family, date(2030, 7, 15))
    print(f"{family} / {target}: median 2025 {np.median(a):.2f} -> 2030 {np.median(b):.2f}")

# 3. Station types shown vs measured
types = full.groupby("MonitoringLocationIdentifier")["MonitoringLocationTypeName"].first()
for target in ("Nitrate", "E. coli", "Specific Conductance"):
    measured = set(full.loc[full[app.TARGET_COLS[target]].notna(), "MonitoringLocationIdentifier"])
    print(target, "| measured at", len(measured), "| never measured, by type:",
          types[~types.index.isin(measured)].value_counts().head(4).to_dict())

# 4. Served prediction vs the station's own history
for target, family in [("Specific Conductance", "Random Forest"), ("Nitrate + Nitrite", "Gradient Boosting")]:
    col = app.TARGET_COLS[target]
    g = full[full[col].notna()].groupby("MonitoringLocationIdentifier")[col]
    hist = g.median()[g.size() >= 5]
    pred = pd.Series(served(target, family, date(2026, 7, 15)), index=S["MonitoringLocationIdentifier"])
    both = pd.concat([pred, hist], axis=1, join="inner")
    print(f"{target}: Spearman(served, station median) = {both.corr(method='spearman').iloc[0, 1]:.2f} over {len(both)} stations")
```

Output with the models committed as of this writing:

```
median last visit: 2022-09-12 | share Apr-Oct: 0.868
Water Temperature month  1: observed   0.6  served  11.6
Water Temperature month  4: observed  10.5  served  15.5
Water Temperature month  7: observed  24.5  served  20.8
Water Temperature month 10: observed  14.4  served  16.4
Dissolved Oxygen month  1: observed  12.2  served   9.8
Dissolved Oxygen month  4: observed  10.9  served   9.8
Dissolved Oxygen month  7: observed   8.0  served   8.8
Dissolved Oxygen month 10: observed   8.5  served   8.6
Gradient Boosting / Water Temperature: median 2025 20.76 -> 2030 20.76
Neural Network / Nitrate + Nitrite: median 2025 8.36 -> 2030 18.07
Linear Regression / Turbidity: median 2025 13.76 -> 2030 7.76
Nitrate | measured at 278 | never measured, by type: {'River/Stream': 414, 'Lake': 315, 'Wetland Undifferentiated': 158, 'Well': 136}
E. coli | measured at 434 | never measured, by type: {'River/Stream': 323, 'Well': 186, 'Lake': 159, 'Wetland Undifferentiated': 158}
Specific Conductance | measured at 483 | never measured, by type: {'River/Stream': 494, 'Lake': 127, 'Wetland Undifferentiated': 119, 'Well': 91}
Specific Conductance: Spearman(served, station median) = 0.91 over 287 stations
Nitrate + Nitrite: Spearman(served, station median) = 0.60 over 248 stations
```

The Spearman range quoted in the README (0.60–0.91) spans ten targets; this
script checks its two ends. Once step 2 exists, fold these checks into
`serving_backtest.py` so they stay current.
