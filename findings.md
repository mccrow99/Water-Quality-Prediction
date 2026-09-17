# Findings

What this project has established about predicting water quality in Iowa, and
about the data it used to do it.

**Scope.** Iowa statewide, 2015-01-02 → 2025-12-25. The terminal modeling table
(`data/final/epa-full.csv`) holds **48,251 station-day rows at 1,345 monitoring
stations** — ~971K EPA WQX observations pivoted to one row per station-day, then
joined to daily climate (ISU/IEM + PRISM), USGS streamflow, SSURGO soil, USDA
cropland fractions, county nutrient loading, and regulatory/demographic context.
**29 predictors, 13 targets, 4 model families.**

**Sources.** Numbers here come from `src/05_modeling/model_metrics.csv` (the
source of truth for every score) and `src/04_eda/eda-summary.md`. Where
`model_outcomes.md` prose disagrees with the metrics table, the table wins — some
of that prose predates the random-forest size cap. Detail for everything below
lives in those two documents.

---

## 1. Which predictors are associated with water quality

Two measurements answer this, and they disagree in an informative way.

### Measured on the trained models

Permuting a block of predictors on **held-out stations** and recording the R²
lost (`src/04_eda/block_permutation_importance.py`; full derivation in
`eda-summary.md` §1.12). Correlated columns are permuted **as a block** — the
five temperature columns are one fact (ρ = 0.955 between two of them) and the
four nutrient columns span about two dimensions, so shuffled individually they
all look unimportant because the model reads the survivors.

Mean over the 25 (target × family) fits with base R² > 0.05. Repeat-to-repeat SD
is 0.003, so anything under ~0.01 is noise.

| Block | mean R² drop | share of model skill | largest single drop |
|---|--:|--:|---|
| Air temperature | 0.159 | 32% | 0.90 (Water Temperature) |
| Location (lat/lon) | 0.158 | 47% | 0.65 (Total Dissolved Solids) |
| Season (day-of-year) | 0.123 | 29% | 0.32 (Nitrate + Nitrite) |
| Nutrient budget (fertiliser/manure N, P) | 0.098 | 25% | 0.35 (Nitrate) |
| Land cover (corn, soy, developed, forest) | 0.072 | 20% | 0.25 (Specific Conductance) |
| Network geometry (distance to gauge/station) | 0.038 | 13% | 0.12 (Turbidity) |
| Year | 0.032 | 9% | 0.18 (Nitrate + Nitrite) |
| Precipitation + humidity | 0.028 | 8% | 0.17 (E. coli) |
| Soil (Ksat, available water capacity) | 0.021 | 4% | 0.21 (Specific Conductance) |
| Streamflow discharge | 0.020 | 7% | 0.10 (Specific Conductance) |
| Wind | 0.004 | 1% | 0.01 (Dissolved Oxygen) |
| Snow / snow depth | 0.000 | 0% | 0.00 (pH) |

### The four conclusions

**Air temperature is the only block that is a mechanism rather than a proxy.**
`prism_tmin_c` ↔ Water Temperature is **ρ = +0.849** — the strongest relationship
in the dataset — and it is the one strong relationship that survives being split
into its parts: **+0.809 between stations and +0.868 within a single station.**
It predicts *when* a site is warm, not merely *which* sites are warm. It carries
Dissolved Oxygen (ρ = −0.44) and pH too, by the same physics. Caveat: season
takes a further 0.21 R² off Water Temperature on top of it, so part of what looks
thermal is the calendar, which is already a feature.

**Location ranks second and is not a predictor at all.** `LongitudeMeasure` alone
is the largest single-column drop in the entire run (mean 0.116, more than double
latitude's) — on a column whose correlation with Specific Conductance is
**ρ = 0.0007** while its conditional median runs 327 → 703 µS/cm. Coordinates are
~1 distinct value per station, so the trees are using them to recall station
identity. On the three targets where location is the top block — Specific
Conductance, TDS, Total Phosphorus — the models also fail to beat a no-feature
baseline (§3). Roughly half their measured skill is a station-level mean
recovered from coordinates.

**Agriculture is the strongest non-thermal signal, and it is entirely
cross-sectional.** `pct_corn` → Nitrate is **ρ = +0.489**, second-largest in the
dataset. But it decomposes to **+0.714 between stations and −0.001 within one**,
and it is a cliff rather than a slope: conditional median **0 mg/L across the
bottom four deciles of corn share, 2 mg/L across the middle five, 10 mg/L in the
top decile**. Sites in corn country have more nitrate; a given site's nitrate on a
given day has nothing to do with its corn share. The nutrient-budget block behaves
the same way (between 0.26, within 0.02), as does land cover (0.15 vs 0.04).

**Streamflow is the variable this ranking undersells.** It places 10th of 12, and
its mean |ρ| is only 0.148 — yet it is the **sole block whose within-station
correlation exceeds its between-station one** (0.28 vs 0.14), reaching **+0.557**
within station for Nitrate + Nitrite, **+0.526** for Total Suspended Solids and
**+0.459** for Turbidity. It also produces the clearest sign reversal in the data:
Specific Conductance ← discharge is **+0.150 between stations but −0.364 within
one** — the dilution effect, visible only once you stop comparing different
rivers. A ranking pooled across stations cannot see this, because the
between-station structure it competes against is far larger.

**Wind, snowfall and snow depth contribute nothing** — 0.004, 0.000 and 0.001
mean drop, at or below the noise floor on every target. (Measured on the
post-fix file, after a cleaning bug that had fabricated ~190k zero-snow rows was
corrected, so this is honest sparsity rather than an artefact. It says these
columns *as fed to the model* carry no signal, not that snowmelt is irrelevant.)

### The short version

> **Temperature drives the physical targets. Corn and applied nutrients mark
> which places are polluted, not when. Streamflow drives the day-to-day movement
> in sediment and nitrate but is masked by between-site variation. Everything
> else is close to noise — and the single most "important" column, longitude, is
> not a predictor but a station name.**

### Why the raw correlation table is misleading

Three structural facts make single-feature correlations a poor summary here, all
established in `eda-summary.md` §1.5–1.7:

- **Only one target has any feature above |ρ| = 0.5** (Water Temperature). The
  other ten top out between 0.209 and 0.442.
- **Only 33 of 360 target–feature pairs (9%) are cleanly monotone; 212 (59%)
  carry no shape a single feature could use** — they are flat, irregular or
  single-peaked. Ranking pairs by correlation and by effect size agree at only
  ρ = +0.56.
- **The 30 columns are about 19 independent facts.** 21 feature pairs exceed
  |ρ| = 0.70, eight features have VIF ≥ 10 (the four nutrient columns at 61.3,
  57.1, 55.8, 54.6), and the correlation matrix is singular — `pct_row_crops`
  equals `pct_corn + pct_soybean` exactly on all 48,251 rows. (That column has
  since been excluded from the feature set.)

---

## 2. How well each target can be predicted at an unseen station

Best family per target, scored on **stations held out whole**. `†` = fitted and
read on log10 scale. Margin is against persistence ("repeat this station's
previous value"), a model with no features.

| Target | Best family | R² | Margin vs. persistence | Test stations |
|---|---|--:|--:|--:|
| Water Temperature | Gradient Boosting | 0.940 | +0.302 | 200 |
| Specific Conductance | Random Forest | 0.546 | **−0.306** | 97 |
| Total Dissolved Solids | Random Forest | 0.533 | **−0.276** | 117 |
| Nitrate + Nitrite | Gradient Boosting | 0.515 | +0.285 | 74 |
| Dissolved Oxygen | Random Forest | 0.476 | +0.162 | 182 |
| Nitrate | Random Forest | 0.455 | +0.056 | 56 |
| pH | Random Forest | 0.375 | +0.440 | 222 |
| E. coli † | Gradient Boosting | 0.358 | +0.520 | 87 |
| Total Suspended Solids † | Random Forest | 0.347 | +0.252 | 111 |
| Turbidity † | Random Forest | 0.296 | +0.247 | 169 |
| WQI | Random Forest | 0.288 | +0.121 | 189 |
| Total Phosphorus † | Gradient Boosting | 0.189 | **−0.188** | 91 |
| Nitrite | Random Forest | 0.011 | +0.701 | 46 |

Mean best-family score across the 13: **0.41**. One target is genuinely well
predicted (Water Temperature), four sit around 0.45–0.55, and the rest explain
between a tenth and a third of the variance at a site never seen in training.

---

## 3. The largest finding was a measurement error, not a model result

**Most of the project's original apparent accuracy was station recall.** The
first version split train/test on *rows*, which left 99.1–99.7% of test rows at a
station that also appeared in training. With latitude and longitude in the feature
set at ~1 distinct value per station, a tree could identify the site and recall its
typical level — and that scored as prediction.

Re-splitting on `MonitoringLocationIdentifier` (whole stations held out, no site
on both sides) moved mean raw-scale R² across the 12 targets then modelled from
**0.546 → 0.28** for gradient boosting and **0.538 → 0.34** for random forest.
The collapse concentrated exactly where the diagnosis predicted:

| Target | Row split | Station split (current) |
|---|--:|--:|
| Specific Conductance (GB) | 0.888 | 0.262 |
| Total Dissolved Solids (RF) | 0.824 | 0.533 |
| Water Temperature (GB) | 0.952 | 0.940 |

(Before/after from `model_metrics_random_split.csv`, the frozen pre-split table.
Random-forest "after" figures also carry a later size cap that costs a mean 0.018
R², so for that family the two changes are not fully separated; gradient boosting
is uncapped and isolates the split effect.)

**The models did not get worse; the measurement got honest.** Water Temperature —
the one target with a real mechanism in the features — barely moved.

## 4. Three targets lose to a model with no features at all

**10 of 13 targets beat persistence; three do not** — Specific Conductance
(−0.306), Total Dissolved Solids (−0.276) and Total Phosphorus (−0.188). For
these, *a site's last reading is a better forecast than any model here.*

This is consistent with their deseasonalised autocorrelation: Specific Conductance
and TDS still correlate at **0.77 and 0.75 across visits two to four years apart**.
They are site constants plus a small residual — properties of the place, not of
the day. The right response is to hand the model that last reading explicitly
(`y_prev`, `days_since_prev`) rather than leaving it to be approximated from
coordinates.

**Two targets are one target.** Specific Conductance ↔ TDS correlate at
**ρ = 0.984** (TDS is conventionally *derived* from conductance), so they are
being modelled and reported twice.

## 5. Model family is the second-order decision

Mean score across the 13 targets, each on the scale it is fitted on:

| Family | Mean | Wins |
|---|--:|--:|
| Random Forest | 0.402 | 9 |
| Gradient Boosting | 0.358 | 4 |
| Linear Regression | 0.230 | 0 |
| Neural Network | 0.221 | 0 |

**The neural network adds nothing.** A three-seed MLP ensemble, trained on the
same split with the same transform selection, is beaten on every target, landing
at linear regression's level to within noise. Its losses concentrate where trees
cut sharply on coordinates (Specific Conductance 0.546 → 0.020) and nearly vanish
where the target has a smooth physical driver (Water Temperature 0.938 → 0.921).
Its one nominal win (Turbidity) disappears when read on the log scale the target
is actually fitted on, and was in any case smaller than its own seed-to-seed
spread of 0.090. This is the expected outcome for tabular data at this scale, and
it is worth having measured rather than assumed.

**The gap between the best family and the worst is smaller than the gap created by
fixing the split.** Feature quality and evaluation design dominate estimator
choice here.

## 6. Data artefacts that cap achievable accuracy

Several limits are in the measurements themselves and no model can cross them:

- **Zero-inflation.** Nitrite is **84.1% zeros**, Nitrate 38.3% — almost certainly
  non-detects recorded as 0. Nitrite's interquartile range is exactly 0, which is
  why its best R² is 0.011: it is a hurdle problem being modelled as a continuous one.
- **Coarse rounding.** pH is exactly 8.0 on **16.3%** of observations and DO exactly
  8.0 on 12.7%; 86% of nitrate values are whole numbers. pH is not measured to
  whole units in a lab — this is rounding upstream, and it puts a floor under RMSE.
- **Censoring.** 10.7% of E. coli values sit above the 2,419.6 MPN/100 mL IDEXX
  ceiling, and the maximum (2,420,000) is ~1,000× the ceiling and not plausible.
- **Stale features.** Nutrient-budget columns are up to 8 years old relative to the
  observation they describe.
- **Programme structure.** Two target pairs share *zero* rows — they are measured by
  monitoring programmes that never overlap. Organisation identity explains more of
  some targets than any feature does.
- **WQI is partly sampling design.** `WQI_n_groups` — how many of the eight pollution
  groups a sample actually measured — explains **9.7%** of WQI's variance on its own.
  A WQI from four measurements and one from eight share a scale but are not the same
  quantity.

## 7. The dashboard does not show what was measured

The scores above were computed with **the weather actually recorded on each
sample's date**. The dashboard instead gives each station the weather from its
**last visit**, changing only day-of-year and year. The median station's last visit
was September 2022, and 87% of last visits fell April–October, so the models are
fed summer weather in every season:

| Statewide median water temperature, °C | Jan | Apr | Jul | Oct |
|---|--:|--:|--:|--:|
| Observed | 0.6 | 10.5 | 24.5 | 14.4 |
| Dashboard prediction | 11.6 | 15.5 | 20.8 | 16.4 |

The January map is about **11 °C too warm** and the seasonal swing is flattened by
more than half. Three further gaps compound it: future dates are not forecasts
(trees cannot extrapolate the year at all; LR and the MLP extrapolate trends the
data does not support); every target is drawn at all 1,345 stations, including 190
wells, 318 lakes and 158 wetlands of types the model never trained on; and each
station gets a confident label ("Low", "High Concern") with no uncertainty, though
typical errors are as large as the thresholds themselves — the best E. coli model's
MAE is ~1,490 MPN/100 mL against a 235 MPN/100 mL recreational threshold.

**Use the dashboard to demonstrate the pipeline and compare model families. Do not
use it to decide anything about a real water body.**

---

## What would move the results most

In rough order of expected value:

1. **Give the models the previous observation at the same station** (`y_prev`,
   `days_since_prev`). Persistence alone already beats three targets; handing it
   over as a feature is the single largest available gain.
2. **Model the station level explicitly** (hierarchical / station-effect models,
   or persistence + anomaly) instead of letting a tree memorise it from
   coordinates. Under that framing, streamflow — currently masked — should move up.
3. **Fix the dashboard's feature assembly** so predictions use weather for the
   date being predicted, or state plainly that they do not.
4. **Treat Nitrite and Nitrate as hurdle problems** (detect / not-detect, then
   magnitude) rather than continuous regressions.
5. **Reduce the feature set** — drop the redundant thermal and nutrient columns,
   and the dead snow and wind columns — and add `np_years_stale` so feature age is
   visible to the model.
6. **Name a decision.** "What should we expect at this station on this date" is a
   question, not a decision. "Should this beach post an advisory this weekend?"
   implies a different target, horizon, location set and success metric — and
   choosing one would settle several of the open trade-offs above.

---

*Supporting detail: `src/04_eda/eda-summary.md` (data structure, correlation and
importance analysis), `src/05_modeling/model_outcomes.md` (per-family results),
`src/05_modeling/model_metrics.csv` (authoritative scores), `README.md` (current
status and limitations), `NEXT_STEPS.md` (proposed work), `DATA.md` and `MERGE.md`
(data dictionary and merge plan).*
