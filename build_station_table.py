"""Precompute the deduplicated station table the dashboard runs on.

`app.py` never needs the full modeling table. It reads `epa-full.csv`
(48,251 rows x 318 columns, 81 MB) only to collapse it to one representative
row per monitoring station -- 1,345 rows, 2.7 MB. Everything else in that file
is training data the app does not touch.

That distinction matters for deployment. `data/` is gitignored, so the 81 MB
CSV never reaches the host and the dashboard cannot boot there. Committing the
collapsed table instead ships what the app actually uses, at 3% of the size,
and skips parsing 48k rows on every cold start.

Run this after any change to `epa-full.csv` (notably after
`src/04_eda/wqi-calculation.ipynb` rewrites it):

    python3 build_station_table.py

The collapse must stay identical to what `app.py::load_station_data` used to do
inline -- sort by observation time, then `groupby.last()`, which takes the last
*non-null* value per column and so fills a station's gaps from its own history
rather than trusting one row. `app.py` prefers this file when it exists and
falls back to recomputing from `epa-full.csv` when it does not, so a stale or
absent table degrades startup time, never correctness.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
SOURCE_PATH = BASE_DIR / "data/final/epa-full.csv"
OUTPUT_PATH = BASE_DIR / "data/stations.csv"

GROUP_COL = "MonitoringLocationIdentifier"
DATE_COL = "ActivityStartDateTime"


def build_station_table(source: Path = SOURCE_PATH) -> pd.DataFrame:
    """Collapse the modeling table to one representative row per station."""
    if not source.exists():
        raise FileNotFoundError(
            f"Could not find {source}. This script needs the full modeling "
            "table, which is gitignored — run the merge pipeline first."
        )

    df = pd.read_csv(source, parse_dates=[DATE_COL], low_memory=False)
    df = df.sort_values(DATE_COL)

    stations = (
        df.groupby(GROUP_COL, sort=False)
          .last()
          .reset_index()
    )
    return stations


def main() -> None:
    stations = build_station_table()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    stations.to_csv(OUTPUT_PATH, index=False)

    size_mb = OUTPUT_PATH.stat().st_size / 1e6
    print(
        f"Wrote {len(stations):,} stations x {stations.shape[1]} columns "
        f"-> {OUTPUT_PATH.relative_to(BASE_DIR)} ({size_mb:.1f} MB)"
    )


if __name__ == "__main__":
    main()
