"""
utils/data.py
=============
CSV-based data layer for InfraSight AI.

All pages share the same logfile and call these helpers so that
the schema is defined in exactly one place.
"""

from __future__ import annotations
from pathlib import Path
import pandas as pd



# SCHEMA
LOG_COLUMNS = [
    "lat",          # float  — GPS latitude
    "lon",          # float  — GPS longitude
    "status",       # str    — "serviceable" | "faulty"
    "time",         # str    — "YYYY/MM/DD HH:MM:SS"
    "lighting",     # str    — "Daylight" | "Twilight" | "Night"
    "light_label",  # str    — "Light_On" | "Light_Off" | "Unknown"
    "confidence",   # float  — model confidence [0, 1]
    "fault",        # str    — fault flag string or "✅ Normal"
]

_PROJECT_DIR = Path(__file__).resolve().parent.parent
LOG_PATH     = _PROJECT_DIR / "inspection_logs.csv"



# INITIALISE
def init_logfile(path: Path = LOG_PATH) -> None:
    """Create the CSV with the correct header if it does not yet exist."""
    if not path.exists():
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(path, index=False)


# LOAD
def load_logs(path: Path = LOG_PATH) -> pd.DataFrame:
    """
    Read the inspection log CSV.

    • Adds any missing columns (backward-compatible with old CSVs from InitialClassifier).
    • Parses the time column as datetime.
    • Drops rows where time could not be parsed.
    """
    init_logfile(path)
    df = pd.read_csv(path)

    # Back-fill columns added after initial deployment
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = "Unknown" if df.dtypes.get(col, "O") == object else 0.0

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    return df



# APPEND
def append_entries(entries: list[dict], path: Path = LOG_PATH) -> pd.DataFrame:
    """
    Append a list of entry dicts to the CSV and return the updated DataFrame.

    Each dict should contain the keys in LOG_COLUMNS.
    Missing keys are filled with sensible defaults.
    """
    if not entries:
        return load_logs(path)

    defaults = {
        "lat"        : 0.0,
        "lon"        : 0.0,
        "status"     : "Unknown",
        "time"       : "",
        "lighting"   : "Unknown",
        "light_label": "Unknown",
        "confidence" : 0.0,
        "fault"      : "Unknown",
    }

    rows = []
    for entry in entries:
        row = {**defaults, **{k: entry.get(k, defaults[k]) for k in LOG_COLUMNS}}
        # Ensure time is a plain string for CSV storage
        if hasattr(row["time"], "strftime"):
            row["time"] = row["time"].strftime("%d/%m/%Y %H:%M:%S")
        rows.append(row)

    df_new     = pd.DataFrame(rows, columns=LOG_COLUMNS)
    existing   = load_logs(path)

    # Convert existing time back to string for concat
    existing["time"] = existing["time"].dt.strftime("%d/%m/%Y %H:%M:%S")

    updated = pd.concat([existing, df_new], ignore_index=True)
    updated.to_csv(path, index=False)
    return load_logs(path) 
