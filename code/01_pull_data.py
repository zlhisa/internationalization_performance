"""
01_pull_data.py
---------------
Pull ALL Compustat Global annual variables from WRDS.
Fiscal years 2015–2024, all geographies, all firms.

Output
------
data/raw/<YYYY-MM-DD_HH-MM-SS>/
    fyear_2015.parquet
    fyear_2016.parquet
    ...
    fyear_2024.parquet
    column_schema.csv
    pull_metadata.txt

Each run creates a new timestamped folder — every pull is preserved.
02_clean.py automatically picks the most recent folder.

Key lessons from live demo
--------------------------
- Compustat Global uses datafmt = 'HIST_STD' (not 'STD' like North America)
- Credentials loaded from .env via find_env() — works from any working directory
- from datetime import datetime (not import datetime) to avoid AttributeError
- Relative paths only — never hardcode C:/Users/...

Usage
-----
    python code/01_pull_data.py
    task pull
"""

import os
import sys
from datetime import datetime          # ← must be 'from datetime import datetime'
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv


# ── Find project root by searching upward AND in sibling folders ──────────────
def find_env():
    """Find .env by walking up the directory tree and checking siblings."""
    current = Path(os.getcwd())
    for path in [current] + list(current.parents):
        # Check current level
        if (path / ".env").exists():
            return path / ".env"
        # Check all sibling folders at this level
        try:
            for sibling in path.iterdir():
                if sibling.is_dir() and (sibling / ".env").exists():
                    return sibling / ".env"
        except PermissionError:
            continue
    raise FileNotFoundError(
        "Could not find .env in any parent or sibling directory.\n"
        "Create .env with WRDS_USERNAME=your_username in the project root."
    )


env_file     = find_env()
project_root = env_file.parent
os.chdir(project_root)
print(f"Project root: {project_root}")

# ── Credentials ───────────────────────────────────────────────────────────────
load_dotenv(dotenv_path=env_file, override=True)
WRDS_USER = os.getenv("WRDS_USERNAME")
if not WRDS_USER:
    print("ERROR: WRDS_USERNAME not set in .env")
    sys.exit(1)
print(f"WRDS user:    {WRDS_USER}")

# ── Configuration ─────────────────────────────────────────────────────────────
START_YEAR = 2015
END_YEAR   = 2024

# ── Output folder — new timestamp every run ────────────────────────────────────
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT_DIR   = Path("data") / "raw" / timestamp
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Output folder: {OUT_DIR}\n")

# ── Connect ───────────────────────────────────────────────────────────────────
print("Connecting to WRDS...")
try:
    db = wrds.Connection(wrds_username=WRDS_USER)
except Exception as e:
    print(f"ERROR: Could not connect to WRDS.\n{e}")
    sys.exit(1)

# ── Discover columns ──────────────────────────────────────────────────────────
print("Fetching column schema...")
try:
    schema = db.describe_table(library="comp_global_daily", table="g_funda")
    print(f"  → {len(schema)} columns available")
    schema.to_csv(OUT_DIR / "column_schema.csv", index=False)
    print(f"  → Schema saved to {OUT_DIR / 'column_schema.csv'}")
except Exception as e:
    print(f"  Warning: could not fetch schema ({e}). Continuing.")

# ── Pull by fiscal year ───────────────────────────────────────────────────────
# Compustat Global uses datafmt = 'HIST_STD' (not 'STD' like North America)
# Chunking by year keeps files manageable and allows resuming if interrupted.

total_rows = 0
years_done = []

print(f"\nPulling fiscal years {START_YEAR}–{END_YEAR}...")

for year in range(START_YEAR, END_YEAR + 1):
    print(f"  fyear = {year}", end=" ... ", flush=True)

    query = f"""
        SELECT *
        FROM comp_global_daily.g_funda
        WHERE fyear   = {year}
          AND datafmt = 'HIST_STD'
          AND indfmt  = 'INDL'
          AND popsrc  = 'I'
          AND consol  = 'C'
    """

    try:
        df = db.raw_sql(query, date_cols=["datadate"])
    except Exception as e:
        print(f"FAILED ({e})")
        continue

    if df.empty:
        print("no data — skipping")
        continue

    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]

    out_path = OUT_DIR / f"fyear_{year}.parquet"
    df.to_parquet(out_path, index=False)

    rows  = len(df)
    firms = df["gvkey"].nunique() if "gvkey" in df.columns else "?"
    total_rows += rows
    years_done.append(year)
    print(f"{rows:>8,} rows | {firms:>6,} firms → {out_path.name}")

db.close()

# ── Write metadata ─────────────────────────────────────────────────────────────
meta_path = OUT_DIR / "pull_metadata.txt"
meta_path.write_text(
    f"WRDS Compustat Global Pull\n"
    f"==========================\n"
    f"Pulled:       {datetime.now().isoformat()}\n"
    f"User:         {WRDS_USER}\n"
    f"Source:       comp_global_daily.g_funda\n"
    f"Filters:      datafmt=HIST_STD, indfmt=INDL, popsrc=I, consol=C\n"
    f"Fiscal years: {START_YEAR}–{END_YEAR}\n"
    f"Geography:    All (no filter)\n"
    f"Years pulled: {years_done}\n"
    f"Total rows:   {total_rows:,}\n"
    f"Format:       Parquet (one file per fiscal year)\n"
    f"License:      WRDS subscriber agreement\n"
    f"\nNote: Raw data is NOT committed to Git (see .gitignore).\n"
    f"      To reproduce: run this script with valid WRDS credentials.\n"
)

print(f"\n{'='*55}")
print(f"Pull complete.")
print(f"  Years:      {years_done}")
print(f"  Total rows: {total_rows:,}")
print(f"  Folder:     {OUT_DIR}")
print(f"\nNext step: python code/02_clean.py")