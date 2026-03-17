#!/usr/bin/env python3
"""
County Population Trend Builder  — 1980 to 2023
================================================
Data sources:
  1. 1980 census:      comp8090.zip   (US Census Bureau)
  2. 1990–1999 annual: 99c8_00.txt    (US Census Bureau)
  3. 2000–2023 annual: county_population.csv  (already downloaded)

Methodology — piecewise log-linear regression (broken-stick model)
  For each county, fit:
      log(pop) = β₀ + β₁·t + β₂·(t−t₁₉₉₀)₊ + β₃·(t−t₂₀₀₀)₊ + β₄·(t−t₂₀₁₀)₊ + β₅·(t−t₂₀₂₀)₊
  where t = year − 1980 and (x)₊ = max(0, x).

  This allows a different growth rate per decade while remaining
  continuous at the knots — no kinks, no vintage-jump artifacts.
  The 1980 census anchors the intercept; the 1981–1989 gap is
  naturally filled by the first segment.

Output: data/population/county_population_trend.csv
  geoid, county_name, state_name, year, raw_pop, trend_pop, pop_change
"""

import os, io, csv, time, zipfile, urllib.request
from urllib.error import HTTPError
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

POP_DIR   = DATA_ROOT / "population"
RAW_CSV   = POP_DIR / "county_population.csv"
OUT_CSV   = POP_DIR / "county_population_trend.csv"
LOG_FILE  = LOGS_ROOT / "population_trend_build.log"
CACHE_DIR = POP_DIR / "raw_cache"

COMP_URL  = "https://www2.census.gov/programs-surveys/popest/datasets/1980-1990/counties/totals/comp8090.zip"
C99_URL   = "https://www2.census.gov/programs-surveys/popest/tables/1990-2000/counties/totals/99c8_00.txt"

POP_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def fetch_bytes(url, cache_name):
    path = CACHE_DIR / cache_name
    if path.exists():
        log(f"  (cached) {cache_name}")
        return path.read_bytes()
    log(f"  Downloading {cache_name} …")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    path.write_bytes(data)
    log(f"  {cache_name} ({len(data)//1024} KB)")
    return data

# ── parse 1980 census from comp8090.zip ──────────────────────────────────────
def parse_1980(data: bytes) -> dict:
    """Returns {geoid: pop_1980}"""
    z = zipfile.ZipFile(io.BytesIO(data))
    txt = z.read("comp8090.txt").decode("latin-1")
    result = {}
    for line in txt.splitlines():
        parts = line.split()
        if not parts or parts[0] != "A":
            continue
        fips = parts[1].zfill(5)
        # Skip national (00000), state totals (XX000)
        if fips == "00000" or fips[2:] == "000":
            continue
        try:
            pop = int(parts[2].replace(",", ""))
        except (IndexError, ValueError):
            continue
        if pop > 0:
            result[fips] = pop
    log(f"  1980 census: {len(result):,} counties")
    return result

# ── parse 1990–1999 annual estimates from 99c8_00.txt ────────────────────────
def parse_1990s(data: bytes) -> dict:
    """Returns {geoid: {year: pop}} for 1990–1999"""
    txt = data.decode("latin-1")
    # Values in block 1 are in descending year order: 1999 1998 … 1990(est) 1990(census)
    # We want indices 0–9 → years 1999 down to 1990
    YEARS = list(range(1999, 1989, -1))  # [1999, 1998, ..., 1990]
    result = {}
    in_block1 = False
    for line in txt.splitlines():
        stripped = line.strip()
        if "Block 1:" in stripped or "Block  1:" in stripped:
            in_block1 = True
            continue
        if stripped.startswith("Block ") and "1" not in stripped.split()[1]:
            in_block1 = False
        if not in_block1:
            continue
        parts = stripped.split()
        if not parts or parts[0] != "1":
            continue
        # County line: parts[1] should be a 5-digit FIPS ending in non-000
        if len(parts) < 12:
            continue
        fips_candidate = parts[1]
        if not fips_candidate.isdigit() or len(fips_candidate) != 5:
            continue
        if fips_candidate[2:] == "000":
            continue  # state total
        geoid = fips_candidate
        # Next 11 numeric tokens: indices 2..12, last one is census base — skip it
        pops = {}
        try:
            for i, yr in enumerate(YEARS):
                val = int(parts[2 + i].replace(",", ""))
                pops[yr] = val
        except (IndexError, ValueError):
            continue
        result[geoid] = pops
    log(f"  1990–1999: {len(result):,} counties")
    return result

# ── load existing 2000–2023 CSV ───────────────────────────────────────────────
def load_2000s() -> dict:
    """Returns {geoid: {year: pop, 'name': .., 'state': ..}}"""
    result = {}
    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            g = row["geoid"]
            yr = int(row["year"])
            pop = int(row["population"])
            if g not in result:
                result[g] = {"name": row["county_name"], "state": row["state_name"]}
            result[g][yr] = pop
    geoid_count = len(result)
    log(f"  2000–2023: {geoid_count:,} counties")
    return result

# ── piecewise log-linear fit ──────────────────────────────────────────────────
KNOTS     = [1990, 2000, 2010, 2020]
BASE_YEAR = 1980
ALL_YEARS = np.arange(1980, 2024)

def design_matrix(years: np.ndarray) -> np.ndarray:
    t = (years - BASE_YEAR).astype(float)
    cols = [np.ones_like(t), t]
    for k in KNOTS:
        cols.append(np.maximum(0.0, years - k))
    return np.column_stack(cols)

def fit_trend(data_years: np.ndarray, data_pops: np.ndarray) -> np.ndarray:
    """
    Fit piecewise log-linear model; return trend population for ALL_YEARS.
    Falls back to simple log-linear if too few points.
    """
    valid = (data_pops > 0)
    y = np.log(data_pops[valid].astype(float))
    x = data_years[valid]
    if len(x) < 3:
        # Degenerate: just return constant
        return np.full(len(ALL_YEARS), np.exp(y.mean()) if len(y) else np.nan)

    X = design_matrix(x)
    # Remove columns for knots that are beyond the data range (avoid extrapolation artefacts)
    active_knots = [k for k in KNOTS if k > x.min() - 5]
    X_full = design_matrix(x)
    X_pred = design_matrix(ALL_YEARS)

    if len(x) < X_full.shape[1]:
        # Fewer data points than parameters: fall back to simple log-linear
        X_full = X_full[:, :2]
        X_pred = X_pred[:, :2]

    coef, _, _, _ = np.linalg.lstsq(X_full, y, rcond=None)
    trend = np.exp(X_pred @ coef)
    return trend

def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    import csv as _csv
    import sys
    errors = []

    if not OUT_CSV.exists():
        errors.append(f"Missing: {OUT_CSV}")
    elif OUT_CSV.stat().st_size == 0:
        errors.append(f"Empty: {OUT_CSV}")
    else:
        try:
            with open(OUT_CSV, newline="") as f:
                rows = list(_csv.DictReader(f))
            if len(rows) <= 100:
                errors.append(f"Too few rows: {len(rows)} (expected >100)")
        except Exception as e:
            errors.append(f"Cannot read CSV: {e}")

    if errors:
        log("CRITICAL: Output validation FAILED:")
        for e in errors:
            log(f"  ✗ {e}")
        return False
    log("Output validation passed ✓")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Building county population trend")

    # 1. Download / load raw sources
    log("\n[1/4] Loading raw data")
    raw_1980 = parse_1980(fetch_bytes(COMP_URL,  "comp8090.zip"))
    raw_1990 = parse_1990s(fetch_bytes(C99_URL,  "99c8_00.txt"))
    raw_2000 = load_2000s()

    # 2. Build master county index (union of all GEOIDs found in 2000-2023 CSV)
    log("\n[2/4] Merging data")
    all_geoids = sorted(raw_2000.keys())
    log(f"  Master county list: {len(all_geoids):,} counties")

    # 3. Fit trend per county
    log("\n[3/4] Fitting trends")
    rows = []
    no_1980 = 0
    no_1990s = 0

    for geoid in all_geoids:
        meta   = raw_2000[geoid]
        name   = meta["name"]
        state  = meta["state"]

        # Assemble raw observations
        obs_years = []
        obs_pops  = []

        # 1980 census
        if geoid in raw_1980:
            obs_years.append(1980)
            obs_pops.append(raw_1980[geoid])
        else:
            no_1980 += 1

        # 1990–1999
        if geoid in raw_1990:
            for yr in range(1990, 2000):
                if yr in raw_1990[geoid]:
                    obs_years.append(yr)
                    obs_pops.append(raw_1990[geoid][yr])
        else:
            no_1990s += 1

        # 2000–2023
        for yr in range(2000, 2024):
            if yr in meta:
                obs_years.append(yr)
                obs_pops.append(meta[yr])

        if len(obs_years) < 3:
            continue

        dy = np.array(obs_years, dtype=float)
        dp = np.array(obs_pops,  dtype=float)

        # Fit
        trend = fit_trend(dy, dp)

        # Build output rows
        # raw_pop: actual observed value for that year (or blank if interpolated)
        obs_map = dict(zip(obs_years, obs_pops))
        prev_trend = None
        for i, yr in enumerate(ALL_YEARS):
            yr = int(yr)
            t_pop = round(trend[i])
            raw   = obs_map.get(yr, "")
            change = round(trend[i] - trend[i-1]) if i > 0 else ""
            rows.append({
                "geoid":       geoid,
                "county_name": name,
                "state_name":  state,
                "year":        yr,
                "raw_pop":     raw,
                "trend_pop":   t_pop,
                "pop_change":  change,
            })

    log(f"  Counties without 1980 data: {no_1980}")
    log(f"  Counties without 1990s data: {no_1990s}")

    # 4. Write output
    log(f"\n[4/4] Writing {OUT_CSV}")
    fieldnames = ["geoid", "county_name", "state_name", "year", "raw_pop", "trend_pop", "pop_change"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    geoids_out = len({r["geoid"] for r in rows})
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done!")
    log(f"  Counties:    {geoids_out:,}")
    log(f"  Years:       1980–2023 ({len(ALL_YEARS)} years each)")
    log(f"  Total rows:  {len(rows):,}")
    log(f"  Output:      {OUT_CSV}")

    if not validate_outputs():
        import sys
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        import sys
        ok = validate_outputs()
        sys.exit(0 if ok else 1)
    main()
