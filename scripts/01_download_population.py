#!/usr/bin/env python3
"""
Census Population Estimates Program (PEP) Downloader
Builds a single long-format CSV of annual county population estimates.

Sources (all US Census Bureau):
  - 2000–2009: Intercensal estimates   co-est00int-tot.csv
  - 2010–2019: Vintage 2020            co-est2020-alldata.csv
  - 2020–2023: Vintage 2023            co-est2023-alldata.csv

Output: data/population/county_population.csv
  GEOID, county_name, state_name, year, population

GEOID is the 5-digit FIPS code (2-digit state + 3-digit county).
Use this to join against SPC storm data for per-capita normalization.
"""

import os
import csv
import time
import io
import urllib.request
from urllib.error import HTTPError
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

OUT_DIR  = DATA_ROOT / "population"
OUT_FILE = OUT_DIR / "county_population.csv"
LOG_FILE = LOGS_ROOT / "population_download.log"

SOURCES = [
    {
        "label": "2000–2009 intercensal",
        "url":   "https://www2.census.gov/programs-surveys/popest/datasets/2000-2010/intercensal/county/co-est00int-tot.csv",
        "years": list(range(2000, 2010)),  # skip 2010 — taken from next vintage
        "encoding": "latin-1",
    },
    {
        "label": "2010–2019 vintage 2020",
        "url":   "https://www2.census.gov/programs-surveys/popest/datasets/2010-2020/counties/totals/co-est2020-alldata.csv",
        "years": list(range(2010, 2020)),  # skip 2020 — taken from next vintage
        "encoding": "latin-1",
    },
    {
        "label": "2020–2023 vintage 2023",
        "url":   "https://www2.census.gov/programs-surveys/popest/datasets/2020-2023/counties/totals/co-est2023-alldata.csv",
        "years": list(range(2020, 2024)),
        "encoding": "latin-1",
    },
]

def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def fetch_csv(url, encoding="latin-1"):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research/census-pop)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read()
    return list(csv.DictReader(io.StringIO(raw.decode(encoding))))

def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    import csv as _csv
    import sys
    errors = []

    if not OUT_FILE.exists():
        errors.append(f"Missing: {OUT_FILE}")
    elif OUT_FILE.stat().st_size == 0:
        errors.append(f"Empty: {OUT_FILE}")
    else:
        try:
            with open(OUT_FILE, newline="") as f:
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting population download")

    all_rows = []

    for source in SOURCES:
        log(f"\n  Fetching {source['label']} …")
        try:
            rows = fetch_csv(source["url"], source["encoding"])
        except HTTPError as e:
            log(f"  HTTP {e.code}: {source['url']}")
            continue
        except Exception as e:
            log(f"  Error: {e}")
            continue

        count = 0
        for row in rows:
            # SUMLEV 50 = county; skip state totals (SUMLEV 40) and national (SUMLEV 10)
            # Some vintages zero-pad as "050", others use "50" — normalise
            if row.get("SUMLEV", "").strip().lstrip("0") != "50":
                continue
            # Skip COUNTY == "000" (state-level summary rows that slip through)
            if row.get("COUNTY", "").strip() == "000":
                continue

            state_fips  = row["STATE"].strip().zfill(2)
            county_fips = row["COUNTY"].strip().zfill(3)
            geoid       = state_fips + county_fips
            county_name = row["CTYNAME"].strip()
            state_name  = row["STNAME"].strip()

            for year in source["years"]:
                col = f"POPESTIMATE{year}"
                if col not in row:
                    continue
                val = row[col].strip().replace(",", "")
                if not val or not val.lstrip("-").isdigit():
                    continue
                pop = int(val)
                if pop <= 0:
                    continue
                all_rows.append({
                    "geoid":       geoid,
                    "county_name": county_name,
                    "state_name":  state_name,
                    "year":        year,
                    "population":  pop,
                })
                count += 1

        log(f"  {source['label']}: {count:,} county-year records")

    if not all_rows:
        log("\nNo data collected — aborting.")
        return

    # Sort for readability
    all_rows.sort(key=lambda r: (r["geoid"], r["year"]))

    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["geoid", "county_name", "state_name", "year", "population"])
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    geoids = {r["geoid"] for r in all_rows}
    years  = sorted({r["year"] for r in all_rows})
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done!")
    log(f"  Output:   {OUT_FILE}")
    log(f"  Records:  {len(all_rows):,}")
    log(f"  Counties: {len(geoids):,}")
    log(f"  Years:    {years[0]}–{years[-1]}")
    log(f"\nUsage:")
    log(f"  Join SPC storm data on FIPS (5-digit) + year.")
    log(f"  Normalize: storms_per_100k = storm_count / population * 100_000")

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
