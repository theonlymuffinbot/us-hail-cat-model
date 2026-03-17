#!/usr/bin/env python3
"""
Spatial Neighbourhood Beta — County-Level Population Normalisation
==================================================================
For each county, pools storm + population data from the county AND
all immediately contiguous neighbours, fits a local regression:

    log(storms + 1) ~ beta*log(pop)

This gives each county a locally-calibrated beta that reflects the
population-density context of its neighbourhood (urban cores get
a higher beta; sparse rural counties get a lower one).

All three normalisation methods are then applied per county:

  A) rate_per_100k    = raw / pop * 100,000
  B) adj_count        = raw * (pop_ref / pop)          (linear, beta=1)
  C) spatial_adj      = raw * (pop_ref / pop)^beta_local  (neighbourhood beta)

Inputs:
  data/storms/county_storm_counts.csv        (from 04_build_storm_trends.py)
  data/population/county_population_trend.csv

Output:
  data/storms/county_storm_spatial.csv
      geoid, county_name, state_name, year,
      hail, wind, total, trend_pop,
      n_neighbours, local_beta,
      rate_per_100k, adj_count, spatial_adj

  data/storms/county_beta_map.csv
      geoid, county_name, state_name, n_neighbours, local_beta
      (one row per county — useful for mapping beta variation)
"""

import os, csv, math, time, urllib.request, collections
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

STORMS    = DATA_ROOT / "storms" / "county_storm_counts.csv"
POP_FILE  = DATA_ROOT / "population" / "county_population_trend.csv"
OUT_DIR   = DATA_ROOT / "storms"
OUT_MAIN  = OUT_DIR / "county_storm_spatial.csv"
OUT_BETA  = OUT_DIR / "county_beta_map.csv"
LOG_FILE  = LOGS_ROOT / "spatial_beta_build.log"

ADJ_URL   = "https://www2.census.gov/geo/docs/reference/county_adjacency.txt"
CACHE     = DATA_ROOT / "population" / "raw_cache" / "county_adjacency.txt"

YEARS     = range(2004, 2024)
REF_YEAR  = 2004

def log(msg):
    print(msg, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ── download / load adjacency file ────────────────────────────────────────────
def load_adjacency() -> dict:
    """Returns {geoid: [neighbour_geoid, ...]} — includes self."""
    if CACHE.exists():
        log(f"  (cached) county_adjacency.txt")
        raw = CACHE.read_bytes()
    else:
        log("  Downloading county_adjacency.txt …")
        req = urllib.request.Request(ADJ_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(raw)
        log(f"  county_adjacency.txt ({len(raw)//1024} KB)")

    adj = collections.defaultdict(set)
    current = None
    for line in raw.decode("latin-1").splitlines():
        parts = line.split("\t")
        if parts[0].strip():
            # New focal county — col[1] is its FIPS
            if len(parts) >= 2:
                current = parts[1].strip().zfill(5)
                adj[current].add(current)      # always include self
        else:
            # Neighbour row — col[3] is neighbour FIPS
            if current and len(parts) >= 4:
                nbr = parts[3].strip().zfill(5)
                if nbr:
                    adj[current].add(nbr)

    log(f"  Adjacency: {len(adj):,} counties, avg {sum(len(v) for v in adj.values())/len(adj):.1f} neighbours each")
    return dict(adj)

# ── load storm counts ─────────────────────────────────────────────────────────
def load_storm_counts():
    """Returns counts[geoid][year] = total, meta[geoid] = (name, state)"""
    counts = collections.defaultdict(lambda: collections.defaultdict(int))
    meta   = {}
    with open(STORMS) as f:
        for row in csv.DictReader(f):
            g  = row["geoid"]
            yr = int(row["year"])
            counts[g][yr] = int(row["total"])
            if g not in meta:
                meta[g] = (row["county_name"], row["state_name"])
    return counts, meta

# ── load population ───────────────────────────────────────────────────────────
def load_population():
    """Returns pop[geoid][year] = trend_pop"""
    pop = collections.defaultdict(dict)
    with open(POP_FILE) as f:
        for row in csv.DictReader(f):
            yr = int(row["year"])
            if yr in YEARS:
                pop[row["geoid"]][yr] = int(row["trend_pop"])
    return pop

# ── local beta regression ────────────────────────────────────────────────────
def fit_local_beta(neighbourhood: set, counts: dict, pop: dict) -> float:
    """
    Pool all county-year observations in the neighbourhood,
    fit log(storms+1) ~ log(pop), return beta.
    Falls back to beta=1.0 if insufficient data.
    """
    log_s, log_p = [], []
    for g in neighbourhood:
        for yr in YEARS:
            s = counts.get(g, {}).get(yr, 0)
            p = pop.get(g, {}).get(yr, 0)
            if p > 0:
                log_s.append(math.log(s + 1))
                log_p.append(math.log(p))

    if len(log_s) < 10:
        return 1.0   # not enough data — default to linear

    lp = np.array(log_p)
    ls = np.array(log_s)
    X  = np.column_stack([np.ones_like(lp), lp])
    coef, _, _, _ = np.linalg.lstsq(X, ls, rcond=None)
    beta = float(coef[1])
    # Clamp to a sensible range
    return max(0.1, min(beta, 5.0))

def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    import csv as _csv
    import sys
    errors = []

    for p in [OUT_BETA, OUT_MAIN]:
        if not p.exists():
            errors.append(f"Missing: {p.name}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty: {p.name}")
        else:
            try:
                with open(p, newline="") as f:
                    rows = list(_csv.DictReader(f))
                if len(rows) <= 100:
                    errors.append(f"Too few rows in {p.name}: {len(rows)} (expected >100)")
            except Exception as e:
                errors.append(f"Cannot read {p.name}: {e}")

    if errors:
        log("CRITICAL: Output validation FAILED:")
        for e in errors:
            log(f"  ✗ {e}")
        return False
    log("Output validation passed ✓")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Building spatial beta normalisations")

    log("\n[1/5] Loading adjacency")
    adj = load_adjacency()

    log("\n[2/5] Loading storm counts")
    counts, meta = load_storm_counts()

    log("\n[3/5] Loading population trend")
    pop = load_population()

    log("\n[4/5] Computing local beta for each county")
    all_geoids = sorted(set(pop.keys()))
    beta_map   = {}   # geoid -> local_beta
    nbr_count  = {}   # geoid -> n_neighbours

    for geoid in all_geoids:
        neighbourhood = adj.get(geoid, {geoid})
        beta_map[geoid]  = fit_local_beta(neighbourhood, counts, pop)
        nbr_count[geoid] = len(neighbourhood)

    betas = list(beta_map.values())
    log(f"  beta range: {min(betas):.3f} – {max(betas):.3f}")
    log(f"  beta median: {sorted(betas)[len(betas)//2]:.3f}")
    log(f"  beta mean:   {sum(betas)/len(betas):.3f}")

    log("\n[5/5] Writing outputs")

    # county_storm_spatial.csv
    main_rows = []
    for geoid in all_geoids:
        cname, sname = meta.get(geoid, ("", ""))
        beta    = beta_map[geoid]
        n_nbrs  = nbr_count[geoid]
        p_ref   = pop[geoid].get(REF_YEAR, 0)

        for yr in YEARS:
            # raw counts
            total = counts.get(geoid, {}).get(yr, 0)
            hail  = 0   # not in storm counts by type here — load separately if needed
            wind  = 0
            p     = pop[geoid].get(yr, 0)

            if p > 0 and p_ref > 0:
                rate      = total / p * 100_000
                adj_b     = total * (p_ref / p)                  # beta=1
                spatial   = total * (p_ref / p) ** beta          # local beta
            else:
                rate = adj_b = spatial = 0.0

            main_rows.append({
                "geoid":        geoid,
                "county_name":  cname,
                "state_name":   sname,
                "year":         yr,
                "total":        total,
                "trend_pop":    p,
                "n_neighbours": n_nbrs,
                "local_beta":   round(beta, 4),
                "rate_per_100k":  round(rate,     4),
                "adj_count":      round(adj_b,    2),
                "spatial_adj":    round(spatial,  2),
            })

    fields_main = ["geoid","county_name","state_name","year",
                   "total","trend_pop","n_neighbours","local_beta",
                   "rate_per_100k","adj_count","spatial_adj"]
    with open(OUT_MAIN, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields_main)
        w.writeheader(); w.writerows(main_rows)
    log(f"  county_storm_spatial.csv  ({len(main_rows):,} rows)")

    # county_beta_map.csv
    beta_rows = []
    for geoid in all_geoids:
        cname, sname = meta.get(geoid, ("", ""))
        beta_rows.append({
            "geoid":       geoid,
            "county_name": cname,
            "state_name":  sname,
            "n_neighbours": nbr_count[geoid],
            "local_beta":   round(beta_map[geoid], 4),
        })
    with open(OUT_BETA, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["geoid","county_name","state_name",
                                          "n_neighbours","local_beta"])
        w.writeheader(); w.writerows(beta_rows)
    log(f"  county_beta_map.csv  ({len(beta_rows):,} rows)")

    # Print a few interesting beta examples
    log("\n── Sample local beta values ──────────────────────────────────────────────")
    examples = [
        ("36061", "New York County (Manhattan), NY"),
        ("17031", "Cook County (Chicago), IL"),
        ("48113", "Dallas County, TX"),
        ("26163", "Wayne County (Detroit), MI"),
        ("30003", "Big Horn County, MT  (rural)"),
        ("38037", "Grant County, ND  (rural)"),
        ("48301", "Loving County, TX  (least populous)"),
    ]
    for geoid, label in examples:
        b = beta_map.get(geoid, "N/A")
        n = nbr_count.get(geoid, 0)
        log(f"  {label}: beta={b}  (neighbourhood size={n})")

    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done!")

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
