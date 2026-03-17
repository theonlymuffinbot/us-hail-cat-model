#!/usr/bin/env python3
"""
Hail Raster Population Debiasing
=================================
Reads each raw hail GeoTIFF from data/hail/
and writes a population-debiased version to data/hail_0.05deg_pop_debias/

Method
------
For each grid cell:
  1. Cell is assigned to its nearest county centroid (FIPS)
  2. Per-county correction factor for year Y:
       correction = (pop_2004 / pop_Y) ^ beta_local
     where beta_local comes from the neighbourhood spatial regression
     (county_beta_map.csv) and pop from county_population_trend.csv
  3. debiased_count = raw_count x correction

  beta < 1 for nearly all counties, so corrections are gentle.
  Output dtype: float32 (counts may be fractional after correction).

Reference year: 2004 (first year of SPC data — both lines start together).

Inputs
------
  data/hail/YYYY/hail_YYYYMMDD.tif   (raw rasters)
  data/storms/county_beta_map.csv
  data/population/county_population_trend.csv
  Census county centroids (auto-downloaded)

Output
------
  data/hail_0.05deg_pop_debias/YYYY/hail_YYYYMMDD.tif
  Same grid, same 29 bands, float32, LZW compressed.
"""

import os, csv, time, urllib.request
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

# ── paths ─────────────────────────────────────────────────────────────────────
RAW_DIR   = DATA_ROOT / "hail"
OUT_DIR   = DATA_ROOT / "hail_0.05deg_pop_debias"
LOG_FILE  = LOGS_ROOT / "hail_debias_build.log"
CACHE_DIR = DATA_ROOT / "population" / "raw_cache"

BETA_FILE = DATA_ROOT / "storms" / "county_beta_map.csv"
POP_FILE  = DATA_ROOT / "population" / "county_population_trend.csv"
CENT_URL  = "https://www2.census.gov/geo/docs/reference/cenpop2020/county/CenPop2020_Mean_CO.txt"
CENT_CACHE= CACHE_DIR / "CenPop2020_Mean_CO.txt"

REF_YEAR  = 2004

# Grid definition — must match 06_build_hail_rasters.py exactly
LON_MIN, LON_MAX = -125.0,  -66.0
LAT_MIN, LAT_MAX =   24.0,   50.0
DX = DY = 0.05
NCOLS = int(round((LON_MAX - LON_MIN) / DX))   # 1180
NROWS = int(round((LAT_MAX - LAT_MIN) / DY))   # 520
TRANSFORM = from_origin(LON_MIN, LAT_MAX, DX, DY)
CRS = "EPSG:4326"

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)

# ── logging ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

# ── load county centroids ─────────────────────────────────────────────────────
def load_centroids():
    """Returns list of (geoid, lat, lon)"""
    if not CENT_CACHE.exists():
        log("  Downloading county centroids ...")
        req = urllib.request.Request(CENT_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            data = r.read()
        CENT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        CENT_CACHE.write_bytes(data)
        log(f"  centroids ({len(data)//1024} KB)")
    centroids = []
    with open(CENT_CACHE, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            geoid = row["STATEFP"].zfill(2) + row["COUNTYFP"].zfill(3)
            centroids.append((geoid, float(row["LATITUDE"]), float(row["LONGITUDE"])))
    log(f"  Centroids loaded: {len(centroids):,}")
    return centroids

# ── build county FIPS grid ────────────────────────────────────────────────────
def build_county_grid(centroids):
    """
    Returns county_grid[row, col] = index into centroids list (nearest centroid).
    Uses vectorised nearest-neighbour over the grid.
    """
    log("  Building county assignment grid (nearest centroid) ...")

    # Grid cell centre coordinates
    lons = LON_MIN + (np.arange(NCOLS) + 0.5) * DX          # (NCOLS,)
    lats = LAT_MAX - (np.arange(NROWS) + 0.5) * DY          # (NROWS,)
    grid_lon, grid_lat = np.meshgrid(lons, lats)             # (NROWS, NCOLS)

    c_lats = np.array([c[1] for c in centroids])             # (N,)
    c_lons = np.array([c[2] for c in centroids])             # (N,)

    # Process in row batches to keep memory reasonable
    county_idx = np.empty((NROWS, NCOLS), dtype=np.int16)
    batch = 50
    for r0 in range(0, NROWS, batch):
        r1 = min(r0 + batch, NROWS)
        gl = grid_lat[r0:r1, :, np.newaxis]   # (batch, NCOLS, 1)
        gn = grid_lon[r0:r1, :, np.newaxis]
        # Approximate distance^2 (degrees) — good enough for nearest-centroid
        d2 = (gl - c_lats)**2 + (gn - c_lons)**2
        county_idx[r0:r1, :] = np.argmin(d2, axis=2).astype(np.int16)

    log(f"  County grid built: {NROWS}x{NCOLS}")
    return county_idx   # values are indices into `centroids` list

# ── load beta map ────────────────────────────────────────────────────────────
def load_beta(centroids):
    """Returns beta[centroid_index] as float32 array."""
    geoid_to_idx = {c[0]: i for i, c in enumerate(centroids)}
    beta_by_geoid = {}
    with open(BETA_FILE) as f:
        for row in csv.DictReader(f):
            beta_by_geoid[row["geoid"]] = float(row["local_beta"])
    beta_arr = np.ones(len(centroids), dtype=np.float32)
    for geoid, b in beta_by_geoid.items():
        idx = geoid_to_idx.get(geoid)
        if idx is not None:
            beta_arr[idx] = b
    log(f"  beta loaded: {len(beta_by_geoid):,} counties")
    return beta_arr

# ── load population 2004 + per year ──────────────────────────────────────────
def load_population(centroids):
    """
    Returns:
      pop_ref[centroid_index]   = trend_pop at REF_YEAR
      pop_by_year[year][centroid_index] = trend_pop
    """
    geoid_to_idx = {c[0]: i for i, c in enumerate(centroids)}
    n = len(centroids)
    pop_ref  = np.ones(n, dtype=np.float32)
    pop_year = {}

    with open(POP_FILE) as f:
        for row in csv.DictReader(f):
            yr  = int(row["year"])
            g   = row["geoid"]
            idx = geoid_to_idx.get(g)
            if idx is None:
                continue
            p = max(int(row["trend_pop"]), 1)
            if yr == REF_YEAR:
                pop_ref[idx] = p
            if yr not in pop_year:
                pop_year[yr] = np.ones(n, dtype=np.float32)
            pop_year[yr][idx] = p

    log(f"  Population loaded: {len(pop_year)} years")
    return pop_ref, pop_year

# ── build per-year correction raster ─────────────────────────────────────────
def build_correction_raster(year, county_idx, pop_ref, pop_year, beta_arr):
    """
    Returns correction[NROWS, NCOLS] float32.
    correction = (pop_ref / pop_year) ^ beta
    """
    if year not in pop_year:
        return np.ones((NROWS, NCOLS), dtype=np.float32)

    # Per-centroid correction
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(pop_year[year] > 0,
                         pop_ref / pop_year[year], 1.0).astype(np.float32)
    correction_per_county = ratio ** beta_arr    # (N,)

    # Map to grid
    return correction_per_county[county_idx]     # (NROWS, NCOLS)

# ── process one tif ───────────────────────────────────────────────────────────
def process_tif(src_path, dst_path, correction):
    """Read raw raster, apply correction, write float32 GeoTIFF."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(src_path) as src:
        n_bands = src.count
        profile = src.profile.copy()

    profile.update(dtype=np.float32, compress="lzw")

    with rasterio.open(src_path) as src, \
         rasterio.open(dst_path, "w", **profile) as dst:
        for b in range(1, n_bands + 1):
            raw = src.read(b).astype(np.float32)
            dst.write(raw * correction, b)
            tags = src.tags(b)
            if tags:
                dst.update_tags(b, **tags)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hail population debiasing started")
    log(f"  Source:  {RAW_DIR}")
    log(f"  Output:  {OUT_DIR}")
    log(f"  Ref year: {REF_YEAR}")

    log("\n[1/5] Loading county centroids")
    centroids = load_centroids()

    log("\n[2/5] Building county assignment grid")
    county_idx = build_county_grid(centroids)

    log("\n[3/5] Loading beta map")
    beta_arr = load_beta(centroids)

    log("\n[4/5] Loading population trend")
    pop_ref, pop_year = load_population(centroids)

    log("\n[5/5] Processing rasters")

    # Pre-build correction rasters per year (fast — one per year, not per day)
    years = sorted(pop_year.keys())
    correction_cache = {}
    for yr in years:
        correction_cache[yr] = build_correction_raster(
            yr, county_idx, pop_ref, pop_year, beta_arr)
    log(f"  Correction rasters built for {len(correction_cache)} years")

    # Find all raw tifs and process
    raw_tifs = sorted(RAW_DIR.rglob("hail_????????.tif"))
    log(f"  Raw rasters found: {len(raw_tifs):,}")

    done = skipped = errors = 0
    for src_path in raw_tifs:
        # Parse year from filename: hail_YYYYMMDD.tif
        try:
            year = int(src_path.stem[5:9])   # hail_YYYY...
        except ValueError:
            continue

        rel  = src_path.relative_to(RAW_DIR)
        dst_path = OUT_DIR / rel

        if dst_path.exists():
            skipped += 1
            continue

        correction = correction_cache.get(year,
                         np.ones((NROWS, NCOLS), dtype=np.float32))
        try:
            process_tif(src_path, dst_path, correction)
            done += 1
            if done % 500 == 0 or done == 1:
                log(f"  [{src_path.stem}] done={done:,}  skipped={skipped:,}")
        except Exception as e:
            errors += 1
            log(f"  ERROR {src_path.name}: {e}")

    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done!")
    log(f"  Written:  {done:,}")
    log(f"  Skipped:  {skipped:,}")
    log(f"  Errors:   {errors:,}")

if __name__ == "__main__":
    main()
