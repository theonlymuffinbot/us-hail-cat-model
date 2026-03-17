#!/usr/bin/env python3
"""
Hail Raster Builder
===================
Creates one multi-band GeoTIFF per SPC storm day (12Z–12Z) over CONUS.

Grid:
  CRS:        WGS84 (EPSG:4326)
  Resolution: 0.05° × 0.05°  (~5.5 km at mid-latitudes)
  Extent:     lon [-125.0, -66.0]  lat [24.0, 50.0]
  Dimensions: 1180 cols × 520 rows

Bands:
  Each band = one hail-size bin (hundredths of inches, step=25).
  Band 1 -> size  0– 24
  Band 2 -> size 25– 49
  Band 3 -> size 50– 74
  ...
  Band 29 -> size 700–724
  (bins beyond observed range remain zero)

Cell values:
  uint8 count of hail reports in that cell for that storm day.
  Capped at 255 (no real-world day will approach that in a 0.05° cell).

Output:
  data/hail/
    YYYY/
      hail_YYYYMMDD.tif

Source data:
  data/spc/YYYY/YYMMDD_rpts_hail.csv
"""

import os, csv, sys, time, collections
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
try:
    import rasterio
    from rasterio.transform import from_origin
    HAVE_RASTERIO = True
except ImportError:
    HAVE_RASTERIO = False

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

# ── grid definition ───────────────────────────────────────────────────────────
LON_MIN, LON_MAX =  -125.0,  -66.0
LAT_MIN, LAT_MAX =    24.0,   50.0
DX = DY = 0.05

NCOLS = int(round((LON_MAX - LON_MIN) / DX))   # 1180
NROWS = int(round((LAT_MAX - LAT_MIN) / DY))   # 520

# GeoTIFF transform: top-left corner, pixel size (dy is negative = N->S)
TRANSFORM = from_origin(LON_MIN, LAT_MAX, DX, DY) if HAVE_RASTERIO else None

# ── size bins ─────────────────────────────────────────────────────────────────
BIN_STEP  = 25
MAX_SIZE  = 700           # max observed in dataset
N_BINS    = (MAX_SIZE // BIN_STEP) + 1   # 29  (bins 0..28 -> sizes 0..700)
# Band index = size // BIN_STEP

# ── paths ─────────────────────────────────────────────────────────────────────
SPC_DIR = DATA_ROOT / "spc"
OUT_DIR = DATA_ROOT / "hail"
LOG     = LOGS_ROOT / "hail_rasters_build.log"

CRS = "EPSG:4326"

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(msg + "\n")

def latlon_to_cell(lat, lon):
    """Return (row, col) for a lat/lon point; None if outside grid."""
    if not (LAT_MIN <= lat <= LAT_MAX and LON_MIN <= lon <= LON_MAX):
        return None
    col = int((lon - LON_MIN) / DX)
    row = int((LAT_MAX - lat) / DY)
    col = min(col, NCOLS - 1)
    row = min(row, NROWS - 1)
    return row, col

def parse_hail_file(filepath):
    """
    Parse one SPC hail CSV.
    Returns list of (lat, lon, size_bin) tuples for valid rows.
    """
    records = []
    try:
        with open(filepath, newline="", errors="replace") as f:
            for row in csv.DictReader(f):
                try:
                    lat  = float(row["Lat"])
                    lon  = float(row["Lon"])
                    size = int(row["Size"])
                except (KeyError, ValueError, TypeError):
                    continue
                cell = latlon_to_cell(lat, lon)
                if cell is None:
                    continue
                size_bin = min(size // BIN_STEP, N_BINS - 1)
                records.append((cell[0], cell[1], size_bin))
    except Exception as e:
        pass
    return records

def build_grid(records):
    """
    Accumulate records into a (N_BINS, NROWS, NCOLS) uint8 array.
    """
    grid = np.zeros((N_BINS, NROWS, NCOLS), dtype=np.uint8)
    for row, col, size_bin in records:
        v = grid[size_bin, row, col]
        if v < 255:
            grid[size_bin, row, col] = v + 1
    return grid

def write_geotiff(grid, outpath):
    """Write (N_BINS, NROWS, NCOLS) array as multi-band GeoTIFF."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        outpath,
        "w",
        driver   = "GTiff",
        height   = NROWS,
        width    = NCOLS,
        count    = N_BINS,
        dtype    = np.uint8,
        crs      = CRS,
        transform= TRANSFORM,
        compress = "lzw",
        nodata   = None,
    ) as dst:
        for b in range(N_BINS):
            dst.write(grid[b], b + 1)
            lo = b * BIN_STEP
            hi = lo + BIN_STEP - 1
            dst.update_tags(b + 1, size_range=f"{lo}-{hi} hundredths_of_inches")

def spc_filename_to_date(stem):
    """'100601' -> datetime(2010, 6, 1)"""
    yr2 = int(stem[:2])
    yr  = 2000 + yr2
    mo  = int(stem[2:4])
    dy  = int(stem[4:6])
    return datetime(yr, mo, dy)

def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    import random
    import sys
    errors = []

    if not OUT_DIR.exists():
        errors.append(f"Missing directory: {OUT_DIR}")
    else:
        tifs = list(OUT_DIR.rglob("hail_????????.tif"))
        if len(tifs) <= 4000:
            errors.append(f"Too few TIFFs: {len(tifs)} (expected >4000)")
        else:
            sample = random.sample(tifs, min(10, len(tifs)))
            for p in sample:
                try:
                    import rasterio as _rio
                    with _rio.open(p) as src:
                        src.read(1)
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
    if not HAVE_RASTERIO:
        print("ERROR: rasterio not installed. Run:")
        print("  pip install rasterio")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Hail raster build started")
    log(f"  Grid:    {NCOLS}x{NROWS} cells @ {DX}° ({DX*111:.1f} km at equator)")
    log(f"  Bands:   {N_BINS}  (size 0–{MAX_SIZE} hundredths, step {BIN_STEP})")
    log(f"  Output:  {OUT_DIR}")

    # Collect all hail files across all years
    hail_files = sorted([
        fp for yr_dir in sorted(SPC_DIR.iterdir()) if yr_dir.is_dir()
        for fp in yr_dir.glob("*_rpts_hail.csv")
    ])
    log(f"  Files:   {len(hail_files):,} hail CSVs\n")

    total_files  = 0
    total_reports = 0
    skipped       = 0

    for fp in hail_files:
        stem    = fp.stem                          # e.g. '100601_rpts_hail'
        date    = spc_filename_to_date(stem[:6])
        out_yr  = OUT_DIR / str(date.year)
        outfile = out_yr / f"hail_{date.strftime('%Y%m%d')}.tif"

        if outfile.exists():
            skipped += 1
            continue

        records = parse_hail_file(fp)
        if not records:
            # Write an all-zero file so we know the day was processed
            grid = np.zeros((N_BINS, NROWS, NCOLS), dtype=np.uint8)
        else:
            grid = build_grid(records)

        write_geotiff(grid, outfile)
        total_files   += 1
        total_reports += len(records)

        if total_files % 200 == 0 or total_files == 1:
            log(f"  [{date.strftime('%Y-%m-%d')}] {len(records):>5} reports -> {outfile.name}  "
                f"(total so far: {total_files:,} files, {total_reports:,} reports)")

    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done!")
    log(f"  Files written: {total_files:,}")
    log(f"  Files skipped (already exist): {skipped:,}")
    log(f"  Total reports gridded: {total_reports:,}")
    log(f"  Grid definition:")
    log(f"    LON: {LON_MIN} to {LON_MAX}  ({NCOLS} cols x {DX}°)")
    log(f"    LAT: {LAT_MIN} to {LAT_MAX}  ({NROWS} rows x {DY}°)")
    log(f"    CRS: {CRS}")
    log(f"    Bands: {N_BINS} (band N = size {BIN_STEP}x(N-1) to {BIN_STEP}xN-1 hundredths/inch)")

    if not validate_outputs():
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        ok = validate_outputs()
        sys.exit(0 if ok else 1)
    main()
