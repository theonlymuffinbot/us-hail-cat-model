#!/usr/bin/env python3
"""
12_build_occurrence_probs.py

Generates annual occurrence probability rasters for specific hail size thresholds.

P_occ(T) = fraction of years (2004-2025) where at least one event had
           peak hail >= threshold T at that cell.

Thresholds: 0.25, 0.50, 1.50, 2.00, 3.00, 4.00, 5.00 inches

Note: event_peak stores bin midpoints. Threshold comparison is against the
lower bound of the relevant bin, so e.g. >= 0.25" captures all reports in
the 0.25-0.49" bin (midpoint 0.37") and above.

Output: data/hail_0.25deg/p_occ_{threshold}in.tif
"""

import os, time, glob
import numpy as np
import pandas as pd
import rasterio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

ROOT     = str(DATA_ROOT / "hail_0.25deg")
CAT_PATH = os.path.join(ROOT, "event_catalog.csv")
EPK_PATH = os.path.join(ROOT, "event_peak_array.npy")
REF_TIF  = None  # will use first storm tif for profile

# Thresholds in inches
THRESHOLDS = [0.25, 0.50, 1.50, 2.00, 3.00, 4.00, 5.00]
YEARS      = list(range(2004, 2026))   # 2004-2025 complete years
N_YEARS    = len(YEARS)

t0 = time.time()
print("="*60)
print("OCCURRENCE PROBABILITY RASTERS")
print("="*60)

# Load
print("\nLoading event_peak_array...")
event_peak = np.load(EPK_PATH)          # (n_events, 104, 236)
n_events, nrows, ncols = event_peak.shape
print(f"  Shape: {event_peak.shape}")

print("Loading event catalog...")
event_df    = pd.read_csv(CAT_PATH, parse_dates=["start_date"])
event_years = event_df["start_date"].dt.year.values
print(f"  {len(event_df)} events, years {event_years.min()}–{event_years.max()}")

# Build year -> event indices lookup
year_events = {y: [] for y in YEARS}
for i, yr in enumerate(event_years):
    if yr in year_events:
        year_events[yr].append(i)

# Annual max per cell across all complete years
print(f"\nBuilding annual max array ({N_YEARS} years)...")
annual_max = np.zeros((N_YEARS, nrows, ncols), dtype=np.float32)
for yi, yr in enumerate(YEARS):
    idx = year_events[yr]
    if idx:
        annual_max[yi] = event_peak[idx, :, :].max(axis=0)

print(f"  Annual max range: {annual_max.min():.2f}–{annual_max.max():.2f} in")

# Rasterio profile from reference file
ref_files = sorted(glob.glob(os.path.join(ROOT, "2004/*.tif")))
with rasterio.open(ref_files[0]) as src:
    out_profile = src.profile.copy()
out_profile.update(count=1, dtype="float32", nodata=-9999.0,
                   compress="lzw", predictor=3)

# Write one raster per threshold
print(f"\nWriting {len(THRESHOLDS)} occurrence probability rasters...")
for thresh in THRESHOLDS:
    # P_occ = fraction of years where annual max >= threshold
    exceeds     = (annual_max >= thresh)          # (N_YEARS, nrows, ncols) bool
    p_occ       = exceeds.sum(axis=0) / N_YEARS   # (nrows, ncols) float

    # Mask cells that never had any hail in record at any threshold
    # (leave as 0.0 — already correct)
    out = p_occ.astype(np.float32)

    fname = f"p_occ_{thresh:.2f}in.tif".replace(".", "p")  # e.g. p_occ_0p25in.tif
    path  = os.path.join(ROOT, fname)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)
        dst.update_tags(1,
            threshold_inches=str(thresh),
            description=f"Annual occurrence probability of hail >= {thresh}\"",
            method="fraction of years (2004-2025) with annual max >= threshold",
            n_years=str(N_YEARS),
        )

    # Stats for QC
    nonz = out[out > 0]
    print(f"  p_occ >= {thresh:.2f}\"  -> {fname}")
    print(f"    cells>0: {len(nonz):,}  max: {out.max():.3f}  "
          f"p50(nonzero): {np.median(nonz):.3f}  p90(nonzero): {np.percentile(nonz,90):.3f}")

print(f"\nDone in {time.time()-t0:.1f}s")
print(f"\nOutputs in {ROOT}/:")
for thresh in THRESHOLDS:
    fname = f"p_occ_{thresh:.2f}in.tif".replace(".", "p")
    path  = os.path.join(ROOT, fname)
    size  = os.path.getsize(path)
    print(f"  {fname}  ({size/1024:.0f} KB)")
