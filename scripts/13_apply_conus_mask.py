#!/usr/bin/env python3
"""
13_apply_conus_mask.py

1. Build a strict CONUS land mask using regionmask (US states polygon).
2. Re-apply to all RP and p_occ TIFs — cells outside CONUS -> nodata.
3. Smooth p_occ threshold TIFs with the same 150km pooling kernel
   so they are visually consistent with the smoothed RP maps.
"""

import os, glob, time
import numpy as np
import rasterio
import regionmask
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

ROOT    = str(DATA_ROOT / "hail_0.25deg")
NROWS, NCOLS = 104, 236
CELL_DEG     = 0.25
LAT_ORIG, LON_ORIG = 50.0, -125.0

t0 = time.time()

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    errors = []
    root = DATA_ROOT / "hail_0.25deg"
    rp_fnames = ["rp_10yr_hail.tif", "rp_25yr_hail.tif", "rp_50yr_hail.tif",
                 "rp_100yr_hail.tif", "rp_200yr_hail.tif", "rp_250yr_hail.tif",
                 "rp_500yr_hail.tif", "p_occurrence.tif"]
    pocc_fnames = [f"p_occ_{f'{t:.2f}'.replace('.', 'p')}in.tif"
                   for t in [0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]]

    for fname in rp_fnames + pocc_fnames:
        p = root / fname
        if not p.exists():
            errors.append(f"Missing: {fname}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty: {fname}")
        else:
            try:
                with rasterio.open(p) as src:
                    src.read(1)
            except Exception as e:
                errors.append(f"Cannot read {fname}: {e}")

    if errors:
        log("CRITICAL: Output validation FAILED:")
        for e in errors:
            log(f"  ✗ {e}")
        return False
    log("Output validation passed ✓")
    return True


# ── --validate early exit ──────────────────────────────────────────────────────
import sys as _sys
if "--validate" in _sys.argv:
    ok = validate_outputs()
    _sys.exit(0 if ok else 1)

# ── Build CONUS mask ─────────────────────────────────────────
log("Building CONUS land mask via regionmask...")

# Cell centre coordinates
rows_idx = np.arange(NROWS)
cols_idx = np.arange(NCOLS)
R, C     = np.meshgrid(rows_idx, cols_idx, indexing='ij')
LAT_GRID = LAT_ORIG - R * CELL_DEG - CELL_DEG / 2
LON_GRID = LON_ORIG + C * CELL_DEG + CELL_DEG / 2

# Use Natural Earth US states (regionmask built-in, no shapefile needed)
us_states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
mask_2d   = us_states.mask(LON_GRID[0, :], LAT_GRID[:, 0])  # returns 2D int array

# mask_2d: integer >= 0 = inside a state, -1 = outside
# Note: regionmask.mask uses (lon, lat) ordering and returns (lat, lon)
conus_land = (mask_2d >= 0).values  # (104, 236) bool — True = inside CONUS

n_conus = conus_land.sum()
log(f"  CONUS cells: {n_conus:,} of {NROWS*NCOLS:,}")

# ── Apply mask to all RP + p_occ TIFs ───────────────────────
all_tifs = (
    glob.glob(os.path.join(ROOT, "rp_*yr_hail.tif")) +
    [os.path.join(ROOT, "p_occurrence.tif")] +
    glob.glob(os.path.join(ROOT, "p_occ_*.tif"))
)

log(f"\nApplying CONUS mask to {len(all_tifs)} rasters...")
for path in sorted(all_tifs):
    with rasterio.open(path, "r+") as dst:
        arr     = dst.read(1)
        nodata  = dst.nodata if dst.nodata is not None else -9999.0
        arr     = arr.astype(np.float32)
        # Zero out cells outside CONUS
        arr[~conus_land] = nodata
        dst.write(arr, 1)
    fname = os.path.basename(path)
    valid = arr[conus_land & (arr > 0)]
    log(f"  {fname}  valid: {len(valid):,}  max: {valid.max():.3f}" if len(valid) else f"  {fname}  (no valid cells)")

# ── Smooth p_occ threshold TIFs with spatial pooling ─────────
log("\nSmoothing p_occ threshold TIFs (150km pool, 75km decay)...")

POOL_RADIUS_KM = 150.0
DECAY_KM       = 75.0
MIN_OBS        = 5

def haversine_grid(lat0, lon0):
    dlat = np.radians(LAT_GRID - lat0)
    dlon = np.radians(LON_GRID - lon0)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat0)) * np.cos(np.radians(LAT_GRID)) *
            np.sin(dlon/2)**2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

THRESH_TIFS = sorted(glob.glob(os.path.join(ROOT, "p_occ_*.tif")))

for path in THRESH_TIFS:
    fname = os.path.basename(path)
    with rasterio.open(path) as src:
        raw     = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        nodata  = src.nodata if src.nodata is not None else -9999.0

    smoothed = np.full((NROWS, NCOLS), nodata, dtype=np.float32)

    active_r, active_c = np.where(conus_land)

    for idx in range(len(active_r)):
        ri, ci = int(active_r[idx]), int(active_c[idx])
        lat0   = float(LAT_GRID[ri, ci])
        lon0   = float(LON_GRID[ri, ci])

        dist    = haversine_grid(lat0, lon0)
        weights = np.exp(-dist / DECAY_KM)
        in_pool = (dist <= POOL_RADIUS_KM) & conus_land & (raw != nodata)

        w_pool = weights[in_pool]
        v_pool = raw[in_pool]

        if len(v_pool) < MIN_OBS or w_pool.sum() == 0:
            smoothed[ri, ci] = raw[ri, ci] if raw[ri, ci] != nodata else nodata
            continue

        w_norm = w_pool / w_pool.sum()
        smoothed[ri, ci] = float(np.dot(w_norm, v_pool))

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(smoothed, 1)

    valid = smoothed[conus_land & (smoothed >= 0)]
    log(f"  {fname}  valid: {len(valid):,}  max: {valid.max():.3f}")

log(f"\nDone in {time.time()-t0:.1f}s")

if not validate_outputs():
    import sys
    sys.exit(1)
