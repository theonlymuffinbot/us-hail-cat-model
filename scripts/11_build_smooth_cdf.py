#!/usr/bin/env python3
"""
11_build_smooth_cdf.py

Rebuild hail return period rasters using spatially-pooled CDF fitting.

Instead of fitting lognormal + GPD to each cell's 22 annual maxima independently
(5-15 non-zero obs -> noisy), we pool observations from all cells within a 150 km
radius weighted by exp(-d/75km). This gives ~50-150 effective observations per
cell and produces smooth, stable fits across the domain.

CONUS masking:
  - Cells with zero hail in the entire 22-year record -> nodata
  - Cells north of 49.5° (Canada) or south of 24.0° -> nodata
  - Cells with fewer than 3 pooled non-zero obs -> nodata

Outputs (replace existing files in data/hail_0.25deg/):
  rp_10yr_hail.tif, rp_25yr_hail.tif, rp_50yr_hail.tif,
  rp_100yr_hail.tif, rp_200yr_hail.tif, rp_250yr_hail.tif,
  rp_500yr_hail.tif, p_occurrence.tif

Parameters:
  POOL_RADIUS_KM = 150    kernel search radius
  DECAY_KM       = 75     exponential decay half-radius
  MIN_OBS        = 10     minimum pooled non-zero obs before fitting
  GPD_THRESHOLD  = 2.0    inches — above this, use GPD tail
  YEARS          = 2004-2025 (22 complete years)
"""

import os, time, json, glob
import numpy as np
import pandas as pd
import rasterio
from scipy.stats import lognorm, genpareto
from scipy.optimize import minimize
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

ROOT    = str(DATA_ROOT / "hail_0.25deg")
YEARS   = list(range(2004, 2026))
N_YEARS = len(YEARS)

POOL_RADIUS_KM = 150.0
DECAY_KM       = 75.0
MIN_OBS        = 10
GPD_THRESH     = 2.0

NROWS, NCOLS = 104, 236
CELL_DEG     = 0.25
LAT_ORIG     = 50.0
LON_ORIG     = -125.0

RETURN_PERIODS = [10, 25, 50, 100, 200, 250, 500]

t0 = time.time()
LOG = []

def log(msg):
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}")
    LOG.append(f"[{ts}] {msg}")

def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    errors = []
    root = DATA_ROOT / "hail_0.25deg"
    rp_fnames = ["rp_10yr_hail.tif", "rp_25yr_hail.tif", "rp_50yr_hail.tif",
                 "rp_100yr_hail.tif", "rp_200yr_hail.tif", "rp_250yr_hail.tif",
                 "rp_500yr_hail.tif", "p_occurrence.tif"]

    for fname in rp_fnames:
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

log("="*60)
log("SPATIAL-POOLED CDF REBUILD")
log("="*60)

# ── Cell coordinates ─────────────────────────────────────────
rows_all = np.arange(NROWS)
cols_all = np.arange(NCOLS)
R, C     = np.meshgrid(rows_all, cols_all, indexing='ij')
LAT_GRID = LAT_ORIG - R * CELL_DEG - CELL_DEG / 2   # (104, 236)
LON_GRID = LON_ORIG + C * CELL_DEG + CELL_DEG / 2

def haversine_grid(lat0, lon0):
    """Great-circle distance from (lat0,lon0) to all grid cells (km)."""
    dlat = np.radians(LAT_GRID - lat0)
    dlon = np.radians(LON_GRID - lon0)
    a    = (np.sin(dlat/2)**2 +
            np.cos(np.radians(lat0)) * np.cos(np.radians(LAT_GRID)) *
            np.sin(dlon/2)**2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

# ── Load event peak array ────────────────────────────────────
log("Loading event_peak_array.npy...")
event_peak = np.load(os.path.join(ROOT, "event_peak_array.npy"))  # (N,104,236)
event_df   = pd.read_csv(os.path.join(ROOT, "event_catalog.csv"),
                          parse_dates=["start_date"])
event_years = event_df["start_date"].dt.year.values
log(f"  event_peak shape: {event_peak.shape}")

# ── Build annual max array ───────────────────────────────────
log(f"Building annual max for {N_YEARS} years ({YEARS[0]}–{YEARS[-1]})...")
year_events = {y: [] for y in YEARS}
for i, yr in enumerate(event_years):
    if yr in year_events:
        year_events[yr].append(i)

annual_max = np.zeros((N_YEARS, NROWS, NCOLS), dtype=np.float32)
for yi, yr in enumerate(YEARS):
    idx = year_events[yr]
    if idx:
        annual_max[yi] = event_peak[np.array(idx)].max(axis=0)

log(f"  Annual max range: {annual_max.min():.2f}–{annual_max.max():.2f} in")

# ── CONUS mask ───────────────────────────────────────────────
# Cells that have at least one non-zero year AND are within CONUS lat bounds
has_hail       = (annual_max > 0).any(axis=0)             # (104,236) bool
in_conus_lat   = (LAT_GRID >= 24.5) & (LAT_GRID <= 49.5)  # clip Canada/Mexico
in_conus_lon   = (LON_GRID >= -124.8) & (LON_GRID <= -66.5)
conus_mask     = has_hail & in_conus_lat & in_conus_lon

n_active = conus_mask.sum()
log(f"  Active CONUS cells: {n_active:,} of {NROWS*NCOLS:,}")

# ── Rasterio output profile ──────────────────────────────────
ref_files = sorted(glob.glob(os.path.join(ROOT, "2004/*.tif")))
with rasterio.open(ref_files[0]) as src:
    out_profile = src.profile.copy()
out_profile.update(count=1, dtype="float32", nodata=-9999.0,
                   compress="lzw", predictor=3)

# ── Fitting helpers ──────────────────────────────────────────
def fit_lognormal_mle(data):
    """Fit lognormal to positive data via MLE. Returns (mu, sigma)."""
    log_d = np.log(data[data > 0])
    if len(log_d) < 3:
        return np.nan, np.nan
    return float(log_d.mean()), float(log_d.std(ddof=1))

def fit_gpd_mle(exceedances):
    """Fit GPD to exceedances via MLE. Returns (shape c, scale)."""
    if len(exceedances) < 5:
        return 0.0, float(exceedances.mean()) if len(exceedances) > 0 else 1.0
    from scipy.stats import genpareto
    try:
        c, loc, scale = genpareto.fit(exceedances, floc=0)
        return float(c), float(scale)
    except Exception:
        return 0.0, float(exceedances.mean())

def return_period_value(rp, mu, sigma, c_gpd, scale_gpd,
                        gpd_thresh, p_occ_rate, n_obs):
    """
    Compute hail size at given return period (years).
    Uses zero-inflated lognormal body + GPD tail.
    p_occ_rate = fraction of years with non-zero hail at this cell.
    """
    # Annual non-exceedance probability
    p_annual = 1.0 - 1.0 / rp
    # Conditional prob: P(X <= x | X > 0) = (p_annual - (1-p_occ)) / p_occ
    if p_occ_rate <= 0 or p_occ_rate > 1:
        return np.nan
    p_cond = (p_annual - (1.0 - p_occ_rate)) / p_occ_rate
    if p_cond <= 0:
        return 0.0
    if p_cond >= 1:
        return np.nan

    # Is this in the body (lognormal) or tail (GPD)?
    # P(X <= gpd_thresh | X > 0) from lognormal
    if np.isnan(mu) or np.isnan(sigma) or sigma <= 0:
        return np.nan
    p_body_thresh = lognorm.cdf(gpd_thresh, s=sigma, scale=np.exp(mu))

    if p_cond <= p_body_thresh:
        # Body: invert lognormal
        return float(lognorm.ppf(p_cond, s=sigma, scale=np.exp(mu)))
    else:
        # Tail: invert GPD above threshold
        p_exceed_thresh = 1.0 - p_body_thresh
        if p_exceed_thresh <= 0:
            return float(lognorm.ppf(p_cond, s=sigma, scale=np.exp(mu)))
        # Conditional prob within tail
        p_tail_cond = (p_cond - p_body_thresh) / p_exceed_thresh
        p_tail_cond = np.clip(p_tail_cond, 0, 1 - 1e-9)
        if scale_gpd <= 0:
            return gpd_thresh
        if abs(c_gpd) < 1e-6:
            # Exponential special case
            return float(gpd_thresh - scale_gpd * np.log(1 - p_tail_cond))
        else:
            return float(gpd_thresh + scale_gpd / c_gpd *
                         ((1 - p_tail_cond) ** (-c_gpd) - 1))

# ── Main loop ────────────────────────────────────────────────
log(f"\nFitting pooled CDFs for {n_active:,} active cells...")
log(f"  Pool radius: {POOL_RADIUS_KM} km  |  Decay: {DECAY_KM} km  |  Min obs: {MIN_OBS}")

rp_arrays = {rp: np.full((NROWS, NCOLS), -9999.0, dtype=np.float32)
             for rp in RETURN_PERIODS}
p_occ_arr = np.full((NROWS, NCOLS), -9999.0, dtype=np.float32)

active_rows, active_cols = np.where(conus_mask)
n_active_cells = len(active_rows)

t_start = time.time()
n_fitted = 0
n_nodata = 0

for idx in range(n_active_cells):
    ri, ci = int(active_rows[idx]), int(active_cols[idx])
    lat0   = float(LAT_GRID[ri, ci])
    lon0   = float(LON_GRID[ri, ci])

    # Distance to all cells (km)
    dist = haversine_grid(lat0, lon0)  # (104,236)

    # Kernel weights
    weights = np.exp(-dist / DECAY_KM)
    in_pool = (dist <= POOL_RADIUS_KM) & conus_mask  # only active CONUS cells

    # Pool weighted annual max observations
    # Each cell contributes its N_YEARS annual maxima, weighted by w
    pool_obs = []
    pool_wts = []
    for pr, pc in zip(*np.where(in_pool)):
        w   = float(weights[pr, pc])
        ann = annual_max[:, pr, pc]  # 22 values
        pool_obs.extend(ann.tolist())
        pool_wts.extend([w] * N_YEARS)

    pool_obs = np.array(pool_obs, dtype=np.float32)
    pool_wts = np.array(pool_wts, dtype=np.float32)

    nz_mask = pool_obs > 0
    nz_obs  = pool_obs[nz_mask]
    nz_wts  = pool_wts[nz_mask]

    # Need enough non-zero observations
    eff_n = float(nz_wts.sum()) if len(nz_wts) > 0 else 0.0

    if len(nz_obs) < MIN_OBS or eff_n < MIN_OBS:
        n_nodata += 1
        continue

    # Occurrence rate: weighted fraction of year-cell obs that are non-zero
    total_wt   = pool_wts.sum()
    nz_wt      = nz_wts.sum()
    p_occ_rate = float(nz_wt / total_wt) if total_wt > 0 else 0.0

    # Weighted lognormal fit on non-zero obs
    # Use weighted mean/std of log(x)
    log_x   = np.log(nz_obs)
    w_norm  = nz_wts / nz_wts.sum()
    mu_w    = float(np.dot(w_norm, log_x))
    var_w   = float(np.dot(w_norm, (log_x - mu_w)**2))
    sigma_w = float(np.sqrt(max(var_w, 1e-6)))

    # GPD fit on exceedances above threshold (weighted bootstrap sample)
    exceedances = nz_obs[nz_obs > GPD_THRESH] - GPD_THRESH
    if len(exceedances) >= 5:
        c_gpd, scale_gpd = fit_gpd_mle(exceedances)
        c_gpd    = float(np.clip(c_gpd, -0.5, 0.5))
        scale_gpd = float(max(scale_gpd, 0.05))
    else:
        c_gpd, scale_gpd = 0.0, 0.5

    # Store p_occ (for the target cell itself)
    own_nz = annual_max[:, ri, ci]
    p_occ_arr[ri, ci] = float((own_nz > 0).sum() / N_YEARS)

    # Compute return period values
    for rp in RETURN_PERIODS:
        val = return_period_value(rp, mu_w, sigma_w, c_gpd, scale_gpd,
                                  GPD_THRESH, p_occ_rate, len(nz_obs))
        if val is not None and np.isfinite(val) and val >= 0:
            rp_arrays[rp][ri, ci] = float(np.clip(val, 0, 20))

    n_fitted += 1

    if idx > 0 and idx % 500 == 0:
        elapsed = time.time() - t_start
        rate    = idx / elapsed
        eta     = (n_active_cells - idx) / rate
        log(f"  {idx:,}/{n_active_cells:,}  ({100*idx/n_active_cells:.1f}%)  "
            f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s")

elapsed = time.time() - t_start
log(f"\nFitting complete: {n_fitted:,} cells fitted, {n_nodata:,} nodata  ({elapsed:.1f}s)")

# ── Write rasters ─────────────────────────────────────────────
log("\nWriting output rasters...")

RP_NAMES = {
    10:  "rp_10yr_hail.tif",
    25:  "rp_25yr_hail.tif",
    50:  "rp_50yr_hail.tif",
    100: "rp_100yr_hail.tif",
    200: "rp_200yr_hail.tif",
    250: "rp_250yr_hail.tif",
    500: "rp_500yr_hail.tif",
}

for rp, fname in RP_NAMES.items():
    path = os.path.join(ROOT, fname)
    arr  = rp_arrays[rp]
    valid = arr[arr > 0]
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr, 1)
        dst.update_tags(1,
            return_period_years=str(rp),
            method="spatially_pooled_lognormal_gpd",
            pool_radius_km=str(POOL_RADIUS_KM),
            decay_km=str(DECAY_KM),
            years="2004-2025",
        )
    log(f"  {fname}  valid cells: {len(valid):,}  "
        f"p50: {np.median(valid):.2f}  max: {valid.max():.2f}")

# p_occurrence
pocc_path = os.path.join(ROOT, "p_occurrence.tif")
with rasterio.open(pocc_path, "w", **out_profile) as dst:
    dst.write(p_occ_arr, 1)
    dst.update_tags(1,
        description="Annual occurrence probability (fraction of years >= 1in)",
        threshold_inches="1.0",
        method="spatially_pooled",
        years="2004-2025",
    )
valid_occ = p_occ_arr[p_occ_arr >= 0]
log(f"  p_occurrence.tif  valid: {len(valid_occ):,}  max: {valid_occ.max():.3f}")

# ── Rebuild p_occ threshold TIFs ─────────────────────────────
log("\nRebuilding p_occ threshold TIFs...")
THRESHOLDS = [0.25, 0.50, 1.50, 2.00, 3.00, 4.00, 5.00]

# Annual max array already built above
for thresh in THRESHOLDS:
    exceeds  = (annual_max >= thresh)
    p_occ_t  = np.where(conus_mask,
                        exceeds.sum(axis=0) / N_YEARS,
                        -9999.0).astype(np.float32)
    fname    = f"p_occ_{thresh:.2f}in.tif".replace(".", "p")
    path     = os.path.join(ROOT, fname)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(p_occ_t, 1)
        dst.update_tags(1,
            threshold_inches=str(thresh),
            description=f"Annual occurrence probability hail >= {thresh}\"",
            conus_mask="applied",
            years="2004-2025",
        )
    valid = p_occ_t[p_occ_t >= 0]
    log(f"  p_occ >= {thresh:.2f}\"  cells: {len(valid):,}  max: {valid.max():.3f}")

total = time.time() - t0
log(f"\n{'='*60}")
log(f"DONE in {total/60:.1f} min")
log(f"{'='*60}")

LOGS_ROOT.mkdir(parents=True, exist_ok=True)
with open(LOGS_ROOT / "smooth_cdf.log", "w") as f:
    f.write("\n".join(LOG))

if not validate_outputs():
    import sys
    sys.exit(1)
