#!/usr/bin/env python3
"""
10_hail_catmodel_pipeline.py
Hail Catastrophe Model — Primary Hazard CDF & Spatial Correlation
Steps 0–4 for data/hail_0.25deg/

Adapted from design doc with the following known dataset properties:
  - Band descriptions: None (bin info in band tags: size_range '0-24 hundredths_of_inches')
  - Bands are integer counts (0/1/2), BAND_METHOD = 'max_active_bin'
  - Bin N (1-indexed): lo=(N-1)*25, hi=lo+24, mid=lo+12 hundredths of inches
  - 29 bands, uint16, no nodata, EPSG:4326, 0.25° resolution
  - Spatial extent: lon [-125, -66], lat [24, 50] (CONUS)
"""

import os, re, glob, json, time, random, warnings
warnings.filterwarnings('ignore')

import rasterio
import numpy as np
import pandas as pd
from datetime import datetime
from scipy import stats
from scipy.optimize import brentq, curve_fit
from scipy.ndimage import binary_dilation
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr as scipy_spearmanr
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

T0 = time.time()

def elapsed():
    return f"{(time.time()-T0)/60:.1f} min"

print("="*60)
print("HAIL CAT MODEL PIPELINE")
print("="*60)

# ═══════════════════════════════════════════════════════════════
# STEP 0 — Discover data structure and read bin definitions
# ═══════════════════════════════════════════════════════════════

ROOT = str(DATA_ROOT / "hail_0.25deg")

tif_files = sorted(
    f
    for year_dir in sorted(glob.glob(os.path.join(ROOT, "*")))
    if os.path.isdir(year_dir)
    for f in glob.glob(os.path.join(year_dir, "*.tif"))
)

assert len(tif_files) > 0, f"No .tif files found under {ROOT}"
print(f"\nStep 0 — Data discovery")
print(f"  Found {len(tif_files)} .tif files across "
      f"{len(set(os.path.basename(os.path.dirname(f)) for f in tif_files))} year folders")
print(f"  First: {tif_files[0]}")
print(f"  Last:  {tif_files[-1]}")

# --- 0.2 Full metadata from first file ---
print("\n" + "="*60)
print("RASTER METADATA DISCOVERY")
print("="*60)

with rasterio.open(tif_files[0]) as src:
    N_BANDS      = src.count
    nrows, ncols = src.height, src.width
    ref_crs      = src.crs
    ref_transform = src.transform
    ref_shape    = (nrows, ncols)

    print(f"\nFile:      {tif_files[0]}")
    print(f"Bands:     {N_BANDS}")
    print(f"CRS:       {src.crs}")
    print(f"Transform: {src.transform}")
    print(f"Shape:     {nrows} rows x {ncols} cols")
    print(f"Dtype:     {src.dtypes[0]}")
    print(f"Nodata:    {src.nodata}")

    print("\nBand descriptions + tags (first 5):")
    band_descriptions = []
    for i in range(1, N_BANDS + 1):
        desc = src.descriptions[i - 1]
        band_descriptions.append(desc)
        if i <= 5:
            print(f"  Band {i:2d}: desc={desc!r}  tags={src.tags(i)}")

    print("\nDataset tags:")
    for k, v in src.tags().items():
        print(f"  {k}: {v}")

    sample    = src.read().astype(np.float32)
    band_sums = sample.sum(axis=0)
    print(f"\nValue range:    {sample.min():.4f} – {sample.max():.4f}")
    print(f"Band-sum range: {band_sums.min():.4f} – {band_sums.max():.4f}")
    hint = ("probabilities -> 'weighted_mean'" if abs(band_sums.max() - 1.0) < 0.1
            else "counts/binary -> 'max_active_bin'")
    print(f"Interpretation: {hint}")

print("="*60)

# --- 0.3 Bin midpoints ---
# Try range parser on band tags first, fall back to hardcoded formula

def parse_bins_from_size_range_tags(tif_path):
    """Parse bin midpoints from band tags: size_range = '0-24 hundredths_of_inches'"""
    midpoints = []
    with rasterio.open(tif_path) as src:
        for i in range(1, src.count + 1):
            tag = src.tags(i).get("size_range", "")
            m = re.search(r"(\d+)-(\d+)", tag)
            if not m:
                raise ValueError(f"Band {i} size_range tag not parseable: {tag!r}")
            lo, hi = int(m.group(1)), int(m.group(2))
            midpoints.append((lo + hi) / 2.0 / 100.0)  # convert hundredths -> inches
    return np.array(midpoints, dtype=np.float32)

def parse_bins_from_descriptions(descriptions):
    pattern = re.compile(
        r"([0-9]+\.?[0-9]*)"
        r"[\s_\-\u2013to]+"
        r"([0-9]+\.?[0-9]*)"
    )
    midpoints = []
    for desc in descriptions:
        if desc is None:
            raise ValueError("Band description is None")
        m = pattern.search(str(desc))
        if not m:
            raise ValueError(f"No numeric range in: {desc!r}")
        lo, hi = float(m.group(1)), float(m.group(2))
        midpoints.append((lo + hi) / 2.0)
    return np.array(midpoints, dtype=np.float32)

try:
    bin_midpoints = parse_bins_from_size_range_tags(tif_files[0])
    print("Bin midpoints from band size_range tags:")
except Exception as e1:
    try:
        bin_midpoints = parse_bins_from_descriptions(band_descriptions)
        print("Bin midpoints from band descriptions:")
    except Exception as e2:
        print(f"Both parsers failed ({e1}; {e2}) — using hardcoded formula")
        # Known: Band N (1-indexed): lo=(N-1)*25, hi=lo+24 hundredths of inches
        bin_midpoints = np.array([((i * 25) + 12) / 100.0 for i in range(N_BANDS)],
                                  dtype=np.float32)

print(f"  {np.round(bin_midpoints, 4).tolist()}")
assert len(bin_midpoints) == N_BANDS

with open(os.path.join(ROOT, "bin_midpoints.json"), "w") as f:
    json.dump({
        "source":          tif_files[0],
        "n_bands":         N_BANDS,
        "band_descriptions": band_descriptions,
        "bin_midpoints_in": bin_midpoints.tolist(),
        "note":            "Bin N (1-indexed): lo=(N-1)*25, hi=lo+24 hundredths of inches"
    }, f, indent=2)
print("Saved bin_midpoints.json")

# --- 0.4 Band interpretation ---
# Band-sum max = 2.0 -> integer counts -> max active bin
BAND_METHOD = "max_active_bin"
print(f"\nBAND_METHOD = {BAND_METHOD!r} (integer counts, not probabilities)")

# --- 0.5 Date parsing ---
def parse_date(filepath):
    basename = os.path.splitext(os.path.basename(filepath))[0]
    m = re.search(r"(\d{8})", basename)
    if m:
        return datetime.strptime(m.group(1), "%Y%m%d")
    m = re.search(r"(\d{4})_(\d{2})_(\d{2})", basename)
    if m:
        return datetime.strptime(f"{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%m%d")
    raise ValueError(f"Could not parse date from: {basename!r}")

print(f"\nDate parse test: {os.path.basename(tif_files[0])} -> {parse_date(tif_files[0]).date()}")
dates = [parse_date(f) for f in tif_files]
print(f"Date range: {dates[0].date()} – {dates[-1].date()} ({len(dates)} files)")

dupes = pd.Series(dates)[pd.Series(dates).duplicated()]
if len(dupes):
    print(f"WARNING: {len(dupes)} duplicate dates:\n{dupes.values}")

# --- 0.6 Spatial validation (sample) ---
print("Spatial validation (100 random files)...")
errors = []
for f in random.sample(tif_files, min(100, len(tif_files))):
    with rasterio.open(f) as src:
        if src.count != N_BANDS:          errors.append(f"{f}: bands {src.count}")
        if src.crs != ref_crs:            errors.append(f"{f}: CRS mismatch")
        if (src.height, src.width) != ref_shape: errors.append(f"{f}: shape mismatch")
if errors:
    for e in errors[:5]: print(f"  ERROR: {e}")
    raise RuntimeError("Spatial inconsistencies detected.")
print(f"  All sampled files passed.")

# Coordinate arrays
t   = ref_transform
lats = np.array([t.f + t.e * i for i in range(nrows)])
lons = np.array([t.c + t.a * j for j in range(ncols)])
print(f"\nGrid: {nrows}x{ncols}  lat [{lats[-1]:.2f}, {lats[0]:.2f}]  lon [{lons[0]:.2f}, {lons[-1]:.2f}]")
print(f"Step 0 complete — {elapsed()}")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Characteristic hail size per cell per day
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 1 — Daily characteristic hail stack")
print("="*60)

def compute_characteristic_hail(filepath, bin_midpoints, method="max_active_bin"):
    with rasterio.open(filepath) as src:
        data = src.read().astype(np.float32)
    mid = bin_midpoints.reshape(len(bin_midpoints), 1, 1)
    if method == "max_active_bin":
        return np.nanmax(np.where(data > 0, mid, 0.0), axis=0).astype(np.float32)
    elif method == "weighted_mean":
        band_sum = data.sum(axis=0)
        return np.where(
            band_sum > 0,
            (data * mid).sum(axis=0) / np.where(band_sum > 0, band_sum, 1),
            0.0
        ).astype(np.float32)
    raise ValueError(f"Unknown BAND_METHOD: {method!r}")

nc_path = os.path.join(ROOT, "char_hail_daily.nc")
rebuild = True
if os.path.exists(nc_path):
    print(f"  Found existing {os.path.basename(nc_path)}, checking...")
    try:
        _test = xr.open_dataarray(nc_path)
        if len(_test.time) == len(tif_files):
            print(f"  Matches file count ({len(tif_files)}) — loading")
            rebuild = False
        else:
            print(f"  Mismatch ({len(_test.time)} vs {len(tif_files)}) — rebuilding")
        _test.close()
    except Exception as e:
        print(f"  Load failed ({e}) — rebuilding")

if rebuild:
    t1 = time.time()
    print(f"  Building stack from {len(tif_files)} files...")
    stack = []
    for i, f in enumerate(tif_files):
        stack.append(compute_characteristic_hail(f, bin_midpoints, BAND_METHOD))
        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(tif_files)}  ({time.time()-t1:.0f}s)")
    hail_da = xr.DataArray(
        np.stack(stack, axis=0),
        dims=["time", "lat", "lon"],
        coords={"time": dates, "lat": lats, "lon": lons},
        attrs={
            "units":       "inches",
            "description": "Characteristic hail — max active bin midpoint",
            "source":      ROOT,
            "band_method": BAND_METHOD,
        }
    )
    hail_da.to_netcdf(nc_path)
    print(f"  Saved char_hail_daily.nc  ({time.time()-t1:.0f}s)")
else:
    hail_da = xr.open_dataarray(nc_path)

print(f"  Shape: {hail_da.shape}  max={float(hail_da.max()):.2f} in  min_nonzero>0: {float(hail_da.where(hail_da>0).min()):.2f} in")

# Load into memory for fast event detection
print("  Loading into memory...")
hail_da = hail_da.load()
print(f"Step 1 complete — {elapsed()}")

# ═══════════════════════════════════════════════════════════════
# STEP 2 — Multi-day storm event identification
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 2 — Event identification")
print("="*60)

DAMAGE_THRESHOLD_IN = 1.0  # residential asphalt shingles

# 2.2 Temporal clustering
daily_any_hail = (hail_da >= DAMAGE_THRESHOLD_IN).any(dim=["lat", "lon"]).values

event_windows = []
in_event, start_idx = False, None
for i, active in enumerate(daily_any_hail):
    if active and not in_event:
        in_event, start_idx = True, i
    elif not active and in_event:
        in_event = False
        event_windows.append((start_idx, i - 1))
if in_event:
    event_windows.append((start_idx, len(daily_any_hail) - 1))

print(f"  Candidate temporal windows:  {len(event_windows)}")
if event_windows:
    durations = [e - s + 1 for s, e in event_windows]
    print(f"  Max window duration (days):  {max(durations)}")

# 2.3 Spatial continuity — split disconnected windows
def footprints_overlap(fp1, fp2, buffer_cells=2):
    return np.any(binary_dilation(fp1, iterations=buffer_cells) & fp2)

def split_event_window(start, end, hail_da, threshold, buffer_cells=2):
    sub_events, current_start = [], start
    for i in range(start, end):
        fp_today    = (hail_da.isel(time=i).values   >= threshold)
        fp_tomorrow = (hail_da.isel(time=i+1).values >= threshold)
        if not footprints_overlap(fp_today, fp_tomorrow, buffer_cells):
            sub_events.append((current_start, i))
            current_start = i + 1
    sub_events.append((current_start, end))
    return sub_events

print("  Applying spatial continuity check...")
final_events = []
for start, end in event_windows:
    if start == end:
        final_events.append((start, end))
    else:
        final_events.extend(split_event_window(start, end, hail_da, DAMAGE_THRESHOLD_IN))

print(f"  Final events after spatial splitting: {len(final_events)}")

# 2.4 Build event catalog
event_peak_hail, event_records = [], []

for event_id, (s, e) in enumerate(final_events):
    peak      = hail_da.isel(time=slice(s, e + 1)).max(dim="time").values
    footprint = peak >= DAMAGE_THRESHOLD_IN
    n_cells   = int(footprint.sum())
    if n_cells == 0:
        continue
    event_peak_hail.append(peak)
    event_records.append({
        "event_id":           event_id,
        "start_date":         dates[s],
        "end_date":           dates[e],
        "duration_days":      e - s + 1,
        "n_active_cells":     n_cells,
        "footprint_area_km2": n_cells * (0.25 * 111) ** 2,
        "peak_hail_max_in":   float(peak.max()),
        "peak_hail_mean_in":  float(peak[footprint].mean()),
    })

event_df         = pd.DataFrame(event_records)
event_peak_array = np.stack(event_peak_hail, axis=0)  # (n_events, nrows, ncols)

event_df.to_csv(os.path.join(ROOT, "event_catalog.csv"), index=False)
np.save(os.path.join(ROOT, "event_peak_array.npy"), event_peak_array)
print(f"\n  Event catalog: {len(event_df)} events over {event_df['start_date'].dt.year.nunique()} years")
print(event_df[["start_date","end_date","duration_days","n_active_cells","peak_hail_max_in"]].describe().to_string())

# 2.5 Quality checks
print("\n  Event duration distribution:")
print(event_df["duration_days"].value_counts().sort_index().to_string())
print("\n  Annual event counts:")
print(event_df.groupby(event_df["start_date"].dt.year).size().to_string())
assert event_df["peak_hail_max_in"].max() < 10.5, "Implausible peak hail — check bins"
print(f"\nStep 2 complete — {elapsed()}")

# ═══════════════════════════════════════════════════════════════
# STEP 3 — Fit primary hazard CDFs per grid cell
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 3 — CDF fitting")
print("="*60)

# 3.1 Annual maximum hail series
event_years  = [row["start_date"].year for _, row in event_df.iterrows()]
unique_years = sorted(set(event_years))
n_years      = len(unique_years)

annual_max = np.zeros((n_years, nrows, ncols), dtype=np.float32)
for yi, year in enumerate(unique_years):
    year_mask       = np.array([y == year for y in event_years])
    annual_max[yi]  = event_peak_array[year_mask].max(axis=0)

print(f"  Annual max array: {annual_max.shape}  ({n_years} years: {unique_years[0]}–{unique_years[-1]})")

# 3.2 Zero-inflated occurrence model
def fit_zero_inflated_cell(series):
    nz = series[series > 0]
    return len(nz) / len(series), (nz if len(nz) >= 5 else None)

# 3.3 Lognormal body + GPD tail (L-moments with scipy fallback)
GPD_THRESHOLD_IN = 2.0

def fit_cell_distribution(nonzero_series, tail_threshold=GPD_THRESHOLD_IN):
    result = {}
    # Lognormal body
    shape, loc, scale = stats.lognorm.fit(nonzero_series, floc=0)
    result["lognormal"] = {"shape": shape, "loc": loc, "scale": scale}
    # GPD tail
    exceedances = nonzero_series[nonzero_series > tail_threshold] - tail_threshold
    if len(exceedances) >= 5:
        try:
            from lmoments3 import distr as lm3
            p = lm3.gpa.lmom_fit(exceedances)
            result["gpd"] = {
                "xi":           float(p["c"]),
                "loc":          float(p.get("loc", 0.0)),
                "sigma":        float(p["scale"]),
                "threshold":    tail_threshold,
                "n_exceedances": len(exceedances),
                "rate":         len(exceedances) / len(nonzero_series),
            }
        except Exception:
            # Fallback: scipy MLE with loc=0
            xi, _, sigma = stats.genpareto.fit(exceedances, floc=0)
            result["gpd"] = {
                "xi":           float(xi),
                "loc":          0.0,
                "sigma":        float(sigma),
                "threshold":    tail_threshold,
                "n_exceedances": len(exceedances),
                "rate":         len(exceedances) / len(nonzero_series),
            }
    return result

# 3.4 Composite CDF
def composite_cdf(h, p_occ, lgn, gpd=None, tail_threshold=GPD_THRESHOLD_IN):
    if h <= 0:
        return 1.0 - p_occ
    F_body = stats.lognorm.cdf(h, lgn["shape"], loc=lgn["loc"], scale=lgn["scale"])
    if gpd and h > tail_threshold:
        F_thresh = stats.lognorm.cdf(tail_threshold, lgn["shape"], loc=lgn["loc"], scale=lgn["scale"])
        F_gpd    = stats.genpareto.cdf(h - tail_threshold,
                                       gpd["xi"], loc=gpd["loc"], scale=gpd["sigma"])
        F_sev = min(F_thresh + (1.0 - F_thresh) * F_gpd * gpd["rate"], 1.0)
    else:
        F_sev = F_body
    return (1.0 - p_occ) + p_occ * F_sev

# 3.5 Invert CDF for return periods
RETURN_PERIODS = [10, 25, 50, 100, 200, 250, 500]

def hail_at_return_period(T, p_occ, lgn, gpd=None):
    target = 1.0 - 1.0 / T
    if target <= (1.0 - p_occ):
        return 0.0
    try:
        return brentq(lambda h: composite_cdf(h, p_occ, lgn, gpd) - target,
                      1e-6, 20.0, xtol=1e-4)
    except ValueError:
        return np.nan

# 3.6 Apply across all cells
rp_surfaces   = {T: np.full((nrows, ncols), np.nan, dtype=np.float32) for T in RETURN_PERIODS}
p_occ_surface = np.full((nrows, ncols), np.nan, dtype=np.float32)
n_fitted      = 0

print(f"  Fitting CDFs across {nrows}x{ncols} = {nrows*ncols} cells...")
t3 = time.time()
for i in range(nrows):
    for j in range(ncols):
        series = annual_max[:, i, j]
        p_occ, nz = fit_zero_inflated_cell(series)
        p_occ_surface[i, j] = p_occ
        if nz is None:
            continue
        try:
            params = fit_cell_distribution(nz)
            lgn    = params["lognormal"]
            gpd    = params.get("gpd")
            for T in RETURN_PERIODS:
                rp_surfaces[T][i, j] = hail_at_return_period(T, p_occ, lgn, gpd)
            n_fitted += 1
        except Exception:
            pass
    if i % 10 == 0:
        print(f"    Row {i}/{nrows}  ({time.time()-t3:.0f}s)")

print(f"  Fitted {n_fitted} cells  ({time.time()-t3:.0f}s)")

# Write output rasters
with rasterio.open(tif_files[0]) as src:
    out_profile = src.profile.copy()
    out_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="lzw", predictor=3)

for T, surface in rp_surfaces.items():
    out  = np.where(np.isnan(surface), -9999.0, surface).astype(np.float32)
    path = os.path.join(ROOT, f"rp_{T}yr_hail.tif")
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(out, 1)
        dst.update_tags(1, return_period_years=str(T), units="inches",
                        description=f"{T}-year return period hail size (composite lognormal+GPD)")
    print(f"  Written rp_{T}yr_hail.tif  |  max={out[out!=-9999].max():.2f} in  "
          f"p90={np.percentile(out[out!=-9999], 90):.2f} in")

p_occ_out = np.where(np.isnan(p_occ_surface), -9999.0, p_occ_surface).astype(np.float32)
with rasterio.open(os.path.join(ROOT, "p_occurrence.tif"), "w", **out_profile) as dst:
    dst.write(p_occ_out, 1)
print(f"  Written p_occurrence.tif")
print(f"Step 3 complete — {elapsed()}")

# ═══════════════════════════════════════════════════════════════
# STEP 4 — Spatial correlation
# ═══════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("STEP 4 — Spatial correlation")
print("="*60)

MAX_CORR_CELLS = 800  # cap for Spearman matrix computation

flat_max    = annual_max.reshape(n_years, nrows * ncols)
active_mask = (flat_max > 0).any(axis=0)
active_idx  = np.where(active_mask)[0]
print(f"  Active cells (any hail in record): {len(active_idx)}")

if len(active_idx) > MAX_CORR_CELLS:
    rng      = np.random.default_rng(42)
    sel      = rng.choice(len(active_idx), MAX_CORR_CELLS, replace=False)
    corr_idx = active_idx[np.sort(sel)]
    print(f"  Subsampled to {MAX_CORR_CELLS} cells for correlation computation")
else:
    corr_idx = active_idx

flat_corr  = flat_max[:, corr_idx]      # (n_years, n_corr)
corr_lats  = np.array([lats[i // ncols] for i in corr_idx])
corr_lons  = np.array([lons[i %  ncols] for i in corr_idx])

# 4.1 Spearman correlation matrix
print("  Computing Spearman correlation matrix...")
t4 = time.time()
sr = scipy_spearmanr(flat_corr)
# scipy 1.9+: SpearmanrResult object; older: (corr, pval) tuple
if hasattr(sr, "statistic"):
    corr_matrix = np.array(sr.statistic)
else:
    corr_matrix = np.array(sr[0])

if corr_matrix.ndim == 0:  # scalar edge case (2 columns)
    v = float(corr_matrix)
    corr_matrix = np.array([[1.0, v], [v, 1.0]])

print(f"  Correlation matrix: {corr_matrix.shape}  ({time.time()-t4:.0f}s)")

# 4.2 Distance matrix + exponential decay fit
coords_km   = np.column_stack([
    corr_lats * 111.0,
    corr_lons * 111.0 * np.cos(np.radians(corr_lats.mean()))
])
dist_matrix = cdist(coords_km, coords_km)

tri_idx        = np.triu_indices(len(corr_idx), k=1)
distances_flat = dist_matrix[tri_idx]
corr_flat      = corr_matrix[tri_idx]

# Subsample pairs for curve_fit if huge
n_pairs = len(distances_flat)
if n_pairs > 200_000:
    rng2   = np.random.default_rng(99)
    pi     = rng2.choice(n_pairs, 200_000, replace=False)
    d_fit, c_fit = distances_flat[pi], corr_flat[pi]
else:
    d_fit, c_fit = distances_flat, corr_flat

def exp_decay(d, lam):
    return np.exp(-d / lam)

popt, _   = curve_fit(exp_decay, d_fit, c_fit, p0=[200.0], bounds=(10, 2000))
lambda_km = float(popt[0])
print(f"  Fitted decorrelation length lambda = {lambda_km:.1f} km")

if lambda_km < 100:
    print("  WARNING: lambda < 100 km — may indicate noise or sparse data")
elif lambda_km > 600:
    print("  WARNING: lambda > 600 km — may indicate spatial smoothing artifact")
else:
    print(f"  lambda in expected range (100–600 km)")

with open(os.path.join(ROOT, "lambda_km.json"), "w") as f:
    json.dump({
        "lambda_km":          lambda_km,
        "n_active_cells":     int(len(active_idx)),
        "n_corr_cells":       int(len(corr_idx)),
        "subsampled":         bool(len(active_idx) > MAX_CORR_CELLS),
        "n_years":            n_years,
        "damage_threshold_in": DAMAGE_THRESHOLD_IN,
        "gpd_threshold_in":   GPD_THRESHOLD_IN,
    }, f, indent=2)
print(f"  Saved lambda_km.json")

# 4.3 Smooth model correlation matrix with PSD correction
model_corr = exp_decay(dist_matrix, lambda_km)
np.fill_diagonal(model_corr, 1.0)

eigvals = np.linalg.eigvalsh(model_corr)
if eigvals.min() < 0:
    eigvals2, eigvecs = np.linalg.eigh(model_corr)
    eigvals2          = np.maximum(eigvals2, 1e-8)
    model_corr        = eigvecs @ np.diag(eigvals2) @ eigvecs.T
    np.fill_diagonal(model_corr, 1.0)
    print("  Applied nearest-PSD correction")
else:
    print(f"  Model corr matrix is PSD (min eigval = {eigvals.min():.4f})")

# 4.4 Cholesky factor for simulation
print("  Computing Cholesky decomposition...")
t4c = time.time()
try:
    L = np.linalg.cholesky(model_corr)
    print(f"  Cholesky done in {time.time()-t4c:.0f}s  shape: {L.shape}")
    np.save(os.path.join(ROOT, "cholesky_L.npy"),      L)
    np.save(os.path.join(ROOT, "corr_cell_idx.npy"),   corr_idx)
    print(f"  Saved cholesky_L.npy ({L.shape})")
    print(f"  Saved corr_cell_idx.npy ({len(corr_idx)} cells)")
except np.linalg.LinAlgError as e:
    print(f"  WARNING: Cholesky failed: {e}")

# 4.5 Validation plot
dist_bins   = np.arange(0, 2001, 100)
bin_centers = (dist_bins[:-1] + dist_bins[1:]) / 2.0
emp_means, mod_means = [], []
for lo, hi in zip(dist_bins[:-1], dist_bins[1:]):
    mask = (distances_flat >= lo) & (distances_flat < hi)
    if mask.sum() > 0:
        emp_means.append(float(corr_flat[mask].mean()))
        mod_means.append(float(exp_decay(bin_centers[len(emp_means)-1], lambda_km)))

fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(bin_centers[:len(emp_means)], emp_means, s=10, alpha=0.5,
           label="Empirical (binned mean)")
ax.plot(bin_centers[:len(mod_means)], mod_means, color="red", linewidth=1.5,
        label=f"Exp decay lambda = {lambda_km:.0f} km")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Spearman correlation")
ax.set_title("Spatial correlation decay — empirical vs model")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "correlation_decay_fit.png"), dpi=150)
print("  Saved correlation_decay_fit.png")

historical_active = [(annual_max[yi] >= DAMAGE_THRESHOLD_IN).sum() for yi in range(n_years)]
print(f"  Historical active-cell count: "
      f"mean={np.mean(historical_active):.0f}  var={np.var(historical_active):.1f}")
print(f"Step 4 complete — {elapsed()}")

# ═══════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════

total_min = (time.time() - T0) / 60
print("\n" + "="*60)
print(f"PIPELINE COMPLETE — {total_min:.1f} min")
print("="*60)
print(f"\nOutputs in {ROOT}/")
print(f"  char_hail_daily.nc           — {hail_da.shape[0]} days x {nrows}x{ncols}")
print(f"  event_catalog.csv            — {len(event_df)} events, {n_years} years")
print(f"  p_occurrence.tif             — annual hail occurrence probability")
for T in RETURN_PERIODS:
    rp = rp_surfaces[T]
    nv = rp[~np.isnan(rp)]
    print(f"  rp_{T:3d}yr_hail.tif           — max={nv.max():.2f} in  p90={np.percentile(nv,90):.2f} in")
print(f"  bin_midpoints.json")
print(f"  lambda_km.json               — lambda = {lambda_km:.1f} km")
print(f"  cholesky_L.npy               — {len(corr_idx)}x{len(corr_idx)} Cholesky factor")
print(f"  corr_cell_idx.npy            — active cell indices for copula")
print(f"  correlation_decay_fit.png")
print(f"\nKey flags for review:")
print(f"  BAND_METHOD   = {BAND_METHOD}")
print(f"  Damage thresh = {DAMAGE_THRESHOLD_IN} in")
print(f"  GPD thresh    = {GPD_THRESHOLD_IN} in")
print(f"  lambda        = {lambda_km:.1f} km")
print(f"  n_events      = {len(event_df)}")
print(f"  n_years       = {n_years}")
