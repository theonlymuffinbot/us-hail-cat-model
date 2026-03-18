#!/usr/bin/env python3
"""
15_stochastic_maps.py  —  Per-cell stochastic hazard maps (event-resampling)
===========================================================================
Runs a lean 2,000-year simulation using event-resampling (same methodology
as stage 14) to produce per-cell return period maps and occurrence probability
maps.

Methodology (matches stage 14):
  1. Load event_catalog.csv + event_peak_array.npy (historical templates)
  2. λ = n_hist_events / n_years_of_record
  3. Per simulated year:
     - Draw N ~ Poisson(λ)
     - For each event: resample template weighted by exp(-|doy_diff|/30)
     - Apply log-normal intensity perturbation (σ=0.15)
     - Accumulate per-cell annual max hail
  4. Build return period maps and p_occ maps from 2,000 annual max arrays

Outputs: data/stochastic/maps/
  stoch_rp_{10,25,50,100,200,500}yr_hail.tif
  stoch_p_occurrence.tif  (= stoch_p_occ_1p00in.tif, for compatibility)
  stoch_p_occ_{0p25,...,5p00}in.tif

Comparison figures: docs/figures/maps/
  stoch_vs_hist_rp_100yr_comparison.png
  stoch_vs_hist_p_occ_1p00in_comparison.png

Memory: ~250 MB peak   Runtime: ~3-8 min
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import warnings; warnings.filterwarnings("ignore")

REPO  = Path(__file__).resolve().parent.parent
DATA  = REPO / "data"
STOCH = DATA / "stochastic"
HIST  = DATA / "hail_0.25deg"
OUT_D = STOCH / "maps";           OUT_D.mkdir(parents=True, exist_ok=True)
OUT_F = REPO / "docs/figures/maps"; OUT_F.mkdir(parents=True, exist_ok=True)

N_SIM         = 2_000
SIGMA_PERTURB = 0.15
RNG_SEED      = 42
THRESH        = 0.25          # minimum hail size to count (inches)
RP_YRS        = [10, 25, 50, 100, 200, 500]
POCC_T        = [0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]
MAX_HAIL_PHYS = 10.0          # physical ceiling (inches)

NROWS, NCOLS = 104, 236
LAT_MAX, LAT_MIN = 48.875, 22.875
LON_MIN, LON_MAX = -124.875, -65.125
CS           = 0.25
NODATA       = -9999.0


# ── Output filenames ──────────────────────────────────────────────────────────
def _pocc_tag(t):
    return f"{t:.2f}".replace(".", "p")


def validate_outputs() -> bool:
    """Validate all expected output TIFs exist and are readable."""
    errors = []
    rp_fnames   = [f"stoch_rp_{rp}yr_hail.tif"        for rp in RP_YRS]
    pocc_fnames = [f"stoch_p_occ_{_pocc_tag(t)}in.tif" for t  in POCC_T]
    expected    = rp_fnames + ["stoch_p_occurrence.tif"] + pocc_fnames

    for fname in expected:
        p = OUT_D / fname
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
        print("CRITICAL: Output validation FAILED:")
        for e in errors:
            print(f"  x {e}")
        return False
    print("Output validation passed")
    return True


# ── --validate early exit ─────────────────────────────────────────────────────
if "--validate" in sys.argv:
    ok = validate_outputs()
    sys.exit(0 if ok else 1)


# ── Raster helpers ────────────────────────────────────────────────────────────
def write_tif(arr, path):
    tr = from_bounds(LON_MIN - CS / 2, LAT_MIN - CS / 2,
                     LON_MAX + CS / 2, LAT_MAX + CS / 2, NCOLS, NROWS)
    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                       width=NCOLS, height=NROWS, count=1, crs="EPSG:4326",
                       transform=tr, nodata=NODATA, compress="lzw") as dst:
        dst.write(arr.astype(np.float32), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load inputs
# ═══════════════════════════════════════════════════════════════════════════════
print("=== Loading inputs ===", flush=True)

ec = pd.read_csv(HIST / "event_catalog.csv", parse_dates=["start_date", "end_date"])
n_hist_events  = len(ec)
n_years_record = int(ec["start_date"].dt.year.nunique())
lambda_events  = n_hist_events / n_years_record
event_doys     = ec["start_date"].dt.dayofyear.values.astype(np.int32)

print(f"  Events: {n_hist_events:,} over {n_years_record} years, "
      f"lambda={lambda_events:.2f} ev/yr", flush=True)

print("  Loading event_peak_array.npy ...", flush=True)
event_peak_array = np.load(HIST / "event_peak_array.npy")   # (n_events, 104, 236)
assert event_peak_array.shape == (n_hist_events, NROWS, NCOLS), \
    f"Unexpected shape {event_peak_array.shape}"
print(f"  event_peak_array: {event_peak_array.shape}", flush=True)

# Active cell flat indices (from stage 14 output)
act_idx = np.load(STOCH / "active_flat_idx.npy")            # (N_ACT,)
N_ACT   = len(act_idx)
print(f"  Active cells: {N_ACT:,}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Seasonal DOY distribution
# ═══════════════════════════════════════════════════════════════════════════════
from scipy.ndimage import gaussian_filter1d

daily_count = np.zeros(366, dtype=np.float64)
for d in event_doys:
    daily_count[min(d, 366) - 1] += 1.0
padded     = np.tile(daily_count, 3)
smoothed   = gaussian_filter1d(padded, sigma=10.0)
daily_prob = smoothed[366:732]
daily_prob /= daily_prob.sum()
doy_cdf    = np.cumsum(daily_prob)
doy_values = np.arange(1, 367)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Simulate 2,000 years — event-resampling
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n=== Simulating {N_SIM:,} years (event-resampling) ===", flush=True)

rng     = np.random.default_rng(RNG_SEED)
# Per-cell annual max: (N_SIM, N_ACT)  ~100 MB
ann_max = np.zeros((N_SIM, N_ACT), dtype=np.float32)

for yr in range(N_SIM):
    if yr % 200 == 0:
        print(f"  Year {yr}/{N_SIM}...", flush=True)

    n_evt = int(rng.poisson(lambda_events))
    if n_evt == 0:
        continue

    year_max = np.zeros(N_ACT, dtype=np.float32)

    # Draw all event DOYs for this year at once
    u_date  = rng.random(n_evt)
    ev_doys = doy_values[np.searchsorted(doy_cdf, u_date)]

    for ev_i in range(n_evt):
        doy = int(ev_doys[ev_i])

        # Seasonal template weights
        doy_diff = np.abs(event_doys - doy)
        doy_diff = np.minimum(doy_diff, 366 - doy_diff)
        weights  = np.exp(-doy_diff / 30.0)
        weights /= weights.sum()

        # Resample historical template
        tmpl_idx     = int(rng.choice(n_hist_events, p=weights))
        scale_factor = float(np.exp(SIGMA_PERTURB * rng.standard_normal()))

        # Extract active-cell hail values and apply perturbation
        hail_flat = event_peak_array[tmpl_idx].ravel()[act_idx] * scale_factor
        np.clip(hail_flat, 0.0, MAX_HAIL_PHYS, out=hail_flat)

        np.maximum(year_max, hail_flat, out=year_max)

    ann_max[yr] = year_max

print("  Simulation complete.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-cell return period and p_occ maps
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Computing per-cell statistics ===", flush=True)

rp_g    = {rp: np.full((NROWS, NCOLS), NODATA, np.float32) for rp in RP_YRS}
poc_g   = {t:  np.full((NROWS, NCOLS), NODATA, np.float32) for t  in POCC_T}

for i, flat in enumerate(act_idx):
    row, col = int(flat // NCOLS), int(flat % NCOLS)
    ann      = ann_max[:, i]                          # (N_SIM,)

    for t in POCC_T:
        poc_g[t][row, col] = float(np.mean(ann >= t))

    ann_sorted = np.sort(ann)[::-1]                   # descending
    for rp in RP_YRS:
        rank = int(N_SIM / rp)
        rank = max(0, min(rank, N_SIM - 1))
        q    = float(ann_sorted[rank])
        rp_g[rp][row, col] = q if q >= THRESH else NODATA

del ann_max

# Apply CONUS mask from historical 100yr map
if (HIST / "rp_100yr_hail.tif").exists():
    with rasterio.open(HIST / "rp_100yr_hail.tif") as src:
        hm  = src.read(1)
        hnd = src.nodata if src.nodata is not None else -9999
    ocean = (hm == hnd) | (hm <= 0)
    for g in list(rp_g.values()) + list(poc_g.values()):
        g[ocean] = NODATA

# Sanity check
valid_rp = int(np.sum(rp_g[100] != NODATA))
max_rp   = float(np.max(rp_g[100][rp_g[100] != NODATA])) if valid_rp > 0 else 0
print(f"  rp100: valid={valid_rp} max={max_rp:.2f}in", flush=True)
print(f"  p_occ_1in: valid={int(np.sum(poc_g[1.0] != NODATA))}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Write GeoTIFFs
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Writing GeoTIFFs ===", flush=True)

for rp in RP_YRS:
    write_tif(rp_g[rp], OUT_D / f"stoch_rp_{rp}yr_hail.tif")
    print(f"  stoch_rp_{rp}yr_hail.tif", flush=True)

for t in POCC_T:
    tag = _pocc_tag(t)
    write_tif(poc_g[t], OUT_D / f"stoch_p_occ_{tag}in.tif")
    print(f"  stoch_p_occ_{tag}in.tif", flush=True)

# stoch_p_occurrence.tif = stoch_p_occ_1p00in.tif (for compatibility)
write_tif(poc_g[1.00], OUT_D / "stoch_p_occurrence.tif")
print("  stoch_p_occurrence.tif", flush=True)

print(f"  Total TIFs: {len(list(OUT_D.glob('*.tif')))}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Comparison figures (stochastic vs historical)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Rendering comparison figures ===", flush=True)

HAIL_LEV = [0, .5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7, 9]
PROB_LEV = [0, .02, .05, .10, .15, .20, .30, .40, .50, .65, .80, 1.0]
HCMAP, PCMAP = plt.cm.YlOrRd, plt.cm.plasma
EXT = [LON_MIN - CS/2, LON_MAX + CS/2, LAT_MIN - CS/2, LAT_MAX + CS/2]

def msk(a):
    return np.ma.masked_where(a == NODATA, a)

def compare(stoch_arr, hist_path, title, cmap, levs, path, cbl):
    if not hist_path.exists():
        print(f"  Skipping (no historical): {hist_path.name}", flush=True)
        return
    with rasterio.open(hist_path) as src:
        ha  = src.read(1).astype(np.float32)
        hnd = src.nodata if src.nodata is not None else -9999
    ha[ha == hnd] = np.nan
    norm = BoundaryNorm(levs, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    kw = dict(extent=EXT, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].imshow(np.ma.masked_invalid(ha), **kw)
    axes[0].set_title(f"Historical ({n_years_record}-yr record)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    im = axes[1].imshow(msk(stoch_arr), **kw)
    axes[1].set_title(f"Stochastic ({N_SIM:,}-yr event-resampling)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    fig.suptitle(f"Historical vs Stochastic — {title}", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, ax=axes.tolist(), orientation="horizontal", pad=0.06, shrink=0.6)
    cb.set_label(cbl, fontsize=11)
    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {path.name}", flush=True)

compare(rp_g[100], HIST / "rp_100yr_hail.tif",
        "100-Year Return Period Hail",
        HCMAP, HAIL_LEV,
        OUT_F / "stoch_vs_hist_rp_100yr_comparison.png",
        "Hail Size (inches)")

compare(poc_g[1.00], HIST / "p_occ_1p00in.tif",
        "Annual P(Hail >= 1.0\")",
        PCMAP, PROB_LEV,
        OUT_F / "stoch_vs_hist_p_occ_1p00in_comparison.png",
        "Annual Probability")

print(f"\n=== Done: {len(list(OUT_D.glob('*.tif')))} TIFs, "
      f"{len(list(OUT_F.glob('*.png')))} PNGs ===", flush=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Validate
# ═══════════════════════════════════════════════════════════════════════════════
if not validate_outputs():
    sys.exit(1)
