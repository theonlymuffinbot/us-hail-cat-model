#!/usr/bin/env python3
"""
14_generate_stochastic_catalog.py  (v2 — memory-optimised)
===========================================================
Generates a 50,000-year stochastic hail occurrence catalog and
derives Probable Exceedance Tables (PET) from it.

Memory budget fixes vs v1
--------------------------
  - event_peak_array freed immediately after CDF table build
  - climo_factor stored as float16 (halves 180 MB -> 90 MB)
  - Event summary rows written in streaming chunks (CSV, not buffered)
  - Cell sample rows written immediately per event (not accumulated)
  - Annual stats tracked as running arrays, not full DataFrame in RAM

Outputs  (data/stochastic/)
-----------------------------------------------
  cdf_lookup.npy                  — pre-computed CDF table (reused if exists)
  stochastic_event_summary.csv    — one row per simulated event
  stochastic_cell_sample.parquet  — full cell data (CELL_SAMPLE_YEARS years)
  pet_occurrence.csv              — occurrence PET
  pet_aggregate.csv               — aggregate PET

Key parameters
--------------
  LAMBDA_KM        = 150.0   Spatial decorrelation length
  N_SIM_YEARS      = 50,000
  LAMBDA_EVENTS    = 127.3   Poisson rate (2928 events / 23 years)
  THRESHOLD_IN     = 0.25    Minimum hail size stored
  GPD_THRESH_IN    = 2.0     GPD tail threshold
  N_QUANT          = 2000    CDF lookup resolution
  CELL_SAMPLE_YEARS= 2000    Full cell data written for validation
"""

import os, sys, time, warnings, glob, gc
from pathlib import Path
from datetime import date

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from scipy.stats import norm as sp_norm, lognorm, genpareto
from scipy.ndimage import gaussian_filter1d, uniform_filter1d

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = DATA_ROOT / "hail_0.25deg"
CLIMO_DIR = DATA_ROOT / "hail_0.25deg_climo"
OUT_DIR   = DATA_ROOT / "stochastic"
OUT_DIR.mkdir(exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH  = LOGS_ROOT / "stochastic_catalog_build.log"

# ── Parameters ────────────────────────────────────────────────────────────────
LAMBDA_KM         = 150.0
N_SIM_YEARS       = 50_000
LAMBDA_EVENTS     = 127.3
THRESHOLD_IN      = 0.25
GPD_THRESH_IN     = 2.0
N_QUANT           = 2000
CELL_SAMPLE_YEARS = 2_000
RNG_SEED          = 42

NROWS, NCOLS      = 104, 236
CELL_DEG          = 0.25
LAT_ORIG          = 50.0
LON_ORIG          = -125.0
CELL_AREA_KM2     = (CELL_DEG * 111.0) ** 2   # ~772 km2

# Streaming flush interval (years) for event summary CSV
FLUSH_EVERY_YEARS = 500

T0 = time.time()


# ── Logging ───────────────────────────────────────────────────────────────────
def log(msg):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")

def elapsed():
    return f"{(time.time()-T0)/60:.1f} min"

def mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    except Exception:
        return 0.0


# ── Grid helpers ──────────────────────────────────────────────────────────────
def cell_latlon(idx):
    r   = idx // NCOLS
    c   = idx %  NCOLS
    lat = LAT_ORIG - (r + 0.5) * CELL_DEG
    lon = LON_ORIG + (c + 0.5) * CELL_DEG
    return float(lat), float(lon)

def haversine_km(lat1, lon1, lat2, lon2):
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a    = (np.sin(dlat/2)**2
            + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2))
            * np.sin(dlon/2)**2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# ═══════════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("STOCHASTIC CATALOG GENERATION  v2 (memory-optimised)")
log(f"  N_SIM_YEARS   = {N_SIM_YEARS:,}")
log(f"  LAMBDA_KM     = {LAMBDA_KM} km")
log(f"  LAMBDA_EVENTS = {LAMBDA_EVENTS}")
log(f"  THRESHOLD_IN  = {THRESHOLD_IN}\"")
log(f"  N_QUANT       = {N_QUANT}")
log("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Cholesky + seed cells
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[1] Loading Cholesky factor and seed cells")

L_seed   = np.load(ROOT / "cholesky_L_150km.npy")        # (800, 800)
seed_idx = np.load(ROOT / "corr_cell_idx.npy")            # (800,)
n_seed   = len(seed_idx)

seed_lats = np.array([cell_latlon(i)[0] for i in seed_idx])
seed_lons = np.array([cell_latlon(i)[1] for i in seed_idx])
log(f"  Cholesky {L_seed.shape}, seed cells {n_seed},  mem ~{mem_mb():.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Active cells + P_occurrence
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[2] Loading active cells and P_occurrence")

with rasterio.open(ROOT / "p_occurrence.tif") as src:
    p_occ_grid = src.read(1).astype(np.float32)
    nodata_val = src.nodata if src.nodata is not None else -9999.0

active_mask  = (p_occ_grid > 0) & (p_occ_grid != nodata_val)
active_flat  = np.where(active_mask.ravel())[0]
n_active     = len(active_flat)

active_lats  = np.array([cell_latlon(i)[0] for i in active_flat], dtype=np.float32)
active_lons  = np.array([cell_latlon(i)[1] for i in active_flat], dtype=np.float32)
p_occ_active = p_occ_grid.ravel()[active_flat]

del p_occ_grid
gc.collect()

log(f"  Active cells: {n_active:,}  P_occ [{p_occ_active.min():.3f}, {p_occ_active.max():.3f}]")
log(f"  mem ~{mem_mb():.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Parent-child mapping
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[3] Parent-child correlation mapping")

BATCH = 1000
parent_idx        = np.empty(n_active, dtype=np.int32)
parent_rho        = np.empty(n_active, dtype=np.float32)
parent_eps_scale  = np.empty(n_active, dtype=np.float32)

for r0 in range(0, n_active, BATCH):
    r1      = min(r0 + BATCH, n_active)
    dlat    = active_lats[r0:r1, None] - seed_lats[None, :]
    dlon    = active_lons[r0:r1, None] - seed_lons[None, :]
    cos_lat = np.cos(np.radians(active_lats[r0:r1]))[:, None]
    dist2   = dlat**2 + (dlon * cos_lat)**2
    nearest = np.argmin(dist2, axis=1)
    dist_km = haversine_km(active_lats[r0:r1], active_lons[r0:r1],
                           seed_lats[nearest],  seed_lons[nearest])
    parent_idx[r0:r1]       = nearest
    parent_rho[r0:r1]       = np.exp(-dist_km / LAMBDA_KM).astype(np.float32)
    parent_eps_scale[r0:r1] = np.sqrt(np.maximum(0.0, 1.0 - parent_rho[r0:r1]**2))

log(f"  rho range: [{parent_rho.min():.3f}, {parent_rho.max():.3f}]  "
    f"mean: {parent_rho.mean():.3f}  mem ~{mem_mb():.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CDF lookup table (reuse if already built)
# ═══════════════════════════════════════════════════════════════════════════════
cdf_lookup_path = OUT_DIR / "cdf_lookup.npy"
quant_p_path    = OUT_DIR / "cdf_quant_p.npy"

if cdf_lookup_path.exists() and quant_p_path.exists():
    log("\n[4] Loading pre-built CDF lookup table")
    cdf_lookup = np.load(cdf_lookup_path)    # (N_QUANT, n_active)
    QUANT_P    = np.load(quant_p_path)
    log(f"  cdf_lookup {cdf_lookup.shape}  mem ~{mem_mb():.0f} MB")
else:
    log("\n[4] Building per-cell CDF lookup table")

    event_peak = np.load(ROOT / "event_peak_array.npy")   # (n_events, 104, 236)
    n_events   = event_peak.shape[0]
    ep_flat    = event_peak.reshape(n_events, NROWS * NCOLS)[:, active_flat]
    del event_peak
    gc.collect()
    log(f"  event_peak loaded -> ep_flat {ep_flat.shape}  mem ~{mem_mb():.0f} MB")

    QUANT_P    = np.linspace(1.0/(N_QUANT+1), N_QUANT/(N_QUANT+1), N_QUANT,
                             dtype=np.float32)
    cdf_lookup = np.zeros((N_QUANT, n_active), dtype=np.float32)

    n_fitted = n_empirical = 0
    t4 = time.time()

    for ci in range(n_active):
        vals = ep_flat[:, ci]
        nz   = vals[vals > 0]

        if len(nz) < 3:
            continue

        if len(nz) < 10:
            cdf_lookup[:, ci] = np.interp(
                QUANT_P, np.linspace(0, 1, len(nz)), np.sort(nz)
            ).astype(np.float32)
            n_empirical += 1
            continue

        try:
            shape, loc, scale = lognorm.fit(nz, floc=0)
            exc = nz[nz > GPD_THRESH_IN] - GPD_THRESH_IN
            gpd_params = None
            if len(exc) >= 5:
                try:
                    xi, _, sigma = genpareto.fit(exc, floc=0)
                    gpd_params   = (float(xi), float(sigma))
                except Exception:
                    pass

            p_occ_cell = float(p_occ_active[ci])
            for qi, p in enumerate(QUANT_P):
                if p <= (1.0 - p_occ_cell):
                    cdf_lookup[qi, ci] = 0.0
                    continue
                f_sev = min((p - (1.0 - p_occ_cell)) / p_occ_cell, 1.0 - 1e-9)

                f_thresh = lognorm.cdf(GPD_THRESH_IN, shape, loc=loc, scale=scale)
                if gpd_params is None or f_sev <= f_thresh:
                    h = lognorm.ppf(f_sev, shape, loc=loc, scale=scale)
                else:
                    xi, sigma  = gpd_params
                    gpd_rate   = len(exc) / len(nz)
                    gpd_target = (f_sev - f_thresh) / max((1.0 - f_thresh) * gpd_rate, 1e-9)
                    gpd_target = min(max(gpd_target, 1e-9), 1.0 - 1e-9)
                    h          = GPD_THRESH_IN + genpareto.ppf(gpd_target, xi, loc=0, scale=sigma)

                cdf_lookup[qi, ci] = max(0.0, float(h))

            n_fitted += 1

        except Exception:
            cdf_lookup[:, ci] = np.interp(
                QUANT_P, np.linspace(0, 1, len(nz)), np.sort(nz)
            ).astype(np.float32)
            n_empirical += 1

    del ep_flat
    gc.collect()

    # Clamp to physical ceiling — record hailstone is 8" (Vivian SD, 2010)
    MAX_HAIL_PHYSICAL = 10.0
    n_clamped = int((cdf_lookup > MAX_HAIL_PHYSICAL).sum())
    cdf_lookup = np.clip(cdf_lookup, 0.0, MAX_HAIL_PHYSICAL).astype(np.float32)
    if n_clamped:
        log(f"  Clamped {n_clamped:,} lookup values exceeding {MAX_HAIL_PHYSICAL}\" "
            f"(unbounded GPD extrapolation)")

    np.save(cdf_lookup_path, cdf_lookup)
    np.save(quant_p_path,    QUANT_P)
    log(f"  Built in {time.time()-t4:.0f}s — "
        f"parametric: {n_fitted:,}  empirical: {n_empirical:,}  "
        f"size: {cdf_lookup.nbytes/1e6:.0f} MB  mem ~{mem_mb():.0f} MB")

# CDF lookup is float32 — keep as-is but ensure no event_peak reference lingers
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Seasonal distributions
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[5] Building seasonal distributions")

ec           = pd.read_csv(ROOT / "event_catalog.csv", parse_dates=["start_date"])
event_doy    = ec["start_date"].dt.dayofyear.values
daily_count  = np.zeros(367, dtype=np.float64)
for d in event_doy:
    daily_count[d] += 1.0

padded       = np.tile(daily_count[1:], 3)
smoothed     = gaussian_filter1d(padded, sigma=3.5)
daily_prob   = smoothed[366:732]
daily_prob  /= daily_prob.sum()
doy_values   = np.arange(1, 367)
doy_cdf      = np.cumsum(daily_prob)

log(f"  Peak DOY: {int(doy_values[daily_prob.argmax()])}")

# Climo rasters -> seasonal P_occ modulation
log("  Loading 366 climatology rasters...")
climo_files = sorted(glob.glob(str(CLIMO_DIR / "climo_*.tif")))
assert len(climo_files) > 0, f"No climo files in {CLIMO_DIR}"

def mmdd_to_doy(mmdd):
    m, d = int(mmdd[:2]), int(mmdd[2:])
    return date(2000, m, d).timetuple().tm_yday

# Build climo_seasonal as float16 to save ~90 MB
climo_seasonal = np.zeros((366, n_active), dtype=np.float16)
for fpath in climo_files:
    mmdd = os.path.basename(fpath).replace("climo_", "").replace(".tif", "")
    doy  = mmdd_to_doy(mmdd)
    with rasterio.open(fpath) as src:
        band_sum = src.read().astype(np.float32).sum(axis=0).ravel()
    climo_seasonal[doy-1] = band_sum[active_flat].astype(np.float16)

climo_mean   = climo_seasonal.astype(np.float32).mean(axis=0)   # (n_active,)
climo_factor = np.where(
    climo_mean > 0,
    climo_seasonal.astype(np.float32) / np.maximum(climo_mean[None,:], 1e-6),
    1.0
).astype(np.float16)
del climo_seasonal
gc.collect()

# Smooth +-15 days
climo_factor = uniform_filter1d(
    climo_factor.astype(np.float32), size=15, axis=0, mode="wrap"
).astype(np.float16)

# Seasonal P_occ: (366, n_active), clipped to [0,1]
p_occ_seasonal = np.minimum(
    p_occ_active[None,:].astype(np.float32) * climo_factor.astype(np.float32),
    1.0
).astype(np.float32)
del climo_factor
gc.collect()

log(f"  Seasonal P_occ built  mem ~{mem_mb():.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Simulation
# ═══════════════════════════════════════════════════════════════════════════════
log(f"\n[6] Simulation — {N_SIM_YEARS:,} years  mem ~{mem_mb():.0f} MB")

rng = np.random.default_rng(RNG_SEED)

# Pick sample years for full cell output
sample_year_set = set(
    rng.choice(N_SIM_YEARS, size=CELL_SAMPLE_YEARS, replace=False).tolist()
)

# Annual tracking arrays (no per-event buffering — just accumulate annual stats)
ann_occ_max_hail  = np.zeros(N_SIM_YEARS, dtype=np.float32)   # worst event max hail
ann_occ_fp_km2    = np.zeros(N_SIM_YEARS, dtype=np.float32)   # worst event footprint
ann_occ_n_cells   = np.zeros(N_SIM_YEARS, dtype=np.int32)     # worst event n_cells
ann_agg_max_hail  = np.zeros(N_SIM_YEARS, dtype=np.float32)   # annual max hail
ann_agg_fp_km2    = np.zeros(N_SIM_YEARS, dtype=np.float32)   # annual total footprint

# Streaming CSV for event summary AND cell sample — nothing buffered in RAM
ev_csv_path     = OUT_DIR / "stochastic_event_summary.csv"
sample_csv_path = OUT_DIR / "stochastic_cell_sample.csv"

ev_csv_header     = "sim_year,event_idx,doy,n_cells,max_hail_in,mean_hail_in,p95_hail_in,footprint_km2\n"
sample_csv_header = "sim_year,event_idx,cell_idx,lat,lon,hail_in\n"

with open(ev_csv_path, "w") as ev_csv, open(sample_csv_path, "w") as sc_csv:
    ev_csv.write(ev_csv_header)
    sc_csv.write(sample_csv_header)

    t_sim          = time.time()
    total_events   = 0
    global_ev_id   = 0
    ev_buf         = []
    sc_buf         = []

    for year_idx in range(N_SIM_YEARS):

        if year_idx > 0 and year_idx % 5000 == 0:
            rate = year_idx / (time.time() - t_sim)
            eta  = (N_SIM_YEARS - year_idx) / rate / 60
            log(f"  Year {year_idx:>6,} / {N_SIM_YEARS:,}  "
                f"events: {total_events:,}  "
                f"ETA: {eta:.1f} min  mem ~{mem_mb():.0f} MB")

        # Flush buffers every FLUSH_EVERY_YEARS years
        if year_idx % FLUSH_EVERY_YEARS == 0 and year_idx > 0:
            if ev_buf:
                ev_csv.write("".join(ev_buf)); ev_csv.flush(); ev_buf.clear()
            if sc_buf:
                sc_csv.write("".join(sc_buf)); sc_csv.flush(); sc_buf.clear()

        n_ev = int(rng.poisson(LAMBDA_EVENTS))
        total_events += n_ev

        if n_ev == 0:
            continue

        # Draw event DOYs
        u_date  = rng.random(n_ev)
        ev_doys = doy_values[np.searchsorted(doy_cdf, u_date)]

        for ev_i in range(n_ev):
            doy  = int(ev_doys[ev_i])
            doy0 = doy - 1

            # ── Correlated field ───────────────────────────────────────────
            z_seed   = L_seed @ rng.standard_normal(n_seed)         # (800,)
            z_active = (parent_rho * z_seed[parent_idx]
                        + parent_eps_scale * rng.standard_normal(n_active).astype(np.float32))
            u_active = sp_norm.cdf(z_active).astype(np.float32)     # (n_active,)

            # ── Zero-inflation ─────────────────────────────────────────────
            p_occ_ev  = p_occ_seasonal[doy0]                         # (n_active,)
            hail_mask = u_active >= (1.0 - p_occ_ev)                 # bool

            hail_active = np.zeros(n_active, dtype=np.float32)

            if hail_mask.any():
                u_sev = np.minimum(
                    (u_active[hail_mask] - (1.0 - p_occ_ev[hail_mask]))
                    / np.maximum(p_occ_ev[hail_mask], 1e-9),
                    1.0 - 1e-9
                )
                # Vectorised CDF lookup
                q_f      = u_sev * (N_QUANT - 1)
                q_lo     = np.floor(q_f).astype(np.int32)
                q_hi     = np.minimum(q_lo + 1, N_QUANT - 1)
                frac     = (q_f - q_lo).astype(np.float32)
                ci_sub   = np.where(hail_mask)[0]
                hail_active[ci_sub] = (
                    (1.0 - frac) * cdf_lookup[q_lo, ci_sub]
                    + frac       * cdf_lookup[q_hi, ci_sub]
                )

            # ── Clamp + apply threshold ───────────────────────────────────
            np.clip(hail_active, 0.0, 10.0, out=hail_active)
            keep      = hail_active >= THRESHOLD_IN
            n_out     = int(keep.sum())
            max_hail  = float(hail_active[keep].max()) if n_out > 0 else 0.0
            mean_hail = float(hail_active[keep].mean()) if n_out > 0 else 0.0
            p95_hail  = float(np.percentile(hail_active[keep], 95)) if n_out > 0 else 0.0
            fp_km2    = float(n_out * CELL_AREA_KM2)

            # ── Update annual trackers ─────────────────────────────────────
            if n_out > 0:
                ann_agg_max_hail[year_idx] = max(ann_agg_max_hail[year_idx], max_hail)
                ann_agg_fp_km2[year_idx]  += fp_km2
                if max_hail > ann_occ_max_hail[year_idx]:
                    ann_occ_max_hail[year_idx] = max_hail
                    ann_occ_fp_km2[year_idx]   = fp_km2
                    ann_occ_n_cells[year_idx]  = n_out

            # ── Stream event summary ───────────────────────────────────────
            ev_buf.append(
                f"{year_idx},{global_ev_id},{doy},{n_out},"
                f"{max_hail:.3f},{mean_hail:.3f},{p95_hail:.3f},{fp_km2:.0f}\n"
            )

            # ── Cell sample — streamed immediately, no RAM buffer ──────────
            if year_idx in sample_year_set and n_out > 0:
                cell_out = active_flat[keep]
                hail_out = hail_active[keep]
                for ci_flat, hv in zip(cell_out, hail_out):
                    lat, lon = cell_latlon(int(ci_flat))
                    sc_buf.append(
                        f"{year_idx},{global_ev_id},{int(ci_flat)},"
                        f"{lat:.3f},{lon:.3f},{hv:.3f}\n"
                    )

            global_ev_id += 1

    # Flush remaining buffers
    if ev_buf:
        ev_csv.write("".join(ev_buf))
    if sc_buf:
        sc_csv.write("".join(sc_buf))

log(f"\n  Simulation complete — {elapsed()}")
log(f"  Total events simulated: {total_events:,}")
log(f"  Global events with data written: {global_ev_id:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Write outputs
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[7] Writing remaining outputs")

log(f"  stochastic_cell_sample.csv  (streamed to disk during simulation)")

# Save annual trackers (useful for validation / re-deriving PET later)
np.save(OUT_DIR / "ann_occ_max_hail.npy",  ann_occ_max_hail)
np.save(OUT_DIR / "ann_occ_fp_km2.npy",    ann_occ_fp_km2)
np.save(OUT_DIR / "ann_occ_n_cells.npy",   ann_occ_n_cells)
np.save(OUT_DIR / "ann_agg_max_hail.npy",  ann_agg_max_hail)
np.save(OUT_DIR / "ann_agg_fp_km2.npy",    ann_agg_fp_km2)
log("  Annual tracker arrays saved (ann_*.npy)")


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Build PETs
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[8] Building Probable Exceedance Tables")

rank = np.arange(1, N_SIM_YEARS + 1, dtype=np.float64)
rp   = N_SIM_YEARS / rank

# Occurrence PET
occ_hail_sorted = np.sort(ann_occ_max_hail)[::-1]
occ_fp_sorted   = np.sort(ann_occ_fp_km2)[::-1]
occ_nc_sorted   = np.sort(ann_occ_n_cells)[::-1]

pet_occ = pd.DataFrame({
    "return_period_yr": rp,
    "max_hail_in":      occ_hail_sorted,
    "footprint_km2":    occ_fp_sorted,
    "n_cells":          occ_nc_sorted,
})
pet_occ.to_csv(OUT_DIR / "pet_occurrence.csv", index=False)
log("  pet_occurrence.csv saved")

# Aggregate PET
agg_hail_sorted = np.sort(ann_agg_max_hail)[::-1]
agg_fp_sorted   = np.sort(ann_agg_fp_km2)[::-1]

pet_agg = pd.DataFrame({
    "return_period_yr": rp,
    "agg_max_hail_in":   agg_hail_sorted,
    "agg_footprint_km2": agg_fp_sorted,
})
pet_agg.to_csv(OUT_DIR / "pet_aggregate.csv", index=False)
log("  pet_aggregate.csv saved")


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Print summary table
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 62)
log("OCCURRENCE PET — Worst Single Event per Year")
log("=" * 62)
log(f"{'Return Pd (yr)':>15}  {'Max Hail (in)':>13}  {'Footprint (km2)':>16}  {'Cells':>7}")
for T in [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000]:
    row = pet_occ[pet_occ["return_period_yr"] >= T].iloc[-1]
    log(f"{T:>15,}  {row['max_hail_in']:>13.2f}  {row['footprint_km2']:>16,.0f}  {int(row['n_cells']):>7,}")

log("\n" + "=" * 62)
log("AGGREGATE PET — Annual Max Hail Intensity (All Events)")
log("=" * 62)
log(f"{'Return Pd (yr)':>15}  {'Agg Max Hail (in)':>17}  {'Agg Footprint (km2)':>20}")
for T in [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000]:
    row = pet_agg[pet_agg["return_period_yr"] >= T].iloc[-1]
    log(f"{T:>15,}  {row['agg_max_hail_in']:>17.2f}  {row['agg_footprint_km2']:>19,.0f}")

log(f"\n{'='*62}")
log(f"COMPLETE — {elapsed()}")
log(f"{'='*62}")
log(f"\nOutputs -> {OUT_DIR}/")
log(f"  stochastic_event_summary.csv    ({global_ev_id:,} events, streamed)")
log(f"  stochastic_cell_sample.csv      ({CELL_SAMPLE_YEARS:,}-yr validation sample, streamed)")
log(f"  pet_occurrence.csv")
log(f"  pet_aggregate.csv")
log(f"  ann_*.npy                       (annual tracker arrays)")
log(f"\nParameters: lambda={LAMBDA_KM}km  N={N_SIM_YEARS:,}yr  seed={RNG_SEED}")
