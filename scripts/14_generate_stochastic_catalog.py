#!/usr/bin/env python3
"""
14_generate_stochastic_catalog.py  (v3 — event-resampling bootstrap)
===========================================================
Generates a 50,000-year stochastic hail occurrence catalog using an
event-resampling (bootstrap) approach.  Each simulated event is drawn
from the library of 2,982+ historical event footprints (event_peak_array.npy),
preserving real spatial geometry instead of synthesising fields from a copula.

Why event-resampling instead of per-cell field generation?
  - Spatial footprint shapes come from real synoptic-scale storm geometry
  - No dependence on a Cholesky copula whose λ is poorly constrained by SPC data
  - Seasonal weighting + log-normal intensity perturbation add realistic variability
  - PET metrics are event-level (occurrence) and annual totals (aggregate),
    matching how insurance cat models define loss exceedance curves

Previous approach (v2) was WRONG because:
  - It synthesised per-cell hail from independent CDF lookups with Cholesky copula
  - Spatial footprint shapes were purely emergent from correlation structure, not
    real storm geometry → systematic underestimate of aggregate footprint variance
  - ann_occ_fp_km2 / ann_agg_fp_km2 were raw area metrics, not insurance-relevant
    PET metrics (intensity occurrence and aggregate exposure)

Methodology
-----------
Pre-computation:
  1. Load event_catalog.csv + event_peak_array.npy (n_events × 104 × 236 float32)
  2. Fit Poisson rate λ = n_events / n_years_of_record
  3. Build seasonal KDE of historical event DOYs (Gaussian, σ=10 days, wrapped)

Per simulated year (50,000 years):
  1. Draw N_events ~ Poisson(λ)
  2. For each event:
     a. Draw calendar DOY from seasonal distribution
     b. Resample template: weight each historical event by
        exp(-|historical_doy - drawn_doy| / 30), with day-of-year wrap-around
     c. Intensity perturbation: multiply all hail values by
        exp(σ_perturb × ε), σ_perturb = 0.15, ε ~ N(0,1)
     d. [Optional] Spatial translation ±1–2 cells (disabled by default)

PET metrics (what matters for insurance):
  ann_occ_max_hail[year] = max single-event peak hail (occurrence intensity)
  ann_occ_n_cells[year]  = n_cells of that max-hail event (occurrence footprint)
  ann_agg_n_cells[year]  = sum of n_cells across all events (aggregate exposure)
  ann_agg_events[year]   = count of events in the year

Outputs  (data/stochastic/)
-----------------------------------------------
  stochastic_event_summary.csv  — one row per simulated event
  pet_occurrence.csv            — occurrence PET: return_period_yr, max_hail_in, n_cells
  pet_aggregate.csv             — aggregate PET: return_period_yr, agg_n_cells, agg_events
  ann_occ_max_hail.npy          — annual occurrence max hail  (50,000,)
  ann_occ_n_cells.npy           — annual occurrence n_cells   (50,000,)
  ann_agg_n_cells.npy           — annual aggregate n_cells    (50,000,)
  active_flat_idx.npy           — active cell flat indices (for downstream stages)

Key parameters
--------------
  N_SIM_YEARS       = 50,000
  SIGMA_PERTURB     = 0.15    Log-normal intensity perturbation std
  SPATIAL_TRANSLATE = False   Optional spatial translation (disabled by default)
  DAMAGE_THRESHOLD_IN = 1.0   Footprint threshold (residential shingles)
  RNG_SEED          = 42
"""

import os, sys, time, warnings, gc
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

ROOT    = DATA_ROOT / "hail_0.25deg"
OUT_DIR = DATA_ROOT / "stochastic"
OUT_DIR.mkdir(exist_ok=True)
LOGS_ROOT.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOGS_ROOT / "stochastic_catalog_build.log"

# ── Parameters ────────────────────────────────────────────────────────────────
N_SIM_YEARS         = 50_000
SIGMA_PERTURB       = 0.15    # log-normal intensity perturbation std per event
SPATIAL_TRANSLATE   = False   # shift footprint ±1-2 cells (disabled by default)
DAMAGE_THRESHOLD_IN = 1.0     # cells below this excluded from n_cells count
MAX_HAIL_PHYSICAL   = 10.0    # physical ceiling (record: 8" Vivian SD 2010)
RNG_SEED            = 42
FLUSH_EVERY_YEARS   = 500     # CSV flush interval

NROWS, NCOLS  = 104, 236
CELL_DEG      = 0.25
CELL_AREA_KM2 = (CELL_DEG * 111.0) ** 2   # ~770.06 km²

T0 = time.time()


def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass."""
    errors = []
    files_to_check = [
        OUT_DIR / "stochastic_event_summary.csv",
        OUT_DIR / "pet_occurrence.csv",
        OUT_DIR / "pet_aggregate.csv",
        OUT_DIR / "active_flat_idx.npy",
        OUT_DIR / "ann_occ_max_hail.npy",
        OUT_DIR / "ann_occ_n_cells.npy",
        OUT_DIR / "ann_agg_n_cells.npy",
    ]
    for p in files_to_check:
        if not p.exists():
            errors.append(f"Missing: {p.name}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty: {p.name}")

    for csv_name in ["stochastic_event_summary.csv", "pet_occurrence.csv"]:
        p = OUT_DIR / csv_name
        if p.exists() and p.stat().st_size > 0:
            try:
                import csv as _csv
                with open(p, newline="") as f:
                    rows = list(_csv.DictReader(f))
                if len(rows) <= 1000:
                    errors.append(f"Too few rows in {csv_name}: {len(rows)} (expected >1000)")
            except Exception as e:
                errors.append(f"Cannot read {csv_name}: {e}")

    if errors:
        print("CRITICAL: Output validation FAILED:")
        for e in errors:
            print(f"  ✗ {e}")
        return False
    print("Output validation passed ✓")
    return True


# ── --validate early exit ──────────────────────────────────────────────────────
if "--validate" in sys.argv:
    ok = validate_outputs()
    sys.exit(0 if ok else 1)


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


log("=" * 60)
log("STOCHASTIC CATALOG GENERATION  v3 (event-resampling bootstrap)")
log(f"  N_SIM_YEARS      = {N_SIM_YEARS:,}")
log(f"  SIGMA_PERTURB    = {SIGMA_PERTURB}")
log(f"  SPATIAL_TRANSLATE= {SPATIAL_TRANSLATE}")
log(f"  RNG_SEED         = {RNG_SEED}")
log("=" * 60)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Load historical event catalog and peak arrays
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[1] Loading historical event catalog and peak arrays")

ec = pd.read_csv(ROOT / "event_catalog.csv", parse_dates=["start_date", "end_date"])
n_hist_events = len(ec)

# (n_hist_events, NROWS, NCOLS) float32 — peak hail at each cell per event
event_peak_array = np.load(ROOT / "event_peak_array.npy")
assert event_peak_array.shape[0] == n_hist_events, (
    f"event_peak_array shape {event_peak_array.shape[0]} != catalog length {n_hist_events}"
)

# Poisson rate from historical record
n_years_record = int(ec["start_date"].dt.year.nunique())
lambda_events  = n_hist_events / n_years_record

# DOY for each historical event (for seasonal weighting)
event_doys = ec["start_date"].dt.dayofyear.values.astype(np.int32)
event_ids  = ec["event_id"].values

log(f"  Historical events: {n_hist_events:,} over {n_years_record} years")
log(f"  Poisson rate λ = {lambda_events:.2f} events/year")
log(f"  event_peak_array: {event_peak_array.shape}  mem ~{mem_mb():.0f} MB")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Active cell indices (required by downstream stage 15)
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[2] Computing active cell indices from p_occurrence.tif")

with rasterio.open(ROOT / "p_occurrence.tif") as src:
    p_occ_grid = src.read(1).astype(np.float32)
    nodata_val = src.nodata if src.nodata is not None else -9999.0

active_mask = (p_occ_grid > 0) & (p_occ_grid != nodata_val)
active_flat = np.where(active_mask.ravel())[0]
np.save(OUT_DIR / "active_flat_idx.npy", active_flat)
log(f"  Saved active_flat_idx.npy ({len(active_flat):,} cells)")
del p_occ_grid
gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Seasonal DOY distribution for event date sampling
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[3] Building seasonal DOY distribution")

# KDE-smoothed daily probability (Gaussian σ=10 days, wrapped at year boundaries)
daily_count = np.zeros(366, dtype=np.float64)
for d in event_doys:
    daily_count[min(d, 366) - 1] += 1.0

padded     = np.tile(daily_count, 3)                  # wrap-around for Gaussian
smoothed   = gaussian_filter1d(padded, sigma=10.0)
daily_prob = smoothed[366:732]
daily_prob /= daily_prob.sum()
doy_values = np.arange(1, 367)
doy_cdf    = np.cumsum(daily_prob)

log(f"  Peak DOY: {int(doy_values[daily_prob.argmax()])}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Simulation — 50,000 years
# ═══════════════════════════════════════════════════════════════════════════════
log(f"\n[4] Simulation — {N_SIM_YEARS:,} years  mem ~{mem_mb():.0f} MB")

rng = np.random.default_rng(RNG_SEED)

# Annual tracking arrays
ann_occ_max_hail = np.zeros(N_SIM_YEARS, dtype=np.float32)
ann_occ_n_cells  = np.zeros(N_SIM_YEARS, dtype=np.int32)
ann_agg_n_cells  = np.zeros(N_SIM_YEARS, dtype=np.int64)
ann_agg_events   = np.zeros(N_SIM_YEARS, dtype=np.int32)

ev_csv_path = OUT_DIR / "stochastic_event_summary.csv"
ev_header   = ("sim_year,event_idx,template_event_id,doy,"
               "n_cells,max_hail_in,mean_hail_in,footprint_km2\n")

with open(ev_csv_path, "w") as ev_csv:
    ev_csv.write(ev_header)

    t_sim        = time.time()
    total_events = 0
    global_ev_id = 0
    ev_buf       = []

    for year_idx in range(N_SIM_YEARS):

        if year_idx > 0 and year_idx % 5000 == 0:
            rate = year_idx / (time.time() - t_sim)
            eta  = (N_SIM_YEARS - year_idx) / rate / 60
            log(f"  Year {year_idx:>6,} / {N_SIM_YEARS:,}  "
                f"events: {total_events:,}  "
                f"ETA: {eta:.1f} min  mem ~{mem_mb():.0f} MB")

        # Flush CSV buffer periodically
        if year_idx % FLUSH_EVERY_YEARS == 0 and year_idx > 0 and ev_buf:
            ev_csv.write("".join(ev_buf))
            ev_csv.flush()
            ev_buf.clear()

        n_ev = int(rng.poisson(lambda_events))
        total_events += n_ev
        if n_ev == 0:
            continue

        # Draw event DOYs from seasonal distribution
        u_date  = rng.random(n_ev)
        ev_doys = doy_values[np.searchsorted(doy_cdf, u_date)]

        for ev_i in range(n_ev):
            doy = int(ev_doys[ev_i])

            # ── Seasonal weighting for template selection ──────────────────
            # Weight each historical event by exp(-|doy_diff| / 30),
            # wrapping around the year boundary
            doy_diff = np.abs(event_doys - doy)
            doy_diff = np.minimum(doy_diff, 366 - doy_diff)
            weights  = np.exp(-doy_diff / 30.0)
            weights /= weights.sum()

            # ── Resample a historical template event ───────────────────────
            template_idx      = int(rng.choice(n_hist_events, p=weights))
            template_event_id = int(event_ids[template_idx])
            peak_template     = event_peak_array[template_idx]  # view, no copy

            # ── Log-normal intensity perturbation ──────────────────────────
            # Multiply all hail values by exp(σ × ε), ε ~ N(0,1)
            # Preserves spatial structure; adds year-to-year intensity variability
            scale_factor = float(np.exp(SIGMA_PERTURB * rng.standard_normal()))
            perturbed    = np.clip(peak_template * scale_factor,
                                   0.0, MAX_HAIL_PHYSICAL)

            # ── Optional spatial translation ───────────────────────────────
            if SPATIAL_TRANSLATE:
                dr = int(rng.integers(-2, 3))
                dc = int(rng.integers(-2, 3))
                if dr != 0 or dc != 0:
                    perturbed = np.roll(np.roll(perturbed, dr, axis=0), dc, axis=1)
                    if dr > 0:
                        perturbed[:dr, :] = 0.0
                    elif dr < 0:
                        perturbed[dr:, :] = 0.0
                    if dc > 0:
                        perturbed[:, :dc] = 0.0
                    elif dc < 0:
                        perturbed[:, dc:] = 0.0

            # ── Count footprint cells (damage threshold = 1.0") ────────────
            footprint_mask = perturbed >= DAMAGE_THRESHOLD_IN
            n_cells        = int(footprint_mask.sum())

            if n_cells == 0:
                continue

            active_hail = perturbed[footprint_mask]
            max_hail    = float(active_hail.max())
            mean_hail   = float(active_hail.mean())
            fp_km2      = float(n_cells * CELL_AREA_KM2)

            # ── Update annual trackers ─────────────────────────────────────
            ann_agg_n_cells[year_idx] += n_cells
            ann_agg_events[year_idx]  += 1
            if max_hail > ann_occ_max_hail[year_idx]:
                ann_occ_max_hail[year_idx] = max_hail
                ann_occ_n_cells[year_idx]  = n_cells

            # ── Stream event summary ───────────────────────────────────────
            ev_buf.append(
                f"{year_idx},{global_ev_id},{template_event_id},"
                f"{doy},{n_cells},{max_hail:.3f},{mean_hail:.3f},{fp_km2:.0f}\n"
            )
            global_ev_id += 1

    # Flush remaining buffer
    if ev_buf:
        ev_csv.write("".join(ev_buf))

log(f"\n  Simulation complete — {elapsed()}")
log(f"  Total event draws: {total_events:,}")
log(f"  Events written to CSV (n_cells > 0): {global_ev_id:,}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Save annual tracker arrays
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[5] Writing annual tracker arrays")

np.save(OUT_DIR / "ann_occ_max_hail.npy", ann_occ_max_hail)
np.save(OUT_DIR / "ann_occ_n_cells.npy",  ann_occ_n_cells)
np.save(OUT_DIR / "ann_agg_n_cells.npy",  ann_agg_n_cells)
log("  Saved ann_occ_max_hail.npy, ann_occ_n_cells.npy, ann_agg_n_cells.npy")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Build Probable Exceedance Tables (marginal, sorted independently)
# ═══════════════════════════════════════════════════════════════════════════════
log("\n[6] Building Probable Exceedance Tables")

rank = np.arange(1, N_SIM_YEARS + 1, dtype=np.float64)
rp   = N_SIM_YEARS / rank

# Occurrence PET — worst single event per year by intensity
occ_hail_sorted = np.sort(ann_occ_max_hail)[::-1]
occ_nc_sorted   = np.sort(ann_occ_n_cells)[::-1]

pet_occ = pd.DataFrame({
    "return_period_yr": rp,
    "max_hail_in":      occ_hail_sorted,
    "n_cells":          occ_nc_sorted,
})
pet_occ.to_csv(OUT_DIR / "pet_occurrence.csv", index=False)
log("  pet_occurrence.csv saved")

# Aggregate PET — annual total geographic exposure
agg_nc_sorted  = np.sort(ann_agg_n_cells)[::-1]
agg_ev_sorted  = np.sort(ann_agg_events)[::-1]

pet_agg = pd.DataFrame({
    "return_period_yr": rp,
    "agg_n_cells":      agg_nc_sorted,
    "agg_events":       agg_ev_sorted,
})
pet_agg.to_csv(OUT_DIR / "pet_aggregate.csv", index=False)
log("  pet_aggregate.csv saved")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Print summary tables
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 64)
log("OCCURRENCE PET — Worst Single Event per Year (by Intensity)")
log("=" * 64)
log(f"{'Return Pd (yr)':>15}  {'Max Hail (in)':>13}  {'N Cells':>8}")
for T in [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000]:
    row = pet_occ[pet_occ["return_period_yr"] >= T].iloc[-1]
    log(f"{T:>15,}  {row['max_hail_in']:>13.2f}  {int(row['n_cells']):>8,}")

log("\n" + "=" * 64)
log("AGGREGATE PET — Annual Total Cell-Events (Geographic Exposure)")
log("=" * 64)
log(f"{'Return Pd (yr)':>15}  {'Agg N Cells':>12}  {'Agg Events':>10}")
for T in [2, 5, 10, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000, 10000, 25000]:
    row = pet_agg[pet_agg["return_period_yr"] >= T].iloc[-1]
    log(f"{T:>15,}  {int(row['agg_n_cells']):>12,}  {int(row['agg_events']):>10,}")

log(f"\n{'='*64}")
log(f"COMPLETE — {elapsed()}")
log(f"{'='*64}")
log(f"\nOutputs -> {OUT_DIR}/")
log(f"  stochastic_event_summary.csv  ({global_ev_id:,} events, streamed)")
log(f"  pet_occurrence.csv            (occurrence PET: max_hail_in + n_cells)")
log(f"  pet_aggregate.csv             (aggregate PET: agg_n_cells + agg_events)")
log(f"  ann_*.npy                     (annual tracker arrays)")
log(f"\nParameters: N={N_SIM_YEARS:,}yr  λ={lambda_events:.1f}ev/yr  "
    f"σ_perturb={SIGMA_PERTURB}  seed={RNG_SEED}")
log(f"Template library: {n_hist_events:,} historical events over {n_years_record} years")

if not validate_outputs():
    sys.exit(1)
