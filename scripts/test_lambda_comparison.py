#!/usr/bin/env python3
"""
test_lambda_comparison.py

Tests λ = 100, 150, 200 km using a Gaussian copula Monte Carlo.
Validation metric: variance of annual aggregate hail footprint vs historical.

Method:
  1. For each λ, build model correlation matrix → Cholesky L
  2. Simulate N_SIM_YEARS of annual hail using:
     - Draw correlated standard normals via L
     - Map uniform marginals through empirical CDF per cell
     - Aggregate across cells and events per year
  3. Compare simulated annual variance to historical (558,002)
  4. Save the best/chosen Cholesky (default: 200km)
"""

import os, json, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import norm as sp_norm
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

ROOT              = str(DATA_ROOT / "hail_0.25deg")
DAMAGE_THRESHOLD  = 1.0
CHOSEN_LAMBDA     = 200.0   # default choice unless data says otherwise
LAMBDAS_TO_TEST   = [100.0, 150.0, 200.0]
N_SIM_YEARS       = 2000    # simulation length
RNG_SEED          = 42

print("="*60)
print("LAMBDA COMPARISON — 100 / 150 / 200 km")
print("="*60)

# ── Load data ────────────────────────────────────────────────
print("\nLoading data...")
event_peak = np.load(os.path.join(ROOT, "event_peak_array.npy"))
n_events, nrows, ncols = event_peak.shape
print(f"  event_peak_array: {event_peak.shape}")

event_df = pd.read_csv(os.path.join(ROOT, "event_catalog.csv"),
                        parse_dates=["start_date", "end_date"])
event_years = event_df["start_date"].dt.year.values
unique_years = sorted(set(event_years))
n_hist_years = len(unique_years)
print(f"  event_catalog: {len(event_df)} events, {n_hist_years} years")

import xarray as xr
hail_da = xr.open_dataarray(os.path.join(ROOT, "char_hail_daily.nc"))
lats = hail_da.lat.values
lons = hail_da.lon.values
hail_da.close()

# Load existing corr_cell_idx (800 subsampled cells from earlier)
corr_idx = np.load(os.path.join(ROOT, "corr_cell_idx.npy"))
n_cells  = len(corr_idx)
print(f"  corr_cell_idx: {n_cells} cells")

corr_lats = np.array([lats[i // ncols] for i in corr_idx])
corr_lons = np.array([lons[i %  ncols] for i in corr_idx])

# Per-cell event intensity matrix for the corr cells
flat      = event_peak.reshape(n_events, nrows * ncols)
cell_data = flat[:, corr_idx]  # (n_events, n_cells)

# ── Historical reference metrics ─────────────────────────────
# Annual footprint = total cell-events above threshold per year (across corr cells)
print("\nComputing historical reference...")

# Events per year
events_per_year = {y: [] for y in unique_years}
for i, yr in enumerate(event_years):
    if yr in events_per_year:
        events_per_year[yr].append(i)

mean_events_per_year = np.mean([len(v) for v in events_per_year.values()])
print(f"  Mean events/year: {mean_events_per_year:.1f}")

# Historical annual aggregate: sum of cell intensities per year
hist_annual_agg = []
for yr in unique_years:
    idx = events_per_year[yr]
    if idx:
        yr_peak = cell_data[idx, :]         # (n_events_yr, n_cells)
        # Annual max per cell
        ann_max = yr_peak.max(axis=0)       # (n_cells,)
        hist_annual_agg.append(ann_max.sum())
    else:
        hist_annual_agg.append(0.0)

hist_annual_agg = np.array(hist_annual_agg)
hist_mean = hist_annual_agg.mean()
hist_var  = hist_annual_agg.var()
hist_p90  = np.percentile(hist_annual_agg, 90)
hist_p99  = np.percentile(hist_annual_agg, 99)

print(f"  Historical annual aggregate:")
print(f"    mean = {hist_mean:.1f}  var = {hist_var:.0f}")
print(f"    p90  = {hist_p90:.1f}  p99 = {hist_p99:.1f}")

# Also compute historical annual footprint (active cells above threshold)
hist_annual_footprint = []
for yr in unique_years:
    idx = events_per_year[yr]
    if idx:
        yr_peak = cell_data[idx, :]
        ann_max = yr_peak.max(axis=0)
        hist_annual_footprint.append((ann_max >= DAMAGE_THRESHOLD).sum())
    else:
        hist_annual_footprint.append(0)

hist_fp = np.array(hist_annual_footprint)
hist_fp_var = hist_fp.var()
print(f"  Historical annual active cells: mean={hist_fp.mean():.0f}  var={hist_fp_var:.0f}")
print(f"  (Full-grid footprint var from pipeline: 558,002)")

# ── Build empirical marginal CDF per cell ────────────────────
# For each cell: sorted event intensities (including zeros) + ranks
print("\nBuilding empirical marginals...")
n_ev = cell_data.shape[0]

# Sort each cell's event intensities
sorted_cell = np.sort(cell_data, axis=0)  # (n_events, n_cells)
# Plotting positions (Weibull: i/(n+1))
pp = (np.arange(1, n_ev + 1) / (n_ev + 1))  # (n_events,)

# Fraction of events with zero hail per cell (= 1 - p_occ)
p_zero = (cell_data == 0).sum(axis=0) / n_ev  # (n_cells,)

def quantile_transform(u_vals, sorted_col, pp_arr, p_z):
    """
    Map uniform u to hail intensity using empirical CDF.
    u <= p_zero → 0.0 (no hail)
    u >  p_zero → interpolate from sorted nonzero values
    """
    out = np.zeros_like(u_vals)
    hail_mask = u_vals > p_z
    if hail_mask.any():
        u_hail = u_vals[hail_mask]
        out[hail_mask] = np.interp(u_hail, pp_arr, sorted_col)
    return out

# ── Distance matrix for corr cells ───────────────────────────
coords_km   = np.column_stack([
    corr_lats * 111.0,
    corr_lons * 111.0 * np.cos(np.radians(corr_lats.mean()))
])
dist_matrix = cdist(coords_km, coords_km)

# ── Monte Carlo simulation for each λ ────────────────────────
def simulate_annual_agg(L, sorted_cell, pp, p_zero, n_years, n_ev_per_year, rng):
    """
    Simulate annual aggregate hail using Gaussian copula.
    Returns array of shape (n_years,) — annual sum of peak intensities.
    """
    n_cells = L.shape[0]
    annual_agg      = np.zeros(n_years)
    annual_footprint = np.zeros(n_years, dtype=int)

    # We simulate one "peak event" per year for efficiency
    # (draws correlated intensity fields and takes max across events)
    # For each year: draw n_ev_per_year correlated fields, take cell-wise max
    for yr in range(n_years):
        # Draw n_ev correlated standard normals: (n_ev_per_year, n_cells)
        Z_raw = rng.standard_normal((n_ev_per_year, n_cells))  # (n_ev, n_cells)
        Z_cor = (L @ Z_raw.T).T                                 # (n_ev, n_cells)
        U     = sp_norm.cdf(Z_cor)                              # uniform marginals

        # Map each event through empirical CDF per cell
        event_intensities = np.zeros_like(U)
        for k in range(n_cells):
            event_intensities[:, k] = quantile_transform(
                U[:, k], sorted_cell[:, k], pp, p_zero[k]
            )

        # Annual max per cell across events
        ann_max = event_intensities.max(axis=0)
        annual_agg[yr]       = ann_max.sum()
        annual_footprint[yr] = (ann_max >= DAMAGE_THRESHOLD).sum()

    return annual_agg, annual_footprint

rng = np.random.default_rng(RNG_SEED)
n_ev_per_year = int(round(mean_events_per_year))
results = {}

print(f"\nRunning Monte Carlo ({N_SIM_YEARS} years × {n_ev_per_year} events/yr × {n_cells} cells)...")

for lam in LAMBDAS_TO_TEST:
    t_lam = time.time()
    print(f"\n  λ = {lam:.0f} km")

    # Build model correlation matrix
    model_corr = np.exp(-dist_matrix / lam)
    np.fill_diagonal(model_corr, 1.0)

    # PSD correction if needed
    eigvals = np.linalg.eigvalsh(model_corr)
    if eigvals.min() < 0:
        ev, evec = np.linalg.eigh(model_corr)
        ev        = np.maximum(ev, 1e-8)
        model_corr = evec @ np.diag(ev) @ evec.T
        np.fill_diagonal(model_corr, 1.0)
        print(f"    PSD correction applied")

    # Cholesky
    L = np.linalg.cholesky(model_corr)
    print(f"    Cholesky done  min_eigval={eigvals.min():.4f}")

    # Simulate
    rng_lam = np.random.default_rng(RNG_SEED)  # same seed for fair comparison
    sim_agg, sim_fp = simulate_annual_agg(
        L, sorted_cell, pp, p_zero, N_SIM_YEARS, n_ev_per_year, rng_lam
    )

    sim_mean = sim_agg.mean()
    sim_var  = sim_agg.var()
    sim_p90  = np.percentile(sim_agg, 90)
    sim_p99  = np.percentile(sim_agg, 99)
    sim_fp_var = sim_fp.var()

    var_ratio = sim_var / hist_var if hist_var > 0 else np.nan
    fp_ratio  = sim_fp_var / hist_fp_var if hist_fp_var > 0 else np.nan

    results[lam] = {
        "lambda_km":      lam,
        "sim_mean":       sim_mean,
        "sim_var":        sim_var,
        "sim_p90":        sim_p90,
        "sim_p99":        sim_p99,
        "sim_fp_var":     sim_fp_var,
        "var_ratio_agg":  var_ratio,
        "var_ratio_fp":   fp_ratio,
        "sim_agg":        sim_agg,
        "sim_fp":         sim_fp,
        "L":              L,
        "model_corr":     model_corr,
    }

    print(f"    Aggregate: mean={sim_mean:.1f}  var={sim_var:.0f}  "
          f"p90={sim_p90:.1f}  p99={sim_p99:.1f}")
    print(f"    Footprint var: {sim_fp_var:.0f}")
    print(f"    Var ratio (sim/hist) agg={var_ratio:.3f}  fp={fp_ratio:.3f}")
    print(f"    ({time.time()-t_lam:.0f}s)")

# ── Summary table ─────────────────────────────────────────────
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
print(f"\n{'Metric':<30} {'Historical':>12} {'λ=100km':>12} {'λ=150km':>12} {'λ=200km':>12}")
print("-"*78)

def row(label, hist_val, fmt=".1f"):
    vals = [f"{results[lam]['sim_' + label]:{fmt}}" if f'sim_{label}' in results[lam]
            else "—" for lam in LAMBDAS_TO_TEST]
    print(f"{label:<30} {hist_val:>12{fmt}} " + "  ".join(f"{v:>12}" for v in vals))

print(f"{'Annual agg mean':<30} {hist_mean:>12.1f} " +
      "  ".join(f"{results[l]['sim_mean']:>12.1f}" for l in LAMBDAS_TO_TEST))
print(f"{'Annual agg var':<30} {hist_var:>12.0f} " +
      "  ".join(f"{results[l]['sim_var']:>12.0f}" for l in LAMBDAS_TO_TEST))
print(f"{'Annual agg p90':<30} {hist_p90:>12.1f} " +
      "  ".join(f"{results[l]['sim_p90']:>12.1f}" for l in LAMBDAS_TO_TEST))
print(f"{'Annual agg p99':<30} {hist_p99:>12.1f} " +
      "  ".join(f"{results[l]['sim_p99']:>12.1f}" for l in LAMBDAS_TO_TEST))
print(f"{'Footprint var':<30} {hist_fp_var:>12.0f} " +
      "  ".join(f"{results[l]['sim_fp_var']:>12.0f}" for l in LAMBDAS_TO_TEST))
print(f"{'Var ratio (agg)':<30} {'1.000':>12} " +
      "  ".join(f"{results[l]['var_ratio_agg']:>12.3f}" for l in LAMBDAS_TO_TEST))
print(f"{'Var ratio (footprint)':<30} {'1.000':>12} " +
      "  ".join(f"{results[l]['var_ratio_fp']:>12.3f}" for l in LAMBDAS_TO_TEST))

# Pick λ with var ratio closest to 1.0
best_lam = min(LAMBDAS_TO_TEST,
               key=lambda l: abs(results[l]['var_ratio_fp'] - 1.0))
print(f"\nClosest var ratio to historical: λ = {best_lam:.0f} km")
print(f"Default choice:                  λ = {CHOSEN_LAMBDA:.0f} km")
final_lam = CHOSEN_LAMBDA  # stick with 200km as instructed

# ── Validation plot ───────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
colors = {100.0: 'steelblue', 150.0: 'darkorange', 200.0: 'seagreen'}

for col, lam in enumerate(LAMBDAS_TO_TEST):
    r = results[lam]

    # Top row: annual aggregate distribution
    ax = axes[0, col]
    ax.hist(hist_annual_agg, bins=20, alpha=0.5, color='gray',
            label=f"Historical (n={n_hist_years})", density=True)
    ax.hist(r['sim_agg'], bins=40, alpha=0.6, color=colors[lam],
            label=f"Simulated (n={N_SIM_YEARS})", density=True)
    ax.axvline(hist_mean, color='black', linewidth=1.5, linestyle='--')
    ax.axvline(r['sim_mean'], color=colors[lam], linewidth=1.5)
    ax.set_title(f"λ = {lam:.0f} km — Annual agg intensity\nvar_ratio={r['var_ratio_agg']:.3f}")
    ax.set_xlabel("Sum of annual max intensities (in)")
    ax.legend(fontsize=7)

    # Bottom row: annual footprint distribution
    ax2 = axes[1, col]
    ax2.hist(hist_annual_footprint, bins=15, alpha=0.5, color='gray',
             label="Historical", density=True)
    ax2.hist(r['sim_fp'], bins=30, alpha=0.6, color=colors[lam],
             label="Simulated", density=True)
    ax2.set_title(f"λ = {lam:.0f} km — Annual footprint (cells)\nvar_ratio={r['var_ratio_fp']:.3f}")
    ax2.set_xlabel("Active cells ≥ 1.0 in")
    ax2.legend(fontsize=7)

fig.suptitle(
    f"Gaussian copula λ comparison — {N_SIM_YEARS}-year Monte Carlo\n"
    f"Historical: {n_hist_years} years, {n_cells} cells, DAMAGE_THRESH={DAMAGE_THRESHOLD}\"",
    fontsize=10
)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "lambda_comparison.png"), dpi=150)
print(f"\nSaved lambda_comparison.png")

# ── Save final choice ────────────────────────────────────────
print(f"\nSaving final Cholesky for λ = {final_lam:.0f} km...")
np.save(os.path.join(ROOT, "cholesky_L.npy"),    results[final_lam]["L"])
np.save(os.path.join(ROOT, "corr_cell_idx.npy"), corr_idx)

# Save all three for reference
for lam in LAMBDAS_TO_TEST:
    np.save(os.path.join(ROOT, f"cholesky_L_{lam:.0f}km.npy"), results[lam]["L"])
    print(f"  Saved cholesky_L_{lam:.0f}km.npy")

with open(os.path.join(ROOT, "lambda_km.json"), "w") as f:
    json.dump({
        "lambda_km":          final_lam,
        "method":             "literature_informed_option_b",
        "candidates_tested":  LAMBDAS_TO_TEST,
        "var_ratios_agg":     {str(l): round(results[l]["var_ratio_agg"], 4) for l in LAMBDAS_TO_TEST},
        "var_ratios_fp":      {str(l): round(results[l]["var_ratio_fp"], 4) for l in LAMBDAS_TO_TEST},
        "best_fit_lambda_km": float(best_lam),
        "chosen_lambda_km":   final_lam,
        "n_sim_years":        N_SIM_YEARS,
        "n_corr_cells":       int(n_cells),
        "damage_threshold_in": DAMAGE_THRESHOLD,
        "note":               f"200km chosen per model design. Best empirical fit: {best_lam:.0f}km"
    }, f, indent=2)
print(f"  Saved lambda_km.json  (chosen λ = {final_lam:.0f} km)")

print(f"\n{'='*60}")
print(f"LAMBDA COMPARISON COMPLETE — {elapsed()}")
print(f"{'='*60}")
print(f"  Best empirical fit: λ = {best_lam:.0f} km")
print(f"  Chosen (design):    λ = {final_lam:.0f} km")
print(f"  Cholesky saved:     cholesky_L.npy  ({n_cells}×{n_cells})")
print(f"  All three saved:    cholesky_L_100km/150km/200km.npy")
print(f"  Validation plot:    lambda_comparison.png")
