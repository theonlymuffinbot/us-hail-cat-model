#!/usr/bin/env python3
"""
15_stochastic_maps.py  —  Per-cell stochastic hazard maps
==========================================================
Runs a lean 2,000-year simulation using cached CDF lookup + Cholesky.
Everything preloaded; zero I/O inside the event loop.

Key fix vs previous version:
  - p_occurrence.tif stores ANNUAL probability (0.1–0.75 range).
    Per-event fire probability = p_annual / lambda_events  (~0.001–0.006).
    This is the correct per-event zero-inflation probability.
  - Vectorised CDF lookup: cdf[bins, np.arange(N_ACT)] instead of per-cell loop.
  - ann_max updated with np.maximum.at() which handles duplicate indices safely.

Memory: ~1.2 GB peak  Runtime: ~20-30 min
"""

import sys, datetime
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.ndimage import uniform_filter1d
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
CLIMO = DATA / "hail_0.25deg_climo"
OUT_D = STOCH / "maps";           OUT_D.mkdir(exist_ok=True)
OUT_F = REPO / "docs/figures/stochastic"; OUT_F.mkdir(exist_ok=True)

N_SIM      = 2_000
LAM_EVT    = 127.3
LAM_KM     = 150.0
SEED       = 77
THRESH     = 0.25
RP_YRS     = [10, 25, 50, 100, 200, 500]
POCC_T     = [0.25, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 5.00]
NROWS, NCOLS = 104, 236
LAT_MAX, LAT_MIN = 48.875, 22.875
LON_MIN, LON_MAX = -124.875, -65.125
CS         = 0.25
NODATA     = -9999.0
KM_LAT     = 111.0

def write_tif(arr, path):
    tr = from_bounds(LON_MIN-CS/2, LAT_MIN-CS/2, LON_MAX+CS/2, LAT_MAX+CS/2, NCOLS, NROWS)
    with rasterio.open(path, "w", driver="GTiff", dtype="float32",
                       width=NCOLS, height=NROWS, count=1, crs="EPSG:4326",
                       transform=tr, nodata=NODATA, compress="lzw") as dst:
        dst.write(arr.astype(np.float32), 1)

def doy_to_mmdd(doy):
    d = datetime.date(2000,1,1) + datetime.timedelta(days=int(doy)-1)
    return f"{d.month:02d}{d.day:02d}"

print("=== Loading inputs ===", flush=True)

act_idx  = np.load(STOCH/"active_flat_idx.npy")   # (12811,)
cdf_p    = np.load(STOCH/"cdf_quant_p.npy")       # (2000,)
chol_L   = np.load(HIST/"cholesky_L_150km.npy")   # (800,800)
seed_idx = np.load(HIST/"corr_cell_idx.npy")       # (800,)
N_ACT    = len(act_idx)
N_SEED   = len(seed_idx)

print(f"  active={N_ACT}, seed={N_SEED}", flush=True)
print("  Loading CDF lookup (98 MB)...", flush=True)
cdf_lut = np.load(STOCH/"cdf_lookup.npy")         # (2000, 12811) — loads fully
print(f"  CDF loaded: {cdf_lut.shape}", flush=True)

# Annual p_occ per active cell
with rasterio.open(HIST/"p_occurrence.tif") as src:
    p_occ_full = src.read(1).astype(np.float32)
p_occ_ann = p_occ_full.flat[act_idx].copy()       # (12811,) ANNUAL probability

# ── KEY FIX: per-event fire probability ──────────────────────────────────
# p_occurrence.tif = P(cell hit at least once per year)
# Assuming Poisson events: P(hit in one event) ≈ p_annual / lambda_events
# More precisely: p_event = 1 - (1-p_annual)^(1/lambda) ≈ p_annual/lambda for small p
p_evt = p_occ_ann / LAM_EVT                       # (12811,)  ~0.001 to 0.006
print(f"  p_evt range: {p_evt.min():.5f} to {p_evt.max():.5f}, mean={p_evt.mean():.5f}", flush=True)

# ── Pre-load climo seasonal weights ──────────────────────────────────────
print("  Pre-loading climo rasters (366 files)...", flush=True)
# seasonal_scale[doy] = per-cell weight (not probability) summing seasonal variation
seasonal_scale = np.ones((367, N_ACT), dtype=np.float32)
if CLIMO.exists():
    climo_files = {f.stem.replace("climo_",""): f for f in sorted(CLIMO.glob("climo_*.tif"))}
    for doy in range(1, 367):
        mmdd = doy_to_mmdd(doy)
        if mmdd not in climo_files:
            continue
        with rasterio.open(climo_files[mmdd]) as src:
            arr = src.read()                       # (29, 104, 236)
        total = arr.sum(axis=0).flat[act_idx].astype(np.float32)
        mx = total.max()
        if mx > 0:
            seasonal_scale[doy] = total / mx
    print(f"  Loaded {len(climo_files)} climo files", flush=True)

# ── Event DOY distribution ────────────────────────────────────────────────
cat = pd.read_csv(HIST/"event_catalog.csv")
date_col = next((c for c in ["start_date","date","Date","event_date"] if c in cat.columns), None)
if date_col:
    doys_hist = pd.to_datetime(cat[date_col]).dt.dayofyear.values
    doy_w = np.zeros(366, dtype=np.float32)
    for d in doys_hist:
        doy_w[min(d-1,365)] += 1
    doy_w = uniform_filter1d(doy_w, size=28, mode='wrap')
    doy_w /= doy_w.sum()
else:
    doy_w = np.full(366, 1/366, dtype=np.float32)
doy_choices = np.arange(1, 367)

# ── Correlation mapping ───────────────────────────────────────────────────
print("  Building correlation map...", flush=True)
sr = seed_idx // NCOLS; sc = seed_idx % NCOLS
sl = LAT_MAX - sr*CS;   so = LON_MIN + sc*CS
ar = act_idx  // NCOLS; ac = act_idx  % NCOLS
al = LAT_MAX - ar*CS;   ao = LON_MIN + ac*CS
parent = np.zeros(N_ACT, dtype=np.int32)
rho    = np.zeros(N_ACT, dtype=np.float32)
for s in range(0, N_ACT, 500):
    e = min(s+500, N_ACT)
    dy = (al[s:e,None] - sl[None,:]) * KM_LAT
    dx = (ao[s:e,None] - so[None,:]) * KM_LAT * np.cos(np.radians(al[s:e,None]))
    d  = np.sqrt(dx**2 + dy**2)
    nn = np.argmin(d, axis=1)
    parent[s:e] = nn
    rho[s:e]    = np.exp(-d[np.arange(e-s), nn] / LAM_KM).astype(np.float32)
rho_c = np.sqrt(np.maximum(0.0, 1.0-rho**2)).astype(np.float32)
print("  Correlation map done.", flush=True)

# ══════════════════════════════════════════════════════════════════════════
print(f"\n=== Simulating {N_SIM:,} years ===", flush=True)
rng     = np.random.default_rng(SEED)
# Per-cell annual max: shape (N_SIM, N_ACT)
ann_max = np.zeros((N_SIM, N_ACT), dtype=np.float32)

for yr in range(N_SIM):
    if yr % 200 == 0:
        print(f"  Year {yr}/{N_SIM}...", flush=True)

    n_evt = rng.poisson(LAM_EVT)
    if n_evt == 0:
        continue

    evt_doys = rng.choice(doy_choices, size=n_evt, p=doy_w)

    for doy in evt_doys:
        # Per-event fire prob: scale annual p_evt by seasonal weight
        p_fire = p_evt * seasonal_scale[doy]       # (N_ACT,)

        # Correlated z-scores
        z_seed = (chol_L @ rng.standard_normal(N_SEED)).astype(np.float32)
        z_act  = rho * z_seed[parent] + rho_c * rng.standard_normal(N_ACT).astype(np.float32)

        # Fire decision
        u     = ndtr(z_act).astype(np.float32)
        fired = u < p_fire                          # (N_ACT,) bool
        if not fired.any():
            continue

        fi = np.where(fired)[0]                    # indices into active cells

        # CDF inversion — conditional on firing
        # Re-draw uniform conditional on u < p_fire using the correlated z directly
        # u_cond in (0,1) given firing: u_cond = u_fired / p_fire_fired
        u_cond = np.clip(u[fired] / np.maximum(p_fire[fired], 1e-9), 0.0, 0.9999)

        # Map through CDF lookup
        bins = np.searchsorted(cdf_p, u_cond).clip(0, len(cdf_p)-1)
        hail = cdf_lut[bins, fi]                   # (n_fired,) — vectorised

        # Update annual max (use np.maximum.at for safety with repeated indices)
        big = hail >= THRESH
        if big.any():
            np.maximum.at(ann_max[yr], fi[big], hail[big])

print("  Simulation complete.", flush=True)

# ══════════════════════════════════════════════════════════════════════════
print("\n=== Computing per-cell statistics ===", flush=True)

rp_g    = {rp: np.full((NROWS,NCOLS), NODATA, np.float32) for rp in RP_YRS}
poc_g   = {t:  np.full((NROWS,NCOLS), NODATA, np.float32) for t  in POCC_T}
poc_any = np.full((NROWS,NCOLS), NODATA, np.float32)

for i, flat in enumerate(act_idx):
    row, col = int(flat // NCOLS), int(flat % NCOLS)
    ann = ann_max[:, i]                            # (N_SIM,)
    p   = float(np.mean(ann >= THRESH))
    poc_any[row, col] = p
    for t in POCC_T:
        poc_g[t][row, col] = float(np.mean(ann >= t))
    for rp in RP_YRS:
        q = float(np.quantile(ann, 1.0 - 1.0 / rp))
        rp_g[rp][row, col] = q if q >= THRESH else NODATA

del ann_max

# Apply CONUS mask
if (HIST/"rp_100yr_hail.tif").exists():
    with rasterio.open(HIST/"rp_100yr_hail.tif") as src:
        hm  = src.read(1); hnd = src.nodata or -9999
    out = (hm == hnd) | (hm <= 0)
    for g in list(rp_g.values()) + list(poc_g.values()) + [poc_any]:
        g[out] = NODATA

# Sanity check
valid_rp = np.sum(rp_g[100] != NODATA)
valid_po = np.sum(poc_any != NODATA)
max_rp   = np.max(rp_g[100][rp_g[100] != NODATA]) if valid_rp > 0 else 0
print(f"  Sanity: rp100 valid={valid_rp} max={max_rp:.2f}in  p_occ valid={valid_po}", flush=True)

# ══════════════════════════════════════════════════════════════════════════
print("\n=== Writing GeoTIFFs ===", flush=True)
for rp in RP_YRS:
    write_tif(rp_g[rp], OUT_D/f"stoch_rp_{rp}yr_hail.tif")
    print(f"  stoch_rp_{rp}yr_hail.tif", flush=True)
write_tif(poc_any, OUT_D/"stoch_p_occurrence.tif")
for t in POCC_T:
    tag = f"{t:.2f}".replace(".","p")
    write_tif(poc_g[t], OUT_D/f"stoch_p_occ_{tag}in.tif")
print("  TIFs written.", flush=True)

# ══════════════════════════════════════════════════════════════════════════
print("\n=== Rendering figures ===", flush=True)

HAIL_LEV = [0,.5,1,1.5,2,2.5,3,3.5,4,4.5,5,6,7,9]
PROB_LEV = [0,.02,.05,.10,.15,.20,.30,.40,.50,.65,.80,1.0]
HCMAP, PCMAP = plt.cm.YlOrRd, plt.cm.plasma
EXT  = [LON_MIN-CS/2, LON_MAX+CS/2, LAT_MIN-CS/2, LAT_MAX+CS/2]
LABEL = f"{N_SIM:,}-yr sim, λ={LAM_KM:.0f} km"

def msk(a): return np.ma.masked_where(a==NODATA, a)

def single(arr, title, cmap, levs, path, cbl):
    norm = BoundaryNorm(levs, cmap.N)
    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(msk(arr), extent=EXT, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.08, shrink=0.7)
    cb.set_label(cbl, fontsize=10)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {path.name}", flush=True)

def panel(arrs, titles, suptitle, cmap, levs, path, cbl):
    norm = BoundaryNorm(levs, cmap.N)
    n = len(arrs); cols = 3; rows = (n+2)//3
    fig, axes = plt.subplots(rows, cols, figsize=(20, 7*rows))
    axes = axes.flatten()
    im = None
    for i,(arr,tit) in enumerate(zip(arrs,titles)):
        im = axes[i].imshow(msk(arr), extent=EXT, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
        axes[i].set_title(tit, fontsize=11, fontweight="bold")
        axes[i].set_xlabel("Lon"); axes[i].set_ylabel("Lat")
    for j in range(len(arrs), len(axes)): axes[j].set_visible(False)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    if im:
        cb = fig.colorbar(im, ax=axes[:len(arrs)].tolist(), orientation="vertical", shrink=0.7, pad=0.02)
        cb.set_label(cbl, fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {path.name}", flush=True)

def compare(stoch, hist_path, title, cmap, levs, path, cbl):
    if not hist_path.exists(): return
    with rasterio.open(hist_path) as src:
        ha = src.read(1).astype(np.float32); hnd = src.nodata or -9999
    ha[ha==hnd] = np.nan
    norm = BoundaryNorm(levs, cmap.N)
    fig, axes = plt.subplots(1, 2, figsize=(20,7))
    kw = dict(extent=EXT, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    axes[0].imshow(np.ma.masked_invalid(ha), **kw)
    axes[0].set_title(f"Historical\n(22-yr record, smoothed CDF)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Longitude"); axes[0].set_ylabel("Latitude")
    im = axes[1].imshow(msk(stoch), **kw)
    axes[1].set_title(f"Stochastic\n({LABEL})", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Longitude"); axes[1].set_ylabel("Latitude")
    fig.suptitle(f"Historical vs Stochastic — {title}", fontsize=13, fontweight="bold")
    cb = fig.colorbar(im, ax=axes.tolist(), orientation="horizontal", pad=0.06, shrink=0.6)
    cb.set_label(cbl, fontsize=11)
    plt.tight_layout(); fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"  {path.name}", flush=True)

for rp in RP_YRS:
    single(rp_g[rp], f"Stochastic {rp}-Year Return Period Hail  ({LABEL})",
           HCMAP, HAIL_LEV, OUT_F/f"stoch_rp_{rp}yr_hail.png", "Hail Size (inches)")

panel([rp_g[r] for r in RP_YRS], [f"{r}-Year RP" for r in RP_YRS],
      f"Stochastic Return Period Hail Maps  ({LABEL})",
      HCMAP, HAIL_LEV, OUT_F/"stoch_rp_all_panel.png", "Hail Size (inches)")

single(poc_any, f"Stochastic Annual P(Hail ≥ 0.25\")  ({LABEL})",
       PCMAP, PROB_LEV, OUT_F/"stoch_p_occurrence.png", "Annual Probability")

for t in POCC_T:
    tag = f"{t:.2f}".replace(".","p")
    single(poc_g[t], f"Stochastic Annual P(Hail ≥ {t}\")  ({LABEL})",
           PCMAP, PROB_LEV, OUT_F/f"stoch_p_occ_{tag}in.png", "Annual Probability")

panel([poc_g[t] for t in [0.25,0.50,1.00,1.50,2.00,3.00]],
      [f"P(Hail ≥ {t}\")" for t in [0.25,0.50,1.00,1.50,2.00,3.00]],
      f"Stochastic Annual Hail Occurrence Probabilities  ({LABEL})",
      PCMAP, PROB_LEV, OUT_F/"stoch_p_occ_all_panel.png", "Annual Probability")

compare(rp_g[10],  HIST/"rp_10yr_hail.tif",  "10-Year Return Period Hail",
        HCMAP, HAIL_LEV, OUT_F/"stoch_vs_hist_rp_10yr_comparison.png",  "Hail Size (inches)")
compare(rp_g[100], HIST/"rp_100yr_hail.tif", "100-Year Return Period Hail",
        HCMAP, HAIL_LEV, OUT_F/"stoch_vs_hist_rp_100yr_comparison.png", "Hail Size (inches)")
compare(poc_any,  HIST/"p_occurrence.tif",
        "Annual Hail Occurrence Probability (≥ 0.25\")",
        PCMAP, PROB_LEV, OUT_F/"stoch_vs_hist_p_occurrence_comparison.png", "Annual Probability")

print(f"\n=== Done: {len(list(OUT_D.glob('*.tif')))} TIFs, {len(list(OUT_F.glob('*.png')))} PNGs ===")
