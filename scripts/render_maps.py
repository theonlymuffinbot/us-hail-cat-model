#!/usr/bin/env python3
"""
render_maps.py — Fifth pass. Core fix: use contourf instead of imshow.

contourf with a masked array:
  - Naturally stops at the CONUS boundary (no bilinear bleed into Canada)
  - Produces smooth gradient contours (no grid-cell blockiness)
  - Handles NaN/nodata correctly via np.ma.masked_where
"""

import os, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

import numpy as np
import rasterio
import regionmask
from scipy.ndimage import gaussian_filter, distance_transform_edt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ROOT    = str(DATA_ROOT / "hail_0.25deg")
OUT_DIR = str(REPO_ROOT / "docs" / "figures" / "maps")
os.makedirs(OUT_DIR, exist_ok=True)

PROJ      = ccrs.LambertConformal(central_longitude=-96, central_latitude=37.5,
                                   standard_parallels=(29.5, 45.5))
DATA_PROJ = ccrs.PlateCarree()
LON0, LON1 = -125.0, -66.25
LAT0, LAT1 =   24.0,  50.00

NROWS, NCOLS = 104, 236
CELL_DEG     = 0.25
LAT_ORIG, LON_ORIG = 50.0, -125.0

OCEAN_COLOR = '#a8cfe0'
LAND_BG     = '#ebebeb'   # light grey — non-CONUS land

# ── CONUS mask ────────────────────────────────────────────────────────────────
print("Building CONUS mask...")
_r = np.arange(NROWS); _c = np.arange(NCOLS)
_R, _C = np.meshgrid(_r, _c, indexing='ij')
LAT_GRID = LAT_ORIG - _R * CELL_DEG - CELL_DEG / 2
LON_GRID = LON_ORIG + _C * CELL_DEG + CELL_DEG / 2

_us   = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50
_m2d  = _us.mask(LON_GRID[0, :], LAT_GRID[:, 0])
CONUS = (_m2d >= 0).values   # (104, 236)
print(f"  CONUS cells: {CONUS.sum():,}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def add_map_features(ax, lw=0.65):
    ax.set_extent([LON0, LON1, LAT0, LAT1], crs=DATA_PROJ)
    ax.add_feature(cfeature.LAND.with_scale('50m'),
                   facecolor=LAND_BG, zorder=1)
    ax.add_feature(cfeature.OCEAN.with_scale('50m'),
                   facecolor=OCEAN_COLOR, zorder=1)
    ax.add_feature(cfeature.LAKES.with_scale('50m'),
                   facecolor=OCEAN_COLOR, zorder=1)
    ax.add_feature(cfeature.STATES.with_scale('50m'),
                   edgecolor='#555555', linewidth=lw, facecolor='none', zorder=5)
    ax.add_feature(cfeature.COASTLINE.with_scale('50m'),
                   edgecolor='#2a2a2a', linewidth=lw + 0.15, zorder=5)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'),
                   edgecolor='#2a2a2a', linewidth=lw + 0.3, zorder=5)

def read_raster(path, nodata=-9999.0):
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float64)
    data[data == nodata] = np.nan
    return data

def fill_nan_nearest_conus(arr):
    """Fill NaN within CONUS only using nearest valid neighbour."""
    nan_m = np.isnan(arr) & CONUS
    if not nan_m.any():
        return arr.copy()
    idx   = distance_transform_edt(nan_m, return_distances=False,
                                   return_indices=True)
    filled = arr.copy()
    filled[nan_m] = arr[idx[0][nan_m], idx[1][nan_m]]
    return filled

def prep_rp(path, sigma=1.8):
    data = read_raster(path)
    data = np.clip(data, 0, 10)
    data = fill_nan_nearest_conus(data)
    data = gaussian_filter(np.nan_to_num(data, 0), sigma=sigma)
    # Mask everything outside CONUS
    masked = np.ma.masked_where(~CONUS, data)
    return masked

def prep_occ(path, sigma=1.4):
    data = read_raster(path)
    data = np.where(CONUS & ~np.isnan(data), data, 0.0)
    data = gaussian_filter(data, sigma=sigma)
    masked = np.ma.masked_where(~CONUS, data)
    return masked

def draw_contourf(ax, data, lons, lats, cmap, norm, levels, zorder=3):
    """Draw filled contours on a cartopy axes."""
    cf = ax.contourf(lons, lats, data, levels=levels,
                     cmap=cmap, norm=norm,
                     transform=DATA_PROJ, zorder=zorder,
                     extend='neither')
    return cf

def savefig(fig, name, dpi=300):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    kb = os.path.getsize(path) // 1024
    print(f"  Saved {name}  ({kb} KB)")

# Grid lon/lat centres for contourf
LONS2D = LON_GRID   # (104,236)
LATS2D = LAT_GRID   # (104,236)

# ── Return period colormap — finer levels for contourf ───────────────────────
RP_COLORS_FINE = [
    '#daf0fc','#b8e2f8','#8ecae6','#63b5d8','#4da6d6','#3594c4',
    '#2a8f50','#3da048','#5cb85c','#7ec850','#a8d84e','#c8e040',
    '#f5e030','#f5cc2c','#f5b82e','#f5a028','#f57f20','#f06018',
    '#e03520','#cc1a10','#b01010','#8c0c0c','#720000','#500000','#380000',
]
RP_LEVELS = np.concatenate([
    np.arange(0, 1.0, 0.1),
    np.arange(1.0, 2.0, 0.1),
    np.arange(2.0, 3.0, 0.2),
    np.arange(3.0, 4.5, 0.25),
    [4.5, 5.0, 6.0, 7.0, 10.0],
])
RP_NORM_CF = BoundaryNorm(RP_LEVELS, 256, clip=True)

# Colormap interpolated from key anchor points
from matplotlib.colors import LinearSegmentedColormap
RP_CMAP_CF = LinearSegmentedColormap.from_list('hail_rp', [
    (0/10, '#daf0fc'),    # 0"  — very pale blue
    (0.5/10,'#4da6d6'),   # 0.5"
    (1.0/10,'#2a8f50'),   # 1.0" — green (damage threshold)
    (1.5/10,'#a8d84e'),   # 1.5"
    (2.0/10,'#f5e030'),   # 2.0" — golf ball / yellow
    (2.5/10,'#f57f20'),   # 2.5" — orange
    (3.0/10,'#e03520'),   # 3.0" — baseball / red
    (3.5/10,'#b01010'),
    (4.0/10,'#720000'),   # 4.0" — softball / dark red
    (1.0,   '#200000'),   # 10"  — extreme
], N=512)

# Discrete colorbar bins for legend
RP_CB_BOUNDS = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75,
                2.0, 2.5, 3.0, 3.5, 4.0, 10.0]
RP_CB_NORM   = BoundaryNorm(RP_CB_BOUNDS, 256)
RP_CB_LABELS = ['< 0.25"','0.25–0.5"','0.5–0.75"','0.75–1.0"',
                '1.0–1.25"\n(Quarter)','1.25–1.5"','1.5–1.75"','1.75–2.0"',
                '2.0–2.5"\n(Golf Ball)','2.5–3.0"',
                '3.0–3.5"\n(Baseball)','3.5–4.0"','> 4.0"\n(Softball+)']

SOURCE_RP = ('Source: NOAA SPC Hail Reports 2004–2025  ·  '
             'Zero-inflated Lognormal + GPD tail (L-moments)  ·  '
             '150 km spatial pooling  ·  0.25° grid')

# Probability colormap
PROB_CMAP_CF = LinearSegmentedColormap.from_list('hail_prob', [
    (0.0, '#f7fbff'),
    (0.2, '#9ecae1'),
    (0.5, '#3182bd'),
    (0.8, '#08519c'),
    (1.0, '#08306b'),
], N=512)

SOURCE_OCC = ('Source: NOAA SPC Hail Reports 2004–2025  ·  '
              'Smoothed annual occurrence probability  ·  0.25° grid')

OCC_SCALES = {
    "p_occ_0p25in.tif": ([0,0.10,0.20,0.30,0.40,0.55,0.70,0.85,1.01],
                          ['0–10%','10–20%','20–30%','30–40%','40–55%','55–70%','70–85%','85–100%']),
    "p_occ_0p50in.tif": ([0,0.10,0.20,0.30,0.40,0.55,0.70,0.85,1.01],
                          ['0–10%','10–20%','20–30%','30–40%','40–55%','55–70%','70–85%','85–100%']),
    "p_occ_1p00in.tif":  ([0,0.05,0.10,0.20,0.30,0.45,0.60,0.75,1.01],
                          ['0–5%','5–10%','10–20%','20–30%','30–45%','45–60%','60–75%','75–100%']),
    "p_occ_1p50in.tif":  ([0,0.05,0.10,0.15,0.20,0.30,0.40,0.55,1.01],
                          ['0–5%','5–10%','10–15%','15–20%','20–30%','30–40%','40–55%','55–100%']),
    "p_occ_2p00in.tif":  ([0,0.02,0.05,0.08,0.12,0.18,0.26,0.36,1.01],
                          ['0–2%','2–5%','5–8%','8–12%','12–18%','18–26%','26–36%','36–100%']),
    "p_occ_3p00in.tif":  ([0,0.01,0.02,0.04,0.06,0.09,0.13,0.20,1.01],
                          ['0–1%','1–2%','2–4%','4–6%','6–9%','9–13%','13–20%','20–100%']),
    "p_occ_4p00in.tif":  ([0,0.005,0.01,0.02,0.04,0.06,0.09,0.14,1.01],
                          ['0–0.5%','0.5–1%','1–2%','2–4%','4–6%','6–9%','9–14%','14–100%']),
    "p_occ_5p00in.tif":  ([0,0.001,0.003,0.006,0.01,0.02,0.03,0.05,1.01],
                          ['0–0.1%','0.1–0.3%','0.3–0.6%','0.6–1%','1–2%','2–3%','3–5%','5–100%']),
}
OCC_ORDER = [
    ("p_occ_0p25in.tif", "≥ 0.25\"",            "a"),
    ("p_occ_0p50in.tif", "≥ 0.50\"",            "b"),
    ("p_occ_1p00in.tif", "≥ 1.00\" (Quarter)",   "c"),
    ("p_occ_1p50in.tif", "≥ 1.50\"",            "d"),
    ("p_occ_2p00in.tif", "≥ 2.00\" (Golf Ball)", "e"),
    ("p_occ_3p00in.tif", "≥ 3.00\" (Baseball)",  "f"),
    ("p_occ_4p00in.tif", "≥ 4.00\" (Softball)",  "g"),
    ("p_occ_5p00in.tif", "≥ 5.00\" (Grapefruit)","h"),
]

RP_FILES = [
    ("rp_10yr_hail.tif",  "10-Year Return Period",  "rp_10yr"),
    ("rp_25yr_hail.tif",  "25-Year Return Period",  "rp_25yr"),
    ("rp_50yr_hail.tif",  "50-Year Return Period",  "rp_50yr"),
    ("rp_100yr_hail.tif", "100-Year Return Period", "rp_100yr"),
    ("rp_200yr_hail.tif", "200-Year Return Period", "rp_200yr"),
    ("rp_250yr_hail.tif", "250-Year Return Period", "rp_250yr"),
    ("rp_500yr_hail.tif", "500-Year Return Period", "rp_500yr"),
]

# ════════════════════════════════════════════════════════════════════════════
# Individual RP maps
# ════════════════════════════════════════════════════════════════════════════
print("\nRendering return period maps...")
for fname, rp_label, outname in RP_FILES:
    data = prep_rp(os.path.join(ROOT, fname))

    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
    add_map_features(ax, lw=0.7)

    # contourf — stops at CONUS mask, smooth gradients
    cf = ax.contourf(LONS2D, LATS2D, data, levels=RP_LEVELS,
                     cmap=RP_CMAP_CF, transform=DATA_PROJ, zorder=3,
                     extend='max')

    # Discrete colorbar using anchor colours
    sm = plt.cm.ScalarMappable(cmap=RP_CMAP_CF, norm=RP_CB_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, fraction=0.025)
    cbar.set_label('Max Hail Size (inches)', fontsize=12, labelpad=12)
    cbar.set_ticks([(RP_CB_BOUNDS[i]+RP_CB_BOUNDS[i+1])/2
                    for i in range(len(RP_CB_BOUNDS)-1)])
    cbar.set_ticklabels(RP_CB_LABELS, fontsize=9)
    cbar.ax.tick_params(length=0)

    ax.set_title(f'CONUS Hail Hazard — {rp_label}\nMaximum Expected Hail Size',
                 fontsize=14, fontweight='bold', pad=12)
    fig.text(0.12, 0.03, SOURCE_RP, fontsize=7.5, color='#444', ha='left')
    savefig(fig, f"{outname}_hail.png")

# ════════════════════════════════════════════════════════════════════════════
# Combined RP panel
# ════════════════════════════════════════════════════════════════════════════
print("\nRendering combined RP panel...")
fig = plt.figure(figsize=(28, 18), facecolor='white')
for idx, (fname, label, _) in enumerate(RP_FILES):
    ax   = fig.add_subplot(2, 4, idx + 1, projection=PROJ)
    data = prep_rp(os.path.join(ROOT, fname))
    add_map_features(ax, lw=0.45)
    ax.contourf(LONS2D, LATS2D, data, levels=RP_LEVELS,
                cmap=RP_CMAP_CF, transform=DATA_PROJ, zorder=3, extend='max')
    ax.set_title(label, fontsize=11, fontweight='bold', pad=4)
    ax.text(0.02, 0.97, f'({chr(97+idx)})', transform=ax.transAxes,
            fontsize=10, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

ax_cb = fig.add_subplot(2, 4, 8); ax_cb.axis('off')
sm = plt.cm.ScalarMappable(cmap=RP_CMAP_CF, norm=RP_CB_NORM); sm.set_array([])
cbar = fig.colorbar(sm, ax=ax_cb, orientation='vertical', fraction=0.55, pad=0.06)
cbar.set_label('Max Hail Size (inches)', fontsize=13, labelpad=14)
cbar.set_ticks([(RP_CB_BOUNDS[i]+RP_CB_BOUNDS[i+1])/2
                for i in range(len(RP_CB_BOUNDS)-1)])
cbar.set_ticklabels(RP_CB_LABELS, fontsize=10)
cbar.ax.tick_params(length=0)
fig.suptitle('CONUS Hail Return Period Hazard Surfaces\n'
             'Source: NOAA SPC 2004–2025  ·  150 km Spatial Pooling  ·  0.25° Grid',
             fontsize=14, fontweight='bold', y=0.995)
fig.tight_layout(rect=[0, 0.01, 1, 0.97], h_pad=2.0, w_pad=1.5)
savefig(fig, "rp_all_panel.png")

# ════════════════════════════════════════════════════════════════════════════
# Individual occurrence maps
# ════════════════════════════════════════════════════════════════════════════
print("\nRendering individual occurrence maps...")
for fname, label, pl in OCC_ORDER:
    fpath = os.path.join(ROOT, fname)
    if not os.path.exists(fpath):
        print(f"  Skipping {fname} (not yet available)")
        continue
    data = prep_occ(fpath)
    bounds, tick_labels = OCC_SCALES[fname]
    max_val = float(np.nanmax(data))
    # Fine levels for smooth contourf
    fine_levels = np.linspace(0, min(max_val * 1.05, bounds[-2] * 1.1), 100)
    fine_levels = np.unique(np.concatenate([fine_levels, bounds[1:-1]]))

    # Prob cmap scaled to this threshold's range
    prob_cmap_local = LinearSegmentedColormap.from_list('prob', [
        '#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b'], N=256)

    fig = plt.figure(figsize=(14, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=PROJ)
    add_map_features(ax, lw=0.7)
    ax.contourf(LONS2D, LATS2D, data, levels=fine_levels,
                cmap=prob_cmap_local, transform=DATA_PROJ, zorder=3,
                extend='neither')

    # Discrete colorbar
    from matplotlib.colors import LinearSegmentedColormap as LSC
    sm = plt.cm.ScalarMappable(
        cmap=LSC.from_list('', ['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b']),
        norm=BoundaryNorm(bounds, 256))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', pad=0.02, fraction=0.025)
    cbar.set_label('Annual Occurrence Probability', fontsize=12, labelpad=12)
    cbar.set_ticks([(bounds[i]+bounds[i+1])/2 for i in range(len(bounds)-1)])
    cbar.set_ticklabels(tick_labels, fontsize=9)
    cbar.ax.tick_params(length=0)

    outname = fname.replace('.tif', '')
    ax.set_title(f'CONUS Annual Hail Occurrence — {label}\n'
                 f'Smoothed Annual Probability of Occurrence',
                 fontsize=13, fontweight='bold', pad=12)
    fig.text(0.12, 0.03, SOURCE_OCC, fontsize=7.5, color='#444', ha='left')
    savefig(fig, f"{outname}.png")

# ════════════════════════════════════════════════════════════════════════════
# Combined occurrence panel
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.colors import LinearSegmentedColormap as LSC
print("\nRendering combined occurrence panel...")
fig2 = plt.figure(figsize=(26, 14), facecolor='white')
gs   = GridSpec(2, 4, figure=fig2, hspace=0.08, wspace=0.06,
                left=0.01, right=0.99, top=0.91, bottom=0.06)

for idx, (fname, label, pl) in enumerate(OCC_ORDER):
    row, col = divmod(idx, 4)
    fpath = os.path.join(ROOT, fname)
    if not os.path.exists(fpath):
        ax = fig2.add_subplot(gs[row, col])
        ax.set_facecolor('#f0f0f0')
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#cccccc')
        ax.text(0.5, 0.5, 'Data\npending', transform=ax.transAxes,
                ha='center', va='center', fontsize=13, color='#999999',
                style='italic', fontweight='bold')
        ax.set_title(f'({pl}) {label}', fontsize=9.5, fontweight='bold', pad=3)
        print(f"  Placeholder for {fname} (not yet available)")
        continue

    ax = fig2.add_subplot(gs[row, col], projection=PROJ)
    data = prep_occ(fpath)
    bounds, tick_labels = OCC_SCALES[fname]
    max_val = float(np.nanmax(data))
    fine_levels = np.linspace(0, min(max_val * 1.05, bounds[-2] * 1.1), 60)
    fine_levels = np.unique(np.concatenate([fine_levels, bounds[1:-1]]))
    prob_cmap_local = LSC.from_list('', ['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b'])

    add_map_features(ax, lw=0.38)
    cf = ax.contourf(LONS2D, LATS2D, data, levels=fine_levels,
                     cmap=prob_cmap_local, transform=DATA_PROJ, zorder=3,
                     extend='neither')
    ax.set_title(f'({pl}) {label}', fontsize=9.5, fontweight='bold', pad=3)

    sm = plt.cm.ScalarMappable(
        cmap=LSC.from_list('', ['#f7fbff','#c6dbef','#6baed6','#2171b5','#08306b']),
        norm=BoundaryNorm(bounds, 256))
    sm.set_array([])
    cbar = fig2.colorbar(sm, ax=ax, orientation='horizontal', pad=0.04,
                         fraction=0.06, aspect=22)
    cbar.ax.tick_params(labelsize=5.5, length=0)
    sel = [0, 2, 4, 6]
    cbar.set_ticks([(bounds[i]+bounds[i+1])/2 for i in sel])
    cbar.set_ticklabels([tick_labels[i] for i in sel], fontsize=6.5)
    cbar.set_label(f'Max: {max_val:.1%}', fontsize=6.5, labelpad=1)

fig2.suptitle(
    'CONUS Annual Hail Occurrence Probability by Size Threshold\n'
    'Source: NOAA SPC 2004–2025  ·  150 km spatial pooling  ·  0.25° Grid',
    fontsize=13, fontweight='bold', y=0.995)
savefig(fig2, "p_occ_all_panel.png")

print(f"\nDone. All maps in {OUT_DIR}/")
for f in sorted(os.listdir(OUT_DIR)):
    if f.endswith('.png'):
        kb = os.path.getsize(os.path.join(OUT_DIR, f)) // 1024
        print(f"  {f}  ({kb} KB)")
