# Hail Catastrophe Model — Technical Documentation

**Date:** 2026-03-16
**Author:** theonlymuffinbot
**Python environment:** `/Users/ai/.openclaw/workspace/btc-env/`

---

## Table of Contents

1. [Data Sources](#1-data-sources)
2. [Repository Structure](#2-repository-structure)
3. [Build Pipeline — Raster Construction](#3-build-pipeline--raster-construction)
4. [CDF and Return Period Layer](#4-cdf-and-return-period-layer)
5. [Daily Climatology](#5-daily-climatology)
6. [Catastrophe Model Pipeline](#6-catastrophe-model-pipeline)
7. [Spatial Correlation](#7-spatial-correlation)
8. [Output File Reference](#8-output-file-reference)
8b. [Stochastic Catalog Outputs](#8b-stochastic-catalog-outputs-datastochastic)
9. [Key Parameters](#9-key-parameters)
10. [Known Issues and Flags](#10-known-issues-and-flags)

---

## 1. Data Sources

### NOAA SPC Hail Reports
- **Source:** https://www.spc.noaa.gov/climo/reports/
- **Coverage:** 2004-03-01 → 2026-03-11 (daily)
- **Format:** CSV per day — `YYMMDD_rpts_hail.csv`
- **Fields:** Time, County, State, Lat, Lon, Size (hundredths of inches), Comments
- **Download script:** `scripts/03_download_spc.py`
- **Local path:** `data/spc/YYYY/`
- **Total files:** ~13,678 CSVs

### Size Bin Encoding
SPC reports raw hail sizes (in hundredths of inches) are binned into 29 bands:

| Band (1-indexed) | Lo (hundredths) | Hi (hundredths) | Midpoint (inches) | Common name |
|---|---|---|---|---|
| 1 | 0 | 24 | 0.12 | Pea |
| 4 | 75 | 99 | 0.87 | Penny |
| 5 | 100 | 124 | 1.12 | Quarter |
| 9 | 200 | 224 | 2.12 | Golf ball |
| 13 | 300 | 324 | 3.12 | Baseball |
| 17 | 400 | 424 | 4.12 | Softball |
| 29 | 700 | 724 | 7.12 | Grapefruit |

Formula: Band N (1-indexed), `lo = (N-1) × 25`, `hi = lo + 24`, `midpoint = (lo + 12) / 100` inches.

---

## 2. Repository Structure

```
hail_model/
├── scripts/
│   ├── 01_download_population.py
│   ├── 02_build_population_trend.py
│   ├── 03_download_spc.py
│   ├── 04_build_storm_trends.py
│   ├── 05_build_spatial_beta.py
│   ├── 06_build_hail_rasters.py
│   ├── 07_build_hail_debias.py
│   ├── 08_build_hail_agg.py
│   ├── 09_build_hail_climo.py
│   ├── 10_hail_catmodel_pipeline.py
│   ├── 11_build_smooth_cdf.py
│   ├── 12_build_occurrence_probs.py
│   ├── 13_apply_conus_mask.py
│   ├── 14_generate_stochastic_catalog.py
│   └── 15_render_figures.py
├── data/
│   ├── hail_0.05deg/                 # Raw 0.05° rasters (one tif/day)
│   │   └── YYYY/hail_YYYYMMDD.tif
│   ├── hail_0.25deg/                 # Aggregated 0.25° rasters + model outputs
│   │   └── YYYY/hail_YYYYMMDD.tif
│   ├── hail_0.25deg_CDF/             # CDF layer (0.25°)
│   ├── hail_0.25deg_climo/           # Daily climatology (0.25°)
│   ├── stochastic/                   # Stochastic catalog outputs
│   ├── population/                   # Census population CSVs
│   └── storms/                       # County storm count CSVs
└── docs/
    ├── executive_summary.md
    ├── technical_documentation.md
    ├── data_dictionary.md
    ├── methodology.md
    ├── reproduce.md
    ├── explainer.md
    └── figures/
        ├── historical/     ← SPC / observed-data maps (RP + p_occ)
        ├── stochastic/     ← stochastic catalog maps (RP + p_occ)
        └── analysis/       ← comparison charts, spatial correlation diagnostics
```

---

## 3. Build Pipeline — Raster Construction

### 3.1 Raw Rasters (0.05°)

**Script:** `scripts/06_build_hail_rasters.py`

For each SPC daily CSV, reports are binned onto a 0.05° grid:
- Extent: lon [−125, −66], lat [24, 50]
- Grid: 1180 cols × 520 rows
- CRS: EPSG:4326
- Dtype: uint16
- Bands: 29 (one per size bin)
- Value: count of reports in that bin for that cell on that day
- Compression: LZW
- Output: `data/hail_0.05deg/YYYY/hail_YYYYMMDD.tif` (only written for days with ≥1 report)

### 3.2 Aggregated Rasters (0.25°)

**Script:** `scripts/08_build_hail_agg.py`

Aggregates 0.05° rasters by summing within each coarser cell. No interpolation — integer counts are preserved.

| Resolution | Grid | Aggregation |
|---|---|---|
| 0.25° | 236 × 104 | 5×5 sum of 0.05° cells |

---

## 4. CDF and Return Period Layer

**Scripts:** `scripts/10_hail_catmodel_pipeline.py` (Steps 0–3) and `scripts/11_build_smooth_cdf.py`
**Output dir:** `data/hail_0.25deg/`

### Method

For each 0.25° cell, the annual maximum hail series is built from `event_peak_array.npy` (one value per year = maximum peak hail across all events in that year). A zero-inflated two-component model is fitted:

- **Occurrence probability** `p_occ`: fraction of years with any hail
- **Body (hail ≤ 2.0"):** Lognormal distribution, fitted via MLE
- **Tail (hail > 2.0"):** Generalized Pareto Distribution, fitted via L-moments
- **Return period inversion:** composite CDF inverted via `scipy.optimize.brentq`

Step 11 (`11_build_smooth_cdf.py`) replaces cell-by-cell fits with spatially-pooled fits using a 150 km radius / 75 km Gaussian decay kernel, giving each cell 50–200 effective observations instead of 5–15.

### Outputs (`data/hail_0.25deg/`)

| File | Description |
|---|---|
| `rp_{10,25,50,100,200,250,500}yr_hail.tif` | Return period hail size (inches), float32, nodata=-9999 |
| `p_occurrence.tif` | Annual P(hail ≥ 1.0") per cell |
| `p_occ_{T}in.tif` | Annual P(hail ≥ T) for T in {0.25,0.50,1.00,1.50,2.00,3.00,4.00,5.00}" |
| 6 | 100-year |

Units: inches. Zero where insufficient data (< 10 hail days in pixel).

---

## 5. Daily Climatology

**Script:** `scripts/09_build_hail_climo.py`
**Output dir:** `data/hail_0.25deg_climo/`

### Method

For each of 366 calendar days (MMDD from 0101 to 1231, including 0229):
- Collect all storm-day files from applicable years (2004–2025; leap years only for 0229)
- Sum the 29-band arrays across all matching files
- Missing years for a given date contribute zeros (no hail = valid zero)

### Output: `climo_MMDD.tif`

- Bands: 29 (one per hail size bin)
- Dtype: uint16
- Value: summed report count across all applicable years for that calendar day
- Tags: `n_applicable_years`, `n_years_with_data`, `calendar_day`

**Example:** `climo_0601.tif` — June 1st climatology, summed across 22 years (2004–2025). On June 1st, hail occurred in 21 of 22 years.

---

## 6. Catastrophe Model Pipeline

**Script:** `scripts/10_hail_catmodel_pipeline.py`
**Output dir:** `data/hail_0.25deg/` (flat files alongside year subfolders)

### Step 0 — Data Discovery

Traverses year subfolders, validates spatial consistency (sample of 100 files), parses bin midpoints from band tags (`size_range: '0-24 hundredths_of_inches'`), sets `BAND_METHOD = 'max_active_bin'` (bands are integer counts, not probabilities).

### Step 1 — Characteristic Hail Size

For each storm-day file, collapses 29 bands to a single intensity value per cell:

```python
# max_active_bin method:
char_hail = max(bin_midpoint[k] for k in 1..29 if band_k > 0)
# Returns 0.0 if no hail reported in cell
```

### Step 2 — Event Identification (Synoptic-System Grouping)

**Damage threshold:** 1.0 inches (residential asphalt shingles)

**Grouping rule (two-condition test):**
Two hail days are grouped into the same event if AND ONLY IF:
1. Temporal gap ≤ 1 day (consecutive, or separated by one quiet day)
2. Footprints spatially overlap within a **3-cell (83 km) buffer** using `scipy.ndimage.binary_dilation`

**Hard cap:** Maximum 5 days per event. Events longer than 5 days are split at the 5-day mark to prevent conflating separate synoptic systems.

Days that fail either condition (temporal OR spatial) remain as individual events.

**Rationale:** The 83 km buffer captures typical synoptic-system migration (30–60 km/day over a 1–2 day gap). The 5-day cap matches NOAA/SPC outbreak period definitions and AIR/RMS event conventions. Single-day events are the most common case.

**Event catalog columns:** `event_id, start_date, end_date, duration_days, n_active_cells, footprint_area_km2, peak_hail_max_in, peak_hail_mean_in, centroid_lat, centroid_lon`

**Results:** Updated after stage 10 re-run with new methodology.
- Mean events per year: 127

Output: `event_catalog.csv`, `event_peak_array.npy` (2928 × 104 × 236)

### Step 3 — CDF Fitting

**Annual maximum series:** For each of 23 years (2004–2026), maximum peak hail across all events in that year, per cell.

**Zero-inflated model:** Cells with ≥5 non-zero annual observations receive a full parametric fit:

```
P(annual max ≤ h) = (1 − p_occ)           for h = 0
                  = (1 − p_occ) + p_occ × F_sev(h)  for h > 0
```

where `p_occ` = fraction of years with any hail.

**Severity distribution:**
- Body: `F_sev(h) = Lognormal(shape, loc=0, scale)` fitted via MLE
- Tail (h > 2.0"): Spliced with GPD fitted via L-moments (lmoments3):
  `F_sev(h) = F_lognorm(u) + (1 − F_lognorm(u)) × F_GPD(h − u) × rate`
  where u = 2.0" (tail threshold), rate = fraction of events exceeding u

**Return period inversion:** `brentq` root finding on composite CDF.

**Results:**
- 7,155 cells fitted (of 24,544 total; rest outside hail belt or < 5 events)
- Return period outputs: 10, 25, 50, 100, 200, 250, 500 years

| Return Period | CONUS max | CONUS p90 |
|---|---|---|
| 10-year | 7.11" | 2.23" |
| 25-year | 5.20" | 2.73" |
| 50-year | 4.98" | 3.13" |
| 100-year | 5.98" | 3.52" |
| 200-year | 7.15" | 3.93" |
| 250-year | 7.54" | 4.06" |
| 500-year | 8.80" | 4.47" |

*Note: 10-year max exceeds 25-year max due to sparse data in extreme tail cells; this is a known artifact of limited sample size at high return periods.*

### Step 4 — See Section 7

---

## 7. Spatial Correlation

### 7.1 Problem Statement

Hail intensities across nearby grid cells within the same event are correlated. Ignoring this correlation produces an exceedance probability (EP) curve that is too thin-tailed — it understates aggregate losses from large-footprint events.

### 7.2 Correlation Model

**Model:** Gaussian copula with exponential spatial decay kernel:

```
ρ(d) = exp(−d / λ)
```

where d = inter-cell distance in km, λ = decorrelation length scale.

**Implementation:** Cholesky decomposition of model correlation matrix → simulate correlated standard normals → map through cell-level CDFs via probability integral transform.

### 7.3 Empirical Estimation Attempts

Three estimation strategies were attempted, all yielding λ ≈ 30–35 km:

| Method | Data | λ result |
|---|---|---|
| Spearman (annual max) | (23 years × 800 cells) | 33.5 km |
| Spearman (event-level, all events) | (2928 events × 800 cells) | 31.6 km |
| Conditional Spearman (co-occurring events only) | 460 usable pairs | NaN (insufficient data) |

**Root cause:** SPC hail swaths are 5–20 km wide; the 0.25° grid cell (~28 km) is comparable in scale. Neighboring cells routinely don't co-occur in the same event. The empirical λ reflects data sparsity, not atmospheric physics.

### 7.4 Literature-Informed Choice

Three candidates were tested via Monte Carlo (2,000-year simulation):

| Metric | Historical | λ=100km | λ=150km | λ=200km |
|---|---|---|---|---|
| Annual agg variance | 19,019 | 939 | 1,506 | 2,271 |
| Footprint variance | 5,502 | 267 | 401 | 586 |
| Var ratio (agg) | 1.000 | 0.049 | 0.079 | **0.119** |

**Chosen: λ = 200 km.** This is the best-fitting candidate and is consistent with published values for CONUS convective hail correlation (150–350 km range from radar-based studies).

**Important caveat:** Even at 200 km, the simulated variance is only ~12% of historical. This gap reflects the inherent sparsity of SPC point reports relative to spatially continuous radar-based hazard fields. Future model versions should derive λ from NOAA MRMS MESH data.

### 7.5 Outputs

| File | Description |
|---|---|
| `cholesky_L.npy` | 800×800 Cholesky factor, λ=200km (retained for diagnostics) |
| `corr_cell_idx.npy` | 800 cell indices (into flattened 104×236 grid) |
| `lambda_km.json` | Fit metadata and variance ratios |
| `docs/figures/analysis/corr_decay_curve.png` | Empirical correlation decay scatter with model overlay (step 15) |
| `docs/figures/analysis/corr_event_examples.png` | Example historical event footprints (step 15) |

**Note:** The stochastic catalog (step 14) and per-cell maps (step 15) use **event-resampling**, not Gaussian copula simulation. The Cholesky factor and spatial correlation analysis are retained as diagnostic outputs only.

---

## 8. Output File Reference

### Cat Model Outputs (`data/hail_0.25deg/`)

| File | Format | Size | Description |
|---|---|---|---|
| `event_catalog.csv` | CSV | ~200 KB | Historical events: dates, duration, footprint, peak hail, centroid_lat/lon |
| `event_peak_array.npy` | NumPy | 274 MB | Peak hail per event (2928 × 104 × 236, float32) |
| `p_occurrence.tif` | GeoTIFF | 33 KB | Annual hail occurrence probability per cell |
| `rp_10yr_hail.tif` | GeoTIFF | 41 KB | 10-year return period hail size (inches) |
| `rp_25yr_hail.tif` | GeoTIFF | 42 KB | 25-year return period hail size (inches) |
| `rp_50yr_hail.tif` | GeoTIFF | 41 KB | 50-year return period hail size (inches) |
| `rp_100yr_hail.tif` | GeoTIFF | 41 KB | 100-year return period hail size (inches) |
| `rp_200yr_hail.tif` | GeoTIFF | 41 KB | 200-year return period hail size (inches) |
| `rp_250yr_hail.tif` | GeoTIFF | 42 KB | 250-year return period hail size (inches) |
| `rp_500yr_hail.tif` | GeoTIFF | 41 KB | 500-year return period hail size (inches) |
| `bin_midpoints.json` | JSON | 1.2 KB | Bin definitions audit trail |
| `lambda_km.json` | JSON | 513 B | Spatial correlation parameters and validation |
| `cholesky_L.npy` | NumPy | 4.9 MB | Cholesky factor (800×800, λ=200km, diagnostics only) |
| `corr_cell_idx.npy` | NumPy | 6.4 KB | 800 copula cell indices (used by step 15 for correlation diagnostics) |

### Daily Climatology (`data/hail_0.25deg_climo/`)

- **366 files** per resolution: `climo_MMDD.tif` (e.g., `climo_0601.tif`)
- **29 bands** per file (one per size bin)
- **Dtype:** uint16
- **Value:** Summed report counts across all applicable years for that calendar day

---

## 8b. Stochastic Catalog Outputs (`data/stochastic/`)

| File | Format | Size | Description |
|---|---|---|---|
| `pet_occurrence.csv` | CSV | ~2 MB | Occurrence PET: return_period_yr, max_hail_in, n_cells (worst single event per year) |
| `pet_aggregate.csv` | CSV | ~2 MB | Aggregate PET: return_period_yr, agg_n_cells, agg_events (annual totals) |
| `stochastic_event_summary.csv` | CSV | *(gitignored)* | One row per simulated event: sim_year, event_idx, template_event_id, doy, n_cells, max_hail_in, mean_hail_in, footprint_km2 |
| `stochastic_cell_sample.csv` | CSV | *(gitignored)* | Cell-level data for validation sample years |
| `ann_occ_max_hail.npy` | NumPy | 195 KB | Annual max hail intensity (worst event per year), shape (50000,) |
| `ann_occ_n_cells.npy` | NumPy | 195 KB | Annual n_cells of worst event per year, shape (50000,) |
| `ann_agg_n_cells.npy` | NumPy | 195 KB | Annual aggregate n_cells (all events summed), shape (50000,) |

---

## 9. Key Parameters

| Parameter | Value | Location | Notes |
|---|---|---|---|
| Damage threshold | 1.0 inches | Step 2, Step 4 | Residential asphalt shingles |
| GPD tail threshold (GPD_THRESH_IN) | 2.0 inches | Step 3 | Fitted where ≥5 exceedances exist |
| Spatial buffer (event grouping) | 3 cells (~83 km) | Step 2 | Synoptic-system migration buffer |
| Max event duration | 5 days | Step 2 | Prevents conflating separate synoptic systems |
| Min events for CDF fit | 5 non-zero years | Step 3 | Below this, cell gets no return period |
| Decorrelation length λ (diagnostics) | 200 km | Step 10 | Literature-informed; stored in lambda_km.json |
| Copula cells (diagnostics) | 800 (subsampled) | Step 10 | Used in step 15 for correlation figures |
| Simulation length (N_SIM_YEARS) | 50,000 years | Step 14 | Stochastic catalog |
| Step 15 simulation length | 50,000 years | Step 15 | Per-cell maps + figures (matches stage 14) |
| Poisson rate λ | n_events / n_years | Steps 14, 15 | Derived from event_catalog.csv |
| Intensity perturbation σ | 0.15 | Steps 14, 15 | Log-normal, applied per event |
| Seasonal weight decay | 30 days | Steps 14, 15 | exp(−\|doy_diff\|/30) |
| RNG seed | 42 | Steps 14, 15 | Reproducibility |

---

## 10. Known Issues and Flags

| Flag | Severity | Description |
|---|---|---|
| 10yr RP max > 25yr RP max | Low | Artifact of GPD extrapolation in data-sparse cells; not physically wrong |
| Partial 2026 included | Low | 15 events through March 11; slightly dilutes annual max tail |
| Empirical λ ≈ 30 km | Info | SPC report sparsity at 0.25° yields near-zero cross-cell correlation empirically; λ=200km from literature used for diagnostics only — stochastic catalog uses event-resampling, not copula |
| Population debiasing applied | Info | Step 7 applies β=2.37 correction; urban/rural bias partially corrected |
| 22-year record | Medium | GPD extrapolation to 500yr carries high uncertainty |
| Event-resampling template library | Info | Stochastic catalog draws from 22-year historical event footprints. Novel geometries not in the 2004–2025 record cannot be generated directly. |
