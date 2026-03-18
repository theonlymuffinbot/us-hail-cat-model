# Hail Catastrophe Model — Data Dictionary

**Date:** 2026-03-16

---

## Raster Files — Common Properties

All GeoTIFF rasters share:
- **CRS:** EPSG:4326 (WGS84 geographic)
- **Nodata:** -9999.0 (float32 outputs) or none (uint16 outputs)
- **Compression:** LZW

### 0.25° Grid

| Property | Value |
|---|---|
| Cols × Rows | 236 × 104 |
| Cell size | 0.25° × 0.25° (~28 km at 37°N) |
| Lon extent | −125.00 to −66.25 |
| Lat extent | 24.00 to 50.00 |
| Origin (upper-left) | (−125.00, 50.00) |

---

## Daily Storm Rasters

**Path:** `data/hail_0.25deg/YYYY/hail_YYYYMMDD.tif`

Only written for days with at least one SPC hail report. Missing dates = no hail.

| Band | Content | Unit | Notes |
|---|---|---|---|
| 1 | Hail report count, 0.00–0.24" | integer count | Pea-size or smaller |
| 2 | Hail report count, 0.25–0.49" | integer count | |
| 3 | Hail report count, 0.50–0.74" | integer count | |
| 4 | Hail report count, 0.75–0.99" | integer count | Penny-size |
| 5 | Hail report count, 1.00–1.24" | integer count | Quarter-size — damage threshold |
| 6–8 | Hail report count, 1.25–1.99" | integer count | |
| 9 | Hail report count, 2.00–2.24" | integer count | Golf ball |
| 10–12 | Hail report count, 2.25–2.99" | integer count | |
| 13 | Hail report count, 3.00–3.24" | integer count | Baseball |
| 14–16 | Hail report count, 3.25–3.99" | integer count | |
| 17 | Hail report count, 4.00–4.24" | integer count | Softball |
| 18–29 | Hail report count, 4.25–7.24" | integer count | Grapefruit (Band 29 = 7.00–7.24") |

**Band tag:** `size_range = 'LO-HI hundredths_of_inches'`
**Dtype:** uint16
**Value range:** 0 to ~5 (rarely more than 2 per cell per day at 0.25°)

---

---

## Daily Climatology Files

**Path:** `data/hail_0.25deg_climo/climo_MMDD.tif`

366 files total (including Feb 29). Naming: `climo_0101.tif` through `climo_1231.tif`.

| Band | Content | Unit |
|---|---|---|
| 1 | Summed hail count, 0.00–0.24" | integer count |
| 2 | Summed hail count, 0.25–0.49" | integer count |
| … | … | … |
| 29 | Summed hail count, 7.00–7.24" | integer count |

**Dtype:** uint16
**Dataset tags:** `calendar_day`, `n_applicable_years`, `n_years_with_data`, `years_in_period`
**Band tags:** `bin_lo_hundredths`, `bin_hi_hundredths`, `bin_mid_inches`, `description`

**Leap day:** `climo_0229.tif` uses only leap years: 2004, 2008, 2012, 2016, 2020, 2024 (6 years).

---

## Cat Model Files (`data/hail_0.25deg/`)

### `char_hail_daily.nc`

**Format:** NetCDF4 via xarray
**Dimensions:** `time` (4720) × `lat` (104) × `lon` (236)
**Variable:** `char_hail` — float32, units inches
**Coordinate `time`:** Python datetime objects, storm days only (days with ≥1 SPC report)
**Value:** 0.0 = no hail; positive = max hail size in inches (bin midpoint of highest active band)

### `event_catalog.csv`

| Column | Type | Description |
|---|---|---|
| `event_id` | int | Sequential event identifier (0-indexed) |
| `start_date` | datetime | First day of event |
| `end_date` | datetime | Last day of event |
| `duration_days` | int | Number of days (1 = single-day event) |
| `n_active_cells` | int | Cells with peak hail ≥ 1.0" |
| `footprint_area_km2` | float | n_active_cells × (0.25 × 111)² km² |
| `peak_hail_max_in` | float | Maximum hail size across all cells in event (inches) |
| `peak_hail_mean_in` | float | Mean hail size across active cells (inches) |

### `event_peak_array.npy`

**Shape:** (2928, 104, 236)
**Dtype:** float32
**Units:** inches
**Value:** Peak hail size at each cell for each event. 0.0 = no hail at that cell in that event.
**Memory:** ~274 MB

### Return Period Rasters (`rp_Tyr_hail.tif`)

Available for T = 10, 25, 50, 100, 200, 250, 500 years.

| Property | Value |
|---|---|
| Band | 1 |
| Dtype | float32 |
| Unit | inches |
| Nodata | -9999.0 |
| Method | Zero-inflated lognormal + GPD tail, inverted via brentq |

### `p_occurrence.tif`

Annual probability of at least one hail event ≥ 1.0" at the cell.
Range: 0.0 to 1.0. Nodata: -9999.0.

### `bin_midpoints.json`

Audit trail for bin definitions:

```json
{
  "source": "data/hail_0.25deg/2004/hail_20040301.tif",
  "n_bands": 29,
  "band_descriptions": [null, null, ...],
  "bin_midpoints_in": [0.12, 0.37, 0.62, ...],
  "note": "Bin N (1-indexed): lo=(N-1)*25, hi=lo+24 hundredths of inches"
}
```

### `lambda_km.json`

Spatial correlation parameters:

```json
{
  "lambda_km": 200.0,
  "method": "literature_informed_option_b",
  "candidates_tested": [100.0, 150.0, 200.0],
  "var_ratios_agg": {"100.0": 0.0494, "150.0": 0.0792, "200.0": 0.1194},
  "var_ratios_fp":  {"100.0": 0.0485, "150.0": 0.0729, "200.0": 0.1065},
  "best_fit_lambda_km": 200.0,
  "chosen_lambda_km": 200.0,
  "n_sim_years": 2000,
  "n_corr_cells": 800
}
```

### `cholesky_L.npy`

Shape: (800, 800), dtype float64. Cholesky factor for the spatial correlation matrix at λ=200 km. **Retained for diagnostics only** — the stochastic catalog (step 14) and per-cell maps (step 15) use event-resampling, not copula simulation.

### `corr_cell_idx.npy`

Shape: (800,)
Dtype: int64
Values: Flattened cell indices into the 104×236 grid (row-major order).
To recover (row, col): `row = idx // 236`, `col = idx % 236`

---

## Build Scripts Reference

| Script | Output | Runtime |
|---|---|---|
| `scripts/01_download_population.py` | `data/population/county_population.csv` | 2 min |
| `scripts/02_build_population_trend.py` | `data/population/county_population_trend.csv` | 1 min |
| `scripts/03_download_spc.py` | `data/spc/` | 5 min |
| `scripts/04_build_storm_trends.py` | `data/storms/` | 2 min |
| `scripts/05_build_spatial_beta.py` | `data/storms/county_beta_map.csv` | 1 min |
| `scripts/06_build_hail_rasters.py` | `data/hail_0.05deg/` | 2 hrs |
| `scripts/07_build_hail_debias.py` | `data/hail_0.05deg_pop_debias/` | 30 min |
| `scripts/08_build_hail_agg.py` | `data/hail_0.25deg/` | 30 min |
| `scripts/09_build_hail_climo.py` | `data/hail_0.25deg_climo/` | 5 min |
| `scripts/10_hail_catmodel_pipeline.py` | Cat model outputs | 6 min |
| `scripts/11_build_smooth_cdf.py` | Updated RP rasters | 10 min |
| `scripts/12_build_occurrence_probs.py` | `data/hail_0.25deg/p_occ_*.tif` | 5 min |
| `scripts/13_apply_conus_mask.py` | Masked rasters | 5 min |
| `scripts/14_generate_stochastic_catalog.py` | `data/stochastic/` | 2.5 hrs |
| `scripts/15_render_figures.py` | `docs/figures/{historical,stochastic,analysis}/` | ~15 min |

---

## Stochastic Catalog (`data/stochastic/`)

### `pet_occurrence.csv`

Occurrence Probable Exceedance Table. Each row represents one return period step derived from 50,000 simulated annual maxima (worst single event per year by intensity).

| Column | Type | Unit | Description |
|---|---|---|---|
| `return_period_yr` | float | years | Return period (e.g., 2, 5, 10, 25, 50, 100, 200, 500, 1000, 10000) |
| `max_hail_in` | float | inches | Peak hail size of the worst single event |
| `n_cells` | int | count | Number of 0.25° cells with hail ≥ 1.0" in the worst single event |

### `pet_aggregate.csv`

Aggregate Probable Exceedance Table. Annual aggregate geographic exposure across all events in a year.

| Column | Type | Unit | Description |
|---|---|---|---|
| `return_period_yr` | float | years | Return period |
| `agg_n_cells` | int | count | Annual sum of cell-events (cells ≥ 1.0" across all events in year) |
| `agg_events` | int | count | Total number of events in the year |

### `stochastic_event_summary.csv` *(gitignored)*

One row per simulated event.

| Column | Type | Unit | Description |
|---|---|---|---|
| `sim_year` | int | — | Simulated year (1 to 50,000) |
| `template_event_id` | int | — | Historical event used as spatial template |
| `doy` | int | — | Day of year the event occurs (1–366) |
| `n_cells` | int | count | Cells with hail ≥ 1.0" (damage threshold) |
| `max_hail_in` | float | inches | Maximum hail size across all cells in this event |
| `footprint_km2` | float | km² | Footprint area (n_cells × 770 km²) |

### Annual Tracker Arrays (`ann_*.npy`)

Three NumPy arrays, each shape (50000,) float32 — one value per simulated year.

| File | Description | Unit |
|---|---|---|
| `ann_occ_max_hail.npy` | Annual max hail intensity (worst single event) | inches |
| `ann_occ_n_cells.npy` | Annual cell count of worst single event | count |
| `ann_agg_n_cells.npy` | Annual aggregate cell-events (all events summed) | count |

### `active_flat_idx.npy`

Shape: (N_ACT,), dtype int64. Flat indices (row-major into 104×236) of cells with any hail activity in the historical record. Required input for `15_render_figures.py`.
