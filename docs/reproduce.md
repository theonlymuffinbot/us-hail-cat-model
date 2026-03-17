# Reproduction Guide: Storm/Population Pipeline + Hail Catastrophe Model

Reproduces all results from scratch. Run steps in order. Steps 1–5 are the population normalization pipeline; Steps 6–13 are the hail cat model.

## Prerequisites

```bash
pip install numpy rasterio xarray scipy pandas regionmask lmoments3 pyarrow
```

- Python 3.9+
- Internet access (Census Bureau, SPC, and Census centroid URLs)
- ~50 GB disk space (raw hail rasters are large)
- ~50 GB disk space for rasters

---

## Directory Layout

```
data/                         # population normalization pipeline
  download_population.py                  # Step 1
  build_population_trend.py               # Step 2
  download_spc.py                         # Step 3
  build_storm_trends.py                   # Step 4
  build_spatial_beta.py                   # Step 5
  population/
    county_population.csv                 ← Step 1 output
    county_population_trend.csv           ← Step 2 output
    raw_cache/                            ← cached Census raw files
  spc/
    2004/ … 2025/                         ← Step 3 output (daily CSVs)
  storms/
    county_storm_counts.csv               ← Step 4 output
    county_storm_normalized.csv           ← Step 4 output
    national_storm_trends.csv             ← Step 4 output
    county_storm_spatial.csv              ← Step 5 output
    county_beta_map.csv                   ← Step 5 output

data/                    # hail cat model
  build_hail_rasters.py                   # Step 6
  build_hail_debias.py                    # Step 7
  build_hail_agg.py                       # Step 8
  build_hail_climo.py                     # Step 9
  hail_catmodel_pipeline.py               # Step 10
  build_smooth_cdf.py                     # Step 11
  build_occurrence_probs.py               # Step 12
  apply_conus_mask.py                     # Step 13
  hail/
    YYYY/hail_YYYYMMDD.tif                ← Step 6 output (0.05°, raw)
  hail_0.05deg_pop_debias/
    YYYY/hail_YYYYMMDD.tif                ← Step 7 output (0.05°, debiased)
  hail_0.25deg/
    YYYY/hail_YYYYMMDD.tif                ← Step 8 output
    char_hail_daily.nc                    ← Step 10 output
    event_catalog.csv                     ← Step 10 output
    event_peak_array.npy                  ← Step 10 output
    p_occurrence.tif                      ← Step 10 output
    rp_{10,25,50,100,200,250,500}yr_hail.tif  ← Steps 10/11 output
    p_occ_{T}in.tif                       ← Step 12 output
    lambda_km.json                        ← Step 10 output
    cholesky_L.npy                        ← Step 10 output
  hail_0.25deg_CDF/
    daily_cdf.tif                         ← Step 10b output
    weibull_params.tif                    ← Step 10b output
    return_periods.tif                    ← Step 10b output
  hail_0.25deg_climo/
    climo_MMDD.tif (366 files)            ← Step 9 output
```

---

## PART 1 — Population Normalization Pipeline

### Step 1 — Download Census Population Estimates

**Script:** `data/download_population.py`

Downloads three Census PEP vintage files and merges them into a single long-format CSV.

**Sources:**
- 2000–2009: `co-est00int-tot.csv` (intercensal)
- 2010–2019: `co-est2020-alldata.csv` (vintage 2020)
- 2020–2023: `co-est2023-alldata.csv` (vintage 2023)

```bash
cd hail_model
python3 download_population.py
```

**Output:** `population/county_population.csv`
- Columns: `geoid, county_name, state_name, year, population`
- ~75,000 rows (3,156 counties × 24 years, 2000–2023)
- `geoid` = 5-digit FIPS code (zero-padded, e.g. `"06037"`)
- Skips `SUMLEV != 50` (removes state/national totals)
- Skips 2010 from 2000s vintage and 2020 from 2010s vintage (taken from next vintage instead)

---

### Step 2 — Build Smoothed Population Trend

**Script:** `data/build_population_trend.py`

Extends population back to 1980, then fits a piecewise log-linear regression per county to eliminate vintage-jump artifacts.

**Auto-downloads (cached in `population/raw_cache/`):**
- 1980 census: `comp8090.zip`
- 1990–1999 annual estimates: `99c8_00.txt`

**Methodology:**
```
log(pop) = β₀ + β₁·t + β₂·(t−1990)₊ + β₃·(t−2000)₊ + β₄·(t−2010)₊ + β₅·(t−2020)₊
```
where `t = year − 1980` and `(x)₊ = max(0, x)`. Allows different growth rate per decade, continuous at knots.

```bash
python3 build_population_trend.py
```

**Output:** `population/county_population_trend.csv`
- Columns: `geoid, county_name, state_name, year, raw_pop, trend_pop, pop_change`
- Years: 1980–2023 (44 years per county)
- Use `trend_pop` (not `raw_pop`) for all normalization downstream

---

### Step 3 — Download SPC Storm Reports

**Script:** `data/download_spc.py`

Downloads daily storm report CSVs from NOAA SPC for 2004 through yesterday.

**Source:** `https://www.spc.noaa.gov/climo/reports/YYMMDD_rpts_TYPE.csv`

```bash
python3 download_spc.py
```

**Output:** `spc/YYYY/YYMMDD_rpts_{torn,hail,wind}.csv`
- Three types per day: tornadoes, hail, wind
- ~13,678 files total (storm-free days produce empty/missing files)
- Fields: `Time, Location, County, State, Lat, Lon, Comments` + type-specific fields
- Re-runnable: already-downloaded files are skipped

---

### Step 4 — Build Storm Trend Files

**Script:** `data/build_storm_trends.py`

Joins SPC reports to county population via FIPS code; produces three population-normalized outputs.

**Inputs:**
- `spc/YYYY/*.csv` (Step 3)
- `population/county_population_trend.csv` (Step 2)

**County name matching:**
- Strip suffixes: "County", "Parish", "Borough", "Census Area", "Municipality", "City and Borough"
- Convert "Saint" → "St."
- Uppercase everything
- Manual alias table for remaining mismatches:

```python
SPC_ALIASES = {
    ("PRINCE GEORGES",    "MD"): "PRINCE GEORGE'S",
    ("ST. MARYS",         "MD"): "ST. MARY'S",
    ("LASALLE",           "IL"): "LASALLE",
    ("LA SALLE",          "IL"): "LASALLE",
    ("DISTRICT OF COLUM", "DC"): "DISTRICT OF COLUMBIA",
    ("DE KALB",           "IL"): "DEKALB",
    ("LA PORTE",          "IN"): "LAPORTE",
    ("CITY OF ROANOKE",   "VA"): "ROANOKE",
    ("ST. LOUIS CITY",    "MO"): "ST. LOUIS",
    ("DONA ANA",          "NM"): "DOÑA ANA",
    ("BALTIMORE CITY",    "MD"): "BALTIMORE",
    ("DE WITT",           "IL"): "DEWITT",
    ("DE WITT",           "TX"): "DEWITT",
    ("DU PAGE",           "IL"): "DUPAGE",
    ("OBRIEN",            "IA"): "O'BRIEN",
}
```

**Three normalization methods:**

| Method | Formula | Assumption |
|---|---|---|
| A) `rate_per_100k` | `count / pop × 100,000` | β = 1 (linear) |
| B) `adj_count` | `count × (pop_2004 / pop_year)` | β = 1 (linear) |
| C) `reg_total` | `observed / predicted_from_pop` | β estimated from data |

Method C OLS: `log(storms) = α + β·log(population)` → β = **2.37**. Index anchored to 1.0 in 2004.

```bash
python3 build_storm_trends.py
```

**Outputs:**
- `storms/county_storm_counts.csv` — `geoid, county_name, state_name, year, hail, wind, total`
- `storms/county_storm_normalized.csv` — adds `trend_pop, rate_per_100k, adj_count`
- `storms/national_storm_trends.csv` — `year, raw_*, rate_*, adj_*, reg_total, beta_total`

---

### Step 5 — Build Spatial Neighborhood Beta

**Script:** `data/build_spatial_beta.py`

Computes a local β per county using county + all contiguous neighbors. Auto-downloads county adjacency file.

**Methodology:**
Pool (year, storms, population) from county + neighbors. Fit: `log(storms + 1) ~ β·log(population)`
```
spatial_adj = raw_count × (pop_2004 / pop_year)^local_β
```

```bash
python3 build_spatial_beta.py
```

**Outputs:**
- `storms/county_storm_spatial.csv` — adds `n_neighbours, local_beta, spatial_adj`
- `storms/county_beta_map.csv` — one row per county, just `geoid, local_beta` (for mapping)

---

## PART 2 — Hail Catastrophe Model

All scripts in `data/`. Requires Steps 3 and 5 to be complete first.

### Step 6 — Build Raw Hail Rasters

**Script:** `data/build_hail_rasters.py`

Converts SPC hail point reports into daily multi-band GeoTIFFs.

**Grid definition:**
```
CRS:        EPSG:4326 (WGS84)
Resolution: 0.05° × 0.05°  (~5.5 km at mid-latitudes)
Extent:     lon [-125, -66], lat [24, 50]
Dimensions: 1,180 cols × 520 rows
Bands:      29  (size bins in hundredths of inches, step = 25)
  Band 1  = 0–24 hundredths (0–0.24")
  Band 2  = 25–49 hundredths (0.25–0.49")
  ...
  Band 29 = 700–724 hundredths (7.00–7.24")
Values:     uint8 count of reports per cell per band (capped at 255)
Compression: LZW
```

**Input:** `data/spc/YYYY/YYMMDD_rpts_hail.csv`  
**Output:** `data/hail/YYYY/hail_YYYYMMDD.tif`

```bash
cd hail_model
python3 build_hail_rasters.py
```

Re-runnable — skips existing files. Zero-report days write an all-zero file (day was processed, just no hail).

---

### Step 7 — Population-Debias the Rasters

**Script:** `data/build_hail_debias.py`

Applies a per-cell population correction using county local β and population trend data.

**Method:**
1. Each grid cell is assigned to its nearest county centroid (2020 Census population-weighted centroids, auto-downloaded)
2. Per-cell correction factor: `correction = (pop_2004 / pop_year)^local_β`
3. `debiased_count = raw_count × correction`

**Inputs:**
- `data/hail/YYYY/hail_YYYYMMDD.tif` (Step 6)
- `data/storms/county_beta_map.csv` (Step 5)
- `data/population/county_population_trend.csv` (Step 2)
- Census county centroids (auto-downloaded and cached)

**Output:** `data/hail_0.05deg_pop_debias/YYYY/hail_YYYYMMDD.tif`
- Same grid, same 29 bands, float32 (fractional after correction), LZW compressed

```bash
python3 build_hail_debias.py
```

---

### Step 8 — Spatial Aggregation to 0.25°

**Script:** `data/build_hail_agg.py`

Block-sums the 0.05° rasters to coarser resolution.

**Method:** Sum within each N×N block (aggregation = sum, not average — counts are additive)
- 0.25° = 5×5 block sum → 236 × 104 grid

**Input:** `data/hail_0.05deg_pop_debias/` (Step 7)
**Output:**
- `data/hail_0.25deg/YYYY/hail_YYYYMMDD.tif` (uint16)

```bash
python3 build_hail_agg.py
```

Re-runnable — skips existing output files.

---

### Step 9 — Daily Climatology

**Script:** `data/build_hail_climo.py`

For each calendar day of the year (366 days), sums report counts across all years (2004–2025).

**Method:** For each MMDD, sum band counts over all available years. Leap day (0229) uses only leap years: 2004, 2008, 2012, 2016, 2020, 2024. Missing year-days (no storm file) treated as zero.

**Input:** `hail_0.25deg/` (Step 8)
**Output:**
- `hail_0.25deg_climo/climo_MMDD.tif` (366 files, uint16, 29 bands each)

```bash
python3 build_hail_climo.py
```

---

### Step 10 — Primary Cat Model Pipeline (CDF + Spatial Correlation)

**Script:** `data/hail_catmodel_pipeline.py`

The core pipeline. Runs Steps 0–4 of the cat model.

**Input:** `data/hail_0.25deg/YYYY/*.tif` (Step 8)

**Sub-steps:**

**Step 0 — Data discovery:**  
Reads band metadata. Parses bin midpoints from `size_range` band tags: `Band N: lo=(N-1)×25, hi=lo+24 hundredths`. Validates spatial consistency across 100 random files.

**Step 1 — Daily characteristic hail:**  
For each day and cell, computes characteristic hail size = **max active bin midpoint** (the largest bin with count > 0). Result: `char_hail_daily.nc` — a 3D xarray (days × rows × cols).

**Step 2 — Event identification:**  
Threshold: **1.0 inch** (residential shingle damage threshold).
1. Find temporal windows where any cell has hail ≥ 1"
2. Check spatial continuity (binary dilation, 2-cell buffer) — split windows where footprints don't overlap
3. Result: 2,928 events across 23 years (2004–2025)

Event catalog columns: `event_id, start_date, end_date, duration_days, n_active_cells, footprint_area_km2, peak_hail_max_in, peak_hail_mean_in`

Also writes `event_peak_array.npy` — shape `(n_events, 104, 236)` — peak hail per event per cell.

**Step 3 — CDF fitting:**  
For each grid cell (104 × 236 = 24,544 cells):
1. Build annual maximum hail series (22 years, 2004–2025)
2. Fit zero-inflated model: `P_occurrence = fraction of years with any hail`
3. Fit lognormal to non-zero annual maxima (`scipy.stats.lognorm`, `floc=0`)
4. Fit GPD to exceedances above **2.0 inches** (`lmoments3` if available, else scipy MLE)
5. Invert composite CDF for return periods: 10, 25, 50, 100, 200, 250, 500 years

Composite CDF:
```python
F(h) = 1 − P_occ                                     # h ≤ 0
F(h) = (1 − P_occ) + P_occ × Lognormal_CDF(h)        # 0 < h ≤ 2"
F(h) = (1 − P_occ) + P_occ × [Lognormal up to 2" + GPD tail]  # h > 2"
```

**Step 4 — Spatial correlation:**  
1. Subsample up to 800 active cells (cap for Spearman matrix computation)
2. Compute Spearman correlation matrix (800 × 800)
3. Compute distance matrix (km, haversine-equivalent)
4. Fit exponential decay: `corr(d) = exp(−d / λ)` → **λ = 33.5 km**
5. Apply nearest-PSD correction if needed
6. Cholesky decomposition → `cholesky_L.npy` (800 × 800)

```bash
python3 hail_catmodel_pipeline.py
```

**Outputs in `hail_0.25deg/`:**
- `char_hail_daily.nc` — 4,720 days × 104 × 236
- `event_catalog.csv` — 2,928 events
- `event_peak_array.npy` — (2928, 104, 236) float32
- `p_occurrence.tif` — annual occurrence probability
- `rp_{T}yr_hail.tif` — return period rasters (7 files: 10–500yr)
- `bin_midpoints.json` — bin definitions
- `lambda_km.json` — λ = 33.5 km, fitted parameters
- `cholesky_L.npy` — 800 × 800 Cholesky factor
- `corr_cell_idx.npy` — active cell indices

**Also covers `build_hail_cdf.py` (Weibull alternative):**  
For each resolution, separately fits a 2-parameter Weibull to non-zero hail days (not annual maxima), and derives return periods via compound Poisson:
```
x = λ × (−ln(−ln(1−1/T) / rate))^(1/k)
```
Output: `hail_0.25deg_CDF/daily_cdf.tif`, `weibull_params.tif`, `return_periods.tif`

---

### Step 11 — Spatially-Pooled CDF Rebuild

**Script:** `data/build_smooth_cdf.py`

Replaces the cell-by-cell CDF fits from Step 10 with spatially-pooled fits. This is the **final/preferred return period output**.

**Parameters:**
```
POOL_RADIUS_KM = 150     kernel search radius
DECAY_KM       = 75      exponential decay half-radius
MIN_OBS        = 10      minimum pooled non-zero obs before fitting
GPD_THRESHOLD  = 2.0"    above this, use GPD tail
YEARS          = 2004–2025 (22 complete years)
```

**Method:**
For each active cell, gather annual maxima from all cells within 150 km, weighted by `exp(−distance / 75km)`. Fit lognormal + GPD composite to the pooled weighted sample. Invert CDF for return periods.

**Input:** `event_peak_array.npy`, `event_catalog.csv` (Step 10)  
**Output:** Overwrites `rp_{T}yr_hail.tif` and `p_occurrence.tif` in `hail_0.25deg/`

```bash
python3 build_smooth_cdf.py
```

---

### Step 12 — Occurrence Probability Rasters

**Script:** `data/build_occurrence_probs.py`

Computes annual occurrence probability rasters for 7 hail size thresholds.

**Method:**
```
P_occ(threshold) = fraction of years (out of 22) where annual max ≥ threshold
```

**Thresholds:** 0.25, 0.50, 1.50, 2.00, 3.00, 4.00, 5.00 inches

**Input:** `event_peak_array.npy`, `event_catalog.csv` (Step 10)  
**Output:** `hail_0.25deg/p_occ_{T}in.tif` (7 files, float32, values 0–1)

File naming: dots replaced with 'p', e.g. `p_occ_0p25in.tif`, `p_occ_1p50in.tif`

```bash
python3 build_occurrence_probs.py
```

---

### Step 13 — Apply CONUS Mask and Smooth

**Script:** `data/apply_conus_mask.py`

Final cleanup: masks out non-CONUS cells and applies spatial smoothing to p_occ rasters.

**Sub-steps:**

**CONUS mask:**  
Uses `regionmask` with Natural Earth US states polygons (no shapefile download needed). Sets cells outside CONUS to nodata (-9999). Applied to all `rp_*.tif` and `p_occ*.tif` files.

**Spatial smoothing (p_occ threshold rasters only):**  
Parameters match Step 11:
```
POOL_RADIUS_KM = 150
DECAY_KM       = 75
MIN_OBS        = 5
```
Gaussian-weighted spatial average of empirical p_occ values within 150 km. Makes p_occ maps visually consistent with the smoothed RP maps.

```bash
python3 apply_conus_mask.py
```

---

---

### Step 14 — Generate Stochastic Catalog and PET

**Script:** `data/generate_stochastic_catalog.py`

Generates a 50,000-year stochastic occurrence catalog and derives Probable Exceedance Tables (PET).

**Inputs:**
- `hail_0.25deg/cholesky_L_150km.npy` — 800×800 Cholesky factor (Step 10)
- `hail_0.25deg/corr_cell_idx.npy` — seed cell indices (Step 10)
- `hail_0.25deg/p_occurrence.tif` — annual P_occ per cell (Step 11)
- `hail_0.25deg/event_peak_array.npy` — for CDF table build (Step 10)
- `hail_0.25deg/event_catalog.csv` — for Poisson rate + seasonality (Step 10)
- `hail_0.25deg_climo/climo_MMDD.tif` — 366 daily climatology files (Step 9)

**Parameters:**
```
LAMBDA_KM         = 150.0    Spatial decorrelation length (km)
N_SIM_YEARS       = 50,000
LAMBDA_EVENTS     = 127.3    Poisson rate (2928 events / 23 years)
THRESHOLD_IN      = 0.25     Minimum hail size stored (inches)
GPD_THRESH_IN     = 2.0      GPD tail threshold
N_QUANT           = 2000     CDF lookup table resolution
CELL_SAMPLE_YEARS = 2,000    Years of full cell data for validation
RNG_SEED          = 42
```

**Algorithm:**

*Pre-computation (done once, cached to disk):*
1. Build parent-child correlation mapping: each of 12,811 active cells assigned to its nearest seed cell; `ρ = exp(−dist / 150km)`.
2. Build per-cell CDF lookup table `(2000 × 12811)` float32 from lognormal+GPD fits. **Important:** 88 cells with blown-up GPD extrapolations are refitted using empirical quantiles only. All values clamped to ≤ 10" physical ceiling.

*Per simulated year:*
1. Draw `N_events ~ Poisson(127.3)`
2. For each event:
   - Draw calendar date from smoothed historical event-date distribution (seasonal)
   - Load climo seasonal P_occ for that DOY to modulate cell-level occurrence probability
   - Generate correlated z-scores for 800 seed cells via `z = L @ randn(800)`
   - Extend to all 12,811 active cells: `z_cell = ρ·z_parent + √(1−ρ²)·ε`
   - Map to uniform via Φ(z); apply zero-inflation using seasonal P_occ
   - Map non-zero uniforms through CDF lookup table → hail size in inches
   - Retain cells ≥ 0.25"
3. Stream event summary row to CSV; stream cell rows for sample years

*Post-simulation:*
- Sort 50,000 annual maxima to derive occurrence and aggregate PETs

**Run:**
```bash
cd hail_model
PYTHONPATH=/path/to/site-packages python3 generate_stochastic_catalog.py
```

Runtime: ~2.5 hours. Peak memory: ~1.4 GB. CDF table is cached — re-runs skip the build step.

**Outputs in `data/stochastic/`:**

| File | Description |
|---|---|
| `cdf_lookup.npy` | (2000 × 12811) CDF lookup table, cached |
| `cdf_quant_p.npy` | Quantile probability grid |
| `active_flat_idx.npy` | Active cell flat indices |
| `stochastic_event_summary.csv` | 6,367,856 rows: sim_year, event_idx, doy, n_cells, max_hail_in, mean_hail_in, p95_hail_in, footprint_km2 |
| `stochastic_cell_sample.csv` | Full cell-level data for 2,000 validation years |
| `pet_occurrence.csv` | Occurrence PET: return_period_yr, max_hail_in, footprint_km2, n_cells |
| `pet_aggregate.csv` | Aggregate PET: return_period_yr, agg_max_hail_in, agg_footprint_km2 |
| `ann_*.npy` | Annual tracker arrays for PET recomputation |

---

### Step 15 — Per-Cell Stochastic Hazard Maps

**Script:** `scripts/15_stochastic_maps.py`

Runs a fast 3,000-year partial re-simulation using all cached inputs (CDF lookup table, Cholesky factor, p_occurrence, daily climatology) to produce per-cell stochastic return period and occurrence probability maps. All 366 climatology rasters are pre-loaded into memory — no per-event disk I/O. Peak memory ~1.5 GB.

**Requires Step 14 outputs** (cdf_lookup.npy, cholesky_L_150km.npy, active_flat_idx.npy, p_occurrence.tif, climo rasters).

```bash
python scripts/15_stochastic_maps.py
```

**Runtime:** ~15 minutes. **Memory:** ~1.5 GB peak.

**GeoTIFF outputs in `data/stochastic/maps/`:**

| File | Description |
|---|---|
| `stoch_rp_{10,25,50,100,200,500}yr_hail.tif` | Per-cell stochastic return period hail size (inches) |
| `stoch_p_occurrence.tif` | Annual P(hail ≥ 0.25") from simulation |
| `stoch_p_occ_{threshold}in.tif` | Annual P(hail ≥ threshold) for 8 thresholds (0.25"–5.00") |

**Figure outputs in `docs/figures/stochastic/`:**
- Individual maps for each return period and threshold
- `stoch_rp_all_panel.png` — 6-panel return period comparison
- `stoch_p_occ_all_panel.png` — 6-panel occurrence probability comparison
- `stoch_vs_hist_rp_10yr_comparison.png` — side-by-side historical vs stochastic 10yr RP
- `stoch_vs_hist_rp_100yr_comparison.png` — side-by-side historical vs stochastic 100yr RP
- `stoch_vs_hist_p_occurrence_comparison.png` — side-by-side p_occ comparison

---

## Expected Key Results

| Metric | Value |
|---|---|
| National storm β (pop. scaling) | 2.37 |
| SPC match rate | ~87% |
| Total hail events (2004–2025) | 2,928 |
| Spatial decorrelation length λ | 33.5 km |
| 100-yr hail max (CONUS) | ~6 in |
| 100-yr hail p90 (CONUS) | ~3.5 in |
| p_occ ≥ 1.5" max (any cell) | ~0.68 |
| Stochastic catalog events (50,000yr) | 6,367,856 |
| Occurrence PET 2yr footprint | ~1,897,434 km² |
| Occurrence PET 100yr footprint | ~2,917,767 km² |
| Occurrence PET 100yr max hail | 8.97" |

**Note on max hail PET:** The max hail metric shows limited variation across return periods (8.89"–9.53") because with 127 events/year × 12,811 active cells, extreme hail *somewhere in CONUS* is virtually certain every year. The **footprint** column carries the exceedance signal — it grows meaningfully from ~1.9M km² at 2yr to ~3.2M km² at 10,000yr. For per-location or portfolio work, slice `stochastic_event_summary.csv` by cell rather than reading the CONUS-wide PET.

---

## Caveats and Known Issues

- **Match rate ~87%:** ~13% of SPC rows don't match a FIPS code (non-standard names, territories, offshore). The alias table covers the most common cases.
- **No tornadoes in trend pipeline:** `build_storm_trends.py` processes `_hail.csv` and `_wind.csv` only.
- **λ = 33.5 km is short:** The pipeline warns `λ < 100 km — may indicate noise or sparse data`. At 0.25° resolution (~28 km/cell), this decorrelation length implies near-zero correlation beyond 2 cells. May reflect genuine local nature of hail or sparse data at this resolution.
- **Stochastic λ = 150 km:** The lambda comparison test found 200 km best matches historical variance ratios. 150 km was used per design choice; consider re-running with 200 km (`cholesky_L_200km.npy` already exists).
- **22–23 years is a short record:** Return periods beyond ~50 years are extrapolated, not empirically observed. GPD tail fit helps but adds uncertainty.
- **88 cells use empirical CDF only:** These cells had GPD extrapolations that blew up; refitted with empirical quantiles. Their maximum is capped at the observed historical maximum (~9.54").
- **Population debiasing is approximate:** Nearest-centroid grid assignment is coarser than county polygons. Rural counties with large geographic extent may be assigned cells from an adjacent county.
- **β = 2.37 is national-level:** Local betas from Step 5 vary significantly by county. The debiasing in Step 7 uses local β correctly.
- **SPC data starts 2004-03-01:** Earlier dates not available in the daily archive. Some scripts treat 2004 as a partial year.
- **lmoments3 optional:** If not installed, GPD tail fitting falls back to scipy MLE. Both are valid; lmoments3 is generally more robust for small samples.
