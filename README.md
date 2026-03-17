# CONUS Hail Catastrophe Model

**A ground-up probabilistic hail hazard model for the Continental United States, built from 22 years of NOAA SPC storm reports.**

![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Data: NOAA SPC](https://img.shields.io/badge/data-NOAA%20SPC-orange)

---

## Table of Contents

- [What This Produces](#what-this-produces)
- [Data Sources](#data-sources)
- [Quick Start](#quick-start)
- [Pipeline](#pipeline)
- [Directory Structure](#directory-structure)
- [Key Results](#key-results)
- [Known Limitations](#known-limitations)
- [License](#license)

---

## What This Produces

This model builds a full **probabilistic hail hazard layer** for the contiguous United States (CONUS):

- **Return period maps** — Hail size (inches) at 10, 25, 50, 100, 200, 250, and 500-year return periods for every 0.25° grid cell (~28 km) across CONUS.
- **Annual occurrence probability rasters** — Probability that hail exceeds 0.25", 0.50", 1.50", 2.00", 3.00", 4.00", or 5.00" at each cell in a given year.
- **Event catalog** — 2,928 discrete historical storm events (2004–2025), each with dates, footprint, and peak hail size.
- **Daily climatology** — 366 daily climatology rasters (by calendar day) capturing seasonal variation.
- **Spatial correlation structure** — Gaussian copula with exponential decay kernel (λ = 150–200 km) encoded as a Cholesky factor for correlated simulation.
- **50,000-year stochastic catalog** — 6,367,856 synthetic hail events with full CONUS footprints and intensity fields, plus Probable Exceedance Tables (PETs) at standard return periods.

The model uses **no commercial hazard data** — only publicly available NOAA SPC storm reports and US Census Bureau population estimates.

---

## Data Sources

| Source | Description | URL |
|---|---|---|
| NOAA Storm Prediction Center | Daily hail/wind/tornado report CSVs, 2004–present | https://www.spc.noaa.gov/climo/reports/ |
| US Census Bureau PEP | County-level population estimates, 2000–2023 | https://www.census.gov/programs-surveys/popest.html |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/theonlymuffinbot/hail-model.git
cd hail-model
```

### 2. Install dependencies

```bash
pip install numpy pandas scipy rasterio xarray regionmask lmoments3 pyarrow matplotlib
```

Python 3.9+ required.

### 3. Run the pipeline (Steps 1–14 in order)

```bash
# Steps 1–5: Population normalization pipeline (~10 min total)
python scripts/01_download_population.py
python scripts/02_build_population_trend.py
python scripts/03_download_spc.py
python scripts/04_build_storm_trends.py
python scripts/05_build_spatial_beta.py

# Steps 6–13: Hail raster pipeline (~3.5 hrs total)
python scripts/06_build_hail_rasters.py
python scripts/07_build_hail_debias.py
python scripts/08_build_hail_agg.py
python scripts/09_build_hail_climo.py
python scripts/10_hail_catmodel_pipeline.py
python scripts/11_build_smooth_cdf.py
python scripts/12_build_occurrence_probs.py
python scripts/13_apply_conus_mask.py

# Step 14: Stochastic catalog (~2.5 hrs)
python scripts/14_generate_stochastic_catalog.py

# Step 15: Per-cell stochastic maps (~15 min)
python scripts/15_stochastic_maps.py

# Optional: render historical figures
python scripts/render_maps.py
python scripts/render_spatial_corr.py
python scripts/test_lambda_comparison.py
```

Steps 3, 6, 7, 8 are re-runnable — they skip existing files.

---

## Pipeline

| Step | Script | What it does | Approx runtime |
|---|---|---|---|
| 1 | `01_download_population.py` | Download Census PEP county population estimates (2000–2023) | 2 min |
| 2 | `02_build_population_trend.py` | Fit piecewise log-linear population trend (1980–2023) | 1 min |
| 3 | `03_download_spc.py` | Download NOAA SPC daily hail/wind/tornado CSVs (2004–present) | 5 min |
| 4 | `04_build_storm_trends.py` | Normalize storm counts by county population, compute beta | 2 min |
| 5 | `05_build_spatial_beta.py` | Compute spatial neighborhood beta per county | 1 min |
| 6 | `06_build_hail_rasters.py` | Build raw 0.05° daily hail rasters from SPC reports | 2 hrs |
| 7 | `07_build_hail_debias.py` | Apply population debiasing (β = 2.37) to raw rasters | 30 min |
| 8 | `08_build_hail_agg.py` | Aggregate debiased rasters to 0.25° and 0.50° | 30 min |
| 9 | `09_build_hail_climo.py` | Build 366-file daily climatology by calendar day | 5 min |
| 10 | `10_hail_catmodel_pipeline.py` | Fit CDFs, identify events, compute return periods, spatial correlation | 6 min |
| 11 | `11_build_smooth_cdf.py` | Spatially-pooled CDF rebuild with 150 km smoothing | 10 min |
| 12 | `12_build_occurrence_probs.py` | Build annual occurrence probability rasters by threshold | 5 min |
| 13 | `13_apply_conus_mask.py` | Apply CONUS mask and spatial smoothing to output rasters | 5 min |
| 14 | `14_generate_stochastic_catalog.py` | Generate 50,000-year stochastic catalog and PETs | 2.5 hrs |
| 15 | `15_stochastic_maps.py` | Per-cell stochastic RP + p_occ maps from 3,000-yr re-simulation | 15 min |
| — | `render_maps.py` | Render historical return period and p_occ maps to PNG | 5 min |
| — | `render_spatial_corr.py` | Render spatial correlation figures | 2 min |
| — | `test_lambda_comparison.py` | Validate lambda candidates via Monte Carlo | 10 min |

---

## Directory Structure

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
│   ├── 15_stochastic_maps.py
│   ├── render_maps.py
│   ├── render_spatial_corr.py
│   └── test_lambda_comparison.py
│
├── data/
│   ├── population/
│   │   ├── county_population.csv          ← Step 1 output
│   │   └── county_population_trend.csv    ← Step 2 output
│   ├── spc/
│   │   └── YYYY/YYMMDD_rpts_{hail,wind,torn}.csv  ← Step 3 output
│   ├── storms/
│   │   ├── county_storm_counts.csv        ← Step 4 output
│   │   ├── county_storm_normalized.csv    ← Step 4 output
│   │   ├── national_storm_trends.csv      ← Step 4 output
│   │   ├── county_storm_spatial.csv       ← Step 5 output
│   │   └── county_beta_map.csv            ← Step 5 output
│   ├── hail_0.05deg/                      ← Step 6 output (gitignored, ~80 GB)
│   ├── hail_0.05deg_pop_debias/           ← Step 7 output (gitignored)
│   ├── hail_0.25deg/                      ← Step 8+10+11+12+13 output (tifs gitignored)
│   │   ├── bin_midpoints.json             ← committed
│   │   ├── lambda_km.json                 ← committed
│   │   └── event_catalog.csv             ← committed
│   ├── hail_0.25deg_climo/                ← Step 9 output (gitignored)
│   ├── hail_0.50deg/                      ← Step 8 output (gitignored)
│   ├── hail_0.50deg_CDF/                  ← gitignored
│   ├── hail_0.50deg_climo/                ← gitignored
│   └── stochastic/
│       ├── pet_occurrence.csv             ← committed (2.2 MB)
│       ├── pet_aggregate.csv              ← committed (2.0 MB)
│       ├── stochastic_event_summary.csv   ← gitignored (289 MB)
│       ├── stochastic_cell_sample.csv     ← gitignored (15 GB)
│       ├── ann_*.npy                      ← gitignored
│       └── maps/                          ← Step 15 output (committed)
│           ├── stoch_rp_{10,25,50,100,200,500}yr_hail.tif
│           ├── stoch_p_occurrence.tif
│           └── stoch_p_occ_{threshold}in.tif
│
├── docs/
│   ├── README.md                          ← this file
│   ├── executive_summary.md
│   ├── technical_documentation.md
│   ├── data_dictionary.md
│   ├── methodology.md
│   ├── reproduce.md
│   ├── explainer.md
│   └── figures/
│       ├── maps/                          ← historical return period + p_occ maps
│       ├── analysis/                      ← correlation and lambda figures
│       └── stochastic/                    ← Step 15 stochastic maps + comparisons
│
└── logs/
    ├── catmodel_pipeline.log
    ├── hail_agg_build.log
    └── ...
```

---

## Key Results

| Metric | Value |
|---|---|
| Period of record | 2004–2025 (22 complete years + partial 2026) |
| National storm population-scaling exponent β | **2.37** |
| Historical hail events (2004–2025) | **2,928** (mean 127/yr) |
| Grid resolution (primary) | 0.25° (~28 km) |
| Cells with fitted CDFs | 7,155 of 24,544 (~29%) |
| Spatial decorrelation length (CDF layer) | λ = 200 km |
| Stochastic catalog length | **50,000 years** |
| Synthetic events generated | **6,367,856** |
| Spatial decorrelation length (stochastic) | λ = 150 km |

### Occurrence Probable Exceedance Table (CONUS-wide)

| Return Period | Max Hail | Footprint km² |
|---|---|---|
| 2-year | 8.89" | 1,897,434 |
| 10-year | 8.89" | 2,661,336 |
| 100-year | 8.97" | 2,917,767 |
| 500-year | 9.43" | 3,029,426 |
| 10,000-year | 9.53" | 3,196,530 |

> **Note on max hail:** Max hail varies little across return periods because with 127 events/yr across 12,811 active cells, near-record hail *somewhere in CONUS* is virtually certain every year. The **footprint** carries the exceedance signal — it grows 70% from the 2-year to the 10,000-year level. For portfolio or location-specific work, slice `stochastic_event_summary.csv` by cell rather than using these CONUS-wide numbers.

---

## Known Limitations

- **SPC report density bias:** SPC hail reports depend on storm spotters and are denser in populated areas. Urban cells show systematically higher report rates than equivalent rural cells. The population debiasing correction (β = 2.37, Step 7) partially but not fully removes this effect.

- **22-year record:** The GPD tail is extrapolating well beyond the observation window for return periods above ~50 years. L-moments improve fit stability but cannot substitute for a longer empirical record. The 500-year and 10,000-year estimates carry significant uncertainty.

- **Lambda variance gap:** Even at λ = 200 km, the simulated aggregate variance is only ~12% of historical. This gap reflects the inherent sparsity of SPC point reports versus spatially continuous radar-derived hail fields. Future versions should derive λ from NOAA MRMS MESH data.

- **No topographic correction:** Elevation and terrain features (e.g., the Front Range, Appalachians) affect hail survivability and spotter access. No terrain adjustment is applied.

- **No time-trend correction:** Any climate change signal in hail frequency or intensity is treated as noise. The model assumes stationarity.

- **Hazard only — no vulnerability or exposure:** This is a hazard-layer model. It produces hail intensity and probability, not insured losses. Integration with vulnerability curves and exposure databases is required for financial risk quantification.

---

## License

MIT License. See [LICENSE](LICENSE).

Data from NOAA SPC is in the public domain. Data from the US Census Bureau is in the public domain.
