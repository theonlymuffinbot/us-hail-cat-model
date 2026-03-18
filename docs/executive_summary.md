# Hail Catastrophe Model — Executive Summary

**Date:** 2026-03-17
**Status:** Complete — primary hazard layer + 50,000-year stochastic catalog (event-resampling methodology)

---

## What Was Built

A ground-up hail catastrophe model hazard layer for the Continental United States (CONUS), covering the period 2004–2025. The model quantifies the probability and intensity of damaging hail at every 0.25° grid cell (~28 km × 28 km) across CONUS, and characterizes how hail intensities are spatially correlated within and across storm events.

The model is built entirely from publicly available NOAA Storm Prediction Center (SPC) hail report data — no commercial hazard data was used.

---

## Key Outputs

| Product | Description |
|---|---|
| **Return period maps** | Hail size (inches) at 10, 25, 50, 100, 200, 250, and 500-year return periods for every CONUS grid cell |
| **Occurrence probability** | Annual probability of damaging hail (≥1.0") per cell |
| **Event catalog** | Discrete historical storm events from 2004–2025, each with peak hail footprint, centroid, and duration. Events grouped by synoptic-system rule (≤1-day gap + 83 km spatial overlap + 5-day cap) |
| **Daily climatology** | 366 daily climatology rasters (by calendar day) for seasonal risk profiling |
| **Stochastic catalog** | 50,000-year simulation using event-resampling (bootstrap) from historical catalog with log-normal intensity perturbation (σ=0.15). Occurrence and aggregate PETs in n_cells and max_hail_in |

---

## Key Metrics

| Metric | Value |
|---|---|
| Period of record | 2004–2025 (22 complete years + partial 2026) |
| Grid resolution | 0.25° (~28 km) |
| Total events identified | Updated after stage 10 re-run with new event definition |
| Mean events per year | Updated after stage 10 re-run |
| 100-year return period hail (CONUS p90) | Updated after stage 11 re-run |
| 500-year return period hail (CONUS p90) | Updated after stage 11 re-run |
| Cells with fitted CDFs | 8,362 of 24,544 (~34%, CONUS-masked) |
| Stochastic catalog | 50,000 years, event-resampling methodology |
| Synthetic events | Updated after stage 14 re-run |
| PET metrics | max_hail_in + n_cells (occurrence); agg_n_cells + agg_events (aggregate) |

---

## Modeling Approach

**Hazard intensity metric:** Maximum hail size per grid cell per day, derived from 29 SPC size bins (0.12" to 7.12" in 0.25" steps). Bin midpoints read directly from raster metadata.

**CDF fitting:** Zero-inflated two-component model per cell:
- Body: Lognormal distribution (fitted via MLE)
- Tail (≥2.0"): Generalized Pareto Distribution fitted via L-moments (more stable than MLE at small samples)
- Return periods derived by inverting the composite CDF

**Event identification:** Synoptic-system grouping — two hail days are combined into one event if the temporal gap is ≤1 day AND the footprints overlap within an 83 km buffer (3 grid cells). Hard cap of 5 days maximum per event to prevent conflating separate synoptic systems. Single storm days default to individual events. Consistent with NOAA/SPC outbreak definitions and AIR/RMS conventions. Damage threshold: 1.0" (residential asphalt shingles). Event catalog includes centroid_lat/lon.

**Stochastic catalog (event-resampling):** Each simulated event draws a historical event template from the event catalog, weighted by seasonal proximity (exp(−|doy_diff|/30)). Intensity is perturbed by a log-normal factor (σ=0.15) to add year-to-year variability while preserving real spatial footprint geometry.

---

## Known Limitations

1. **SPC report bias:** SPC hail reports are denser in populated areas. Population debiasing (β=2.37) partially corrects this; rural hail may still be undercounted despite the correction.

2. **22-year template library:** The event-resampling stochastic catalog can only generate events whose spatial footprints resemble the 22 years of historical events. Truly novel event geometries (e.g., a massive unprecedented outbreak) are not representable beyond intensity perturbation. A longer record would improve this.

3. **22-year record for CDF fitting:** GPD tail extrapolation for return periods >22 years carries significant uncertainty. L-moments improve fit stability but cannot substitute for a longer observational record.

4. **West coast underrepresentation:** The west coast has <0.7% of national hail reports — cells in CA/OR/WA fall below the minimum observation threshold and appear as nodata in the RP maps. This is physically correct (the Cascades/Sierra block Gulf moisture), not a model error.

5. **Partial 2026 data:** Events through March 11, 2026 are included in the event catalog. These are unlikely to materially affect return period estimates.

---

## Next Steps

| Priority | Task |
|---|---|
| High | Integrate vulnerability curves (lognormal body MDR by roof type/age) |
| High | Attach exposure database (TIV by location, construction class) |
| Medium | Replace λ with MRMS MESH-based spatial correlation |
| Medium | Validate return period maps against industry benchmarks (AIR/RMS) |
| Low | Extend to 0.05° resolution for high-value concentrated portfolios |
