# Hail Catastrophe Model — Executive Summary

**Date:** 2026-03-16
**Status:** Complete — primary hazard layer + 50,000-year stochastic catalog

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
| **Event catalog** | 2,928 discrete storm events from 2004–2025, each with peak hail footprint |
| **Daily climatology** | 366 daily climatology rasters (by calendar day) for seasonal risk profiling |
| **Spatial correlation** | Gaussian copula with fitted decorrelation length λ = 200 km for correlated loss simulation |
| **Stochastic catalog** | 50,000-year simulation, 6,367,856 synthetic events, occurrence and aggregate PETs |

---

## Key Metrics

| Metric | Value |
|---|---|
| Period of record | 2004–2025 (22 complete years + partial 2026) |
| Grid resolution | 0.25° (~28 km) |
| Total events identified | 2,928 |
| Mean events per year | 127 |
| Largest single event footprint | 867 grid cells (~670,000 km²) |
| 100-year return period hail (CONUS p90) | 3.52 inches |
| 500-year return period hail (CONUS p90) | 4.47 inches |
| Cells with fitted CDFs | 7,155 of 24,544 (~29%, rest are outside hail belt) |
| Spatial decorrelation length | 200 km (literature-informed) |
| Stochastic catalog | 50,000 years |
| Synthetic events | 6,367,856 |
| Stochastic lambda | 150 km |
| Occurrence PET 100yr max hail | 8.97 inches |
| Occurrence PET 100yr footprint | 2,917,767 km² |

---

## Modeling Approach

**Hazard intensity metric:** Maximum hail size per grid cell per day, derived from 29 SPC size bins (0.12" to 7.12" in 0.25" steps). Bin midpoints read directly from raster metadata.

**CDF fitting:** Zero-inflated two-component model per cell:
- Body: Lognormal distribution (fitted via MLE)
- Tail (≥2.0"): Generalized Pareto Distribution fitted via L-moments (more stable than MLE at small samples)
- Return periods derived by inverting the composite CDF

**Event identification:** Temporal clustering of hail days, followed by spatial continuity checking to split simultaneous disconnected storms (2-cell buffer, ~56 km). Damage threshold: 1.0" (residential asphalt shingles).

**Spatial correlation:** Gaussian copula with exponential correlation kernel ρ(d) = exp(−d/λ). Three candidate values (100, 150, 200 km) were tested via Monte Carlo. 200 km best reproduces historical aggregate variance and is consistent with peer-reviewed literature for CONUS hail.

---

## Known Limitations

1. **SPC report bias:** SPC hail reports are submitted by storm spotters and are denser in populated areas. Urban cells may show systematically higher report counts than equivalent rural cells. The 0.25° aggregation partially mitigates but does not eliminate this bias.

2. **Spatial correlation underestimation:** Empirical correlation from SPC data yields λ ≈ 30 km — much shorter than physics would suggest — because hail swaths (~5–20 km wide) are narrower than the grid cell. The literature-informed 200 km value is preferred but should be replaced with radar-derived (MRMS MESH) correlation estimates in future model versions.

3. **22-year record:** The GPD tail is extrapolating beyond the observation period for return periods >22 years. The 500-year estimate in particular carries significant uncertainty. L-moments improve stability but cannot substitute for a longer record.

4. **Population debiasing was applied to the raster layer (Step 7, β=2.37).** SPC report bias from urban spotter density partially corrected; rural hail may still be undercounted.

5. **Partial 2026 data:** 15 events through March 11, 2026 are included in the event catalog and annual max arrays. These were not excluded from CDF fitting.

---

## Next Steps

| Priority | Task |
|---|---|
| High | Integrate vulnerability curves (lognormal body MDR by roof type/age) |
| High | Attach exposure database (TIV by location, construction class) |
| Medium | Replace λ with MRMS MESH-based spatial correlation |
| Medium | Validate return period maps against industry benchmarks (AIR/RMS) |
| Low | Extend to 0.05° resolution for high-value concentrated portfolios |
