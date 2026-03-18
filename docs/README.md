# Hail Catastrophe Model

**Built:** 2026-03-17  
**Status:** Primary hazard layer complete — re-running stages 10–15 with corrected event definition and event-resampling stochastic methodology

## Documents

| File | Description |
|---|---|
| [`executive_summary.md`](executive_summary.md) | 1-page overview, key results, limitations, next steps |
| [`technical_documentation.md`](technical_documentation.md) | Full methodology: data, build pipeline, CDF fitting, spatial correlation |
| [`data_dictionary.md`](data_dictionary.md) | Schema for every output file, band definitions, units, nodata values |

## Quick Reference

| Want to know... | See |
|---|---|
| What was built and why | Executive Summary |
| How the event catalog was constructed | Technical Doc — Step 2 (synoptic-system grouping) |
| How the stochastic catalog works | Methodology §4 Step 14 (event-resampling) |
| What's in `rp_100yr_hail.tif` | Data Dictionary — Return Period Rasters |
| What the PET metrics mean | Reproduce Guide — PET metric interpretation |
| Known issues before production use | Executive Summary — Known Limitations |

## Data Location

```
/Volumes/bitcoin/GitHub/us-hail-cat-model/data/
├── hail_0.25deg/          ← Cat model outputs + storm rasters
├── hail_0.25deg_climo/    ← 366 daily climatology files
├── stochastic/            ← Stochastic catalog outputs
```
