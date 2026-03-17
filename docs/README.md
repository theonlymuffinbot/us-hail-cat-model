# Hail Catastrophe Model

**Built:** 2026-03-14  
**Status:** Primary hazard layer complete

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
| How the event catalog was constructed | Technical Doc §6.2 |
| Why λ = 200 km | Technical Doc §7 |
| What's in `rp_100yr_hail.tif` | Data Dictionary — Return Period Rasters |
| How to use `cholesky_L.npy` | Technical Doc §7.2, Data Dictionary |
| Known issues before production use | Executive Summary — Known Limitations |

## Data Location

```
/Volumes/bitcoin/data/
├── hail_0.25deg/          ← Cat model outputs + storm rasters
├── hail_0.25deg_CDF/      ← Empirical CDF + Weibull + return periods
├── hail_0.25deg_climo/    ← 366 daily climatology files
```
