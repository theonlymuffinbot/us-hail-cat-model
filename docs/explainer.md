# Storm Reports, Population & the Hail Catastrophe Model: A Plain-Language Deep Dive

## Part 1: Are Storms Actually Getting Worse?

### What Are We Even Trying to Figure Out?

Here's the big question driving the first part of this project:

> **Are severe storms (hail and wind) actually getting more common across the US, or does it just *look* that way because there are more people around to report them?**

Think about it this way: if you put 1,000 new people in a county, you're probably going to get more storm reports from that county — not because storms got worse, but because there are more eyeballs, more phones, more people calling the National Weather Service. We want to separate the "real" trend from the "more people = more reports" effect.

---

### The Data We Used

#### Storm Reports — NOAA's Storm Prediction Center (SPC)

NOAA's Storm Prediction Center tracks every reported severe weather event in the US. They publish daily CSV files — one for tornadoes, one for hail, one for high winds. Each row is a single storm report with the county, state, date, and location.

We downloaded every hail and wind report from **2004 through 2023** — about **13,678 CSV files** total. That's roughly 20 years of daily data.

- **Source:** `https://www.spc.noaa.gov/climo/reports/`
- **What we kept:** Hail reports and wind reports (skipped tornadoes for this analysis)
- **Key fields we used:** County name, state abbreviation

#### Population Data — US Census Bureau

The Census Bureau estimates how many people live in each county every single year. This is called the **Population Estimates Program (PEP)**. We pulled data from three different Census files to cover the full range:

| Time Period | Census File |
|---|---|
| 2000–2009 | Intercensal estimates (`co-est00int-tot.csv`) |
| 2010–2019 | Vintage 2020 estimates (`co-est2020-alldata.csv`) |
| 2020–2023 | Vintage 2023 estimates (`co-est2023-alldata.csv`) |

This gave us annual population for **3,156 counties** across 24 years — about **75,000 data points**.

The join key (how we connect storms to population) is the **FIPS code**: a 5-digit number that uniquely identifies every county in the US. Example: `06037` = Los Angeles County, California.

---

### The Messy Part: Connecting Storms to Counties

Here's where it got tricky. The SPC storm files use county names written by whoever filed the report. The Census files use official county names. These don't always match.

Examples of mismatches we had to handle:
- SPC says `"PRINCE GEORGES"`, Census says `"PRINCE GEORGE'S"` (missing apostrophe + S)
- SPC says `"LA SALLE"`, Census says `"LASALLE"` (space vs. no space)
- SPC says `"CITY OF ROANOKE"`, Census has it as `"ROANOKE"` (Virginia independent cities)
- SPC says `"DONA ANA"`, Census says `"DOÑA ANA"` (missing accent mark)

We built a manual alias table to fix the most common mismatches, and automated cleanup to strip county suffixes like "County", "Parish", "Borough" before matching.

**Final match rate: ~87% of all storm report rows were successfully linked to a county FIPS code.** The ~13% that didn't match were mostly reports with weird county name spellings, offshore reports, or counties that don't exist in the Census data (territories, etc.).

---

### Building Smooth Population Trends

Here's another problem: Census data has "vintage jumps." Every 10 years after a new census, the estimates get revised. If you plot raw Census estimates, you see little jumps in 2010 and 2020 where the estimates suddenly shift — not because the population actually jumped, but because the Census corrected its math.

To fix this, we built a **piecewise log-linear trend** for each county. Fancy words for a simple idea:

1. Collect every real population observation for each county going back to **1980** (yes, we pulled 1980 and 1990s data too, to anchor the trend)
2. Fit a smooth curve through all those points using a regression model that's allowed to change slope at decade boundaries (1990, 2000, 2010, 2020)
3. Use the smooth curve instead of raw Census numbers when we calculate rates

This gives us a clean, continuous population estimate for every county for every year — no jumps, no artifacts.

---

### Three Ways to Normalize

Once we had storms and population matched up, we computed three different ways to remove the population effect:

#### Method A: Rate per 100,000 People

```
rate = storm_count / population × 100,000
```

Simplest approach. "How many storm reports per 100,000 people that year?" This is the same way public health tracks disease rates. Easy to understand, easy to compare across counties of very different sizes.

**Downside:** If storm reports don't scale *linearly* with population (maybe doubling the population doesn't exactly double the reports), this method can still be biased.

#### Method B: Population-Adjusted Count

```
adjusted = storm_count × (population_2004 / population_that_year)
```

"How many reports *would we have seen* if the population had stayed at 2004 levels?" This re-weights historical counts so they're all on the same population base. 2004 is our reference year because that's the start of our SPC data.

**Downside:** Same as Method A — assumes linear scaling.

#### Method C: Regression-Adjusted Index

This one is the most sophisticated.

We fit a statistical model:
```
log(national_storms) = α + β × log(national_population)
```

This estimates how much storm reports *tend* to grow with population, based on the actual data. The **β coefficient** we found was **2.37** — meaning storms scaled with population *faster* than linearly. For every 1% increase in national population, storm reports grew about 2.37%.

Then the adjusted index is:
```
adjusted_index = observed_storms / predicted_from_population_alone
```

Anchored to 1.0 in 2004. A value of 1.5 means 50% more storms than you'd expect from population growth alone. A value of 0.9 means fewer storms than expected.

**This is the most intellectually honest method** because it doesn't assume β=1 (linear). It lets the data tell us how storms and population are actually related, then removes that relationship.

---

### What the National Numbers Show

| Year | Raw Total | Rate/100k | Pop-Adjusted | Regression Index |
|------|-----------|-----------|--------------|-----------------|
| 2004 | 16,982 | 5.76 | 16,982 | 1.000 |
| 2005 | 20,966 | 7.04 | 20,752 | 1.205 |
| 2006 | 24,606 | 8.18 | 24,102 | 1.380 |
| 2007 | 21,161 | 6.96 | 20,510 | 1.157 |
| 2008 | 28,117 | 9.15 | 26,961 | 1.499 |
| 2009 | 20,769 | 6.68 | 19,700 | 1.079 |
| 2010 | 18,812 | 5.99 | 17,648 | 0.952 |
| 2011 | 36,334 | 11.49 | 33,882 | 1.813 |
| 2012 | 25,659 | 8.07 | 23,783 | 1.262 |
| 2013 | 23,893 | 7.47 | 22,010 | 1.159 |
| 2014 | 22,652 | 7.03 | 20,738 | 1.082 |
| 2015 | 23,357 | 7.21 | 21,250 | 1.100 |
| 2016 | 24,358 | 7.47 | 22,020 | 1.130 |
| 2017 | 27,620 | 8.42 | 24,810 | 1.262 |
| 2018 | 23,699 | 7.17 | 21,151 | 1.066 |
| 2019 | 28,391 | 8.54 | 25,173 | 1.258 |
| 2020 | 28,930 | 8.64 | 25,482 | 1.262 |
| 2021 | 22,751 | 6.77 | 19,972 | 0.984 |
| 2022 | 26,373 | 7.82 | 23,069 | 1.131 |
| 2023 | 35,642 | 10.53 | 31,057 | 1.515 |

**Key takeaways:**
- **2011** was an outlier — massive tornado outbreaks and wind events drove the index to 1.81, the highest on record
- **2023** was also a big year (index 1.52) — second highest after 2011
- **2010, 2021** were quiet years (index < 1.0), meaning fewer storms than population growth would predict
- High year-to-year variability — no clear upward trend yet, but later years (2019–2023) trend slightly elevated

---

### The Spatial Twist: Neighborhood Beta

We took the regression idea one step further. Instead of one national β for the whole country, we computed a **local β for every county** — using data from the county itself *plus* all its immediate neighbors.

Why? Because the relationship between population and storm reports varies a lot by geography:
- Dense urban counties: lots of people who might report storms, but not necessarily more actual storms
- Rural counties: few people, so even one major hailstorm might go unreported

By fitting the regression locally (pooling a county with its neighbors), each county gets a β that reflects its own population-density context. Urban cores tend to get a higher β; sparse rural counties get a lower one.

This produces a third normalization at the county level:
```
spatial_adj = raw_count × (pop_ref / pop)^local_β
```

---

## Part 2: The Hail Catastrophe Model

### What's a Catastrophe Model?

The first part of the project was about *trends over time* using county-level report counts. The second part goes much deeper: we built a **catastrophe model** — the kind of thing insurance companies use to price hail policies and understand worst-case scenarios.

A cat model answers questions like:
- "What's the biggest hailstorm a typical point in Kansas should expect over 100 years?"
- "What fraction of years does a cell in Oklahoma see damaging hail (≥1 inch)?"
- "If a hailstorm hits the Dallas metro, how correlated is it with simultaneously hitting Fort Worth?"

To do this, we needed to go from county-level report counts to **spatial raster grids** — essentially a pixelated map of the entire US where every pixel knows its own hail history.

---

### Step 1: From Point Reports to a Raster Grid

The SPC data is a bunch of individual report points — each one is "a hailstone of X size was observed at this lat/lon." That's not ideal for spatial analysis. What we want is a grid where every cell has a count of reports and their sizes.

We created a grid covering the contiguous US:
- **Resolution:** 0.05° × 0.05° (~5.5 km per pixel at mid-latitudes)
- **Extent:** longitude -125° to -66°, latitude 24° to 50°
- **Size:** 1,180 columns × 520 rows = ~614,000 cells
- **Bands:** 29 size bands, each representing a 25-hundredths-of-an-inch range  
  (Band 1 = 0–0.24 inches, Band 2 = 0.25–0.49 inches, ... Band 29 = 7.00–7.24 inches)

Each daily file is a GeoTIFF where each pixel-band stores the *count* of SPC reports in that cell for that size bin on that day.

---

### Step 2: Population Debiasing the Rasters

Here's where we connect the two halves of the project. The population-reporting bias problem doesn't disappear just because we made a grid — densely populated areas still have more reports per actual storm than rural areas.

Using the **local β values** we computed per county in Part 1, we applied a correction factor to each grid cell:

```
correction = (population_2004 / population_year)^local_β
debiased_count = raw_count × correction
```

Every grid cell gets assigned to its nearest county centroid. That county's β determines how aggressively to correct for population growth. The result is a population-debiased raster dataset.

---

### Step 3: Spatial Aggregation

The raw 0.05° grid is quite fine — useful for accuracy, but a lot of data (~4,700 days × 614,000 cells). For the cat model work, we aggregated to a coarser grid:
- **0.25°** (~28 km): 236 columns × 104 rows — the main working resolution

Aggregation method: **sum**. If 5 hail reports land in a 0.25° cell, the aggregated cell stores 5 (not an average). This preserves the total count of reports.

---

### Step 4: Daily Climatology

We built a **climatology** — for each calendar day of the year (January 1st through December 31st, plus leap day), we summed up all the hail counts across every year in the record.

This tells you things like: "Historically, May 20th has more hail activity than November 20th." It's used as background context — a "what's normal for this day of year" baseline.

---

### Step 5: The Core Statistical Model — CDFs and Return Periods

This is the heart of the cat model.

#### What's a Return Period?

A **return period** is a way to express the probability of rare events. "100-year hail" doesn't mean it happens exactly once per century — it means there's a 1% chance of that size or larger occurring in any given year at that location. A 10-year event has a 10% annual chance.

#### The Characteristic Hail Size

For each day with hail, instead of working with 29 separate size-band counts, we reduce each grid cell to a single number: the **maximum active bin midpoint**. If a cell has reports in both the 0.75–0.99 inch band and the 1.00–1.24 inch band, its characteristic size is 1.12 inches (the midpoint of the largest active bin).

#### Annual Maxima

For each grid cell, we find the **annual maximum hail size** — the largest characteristic hail that fell in that cell during each year of the record. This gives us a series of ~22 data points per cell (2004–2025).

#### The Zero-Inflation Problem

Most cells don't get damaging hail every year. A cell in rural Wyoming might have hail 3 years out of 22. This is "zero inflation" — most annual maxima are zero. We handle this with a two-part model:
1. **P(occurrence):** What fraction of years does this cell see any significant hail?
2. **Conditional distribution:** Given that hail does occur, how large is it?

#### Lognormal Body + GPD Tail

For the conditional distribution, we use a composite model:
- **Lognormal distribution** for typical hail sizes (the "body" of the distribution)
- **Generalized Pareto Distribution (GPD)** for extreme sizes above 2 inches (the "tail")

The lognormal is a good fit for everyday hail sizes. The GPD is specifically designed for extreme value modeling — it's the same technique actuaries and engineers use for modeling flood heights, wind speeds, and earthquake intensities.

The composite CDF (cumulative distribution function) at hail size *h* is:
```
If h ≤ 0:    P(hail ≤ h) = 1 − P_occurrence
If h ≤ 2":   P(hail ≤ h) = (1 − P_occ) + P_occ × Lognormal_CDF(h)
If h > 2":   P(hail ≤ h) = (1 − P_occ) + P_occ × [Lognormal up to 2" + GPD tail above 2"]
```

We invert this CDF to get return period estimates: "what size corresponds to the 1% annual exceedance probability?"

#### Spatially-Pooled Fitting

A problem with fitting the model cell-by-cell: 22 years of data is not a lot. Many cells have only 3–5 non-zero observations — not enough to fit a reliable distribution.

The solution: **spatial pooling**. For each cell, we pool all observations from cells within a 150 km radius, weighted by distance (`weight = exp(-distance / 75km)`). This gives ~50–150 effective observations per cell and produces smooth, stable fits across the domain.

---

### Step 6: Storm Event Identification

Rather than treating each day independently, we identified **multi-day storm events** — consecutive days where the hail footprint overlapped spatially (within 2 grid cells).

Results: **2,928 events** identified across 23 years (2004–2025), ranging from 1-day local events to week-long outbreaks covering multiple states.

The event catalog records for each event:
- Start and end date
- Duration in days
- Number of active grid cells (≥1 inch threshold)
- Footprint area (km²)
- Peak hail size (max and mean)

---

### Step 7: Spatial Correlation

For insurance purposes, it's not enough to know what hail size a single cell expects — you also need to know how correlated nearby cells are. If a hailstorm hits one cell, does it also hit the next one over?

We computed the **Spearman rank correlation** between annual maximum hail series for pairs of cells across the domain, then fit a simple model:
```
correlation(distance) = exp(−distance / λ)
```

The fitted decorrelation length was **λ = 33.5 km** — meaning correlation drops to ~37% (1/e) at 33.5 km separation, and is essentially zero by ~100 km.

This is stored as a Cholesky decomposition (a mathematical factoring of the correlation matrix) so it can be used to generate correlated random hail events in simulations.

---

### Step 8: Occurrence Probability Rasters

We computed, for each grid cell, the **annual probability of seeing hail above specific size thresholds**:
- 0.25 inches (small hail, crop damage)
- 0.50 inches (quarter-size, minor vehicle damage)
- 1.50 inches (golf ball, significant damage threshold)
- 2.00 inches (hen egg, severe)
- 3.00 inches (baseball, major)
- 4.00 inches (softball, catastrophic)
- 5.00 inches (grapefruit, extreme)

Each raster shows the fraction of years (out of 22) where that size was exceeded. After smoothing with the same 150 km spatial kernel, these become smooth probability maps across the US.

---

### The Complete Output File Inventory

| Location | File | What It Contains |
|---|---|---|
| `data/population/` | `county_population.csv` | Raw Census estimates, 2000–2023 |
| `data/population/` | `county_population_trend.csv` | Smoothed trend, 1980–2023 |
| `data/storms/` | `county_storm_counts.csv` | Annual hail/wind counts per county |
| `data/storms/` | `county_storm_normalized.csv` | County counts + normalizations |
| `data/storms/` | `national_storm_trends.csv` | National totals, all methods |
| `data/storms/` | `county_storm_spatial.csv` | County data with spatial β |
| `data/storms/` | `county_beta_map.csv` | Local β per county (for mapping) |
| `data/hail/` | `YYYY/hail_YYYYMMDD.tif` | Raw point reports → 0.05° rasters (29 bands, uint8) |
| `data/hail_0.05deg_pop_debias/` | `YYYY/hail_YYYYMMDD.tif` | Population-debiased rasters (float32) |
| `data/hail_0.25deg/` | `YYYY/hail_YYYYMMDD.tif` | 0.25° aggregated rasters |
| `data/hail_0.25deg/` | `char_hail_daily.nc` | Daily characteristic hail stack (xarray NetCDF) |
| `data/hail_0.25deg/` | `event_catalog.csv` | 2,928 storm events, 2004–2025 |
| `data/hail_0.25deg/` | `event_peak_array.npy` | Peak hail per event per cell (n_events × 104 × 236) |
| `data/hail_0.25deg/` | `p_occurrence.tif` | Annual hail occurrence probability |
| `data/hail_0.25deg/` | `p_occ_{T}in.tif` | Occurrence probability by threshold (7 files) |
| `data/hail_0.25deg/` | `rp_{T}yr_hail.tif` | Return period maps: 10/25/50/100/200/250/500yr |
| `data/hail_0.25deg/` | `lambda_km.json` | Fitted decorrelation length (33.5 km) |
| `data/hail_0.25deg/` | `cholesky_L.npy` | 800×800 Cholesky factor for simulation |

| `data/hail_0.25deg_climo/` | `climo_MMDD.tif` | Daily climatology (366 files, 29 bands each) |

---

### What Does This Enable?

With this model in hand, you can:
1. **Look up the hail hazard at any location** — "what does the 100-year hail look like in Wichita, KS?"
2. **Run portfolio loss estimates** — given a set of insured properties, how much would a 1-in-50-year hail year cost?
3. **Generate synthetic event sets** — using the event-resampling stochastic catalog, simulate thousands of years of hail activity preserving real event footprint geometry
4. **Produce risk maps** — show where hail risk is highest after controlling for population reporting bias

---

## Part 3: The 50,000-Year Stochastic Catalog and PET

### What Is a Stochastic Catalog?

A stochastic catalog is a synthetic dataset of millions of simulated events designed to represent what *could* happen over a very long time period — in this case, 50,000 years. It's the bridge between the cat model (which describes hazard at a point) and actual risk quantification.

Why 50,000 years? Because rare events — the 1-in-1,000-year hailstorm, the 1-in-10,000-year event — don't show up in 23 years of historical data. By simulating 50,000 years of synthetic hail activity, we get robust statistics even at very long return periods.

The result: millions of synthetic hail events, each preserving the spatial geometry of a real historical event.

---

### How the Simulation Works (Event-Resampling)

**Step 1: How many events this year?**

The historical record contains a fixed number of events over the years observed. We compute λ = n_events / n_years and treat event occurrence as a Poisson process — each simulated year draws a random count from Poisson(λ). Some years get fewer events, some more.

**Step 2: When does each event occur?**

We fit a KDE-smoothed seasonal distribution from historical event dates (Gaussian, σ=10 days, wrapped at year boundaries). Every synthetic event draws a calendar date from this distribution — more events in late spring and summer, fewer in winter.

**Step 3: Pick a historical template**

For each synthetic event, we pick one real historical event as a template. Events are weighted by seasonal proximity: `weight = exp(−|historical_doy − drawn_doy| / 30)`. A May event gets May templates; an August event gets August templates. This preserves both the seasonality and the real spatial footprint geometry of actual storm systems.

**Step 4: Perturb intensity**

The selected template's hail field is multiplied by a log-normal scaling factor: `exp(σ·ε)` where σ=0.15 and ε ~ N(0,1). This adds ±~15% year-to-year variability in log-space while keeping the spatial structure of the event intact. All values are clamped to [0, 10"] physical ceiling.

**Step 5: Count and store**

Cells with perturbed hail ≥ 1.0" (damage threshold) are counted. Annual trackers record the worst single event (occurrence) and the season total (aggregate).

---

### The Probable Exceedance Table (PET)

After simulating all 50,000 years, we rank the **worst single event per year** by intensity (the "occurrence" metric) and the **annual total geographic exposure** (the "aggregate" metric):

- **`pet_occurrence.csv`:** `return_period_yr, max_hail_in, n_cells` — the worst single event per year
- **`pet_aggregate.csv`:** `return_period_yr, agg_n_cells, agg_events` — annual total cells hit across all events

**Why is max hail nearly flat across return periods?** With ~127 events/year across thousands of active grid cells, near-record hail *somewhere* in CONUS is virtually certain every year. The CONUS-wide annual maximum is always near the top of the distribution. The real exceedance signal lives in the **footprint**: how large an area gets hit simultaneously — which grows substantially from 2-year to 10,000-year return periods.

For location-specific or portfolio work, join `stochastic_event_summary.csv` to cells of interest via `event_peak_array.npy` and the event catalog.

---

### The Output Files

| File | Description |
|---|---|
| `stochastic/stochastic_event_summary.csv` | One row per synthetic event (sim_year, template_event_id, doy, n_cells, max_hail_in, footprint_km2) |
| `stochastic/pet_occurrence.csv` | Occurrence PET: return_period_yr, max_hail_in, n_cells |
| `stochastic/pet_aggregate.csv` | Aggregate PET: return_period_yr, agg_n_cells, agg_events |
| `stochastic/ann_occ_max_hail.npy` | Annual max hail intensity (worst event per year), shape (50000,) |
| `stochastic/ann_occ_n_cells.npy` | Annual n_cells of worst event per year, shape (50000,) |
| `stochastic/ann_agg_n_cells.npy` | Annual aggregate cell-events (all events summed), shape (50000,) |
