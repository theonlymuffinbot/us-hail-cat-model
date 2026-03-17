#!/usr/bin/env python3
"""
Storm Trend Builder — Hail & Wind, 2004–2023
=============================================
Joins SPC storm reports to county population trends and produces
three population-normalised outputs.

Outputs (written to data/storms/):

  county_storm_counts.csv
      geoid, county_name, state_name, year, hail, wind, total
      Raw annual report counts per county.

  national_storm_trends.csv
      year, raw_hail, raw_wind, raw_total,
      rate_hail, rate_wind, rate_total,        <- per 100k  (Option A)
      adj_hail, adj_wind, adj_total,           <- pop-adjusted count (Option B, ref=2004)
      reg_hail, reg_wind, reg_total            <- regression-adjusted index (Option C)

  county_storm_normalized.csv
      geoid, county_name, state_name, year,
      raw_total, rate_per_100k, adj_count      <- county-level normalised data
      (use this for spatial analysis / mapping)

Methodology
-----------
  A) Per-capita rate:    rate = count / trend_pop * 100_000
  B) Pop-adjusted count: adj  = count * (pop_2004 / trend_pop_year)
     -> "how many reports would we expect if population stayed at 2004 levels?"
  C) Regression residual: fit log(national_storms) ~ beta*log(national_pop)
     -> adjusted_index = observed / predicted_from_pop_alone
"""

import os, csv, re, math, collections, time
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

SPC_DIR  = DATA_ROOT / "spc"
POP_FILE = DATA_ROOT / "population" / "county_population_trend.csv"
OUT_DIR  = DATA_ROOT / "storms"
OUT_DIR.mkdir(exist_ok=True)

LOG_FILE = LOGS_ROOT / "storm_trends_build.log"

YEARS    = range(2004, 2024)   # overlap of SPC data and population trend
REF_YEAR = 2004                # reference year for Option B

# US state name -> 2-letter abbreviation
STATE_ABBR = {
    'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
    'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
    'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA','Kansas':'KS',
    'Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD','Massachusetts':'MA',
    'Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO','Montana':'MT',
    'Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
    'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND',
    'Ohio':'OH','Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI',
    'South Carolina':'SC','South Dakota':'SD','Tennessee':'TN','Texas':'TX',
    'Utah':'UT','Vermont':'VT','Virginia':'VA','Washington':'WA',
    'West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY',
    'District of Columbia':'DC',
}

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    print(msg, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

SUFFIX_RE = re.compile(
    r'\s+(County|Parish|Borough|Census Area|City and Borough|'
    r'Municipality|city|City|Island)\s*$', re.I)

def clean_county_name(name: str) -> str:
    """Normalise Census county name to uppercase bare name for matching."""
    name = SUFFIX_RE.sub('', name).strip()
    # 'Saint' -> 'ST.' to match SPC convention
    name = re.sub(r'\bSaint\b', 'St.', name, flags=re.I)
    return name.upper()

# Manual aliases: SPC county name (upper) -> Census county name (upper, pre-clean)
# Covers apostrophes, spaces, accents, and independent cities
SPC_ALIASES = {
    ("PRINCE GEORGES",   "MD"): "PRINCE GEORGE'S",
    ("ST. MARYS",        "MD"): "ST. MARY'S",
    ("LASALLE",          "IL"): "LASALLE",
    ("LA SALLE",         "IL"): "LASALLE",
    ("DISTRICT OF COLUM","DC"): "DISTRICT OF COLUMBIA",
    ("DE KALB",          "IL"): "DEKALB",
    ("DEKALB",           "IL"): "DEKALB",
    ("LA PORTE",         "IN"): "LAPORTE",
    ("LAPORTE",          "IN"): "LAPORTE",
    ("CITY OF ROANOKE",  "VA"): "ROANOKE",
    ("ST. LOUIS CITY",   "MO"): "ST. LOUIS",
    ("DONA ANA",         "NM"): "DONA ANA",
    ("BALTIMORE CITY",   "MD"): "BALTIMORE",
    ("DONA ANA",         "NM"): "DONA ANA",
    ("DE WITT",          "IL"): "DEWITT",
    ("DE WITT",          "TX"): "DEWITT",
    ("DU PAGE",          "IL"): "DUPAGE",
    ("ST. MARYS",        "WV"): "ST. MARYS",
    ("OBRIEN",           "IA"): "O'BRIEN",
}

# ── build FIPS lookup ─────────────────────────────────────────────────────────
def build_fips_lookup(pop_file: Path):
    """
    Returns lookup: (county_upper, state_abbr) -> geoid
    Uses the first unique match; logs ambiguous cases.
    """
    lookup = {}
    ambiguous = {}

    with open(pop_file) as f:
        for row in csv.DictReader(f):
            if row['year'] != '2004':   # just need one year for names
                continue
            geoid  = row['geoid']
            state  = STATE_ABBR.get(row['state_name'], '')
            if not state:
                continue
            key = (clean_county_name(row['county_name']), state)
            if key in lookup:
                ambiguous[key] = ambiguous.get(key, [lookup[key]]) + [geoid]
            else:
                lookup[key] = geoid

    # For ambiguous entries, prefer the one where geoid[2:] != '510'...'999'
    # (i.e., prefer county over independent city in Virginia, etc.)
    for key, geoids in ambiguous.items():
        counties = [g for g in geoids if int(g[2:]) < 500]
        lookup[key] = counties[0] if counties else geoids[0]

    log(f"  FIPS lookup: {len(lookup):,} county keys  "
        f"({len(ambiguous)} ambiguous names resolved)")
    return lookup

# ── parse SPC files ───────────────────────────────────────────────────────────
def parse_spc_files(spc_dir: Path, fips_lookup: dict):
    """
    Walk all hail + wind CSVs for YEARS.
    Returns: counts[geoid][year]['hail'|'wind'] = int
             unmatched: set of (county_upper, state) that had no FIPS match
    """
    counts    = collections.defaultdict(lambda: collections.defaultdict(
                    lambda: collections.defaultdict(int)))
    unmatched = collections.Counter()
    total_rows = 0
    matched    = 0

    for year in YEARS:
        yr_dir = spc_dir / str(year)
        if not yr_dir.is_dir():
            continue
        for fn in sorted(yr_dir.iterdir()):
            name = fn.name
            if not (name.endswith('_hail.csv') or name.endswith('_wind.csv')):
                continue
            storm_type = 'hail' if '_hail' in name else 'wind'
            try:
                with open(fn, newline='', errors='replace') as f:
                    for row in csv.DictReader(f):
                        county = row.get('County', '').strip().upper()
                        state  = row.get('State',  '').strip().upper()
                        if not county or not state:
                            continue
                        total_rows += 1
                        # Apply alias if SPC name doesn't match Census directly
                        county_resolved = SPC_ALIASES.get((county, state), county)
                        key = (county_resolved, state)
                        geoid = fips_lookup.get(key)
                        if geoid:
                            counts[geoid][year][storm_type] += 1
                            matched += 1
                        else:
                            unmatched[key] += 1
            except Exception:
                pass

    match_pct = matched / total_rows * 100 if total_rows else 0
    log(f"  SPC rows parsed:  {total_rows:,}")
    log(f"  Matched to FIPS:  {matched:,}  ({match_pct:.1f}%)")
    log(f"  Unique unmatched: {len(unmatched):,}")
    # Show top unmatched
    top = unmatched.most_common(10)
    if top:
        log("  Top unmatched county/state pairs:")
        for (cty, st), n in top:
            log(f"    {cty}, {st}: {n} reports")
    return counts

# ── load population trend ─────────────────────────────────────────────────────
def load_population(pop_file: Path):
    """
    Returns pop[geoid][year] = trend_pop
             meta[geoid] = (county_name, state_name)
    """
    pop  = collections.defaultdict(dict)
    meta = {}
    with open(pop_file) as f:
        for row in csv.DictReader(f):
            yr = int(row['year'])
            if yr not in YEARS:
                continue
            g = row['geoid']
            pop[g][yr] = int(row['trend_pop'])
            if g not in meta:
                meta[g] = (row['county_name'], row['state_name'])
    return pop, meta

# ── regression helper ─────────────────────────────────────────────────────────
def regression_adjust(raw_series: dict, pop_series: dict):
    """
    Fit log(storms_t) = alpha + beta*log(pop_t) across available years.
    Returns (beta, {year: adjusted_index}) where adjusted_index =
    observed / predicted_from_pop_alone, anchored so index=1.0 at REF_YEAR.
    """
    years = sorted(set(raw_series) & set(pop_series))
    if len(years) < 5:
        return None, {}
    y = np.array([math.log(raw_series[yr]) for yr in years if raw_series[yr] > 0])
    x = np.array([math.log(pop_series[yr]) for yr in years if raw_series[yr] > 0])
    if len(y) < 5:
        return None, {}
    # OLS: y = a + b*x
    X = np.column_stack([np.ones_like(x), x])
    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta = coef

    # Predicted from population alone
    adjusted = {}
    for yr in years:
        if raw_series.get(yr, 0) > 0:
            predicted = math.exp(alpha + beta * math.log(pop_series[yr]))
            adjusted[yr] = raw_series[yr] / predicted   # ratio: observed/expected
    # Normalise so REF_YEAR = 1.0
    ref_val = adjusted.get(REF_YEAR, 1.0)
    if ref_val:
        adjusted = {yr: v / ref_val for yr, v in adjusted.items()}
    return round(beta, 4), adjusted

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Building storm trend files")

    # 1. FIPS lookup
    log("\n[1/5] Building county FIPS lookup")
    fips_lookup = build_fips_lookup(POP_FILE)

    # 2. Parse SPC files
    log("\n[2/5] Parsing SPC hail + wind files")
    counts = parse_spc_files(SPC_DIR, fips_lookup)
    log(f"  Counties with any report: {len(counts):,}")

    # 3. Load population
    log("\n[3/5] Loading population trend")
    pop, meta = load_population(POP_FILE)

    # 4. Build county-year table + national aggregates
    log("\n[4/5] Computing normalised outputs")

    # Reference population per county at REF_YEAR
    ref_pop = {g: pop[g].get(REF_YEAR, 0) for g in pop}

    # National totals
    nat_raw  = {yr: {'hail': 0, 'wind': 0} for yr in YEARS}
    nat_pop  = {yr: 0 for yr in YEARS}   # sum of county trend pops

    county_rows = []
    all_geoids = sorted(set(pop.keys()))

    for geoid in all_geoids:
        cname, sname = meta.get(geoid, ('', ''))
        for yr in YEARS:
            hail = counts[geoid][yr].get('hail', 0)
            wind = counts[geoid][yr].get('wind', 0)
            total = hail + wind
            p = pop[geoid].get(yr, 0)
            p_ref = ref_pop.get(geoid, 0)

            if p > 0:
                rate   = total / p * 100_000
                adj    = total * (p_ref / p) if p_ref > 0 else total
            else:
                rate   = 0
                adj    = total

            county_rows.append({
                'geoid':        geoid,
                'county_name':  cname,
                'state_name':   sname,
                'year':         yr,
                'hail':         hail,
                'wind':         wind,
                'total':        total,
                'trend_pop':    p,
                'rate_per_100k': round(rate, 4),
                'adj_count':    round(adj, 2),
            })

            nat_raw[yr]['hail'] += hail
            nat_raw[yr]['wind'] += wind
            nat_pop[yr] += p

    # National per-capita & adjusted
    nat_ref_pop = nat_pop.get(REF_YEAR, 1)

    nat_rows_work = {}
    for yr in YEARS:
        h = nat_raw[yr]['hail']
        w = nat_raw[yr]['wind']
        t = h + w
        p = nat_pop[yr]
        p_ref = nat_ref_pop

        rate_h = h / p * 100_000 if p else 0
        rate_w = w / p * 100_000 if p else 0
        rate_t = t / p * 100_000 if p else 0

        adj_h  = h * (p_ref / p) if p else h
        adj_w  = w * (p_ref / p) if p else w
        adj_t  = t * (p_ref / p) if p else t

        nat_rows_work[yr] = {
            'year': yr,
            'raw_hail': h, 'raw_wind': w, 'raw_total': t,
            'rate_hail': round(rate_h, 4),
            'rate_wind': round(rate_w, 4),
            'rate_total': round(rate_t, 4),
            'adj_hail': round(adj_h, 1),
            'adj_wind': round(adj_w, 1),
            'adj_total': round(adj_t, 1),
            'reg_total': None,  # filled below
            'beta_total': None,
        }

    # Regression adjustment (Option C) — national total
    beta_total, reg_total = regression_adjust(
        {yr: nat_raw[yr]['hail'] + nat_raw[yr]['wind'] for yr in YEARS},
        nat_pop)
    log(f"  Regression beta (total storms ~ population): {beta_total}")
    log(f"  (beta~1.0 means storms scale linearly with population; "
        f"beta<1 means population growth overstates the storm increase)")

    for yr in YEARS:
        nat_rows_work[yr]['reg_total']  = round(reg_total.get(yr, 0), 4)
        nat_rows_work[yr]['beta_total'] = beta_total

    # 5. Write outputs
    log("\n[5/5] Writing output files")

    # county_storm_counts.csv
    f1 = OUT_DIR / "county_storm_counts.csv"
    fields1 = ['geoid','county_name','state_name','year','hail','wind','total']
    with open(f1, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields1, extrasaction='ignore')
        w.writeheader(); w.writerows(county_rows)
    log(f"  county_storm_counts.csv  ({len(county_rows):,} rows)")

    # county_storm_normalized.csv
    f2 = OUT_DIR / "county_storm_normalized.csv"
    fields2 = ['geoid','county_name','state_name','year','total',
               'trend_pop','rate_per_100k','adj_count']
    with open(f2, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields2, extrasaction='ignore')
        w.writeheader(); w.writerows(county_rows)
    log(f"  county_storm_normalized.csv  ({len(county_rows):,} rows)")

    # national_storm_trends.csv
    f3 = OUT_DIR / "national_storm_trends.csv"
    fields3 = ['year',
               'raw_hail','raw_wind','raw_total',
               'rate_hail','rate_wind','rate_total',
               'adj_hail','adj_wind','adj_total',
               'reg_total','beta_total']
    nat_rows = [nat_rows_work[yr] for yr in YEARS]
    with open(f3, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields3)
        w.writeheader(); w.writerows(nat_rows)
    log(f"  national_storm_trends.csv  ({len(nat_rows)} rows)")

    # Print national summary table
    log("\n── National Summary ──────────────────────────────────────────────────")
    log(f"{'Year':>4}  {'Raw':>7}  {'Rate/100k':>9}  {'Pop-Adj':>8}  {'Reg-Idx':>8}")
    for r in nat_rows:
        log(f"{r['year']:>4}  {r['raw_total']:>7,}  {r['rate_total']:>9.2f}  "
            f"{r['adj_total']:>8,.0f}  {r['reg_total']:>8.3f}")

    log(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done!")
    log(f"  Output dir: {OUT_DIR}")

if __name__ == "__main__":
    main()
