#!/usr/bin/env python3
"""
build_hail_climo.py

Generate daily climatology rasters for 0.25deg and 0.50deg hail data.

For each calendar day (MMDD, 0101–1231 + 0229):
  - Sum the 29-band report counts across all available years (2004–2025)
  - Write one .tif per calendar day with 29 bands (uint16, integer counts)

Missing days (no storm file for a year) are treated as zero (no hail).
Leap day (0229) uses only leap years: 2004,2008,2012,2016,2020,2024.

Output:
  data/hail_0.25deg_climo/climo_MMDD.tif  (366 files)
  data/hail_0.50deg_climo/climo_MMDD.tif  (366 files)
"""

import numpy as np
import rasterio
import os
import datetime
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

YEARS = list(range(2004, 2026))  # 2004-2025 inclusive

RESOLUTIONS = [
    {
        'label':   '0.25deg',
        'in_dir':  str(DATA_ROOT / 'hail_0.25deg'),
        'out_dir': str(DATA_ROOT / 'hail_0.25deg_climo'),
    },
    {
        'label':   '0.50deg',
        'in_dir':  str(DATA_ROOT / 'hail_0.50deg'),
        'out_dir': str(DATA_ROOT / 'hail_0.50deg_climo'),
    },
]


def build_file_index(in_dir):
    """
    Returns dict: {(year, 'MMDD'): filepath}
    """
    index = {}
    for year in YEARS:
        year_dir = os.path.join(in_dir, str(year))
        if not os.path.isdir(year_dir):
            continue
        for fname in os.listdir(year_dir):
            if not fname.endswith('.tif'):
                continue
            # fname = hail_YYYYMMDD.tif
            datestr = fname.replace('hail_', '').replace('.tif', '')
            mmdd = datestr[4:8]  # MMDD
            yr   = int(datestr[:4])
            index[(yr, mmdd)] = os.path.join(year_dir, fname)
    return index


def all_calendar_days():
    """
    Return list of 'MMDD' strings for every calendar day, 0101–1231 incl. 0229.
    """
    days = []
    # Use a leap year (2000) to get all 366 days
    ref = datetime.date(2000, 1, 1)
    for i in range(366):
        d = ref + datetime.timedelta(days=i)
        days.append(f'{d.month:02d}{d.day:02d}')
    return days


def leap_years(years):
    return [y for y in years if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))]


def process_resolution(label, in_dir, out_dir):
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  {label} climatology")
    print(f"{'='*60}")
    os.makedirs(out_dir, exist_ok=True)

    # Build file index
    index = build_file_index(in_dir)
    print(f"Indexed {len(index)} storm-day files")

    # Get grid metadata from any file
    sample = list(index.values())[0]
    with rasterio.open(sample) as src:
        H, W = src.shape
        crs = src.crs
        transform = src.transform
        n_bands = src.count  # 29

    out_profile = {
        'driver':    'GTiff',
        'dtype':     'uint16',
        'width':     W,
        'height':    H,
        'count':     n_bands,
        'crs':       crs,
        'transform': transform,
        'compress':  'lzw',
        'predictor': 2,  # horizontal differencing (good for integer data)
    }

    calendar_days = all_calendar_days()  # 366 MMDD strings
    ly = leap_years(YEARS)

    written = 0
    for mmdd in calendar_days:
        # Determine which years apply
        if mmdd == '0229':
            applicable_years = ly
        else:
            applicable_years = YEARS

        # Sum counts across all applicable years
        climo = np.zeros((n_bands, H, W), dtype=np.uint32)
        n_years_with_data = 0

        for yr in applicable_years:
            fpath = index.get((yr, mmdd))
            if fpath is not None:
                with rasterio.open(fpath) as src:
                    climo += src.read().astype(np.uint32)
                n_years_with_data += 1
            # else: year had no hail on this day -> zeros (already)

        # Clip to uint16 (max 65535 — very unlikely to overflow with these counts)
        climo_u16 = np.clip(climo, 0, 65535).astype(np.uint16)

        out_path = os.path.join(out_dir, f'climo_{mmdd}.tif')
        with rasterio.open(out_path, 'w', **out_profile) as dst:
            dst.write(climo_u16)
            # Tag bands with size bin info
            for b in range(1, n_bands + 1):
                lo  = (b - 1) * 25
                hi  = lo + 24
                mid = lo + 12
                dst.update_tags(b,
                    bin_lo_hundredths=str(lo),
                    bin_hi_hundredths=str(hi),
                    bin_mid_inches=f'{mid/100:.2f}',
                    description=f'Hail size {lo/100:.2f}–{hi/100:.2f} in ({lo}–{hi} hundredths)'
                )
            dst.update_tags(
                calendar_day=mmdd,
                years_in_period='2004-2025',
                n_applicable_years=str(len(applicable_years)),
                n_years_with_data=str(n_years_with_data),
                value='summed report counts across all applicable years',
            )

        written += 1
        if written % 50 == 0:
            print(f"  {written}/366 days written  ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n{label} done — {written} files written in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {out_dir}/climo_MMDD.tif")


if __name__ == '__main__':
    for res in RESOLUTIONS:
        process_resolution(**res)
    print("\nAll done.")
