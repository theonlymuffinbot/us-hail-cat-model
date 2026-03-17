#!/usr/bin/env python3
"""
Hail Daily Spatial Aggregation
===============================
Reads debiased 0.05° daily hail GeoTIFFs and writes block-summed versions
at 0.25° (5x5) and 0.50° (10x10).

Source:  data/hail_0.05deg_pop_debias/YYYY/hail_YYYYMMDD.tif  (float32, 29 bands)
Output:  data/hail_0.25deg/YYYY/hail_YYYYMMDD.tif  (uint16)

Aggregation: SUM — counts are additive across space.
uint16 is sufficient: max = 100 cells x 255 counts = 25,500 < 65,535.
Resumable: skips files that already exist.
"""

import gc
import time
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

# ── config ────────────────────────────────────────────────────────────────────
SRC_DIR = DATA_ROOT / "hail_0.05deg_pop_debias"
LOG     = LOGS_ROOT / "hail_agg_build.log"

LON_MIN = -125.0
LAT_MAX =   50.0
DX_SRC  =    0.05
NROWS   =  520
NCOLS   = 1180

RESOLUTIONS = {
    "hail_0.25deg": 5,
}

# ── helpers ───────────────────────────────────────────────────────────────────
def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def block_sum(data, factor):
    """data: (bands, rows, cols) -> (bands, rows//factor, cols//factor)"""
    b, r, c = data.shape
    r2, c2 = r // factor, c // factor
    return (data[:, :r2*factor, :c2*factor]
            .reshape(b, r2, factor, c2, factor)
            .sum(axis=(2, 4)))


def make_profile(factor, n_bands):
    dx = DX_SRC * factor
    return {
        "driver":    "GTiff",
        "dtype":     "uint16",
        "width":     NCOLS // factor,
        "height":    NROWS // factor,
        "count":     n_bands,
        "crs":       "EPSG:4326",
        "transform": from_origin(LON_MIN, LAT_MAX, dx, dx),
        "compress":  "lzw",
        "tiled":     True,
        "blockxsize": 256,
        "blockysize": 256,
    }


def validate_outputs() -> bool:
    """Validate all outputs produced by this stage. Returns True if all pass.

    Stage 08 does a FULL scan of hail_0.25deg/:
      - File opens without error (catches corrupt/truncated TIFFs)
      - CRS == EPSG:4326 (catches CRS-mismatch files that break stage 10)
      - Shape == (NROWS//factor, NCOLS//factor) (catches dimension mismatches)

    Uses metadata-only open (no pixel reads) for speed — opening the file
    header is sufficient to detect both corruption and CRS/shape issues.
    """
    import sys
    errors = []
    out_dir = DATA_ROOT / "hail_0.25deg"
    factor  = 5
    exp_h   = NROWS // factor   # 104
    exp_w   = NCOLS // factor   # 236
    exp_crs = "EPSG:4326"

    if not out_dir.exists():
        errors.append(f"Missing directory: {out_dir}")
    else:
        tifs = sorted(out_dir.rglob("hail_????????.tif"))
        if len(tifs) == 0:
            errors.append(f"No TIFFs found in {out_dir}")
        else:
            log(f"Scanning all {len(tifs):,} TIFFs for corruption, CRS and shape consistency...")
            for p in tifs:
                try:
                    with rasterio.open(p) as src:
                        # CRS check
                        crs_str = src.crs.to_epsg() if src.crs else None
                        if str(src.crs) != exp_crs and crs_str != 4326:
                            errors.append(f"CRS mismatch {p.name}: got {src.crs}, expected {exp_crs}")
                        # Shape check
                        if src.height != exp_h or src.width != exp_w:
                            errors.append(f"Shape mismatch {p.name}: got ({src.height},{src.width}), expected ({exp_h},{exp_w})")
                except Exception as e:
                    errors.append(f"Corrupt/unreadable {p.name}: {e}")

    if errors:
        log(f"CRITICAL: Output validation FAILED ({len(errors)} issue(s)):")
        for e in errors[:20]:   # cap at 20 to avoid log spam
            log(f"  ✗ {e}")
        if len(errors) > 20:
            log(f"  ... and {len(errors)-20} more")
        # Delete corrupt/mismatched files so they can be regenerated on re-run
        for e in errors:
            parts = e.split(" ")
            for part in parts:
                if part.endswith(".tif"):
                    p = out_dir / "**" / part
                    matches = list(out_dir.rglob(part))
                    for m in matches:
                        log(f"  Deleting bad file for regeneration: {m}")
                        m.unlink(missing_ok=True)
        return False
    log(f"Output validation passed ✓ ({len(tifs):,} files checked)")
    return True


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    tifs = sorted(SRC_DIR.rglob("hail_????????.tif"))
    log(f"Hail daily aggregation started — {len(tifs):,} source files")
    log(f"  Resolutions: { {k: f'{NCOLS//v}x{NROWS//v}' for k,v in RESOLUTIONS.items()} }")

    done = {r: 0 for r in RESOLUTIONS}
    skip = {r: 0 for r in RESOLUTIONS}
    errors = 0

    for i, src_path in enumerate(tifs, 1):
        rel = src_path.relative_to(SRC_DIR)   # YYYY/hail_YYYYMMDD.tif

        # skip if all outputs exist
        if all((DATA_ROOT / res / rel).exists() for res in RESOLUTIONS):
            for r in RESOLUTIONS:
                skip[r] += 1
            continue

        try:
            with rasterio.open(src_path) as src:
                data      = src.read().astype(np.uint32)   # read as uint32 for safe sum
                n_bands   = src.count
                band_tags = {b: src.tags(b) for b in range(1, n_bands + 1)}
        except Exception as e:
            log(f"  ERROR reading {src_path.name}: {e}")
            errors += 1
            continue

        for res_name, factor in RESOLUTIONS.items():
            dst_path = DATA_ROOT / res_name / rel
            if dst_path.exists():
                skip[res_name] += 1
                continue

            dst_path.parent.mkdir(parents=True, exist_ok=True)
            agg = block_sum(data, factor).astype(np.uint16)

            try:
                with rasterio.open(dst_path, "w", **make_profile(factor, n_bands)) as dst:
                    for b in range(n_bands):
                        dst.write(agg[b], b + 1)
                        if band_tags[b + 1]:
                            dst.update_tags(b + 1, **band_tags[b + 1])
                done[res_name] += 1
            except Exception as e:
                log(f"  ERROR writing {dst_path.name}: {e}")
                errors += 1
                dst_path.unlink(missing_ok=True)
            finally:
                del agg

        # Explicit GC every 100 files to prevent memory buildup on low-RAM systems
        del data, band_tags
        if i % 100 == 0:
            gc.collect()

        if i % 500 == 0 or i == len(tifs):
            log(f"  {i:,}/{len(tifs):,} | "
                f"0.25°: done={done['hail_0.25deg']:,} skip={skip['hail_0.25deg']:,} | "
                f"err={errors}")

    log("Done.")
    for r, d in done.items():
        log(f"  {r}: written={d:,}  skipped={skip[r]:,}")
    log(f"  Errors: {errors}")

    if not validate_outputs():
        import sys
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    if args.validate:
        import sys
        ok = validate_outputs()
        sys.exit(0 if ok else 1)
    main()
