#!/usr/bin/env python3
"""
run_pipeline.py — CONUS Hail Cat Model: Full Pipeline Runner
=============================================================
Runs all 15 pipeline stages in order. Stops on the first failure
and prints a clear error summary.

Usage:
    python run_pipeline.py             # run all stages
    python run_pipeline.py --from 6    # resume from stage 6
    python run_pipeline.py --only 5    # run a single stage
    python run_pipeline.py --dry-run   # print what would run, don't execute

Estimated total runtime: ~6–7 hours on a modern laptop.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent / "scripts"

STAGES = [
    (1,  "01_download_population.py",        "Download Census population data"),
    (2,  "02_build_population_trend.py",      "Build population trend (1980–2023)"),
    (3,  "03_download_spc.py",                "Download NOAA SPC storm reports"),
    (4,  "04_build_storm_trends.py",          "Build county storm trends"),
    (5,  "05_build_spatial_beta.py",          "Compute spatial neighbourhood beta"),
    (6,  "06_build_hail_rasters.py",          "Build raw hail GeoTIFFs (~30 min)"),
    (7,  "07_build_hail_debias.py",           "Population-debias hail rasters"),
    (8,  "08_build_hail_agg.py",              "Aggregate to 0.25° and 0.50°"),
    (9,  "09_build_hail_climo.py",            "Build daily climatology rasters"),
    (10, "10_hail_catmodel_pipeline.py",      "Fit CDFs + spatial correlation (~2 hrs)"),
    (11, "11_build_smooth_cdf.py",            "Build smoothed CDF surface"),
    (12, "12_build_occurrence_probs.py",      "Compute annual occurrence probabilities"),
    (13, "13_apply_conus_mask.py",            "Apply CONUS land mask"),
    (14, "14_generate_stochastic_catalog.py", "Generate 50,000-yr stochastic catalog (~2.5 hrs)"),
    (15, "15_stochastic_maps.py",             "Generate stochastic PET maps (~15 min)"),
]


def fmt_duration(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    if td.days:
        h += td.days * 24
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    elif m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def run_stage(num: int, script: str, description: str, dry_run: bool) -> bool:
    script_path = SCRIPTS_DIR / script
    if not script_path.exists():
        print(f"\n  ❌ Script not found: {script_path}")
        return False

    print(f"\n{'='*60}")
    print(f"  Stage {num:02d}/15 — {description}")
    print(f"  Script: scripts/{script}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    if dry_run:
        print("  [dry-run] skipping execution")
        return True

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_path.parent.parent),  # repo root
        )
        elapsed = time.time() - t0
        if result.returncode == 0:
            print(f"\n  ✅ Stage {num:02d} complete ({fmt_duration(elapsed)})")
            return True
        else:
            print(f"\n  ❌ Stage {num:02d} FAILED (exit code {result.returncode}) after {fmt_duration(elapsed)}")
            return False
    except KeyboardInterrupt:
        print(f"\n  ⚠️  Interrupted by user at stage {num:02d}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Run the CONUS hail cat model pipeline.")
    parser.add_argument("--from", dest="from_stage", type=int, default=1, metavar="N",
                        help="Start from stage N (default: 1)")
    parser.add_argument("--only", dest="only_stage", type=int, default=None, metavar="N",
                        help="Run only stage N")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would run without executing")
    args = parser.parse_args()

    # Filter stages to run
    if args.only_stage is not None:
        stages = [(n, s, d) for n, s, d in STAGES if n == args.only_stage]
        if not stages:
            print(f"Error: no stage {args.only_stage}")
            sys.exit(1)
    else:
        stages = [(n, s, d) for n, s, d in STAGES if n >= args.from_stage]

    if not stages:
        print("No stages to run.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f"  CONUS Hail Cat Model Pipeline")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.dry_run:
        print(f"  Mode: DRY RUN")
    elif args.only_stage:
        print(f"  Mode: single stage ({args.only_stage})")
    else:
        print(f"  Mode: stages {stages[0][0]}–{stages[-1][0]} of 15")
    print(f"{'='*60}")

    pipeline_start = time.time()
    completed = []
    failed = None

    try:
        for num, script, description in stages:
            ok = run_stage(num, script, description, args.dry_run)
            if ok:
                completed.append(num)
            else:
                failed = num
                break
    except KeyboardInterrupt:
        pass

    # Summary
    total_elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print(f"  Pipeline Summary ({fmt_duration(total_elapsed)} total)")
    print(f"{'='*60}")
    for num, script, description in STAGES:
        if num in completed:
            print(f"  ✅ Stage {num:02d} — {description}")
        elif num == failed:
            print(f"  ❌ Stage {num:02d} — {description}  ← FAILED")
        elif num > (failed or 0) and num not in completed:
            skipped = num >= (failed or 999)
            if skipped and failed:
                print(f"  ⏭️  Stage {num:02d} — {description}  (skipped)")

    if failed:
        print(f"\n  ❌ Pipeline stopped at stage {failed}.")
        print(f"     Fix the issue and resume with:  python run_pipeline.py --from {failed}")
        sys.exit(1)
    elif not args.dry_run:
        print(f"\n  🎉 All stages complete!")
    else:
        print(f"\n  [dry-run complete — nothing was executed]")


if __name__ == "__main__":
    main()
