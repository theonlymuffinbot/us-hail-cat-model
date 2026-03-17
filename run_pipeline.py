#!/usr/bin/env python3
"""
run_pipeline.py — CONUS Hail Cat Model: Full Pipeline Runner
=============================================================
Runs all 15 pipeline stages in order. Stops on any failure.
Logs each stage to logs/<stage>.log and prints a live summary.

Usage:
    python run_pipeline.py              # Run all stages
    python run_pipeline.py --from 6     # Resume from stage 6
    python run_pipeline.py --only 5     # Run a single stage
    python run_pipeline.py --dry-run    # Print stages without running
    python run_pipeline.py --skip 15    # Skip a stage (comma-separated)
    python run_pipeline.py --validate   # Validate outputs of all (or selected) stages
"""

import argparse
import importlib
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── dependency preflight ──────────────────────────────────────────────────────
REQUIRED_PACKAGES = [
    ("numpy",       "numpy"),
    ("pandas",      "pandas"),
    ("scipy",       "scipy"),
    ("rasterio",    "rasterio"),
    ("xarray",      "xarray"),
    ("regionmask",  "regionmask"),
    ("lmoments3",   "lmoments3"),
    ("pyarrow",     "pyarrow"),
    ("matplotlib",  "matplotlib"),
    ("cartopy",     "cartopy"),
]

def check_dependencies() -> bool:
    """Check all required packages are importable. Print missing ones and return False if any."""
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append(pip_name)
    if missing:
        print(f"\033[91m✗ Missing required packages: {', '.join(missing)}\033[0m")
        print(f"  Run: pip install {' '.join(missing)}")
        print(f"  Or:  pip install -r requirements.txt\n")
        return False
    return True

REPO_ROOT = Path(__file__).resolve().parent
SCRIPTS   = REPO_ROOT / "scripts"
LOGS      = REPO_ROOT / "logs"

STAGES = [
    (1,  "01_download_population.py",       "Download Census population data",          "~1 min"),
    (2,  "02_build_population_trend.py",    "Build county population trend (1980–2023)","~1 min"),
    (3,  "03_download_spc.py",              "Download NOAA SPC storm reports",          "~5 min"),
    (4,  "04_build_storm_trends.py",        "Build storm trend / normalisation files",  "~2 min"),
    (5,  "05_build_spatial_beta.py",        "Compute spatial neighbourhood beta",       "~2 min"),
    (6,  "06_build_hail_rasters.py",        "Build raw 0.05° hail GeoTIFFs",           "~20 min"),
    (7,  "07_build_hail_debias.py",         "Apply population debiasing to rasters",    "~20 min"),
    (8,  "08_build_hail_agg.py",            "Aggregate to 0.25° and 0.50°",            "~10 min"),
    (9,  "09_build_hail_climo.py",          "Build 366-day daily climatology",          "~5 min"),
    (10, "10_hail_catmodel_pipeline.py",    "Fit CDFs + spatial correlation structure", "~30 min"),
    (11, "11_build_smooth_cdf.py",          "Build spatially-pooled smooth CDF",        "~10 min"),
    (12, "12_build_occurrence_probs.py",    "Compute annual occurrence probabilities",  "~5 min"),
    (13, "13_apply_conus_mask.py",          "Apply CONUS land mask",                    "~5 min"),
    (14, "14_generate_stochastic_catalog.py","Generate 50,000-yr stochastic catalog",  "~2.5 hrs"),
    (15, "15_stochastic_maps.py",           "Build per-cell stochastic PET maps",       "~15 min"),
]

# ANSI colours
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"

def print_header():
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  CONUS Hail Cat Model — Pipeline Runner{RESET}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Repo: {REPO_ROOT}")
    print(f"{BOLD}{'='*60}{RESET}\n")

def run_stage(num: int, script: str, desc: str, eta: str, dry_run: bool,
              validate_only: bool = False) -> bool:
    script_path = SCRIPTS / script
    log_path    = LOGS / f"{script.replace('.py', '')}.log"
    LOGS.mkdir(exist_ok=True)

    mode_label = "validate" if validate_only else "run"
    print(f"{BOLD}[{num:02d}/15]{RESET} {desc}")
    print(f"        Script: {script}  (est. {eta})")
    print(f"        Log:    {log_path.relative_to(REPO_ROOT)}")

    if dry_run:
        print(f"        {YELLOW}[DRY RUN — skipped]{RESET}\n")
        return True

    if not script_path.exists():
        print(f"        {RED}✗ Script not found: {script_path}{RESET}\n")
        return False

    t0 = time.time()
    print(f"        {CYAN}▶ {'Validating' if validate_only else 'Running'}...{RESET}", flush=True)

    cmd = [sys.executable, str(script_path)]
    if validate_only:
        cmd.append("--validate")

    try:
        with open(log_path, "w") as log_fh:
            log_fh.write(f"[{datetime.now().isoformat()}] Starting {script} ({mode_label})\n\n")
            log_fh.flush()
            proc = subprocess.Popen(
                cmd,
                cwd=str(SCRIPTS),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                log_fh.write(line)
                log_fh.flush()
                # Print lines that look like progress/summary markers
                stripped = line.strip()
                if stripped and any(marker in stripped for marker in [
                    "[", "Done", "Error", "WARNING", "✓", "✗",
                    "written", "complete", "finished", "failed",
                ]):
                    print(f"        {stripped}")

            proc.wait()
            elapsed = time.time() - t0
            log_fh.write(f"\n[{datetime.now().isoformat()}] Exit code: {proc.returncode} ({fmt_duration(elapsed)})\n")

    except KeyboardInterrupt:
        proc.terminate()
        print(f"\n        {YELLOW}⚠ Interrupted by user{RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"        {RED}✗ Exception: {e}{RESET}\n")
        return False

    elapsed = time.time() - t0
    if proc.returncode == 0:
        print(f"        {GREEN}✓ Done in {fmt_duration(elapsed)}{RESET}\n")
        return True
    else:
        print(f"        {RED}✗ Failed (exit {proc.returncode}) after {fmt_duration(elapsed)}{RESET}")
        print(f"        {RED}  See {log_path} for details{RESET}\n")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run the CONUS hail cat model pipeline.")
    parser.add_argument("--from",   dest="from_stage", type=int, default=1,
                        help="Start from this stage number (default: 1)")
    parser.add_argument("--only",   dest="only_stage", type=int, default=None,
                        help="Run only this stage number")
    parser.add_argument("--skip",   dest="skip_stages", type=str, default="",
                        help="Comma-separated stage numbers to skip")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="Print stages without running them")
    parser.add_argument("--validate", dest="validate_only", action="store_true",
                        help="Run each stage with --validate (skip computation, check outputs only)")
    args = parser.parse_args()

    # Preflight: check all dependencies are installed (skip for dry-run)
    if not args.dry_run:
        if not check_dependencies():
            sys.exit(1)

    skip = set()
    if args.skip_stages:
        try:
            skip = {int(s.strip()) for s in args.skip_stages.split(",")}
        except ValueError:
            print(f"{RED}Invalid --skip value: {args.skip_stages}{RESET}")
            sys.exit(1)

    print_header()

    if args.validate_only:
        print(f"  {YELLOW}Mode: VALIDATE ONLY — checking outputs, skipping computation{RESET}")

    # Determine which stages to run
    if args.only_stage is not None:
        stages_to_run = [s for s in STAGES if s[0] == args.only_stage]
        if not stages_to_run:
            print(f"{RED}No stage with number {args.only_stage}{RESET}")
            sys.exit(1)
    else:
        stages_to_run = [s for s in STAGES if s[0] >= args.from_stage and s[0] not in skip]

    total = len(stages_to_run)
    print(f"  Stages to run: {total}")
    if skip:
        print(f"  Skipping:      {sorted(skip)}")
    if args.from_stage > 1:
        print(f"  Resuming from: stage {args.from_stage}")
    print()

    pipeline_start = time.time()
    results = []

    for num, script, desc, eta in stages_to_run:
        ok = run_stage(num, script, desc, eta, args.dry_run, args.validate_only)
        results.append((num, script, desc, ok))
        if not ok and not args.dry_run:
            print(f"{RED}{BOLD}Pipeline stopped at stage {num}.{RESET}")
            print(f"Fix the issue and resume with:  python run_pipeline.py --from {num}\n")
            break

    # Summary
    total_elapsed = time.time() - pipeline_start
    passed = sum(1 for _, _, _, ok in results if ok)
    failed = sum(1 for _, _, _, ok in results if not ok)

    print(f"{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  Summary{RESET}  ({fmt_duration(total_elapsed)} total)")
    print(f"{BOLD}{'='*60}{RESET}")
    for num, script, desc, ok in results:
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"  {icon}  [{num:02d}] {desc}")

    print()
    if failed == 0 and not args.dry_run:
        mode_word = "validation" if args.validate_only else "stage(s)"
        print(f"{GREEN}{BOLD}  ✓ All {passed} {mode_word} completed successfully!{RESET}\n")
    elif args.dry_run:
        print(f"{YELLOW}  Dry run complete — nothing was executed.{RESET}\n")
    else:
        print(f"{RED}{BOLD}  {failed} stage(s) failed. {passed} succeeded.{RESET}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
