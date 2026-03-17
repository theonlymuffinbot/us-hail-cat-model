#!/usr/bin/env python3
"""SPC NOAA Full Archive Downloader — 2004-03-01 to yesterday"""

import os
import sys
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent  # scripts/ -> hail_model/
DATA_ROOT = REPO_ROOT / "data"
LOGS_ROOT = REPO_ROOT / "logs"

BASE_URL = "https://www.spc.noaa.gov/climo/reports"
OUT_DIR  = DATA_ROOT / "spc"
LOG_FILE = LOGS_ROOT / "spc_download.log"
TYPES = ["torn", "hail", "wind"]
WORKERS = 10
HEADER_SIZE = 60  # bytes — header-only files are ~52 bytes (no data)

def log(msg):
    print(msg, flush=True)
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(msg + "\n")

def download_one(url, outfile):
    if os.path.exists(outfile) and os.path.getsize(outfile) > HEADER_SIZE:
        return "skip"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research/archive-download)"})
        with urlopen(req, timeout=15) as resp:
            content = resp.read()
        if len(content) > HEADER_SIZE:
            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            with open(outfile, "wb") as f:
                f.write(content)
            return "ok"
        return "empty"
    except HTTPError:
        return "miss"
    except Exception as e:
        return f"err:{e}"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = date(2004, 3, 1)
    end = date.today() - timedelta(days=1)

    # Build task list
    tasks = []
    d = start
    while d <= end:
        yy = d.strftime("%y")
        mm = d.strftime("%m")
        dd = d.strftime("%d")
        year = d.strftime("%Y")
        for t in TYPES:
            filename = f"{yy}{mm}{dd}_rpts_{t}.csv"
            url = f"{BASE_URL}/{filename}"
            outfile = str(OUT_DIR / year / filename)
            tasks.append((url, outfile))
        d += timedelta(days=1)

    total = len(tasks)
    log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting: {total} files to check")
    log(f"Output: {OUT_DIR}")

    counts = {"ok": 0, "skip": 0, "miss": 0, "empty": 0, "err": 0}
    done = 0

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, url, outfile): (url, outfile) for url, outfile in tasks}
        for future in as_completed(futures):
            result = future.result()
            done += 1
            key = result if result in counts else "err"
            counts[key] += 1
            if done % 500 == 0 or done == total:
                pct = done / total * 100
                log(f"  [{done}/{total}] {pct:.1f}% — saved:{counts['ok']} skipped:{counts['skip']} empty/404:{counts['miss']+counts['empty']} errors:{counts['err']}")

    saved = counts["ok"] + counts["skip"]
    log(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Done! Files with data: {counts['ok']} new + {counts['skip']} already had. Total in archive: {counts['ok']+counts['skip']}")

if __name__ == "__main__":
    main()
