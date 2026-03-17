#!/usr/bin/env bash
# SPC NOAA Storm Reports - Full Archive Downloader
# Covers 2004-03-01 through yesterday

BASE_URL="https://www.spc.noaa.gov/climo/reports"
OUT_DIR="$(dirname "$0")/archive"
LOG="$(dirname "$0")/download.log"
PARALLEL=8

mkdir -p "$OUT_DIR"
echo "Starting SPC archive download at $(date)" | tee "$LOG"

# Use Python to generate date list (avoids bash date portability issues)
URLFILE=$(mktemp)
python3 - <<EOF > "$URLFILE"
from datetime import date, timedelta
start = date(2004, 3, 1)
end = date.today() - timedelta(days=1)
types = ["torn", "hail", "wind"]
d = start
while d <= end:
    yy = d.strftime("%y")
    mm = d.strftime("%m")
    dd = d.strftime("%d")
    year = d.strftime("%Y")
    for t in types:
        filename = f"{yy}{mm}{dd}_rpts_{t}.csv"
        url = f"$BASE_URL/{filename}"
        outfile = f"$OUT_DIR/{year}/{filename}"
        print(f"{url} {outfile}")
    d += timedelta(days=1)
EOF

# Create year directories
python3 -c "
from datetime import date, timedelta
import os
d = date(2004, 3, 1)
end = date.today() - timedelta(days=1)
while d <= end:
    os.makedirs('$OUT_DIR/' + d.strftime('%Y'), exist_ok=True)
    d += timedelta(days=1)
"

TOTAL=$(wc -l < "$URLFILE" | tr -d ' ')
echo "Files to attempt: $TOTAL" | tee -a "$LOG"

# Download with xargs parallel — skip already-downloaded non-empty files
grep -v "" "$URLFILE" | while IFS=' ' read -r url outfile; do
  if [[ -s "$outfile" ]]; then
    echo "skip $url"
  else
    echo "$url $outfile"
  fi
done | xargs -P "$PARALLEL" -L 1 bash -c '
  url="$0"; outfile="$1"
  [[ -z "$url" || -z "$outfile" ]] && exit 0
  code=$(curl -sS -w "%{http_code}" -o "$outfile" "$url" 2>/dev/null)
  if [[ "$code" == "200" ]]; then
    size=$(wc -c < "$outfile" | tr -d " ")
    if [[ "$size" -le 60 ]]; then rm -f "$outfile"; fi
  else
    rm -f "$outfile"
  fi
'

rm -f "$URLFILE"

downloaded=$(find "$OUT_DIR" -name "*.csv" | wc -l | tr -d ' ')
echo "Done at $(date). Files saved: $downloaded" | tee -a "$LOG"
