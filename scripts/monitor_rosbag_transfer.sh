#!/usr/bin/env bash

set -u

remote_host="HPZ3_at_SR"
remote_dir="/home/unito/advis/bags"
local_dir="/Users/rashid/data/DS/SR/v6/recorded_bags"
interval=30

usage() {
  cat <<'EOF'
Monitor an rsync rosbag download and estimate its remaining time.

Usage:
  scripts/monitor_rosbag_transfer.sh [options]

Options:
  --remote-host HOST   SSH host or alias (default: HPZ3_at_SR)
  --remote-dir PATH    Remote rosbag directory
  --local-dir PATH     Local download directory
  --interval SECONDS   Refresh interval (default: 30)
  -h, --help           Show this help
EOF
}

while (($#)); do
  case "$1" in
    --remote-host)
      remote_host=${2:?"--remote-host requires a value"}
      shift 2
      ;;
    --remote-dir)
      remote_dir=${2:?"--remote-dir requires a value"}
      shift 2
      ;;
    --local-dir)
      local_dir=${2:?"--local-dir requires a value"}
      shift 2
      ;;
    --interval)
      interval=${2:?"--interval requires a value"}
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! $interval =~ ^[1-9][0-9]*$ ]]; then
  echo "--interval must be a positive integer" >&2
  exit 2
fi

if [[ ! -d $local_dir ]]; then
  echo "Local directory does not exist: $local_dir" >&2
  exit 1
fi

previous_bytes=0
previous_time=$(date +%s)

echo "Monitoring ${remote_host}:${remote_dir}"
echo "Destination: ${local_dir}"
echo "Refresh interval: ${interval}s"

while true; do
  remote_bytes=$(
    ssh "$remote_host" \
      "find '$remote_dir' -type f -printf '%s\n' | awk '{total += \$1} END {print total+0}'"
  )
  ssh_result=$?

  if ((ssh_result != 0)) || [[ ! $remote_bytes =~ ^[0-9]+$ ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Could not read remote size; retrying." >&2
    sleep "$interval"
    continue
  fi

  local_bytes=$(python3 - "$local_dir" <<'PY'
import os
import sys

total = 0
for root, _, files in os.walk(sys.argv[1]):
    for name in files:
        try:
            total += os.path.getsize(os.path.join(root, name))
        except FileNotFoundError:
            # rsync can rename a temporary file while this scan is running.
            pass
print(total)
PY
  )

  current_time=$(date +%s)
  elapsed=$((current_time - previous_time))
  measured_rate=0
  if ((previous_bytes > 0 && elapsed > 0 && local_bytes >= previous_bytes)); then
    measured_rate=$(((local_bytes - previous_bytes) / elapsed))
  fi

  if [[ -t 1 ]]; then
    printf '\033[2J\033[H'
  fi

  python3 - "$remote_bytes" "$local_bytes" "$measured_rate" <<'PY'
import sys
from datetime import timedelta
from time import strftime

remote = int(sys.argv[1])
local = int(sys.argv[2])
rate = int(sys.argv[3])
remaining = max(0, remote - local)
progress = min(100.0, 100 * local / remote) if remote else 0.0

print(f"Rosbag transfer monitor — {strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 56)
print(f"Remote total : {remote / 1e9:9.2f} GB")
print(f"Downloaded   : {local / 1e9:9.2f} GB")
print(f"Remaining    : {remaining / 1e9:9.2f} GB")
print(f"Progress     : {progress:9.2f}%")
if rate > 0:
    print(f"Current rate : {rate / 1e6:9.2f} MB/s")
    print(f"ETA          : {timedelta(seconds=int(remaining / rate))}")
else:
    print("Current rate : sampling or stalled")
    print("ETA          : unavailable until data advances")
print("=" * 56)
PY

  if ((remote_bytes > 0 && local_bytes >= remote_bytes)); then
    echo "Transfer size reached 100%. Run an rsync checksum dry-run to verify it."
    exit 0
  fi

  previous_bytes=$local_bytes
  previous_time=$current_time
  sleep "$interval"
done
