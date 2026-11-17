#!/usr/bin/env bash
# Starts (or restarts) the federated-learning server behind Caddy. Run setup_caddy.sh once before.
# It keeps running after you close the SSH window. Its output goes to server.log in this folder.
#
#   bash start_server.sh YOUR_PASSWORD                       a normal run
#   bash start_server.sh YOUR_PASSWORD --round-timeout 0     the first phone waits for the other as long as needed
#   bash start_server.sh YOUR_PASSWORD --secure              secure aggregation
#   bash start_server.sh YOUR_PASSWORD --sweep sensor_study  the sensor study (phones: tick "Keep following new runs")
set -euo pipefail
cd "$(dirname "$0")"

PASSWORD="${1:-}"
if [ -z "$PASSWORD" ]; then
  echo "Usage: bash start_server.sh YOUR_PASSWORD [other fl_server.py options]"
  exit 1
fi
shift
URL=$(cat "$HOME/.posefl_url" 2>/dev/null || true)
if [ -z "$URL" ]; then
  echo "Run  bash setup_caddy.sh  first."
  exit 1
fi

# stop the server this script started last time, if it is still running
if [ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null; then
  echo "Stopped the server that was running."
  sleep 1
fi
# The password goes in an environment variable, so it does not show up in the list of running programs.
# --host 127.0.0.1: the server only accepts connections from Caddy on this machine (always HTTPS from outside).
POSEFL_PASSWORD="$PASSWORD" nohup python3 fl_server.py --host 127.0.0.1 --port 8000 --public-url "$URL" "$@" \
  > server.log 2>&1 &
echo $! > server.pid
sleep 2
cat server.log
echo "(Follow the log with:  tail -f server.log    Stop the server with:  kill \$(cat server.pid))"
