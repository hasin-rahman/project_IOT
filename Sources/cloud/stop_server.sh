#!/usr/bin/env bash
# Stops the server started by start_server.sh.
cd "$(dirname "$0")"
if [ -f server.pid ] && kill "$(cat server.pid)" 2>/dev/null; then echo "Server stopped."; else echo "No server was running."; fi
rm -f server.pid
