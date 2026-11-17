#!/usr/bin/env bash
# Runs every automated check: Python tests, Kotlin compile + selftest, Kotlin<->Python parity, live dropout test.
# Needs kotlinc on PATH (e.g. `npm install kotlin-compiler`, then add node_modules/kotlin-compiler/bin to PATH) and java.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
JAR="${JAR:-/tmp/fedpose-core.jar}"
cd "$ROOT/python"
[ -d ../data_synth ] || python scripts/make_synth_data.py --out ../data_synth
python -m pytest tests -q
kotlinc "$ROOT/android/core/src/main/kotlin" -include-runtime -d "$JAR"
java -cp "$JAR" com.fedpose.cli.VirtualPhoneKt selftest
python scripts/check_kotlin_parity.py --jar "$JAR" --data ../data_synth
python scripts/check_dropout_live.py --jar "$JAR" --data ../data_synth
echo "ALL CHECKS PASSED"
