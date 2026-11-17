#!/usr/bin/env python3
"""Generate SYNTHETIC raw sessions to test the pipeline (never report results from this data).

    python scripts/make_synth_data.py --out ../data_synth --seed 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.synth import generate_dataset  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../data_synth")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--duration", type=float, default=24.0, help="seconds per session")
a = ap.parse_args()
generate_dataset(a.out, seed=a.seed, duration_s=a.duration)
print(f"wrote synthetic dataset to {os.path.abspath(a.out)}")
