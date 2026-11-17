"""Label sanity check for recorded sessions.

Each pose has a typical gravity direction (which device axis points up or down). For every CSV this
script computes the mean gravity vector and compares it with the median vector of all files that
share its pose label. A file that points somewhere else is probably mislabeled.

  python3 scripts/check_labels.py --root ../data/raw/B          (folder that contains train/ val/ test/ newuser/)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

POSES = ["screen_up", "screen_down", "portrait_up", "portrait_down", "landscape_left", "landscape_right"]


def mean_grav(path: str):
    v = [(float(r[2]), float(r[3]), float(r[4])) for r in csv.reader(open(path)) if len(r) >= 5 and r[1] == "grav"]
    return np.mean(v, axis=0) if v else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="folder with split subfolders, e.g. raw/B")
    ap.add_argument("--min-cos", type=float, default=0.8, help="below this similarity a file is flagged")
    a = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "*.csv"))):
        pose = os.path.basename(f).split("__")[0]
        g = mean_grav(f)
        if pose in POSES and g is not None:
            rows.append((f, pose, g))
    if not rows:
        raise SystemExit("no csv files found under " + a.root)

    by_pose = defaultdict(list)
    for _, p, g in rows:
        by_pose[p].append(g / (np.linalg.norm(g) + 1e-9))
    ref = {p: np.median(np.array(v), axis=0) for p, v in by_pose.items()}

    bad = 0
    for f, pose, g in rows:
        u = g / (np.linalg.norm(g) + 1e-9)
        r = ref[pose] / (np.linalg.norm(ref[pose]) + 1e-9)
        cos = float(u @ r)
        best = max(ref, key=lambda p: float(u @ (ref[p] / (np.linalg.norm(ref[p]) + 1e-9))))
        if cos < a.min_cos or best != pose:
            bad += 1
            print(f"CHECK  {os.path.relpath(f, a.root):55s} labeled {pose}, looks like {best} (similarity {cos:.2f})")
    print(f"\n{len(rows)} files checked, {bad} flagged")
    print("per pose files:", {p: len(by_pose[p]) for p in POSES if p in by_pose})


if __name__ == "__main__":
    main()
