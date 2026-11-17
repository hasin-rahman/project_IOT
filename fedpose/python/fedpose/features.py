"""Raw session CSV -> windowed feature matrix.

Raw CSV format (one file per recording session, written by the phone):

    t_ns,sensor,x,y,z
    123456789000,acc,0.01,0.02,9.80
    123456791000,grav,0.00,0.01,9.81
    ...

`sensor` is one of acc, grav, gyro, rot. `t_ns` is the Android event timestamp (ns).
Windows are time based, so sensors with different rates need no resampling.
Kotlin twin: Features.kt (must produce the same numbers).
"""
from __future__ import annotations

import csv

import numpy as np

from .constants import (
    DEFAULT_FEATURE_SET,
    FEATURE_SETS,
    MIN_SAMPLES,
    SENSOR_SCALE,
    SENSORS,
    STRIDE_S,
    TRIM_S,
    WINDOW_S,
    n_features,
)

Session = dict  # sensor -> (t_ns int64[n], xyz float64[n, 3]) sorted by time


def load_session_csv(path: str) -> Session:
    rows = {s: ([], []) for s in SENSORS}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        if header != ["t_ns", "sensor", "x", "y", "z"]:
            raise ValueError(f"{path}: unexpected header {header}")
        for r in reader:
            if not r:
                continue
            s = r[1].strip()
            if s not in rows:
                continue  # unknown sensors are ignored
            rows[s][0].append(int(r[0]))
            rows[s][1].append((float(r[2]), float(r[3]), float(r[4])))
    out: Session = {}
    for s, (t, xyz) in rows.items():
        if not t:
            continue
        t = np.asarray(t, dtype=np.int64)
        xyz = np.asarray(xyz, dtype=np.float64)
        order = np.argsort(t, kind="stable")
        out[s] = (t[order], xyz[order])
    return out


def window_vector(session: Session, sensors, start_ns: int, win_ns: int, min_samples: int = MIN_SAMPLES):
    """Features of the window [start, start + win). None if a sensor has too few samples.

    Layout per sensor: mean(x, y, z), std(x, y, z) (population std), values divided by the
    sensor's fixed scale first. Sensors are concatenated in feature-set order.
    """
    parts = []
    for s in sensors:
        t, xyz = session[s]
        lo = int(np.searchsorted(t, start_ns, "left"))
        hi = int(np.searchsorted(t, start_ns + win_ns, "left"))
        if hi - lo < min_samples:
            return None
        seg = xyz[lo:hi] / SENSOR_SCALE[s]
        parts.append(seg.mean(axis=0))
        parts.append(seg.std(axis=0))
    return np.concatenate(parts)


def window_features(
    session: Session,
    feature_set: str = DEFAULT_FEATURE_SET,
    window_s: float = WINDOW_S,
    stride_s: float = STRIDE_S,
    trim_s: float = TRIM_S,
    min_samples: int = MIN_SAMPLES,
) -> np.ndarray:
    sensors = FEATURE_SETS[feature_set]
    nf = n_features(feature_set)
    if any(s not in session for s in sensors):
        return np.zeros((0, nf))
    win = int(round(window_s * 1e9))
    stride = int(round(stride_s * 1e9))
    trim = int(round(trim_s * 1e9))
    t_lo = max(int(session[s][0][0]) for s in sensors) + trim
    t_hi = min(int(session[s][0][-1]) for s in sensors) - trim
    out = []
    start = t_lo
    while start + win <= t_hi:
        v = window_vector(session, sensors, start, win, min_samples)
        if v is not None:
            out.append(v)
        start += stride
    return np.asarray(out, dtype=np.float64).reshape(-1, nf)


def feature_names(feature_set: str) -> list[str]:
    names = []
    for s in FEATURE_SETS[feature_set]:
        names += [f"{s}_mean_{a}" for a in "xyz"] + [f"{s}_std_{a}" for a in "xyz"]
    return names
