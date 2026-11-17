"""Windowing and feature extraction. Mirrors android/.../core/Features.kt exactly.

Raw session CSV columns (one row per accelerometer event, ~50 Hz):
    t_ms, ax, ay, az, gx, gy, gz, wx, wy, wz, rx, ry, rz, rw
    acc = accelerometer (m/s^2), grav = gravity sensor (m/s^2),
    gyro = gyroscope (rad/s), rot = rotation-vector quaternion (x, y, z, w).

Feature blocks (each gives 6 numbers: mean x/y/z then population std x/y/z):
    acc  : accelerometer / 9.81
    grav : gravity / 9.81
    gyro : gyroscope in rad/s
    rot  : device-frame "up" vector derived from the rotation vector (heading removed)
A feature set is blocks joined by '+', e.g. "acc+grav" (the 12-feature baseline).
"""
import numpy as np

HEADER = "t_ms,ax,ay,az,gx,gy,gz,wx,wy,wz,rx,ry,rz,rw"
N_COLS = 14
G = 9.81
BLOCKS = ["acc", "grav", "gyro", "rot"]
BASELINE = "acc+grav"


class WindowConfig:
    def __init__(self, window_ms=1000.0, step_ms=500.0, trim_ms=1000.0, min_samples=25):
        self.window_ms = window_ms
        self.step_ms = step_ms
        self.trim_ms = trim_ms
        self.min_samples = min_samples


def parse_feature_set(spec):
    blocks = [b.strip() for b in spec.split("+") if b.strip()]
    if not blocks:
        raise ValueError("empty feature set")
    for b in blocks:
        if b not in BLOCKS:
            raise ValueError(f"unknown feature block '{b}' (expected one of {BLOCKS})")
    return blocks


def feature_size(spec):
    return 6 * len(parse_feature_set(spec))


def feature_names(spec):
    names = []
    for b in parse_feature_set(spec):
        names += [f"{b}_mean_{a}" for a in "xyz"] + [f"{b}_std_{a}" for a in "xyz"]
    return names


def read_csv(path):
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    if rows.size == 0:
        return np.zeros((0, N_COLS))
    return rows


def up_from_quaternion(q):
    """Third row of Android's getRotationMatrixFromVector: world-up expressed in device axes."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=1)


def window_ranges(t, cfg):
    """Return [(lo, hi)] index ranges; window k covers t in [start + k*step, start + k*step + window)."""
    if len(t) == 0:
        return []
    start = t[0] + cfg.trim_ms
    end_limit = t[-1] - cfg.trim_ms
    out = []
    k = 0
    while True:
        ws = start + k * cfg.step_ms
        we = ws + cfg.window_ms
        if we > end_limit + 1e-9:
            break
        lo = int(np.searchsorted(t, ws, side="left"))
        hi = int(np.searchsorted(t, we, side="left"))
        if hi - lo >= cfg.min_samples:
            out.append((lo, hi))
        k += 1
    return out


def _block_values(rows, block):
    if block == "acc":
        return rows[:, 1:4] / G
    if block == "grav":
        return rows[:, 4:7] / G
    if block == "gyro":
        return rows[:, 7:10]
    if block == "rot":
        return up_from_quaternion(rows[:, 10:14])
    raise ValueError(block)


def window_features(rows, spec):
    feats = []
    for b in parse_feature_set(spec):
        v = _block_values(rows, b)
        feats.extend(v.mean(axis=0))
        feats.extend(v.std(axis=0))  # population std (ddof=0)
    return np.array(feats)


def extract(rows, spec=BASELINE, cfg=None):
    """rows: (n, 14) array for one session -> (n_windows, n_features) array."""
    cfg = cfg or WindowConfig()
    rows = np.asarray(rows, dtype=float)
    ranges = window_ranges(rows[:, 0] if len(rows) else np.zeros(0), cfg)
    if not ranges:
        return np.zeros((0, feature_size(spec)))
    return np.stack([window_features(rows[lo:hi], spec) for lo, hi in ranges])
