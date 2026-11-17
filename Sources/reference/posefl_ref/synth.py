"""Synthetic sensor sessions for testing the pipeline before real phone data exists.

THIS IS NOT REAL DATA. It is a rough physical model (gravity direction + tilt + hand
tremor + sensor bias) used to check that code paths work and that the FL protocol
behaves sensibly. All reported results must come from real phone recordings.
"""
import numpy as np

from .poses import up_vector

G = 9.81


def _rodrigues(v, axis, angle):
    axis = axis / np.linalg.norm(axis)
    return (v * np.cos(angle) + np.cross(axis, v) * np.sin(angle)
            + axis * np.dot(axis, v) * (1 - np.cos(angle)))


def _random_perp(u, rng):
    a = rng.normal(size=3)
    a -= a.dot(u) * u
    return a / np.linalg.norm(a)


def quaternion_from_rows(e, n, u):
    """Quaternion (x, y, z, w) of rotation matrix R whose rows are (e, n, u)."""
    R = np.array([e, n, u])
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    if q[3] < 0:  # Android reports w >= 0
        q = -q
    return q / np.linalg.norm(q)


class Person:
    """A user/device combination with its own habits (tilt, tremor) and sensor bias."""

    def __init__(self, rng, max_tilt_deg=25.0, tremor=0.15, bias=0.08):
        self.max_tilt = np.radians(max_tilt_deg)
        self.tremor = tremor
        self.acc_bias = rng.normal(scale=bias, size=3)
        self.gyro_bias = rng.normal(scale=0.005, size=3)


def make_session(pose, person, rng, duration_s=20.0, rate_hz=50.0, holding="hand"):
    """Return an (n, 14) array in the raw CSV layout for one recording."""
    n = int(duration_s * rate_hz)
    t = np.arange(n) * (1000.0 / rate_hz) + rng.normal(scale=1.0, size=n)
    t = np.sort(t - t[0])

    u0 = np.array(up_vector(pose))
    tilt = rng.uniform(0, person.max_tilt)
    u_pose = _rodrigues(u0, _random_perp(u0, rng), tilt)
    heading = _random_perp(u_pose, rng)

    in_hand = holding == "hand"
    tremor = person.tremor if in_hand else 0.02
    sway_amp = np.radians(4.0) if in_hand else 0.0
    sway_axis = _random_perp(u_pose, rng)

    # first/last second: the phone is moving into / out of the pose
    start_dir = rng.normal(size=3)
    start_dir /= np.linalg.norm(start_dir)
    rows = np.zeros((n, 14))
    for i in range(n):
        ts = t[i] / 1000.0
        u = _rodrigues(u_pose, sway_axis, sway_amp * np.sin(2 * np.pi * 0.4 * ts))
        if ts < 1.0:
            a = ts
            u = (1 - a) * start_dir + a * u
        elif ts > duration_s - 1.0:
            a = (duration_s - ts)
            u = a * u + (1 - a) * start_dir
        u = u / np.linalg.norm(u)
        grav = G * u
        acc = grav + rng.normal(scale=tremor, size=3) + person.acc_bias
        gyro = rng.normal(scale=0.3 * tremor, size=3) + person.gyro_bias
        e = heading - heading.dot(u) * u
        e /= np.linalg.norm(e)
        nvec = np.cross(u, e)
        q = quaternion_from_rows(e, nvec, u)
        rows[i] = [t[i], *acc, *grav, *gyro, *q]
    return rows
