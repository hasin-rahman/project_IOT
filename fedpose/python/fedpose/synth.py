"""SYNTHETIC sensor data generator.

Only for testing the pipeline before real phone data exists. Numbers produced from this data
must NOT be reported as results. It models: gravity direction per pose, user tilt habits
(resting vs handheld vs an unseen user), per-device sensor bias, hand jitter, random yaw (which
makes rotation-vector features heading dependent, as on a real phone).

Device frame follows Android: x right, y toward top edge, z out of the screen. An accelerometer
at rest reads +9.81 along whichever device axis points UP.
"""
from __future__ import annotations

import os

import numpy as np

from .constants import CLIENT_POSES, POSES

G = 9.81
# "Up" direction expressed in the device frame for each pose.
POSE_UP = {
    0: (0, 0, 1),    # screen up
    1: (0, 0, -1),   # screen down
    2: (0, 1, 0),    # portrait up (top edge up)
    3: (0, -1, 0),   # portrait down
    4: (-1, 0, 0),   # landscape left (left edge up)
    5: (1, 0, 0),    # landscape right (right edge up)
}
USERS = {  # tilt std (deg), jitter scale, mode
    "alex": (4.0, 1.0),
    "sam": (10.0, 3.0),
    "pat": (18.0, 5.0),  # unseen user: sloppier, shakier
}


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def rot_between(u, v):
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    c = float(u @ v)
    if c > 1 - 1e-12:
        return np.eye(3)
    if c < -1 + 1e-12:
        a = np.cross(u, [1.0, 0, 0])
        if np.linalg.norm(a) < 1e-6:
            a = np.cross(u, [0, 1.0, 0])
        a /= np.linalg.norm(a)
        return 2 * np.outer(a, a) - np.eye(3)
    a = np.cross(u, v)
    s2 = float(a @ a)
    K = _skew(a)
    return np.eye(3) + K + K @ K * ((1 - c) / s2)


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def quat_xyz(R):
    """Rotation matrix -> (x, y, z) of the unit quaternion with w >= 0 (Android rotation vector)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    if w < 0:
        x, y, z = -x, -y, -z
    return x, y, z


def _small_rot(rng, sigma_deg):
    ang = np.deg2rad(abs(rng.normal(0, sigma_deg)))
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    K = _skew(axis)
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


def make_session(rng, pose: int, user: str, device_bias: dict, duration_s: float = 24.0, rate_hz: float = 50.0):
    tilt_deg, jit = USERS[user]
    n = int(duration_s * rate_hz)
    u0 = _small_rot(rng, tilt_deg) @ np.asarray(POSE_UP[pose], float)
    # slow hand wobble (AR(1)) around the base direction
    wob = np.zeros((n, 3))
    for i in range(1, n):
        wob[i] = 0.97 * wob[i - 1] + rng.normal(0, 0.004 * jit, 3)
    u = u0 + wob
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    yaw = rng.uniform(0, 2 * np.pi)
    Rz = rot_z(yaw)
    quats = np.array([quat_xyz(Rz @ rot_between(ui, [0, 0, 1.0])) for ui in u])
    lin = rng.normal(0, 0.03 * jit, (n, 3))  # linear acceleration
    ang = rng.normal(0, 0.01 * jit, (n, 3))  # angular velocity
    # gravity sensor = smooth part only
    kernel = np.ones(25) / 25
    u_smooth = np.stack([np.convolve(u[:, k], kernel, mode="same") for k in range(3)], axis=1)
    u_smooth /= np.linalg.norm(u_smooth, axis=1, keepdims=True)

    t0 = int(rng.integers(5_000_000_000, 9_000_000_000))
    rows = []
    for sensor in ("acc", "grav", "gyro", "rot"):
        phase = rng.uniform(0, 20e6)
        ts = t0 + phase + np.arange(n) * (1e9 / rate_hz) + rng.normal(0, 1.5e6, n)
        ts = np.sort(ts).astype(np.int64)
        if sensor == "acc":
            v = G * u * device_bias["acc_scale"] + lin + device_bias["acc_off"] + rng.normal(0, 0.02, (n, 3))
        elif sensor == "grav":
            v = G * u_smooth + rng.normal(0, 0.004, (n, 3))
        elif sensor == "gyro":
            v = ang + device_bias["gyro_off"] + rng.normal(0, 0.004, (n, 3))
        else:
            v = quats + rng.normal(0, 0.002, (n, 3))
        for t, vv in zip(ts, v):
            rows.append((int(t), sensor, vv[0], vv[1], vv[2]))
    rows.sort(key=lambda r: r[0])
    return rows


def write_session(path: str, rows) -> None:
    with open(path, "w") as f:
        f.write("t_ns,sensor,x,y,z\n")
        for t, s, x, y, z in rows:
            f.write(f"{t},{s},{x:.6g},{y:.6g},{z:.6g}\n")


def generate_dataset(root: str, seed: int = 0, duration_s: float = 24.0,
                     train_per_pose: int = 4, val_per_pose: int = 1, test_per_pose: int = 2,
                     newuser_per_pose: int = 2) -> None:
    rng = np.random.default_rng(seed)
    for client, own_poses in CLIENT_POSES.items():
        bias = {
            "acc_scale": 1 + rng.normal(0, 0.01, 3),
            "acc_off": rng.normal(0, 0.05, 3),
            "gyro_off": rng.normal(0, 0.003, 3),
        }
        plan = {
            "train": (own_poses, train_per_pose, ("alex", "sam")),
            "val": (range(6), val_per_pose, ("alex", "sam")),
            "test": (range(6), test_per_pose, ("alex", "sam")),
            "newuser": (range(6), newuser_per_pose, ("pat",)),
        }
        for split, (poses, count, users) in plan.items():
            d = os.path.join(root, "raw", client, split)
            os.makedirs(d, exist_ok=True)
            for pose in poses:
                for k in range(count):
                    user = users[k % len(users)]
                    rows = make_session(rng, pose, user, bias, duration_s)
                    write_session(os.path.join(d, f"{POSES[pose]}__{user}__{k:02d}.csv"), rows)
