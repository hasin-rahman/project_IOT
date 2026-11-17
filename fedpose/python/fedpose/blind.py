"""Blind-test prediction: ONE pose label from a short 2 s recording (demo day format).

Judges place the phone, a 3 s countdown runs, the app records 2 s. We cut that recording into
the same 1 s windows used in training (no trimming: the clip is only 2 s long) and average the
softmax probabilities over the windows. Kotlin twin: BlindTest.kt.

`CLIP_STARTS_S` are window start offsets from the start of the recording. Default (0.5, 1.0) skips
the first half second, when the hand may still be settling, and keeps two overlapping windows.
Use (0.0, 0.5, 1.0) to use all of the clip. Choose with scripts/blind_test_sim.py on YOUR validation
data, not by guessing.
"""
from __future__ import annotations

import numpy as np

from .constants import FEATURE_SETS, MIN_SAMPLES, N_CLASSES, WINDOW_S, n_features
from .features import window_vector
from .model import predict_proba

CLIP_S = 2.0
CLIP_STARTS_S = (0.5, 1.0)


def clip_windows(session, feature_set: str, clip_start_ns: int, starts_s=CLIP_STARTS_S) -> np.ndarray:
    sensors = FEATURE_SETS[feature_set]
    win = int(round(WINDOW_S * 1e9))
    out = []
    for s in starts_s:
        v = window_vector(session, sensors, clip_start_ns + int(round(s * 1e9)), win, MIN_SAMPLES)
        if v is not None:
            out.append(v)
    return np.asarray(out, dtype=np.float64).reshape(-1, n_features(feature_set))


def clip_start_of(session, feature_set: str) -> int:
    """Same rule as the phone: the clip starts when every needed sensor has produced data."""
    return max(int(session[s][0][0]) for s in FEATURE_SETS[feature_set])


def clip_proba(w: np.ndarray, windows: np.ndarray, nf: int) -> np.ndarray:
    if len(windows) == 0:
        return np.full(N_CLASSES, np.nan)
    return predict_proba(w, windows, nf).mean(axis=0)


def clip_label(w: np.ndarray, windows: np.ndarray, nf: int) -> int:
    p = clip_proba(w, windows, nf)
    return -1 if np.isnan(p).any() else int(np.argmax(p))


def make_clips(session, feature_set: str, rng, n: int, trim_s: float = 1.0, starts_s=CLIP_STARTS_S):
    """Random 2 s clips from the steady part of a session (judges place the phone, it settles, then records)."""
    sensors = FEATURE_SETS[feature_set]
    if any(s not in session for s in sensors):
        return []
    lo = max(int(session[s][0][0]) for s in sensors) + int(trim_s * 1e9)
    hi = min(int(session[s][0][-1]) for s in sensors) - int(trim_s * 1e9) - int(CLIP_S * 1e9)
    if hi <= lo:
        return []
    clips = []
    for _ in range(n):
        start = int(rng.integers(lo, hi))
        w = clip_windows(session, feature_set, start, starts_s)
        if len(w):
            clips.append(w)
    return clips


def rubric_score(correct_global: int, n_global: int, correct_a: int, n_a: int, correct_b: int, n_b: int) -> dict:
    """Part A of the demo-day rubric: A1 = 30 x global acc, A2 = 5 x local A acc on A's poses, A3 = 5 x local B acc on B's poses."""
    ga = correct_global / n_global if n_global else 0.0
    la = correct_a / n_a if n_a else 0.0
    lb = correct_b / n_b if n_b else 0.0
    return {"global_acc": ga, "local_A_acc_on_A_poses": la, "local_B_acc_on_B_poses": lb,
            "A1": 30 * ga, "A2": 5 * la, "A3": 5 * lb, "part_A": 30 * ga + 5 * la + 5 * lb}
