"""Dataset loading for offline experiments.

Directory layout (the same files the phones write; pulled over USB with `adb pull`):

    <root>/raw/<client A|B>/<split train|val|test|newuser>/<pose>__<user>__<id>.csv

Splits are by SESSION (file), and windowing happens after the split, so overlapping windows
never cross a split boundary.

Privacy note: the deployed FL system never uploads these files. This loader exists so the team
can run repeatable controlled experiments (3 seeds, sensor study, extra-credit variants) on
data from its own phones.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass

import numpy as np

from .constants import CLIENT_POSES, DEFAULT_FEATURE_SET, POSE_INDEX, SPLITS, n_features
from .features import load_session_csv, window_features


def session_files(root: str, client: str, split: str) -> list[tuple[str, int, str]]:
    d = os.path.join(root, "raw", client, split)
    out = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".csv"):
            continue
        parts = name[:-4].split("__")
        if parts[0] not in POSE_INDEX:
            warnings.warn(f"skipping {name}: unknown pose prefix")
            continue
        user = parts[1] if len(parts) > 1 else "unknown"
        out.append((os.path.join(d, name), POSE_INDEX[parts[0]], user))
    return out


def load_split(root: str, client: str, split: str, feature_set: str = DEFAULT_FEATURE_SET, **kw):
    nf = n_features(feature_set)
    Xs, ys = [], []
    for path, pose, _user in session_files(root, client, split):
        if split == "train" and pose not in CLIENT_POSES[client]:
            warnings.warn(f"client {client} train split contains pose {pose}; ignored (non-IID rule)")
            continue
        X = window_features(load_session_csv(path), feature_set, **kw)
        if len(X):
            Xs.append(X)
            ys.append(np.full(len(X), pose, dtype=np.int64))
    if not Xs:
        return np.zeros((0, nf)), np.zeros(0, dtype=np.int64)
    return np.vstack(Xs), np.concatenate(ys)


@dataclass
class ClientData:
    client: str
    feature_set: str
    train: tuple
    val: tuple
    test: tuple
    newuser: tuple

    @property
    def nf(self) -> int:
        return n_features(self.feature_set)


def load_client(root: str, client: str, feature_set: str = DEFAULT_FEATURE_SET, **kw) -> ClientData:
    sp = {s: load_split(root, client, s, feature_set, **kw) for s in SPLITS}
    return ClientData(client, feature_set, sp["train"], sp["val"], sp["test"], sp["newuser"])


def load_all(root: str, feature_set: str = DEFAULT_FEATURE_SET, **kw) -> dict[str, ClientData]:
    return {c: load_client(root, c, feature_set, **kw) for c in CLIENT_POSES}


def pooled(clients: dict[str, ClientData], split: str):
    """Union of one split across clients (the 'same six-pose test data' for all three models)."""
    Xs = [getattr(c, split)[0] for c in clients.values() if len(getattr(c, split)[1])]
    ys = [getattr(c, split)[1] for c in clients.values() if len(getattr(c, split)[1])]
    return np.vstack(Xs), np.concatenate(ys)
