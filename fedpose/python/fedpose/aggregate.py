from __future__ import annotations

import numpy as np


def fedavg(global_w: np.ndarray, updates: list[tuple[np.ndarray, int]]) -> np.ndarray:
    """Sample-weighted FedAvg expressed on deltas.

    new = global + sum_i n_i * delta_i / sum_i n_i
    which equals sum_i n_i * w_i / sum_i n_i when delta_i = w_i - global (no compression).
    With no updates the model is returned unchanged.
    """
    updates = [(d, n) for d, n in updates if n > 0]
    if not updates:
        return np.asarray(global_w, dtype=np.float64).copy()
    total = float(sum(n for _, n in updates))
    acc = np.zeros_like(np.asarray(global_w, dtype=np.float64))
    for d, n in updates:
        acc += (n / total) * d
    return np.asarray(global_w, dtype=np.float64) + acc
