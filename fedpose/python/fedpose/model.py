"""Softmax regression with minibatch SGD. Kotlin twin: SoftmaxModel.kt.

Parameter vector layout (float64, length 6*nf + 6):
    [ W[0][0..nf-1], W[1][0..nf-1], ..., W[5][0..nf-1], b[0..5] ]
For nf = 12 that is 78 parameters, matching the proposal.
"""
from __future__ import annotations

import numpy as np

from .constants import N_CLASSES, n_params
from .rng import SplitMix64


def init_weights(nf: int, seed: int = 0, scale: float = 0.0) -> np.ndarray:
    """Zeros by default; otherwise uniform in [-scale, scale] from SplitMix64."""
    w = np.zeros(n_params(nf))
    if scale > 0:
        rng = SplitMix64(seed)
        for i in range(len(w)):
            w[i] = (rng.next_double() - 0.5) * 2.0 * scale
    return w


def f32_round(w: np.ndarray) -> np.ndarray:
    """What a phone receives: weights rounded to float32 (the wire format)."""
    return np.asarray(w, dtype=np.float64).astype(np.float32).astype(np.float64)


def unpack(w: np.ndarray, nf: int):
    return w[: N_CLASSES * nf].reshape(N_CLASSES, nf), w[N_CLASSES * nf:]


def logits(w: np.ndarray, X: np.ndarray, nf: int) -> np.ndarray:
    W, b = unpack(w, nf)
    return X @ W.T + b


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_proba(w: np.ndarray, X: np.ndarray, nf: int) -> np.ndarray:
    return softmax(logits(w, X, nf))


def predict(w: np.ndarray, X: np.ndarray, nf: int) -> np.ndarray:
    return logits(w, X, nf).argmax(axis=1)


def loss_acc(w: np.ndarray, X: np.ndarray, y: np.ndarray, nf: int):
    if len(y) == 0:
        return 0.0, 0.0
    p = predict_proba(w, X, nf)
    ce = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
    return float(ce), float((p.argmax(axis=1) == y).mean())


def train_local(
    w0: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    nf: int,
    *,
    epochs: int,
    lr: float,
    batch_size: int = 32,
    l2: float = 0.0,
    mu: float = 0.0,
    w_ref: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Minibatch SGD. `mu` > 0 adds a FedProx proximal term mu * (w - w_ref)."""
    w = np.asarray(w0, dtype=np.float64).copy()
    W, b = unpack(w, nf)  # views into w
    if w_ref is None:
        w_ref = w0
    Wr, br = unpack(np.asarray(w_ref, dtype=np.float64), nf)
    n = len(y)
    if n == 0:
        return w
    rng = SplitMix64(seed)
    idx = list(range(n))
    for _ in range(epochs):
        rng.shuffle(idx)
        for s in range(0, n, batch_size):
            bi = idx[s: s + batch_size]
            Xb, yb = X[bi], y[bi]
            m = len(bi)
            P = softmax(Xb @ W.T + b)
            P[np.arange(m), yb] -= 1.0
            gW = P.T @ Xb / m + l2 * W
            gb = P.mean(axis=0)
            if mu > 0:
                gW = gW + mu * (W - Wr)
                gb = gb + mu * (b - br)
            W -= lr * gW
            b -= lr * gb
    return w
