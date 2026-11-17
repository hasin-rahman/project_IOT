"""Offline federated-learning simulator.

Uses the SAME training, codec and aggregation code as the real system (model.py, wire.py,
aggregate.py), so results are directly comparable with what the Kotlin phones and the server do.
It supports the three extra-credit features so they can be compared with identical splits/seeds:

  * codec="int8"            update compression
  * adaptive=True           adaptive participation (skip small updates, forced every max_skips)
  * dropout + stale_policy  client dropout and recovery (reject stale = version check + resync)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .aggregate import fedavg
from .constants import CLIENT_INDEX, N_CLASSES, n_params
from .metrics import evaluate
from .model import f32_round, init_weights, predict, train_local
from .wire import CODEC_F32, dumps, encode_delta, decode_delta, f32_to_b64, skip_body, update_body

SEED_MUL = 1_000_003


@dataclass
class SimConfig:
    rounds: int = 30
    local_epochs: int = 2
    lr: float = 0.3
    batch_size: int = 32
    l2: float = 0.0
    mu: float = 0.0  # FedProx proximal weight (0 = plain FedAvg)
    seed: int = 0
    init_scale: float = 0.0
    codec: str = CODEC_F32
    adaptive: bool = False
    skip_threshold: float = 0.0  # L2 norm of the delta below which a client may skip
    max_skips: int = 2  # server forces participation after this many consecutive skips
    dropout: dict = field(default_factory=dict)  # client -> (first_round, last_round) offline
    stale_policy: str = "reject"  # "reject" (version check + resync) or "accept" (naive)


def client_seed(base_seed: int, rnd: int, client: str) -> int:
    """Seed for client `client` in round `rnd` (also used by the Kotlin virtual phone)."""
    return base_seed * SEED_MUL + rnd * 1009 + CLIENT_INDEX[client]


def local_baseline_seed(base_seed: int, client: str) -> int:
    return base_seed * SEED_MUL + 777 + CLIENT_INDEX[client]


def train_local_only(X, y, nf: int, cfg: SimConfig, client: str) -> np.ndarray:
    """Independent baseline: from scratch, same total epochs as a client sees in federation."""
    w0 = init_weights(nf, cfg.seed, cfg.init_scale)
    return train_local(
        w0, X, y, nf,
        epochs=cfg.rounds * cfg.local_epochs, lr=cfg.lr, batch_size=cfg.batch_size, l2=cfg.l2,
        seed=local_baseline_seed(cfg.seed, client),
    )


def _eval(w, sets, nf):
    out = {}
    for name, (X, y) in sets.items():
        out[name] = evaluate(y, predict(w, X, nf))
    return out


def simulate(clients: dict, nf: int, cfg: SimConfig, eval_sets: dict | None = None,
             keep_weights: bool = False) -> dict:
    """clients: {"A": (X, y), "B": (X, y)} training data only. Returns the final model and history."""
    eval_sets = eval_sets or {}
    P = n_params(nf)
    gw = init_weights(nf, cfg.seed, cfg.init_scale)
    skips = {c: 0 for c in clients}
    up_bytes = down_bytes = 0
    history = []
    held = {}  # naive-stale mode: model a dropped client was holding
    stale_updates = {}  # client -> (delta, n) computed offline from an old model
    n_stale_rejected = 0
    n_skipped = 0
    epochs_run = 0  # local training cost: total local epochs executed across clients
    for r in range(1, cfg.rounds + 1):
        w32 = f32_round(gw)
        updates = []
        participants, skipped_now, offline_now = [], [], []
        for c, (X, y) in clients.items():
            lo, hi = cfg.dropout.get(c, (None, None))
            offline = lo is not None and lo <= r <= hi
            if offline:
                offline_now.append(c)
                if r == lo:  # went offline mid-training: it holds an old model + local result
                    wl = train_local(w32, X, y, nf, epochs=cfg.local_epochs, lr=cfg.lr,
                                     batch_size=cfg.batch_size, l2=cfg.l2, mu=cfg.mu, w_ref=w32,
                                     seed=client_seed(cfg.seed, r, c))
                    epochs_run += cfg.local_epochs
                    stale_updates[c] = (wl - w32, len(y))
                continue
            if c in stale_updates:  # reconnect round
                d, n = stale_updates.pop(c)
                if cfg.stale_policy == "accept":
                    msg = dumps(update_body(c, r, 0, n, d, cfg.codec))
                    up_bytes += len(msg)
                    updates.append((decode_delta(update_body(c, r, 0, n, d, cfg.codec), P), n))
                    participants.append(c)
                    down_bytes += len(dumps({"w": f32_to_b64(w32)}))  # it still fetches the model
                    continue
                # reject: stale upload is wasted bytes, then resync and train on the latest model
                up_bytes += len(dumps(update_body(c, r, 0, n, d, cfg.codec)))
                n_stale_rejected += 1
            down_bytes += len(dumps({"w": f32_to_b64(w32)}))
            wl = train_local(w32, X, y, nf, epochs=cfg.local_epochs, lr=cfg.lr,
                             batch_size=cfg.batch_size, l2=cfg.l2, mu=cfg.mu, w_ref=w32,
                             seed=client_seed(cfg.seed, r, c))
            epochs_run += cfg.local_epochs
            delta = wl - w32
            norm = float(np.linalg.norm(delta))
            if cfg.adaptive and norm < cfg.skip_threshold and skips[c] < cfg.max_skips:
                skips[c] += 1
                n_skipped += 1
                skipped_now.append(c)
                up_bytes += len(dumps(skip_body(c, r, r, norm)))
                continue
            skips[c] = 0
            body = update_body(c, r, r, len(y), delta, cfg.codec)
            up_bytes += len(dumps(body))
            updates.append((decode_delta(body, P), len(y)))
            participants.append(c)
        gw = fedavg(gw, updates)
        rec = {"round": r, "participants": participants, "skipped": skipped_now,
               "offline": offline_now, "bytes_up": up_bytes, "bytes_down": down_bytes}
        if eval_sets:
            rec["eval"] = _eval(f32_round(gw), eval_sets, nf)
        if keep_weights:
            rec["weights"] = gw.copy()
        history.append(rec)
    return {"weights": gw, "history": history, "bytes_up": up_bytes, "bytes_down": down_bytes,
            "skipped_updates": n_skipped, "stale_rejected": n_stale_rejected,
            "local_epochs_run": epochs_run}
