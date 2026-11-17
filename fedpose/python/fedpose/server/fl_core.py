"""Federated server logic, independent of HTTP (so it can be unit-tested and compared with the
offline reference simulator).

Round protocol (synchronous with timeout)
-----------------------------------------
* Server state is `idle` until started, `training` while rounds run, `done` after max_rounds.
* Every round, each expected client either uploads an update, uploads a SKIP notice (adaptive
  participation), or does nothing. The round closes when all expected clients have answered, or
  when `round_timeout_s` has passed since the round opened (dropout handling).
* Aggregation = sample-weighted FedAvg over the updates that arrived. If none arrived (all
  skipped / all timed out) the global model is kept unchanged and the round still advances.
* Version checks: an update must carry the current `round` and `base_version`. Anything else is
  rejected with HTTP 409 and the current round/version so the client can re-sync.
* Scheduling: after `max_consecutive_skips` skipped rounds in a row a client is told
  `must_participate = true`, so both clients keep contributing.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from ..aggregate import fedavg
from ..constants import n_params
from ..model import init_weights
from ..wire import b64_to_f32, decode_delta, f32_to_b64

MAX_SAMPLES = 10_000_000

# Privacy guard: the ONLY fields a phone may send. Anything else (a stray "features", "samples",
# "raw" ...) is rejected and logged, so the demo can show that raw data cannot reach the server.
ALLOWED_UPDATE_KEYS = {
    "client_id", "round", "base_version", "n_samples", "skipped", "codec", "scale", "payload",
    "train_loss", "train_acc", "class_counts", "update_norm",
}
ALLOWED_REPORT_KEYS = {
    "client_id", "model", "test_set", "accuracy", "macro_f1", "n", "per_class_recall",
    "per_class_f1", "confusion",
}


@dataclass
class ServerConfig:
    n_features: int = 12
    expected_clients: tuple = ("A", "B")
    round_timeout_s: float = 60.0
    max_rounds: int = 30
    max_consecutive_skips: int = 2
    init_scale: float = 0.0
    init_seed: int = 0
    data_dir: str = "server_data"
    autostart: bool = False


class FLState:
    def __init__(self, cfg: ServerConfig, clock=time.monotonic):
        self.cfg = cfg
        self.clock = clock
        self.lock = threading.RLock()
        self.P = n_params(cfg.n_features)
        os.makedirs(cfg.data_dir, exist_ok=True)
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        c = self.cfg
        self.status = "idle"
        self.round = 0
        self.version = 0
        self.global_w = init_weights(c.n_features, c.init_seed, c.init_scale)
        self.round_started = None
        self.pending: dict[str, dict] = {}
        self.skips = {x: 0 for x in c.expected_clients}
        self.missed = {x: 0 for x in c.expected_clients}
        self.registered: set[str] = set()
        self.local_models: dict[str, dict] = {}
        self.reports: list[dict] = []
        self.round_log: list[dict] = []
        self.audit: list[dict] = []
        self.traffic: dict[str, dict] = {}
        self._traffic_mark = (0, 0)
        self.run_id = int(time.time())
        self._save_global()

    def _save_global(self):
        np.save(os.path.join(self.cfg.data_dir, f"global_v{self.version}.npy"), self.global_w)

    def _append_jsonl(self, name: str, obj: dict):
        with open(os.path.join(self.cfg.data_dir, name), "a") as f:
            f.write(json.dumps(obj) + "\n")

    def _audit(self, kind: str, body: dict, outcome: str, **extra):
        """Record WHAT a phone sent (keys, sizes, metadata), never the raw payload bytes."""
        payload = body.get("payload")
        entry = {
            "ts": round(time.time(), 3), "kind": kind, "client_id": body.get("client_id"),
            "outcome": outcome, "keys": sorted(body.keys()),
            "round": body.get("round"), "base_version": body.get("base_version"),
            "n_samples": body.get("n_samples"), "codec": body.get("codec"),
            "payload_chars": len(payload) if isinstance(payload, str) else None,
            "payload_preview": (payload[:16] + "...") if isinstance(payload, str) else None,
            "class_counts": body.get("class_counts"), "train_loss": body.get("train_loss"),
            "skipped": body.get("skipped"),
        }
        entry.update(extra)
        self.audit.append(entry)
        self.audit = self.audit[-500:]
        self._append_jsonl("uploads.jsonl", entry)

    def _traffic_totals(self):
        bi = sum(t["body_in"] for t in self.traffic.values())
        bo = sum(t["body_out"] for t in self.traffic.values())
        return bi, bo

    # --------------------------------------------------------------- traffic
    def count(self, client_id: str | None, endpoint: str, body_in: int, body_out: int,
              hdr_in: int = 0, hdr_out: int = 0):
        with self.lock:
            t = self.traffic.setdefault(client_id or "?", {
                "requests": 0, "body_in": 0, "body_out": 0, "hdr_in": 0, "hdr_out": 0, "by_endpoint": {}})
            t["requests"] += 1
            t["body_in"] += body_in
            t["body_out"] += body_out
            t["hdr_in"] += hdr_in
            t["hdr_out"] += hdr_out
            e = t["by_endpoint"].setdefault(endpoint, {"requests": 0, "body_in": 0, "body_out": 0})
            e["requests"] += 1
            e["body_in"] += body_in
            e["body_out"] += body_out

    # ------------------------------------------------------------- lifecycle
    def start(self, max_rounds: int | None = None, reset: bool = True, n_features: int | None = None):
        """Start (or restart) rounds. `n_features` switches the feature set live (sensor study demo);
        phones pick the new size up when they register again (the app does this on every Step 2)."""
        with self.lock:
            if n_features and n_features != self.cfg.n_features:
                if n_features not in (6, 12, 18, 24):
                    raise ValueError("n_features must be 6, 12, 18 or 24")
                self.cfg.n_features = int(n_features)
                self.P = n_params(self.cfg.n_features)
                reset = True
            if reset and self.status != "idle":
                registered = set(self.registered)
                self._reset_state()
                self.registered = registered
            elif n_features:
                registered = set(self.registered)
                self._reset_state()
                self.registered = registered
            if max_rounds:
                self.cfg.max_rounds = int(max_rounds)
            self.status = "training"
            self.round = 1
            self.round_started = self.clock()
            self._traffic_mark = self._traffic_totals()
            return {"status": self.status, "round": self.round, "version": self.version,
                    "max_rounds": self.cfg.max_rounds}

    def register(self, client_id: str):
        with self.lock:
            if client_id not in self.cfg.expected_clients:
                raise KeyError(client_id)
            self.registered.add(client_id)
            if (self.cfg.autostart and self.status == "idle"
                    and all(c in self.registered for c in self.cfg.expected_clients)):
                self.start(reset=False)
            return {"ok": True, "client_id": client_id, "n_features": self.cfg.n_features,
                    "n_params": self.P, "status": self.status, "round": self.round,
                    "version": self.version, "max_rounds": self.cfg.max_rounds,
                    "round_timeout_s": self.cfg.round_timeout_s}

    # ------------------------------------------------------------------ rounds
    def _tick(self):
        if self.status == "training" and self.clock() - self.round_started >= self.cfg.round_timeout_s:
            self._close_round("timeout")

    def must_participate(self, client_id: str) -> bool:
        return self.skips.get(client_id, 0) >= self.cfg.max_consecutive_skips

    def get_model(self, client_id: str):
        with self.lock:
            self._tick()
            remaining = None
            if self.status == "training":
                remaining = max(0.0, self.cfg.round_timeout_s - (self.clock() - self.round_started))
            return {
                "status": self.status, "round": self.round, "version": self.version,
                "max_rounds": self.cfg.max_rounds, "n_features": self.cfg.n_features,
                "n_params": self.P, "codec": "f32",
                "weights": f32_to_b64(self.global_w),
                "must_participate": self.must_participate(client_id),
                "already_submitted": client_id in self.pending,
                "round_deadline_s": remaining,
            }

    def round_status(self):
        with self.lock:
            self._tick()
            return {"status": self.status, "round": self.round, "version": self.version,
                    "received": sorted(self.pending), "expected": list(self.cfg.expected_clients)}

    def submit_update(self, body: dict):
        """Returns (http_status, response_dict)."""
        with self.lock:
            self._tick()
            cid = body.get("client_id")
            extra_keys = sorted(set(body) - ALLOWED_UPDATE_KEYS)
            if extra_keys:
                self._audit("update", body, "rejected_privacy_guard", unexpected_keys=extra_keys)
                return 400, {"error": "privacy_guard", "unexpected_keys": extra_keys}
            if cid not in self.cfg.expected_clients:
                self._audit("update", body, "rejected_unknown_client")
                return 400, {"error": "unknown_client"}
            if self.status != "training":
                return 409, {"error": "not_training", "status": self.status, "round": self.round,
                             "version": self.version}
            if body.get("round") != self.round or body.get("base_version") != self.version:
                self._audit("update", body, "rejected_stale", server_round=self.round, server_version=self.version)
                return 409, {"error": "stale", "round": self.round, "version": self.version}
            if cid in self.pending:  # retry after a lost response: idempotent
                self._audit("update", body, "duplicate_ignored")
                return 200, {"accepted": True, "duplicate": True, "round": self.round,
                             "version": self.version}
            if body.get("skipped") is True:
                self.pending[cid] = {"skipped": True, "n": 0, "norm": float(body.get("update_norm", 0.0))}
            else:
                n = body.get("n_samples")
                if not isinstance(n, int) or isinstance(n, bool) or not (0 < n <= MAX_SAMPLES):
                    self._audit("update", body, "rejected_bad_n_samples")
                    return 400, {"error": "bad_n_samples"}
                cc = body.get("class_counts")
                if cc is not None and not (isinstance(cc, list) and len(cc) == 6 and all(
                        isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= MAX_SAMPLES for v in cc)):
                    self._audit("update", body, "rejected_bad_class_counts")
                    return 400, {"error": "bad_class_counts"}
                try:
                    delta = decode_delta(body, self.P)
                except ValueError as e:
                    self._audit("update", body, "rejected_bad_payload", detail=str(e))
                    return 400, {"error": "bad_payload", "detail": str(e)}
                meta = {k: body[k] for k in ("train_loss", "train_acc", "class_counts") if k in body}
                self.pending[cid] = {"skipped": False, "n": n, "delta": delta,
                                     "norm": float(np.linalg.norm(delta)), "meta": meta,
                                     "codec": body.get("codec")}
            self._audit("update", body, "accepted_skip" if body.get("skipped") is True else "accepted")
            accepted_round = self.round
            closed = False
            if all(c in self.pending for c in self.cfg.expected_clients):
                self._close_round("all_received")
                closed = True
            return 200, {"accepted": True, "duplicate": False, "round": accepted_round,
                         "round_closed": closed, "next_round": self.round, "version": self.version,
                         "status": self.status}

    def _close_round(self, reason: str):
        contributors = [(c, p) for c, p in self.pending.items() if not p["skipped"]]
        updates = [(p["delta"], p["n"]) for _, p in contributors]
        before = self.version
        if updates:
            # The server keeps float64; phones receive float32 (same as the offline simulator).
            self.global_w = fedavg(self.global_w, updates)
            self.version += 1
            self._save_global()
        bi, bo = self._traffic_totals()
        entry = {
            "run_id": self.run_id, "round": self.round, "reason": reason,
            "version_before": before, "version_after": self.version,
            "contributors": [c for c, _ in contributors],
            "skipped": [c for c, p in self.pending.items() if p["skipped"]],
            "missing": [c for c in self.cfg.expected_clients if c not in self.pending],
            "update_norms": {c: p["norm"] for c, p in self.pending.items()},
            "n_samples": {c: p["n"] for c, p in contributors},
            "class_counts": {c: p.get("meta", {}).get("class_counts") for c, p in contributors},
            "duration_s": round(self.clock() - self.round_started, 3),
            "body_bytes_in": bi - self._traffic_mark[0], "body_bytes_out": bo - self._traffic_mark[1],
            "model_changed": bool(updates),
        }
        self.round_log.append(entry)
        self._append_jsonl("rounds.jsonl", entry)
        for c in self.cfg.expected_clients:
            p = self.pending.get(c)
            if p is None:
                self.missed[c] += 1
            else:
                self.missed[c] = 0
                self.skips[c] = self.skips[c] + 1 if p["skipped"] else 0
        self.pending = {}
        self.round += 1
        if self.round > self.cfg.max_rounds:
            self.status = "done"
            self.round_started = None
        else:
            self.round_started = self.clock()
        self._traffic_mark = self._traffic_totals()

    # ------------------------------------------------- local models and reports
    def put_local_model(self, client_id: str, weights_b64: str):
        with self.lock:
            if client_id not in self.cfg.expected_clients:
                raise KeyError(client_id)
            w = b64_to_f32(weights_b64)
            if w.shape != (self.P,) or not np.all(np.isfinite(w)):
                raise ValueError("bad weights")
            self.local_models[client_id] = {"weights": weights_b64, "uploaded_at": time.time()}
            np.save(os.path.join(self.cfg.data_dir, f"local_{client_id}.npy"), w)

    def get_local_model(self, client_id: str):
        with self.lock:
            return self.local_models.get(client_id)

    def add_report(self, report: dict):
        """Returns None if stored, else the list of unexpected keys (privacy guard)."""
        with self.lock:
            extra_keys = sorted(set(report) - ALLOWED_REPORT_KEYS)
            if extra_keys:
                self._audit("report", report, "rejected_privacy_guard", unexpected_keys=extra_keys)
                return extra_keys
            self._audit("report", report, "accepted")
            report = dict(report)
            report["received_at"] = time.time()
            report["run_id"] = self.run_id
            self.reports.append(report)
            self._append_jsonl("reports.jsonl", report)
            return None

    def status_dict(self):
        with self.lock:
            self._tick()
            return {
                "status": self.status, "round": self.round, "version": self.version,
                "max_rounds": self.cfg.max_rounds, "registered": sorted(self.registered),
                "received_this_round": sorted(self.pending), "skips": dict(self.skips),
                "missed": dict(self.missed), "local_models": sorted(self.local_models),
                "traffic": self.traffic, "rounds_closed": len(self.round_log),
                "last_rounds": self.round_log[-5:], "n_reports": len(self.reports),
                "n_features": self.cfg.n_features, "n_uploads_audited": len(self.audit),
                "run_id": self.run_id,
            }
