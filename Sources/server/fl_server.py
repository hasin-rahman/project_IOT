"""Federated-learning server for the phone-pose project (Python 3.9+, standard library only).

The server never receives sensor data. It receives only model parameters, sample counts and
evaluation metrics, averages client models with FedAvg, and relays models back to the phones.

    python3 fl_server.py                      # uses config.json, listens on port 8000
    python3 fl_server.py --port 8080 --rounds 30 --feature-set acc+grav+gyro

Open http://localhost:8000/ in a browser for the dashboard (dashboard.html).
Server files: fl_server.py (this), fedavg.py, secure_agg.py, curious_server.py, run_summary.py, dashboard.html, config.json.

Protocol (all JSON):
    GET  /health                         liveness + current round
    POST /register      {client_id, device, participant}
    GET  /global?client_id=A             current global model + training settings
    POST /update        {run_id, client_id, base_version, num_samples, model{W,b}, train_loss, ...}
                        -> when every expected client has submitted for this version: FedAvg
                        (with secure aggregation: masked_model instead of model, see secure_agg.py)
    POST /pubkey        {client_id, run_id, public_key}       secure aggregation: publish a public key
    GET  /pubkey?client_id=A                                  ... and fetch the other phone's
    POST /local_model   {client_id, model, num_samples}       store a local-only baseline model
    GET  /local_model?client_id=A                             relay it to the other phone
    POST /metrics       {client_id, model_name, test_set, metrics{...}}
    POST /done          {client_id, run_id}                   client finished evaluating this run
    GET  /status                         full JSON status, including each phone's latest uploads
    GET  /uploads?client_id=A            the exact JSON of the latest upload per endpoint (privacy evidence)
    GET  /results                        latest evaluation metrics per client / model / test set
    POST /device_cost   {client_id, run_id, train_ms_per_round, battery_used_mah, ...}   what a run cost on the phone
    GET  /runs                           one summary per run (Compare runs table; see run_summary.py)
    POST /admin/reset   {optional config overrides}           start a fresh run (new global model)

Password (--password, or the environment variable POSEFL_PASSWORD): needed when the server is on a public
address, e.g. a cloud machine. Every request must then carry it in the X-Auth-Token header (the phones send
it automatically once it is entered in the app); otherwise the server answers 401. The dashboard page asks
for it once and remembers it in that browser. The password is never written to the run folders.

Privacy: every upload is checked against ALLOWED_UPLOAD_FIELDS. A request carrying any other field
(for example sensor rows or feature vectors) is refused with HTTP 400. Every accepted upload is printed
to the console, written to runs/<run_id>/uploads.jsonl and shown on the status page.

Dropout handling: a round waits for every expected client, but once the first update of a round
has arrived, the server waits at most round_timeout_s seconds. It then aggregates whatever arrived
(at least min_clients) and records the missing clients. A late or reconnecting client gets
409 stale_version, downloads the newest global model and continues. If nobody sends an update,
the global model simply stays as it is.

Compressed secure aggregation (--secure --compress): the phones send only the change of their weights,
weighted by their share of the data and rounded to 8 bits, under 16-bit masks: 2 bytes per parameter.

Secure aggregation (--secure, or "secure_aggregation": true): the phones agree on a shared key that
the server never learns, and upload masked weights. Adding the two uploads cancels the masks, so the
server gets the FedAvg result without seeing either phone's model. A round with a missing phone cannot
be unmasked, so it is discarded and repeated (the global model stays as it is).

Experiment sweeps: --sweep sensor_study (or a JSON file with a list of config overrides) queues runs.
When every expected client has called /done for a run, the server starts the next one; phones with
"follow new runs" switched on train again automatically.
"""
import argparse
import hmac
import json
import os
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import run_summary
import secure_agg
from fedavg import fedavg, validate_model, zero_model

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_BLOCKS = ["acc", "grav", "gyro", "rot"]
TRAIN_KEYS = ["local_epochs", "lr", "batch_size", "l2", "seed", "rounds", "feature_set", "baseline_epochs",
              "secure_aggregation", "compression"]
DEFAULTS = {"round_timeout_s": 30, "min_clients": 1, "secure_aggregation": False,
            "compression": False,   # with secure aggregation: 8-bit changes and 16-bit masks (secure_agg.py)
            "quant_clip": 4.0}      # compression: largest change per parameter allowed in the first round
MIN_CLIP = 0.01                     # compression: the scale never shrinks below this
CLIP_MARGIN = 1.5                   # compression: next scale = 1.5 x the largest change the phones reported
SWEEPS = {
    # the required sensor study: 4 feature sets x 3 seeds, everything else fixed
    "sensor_study": [{"feature_set": fs, "seed": seed}
                     for fs in ["acc", "acc+grav", "acc+grav+gyro", "acc+grav+gyro+rot"] for seed in (0, 1, 2)],
    # the baseline repeated with 3 seeds (for mean +/- spread in the report)
    "seeds": [{"seed": seed} for seed in (0, 1, 2)],
    # extra credit: plain FedAvg vs secure aggregation vs compressed secure aggregation, 3 seeds each
    "secure_study": [{"secure_aggregation": secure, "compression": compress, "seed": seed}
                     for secure, compress in ((False, False), (True, False), (True, True)) for seed in (0, 1, 2)],
}


# PRIVACY RULE, enforced in code: these are the ONLY fields a phone may upload.
# There is no field for raw sensor readings or for feature vectors, so the server cannot receive them.
MODEL_FIELDS = {"n_features", "n_classes", "feature_set", "W", "b"}
MASKED_MODEL_FIELDS = {"n_features", "n_classes", "feature_set", "values"}
MASKED_DELTA_FIELDS = {"n_params", "payload"}
ENCRYPTED_FIELDS = {"ciphertext", "tag"}
ALLOWED_UPLOAD_FIELDS = {
    "/register": {"client_id", "device", "participant"},
    "/update": {"client_id", "run_id", "base_version", "num_samples", "class_counts", "model",
                "train_loss", "train_acc", "train_ms", "val_acc",
                "masked_model", "attempt", "key_epoch",            # secure aggregation
                "masked_delta", "max_abs_delta", "clipped"},       # compressed secure aggregation
    "/local_model": {"client_id", "model", "num_samples", "class_counts", "encrypted_model"},
    "/pubkey": {"client_id", "run_id", "public_key", "num_samples"},
    "/metrics": {"client_id", "model_name", "test_set", "feature_set", "participant", "metrics"},
    "/done": {"client_id", "run_id"},
    # what a run cost on the phone: timings, bytes and battery level (no data about the user)
    "/device_cost": {"client_id", "run_id", "run_s", "feature_ms_per_window", "baseline_train_ms",
                     "train_ms_per_round", "predict_ms_per_window", "bytes_sent", "bytes_received", "rounds",
                     "battery_start_pct", "battery_end_pct", "battery_used_mah", "charging"},
}


def check_upload_fields(path, body):
    """Refuse an upload that contains anything other than the allowed fields."""
    allowed = ALLOWED_UPLOAD_FIELDS.get(path)
    if allowed is None or not isinstance(body, dict):
        return
    extra = sorted(set(body) - allowed)
    if extra:
        raise ApiError(400, f"field(s) {extra} are not allowed in {path}: phones may upload only "
                            f"model weights, sample counts and scores")
    for name, fields in (("model", MODEL_FIELDS), ("masked_model", MASKED_MODEL_FIELDS),
                         ("masked_delta", MASKED_DELTA_FIELDS), ("encrypted_model", ENCRYPTED_FIELDS)):
        inner = body.get(name)
        if isinstance(inner, dict):
            extra = sorted(set(inner) - fields)
            if extra:
                raise ApiError(400, f"field(s) {extra} are not allowed inside '{name}'")


def describe_upload(body):
    """One line saying what an upload contains, e.g. 'num_samples=630, model{W[72], b[6]}, train_loss=0.21'."""
    parts = []
    for key, value in body.items():
        if key == "model" and isinstance(value, dict):
            parts.append(f"model{{W[{len(value.get('W', []))}], b[{len(value.get('b', []))}]}}")
        elif key == "masked_model" and isinstance(value, dict):
            parts.append(f"masked_model{{{len(value.get('values', []))} masked numbers}}")
        elif key == "masked_delta" and isinstance(value, dict):
            parts.append(f"masked_delta{{{len(value.get('payload', '')) * 3 // 4} bytes = {value.get('n_params')} x 2-byte masked numbers}}")
        elif key == "encrypted_model" and isinstance(value, dict):
            parts.append(f"encrypted_model{{{len(value.get('ciphertext', '')) // 2} encrypted bytes}}")
        elif key == "public_key":
            parts.append(f"public_key({len(str(value)) * 4} bits)")
        elif key == "metrics" and isinstance(value, dict):
            parts.append(f"metrics{{accuracy={value.get('accuracy')}, macro_f1={value.get('macro_f1')}, n={value.get('n')}, ...}}")
        else:
            parts.append(f"{key}={json.dumps(value)}")
    return ", ".join(parts)


def feature_size(spec):
    blocks = [b for b in spec.split("+") if b]
    for b in blocks:
        if b not in FEATURE_BLOCKS:
            raise ValueError(f"unknown feature block '{b}'")
    return 6 * len(blocks)


class ApiError(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status = status
        self.body = {"error": message, **extra}


class FLState:
    """All server state. Every public method is called with self.lock held."""

    def __init__(self, config, runs_dir, sweep=None, password=""):
        self.lock = threading.Lock()
        self.password = password or ""   # kept out of self.config so it never reaches config.json or /status
        self.runs_dir = runs_dir
        self.clients = {}
        self.traffic = {}
        self.uploads = {}  # client -> endpoint -> the latest upload (privacy evidence for the status page)
        self.base_config = dict(DEFAULTS, **config)
        self.sweep = list(sweep or [])
        self.sweep_index = 0
        self.sweep_runs = []
        self.new_run(dict(self.base_config, **(self.sweep[0] if self.sweep else {})))

    # ------------------------------------------------------------------ runs
    def new_run(self, config):
        if config.get("compression") and not config.get("secure_aggregation"):
            raise ValueError("compression works together with secure aggregation: use --secure --compress")
        self.config = dict(config)
        self.n_features = feature_size(self.config["feature_set"])
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        suffix = 1
        while os.path.exists(os.path.join(self.runs_dir, run_id)):
            suffix += 1
            run_id = time.strftime("run_%Y%m%d_%H%M%S") + f"_{suffix}"
        self.run_id = run_id
        self.version = 0
        self.global_model = zero_model(self.n_features, self.config["n_classes"], self.config["feature_set"])
        self.pending = {}
        self.round_first_at = None
        self.public_keys = {}     # secure aggregation: client -> public key (the server only relays them)
        self.key_epoch = 0        # goes up whenever a phone publishes a new key
        self.attempt = 0          # goes up whenever a round is discarded, so its mask is never reused
        self.discarded = []       # rounds that could not be unmasked because a phone was missing
        self.sample_counts = {}   # compression: client -> number of training windows (sent with the public key)
        self.clip = float(self.config.get("quant_clip", 4.0))   # compression: the scale of this round
        self.history = []
        self.metrics_count = 0
        self.local_models = {}
        self.results = {}
        self.finished = set()
        self.traffic = {}  # bytes are counted per run (written to the run folder as traffic.json)
        self.device_costs = {}   # client -> what the run cost on the phone (time, data, battery)
        for info in self.clients.values():   # per-run facts about the phones start empty again
            info.pop("class_counts", None)
            info.pop("num_samples", None)
        self.run_dir = os.path.join(self.runs_dir, self.run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._write("config.json", self.config)
        self._write("global_v0.json", self.global_model)
        if self.sweep:
            self.sweep_runs.append(self.run_id)
        self.log(f"new run {self.run_id}" + (f" (sweep {self.sweep_index + 1}/{len(self.sweep)})" if self.sweep else "")
                 + f": feature_set={self.config['feature_set']} "
                 f"({self.n_features} features, {self.n_features * self.config['n_classes'] + self.config['n_classes']} params), "
                 f"{self.config['rounds']} rounds, clients={self.config['expected_clients']}"
                 + (", SECURE AGGREGATION ON" if self.secure else ""))

    @property
    def secure(self):
        return bool(self.config.get("secure_aggregation"))

    @property
    def compressed(self):
        return self.secure and bool(self.config.get("compression"))

    def total_samples(self):
        """Compression: all phones' training windows together, once every expected phone has reported its count."""
        counts = [self.sample_counts.get(c) for c in self.config["expected_clients"]]
        return sum(counts) if all(counts) else None

    @property
    def done(self):
        return self.version >= self.config["rounds"]

    def train_config(self):
        return {k: self.config[k] for k in TRAIN_KEYS if k in self.config}

    # ------------------------------------------------------------------ helpers
    def log(self, msg):
        print(time.strftime("[%H:%M:%S] ") + msg, flush=True)

    def _write(self, name, obj):
        with open(os.path.join(self.run_dir, name), "w") as fh:
            json.dump(obj, fh, indent=1)

    def _append(self, name, obj):
        with open(os.path.join(self.run_dir, name), "a") as fh:
            fh.write(json.dumps(obj) + "\n")

    def _require_client(self, body):
        cid = body.get("client_id")
        if not isinstance(cid, str) or not cid:
            raise ApiError(400, "client_id is required")
        return cid

    def check_password(self, given):
        """Refuse the request unless it carries the right password (when one is set)."""
        if self.password and not hmac.compare_digest(str(given or "").encode(), self.password.encode()):
            raise ApiError(401, "wrong or missing server password")

    def touch(self, cid):
        if cid:
            self.clients.setdefault(cid, {})["last_seen"] = time.time()

    def count_traffic(self, cid, endpoint, bytes_in, bytes_out, wire_in, wire_out):
        key = cid or "-"
        t = self.traffic.setdefault(key, {"requests": 0, "payload_in": 0, "payload_out": 0,
                                          "wire_in": 0, "wire_out": 0, "by_endpoint": {}})
        t["requests"] += 1
        t["payload_in"] += bytes_in
        t["payload_out"] += bytes_out
        t["wire_in"] += wire_in
        t["wire_out"] += wire_out
        e = t["by_endpoint"].setdefault(endpoint, {"requests": 0, "payload_in": 0, "payload_out": 0})
        e["requests"] += 1
        e["payload_in"] += bytes_in
        e["payload_out"] += bytes_out

    def record_upload(self, cid, path, body, n_bytes):
        """Keep, print and save exactly what a phone uploaded, so we can show that it is only
        model weights and metadata."""
        if path not in ALLOWED_UPLOAD_FIELDS or not isinstance(body, dict):
            return
        entry = {"time": time.time(), "client_id": cid, "endpoint": path, "bytes": n_bytes,
                 "fields": sorted(body.keys()), "summary": describe_upload(body), "body": body}
        self.uploads.setdefault(cid or "-", {})[path] = entry
        self._append("uploads.jsonl", entry)
        if path in ("/update", "/local_model", "/pubkey"):
            self.log(f"  upload from {cid}: POST {path}, {n_bytes} bytes: {entry['summary']}")

    def get_uploads(self, query):
        cid = (query.get("client_id") or [""])[0]
        return {"client_id": cid, "uploads": self.uploads.get(cid, {})}

    # ------------------------------------------------------------------ endpoints
    def health(self):
        return {"status": "ok", "run_id": self.run_id, "version": self.version,
                "rounds": self.config["rounds"], "done": self.done}

    def register(self, body):
        cid = self._require_client(body)
        info = self.clients.setdefault(cid, {})
        info.update({"device": body.get("device", "?"), "participant": body.get("participant", "?"),
                     "registered_at": time.time(), "last_seen": time.time()})
        self.log(f"client {cid} registered ({info['device']}, participant {info['participant']})")
        return {"ok": True, "run_id": self.run_id, "expected": cid in self.config["expected_clients"],
                "train_config": self.train_config()}

    def get_global(self):
        return {"run_id": self.run_id, "version": self.version, "done": self.done,
                "submitted": sorted(self.pending.keys()),
                "attempt": self.attempt, "key_epoch": self.key_epoch,
                **({"clip": self.clip, "total_samples": self.total_samples()} if self.compressed else {}),
                "model": self.global_model, "train_config": self.train_config()}

    def update(self, body):
        cid = self._require_client(body)
        if cid not in self.config["expected_clients"]:
            raise ApiError(400, f"client '{cid}' is not in expected_clients {self.config['expected_clients']}")
        if body.get("run_id") != self.run_id:
            raise ApiError(409, "stale_run", run_id=self.run_id, version=self.version)
        if self.done:
            raise ApiError(409, "done", version=self.version)
        if body.get("base_version") != self.version:
            raise ApiError(409, "stale_version", version=self.version)
        n = body.get("num_samples")
        if not isinstance(n, int) or n <= 0:
            raise ApiError(400, "num_samples must be a positive integer")
        if self.secure:
            # the server must never receive plain weights in this mode
            if "model" in body:
                raise ApiError(400, "secure aggregation is on: send masked_model, not model")
            self.refuse_label_counts(body)
            if body.get("key_epoch") != self.key_epoch:
                raise ApiError(409, "stale_keys", key_epoch=self.key_epoch)
            if body.get("attempt") != self.attempt:
                raise ApiError(409, "stale_attempt", attempt=self.attempt)
            n_params = self.n_features * self.config["n_classes"] + self.config["n_classes"]
            if self.compressed:
                if "masked_model" in body or not isinstance(body.get("masked_delta"), dict):
                    raise ApiError(400, "compression is on: send masked_delta")
                if n != self.sample_counts.get(cid):
                    raise ApiError(400, "num_samples differs from the count sent with the public key")
                if not isinstance(body.get("max_abs_delta"), (int, float)):
                    raise ApiError(400, "max_abs_delta is required with compression")
                model = body["masked_delta"]
                try:
                    secure_agg.validate_compressed(model.get("payload"), n_params)
                except ValueError as e:
                    raise ApiError(400, str(e))
            else:
                if "masked_delta" in body:
                    raise ApiError(400, "compression is off: send masked_model")
                model = body.get("masked_model") or {}
                try:
                    secure_agg.validate_masked(model.get("values"), n_params)
                except ValueError as e:
                    raise ApiError(400, str(e))
        else:
            if "masked_model" in body or "masked_delta" in body:
                raise ApiError(400, "secure aggregation is off: send model, not masked_model")
            model = body.get("model") or {}
            try:
                validate_model(model, self.n_features, self.config["n_classes"])
            except ValueError as e:
                raise ApiError(400, str(e))
        if not self.pending:
            self.round_first_at = time.time()
        info = self.clients.setdefault(cid, {})
        info["num_samples"] = n
        if isinstance(body.get("class_counts"), list):   # plain mode only (refused with secure aggregation)
            info["class_counts"] = body["class_counts"]
        self.pending[cid] = {"model": model, "num_samples": n,
                             "train_loss": body.get("train_loss"), "train_acc": body.get("train_acc"),
                             "train_ms": body.get("train_ms"), "val_acc": body.get("val_acc"),
                             "max_abs_delta": body.get("max_abs_delta"), "clipped": body.get("clipped"),
                             "received_at": time.time()}
        va = body.get("val_acc")
        self.log(f"round {self.version}: update from {cid} ({n} samples, loss={body.get('train_loss')}"
                 + (f", global v{self.version} val acc={va:.3f}" if isinstance(va, (int, float)) else "") + ")")
        aggregated = False
        if all(c in self.pending for c in self.config["expected_clients"]):
            self.aggregate()
            aggregated = True
        return {"accepted": True, "aggregated": aggregated, "version": self.version, "done": self.done}

    def refuse_label_counts(self, body):
        """Secure aggregation hides which poses a phone has, so the per-pose counts (which would give
        that away directly) must not be sent either. Only the total number of windows is allowed."""
        if "class_counts" in body:
            raise ApiError(400, "secure aggregation is on: class_counts would reveal which poses this phone has; "
                                "send only num_samples")

    def tick(self):
        """Called about twice a second: closes a round whose timeout has expired (client dropout)."""
        timeout = self.config.get("round_timeout_s") or 0
        if (timeout > 0 and self.pending and self.round_first_at is not None and not self.done
                and time.time() - self.round_first_at >= timeout
                and len(self.pending) >= max(1, self.config.get("min_clients", 1))):
            missing = [c for c in self.config["expected_clients"] if c not in self.pending]
            if self.secure:
                self.discard_round(f"timeout after {timeout}s, no update from {missing}")
                return
            self.log(f"round {self.version}: timeout after {timeout}s, no update from {missing}; "
                     f"aggregating the {len(self.pending)} update(s) that arrived")
            self.aggregate()

    def discard_round(self, reason):
        """Secure aggregation: one masked update alone is noise (its mask only cancels against the other
        phone's), so the round is thrown away and repeated. The global model stays as it is."""
        record = {"round": self.version, "attempt": self.attempt, "time": time.time(),
                  "received": sorted(self.pending.keys()), "reason": reason}
        self.discarded.append(record)
        self._append("discarded_rounds.jsonl", record)
        self.log(f"round {self.version}: DISCARDED ({reason}); a single masked update cannot be unmasked. "
                 f"The global model stays at v{self.version} and the round is repeated.")
        self.pending = {}
        self.round_first_at = None
        self.attempt += 1

    def aggregate(self):
        clients = sorted(self.pending.keys())
        missing = [c for c in self.config["expected_clients"] if c not in self.pending]
        clip_used = self.clip
        if self.compressed:
            # add the compressed masked changes: the masks cancel and the FedAvg change is left
            n_w = self.n_features * self.config["n_classes"]
            flat_old = self.global_model["W"] + self.global_model["b"]
            change = secure_agg.sum_compressed([self.pending[c]["model"]["payload"] for c in clients],
                                               len(flat_old), self.clip)
            flat = [w + d for w, d in zip(flat_old, change)]
            W, b = flat[:n_w], flat[n_w:]
            # next round's scale follows the largest change the phones reported (one number each, in the clear)
            largest = max(self.pending[c]["max_abs_delta"] for c in clients)
            self.clip = max(MIN_CLIP, CLIP_MARGIN * largest)
        elif self.secure:
            # add the masked uploads: the masks cancel and the sample-weighted average is left
            total = sum(self.pending[c]["num_samples"] for c in clients)
            flat = secure_agg.sum_masked([self.pending[c]["model"]["values"] for c in clients], total)
            n_w = self.n_features * self.config["n_classes"]
            W, b = flat[:n_w], flat[n_w:]
        else:
            updates = [(self.pending[c]["model"], self.pending[c]["num_samples"]) for c in clients]
            W, b = fedavg(updates)
        self.global_model = dict(self.global_model, W=W, b=b)
        record = {"round": self.version, "new_version": self.version + 1, "time": time.time(), "clients": clients,
                  "missing": missing,
                  "duration_s": round(time.time() - self.round_first_at, 3) if self.round_first_at else None,
                  **({"clip": clip_used} if self.compressed else {}),
                  **{f"{k}_{c}": self.pending[c][k] for c in clients
                     for k in ("num_samples", "train_loss", "train_acc", "train_ms", "val_acc", "clipped")
                     if k != "clipped" or self.compressed}}
        self.history.append(record)
        self._append("rounds.jsonl", record)
        self.version += 1
        self.pending = {}
        self.round_first_at = None
        self._write(f"global_v{self.version}.json", self.global_model)
        self._write("traffic.json", self.traffic)
        mode = "Compressed secure aggregation" if self.compressed else "Secure aggregation" if self.secure else "FedAvg"
        self.log(mode + f" over {clients} -> global v{self.version}"
                 + ("  (all rounds done)" if self.done else ""))

    def put_local_model(self, body):
        cid = self._require_client(body)
        entry = {"num_samples": body.get("num_samples"), "uploaded_at": time.time(), "run_id": self.run_id}
        if self.secure:
            # relayed as an encrypted blob that only the other phone can open
            self.refuse_label_counts(body)
            blob = body.get("encrypted_model")
            if "model" in body or not isinstance(blob, dict) or not isinstance(blob.get("ciphertext"), str):
                raise ApiError(400, "secure aggregation is on: send encrypted_model, not model")
            entry["encrypted_model"] = blob
        else:
            model = body.get("model") or {}
            nf = model.get("n_features", self.n_features)
            try:
                validate_model(model, nf, self.config["n_classes"])
            except ValueError as e:
                raise ApiError(400, str(e))
            entry["model"] = model
        self.local_models[cid] = entry
        self._write(f"local_{cid}.json", entry)
        self.log(f"stored local-only model of client {cid}" + (" (encrypted, the server cannot read it)" if self.secure else ""))
        return {"ok": True}

    def put_public_key(self, body):
        """Secure aggregation: the server only passes public keys between the phones."""
        cid = self._require_client(body)
        if body.get("run_id") != self.run_id:
            raise ApiError(409, "stale_run", run_id=self.run_id, version=self.version)
        key = body.get("public_key")
        if not isinstance(key, str) or not key:
            raise ApiError(400, "public_key is required")
        if isinstance(body.get("num_samples"), int) and body["num_samples"] > 0:
            self.sample_counts[cid] = body["num_samples"]
        if self.public_keys.get(cid) != key:
            self.public_keys[cid] = key
            self.key_epoch += 1
            self.log(f"client {cid} published a public key (key epoch {self.key_epoch})")
            if self.pending:   # updates masked with the old key can no longer be unmasked
                self.discard_round(f"client {cid} changed its key")
        return {"ok": True, "key_epoch": self.key_epoch}

    def get_public_key(self, query):
        cid = (query.get("client_id") or [""])[0]
        if cid not in self.public_keys:
            raise ApiError(404, f"no public key from client '{cid}' yet")
        return {"client_id": cid, "public_key": self.public_keys[cid], "key_epoch": self.key_epoch}

    def get_local_model(self, query):
        cid = (query.get("client_id") or [""])[0]
        if cid not in self.local_models:
            raise ApiError(404, f"no local model from client '{cid}' yet")
        return self.local_models[cid]

    def put_metrics(self, body):
        cid = self._require_client(body)
        rec = dict(body, run_id=self.run_id, global_version=self.version, received_at=time.time())
        self._append("metrics.jsonl", rec)
        self.metrics_count += 1
        key = f"{cid}|{body.get('model_name')}|{body.get('test_set')}"
        self.results[key] = {"client_id": cid, "model_name": body.get("model_name"),
                             "test_set": body.get("test_set"), "metrics": body.get("metrics", {})}
        m = body.get("metrics", {})
        self.log(f"metrics from {cid}: {body.get('model_name')} on {body.get('test_set')}: "
                 f"acc={m.get('accuracy')} macroF1={m.get('macro_f1')}")
        return {"ok": True}

    def put_device_cost(self, body):
        """What the run cost on a phone (time, data, battery), shown on the dashboard and in Compare runs."""
        cid = self._require_client(body)
        if body.get("run_id") != self.run_id:
            return {"ok": True, "ignored": "old run"}
        self.device_costs[cid] = dict(body, received_at=time.time())
        self._write(f"device_cost_{cid}.json", self.device_costs[cid])
        self.log(f"device cost from {cid}: {body.get('train_ms_per_round')} ms per round, "
                 f"{body.get('predict_ms_per_window')} ms per prediction, battery {body.get('battery_used_mah')} mAh")
        return {"ok": True}

    def done_client(self, body):
        cid = self._require_client(body)
        if body.get("run_id") != self.run_id:
            return {"ok": True, "ignored": "old run"}
        self.finished.add(cid)
        self.log(f"client {cid} finished run {self.run_id}")
        if self.sweep and all(c in self.finished for c in self.config["expected_clients"]):
            if self.sweep_index + 1 < len(self.sweep):
                self.sweep_index += 1
                self.new_run(dict(self.base_config, **self.sweep[self.sweep_index]))
            else:
                self.log(f"sweep finished: {len(self.sweep_runs)} runs: {' '.join(self.sweep_runs)}")
                self.log("analyse with:  python3 analyze_run.py " + " ".join(f"runs/{r}" for r in self.sweep_runs))
        return {"ok": True}

    def get_results(self):
        return {"run_id": self.run_id, "results": list(self.results.values())}

    def reset(self, body):
        cfg = dict(DEFAULTS, **self.config)
        for k, v in (body or {}).items():
            if k in cfg:
                cfg[k] = v
        feature_size(cfg["feature_set"])  # validate
        self.sweep = []
        try:
            self.new_run(cfg)
        except ValueError as e:
            raise ApiError(400, str(e))
        return {"ok": True, "run_id": self.run_id, "config": self.config}

    def status(self):
        now = time.time()
        return {
            "run_id": self.run_id, "version": self.version, "rounds": self.config["rounds"], "done": self.done,
            "config": self.config, "n_params": len(self.global_model["W"]) + len(self.global_model["b"]),
            "pending": sorted(self.pending.keys()),
            "round_wait_s": round(time.time() - self.round_first_at, 1) if self.round_first_at else None,
            "finished": sorted(self.finished),
            "sweep": {"index": self.sweep_index, "total": len(self.sweep), "runs": self.sweep_runs} if self.sweep else None,
            "results": list(self.results.values()),
            "clients": {c: dict(v, seconds_since_seen=round(now - v.get("last_seen", now), 1))
                        for c, v in self.clients.items()},
            "local_models": sorted(self.local_models.keys()),
            "secure": self.secure, "compressed": self.compressed, "discarded_rounds": self.discarded[-20:],
            "uploads": self.uploads, "device_costs": self.device_costs,
            "history": self.history[-50:], "traffic": self.traffic, "metrics_received": self.metrics_count,
        }


DASHBOARD_FILE = os.path.join(HERE, "dashboard.html")   # the status page (plain HTML + JavaScript)


def dashboard_page():
    """The dashboard, read from dashboard.html on every request (so it can be edited without a restart)."""
    try:
        with open(DASHBOARD_FILE, "rb") as fh:
            return fh.read()
    except OSError:
        return (b"<p>dashboard.html is missing: copy it next to fl_server.py. "
                b"The server itself works; the data is at <a href=/status>/status</a>.</p>")


class _CountingWriter:
    """Wraps the socket writer so we can measure the exact bytes sent (headers + body)."""

    def __init__(self, raw):
        self.raw = raw
        self.count = 0

    def write(self, data):
        self.count += len(data)
        return self.raw.write(data)

    def __getattr__(self, name):
        return getattr(self.raw, name)


def make_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def setup(self):
            super().setup()
            self.wfile = _CountingWriter(self.wfile)

        def log_message(self, fmt, *args):  # keep the console readable
            pass

        def _send(self, status, payload, content_type="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return len(data)

        def _handle(self, method):
            self.wfile.count = 0
            url = urlparse(self.path)
            query = parse_qs(url.query)
            raw = b""
            if method == "POST":
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            cid = (query.get("client_id") or [None])[0]
            try:
                body = json.loads(raw) if raw else {}
                if isinstance(body, dict):
                    cid = body.get("client_id", cid)
                if not (method == "GET" and url.path == "/"):   # the dashboard page itself holds no data
                    state.check_password(self.headers.get("X-Auth-Token") or (query.get("token") or [""])[0])
                with state.lock:
                    state.touch(cid)
                    if method == "POST":
                        check_upload_fields(url.path, body)   # privacy allow-list
                    status, out = 200, self.route(method, url.path, body, query)
                    if method == "POST":
                        state.record_upload(cid, url.path, body, len(raw))
            except ApiError as e:
                status, out = e.status, e.body
            except json.JSONDecodeError:
                status, out = 400, {"error": "invalid JSON"}
            except Exception as e:  # noqa: BLE001 - report instead of crashing the server
                status, out = 500, {"error": f"{type(e).__name__}: {e}"}
            if url.path == "/" and status == 200:
                n_out = self._send(200, dashboard_page(), "text/html; charset=utf-8")
            else:
                n_out = self._send(status, out)
            wire_in = len(self.requestline) + 2 + len(str(self.headers)) + len(raw)
            with state.lock:
                state.count_traffic(cid, url.path, len(raw), n_out, wire_in, self.wfile.count)

        def route(self, method, path, body, query):
            if method == "GET":
                if path == "/":
                    return {}
                if path == "/health":
                    return state.health()
                if path == "/global":
                    return state.get_global()
                if path == "/local_model":
                    return state.get_local_model(query)
                if path == "/status":
                    return state.status()
                if path == "/results":
                    return state.get_results()
                if path == "/uploads":
                    return state.get_uploads(query)
                if path == "/pubkey":
                    return state.get_public_key(query)
                if path == "/runs":
                    return {"current": state.run_id, "runs": run_summary.summarize_runs(state.runs_dir)}
            else:
                if path == "/register":
                    return state.register(body)
                if path == "/update":
                    return state.update(body)
                if path == "/local_model":
                    return state.put_local_model(body)
                if path == "/pubkey":
                    return state.put_public_key(body)
                if path == "/metrics":
                    return state.put_metrics(body)
                if path == "/device_cost":
                    return state.put_device_cost(body)
                if path == "/done":
                    return state.done_client(body)
                if path == "/admin/reset":
                    return state.reset(body)
            raise ApiError(404, f"unknown endpoint {method} {path}")

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

    return Handler


def lan_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))  # no packet is sent; just picks the outgoing interface
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return "127.0.0.1"


def load_config(path, overrides):
    with open(path) as fh:
        cfg = json.load(fh)
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    return cfg


def make_server(config, runs_dir=None, sweep=None, password=""):
    state = FLState(config, runs_dir or os.path.join(HERE, "runs"), sweep, password)
    httpd = ThreadingHTTPServer((config["host"], config["port"]), make_handler(state))

    def ticker():
        while True:
            time.sleep(0.5)
            with state.lock:
                try:
                    state.tick()
                except Exception as e:  # noqa: BLE001 - never kill the ticker
                    state.log(f"tick error: {e}")

    threading.Thread(target=ticker, daemon=True).start()
    return httpd, state


def load_sweep(name):
    if not name:
        return None
    if name in SWEEPS:
        return SWEEPS[name]
    with open(name) as fh:
        sweep = json.load(fh)
    if not isinstance(sweep, list) or not all(isinstance(x, dict) for x in sweep):
        raise SystemExit("a sweep file must contain a JSON list of config-override objects")
    return sweep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(HERE, "config.json"))
    ap.add_argument("--port", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--feature-set", dest="feature_set")
    ap.add_argument("--local-epochs", dest="local_epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--round-timeout", dest="round_timeout_s", type=float,
                    help="seconds to wait for missing clients once a round has started (0 = wait forever)")
    ap.add_argument("--secure", dest="secure_aggregation", action="store_const", const=True,
                    help="secure aggregation: phones upload masked weights that only make sense once added together")
    ap.add_argument("--compress", dest="compression", action="store_const", const=True,
                    help="with --secure: send 8-bit changes under 16-bit masks (2 bytes per parameter)")
    ap.add_argument("--sweep", help="queue several runs: 'sensor_study', 'seeds', 'secure_study', or a JSON file with a list of overrides")
    ap.add_argument("--runs-dir", default=os.path.join(HERE, "runs"))
    ap.add_argument("--password", default=os.environ.get("POSEFL_PASSWORD", ""),
                    help="require this password on every request (use it when the server is on a public address)")
    ap.add_argument("--public-url", help="the address to show for the phones, e.g. https://34-1-2-3.sslip.io (behind Caddy)")
    ap.add_argument("--public-ip", help="the external IP to show for the phones, when they connect to this port directly")
    ap.add_argument("--host", help="network interface to listen on; 127.0.0.1 = only reachable through Caddy on the same machine")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, {"port": args.port, "rounds": args.rounds, "seed": args.seed,
                                    "feature_set": args.feature_set, "local_epochs": args.local_epochs,
                                    "round_timeout_s": args.round_timeout_s,
                                    "secure_aggregation": args.secure_aggregation, "compression": args.compression,
                                    "host": args.host})
    sweep = load_sweep(args.sweep)
    for o in sweep or []:
        feature_size(o.get("feature_set", cfg["feature_set"]))
    try:
        httpd, _ = make_server(cfg, args.runs_dir, sweep, args.password)
    except ValueError as e:
        raise SystemExit(f"error: {e}")
    if args.public_url:
        url = args.public_url.rstrip("/")
    else:
        url = f"http://{args.public_ip or lan_ip()}" + ("" if cfg["port"] == 80 else f":{cfg['port']}")
    print(f"\nServer listening on {cfg['host']} port {cfg['port']}.")
    print(f"  On the phones, set the server URL to:  {url}")
    if args.password:
        print("  Password required: enter it on the phones (Server password) and on the dashboard.")
    else:
        print("  No password set. Use --password when the server is on a public address.")
    if args.public_url or args.public_ip:
        print(f"  Dashboard:                            {url}/\n")
    else:
        print(f"  Status page on this computer:         http://localhost:{cfg['port']}/")
        print("  (On Google Cloud, use cloud/start_server.sh instead: see README section 13.)\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    sys.exit(main())
