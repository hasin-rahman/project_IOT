"""Federated-learning server for the phone-pose project (Python 3.9+, standard library only).

The server never receives sensor data. It receives only model parameters, sample counts and
evaluation metrics, averages client models with FedAvg, and relays models back to the phones.

    python3 fl_server.py                      # uses config.json, listens on port 8000
    python3 fl_server.py --port 8080 --rounds 30 --feature-set acc+grav+gyro

Open http://localhost:8000/ in a browser for a live status page.

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

import secure_agg
from fedavg import fedavg, validate_model, zero_model

HERE = os.path.dirname(os.path.abspath(__file__))
FEATURE_BLOCKS = ["acc", "grav", "gyro", "rot"]
TRAIN_KEYS = ["local_epochs", "lr", "batch_size", "l2", "seed", "rounds", "feature_set", "baseline_epochs",
              "secure_aggregation"]
DEFAULTS = {"round_timeout_s": 30, "min_clients": 1, "secure_aggregation": False}
SWEEPS = {
    # the required sensor study: 4 feature sets x 3 seeds, everything else fixed
    "sensor_study": [{"feature_set": fs, "seed": seed}
                     for fs in ["acc", "acc+grav", "acc+grav+gyro", "acc+grav+gyro+rot"] for seed in (0, 1, 2)],
    # the baseline repeated with 3 seeds (for mean +/- spread in the report)
    "seeds": [{"seed": seed} for seed in (0, 1, 2)],
    # extra credit: plain FedAvg vs secure aggregation, 3 seeds each
    "secure_study": [{"secure_aggregation": secure, "seed": seed} for secure in (False, True) for seed in (0, 1, 2)],
}


# PRIVACY RULE, enforced in code: these are the ONLY fields a phone may upload.
# There is no field for raw sensor readings or for feature vectors, so the server cannot receive them.
MODEL_FIELDS = {"n_features", "n_classes", "feature_set", "W", "b"}
MASKED_MODEL_FIELDS = {"n_features", "n_classes", "feature_set", "values"}
ENCRYPTED_FIELDS = {"ciphertext", "tag"}
ALLOWED_UPLOAD_FIELDS = {
    "/register": {"client_id", "device", "participant"},
    "/update": {"client_id", "run_id", "base_version", "num_samples", "class_counts", "model",
                "train_loss", "train_acc", "train_ms", "val_acc",
                "masked_model", "attempt", "key_epoch"},          # the last three: secure aggregation
    "/local_model": {"client_id", "model", "num_samples", "class_counts", "encrypted_model"},
    "/pubkey": {"client_id", "run_id", "public_key"},
    "/metrics": {"client_id", "model_name", "test_set", "feature_set", "participant", "metrics"},
    "/done": {"client_id", "run_id"},
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
                         ("encrypted_model", ENCRYPTED_FIELDS)):
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
        self.history = []
        self.metrics_count = 0
        self.local_models = {}
        self.results = {}
        self.finished = set()
        self.traffic = {}  # bytes are counted per run (written to the run folder as traffic.json)
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
            if body.get("key_epoch") != self.key_epoch:
                raise ApiError(409, "stale_keys", key_epoch=self.key_epoch)
            if body.get("attempt") != self.attempt:
                raise ApiError(409, "stale_attempt", attempt=self.attempt)
            model = body.get("masked_model") or {}
            try:
                secure_agg.validate_masked(model.get("values"),
                                           self.n_features * self.config["n_classes"] + self.config["n_classes"])
            except ValueError as e:
                raise ApiError(400, str(e))
        else:
            if "masked_model" in body:
                raise ApiError(400, "secure aggregation is off: send model, not masked_model")
            model = body.get("model") or {}
            try:
                validate_model(model, self.n_features, self.config["n_classes"])
            except ValueError as e:
                raise ApiError(400, str(e))
        if not self.pending:
            self.round_first_at = time.time()
        if isinstance(body.get("class_counts"), list):
            self.clients.setdefault(cid, {})["class_counts"] = body["class_counts"]
        self.pending[cid] = {"model": model, "num_samples": n,
                             "train_loss": body.get("train_loss"), "train_acc": body.get("train_acc"),
                             "train_ms": body.get("train_ms"), "val_acc": body.get("val_acc"),
                             "received_at": time.time()}
        va = body.get("val_acc")
        self.log(f"round {self.version}: update from {cid} ({n} samples, loss={body.get('train_loss')}"
                 + (f", global v{self.version} val acc={va:.3f}" if isinstance(va, (int, float)) else "") + ")")
        aggregated = False
        if all(c in self.pending for c in self.config["expected_clients"]):
            self.aggregate()
            aggregated = True
        return {"accepted": True, "aggregated": aggregated, "version": self.version, "done": self.done}

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
        if self.secure:
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
                  **{f"{k}_{c}": self.pending[c][k] for c in clients
                     for k in ("num_samples", "train_loss", "train_acc", "train_ms", "val_acc")}}
        self.history.append(record)
        self._append("rounds.jsonl", record)
        self.version += 1
        self.pending = {}
        self.round_first_at = None
        self._write(f"global_v{self.version}.json", self.global_model)
        self._write("traffic.json", self.traffic)
        self.log(("Secure aggregation" if self.secure else "FedAvg") + f" over {clients} -> global v{self.version}"
                 + ("  (all rounds done)" if self.done else ""))

    def put_local_model(self, body):
        cid = self._require_client(body)
        entry = {"num_samples": body.get("num_samples"), "uploaded_at": time.time(), "run_id": self.run_id}
        if self.secure:
            # relayed as an encrypted blob that only the other phone can open
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
        self.new_run(cfg)
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
            "secure": self.secure, "discarded_rounds": self.discarded[-20:],
            "uploads": self.uploads,
            "history": self.history[-50:], "traffic": self.traffic, "metrics_received": self.metrics_count,
        }


STATUS_HTML = """<!doctype html><html><head><meta charset=utf-8><title>FL server</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
body{font:15px system-ui,sans-serif;margin:24px auto;max-width:1000px;padding:0 16px;color:#202124}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:26px 0 6px}
td,th{padding:4px 10px;text-align:left;border-bottom:1px solid #e3e3e3}th{color:#5f6368;font-weight:500}
table{border-collapse:collapse;margin:6px 0 10px}.big{font-size:26px;font-weight:650}
.muted{color:#5f6368}.ok{color:#1e8e3e}.warn{color:#b26a00}.bad{color:#c62828}
.pill{display:inline-block;padding:2px 10px;border-radius:12px;background:#eef3f9;margin-right:6px}
.best{font-weight:700}pre{max-height:260px;overflow:auto;background:#f6f8fb;padding:8px;border-radius:6px;font-size:12px;max-width:560px}
summary{cursor:pointer;color:#1F5FA8}td{vertical-align:top}form{background:#f6f8fb;padding:10px 12px;border-radius:8px;display:inline-block}
form *{margin-right:8px}svg text{font-size:12px;fill:#5f6368}
</style></head><body>
<h1>Phone-pose federated learning</h1><div id=s class=muted>loading...</div>
<script>
const C={A:'#2a78d6',B:'#eb6834'};
// server password: asked once and remembered in this browser
let TOKEN='';try{TOKEN=localStorage.getItem('posefl_token')||''}catch(e){}
function api(path,opt){opt=opt||{};opt.headers=Object.assign({'X-Auth-Token':TOKEN},opt.headers||{});return fetch(path,opt)}
function askPassword(){const t=prompt('This server needs a password:');if(t!==null){TOKEN=t;try{localStorage.setItem('posefl_token',t)}catch(e){}}}
function row(c,tag){tag=tag||'td';return '<tr>'+c.map(x=>'<'+tag+'>'+x+'</'+tag+'>').join('')+'</tr>'}
function pct(v){return (typeof v==='number')?(100*v).toFixed(1)+'%':'-'}
function chart(hist,clients){
 const W=640,H=230,L=40,R=14,T=12,B=40,n=hist.length;
 if(!n) return '<p class=muted>No rounds yet. Each phone reports how well the current global model does on its own validation data.</p>';
 const x=i=>L+(n<2?0:(W-L-R)*i/(n-1)), y=v=>T+(H-T-B)*(1-v);
 let g='<svg width='+W+' height='+H+'>';
 for(const v of [0,.25,.5,.75,1]) g+='<line x1='+L+' x2='+(W-R)+' y1='+y(v)+' y2='+y(v)+' stroke=#e6e5e1 /><text x=4 y='+(y(v)+4)+'>'+(v*100)+'%</text>';
 const ticks=[...new Set([0,Math.round((n-1)/2),n-1])];
 for(const i of ticks) g+='<text x='+(x(i)-4)+' y='+(H-B+16)+'>'+hist[i].round+'</text>';
 g+='<text x='+(W/2-70)+' y='+(H-4)+'>round (global model version)</text>';
 let legend='';
 clients.forEach((c,ci)=>{
  const pts=hist.map((r,i)=>[i,r['val_acc_'+c]]).filter(p=>typeof p[1]==='number');
  if(!pts.length) return;
  g+='<polyline fill=none stroke-width=2 stroke='+C[c]+(ci?' stroke-dasharray="6 4"':'')+' points="'+pts.map(p=>x(p[0])+','+y(p[1])).join(' ')+'"/>';
  const l=pts[pts.length-1];
  legend+='<span class=pill><svg width=22 height=10><line x1=0 x2=22 y1=5 y2=5 stroke-width=2 stroke='+C[c]+(ci?' stroke-dasharray="6 4"':'')+' /></svg> phone '+c+': <b>'+pct(l[1])+'</b></span>';
 });
 return '<div>'+legend+'</div>'+g+'</svg><div class=muted>Accuracy of global model v<i>k</i> on each phone&rsquo;s validation set, measured on the phone before it trains round <i>k</i>.</div>';
}
function results(rs){
 if(!rs.length) return '<p class=muted>No evaluation results yet. Phones report them when the last round is done.</p>';
 const sets=[...new Set(rs.map(r=>r.client_id+' / '+r.test_set))].sort();
 const order=['local_A','local_B','global','server_sees'], names={local_A:'Client A local',local_B:'Client B local',global:'Federated global',server_sees:'One upload, as the server sees it'};
 let h='<table><tr><th>model</th>'+sets.map(s=>'<th>'+s+'<br><span class=muted>acc &middot; macro-F1</span></th>').join('')+'</tr>';
 for(const m of order){
  if(m==='server_sees'&&!rs.some(r=>r.model_name===m)) continue;
  const cells=sets.map(s=>{const r=rs.find(r=>r.client_id+' / '+r.test_set===s&&r.model_name===m);
   if(!r) return '-'; const mm=r.metrics; return pct(mm.accuracy)+' &middot; '+(mm.macro_f1??0).toFixed(3)+' <span class=muted>(n='+(mm.n??'?')+')</span>';});
  h+=row(['<b>'+names[m]+'</b>'].concat(cells));
 }
 return h+'</table>';
}
const POSES=['Screen Up','Screen Down','Portrait Up','Portrait Down','Landscape Left','Landscape Right'];
function esc(t){return String(t).replace(/&/g,'&amp;').replace(/</g,'&lt;')}
function recall(rs){
 // per-pose recall of each model on each phone's test recordings (falls back to validation)
 let h='';
 for(const c of [...new Set(rs.map(r=>r.client_id))].sort()){
  const set=['test','val','live_test'].find(t=>rs.some(r=>r.client_id===c&&r.test_set===t)); if(!set) continue;
  h+='<p class=muted>Recall per pose, measured on phone '+c+' ('+set+' recordings). Poses 1&ndash;3 are client A&rsquo;s training poses, 4&ndash;6 are client B&rsquo;s.</p><table>'+row(['model'].concat(POSES),'th');
  for(const m of ['local_A','local_B','global','server_sees']){
   const r=rs.find(r=>r.client_id===c&&r.test_set===set&&r.model_name===m); if(!r||!r.metrics.per_class_recall) continue;
   h+=row(['<b>'+({local_A:'Client A local',local_B:'Client B local',global:'Federated global',server_sees:'One upload, as the server sees it'})[m]+'</b>']
     .concat(r.metrics.per_class_recall.map(v=>'<span class='+(v>=.9?'ok':v<=.1?'bad':'warn')+'>'+pct(v)+'</span>')));
  }
  h+='</table>';
 }
 return h;
}
const openIds=new Set();
function tog(d){if(d.open)openIds.add(d.id);else openIds.delete(d.id)}
function uploads(up){
 const cs=Object.keys(up).filter(c=>c!=='-').sort();
 if(!cs.length) return '<p class=muted>Nothing uploaded yet.</p>';
 let h='<p class=muted>The latest request of each kind from each phone, exactly as it arrived. The server refuses any field that is not on its allow-list, so sensor readings and feature vectors cannot be uploaded.</p>';
 for(const c of cs){
  h+='<table>'+row(['phone '+c,'when','size','what it contains'],'th');
  for(const ep of ['/update','/local_model','/metrics','/register','/done']){
   const u=up[c][ep]; if(!u) continue; const id='u_'+c+ep.replace('/','_');
   h+=row(['<b>POST '+ep+'</b>',new Date(u.time*1000).toLocaleTimeString(),u.bytes+' B',
    esc(u.summary)+'<details id='+id+(openIds.has(id)?' open':'')+' ontoggle="tog(this)"><summary>show the raw JSON</summary><pre>'+esc(JSON.stringify(u.body,null,1))+'</pre></details>']);
  }
  h+='</table>';
 }
 return h;
}
async function reset(ev){
 ev.preventDefault(); const f=ev.target;
 if(!confirm('Start a new run? The current global model is kept on disk, but training starts again from zero.')) return;
 const body={feature_set:f.fs.value,rounds:+f.rounds.value,seed:+f.seed.value,round_timeout_s:+f.to.value,secure_aggregation:f.sec.checked};
 const r=await api('/admin/reset',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 alert(r.ok?'New run started. Phones that are training (or following new runs) pick it up automatically.':'Failed: '+(await r.text()));
}
let formShown=false;
async function load(){
 let s; try{const r=await api('/status');
  if(r.status===401){document.getElementById('s').innerHTML='<span class=bad>Wrong or missing server password.</span> <button onclick="askPassword();load()">Enter password</button>';return;}
  s=await r.json();}catch(e){document.getElementById('s').innerHTML='<span class=bad>Server not reachable</span>';return;}
 const cfg=s.config, exp=cfg.expected_clients;
 const waiting=exp.filter(c=>!s.pending.includes(c));
 let h='<div class=big>Global model v'+s.version+' / '+s.rounds+(s.done?' <span class=ok>&middot; training done</span>':'')+'</div>';
 h+='<p><span class=pill>run '+s.run_id+'</span><span class=pill>features <b>'+cfg.feature_set+'</b> ('+s.n_params+' parameters)</span>'+
   '<span class=pill>'+cfg.local_epochs+' local epoch(s), lr '+cfg.lr+', seed '+cfg.seed+'</span>'+
   (s.secure?'<span class=pill style="background:#e3f4e8"><b>secure aggregation ON</b>: the server sees only masked weights</span>':'<span class=pill>secure aggregation off</span>')+
   (s.sweep?'<span class=pill>sweep run '+(s.sweep.index+1)+' of '+s.sweep.total+'</span>':'')+'</p>';
 if(s.discarded_rounds&&s.discarded_rounds.length) h+='<p class=warn>Discarded rounds (a masked update cannot be used alone): '+
   s.discarded_rounds.map(d=>'round '+d.round+' at '+new Date(d.time*1000).toLocaleTimeString()+' (received only '+(d.received.join(', ')||'nothing')+')').join('; ')+'</p>';
 if(!s.done) h+='<p>'+(s.pending.length?('Round '+s.version+': received '+s.pending.join(', ')+', waiting for '+waiting.join(', ')+
   (s.round_wait_s!=null?' <span class=warn>('+s.round_wait_s+' s; timeout '+cfg.round_timeout_s+' s)</span>':'')):'Waiting for phones to send round '+s.version+' updates.')+'</p>';
 else h+='<p>Clients finished evaluating: '+(s.finished.join(', ')||'none yet')+'</p>';
 h+='<h2>Global model accuracy by round</h2>'+chart(s.history,exp);
 h+='<h2>Model comparison (computed on the phones)</h2>'+results(s.results)+recall(s.results);
 h+='<h2>What the phones upload (privacy evidence)</h2>'+uploads(s.uploads||{});
 h+='<h2>Clients</h2><table>'+row(['id','device','participant','train windows per pose (SU, SD, PU, PD, LL, LR)','last seen'],'th');
 for(const [c,v] of Object.entries(s.clients)){const age=v.seconds_since_seen;
  h+=row([c,v.device||'',v.participant||'',(v.class_counts||[]).join(', ')||'-','<span class='+(age<10?'ok':'bad')+'>'+age+' s ago</span>']);}
 h+='</table><h2>Recent rounds</h2><table>'+row(['round','clients','missing','samples','train loss','val acc'],'th');
 for(const r of s.history.slice(-8).reverse()) h+=row([r.round,r.clients.join(', '),
   (r.missing&&r.missing.length)?'<span class=warn>'+r.missing.join(', ')+'</span>':'-',
   r.clients.map(c=>r['num_samples_'+c]).join(' / '),r.clients.map(c=>(r['train_loss_'+c]??'-')).join(' / '),
   r.clients.map(c=>pct(r['val_acc_'+c])).join(' / ')]);
 h+='</table><h2>Traffic</h2><table>'+row(['client','requests','payload in','payload out','on the wire in','on the wire out'],'th');
 for(const [c,v] of Object.entries(s.traffic)) h+=row([c,v.requests,v.payload_in+' B',v.payload_out+' B',v.wire_in+' B',v.wire_out+' B']);
 h+='</table><p class=muted>Only model parameters, sample counts and metrics reach this server. Local-only models stored: '+(s.local_models.join(', ')||'none')+'.</p>';
 document.getElementById('s').innerHTML=h;
 if(!formShown){formShown=true;const d=document.createElement('div');
  d.innerHTML='<h2>Start a new run</h2><form onsubmit="reset(event)">features <select name=fs>'+
  ['acc','acc+grav','acc+grav+gyro','acc+grav+gyro+rot'].map(f=>'<option'+(f===cfg.feature_set?' selected':'')+'>'+f+'</option>').join('')+
  '</select> rounds <input name=rounds type=number value='+cfg.rounds+' style="width:60px"> seed <input name=seed type=number value='+cfg.seed+
  ' style="width:50px"> timeout (s) <input name=to type=number value='+cfg.round_timeout_s+' style="width:60px"> <label><input name=sec type=checkbox'+(s.secure?' checked':'')+'> secure aggregation</label><button>Start new run</button></form>';
  document.body.appendChild(d);}
}
load();setInterval(load,1500);
</script></body></html>"""


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
                n_out = self._send(200, STATUS_HTML.encode(), "text/html; charset=utf-8")
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
                                    "secure_aggregation": args.secure_aggregation, "host": args.host})
    sweep = load_sweep(args.sweep)
    for o in sweep or []:
        feature_size(o.get("feature_set", cfg["feature_set"]))
    httpd, _ = make_server(cfg, args.runs_dir, sweep, args.password)
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
