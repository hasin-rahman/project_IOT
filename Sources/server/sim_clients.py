"""Two simulated phones that talk to the real server over HTTP, using SYNTHETIC data.

Use it to test the server and protocol without phones, and as the reference for the client
logic the Android app implements:

  1. register
  2. train a local-only baseline from scratch on local data, upload it (for relaying)
  3. FL loop: GET /global -> measure it on the local validation set -> train a copy for
     local_epochs -> POST /update (409 = the round moved on: just fetch the newest model)
  4. download the other client's local model, evaluate all three models on the local six-pose
     test set, POST /metrics, POST /done

    python3 fl_server.py                   # terminal 1
    python3 sim_clients.py                 # terminal 2 (default http://localhost:8000)
"""
import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import numpy as np

import secure_agg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reference"))
from posefl_ref import features as F  # noqa: E402
from posefl_ref.metrics import summary  # noqa: E402
from posefl_ref.model import SoftmaxModel  # noqa: E402
from posefl_ref.poses import CLIENT_TRAIN_POSES, N_CLASSES  # noqa: E402
from posefl_ref.synth import Person, make_session  # noqa: E402


PASSWORD = ""   # set with --password when the server requires one


def call(base, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json", "X-Auth-Token": PASSWORD})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


class SimPhone(threading.Thread):
    """A simulated phone. follow_runs=True keeps training whenever the server starts a new run
    (experiment sweeps). pause_rounds={k: seconds} makes the phone go silent before round k
    (to test the server's dropout timeout)."""

    def __init__(self, base, cid, seed, feature_set=None, follow_runs=False, pause_rounds=None, max_runs=None):
        super().__init__(daemon=True)
        self.base, self.cid = base, cid
        self.other = "B" if cid == "A" else "A"
        self.private = self.key = self.key_epoch = None
        self.rng = np.random.default_rng(seed)
        self.follow_runs, self.pause_rounds, self.max_runs = follow_runs, dict(pause_rounds or {}), max_runs
        self.result = None
        self.results_by_run = {}
        self.stale_count = 0
        person = Person(self.rng)
        # raw sessions stay on the "phone"; features are computed for whatever feature set a run uses
        self.sessions = {"train": [], "val": [], "test": []}
        for p in CLIENT_TRAIN_POSES[cid]:
            for _ in range(3):
                self.sessions["train"].append((p, make_session(p, person, self.rng, holding=self.rng.choice(["hand", "surface"]))))
        for split in ("val", "test"):
            for p in range(N_CLASSES):
                self.sessions[split].append((p, make_session(p, person, self.rng)))

    def data(self, split, spec):
        X, y = [], []
        for p, rows in self.sessions[split]:
            f = F.extract(rows, spec)
            X.append(f)
            y += [p] * len(f)
        return np.concatenate(X), np.array(y)

    def run(self):
        _, reg = call(self.base, "POST", "/register", {"client_id": self.cid, "device": "simulator", "participant": "sim"})
        current_run, finished, runs_done = None, False, 0
        while True:
            _, g = call(self.base, "GET", f"/global?client_id={self.cid}")
            tc = g["train_config"]
            if g["run_id"] != current_run:  # a new run: rebuild features, train the local-only baseline
                current_run, finished = g["run_id"], False
                spec = tc["feature_set"]
                hp = dict(lr=tc["lr"], batch_size=tc["batch_size"], l2=tc["l2"])
                X, y = self.data("train", spec)
                Xv, yv = self.data("val", spec)
                Xt, yt = self.data("test", spec)
                local = SoftmaxModel(F.feature_size(spec))
                local.train(X, y, epochs=tc["baseline_epochs"], rng=self.rng, **hp)
                secure = bool(tc.get("secure_aggregation"))
                if secure:
                    # secure aggregation: publish a fresh public key; the local model is uploaded later, encrypted
                    self.private, self.key, self.key_epoch = secure_agg.new_private_key(), None, None
                    call(self.base, "POST", "/pubkey", {"client_id": self.cid, "run_id": current_run,
                                                        "public_key": secure_agg.public_key(self.private)})
                else:
                    call(self.base, "POST", "/local_model",
                         {"client_id": self.cid, "model": local.to_dict(spec), "num_samples": len(y)})
            if secure and (self.key is None or self.key_epoch != g["key_epoch"]):
                # (re)compute the shared key from the other phone's public key, then upload the encrypted local model
                st, pk = call(self.base, "GET", f"/pubkey?client_id={self.other}")
                if st == 200:
                    self.key, self.key_epoch = secure_agg.shared_key(self.private, pk["public_key"]), pk["key_epoch"]
                    blob = secure_agg.encrypt(self.key, f"local|{current_run}|{self.cid}", json.dumps(local.to_dict(spec)))
                    call(self.base, "POST", "/local_model",
                         {"client_id": self.cid, "encrypted_model": blob, "num_samples": len(y)})
                elif not g["done"]:
                    time.sleep(0.05)   # the other phone has not published its key yet
                    continue
            if g["done"]:
                if not finished:
                    self.evaluate(g, local, Xt, yt, spec)
                    finished, runs_done = True, runs_done + 1
                if not self.follow_runs or (self.max_runs and runs_done >= self.max_runs):
                    return
                time.sleep(0.05)
                continue
            if self.cid in g["submitted"]:
                time.sleep(0.05)
                continue
            if g["version"] in self.pause_rounds:
                # simulated dropout: the phone goes quiet, then sends an update for the old version
                time.sleep(self.pause_rounds.pop(g["version"]))
            m = SoftmaxModel.from_dict(g["model"])
            _, val_acc = m.evaluate(Xv, yv)
            t0 = time.time()
            m.train(X, y, epochs=tc["local_epochs"], rng=self.rng, **hp)
            loss, acc = m.evaluate(X, y)
            update = {
                "client_id": self.cid, "run_id": g["run_id"], "base_version": g["version"],
                "num_samples": len(y), "class_counts": np.bincount(y, minlength=N_CLASSES).tolist(),
                "val_acc": round(val_acc, 4),
                "train_loss": round(loss, 5), "train_acc": round(acc, 4), "train_ms": int(1000 * (time.time() - t0))}
            if secure:
                d = m.to_dict(spec)
                values = secure_agg.mask_update(d["W"] + d["b"], len(y), self.key,
                                                secure_agg.mask_label(g["run_id"], g["version"], g["attempt"]),
                                                secure_agg.adds_mask(self.cid, self.other))
                update.update(masked_model={"n_features": d["n_features"], "n_classes": d["n_classes"],
                                            "feature_set": spec, "values": values},
                              attempt=g["attempt"], key_epoch=self.key_epoch)
            else:
                update["model"] = m.to_dict(spec)
            st, r = call(self.base, "POST", "/update", update)
            if st == 409:
                self.stale_count += 1  # the round moved on without us: download the newest model and continue

    def evaluate(self, g, local, Xt, yt, spec):
        glob = SoftmaxModel.from_dict(g["model"])
        other = "B" if self.cid == "A" else "A"
        lm = None
        other_model = None
        for _ in range(100):
            st, lm = call(self.base, "GET", f"/local_model?client_id={other}")
            if st == 200 and "model" in lm:
                other_model = lm["model"]
            elif st == 200 and self.key is not None:   # secure aggregation: the relayed model is encrypted
                text = secure_agg.decrypt(self.key, f"local|{g['run_id']}|{other}", lm["encrypted_model"])
                other_model = json.loads(text) if text else None
            if other_model:
                break
            time.sleep(0.05)
        models = {f"local_{self.cid}": local, "global": glob}
        if other_model:
            models[f"local_{other}"] = SoftmaxModel.from_dict(other_model)
        self.result = {}
        for name, mdl in models.items():
            s = summary(yt, mdl.predict(Xt))
            s["n"] = int(len(yt))
            self.result[name] = s
            call(self.base, "POST", "/metrics", {"client_id": self.cid, "model_name": name, "feature_set": spec,
                                                 "test_set": "test", "metrics": s})
        self.results_by_run[g["run_id"]] = self.result
        call(self.base, "POST", "/done", {"client_id": self.cid, "run_id": g["run_id"]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://localhost:8000")
    ap.add_argument("--password", default="", help="the server password, if it has one")
    ap.add_argument("--follow", action="store_true", help="keep training when the server starts new runs (sweeps); Ctrl-C to stop")
    args = ap.parse_args()
    global PASSWORD
    PASSWORD = args.password
    phones = [SimPhone(args.server, "A", 1, follow_runs=args.follow), SimPhone(args.server, "B", 2, follow_runs=args.follow)]
    for p in phones:
        p.start()
    for p in phones:
        p.join()
    for p in phones:
        print(f"Phone {p.cid} six-pose test set (synthetic):")
        for name in sorted(p.result):
            s = p.result[name]
            print(f"   {name:<9} accuracy {s['accuracy']:.3f}  macro-F1 {s['macro_f1']:.3f}  recall "
                  + " ".join(f"{v:.2f}" for v in s["per_class_recall"]))


if __name__ == "__main__":
    main()
