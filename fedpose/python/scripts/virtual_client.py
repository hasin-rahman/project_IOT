"""Stand-in phone: a Python client that talks to the real FL server over HTTP.

Use it to test your own phone (Client B) without a second phone, using the synthetic data folder.
It plays the same protocol as the Android app: register, poll /model, train on local data, upload a delta.

  python3 scripts/virtual_client.py --server http://127.0.0.1:8080 --client A --data ../data_synth

Raw data never leaves this script: only the weight delta, counts and metrics are sent.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedpose.constants import CLIENT_POSES, n_features  # noqa: E402
from fedpose.data import load_client  # noqa: E402
from fedpose.fl_sim import client_seed  # noqa: E402
from fedpose.model import train_local  # noqa: E402
from fedpose.wire import b64_to_f32, update_body  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--client", required=True, choices=sorted(CLIENT_POSES))
    ap.add_argument("--data", required=True, help="folder that contains raw/<client>/<split>/")
    ap.add_argument("--features", default="acc_grav")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--codec", default="f32", choices=["f32", "int8"])
    ap.add_argument("--poll", type=float, default=0.5)
    a = ap.parse_args()

    cid, base = a.client, a.server.rstrip("/")
    nf = n_features(a.features)
    X, y = load_client(a.data, cid, a.features).train
    counts = np.bincount(y, minlength=6).tolist()
    print(f"[{cid}] {len(y)} training windows, class counts {counts}")

    reg = requests.post(f"{base}/register", json={"client_id": cid}, timeout=10).json()
    if int(reg.get("n_features", nf)) != nf:
        sys.exit(f"[{cid}] server expects {reg['n_features']} features, this client has {nf}")
    print(f"[{cid}] registered, server max_rounds={reg.get('max_rounds')}")

    done_rounds = set()
    while True:
        try:
            m = requests.get(f"{base}/model", params={"client_id": cid}, timeout=10).json()
        except requests.RequestException as e:
            print(f"[{cid}] server not reachable ({e}), retrying"); time.sleep(1); continue
        if m["status"] == "done":
            print(f"[{cid}] finished at round {m['round']}"); return
        if m["status"] != "training" or m.get("already_submitted") or m["round"] in done_rounds:
            time.sleep(a.poll); continue
        w0 = b64_to_f32(m["weights"])
        w = train_local(w0, X, y, nf, epochs=a.epochs, lr=a.lr, batch_size=a.batch, w_ref=w0,
                        seed=client_seed(a.seed, m["round"], cid))
        body = update_body(cid, m["round"], m["version"], len(y), w - w0, a.codec,
                           extra={"class_counts": counts})
        r = requests.post(f"{base}/update", json=body, timeout=10)
        if r.status_code == 200:
            done_rounds.add(m["round"]); print(f"[{cid}] round {m['round']} uploaded")
        elif r.status_code == 409:
            print(f"[{cid}] round {m['round']} was stale, resyncing")
        else:
            print(f"[{cid}] upload rejected: {r.status_code} {r.text}")
            time.sleep(1)


if __name__ == "__main__":
    main()
