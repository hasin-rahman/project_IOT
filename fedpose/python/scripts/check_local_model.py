"""Quick data-quality check: train a local-only model for one client and score it.

  python3 scripts/check_local_model.py --data ../data --client B

Reads <data>/raw/<client>/{train,val,test,newuser}. Prints accuracy on this client's own poses and on
all six poses, plus per-pose recall. For client B, expect high accuracy on poses 3-5 and about 0 on
poses 0-2 (a local model cannot predict poses it never saw: that is the point of the project).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fedpose.constants import CLIENT_POSES, POSES, n_features  # noqa: E402
from fedpose.data import load_client  # noqa: E402
from fedpose.fl_sim import SimConfig, train_local_only  # noqa: E402
from fedpose.model import predict  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--client", default="B", choices=sorted(CLIENT_POSES))
    ap.add_argument("--features", default="acc_grav")
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.3)
    a = ap.parse_args()

    d = load_client(a.data, a.client, a.features)
    nf = n_features(a.features)
    X, y = d.train
    print(f"client {a.client}: {len(y)} train windows, per pose {np.bincount(y, minlength=6).tolist()}")
    cfg = SimConfig(rounds=a.rounds, local_epochs=a.epochs, lr=a.lr)
    w = train_local_only(X, y, nf, cfg, a.client)
    own = list(CLIENT_POSES[a.client])

    for name in ("val", "test", "newuser"):
        Xs, ys = getattr(d, name)
        if not len(ys):
            print(f"{name:8s} no data"); continue
        p = predict(w, Xs, nf)
        m = np.isin(ys, own)
        short = ["SU", "SD", "PU", "PD", "LL", "LR"]
        rec = [f"{short[k]} {100 * np.mean(p[ys == k] == k):3.0f}%" for k in range(6) if (ys == k).any()]
        print(f"{name:8s} n={len(ys):4d}  own poses {100 * np.mean(p[m] == ys[m]):5.1f}%   all six {100 * np.mean(p == ys):5.1f}%")
        print("         recall:", " | ".join(rec))


if __name__ == "__main__":
    main()
