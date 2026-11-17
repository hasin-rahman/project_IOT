"""Offline simulation of the whole experiment on SYNTHETIC data.

Purpose: check the pipeline and the non-IID FedAvg behaviour before phone data exists,
and serve as a reference the phone/server results can be compared with.
It is NOT a source of results for the report.

    python3 simulate.py                      # baseline feature set, 3 seeds
    python3 simulate.py --features acc+grav+gyro+rot --rounds 20
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from posefl_ref import features as F  # noqa: E402
from posefl_ref.metrics import summary  # noqa: E402
from posefl_ref.model import SoftmaxModel  # noqa: E402
from posefl_ref.poses import CLIENT_TRAIN_POSES, LABELS, N_CLASSES  # noqa: E402
from posefl_ref.synth import Person, make_session  # noqa: E402


def build_dataset(poses, people, rng, spec, sessions_per_pose=3, duration_s=20.0):
    X, y = [], []
    for person in people:
        for p in poses:
            for _ in range(sessions_per_pose):
                holding = rng.choice(["hand", "surface"])
                rows = make_session(p, person, rng, duration_s=duration_s, holding=holding)
                f = F.extract(rows, spec)
                X.append(f)
                y.append(np.full(len(f), p))
    return np.concatenate(X), np.concatenate(y)


def fedavg(models, counts):
    total = float(sum(counts))
    out = models[0].copy()
    out.W = sum(m.W * (c / total) for m, c in zip(models, counts))
    out.b = sum(m.b * (c / total) for m, c in zip(models, counts))
    return out


def run(seed, spec, rounds, local_epochs, baseline_epochs, lr, batch, l2):
    rng = np.random.default_rng(seed)
    person_a, person_b, person_new = Person(rng), Person(rng), Person(rng, max_tilt_deg=30, tremor=0.25)
    Xa, ya = build_dataset(CLIENT_TRAIN_POSES["A"], [person_a], rng, spec)
    Xb, yb = build_dataset(CLIENT_TRAIN_POSES["B"], [person_b], rng, spec)
    # six-pose test sets: one from the training phones' users, one from an unseen user
    Xt, yt = build_dataset(range(N_CLASSES), [person_a, person_b], rng, spec, sessions_per_pose=1)
    Xu, yu = build_dataset(range(N_CLASSES), [person_new], rng, spec, sessions_per_pose=1)

    nf = F.feature_size(spec)
    local_a, local_b = SoftmaxModel(nf), SoftmaxModel(nf)
    local_a.train(Xa, ya, epochs=baseline_epochs, lr=lr, batch_size=batch, l2=l2, rng=rng)
    local_b.train(Xb, yb, epochs=baseline_epochs, lr=lr, batch_size=batch, l2=l2, rng=rng)

    glob = SoftmaxModel(nf)
    curve = []
    for _ in range(rounds):
        ca, cb = glob.copy(), glob.copy()
        ca.train(Xa, ya, epochs=local_epochs, lr=lr, batch_size=batch, l2=l2, rng=rng)
        cb.train(Xb, yb, epochs=local_epochs, lr=lr, batch_size=batch, l2=l2, rng=rng)
        glob = fedavg([ca, cb], [len(ya), len(yb)])
        curve.append(summary(yt, glob.predict(Xt))["accuracy"])

    res = {}
    for name, m in [("Client A local", local_a), ("Client B local", local_b), ("Federated global", glob)]:
        res[name] = {"seen_users": summary(yt, m.predict(Xt)), "unseen_user": summary(yu, m.predict(Xu))}
    return res, curve, (len(ya), len(yb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=F.BASELINE)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--local-epochs", type=int, default=1)
    ap.add_argument("--baseline-epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--l2", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    print(f"SYNTHETIC DATA - feature set '{args.features}' ({F.feature_size(args.features)} features), "
          f"{args.rounds} rounds x {args.local_epochs} local epoch(s), seeds {args.seeds}\n")
    all_res, curves = [], []
    for s in args.seeds:
        res, curve, counts = run(s, args.features, args.rounds, args.local_epochs,
                                 args.baseline_epochs, args.lr, args.batch, args.l2)
        all_res.append(res)
        curves.append(curve)
    print(f"training windows per client: A={counts[0]}, B={counts[1]}\n")

    names = list(all_res[0].keys())
    for split in ["seen_users", "unseen_user"]:
        print(f"== Test set: {split.replace('_', ' ')} (mean +/- std over seeds) ==")
        print(f"{'model':<18} {'accuracy':>16} {'macro-F1':>16}   per-class recall")
        for n in names:
            acc = [r[n][split]["accuracy"] for r in all_res]
            f1 = [r[n][split]["macro_f1"] for r in all_res]
            rec = np.mean([r[n][split]["per_class_recall"] for r in all_res], axis=0)
            print(f"{n:<18} {np.mean(acc):>8.3f} +/- {np.std(acc):.3f} {np.mean(f1):>8.3f} +/- {np.std(f1):.3f}   "
                  + " ".join(f"{v:.2f}" for v in rec))
        print()
    print("classes:", ", ".join(LABELS))
    c = np.mean(curves, axis=0)
    print("global accuracy by round:", " ".join(f"{v:.2f}" for v in c))


if __name__ == "__main__":
    main()
