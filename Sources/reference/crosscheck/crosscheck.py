"""Check that the Kotlin phone code computes exactly what the Python reference computes.

    python3 crosscheck.py make  <dir>                 # write pose_0..5.csv synthetic sessions
    python3 crosscheck.py compare <dir> <kotlin_output.txt>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from posefl_ref import features as F  # noqa: E402
from posefl_ref.model import SoftmaxModel  # noqa: E402
from posefl_ref.synth import Person, make_session  # noqa: E402

SPEC = "acc+grav+gyro+rot"


def make(d):
    os.makedirs(d, exist_ok=True)
    rng = np.random.default_rng(42)
    person = Person(rng)
    for i in range(6):
        rows = make_session(i, person, rng, duration_s=8.0, holding="hand")
        with open(os.path.join(d, f"pose_{i}.csv"), "w") as fh:
            fh.write(F.HEADER + "\n")
            for r in rows:
                fh.write(f"{r[0]:.1f}," + ",".join(f"{v:.6f}" for v in r[1:]) + "\n")


def compare(d, kt_path):
    X, y = [], []
    for i in range(6):
        f = F.extract(F.read_csv(os.path.join(d, f"pose_{i}.csv")), SPEC)
        X.append(f)
        y += [i] * len(f)
    X = np.concatenate(X)
    y = np.array(y)
    m = SoftmaxModel(X.shape[1])
    m.train(X, y, epochs=3, lr=0.1, batch_size=16, l2=1e-4, shuffle=False)
    loss, acc = m.evaluate(X, y)

    kt = {"FEAT": [], "GUESS": []}
    for line in open(kt_path):
        tag, *vals = line.split()
        if tag in ("FEAT", "GUESS"):
            kt[tag].append([float(v) for v in vals])
        else:
            kt[tag] = [float(v) for v in vals]
    kfeat = np.array(kt["FEAT"])
    ok = True

    def check(name, a, b, tol):
        nonlocal ok
        err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        good = err < tol
        ok &= good
        print(f"{'PASS' if good else 'FAIL'}  {name:<28} max abs diff = {err:.2e}")

    print(f"windows: python={len(X)} kotlin={len(kfeat)}")
    ok &= len(X) == len(kfeat)
    check("features (24 per window)", X, kfeat, 1e-9)
    check("trained W", m.W.reshape(-1), kt["W"], 1e-9)
    check("trained b", m.b, kt["B"], 1e-9)
    check("loss / accuracy", [loss, acc], kt["EVAL"], 1e-9)
    guesses = [int(g[1]) for g in kt["GUESS"]]
    print(f"{'PASS' if guesses == list(range(6)) else 'FAIL'}  rule-based pose check       {guesses}")
    ok &= guesses == list(range(6))
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return ok


if __name__ == "__main__":
    if sys.argv[1] == "make":
        make(sys.argv[2])
    else:
        sys.exit(0 if compare(sys.argv[2], sys.argv[3]) else 1)
