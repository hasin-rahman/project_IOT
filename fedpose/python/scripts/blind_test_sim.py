#!/usr/bin/env python3
"""Estimate your demo-day Part A score (blind test, 40 points) from your own recorded sessions.

Mimics the judges: 36 placements (6 per pose), 2 s clips, one label per clip, all three models.
Score = 30 x global accuracy + 5 x local A accuracy on its 3 poses + 5 x local B accuracy on its 3 poses.

    python scripts/blind_test_sim.py --data ../data --out ../results --split newuser

Use --split newuser for the honest estimate (people who gave no training data), --split val to
choose the window offsets, and look at test only at the end. Compare offsets with --starts.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.blind import CLIP_STARTS_S, clip_label, clip_windows, make_clips, rubric_score  # noqa: E402
from fedpose.constants import CLIENT_POSES, FEATURE_SETS, n_features  # noqa: E402
from fedpose.data import load_all, session_files  # noqa: E402
from fedpose.features import load_session_csv  # noqa: E402
from fedpose.fl_sim import SimConfig, simulate, train_local_only  # noqa: E402
from fedpose.model import f32_round  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../data_synth")
ap.add_argument("--out", default="../results_synth")
ap.add_argument("--split", default="newuser", choices=["val", "test", "newuser"])
ap.add_argument("--features", default="acc_grav", choices=list(FEATURE_SETS))
ap.add_argument("--starts", default=None, help="comma list of window starts in s, e.g. 0.5,1.0 or 0,0.5,1.0")
ap.add_argument("--trim", type=float, default=1.0, help="seconds cut from both ends of a session before clips are drawn; use 0 for a pessimistic run that includes hand-settling motion")
ap.add_argument("--trials", type=int, default=2000, help="random 36-placement draws")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

chosen = json.load(open(os.path.join(a.out, "chosen_config.json")))
fs, nf = a.features, n_features(a.features)
cd = load_all(a.data, fs)
cl = {c: d.train for c, d in cd.items()}
cfg = SimConfig(rounds=chosen["rounds"], local_epochs=chosen["local_epochs"], lr=chosen["lr"], seed=a.seed)
models = {"global": f32_round(simulate(cl, nf, cfg)["weights"]),
          "local_A": train_local_only(*cl["A"], nf, cfg, "A"),
          "local_B": train_local_only(*cl["B"], nf, cfg, "B")}

rng = np.random.default_rng(a.seed)
variants = [tuple(float(x) for x in a.starts.split(","))] if a.starts else [(0.0, 0.5, 1.0), (0.5, 1.0), (1.0,)]
for starts in variants:
    pool = {p: [] for p in range(6)}  # pose -> list of clips (window matrices)
    for client in CLIENT_POSES:
        for path, pose, _u in session_files(a.data, client, a.split):
            pool[pose] += make_clips(load_session_csv(path), fs, rng, 25, trim_s=a.trim, starts_s=starts)
    if any(len(v) == 0 for v in pool.values()):
        print(f"split '{a.split}' has no usable clips for some pose; record more sessions"); sys.exit(1)
    ok = {m: {p: np.array([clip_label(w, c, nf) == p for c in pool[p]]) for p in range(6)} for m, w in models.items()}
    scores = []
    for _ in range(a.trials):
        c_g = c_a = c_b = 0
        for p in range(6):
            pick = rng.integers(0, len(pool[p]), 6)
            c_g += int(ok["global"][p][pick].sum())
            if p in CLIENT_POSES["A"]:
                c_a += int(ok["local_A"][p][pick].sum())
            else:
                c_b += int(ok["local_B"][p][pick].sum())
        scores.append(rubric_score(c_g, 36, c_a, 18, c_b, 18))
    pa = np.array([s["part_A"] for s in scores])
    ga = np.mean([s["global_acc"] for s in scores])
    print(f"window starts {starts}: global acc {ga:.3f} | Part A mean {pa.mean():.1f}/40, 5th percentile {np.percentile(pa, 5):.1f}, "
          f"min {pa.min():.1f}  (split={a.split}, features={fs}, trim={a.trim}s, clips/pose={[len(pool[p]) for p in range(6)]})")
    per_pose = {p: float(ok['global'][p].mean()) for p in range(6)}
    print("   global accuracy per pose:", {k: round(v, 3) for k, v in per_pose.items()})
    for m, own in (("local_A", CLIENT_POSES["A"]), ("local_B", CLIENT_POSES["B"])):
        other = [p for p in range(6) if p not in own]
        print(f"   {m}: own poses {np.mean([ok[m][p].mean() for p in own]):.3f}, other client's poses {np.mean([ok[m][p].mean() for p in other]):.3f} (rubric expects ~0)")
