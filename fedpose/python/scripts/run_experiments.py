#!/usr/bin/env python3
"""Controlled experiments for the report (offline, on data pulled from the team's own phones).

    python scripts/run_experiments.py --data ../data_synth --out ../results_synth all

Commands
  tune      pick lr / local epochs / rounds on VALIDATION data only (global model macro-F1)
  baseline  Client A local vs Client B local vs federated global, 3 seeds, test + unseen user
  sensor    required technical extension: acc | acc+grav | +gyro | +rotation vector
  extra     extra credit: int8 compression, adaptive participation, dropout recovery, all combined
  all       tune, baseline, sensor, extra (in that order; later steps reuse chosen_config.json)

The seeds change the minibatch order (and the optional random init); the data is fixed. Spread over
3 seeds therefore understates real-world variance. The unseen-user test and per-session splits are
the better indicators of generalization.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.constants import FEATURE_SETS, n_features  # noqa: E402
from fedpose.data import load_all, pooled  # noqa: E402
from fedpose.fl_sim import SimConfig, simulate, train_local_only  # noqa: E402
from fedpose.metrics import evaluate  # noqa: E402
from fedpose.model import f32_round, predict  # noqa: E402

SEEDS = (0, 1, 2)


def ms(xs):
    xs = np.asarray(xs, dtype=float)
    return {"mean": float(xs.mean()), "std": float(xs.std(ddof=1)) if len(xs) > 1 else 0.0, "values": xs.tolist()}


def save(out, name, obj):
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, name), "w") as f:
        json.dump(obj, f, indent=1)
    print(f"  wrote {os.path.join(out, name)}")


def setup(data, fs):
    cd = load_all(data, fs)
    cl = {c: d.train for c, d in cd.items()}
    ev = {"val": pooled(cd, "val"), "test": pooled(cd, "test"), "newuser": pooled(cd, "newuser")}
    return cd, cl, ev


def load_cfg(out) -> dict:
    p = os.path.join(out, "chosen_config.json")
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing: run the 'tune' command first")
    return json.load(open(p))


def cfg_from(chosen, seed, **kw) -> SimConfig:
    return SimConfig(rounds=chosen["rounds"], local_epochs=chosen["local_epochs"], lr=chosen["lr"],
                     batch_size=chosen.get("batch_size", 32), seed=seed, **kw)


# ------------------------------------------------------------------------------ tune
def cmd_tune(a):
    fs = "acc_grav"
    _, cl, ev = setup(a.data, fs)
    nf = n_features(fs)
    checkpoints = (10, 20, 40)
    rows = []
    for lr, E in itertools.product((0.1, 0.3, 1.0), (1, 2, 5)):
        f1s = {r: [] for r in checkpoints}
        for seed in SEEDS:
            res = simulate(cl, nf, SimConfig(rounds=max(checkpoints), local_epochs=E, lr=lr, seed=seed), {"val": ev["val"]})
            for r in checkpoints:
                f1s[r].append(res["history"][r - 1]["eval"]["val"]["macro_f1"])
        for r in checkpoints:
            rows.append({"lr": lr, "local_epochs": E, "rounds": r, "val_macro_f1": float(np.mean(f1s[r]))})
    # best validation macro-F1; ties (within 0.002) go to the cheaper setting (fewer total epochs)
    best_f1 = max(r["val_macro_f1"] for r in rows)
    cands = [r for r in rows if r["val_macro_f1"] >= best_f1 - 0.002]
    best = min(cands, key=lambda r: (r["rounds"] * r["local_epochs"], r["lr"]))
    chosen = {"lr": best["lr"], "local_epochs": best["local_epochs"], "rounds": best["rounds"], "batch_size": 32,
              "val_macro_f1": best["val_macro_f1"], "selected_on": "validation split only"}
    print("  chosen:", chosen)
    save(a.out, "tuning_grid.json", rows)
    save(a.out, "chosen_config.json", chosen)


# ----------------------------------------------------------------------- baseline/sensor
def three_models(cl, nf, ev, cfg):
    res = simulate(cl, nf, cfg, ev)
    models = {"local_A": train_local_only(*cl["A"], nf, cfg, "A"),
              "local_B": train_local_only(*cl["B"], nf, cfg, "B"),
              "global": f32_round(res["weights"])}
    out = {}
    for name, w in models.items():
        out[name] = {k: evaluate(ev[k][1], predict(w, ev[k][0], nf)) for k in ("val", "test", "newuser")}
    hist = [{"round": h["round"], "bytes_up": h["bytes_up"], "bytes_down": h["bytes_down"],
             **{f"{k}_{m}": h["eval"][k][m] for k in ("test", "newuser", "val") for m in ("accuracy", "macro_f1")}}
            for h in res["history"]]
    return {"models": out, "history": hist, "bytes_up": res["bytes_up"], "bytes_down": res["bytes_down"]}


def cmd_baseline(a, fs="acc_grav", name="baseline.json"):
    chosen = load_cfg(a.out)
    _, cl, ev = setup(a.data, fs)
    nf = n_features(fs)
    runs = []
    for seed in SEEDS:
        r = three_models(cl, nf, ev, cfg_from(chosen, seed))
        r["seed"] = seed
        runs.append(r)
    save(a.out, name, {"feature_set": fs, "config": chosen, "seeds": list(SEEDS), "runs": runs})
    for m in ("local_A", "local_B", "global"):
        for k in ("test", "newuser"):
            f1 = ms([r["models"][m][k]["macro_f1"] for r in runs])
            ac = ms([r["models"][m][k]["accuracy"] for r in runs])
            print(f"  {m:8s} {k:8s} acc {ac['mean']:.3f}±{ac['std']:.3f}  macroF1 {f1['mean']:.3f}±{f1['std']:.3f}")


def cmd_sensor(a):
    chosen = load_cfg(a.out)
    res = {}
    for fs in FEATURE_SETS:
        _, cl, ev = setup(a.data, fs)
        nf = n_features(fs)
        runs = [three_models(cl, nf, ev, cfg_from(chosen, s)) for s in SEEDS]
        res[fs] = {"n_features": nf, "n_params": 6 * nf + 6, "runs": [
            {"seed": s, "models": r["models"]} for s, r in zip(SEEDS, runs)]}
        g = ms([r["models"]["global"]["test"]["macro_f1"] for r in runs])
        gn = ms([r["models"]["global"]["newuser"]["macro_f1"] for r in runs])
        print(f"  {fs:20s} nf={nf:2d} global macroF1 test {g['mean']:.3f}±{g['std']:.3f} newuser {gn['mean']:.3f}±{gn['std']:.3f}")
    save(a.out, "sensor_study.json", {"config": chosen, "seeds": list(SEEDS), "results": res})


# ------------------------------------------------------------------------------ extra
def recovery_rounds(history, start, end, key="val_macro_f1"):
    """Rounds after the outage until macro-F1 is back within 0.01 of the pre-outage level."""
    pre = history[start - 2][key] if start >= 2 else 0.0
    for h in history:
        if h["round"] > end and h[key] >= pre - 0.01:
            return h["round"] - end
    return None


def run_variant(cl, nf, ev, chosen, seed, variant, scenario, tau, drop):
    kw = {}
    if variant in ("int8", "all"):
        kw["codec"] = "int8"
    if variant in ("adaptive", "all"):
        kw.update(adaptive=True, skip_threshold=tau)
    if scenario == "dropout":
        kw["dropout"] = {"B": drop}
        kw["stale_policy"] = "reject" if variant in ("recovery", "all") else "accept"
    cfg = cfg_from(chosen, seed, **kw)
    res = simulate(cl, nf, cfg, ev)
    w = f32_round(res["weights"])
    hist = [{"round": h["round"], "bytes_up": h["bytes_up"],
             **{f"{k}_{m}": h["eval"][k][m] for k in ("test", "newuser", "val") for m in ("accuracy", "macro_f1")}}
            for h in res["history"]]
    final = {k: evaluate(ev[k][1], predict(w, ev[k][0], nf)) for k in ("val", "test", "newuser")}
    first = lambda key, frac: next((h["round"] for h in hist if h[key] >= frac * hist[-1][key]), None)
    return {"seed": seed, "final": final, "bytes_up": res["bytes_up"], "bytes_down": res["bytes_down"],
            "skipped_updates": res["skipped_updates"], "stale_rejected": res["stale_rejected"],
            "local_epochs_run": res["local_epochs_run"],
            "rounds_to_95pct_of_final_val_f1": first("val_macro_f1", 0.95),
            "recovery_rounds": recovery_rounds(hist, *drop) if scenario == "dropout" else None,
            "completed_rounds_with_both": sum(1 for h in res["history"] if len(h["participants"]) == 2),
            "history": hist}


def cmd_extra(a):
    chosen = load_cfg(a.out)
    fs = "acc_grav"
    _, cl, ev = setup(a.data, fs)
    nf = n_features(fs)
    # 1) adaptive threshold chosen on VALIDATION: largest tau whose val macro-F1 is within 0.01 of FedAvg
    base_val = np.mean([run_variant(cl, nf, ev, chosen, s, "fedavg", "stable", 0, None)["final"]["val"]["macro_f1"] for s in SEEDS])
    sweep = []
    for tau in (0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2):
        rs = [run_variant(cl, nf, ev, chosen, s, "adaptive", "stable", tau, None) for s in SEEDS]
        sweep.append({"tau": tau, "val_macro_f1": float(np.mean([r["final"]["val"]["macro_f1"] for r in rs])),
                      "bytes_up": float(np.mean([r["bytes_up"] for r in rs])),
                      "skipped": float(np.mean([r["skipped_updates"] for r in rs]))})
    ok = [s for s in sweep if s["val_macro_f1"] >= base_val - 0.01 and s["skipped"] > 0]
    tau = max(ok, key=lambda s: s["tau"])["tau"] if ok else 0.0
    print(f"  FedAvg val macro-F1 {base_val:.3f}; chosen tau = {tau} "
          f"({'no skipping is safe on this data' if tau == 0 else 'largest within 0.01 of baseline'})")
    save(a.out, "adaptive_threshold_sweep.json", {"baseline_val_macro_f1": float(base_val), "chosen_tau": tau, "sweep": sweep})
    # 2) variants x scenarios x seeds
    R = chosen["rounds"]
    drop = (max(2, R // 4), max(2, R // 4) + max(2, R // 5) - 1)
    out = {"config": chosen, "tau": tau, "dropout_rounds_for_B": list(drop), "results": {}}
    for scenario in ("stable", "dropout"):
        for variant in ("fedavg", "int8", "adaptive", "recovery", "all"):
            runs = [run_variant(cl, nf, ev, chosen, s, variant, scenario, tau, drop) for s in SEEDS]
            out["results"][f"{scenario}/{variant}"] = runs
            f1 = ms([r["final"]["test"]["macro_f1"] for r in runs])
            up = ms([r["bytes_up"] for r in runs])
            print(f"  {scenario:8s} {variant:9s} test macroF1 {f1['mean']:.3f}±{f1['std']:.3f}  uplink {up['mean']:.0f} B  "
                  f"skipped {np.mean([r['skipped_updates'] for r in runs]):.1f} stale {np.mean([r['stale_rejected'] for r in runs]):.1f}")
    save(a.out, "extra_credit.json", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["tune", "baseline", "sensor", "extra", "all"])
    ap.add_argument("--data", default="../data_synth")
    ap.add_argument("--out", default="../results_synth")
    a = ap.parse_args()
    steps = {"tune": cmd_tune, "baseline": cmd_baseline, "sensor": cmd_sensor, "extra": cmd_extra}
    for name in (steps if a.command == "all" else [a.command]):
        print(f"== {name}")
        steps[name](a)


if __name__ == "__main__":
    main()
