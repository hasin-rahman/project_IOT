#!/usr/bin/env python3
"""Figures and tables for the report, from the JSON written by run_experiments.py.

    python scripts/make_plots.py --results ../results_synth --out ../figures_synth
    python scripts/make_plots.py --results ../results --out ../figures --reports ../python/server_data/reports.jsonl

--reports adds the LIVE system evaluation: the phones score all three models on their own test sets
and report only metrics; the two phones' confusion matrices are summed here.

Colors: validated categorical slots (blue, orange, aqua, yellow, magenta) assigned by entity in a
fixed order, blue one-hue sequential ramp for confusion matrices. Every figure has direct labels,
and every number is also written to results_tables.md / CSV (contrast relief for light slots).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.constants import POSES  # noqa: E402
from fedpose.metrics import summarize  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # slots 1-5, fixed order
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
MODEL_COLOR = {"local_A": CAT[0], "local_B": CAT[1], "global": CAT[2]}
MODEL_LABEL = {"local_A": "Client A local", "local_B": "Client B local", "global": "Federated global"}
POSE_SHORT = ["screen up", "screen down", "portrait up", "portrait down", "landscape left", "landscape right"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 130,
})


def ms(v):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def load(res, name):
    p = os.path.join(res, name)
    return json.load(open(p)) if os.path.exists(p) else None


def finish(fig, out, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight")
    plt.close(fig)
    print("  wrote", os.path.join(out, name))


def fig_rounds(base, out):
    runs = base["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, key, title in zip(axes, ("test", "newuser"), ("Same users, held-out sessions", "Unseen user")):
        h = np.array([[r_["%s_macro_f1" % key] for r_ in r["history"]] for r in runs])
        x = np.arange(1, h.shape[1] + 1)
        m, s = h.mean(0), h.std(0, ddof=1) if len(runs) > 1 else np.zeros(h.shape[1])
        ax.fill_between(x, m - s, m + s, color=MODEL_COLOR["global"], alpha=0.18, linewidth=0)
        ax.plot(x, m, color=MODEL_COLOR["global"], lw=2, marker="o", ms=4, label=MODEL_LABEL["global"])
        for name in ("local_A", "local_B"):
            v = np.mean([r["models"][name][key]["macro_f1"] for r in runs])
            ax.axhline(v, color=MODEL_COLOR[name], lw=2, ls="--", label=MODEL_LABEL[name])
        ax.set_title(title)
        ax.set_xlabel("Federated round")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Macro-F1, all six poses")
    axes[0].legend(loc="center right", fontsize=8)
    fig.suptitle("Global model vs independent local baselines (mean ± std, 3 seeds)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold")
    finish(fig, out, "fig_accuracy_by_round.png")


def fig_confusion(base, out, key="test", name="fig_confusion_matrices.png"):
    runs = base["runs"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for ax, m in zip(axes, ("local_A", "local_B", "global")):
        cm = sum(np.array(r["models"][m][key]["confusion"]) for r in runs)
        norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(6):
            for j in range(6):
                ax.text(j, i, f"{norm[i, j]*100:.0f}", ha="center", va="center", fontsize=8,
                        color="white" if norm[i, j] > 0.5 else INK)
        s = summarize(cm)
        ax.set_title(f"{MODEL_LABEL[m]}  (acc {s['accuracy']:.2f}, F1 {s['macro_f1']:.2f})", fontsize=9.5)
        ax.set_xticks(range(6), POSE_SHORT, rotation=45, ha="right")
        ax.set_yticks(range(6), POSE_SHORT if m == "local_A" else [])
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True pose")
    fig.subplots_adjust(wspace=0.08)
    fig.suptitle(f"Confusion matrices, % of each true pose ({'unseen user' if key == 'newuser' else 'held-out sessions'}, summed over seeds)",
                 x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold")
    finish(fig, out, name)


def fig_per_class(base, out):
    runs = base["runs"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    w = 0.26
    for k, m in enumerate(("local_A", "local_B", "global")):
        rec = np.array([r["models"][m]["test"]["per_class_recall"] for r in runs])
        mean, std = rec.mean(0), (rec.std(0, ddof=1) if len(runs) > 1 else 0 * rec[0])
        x = np.arange(6) + (k - 1) * (w + 0.02)
        ax.bar(x, mean, w, yerr=std, color=MODEL_COLOR[m], label=MODEL_LABEL[m], error_kw={"lw": 1, "ecolor": INK2})
    ax.set_xticks(range(6), POSE_SHORT)
    ax.set_ylabel("Per-class accuracy (recall)")
    ax.set_ylim(0, 1.12)
    ax.set_title("Per-pose accuracy: each local model fails on the poses it never saw")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    finish(fig, out, "fig_per_class.png")


def fig_sensor(sensor, out):
    res = sensor["results"]
    names = list(res)
    nice = {"acc": "acc", "acc_grav": "acc+grav\n(baseline)", "acc_grav_gyro": "+gyro", "acc_grav_gyro_rot": "+gyro\n+rotation"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, title in zip(axes, ("test", "newuser"), ("Same users, held-out sessions", "Unseen user")):
        for i, fs in enumerate(names):
            f1 = [r["models"]["global"][key]["macro_f1"] for r in res[fs]["runs"]]
            m, s = ms(f1)
            ax.bar(i, m, 0.62, yerr=s, color=CAT[i], error_kw={"lw": 1, "ecolor": INK2})
            ax.text(i, min(m + s, 1.0) + 0.02, f"{m:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(names)), [nice.get(n, n) for n in names])
        ax.set_title(title)
        ax.set_ylim(0, 1.12)
    axes[0].set_ylabel("Global model macro-F1")
    fig.suptitle("Sensor study: which feature set generalizes? (mean ± std, 3 seeds, all else fixed)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold")
    finish(fig, out, "fig_sensor_study.png")


VARIANTS = ["fedavg", "int8", "adaptive", "recovery", "all"]
VLABEL = {"fedavg": "FedAvg\nbase", "int8": "int8\ncomp.", "adaptive": "adaptive\npartic.",
          "recovery": "dropout\nrecov.", "all": "all\nthree"}


def fig_extra(extra, out):
    r = extra["results"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.8), sharey=True)
    for ax, sc in zip(axes, ("stable", "dropout")):
        for i, v in enumerate(VARIANTS):
            runs = r[f"{sc}/{v}"]
            up = np.mean([x["bytes_up"] for x in runs]) / 1000
            f1 = np.mean([x["final"]["test"]["macro_f1"] for x in runs])
            ax.bar(i, up, 0.62, color=CAT[i])
            ax.text(i, up + 0.15, f"{up:.1f} kB\nF1 {f1:.3f}", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(5), [VLABEL[v] for v in VARIANTS])
        ax.set_title("Connection stable" if sc == "stable" else f"Client B offline, rounds {extra['dropout_rounds_for_B'][0]}-{extra['dropout_rounds_for_B'][1]}")
    axes[0].set_ylabel("Total uplink body bytes (kB, both clients)")
    top = max(np.mean([x["bytes_up"] for x in r[k]]) for k in r) / 1000
    axes[0].set_ylim(0, top * 1.3)
    fig.suptitle("Extra credit: bytes sent vs accuracy (mean of 3 seeds)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold")
    finish(fig, out, "fig_extra_bytes.png")

    lo, hi = extra["dropout_rounds_for_B"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.axvspan(lo - 0.5, hi + 0.5, color=GRID, alpha=0.7, lw=0)
    ax.text((lo + hi) / 2, 0.02, "B offline", ha="center", fontsize=8, color=INK2)
    for i, v in enumerate(["fedavg", "recovery"]):
        h = np.array([[x["test_macro_f1"] for x in run["history"]] for run in r[f"dropout/{v}"]]).mean(0)
        ax.plot(range(1, len(h) + 1), h, color=CAT[i * 3 if i else 0], lw=2, marker="o", ms=4)
        ax.annotate("naive: accept stale update" if v == "fedavg" else "version check + resync",
                    (len(h), h[-1]), xytext=(4, (6 if v == "fedavg" else -10)), textcoords="offset points", fontsize=8, color=INK2)
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Global macro-F1 (test)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Dropout: naive aggregation vs version check + resync")
    finish(fig, out, "fig_dropout.png")


def tables(res, out, base, sensor, extra, sweep):
    L = ["# Result tables (generated by make_plots.py)\n"]
    if base:
        L += ["## Three-model comparison (mean ± std over seeds)\n", "| model | test acc | test macro-F1 | unseen-user acc | unseen-user macro-F1 |", "|---|---|---|---|---|"]
        for m in ("local_A", "local_B", "global"):
            cells = []
            for k in ("test", "newuser"):
                for met in ("accuracy", "macro_f1"):
                    a, s = ms([r["models"][m][k][met] for r in base["runs"]])
                    cells.append(f"{a:.3f} ± {s:.3f}")
            L.append(f"| {MODEL_LABEL[m]} | " + " | ".join(cells) + " |")
        L += ["\nPer-class accuracy on test (mean over seeds):\n", "| model | " + " | ".join(POSE_SHORT) + " |", "|---|" + "---|" * 6]
        for m in ("local_A", "local_B", "global"):
            rec = np.mean([r["models"][m]["test"]["per_class_recall"] for r in base["runs"]], axis=0)
            L.append(f"| {MODEL_LABEL[m]} | " + " | ".join(f"{x:.2f}" for x in rec) + " |")
    if sensor:
        L += ["\n## Sensor study (global model, macro-F1)\n", "| feature set | features | params | test | unseen user |", "|---|---|---|---|---|"]
        for fs, d in sensor["results"].items():
            t = ms([r["models"]["global"]["test"]["macro_f1"] for r in d["runs"]])
            n = ms([r["models"]["global"]["newuser"]["macro_f1"] for r in d["runs"]])
            L.append(f"| {fs} | {d['n_features']} | {d['n_params']} | {t[0]:.3f} ± {t[1]:.3f} | {n[0]:.3f} ± {n[1]:.3f} |")
    if extra:
        L += [f"\n## Extra credit (adaptive threshold tau = {extra['tau']}, chosen on validation)\n",
              "| scenario | variant | uplink B | downlink B | skipped | stale rejected | local epochs | rounds to 95% of final val F1 | recovery rounds | test macro-F1 | unseen-user macro-F1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for key, runs in extra["results"].items():
            sc, v = key.split("/")
            mean = lambda f: np.mean([f(x) for x in runs])
            rec = [x["recovery_rounds"] for x in runs if x["recovery_rounds"] is not None]
            L.append(f"| {sc} | {v} | {mean(lambda x: x['bytes_up']):.0f} | {mean(lambda x: x['bytes_down']):.0f} | "
                     f"{mean(lambda x: x['skipped_updates']):.1f} | {mean(lambda x: x['stale_rejected']):.1f} | "
                     f"{mean(lambda x: x['local_epochs_run']):.0f} | "
                     f"{mean(lambda x: x['rounds_to_95pct_of_final_val_f1'] or 0):.1f} | {(np.mean(rec) if rec else float('nan')):.1f} | "
                     f"{mean(lambda x: x['final']['test']['macro_f1']):.3f} | {mean(lambda x: x['final']['newuser']['macro_f1']):.3f} |")
        L.append("\nNote: adaptive participation saves bytes but not computation: a phone must finish local training to know whether its update is small.")
    if sweep:
        L += ["\n## Adaptive threshold sweep (validation)\n", "| tau | val macro-F1 | uplink B | skipped |", "|---|---|---|---|"]
        for s in sweep["sweep"]:
            L.append(f"| {s['tau']} | {s['val_macro_f1']:.3f} | {s['bytes_up']:.0f} | {s['skipped']:.1f} |")
    with open(os.path.join(out, "results_tables.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("  wrote", os.path.join(out, "results_tables.md"))


def phone_reports(path, out):
    """Sum the two phones' reported confusion matrices per (model, test_set) of the LATEST run."""
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows:
        print("  no phone reports found")
        return
    run = rows[-1]["run_id"]
    agg = {}
    for r in rows:
        if r["run_id"] != run:
            continue
        key = (r["model"], r["test_set"])
        agg.setdefault(key, {})[r["client_id"]] = np.array(r["confusion"])  # last report per phone wins
    L = [f"# Live system evaluation (metrics reported by the phones, run {run})\n",
         "| model | test set | acc | macro-F1 | n windows | phones |", "|---|---|---|---|---|---|"]
    mats = {}
    for (model, ts), per in sorted(agg.items()):
        cm = sum(per.values())
        s = summarize(cm)
        mats[(model, ts)] = cm
        L.append(f"| {model} | {ts} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['n']} | {','.join(sorted(per))} |")
    with open(os.path.join(out, "phone_eval.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("  wrote", os.path.join(out, "phone_eval.md"))
    for ts in sorted({k[1] for k in mats}):
        models = [m for m in ("local_A", "local_B", "global") if (m, ts) in mats]
        fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.2))
        axes = np.atleast_1d(axes)
        for ax, m in zip(axes, models):
            cm = mats[(m, ts)]
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(norm, cmap=SEQ, vmin=0, vmax=1)
            ax.grid(False)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{norm[i, j]*100:.0f}", ha="center", va="center", fontsize=8, color="white" if norm[i, j] > 0.5 else INK)
            ax.set_title(f"{MODEL_LABEL.get(m, m)} (acc {summarize(cm)['accuracy']:.2f})", fontsize=9.5)
            ax.set_xticks(range(6), POSE_SHORT, rotation=45, ha="right")
            ax.set_yticks(range(6), POSE_SHORT if m == models[0] else [])
        fig.subplots_adjust(wspace=0.08)
        fig.suptitle(f"Phone-reported confusion, {ts} (% of true pose)", x=0.01, y=1.04, ha="left", fontsize=11, fontweight="bold")
        finish(fig, out, f"fig_phone_confusion_{ts}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results_synth")
    ap.add_argument("--out", default="../figures_synth")
    ap.add_argument("--reports", default=None, help="server_data/reports.jsonl from the live phones")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    base, sensor, extra = (load(a.results, n) for n in ("baseline.json", "sensor_study.json", "extra_credit.json"))
    sweep = load(a.results, "adaptive_threshold_sweep.json")
    if base:
        fig_rounds(base, a.out); fig_confusion(base, a.out); fig_confusion(base, a.out, "newuser", "fig_confusion_newuser.png"); fig_per_class(base, a.out)
    if sensor:
        fig_sensor(sensor, a.out)
    if extra:
        fig_extra(extra, a.out)
    tables(a.results, a.out, base, sensor, extra, sweep)
    if a.reports:
        phone_reports(a.reports, a.out)


if __name__ == "__main__":
    main()
