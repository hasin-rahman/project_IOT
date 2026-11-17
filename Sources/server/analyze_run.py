"""Turn server run folders into report tables and figures.

    python3 analyze_run.py runs/run_20261120_141502                 # one run
    python3 analyze_run.py runs/run_A runs/run_B runs/run_C ...     # seeds / sensor study (grouped by feature set)
    python3 analyze_run.py --latest 12                              # the 12 most recent runs in runs/

Writes into analysis/<timestamp>/ (or --out DIR):
    summary.md              tables for the report: three-model comparison, per-class recall, sensor study,
                            dropout rounds and communication cost
    results.csv             every metric in long format (one row per run / client / test set / model)
    accuracy_by_round.png   global-model validation accuracy per round
    model_comparison.png    accuracy of the three models on each test set
    confusion_<set>.png     confusion matrices of the three models (summed over both phones)
    sensor_study.png        global accuracy per feature set (only when runs use different feature sets)

Everything here comes from what the phones reported (metrics, never sensor data).
Figures need matplotlib (pip install matplotlib); without it the tables are still written.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
POSES = ["Screen Up", "Screen Down", "Portrait Up", "Portrait Down", "Landscape Left", "Landscape Right"]
SHORT = ["SU", "SD", "PU", "PD", "LL", "LR"]
MODELS = ["local_A", "local_B", "global"]
MODEL_LABEL = {"local_A": "Client A local", "local_B": "Client B local", "global": "Federated global"}
# validated categorical palette (slots 1-3 pass colour-blind checks for all pairs)
MODEL_COLOR = {"global": "#2a78d6", "local_A": "#eb6834", "local_B": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SET_ORDER = ["val", "test", "test_unseen", "live_test"]
SET_LABEL = {"val": "validation", "test": "test", "test_unseen": "test, unseen users", "live_test": "live test (demo)"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_run(d):
    with open(os.path.join(d, "config.json")) as fh:
        cfg = json.load(fh)
    rounds = read_jsonl(os.path.join(d, "rounds.jsonl"))
    metrics = {}
    for m in read_jsonl(os.path.join(d, "metrics.jsonl")):  # keep the latest report per key
        metrics[(m["client_id"], m.get("test_set", "?"), m.get("model_name", "?"))] = m
    traffic = {}
    if os.path.exists(os.path.join(d, "traffic.json")):
        with open(os.path.join(d, "traffic.json")) as fh:
            traffic = json.load(fh)
    return {"dir": d, "id": os.path.basename(os.path.normpath(d)), "config": cfg, "rounds": rounds,
            "metrics": metrics, "traffic": traffic,
            "discarded": read_jsonl(os.path.join(d, "discarded_rounds.jsonl"))}


def mean_sd(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def fmt(m, sd=None, pct=True):
    if m is None:
        return "-"
    s = f"{100 * m:.1f}%" if pct else f"{m:.3f}"
    if sd is not None and sd >= 0.0005:
        s += f" ± {100 * sd:.1f}" if pct else f" ± {sd:.3f}"
    return s


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def collect(runs, model, test_set, key="accuracy"):
    """One value per run: the mean over the phones that reported it."""
    vals = []
    for r in runs:
        xs = [m["metrics"].get(key) for (c, s, mn), m in r["metrics"].items() if s == test_set and mn == model]
        xs = [x for x in xs if x is not None]
        if xs:
            vals.append(statistics.mean(xs))
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (server/runs/run_...)")
    ap.add_argument("--latest", type=int, help="use the N most recent runs in server/runs")
    ap.add_argument("--out", help="output folder (default analysis/<timestamp>)")
    args = ap.parse_args()
    dirs = list(args.runs)
    if args.latest:
        base = os.path.join(HERE, "runs")
        allr = sorted(d for d in os.listdir(base) if d.startswith("run_"))
        dirs += [os.path.join(base, d) for d in allr[-args.latest:]]
    if not dirs:
        ap.error("give run folders or --latest N")
    runs = [load_run(d) for d in dirs]
    runs = [r for r in runs if r["metrics"] or r["rounds"]]
    if not runs:
        sys.exit("none of these runs has rounds or metrics yet")
    out = args.out or os.path.join(HERE, "analysis", time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out, exist_ok=True)

    groups = {}
    for r in runs:
        groups.setdefault(r["config"]["feature_set"], []).append(r)
    sets = [s for s in SET_ORDER if any(k[1] == s for r in runs for k in r["metrics"])]

    md = [f"# Federated phone-pose results\n",
          f"{len(runs)} run(s): " + ", ".join(f"`{r['id']}` ({r['config']['feature_set']}, seed {r['config'].get('seed')})" for r in runs),
          "\nValues are the mean over both phones' local test sets; with several runs, mean ± standard deviation over runs (seeds).\n"]

    # ---------- 1. three-model comparison ----------
    for fs, rs in groups.items():
        md.append(f"\n## Three-model comparison — features `{fs}` ({len(rs)} run(s))\n")
        rows = []
        for model in MODELS:
            row = [MODEL_LABEL[model]]
            for s in sets:
                a = mean_sd(collect(rs, model, s))
                f = mean_sd(collect(rs, model, s, "macro_f1"))
                row.append(f"{fmt(*a)} / {fmt(*f, pct=False)}")
            rows.append(row)
        md.append(md_table(["model"] + [f"{SET_LABEL[s]} (acc / macro-F1)" for s in sets], rows))

        # per-class recall on the main test set
        main_set = "test" if "test" in sets else (sets[0] if sets else None)
        if main_set:
            md.append(f"\nPer-class recall on the {SET_LABEL[main_set]} set:\n")
            rows = []
            for model in MODELS:
                rec = []
                for i in range(6):
                    vals = [m["metrics"]["per_class_recall"][i] for r in rs for (c, s, mn), m in r["metrics"].items()
                            if s == main_set and mn == model and "per_class_recall" in m["metrics"]]
                    rec.append(fmt(statistics.mean(vals)) if vals else "-")
                rows.append([MODEL_LABEL[model]] + rec)
            md.append(md_table(["model"] + POSES, rows))

    # ---------- 2. sensor study ----------
    if len(groups) > 1:
        md.append("\n## Sensor study (global model)\n")
        rows = []
        for fs, rs in groups.items():
            nf = 6 * len(fs.split("+"))
            row = [f"`{fs}`", nf, nf * 6 + 6, len(rs)]
            for s in [x for x in ("test", "test_unseen", "live_test") if x in sets]:
                row.append(fmt(*mean_sd(collect(rs, "global", s))))
                row.append(fmt(*mean_sd(collect(rs, "global", s, "macro_f1")), pct=False))
            rows.append(row)
        hdr = ["features", "#features", "#parameters", "runs"]
        for s in [x for x in ("test", "test_unseen", "live_test") if x in sets]:
            hdr += [f"{SET_LABEL[s]} acc", f"{SET_LABEL[s]} macro-F1"]
        md.append(md_table(hdr, rows))

    # ---------- 3. training dynamics, dropout and communication ----------
    md.append("\n## Rounds, dropout and communication\n")
    rows = []
    for r in runs:
        rr = r["rounds"]
        missing = [x for x in rr if x.get("missing")]
        conv = None
        for x in rr:  # first round where every reporting phone saw >= 95% validation accuracy
            vals = [v for k, v in x.items() if k.startswith("val_acc_") and isinstance(v, (int, float))]
            if vals and min(vals) >= 0.95:
                conv = x["round"]
                break
        up = sum(v.get("by_endpoint", {}).get("/update", {}).get("payload_in", 0) for k, v in r["traffic"].items() if k in ("A", "B"))
        wire = sum(v.get("wire_in", 0) + v.get("wire_out", 0) for k, v in r["traffic"].items() if k in ("A", "B"))
        n_up = sum(v.get("by_endpoint", {}).get("/update", {}).get("requests", 0) for k, v in r["traffic"].items() if k in ("A", "B"))
        rows.append([f"`{r['id']}`", r["config"]["feature_set"], len(rr),
                     ", ".join(f"r{x['round']}: no {'/'.join(x['missing'])}" for x in missing) or "none",
                     "-" if conv is None else conv,
                     f"{up / max(1, n_up):.0f} B" if n_up else "-", f"{wire / 1024:.1f} KiB"])
    md.append(md_table(["run", "features", "rounds", "rounds with a missing phone", "first round ≥95% val acc",
                        "update size (JSON)", "total traffic (both phones)"], rows))

    # ---------- 4. secure aggregation (extra credit) ----------
    if any(r["config"].get("secure_aggregation") for r in runs):
        md.append("\n## Secure aggregation compared with plain FedAvg\n")
        md.append("*One upload as the server sees it*: each phone scores its own last upload the way the server "
                  "received it. In plain FedAvg that is a working model; with secure aggregation it is masked noise "
                  "(guessing among six poses is 16.7%).\n")
        eval_set = "test" if "test" in sets else (sets[0] if sets else "test")
        rows = []
        for secure in (False, True):
            rs = [r for r in runs if bool(r["config"].get("secure_aggregation")) == secure]
            if not rs:
                continue
            sizes, secs = [], []
            for r in rs:
                up = sum(v.get("by_endpoint", {}).get("/update", {}).get("payload_in", 0) for k, v in r["traffic"].items() if k in ("A", "B"))
                n_up = sum(v.get("by_endpoint", {}).get("/update", {}).get("requests", 0) for k, v in r["traffic"].items() if k in ("A", "B"))
                if n_up:
                    sizes.append(up / n_up)
                secs += [x["duration_s"] for x in r["rounds"] if isinstance(x.get("duration_s"), (int, float))]
            rows.append(["secure aggregation" if secure else "plain FedAvg", len(rs),
                         fmt(*mean_sd(collect(rs, "global", eval_set))),
                         fmt(*mean_sd(collect(rs, "server_sees", eval_set))),
                         f"{statistics.mean(sizes):.0f} B" if sizes else "-",
                         f"{statistics.mean(secs):.2f} s" if secs else "-",
                         sum(len(r["discarded"]) for r in rs),
                         sum(1 for r in rs for x in r["rounds"] if x.get("missing"))])
        md.append(md_table(["mode", "runs", f"global accuracy ({SET_LABEL.get(eval_set, eval_set)})",
                            "one upload as the server sees it", "update size (JSON)",
                            "time from first to last update of a round", "rounds discarded and repeated",
                            "rounds averaged with a phone missing"], rows))

    # ---------- results.csv ----------
    with open(os.path.join(out, "results.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "feature_set", "seed", "client", "test_set", "model", "n", "accuracy", "macro_f1"]
                   + [f"recall_{s}" for s in SHORT])
        for r in runs:
            for (c, s, mn), m in sorted(r["metrics"].items()):
                mm = m["metrics"]
                w.writerow([r["id"], r["config"]["feature_set"], r["config"].get("seed"), c, s, mn, mm.get("n"),
                            mm.get("accuracy"), mm.get("macro_f1")] + list(mm.get("per_class_recall", [None] * 6)))

    figs = plots(runs, groups, sets, out)
    if figs:
        md.append("\n## Figures\n\n" + "\n".join(f"![{f}]({f})" for f in figs))
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nWrote {out}/summary.md, results.csv" + (", " + ", ".join(figs) if figs else " (install matplotlib for figures)"))


def plots(runs, groups, sets, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                         "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})
    figs = []

    # accuracy by round: mean over phones (and over runs with the same feature set)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    drew = False
    for i, (fs, rs) in enumerate(groups.items()):
        per_round = {}
        for r in rs:
            for x in r["rounds"]:
                vals = [v for k, v in x.items() if k.startswith("val_acc_") and isinstance(v, (int, float))]
                if vals:
                    per_round.setdefault(x["round"], []).append(statistics.mean(vals))
        if not per_round:
            continue
        xs = sorted(per_round)
        ax.plot(xs, [100 * statistics.mean(per_round[k]) for k in xs], color=SERIES[i % 4], lw=2,
                marker="o", ms=4, label=f"{fs} ({len(rs)} run{'s' if len(rs) > 1 else ''})")
        drew = True
    if drew:
        ax.set(xlabel="round (global model version)", ylabel="validation accuracy (%)", ylim=(0, 102))
        ax.set_title("Global model accuracy on the phones' validation data", loc="left", color=INK)
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(out, "accuracy_by_round.png"))
        figs.append("accuracy_by_round.png")
    plt.close(fig)

    # three-model comparison, one group of bars per test set (first feature-set group = baseline if present)
    fs0 = "acc+grav" if "acc+grav" in groups else next(iter(groups))
    rs = groups[fs0]
    if sets:
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        width = 0.26
        for j, model in enumerate(MODELS):
            vals = [mean_sd(collect(rs, model, s)) for s in sets]
            xs = [i + (j - 1) * (width + 0.02) for i in range(len(sets))]
            bars = ax.bar(xs, [100 * (m or 0) for m, _ in vals], width, color=MODEL_COLOR[model], label=MODEL_LABEL[model],
                          yerr=[100 * (sd or 0) for _, sd in vals], error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
            for b, (m, _) in zip(bars, vals):
                if m is not None:
                    ax.text(b.get_x() + b.get_width() / 2, 100 * m + 2, f"{100 * m:.0f}", ha="center", fontsize=8, color=MUTED)
        ax.set_xticks(range(len(sets)), [SET_LABEL[s] for s in sets])
        ax.set(ylabel="accuracy (%)", ylim=(0, 112))
        ax.set_title(f"Local-only vs federated models (features {fs0})", loc="left", color=INK)
        ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.12))
        fig.savefig(os.path.join(out, "model_comparison.png"))
        figs.append("model_comparison.png")
        plt.close(fig)

    # confusion matrices (latest run of the baseline group, summed over phones)
    last = rs[-1]
    for s in [x for x in ("test", "test_unseen", "live_test") if x in sets]:
        mats = {}
        for model in MODELS:
            tot = [[0] * 6 for _ in range(6)]
            found = False
            for (c, ss, mn), m in last["metrics"].items():
                if ss == s and mn == model and "confusion" in m["metrics"]:
                    found = True
                    for i in range(6):
                        for k in range(6):
                            tot[i][k] += m["metrics"]["confusion"][i][k]
            if found:
                mats[model] = tot
        if not mats:
            continue
        fig, axes = plt.subplots(1, len(mats), figsize=(3.3 * len(mats), 3.3))
        axes = axes if len(mats) > 1 else [axes]
        for ax, (model, cm) in zip(axes, mats.items()):
            rows = [[v / max(1, sum(row)) for v in row] for row in cm]
            ax.imshow(rows, cmap="Blues", vmin=0, vmax=1)
            ax.grid(False)
            for i in range(6):
                for k in range(6):
                    if cm[i][k]:
                        ax.text(k, i, cm[i][k], ha="center", va="center", fontsize=7,
                                color="white" if rows[i][k] > 0.6 else INK)
            ax.set_xticks(range(6), SHORT)
            ax.set_yticks(range(6), SHORT)
            ax.set_xlabel("predicted")
            ax.set_title(MODEL_LABEL[model], color=INK, fontsize=10)
        axes[0].set_ylabel("true pose")
        fig.suptitle(f"Confusion matrices, {SET_LABEL[s]} set (run {last['id']})", x=0.01, ha="left", color=INK, fontsize=10)
        name = f"confusion_{s}.png"
        fig.savefig(os.path.join(out, name))
        figs.append(name)
        plt.close(fig)

    # sensor study
    if len(groups) > 1:
        test_sets = [x for x in ("test", "test_unseen", "live_test") if x in sets]
        fig, ax = plt.subplots(figsize=(6.4, 3.4))
        names = list(groups)
        width = 0.8 / max(1, len(test_sets))
        for j, s in enumerate(test_sets):
            vals = [mean_sd(collect(groups[fs], "global", s)) for fs in names]
            xs = [i + (j - (len(test_sets) - 1) / 2) * width for i in range(len(names))]
            ax.bar(xs, [100 * (m or 0) for m, _ in vals], width * 0.92, color=SERIES[j], label=SET_LABEL[s],
                   yerr=[100 * (sd or 0) for _, sd in vals], error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
        ax.set_xticks(range(len(names)), names)
        lo = min([100 * (mean_sd(collect(groups[fs], "global", s))[0] or 1) for fs in names for s in test_sets] + [100])
        ax.set(ylabel="global model accuracy (%)", ylim=(max(0, lo - 15), 101))
        ax.set_title("Sensor study: which inputs generalise best?", loc="left", color=INK)
        ax.legend(loc="lower right")
        fig.savefig(os.path.join(out, "sensor_study.png"))
        figs.append("sensor_study.png")
        plt.close(fig)
    return figs


if __name__ == "__main__":
    main()
