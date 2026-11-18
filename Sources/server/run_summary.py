"""One line of numbers per run, for the dashboard's "Compare runs" table (and GET /runs).

Everything is read from the run folders the server writes (runs/<run_id>/...), so runs from earlier
server sessions show up too. For each run:

    mode                 plain FedAvg, secure aggregation, or compressed secure aggregation
    accuracy             the three models on the phones' test recordings (average of both phones)
    server_sees          one upload read by the server as a model, scored on the same recordings
    curious              the curious-server attack: how many of the 6 poses it guessed right per upload
    upload_bytes         size of one /update upload (what each phone sends per round)
    train_ms, predict_ms, battery: the phones' device-cost reports
"""
import os
import statistics

import curious_server
from curious_server import read_json, read_jsonl

MODE_LABELS = {"plain": "Plain FedAvg", "secure": "Secure aggregation", "compressed": "Secure + compressed"}
_cache = {}   # run folder -> (signature, summary): a finished run is only read once


def _signature(run_dir):
    sig = []
    for name in ("uploads.jsonl", "rounds.jsonl", "metrics.jsonl", "device_cost_A.json", "device_cost_B.json"):
        try:
            st = os.stat(os.path.join(run_dir, name))
            sig.append((name, st.st_size, st.st_mtime))
        except OSError:
            sig.append((name, None, None))
    return tuple(sig)


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float))]
    return round(statistics.mean(values), 4) if values else None


def model_accuracy(metrics, model_name):
    """Average accuracy of one model over the phones' test sets (validation if there is no test set)."""
    latest = {}
    for m in metrics:   # the last report per phone / model / test set counts
        latest[(m.get("client_id"), m.get("model_name"), m.get("test_set"))] = m.get("metrics", {})
    values = []
    for cid in sorted({k[0] for k in latest}):
        for test_set in ("test", "val"):
            found = latest.get((cid, model_name, test_set))
            if found:
                values.append(found.get("accuracy"))
                break
    return _mean(values)


def summarize_run(run_dir):
    sig = _signature(run_dir)
    cached = _cache.get(run_dir)
    if cached and cached[0] == sig:
        return cached[1]
    config = read_json(os.path.join(run_dir, "config.json"), {})
    rounds = read_jsonl(os.path.join(run_dir, "rounds.jsonl"))
    metrics = read_jsonl(os.path.join(run_dir, "metrics.jsonl"))
    uploads = read_jsonl(os.path.join(run_dir, "uploads.jsonl"))
    discarded = read_jsonl(os.path.join(run_dir, "discarded_rounds.jsonl"))
    costs = [c for c in (read_json(os.path.join(run_dir, f"device_cost_{cid}.json"))
                         for cid in config.get("expected_clients", ["A", "B"])) if c]
    attack = curious_server.attack_run(run_dir)
    update_sizes = [u.get("bytes") for u in uploads if u.get("endpoint") == "/update"]
    mode = curious_server.run_mode(config)
    summary = {
        "run_id": os.path.basename(run_dir),
        "mode": mode, "mode_label": MODE_LABELS[mode],
        "feature_set": config.get("feature_set"), "seed": config.get("seed"),
        "rounds": config.get("rounds"), "rounds_done": len(rounds),
        "global_acc": model_accuracy(metrics, "global"),
        "local_A_acc": model_accuracy(metrics, "local_A"),
        "local_B_acc": model_accuracy(metrics, "local_B"),
        "server_sees_acc": model_accuracy(metrics, "server_sees"),
        "curious_score": attack["average_score"],
        "curious": {cid: {"truth": c["truth"], "last_guess": c.get("last_guess"), "average_score": c["average_score"]}
                    for cid, c in attack["clients"].items()},
        "curious_fully_right": attack["fully_right"],
        "label_counts_visible": attack["label_counts_visible"],
        "upload_bytes": _mean(update_sizes),
        "round_s": _mean([r.get("duration_s") for r in rounds]),
        "discarded_rounds": len(discarded),
        "train_ms_per_round": _mean([c.get("train_ms_per_round") for c in costs]),
        "predict_ms_per_window": _mean([c.get("predict_ms_per_window") for c in costs]),
        "battery_used_mah": _mean([c.get("battery_used_mah") for c in costs if not c.get("charging")]),
        "phones_reporting_cost": len(costs),
    }
    _cache[run_dir] = (sig, summary)
    return summary


def summarize_runs(runs_dir, limit=30):
    """The newest [limit] runs, newest first."""
    if not os.path.isdir(runs_dir):
        return []
    names = sorted((n for n in os.listdir(runs_dir) if n.startswith("run_")), reverse=True)[:limit]
    return [summarize_run(os.path.join(runs_dir, n)) for n in names]
