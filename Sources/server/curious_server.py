"""The "curious server" attack: what can the server work out about each phone's data from its uploads?

The server is assumed to be honest but curious: it runs federated learning correctly, but it also looks
closely at everything the phones send. Raw sensor data never reaches it, but can it still tell WHICH
POSES a phone has been recording? That is private information about the user.

How the attack works (one upload at a time):
    In a federated round the phone starts from the global model the server sent it and trains on its
    own data. Training raises the bias of every pose the phone has examples of and lowers the bias of
    every pose it never saw. The server knows the model it sent, so it subtracts:

        change = bias in the upload - bias of the global model it sent

    and guesses: "change > 0  ->  this phone has that pose".
    Each guess is scored against the answer key (the handout's split: A has poses 1-3, B has 4-6),
    so every upload gets a score out of 6. Guessing at random scores about 3 out of 6.

What the server can read from one upload depends on the mode:
    plain FedAvg               the real weights                      -> the attack works
    secure aggregation         masked weights (random-looking)       -> the attack is a coin flip
    compressed secure agg.     a masked 8-bit change of the weights  -> the attack is a coin flip

    python3 curious_server.py runs/run_20261118_101500      # one run, round by round
    python3 curious_server.py --latest 3                     # the newest 3 runs in runs/
"""
import argparse
import glob
import json
import os

import secure_agg

POSES = ["Screen Up", "Screen Down", "Portrait Up", "Portrait Down", "Landscape Left", "Landscape Right"]
N_POSES = len(POSES)

# ANSWER KEY, from the project handout. Only used to SCORE the guesses; the attack itself never reads it.
TRUE_POSES = {"A": {0, 1, 2}, "B": {3, 4, 5}}


def read_jsonl(path):
    """All records of a .jsonl file (a half-written last line is skipped)."""
    records = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def read_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return default


def bias_change_seen_by_server(body, global_before, mode):
    """The change of the 6 pose biases, as far as the server can read it from ONE /update upload.

    global_before: the global model the server sent for this round. Returns None if unreadable.
    """
    n_classes = len(global_before["b"])
    n_weights = len(global_before["W"])
    if mode == "plain":
        new_bias = (body.get("model") or {}).get("b")
        if not new_bias:
            return None
        return [new - old for new, old in zip(new_bias, global_before["b"])]
    if mode == "secure":
        values = (body.get("masked_model") or {}).get("values")
        if not values:
            return None
        # read the masked numbers as if they were weights (the best the server can do with one upload)
        as_weights = secure_agg.decode_alone(values, body["num_samples"])
        return [new - old for new, old in zip(as_weights[n_weights:], global_before["b"])]
    if mode == "compressed":
        payload = (body.get("masked_delta") or {}).get("payload")
        if not payload:
            return None
        # the upload is already a (masked) change; weight and scale are positive, so 1.0 keeps every sign
        change = secure_agg.decode_compressed_alone(payload, n_weights + n_classes, 1.0, 1.0)
        return change[n_weights:]
    return None


def guess_poses(bias_change):
    """The attacker's guess: the phone has a pose if that pose's bias went up."""
    return [c > 0 for c in bias_change]


def score_guess(guess, client_id):
    """How many of the 6 poses the guess got right (has it / does not have it)."""
    truth = TRUE_POSES.get(client_id, set())
    return sum(1 for pose in range(N_POSES) if guess[pose] == (pose in truth))


def run_mode(config):
    if config.get("secure_aggregation") and config.get("compression"):
        return "compressed"
    return "secure" if config.get("secure_aggregation") else "plain"


def attack_run(run_dir):
    """Runs the attack on every /update upload of one run. Returns a summary per phone."""
    config = read_json(os.path.join(run_dir, "config.json"), {})
    mode = run_mode(config)
    globals_cache = {}
    per_client = {}
    for entry in read_jsonl(os.path.join(run_dir, "uploads.jsonl")):
        if entry.get("endpoint") != "/update":
            continue
        body = entry.get("body") or {}
        cid = entry.get("client_id")
        version = body.get("base_version")
        if version not in globals_cache:
            globals_cache[version] = read_json(os.path.join(run_dir, f"global_v{version}.json"))
        global_before = globals_cache[version]
        if not global_before:
            continue
        change = bias_change_seen_by_server(body, global_before, mode)
        if change is None:
            continue
        guess = guess_poses(change)
        score = score_guess(guess, cid)
        c = per_client.setdefault(cid, {"uploads": 0, "correct": 0, "fully_right": 0, "rounds": [],
                                        "label_counts_visible": False})
        c["uploads"] += 1
        c["correct"] += score
        c["fully_right"] += score == N_POSES
        c["rounds"].append({"round": version, "score": score, "guess": [POSES[p] for p in range(N_POSES) if guess[p]]})
        c["last_guess"] = [POSES[p] for p in range(N_POSES) if guess[p]]
        # the per-pose counts give the answer away directly, without any attack
        if isinstance(body.get("class_counts"), list):
            c["label_counts_visible"] = True
    for cid, c in per_client.items():
        c["average_score"] = round(c["correct"] / c["uploads"], 2) if c["uploads"] else None
        c["truth"] = [POSES[p] for p in sorted(TRUE_POSES.get(cid, set()))]
    uploads = sum(c["uploads"] for c in per_client.values())
    correct = sum(c["correct"] for c in per_client.values())
    return {"mode": mode, "clients": per_client,
            "average_score": round(correct / uploads, 2) if uploads else None,
            "fully_right": round(sum(c["fully_right"] for c in per_client.values()) / uploads, 3) if uploads else None,
            "label_counts_visible": any(c["label_counts_visible"] for c in per_client.values())}


def print_report(run_dir):
    result = attack_run(run_dir)
    print(f"\n{os.path.basename(run_dir)}  ({result['mode']})")
    if result["average_score"] is None:
        print("  no /update uploads to attack")
        return
    for cid in sorted(result["clients"]):
        c = result["clients"][cid]
        print(f"  phone {cid}: truth = {', '.join(c['truth'])}")
        for r in c["rounds"]:
            print(f"    round {r['round']:>2}: guessed {', '.join(r['guess']) or 'nothing':<50} {r['score']}/6")
        print(f"    average {c['average_score']}/6, fully right in {c['fully_right']} of {c['uploads']} uploads"
              + ("; the per-pose counts were also visible in the upload" if c["label_counts_visible"] else ""))
    print(f"  overall: {result['average_score']}/6 per upload (random guessing: 3/6)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders")
    ap.add_argument("--latest", type=int, help="attack the newest N runs in --runs-dir")
    ap.add_argument("--runs-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs"))
    args = ap.parse_args()
    runs = list(args.runs)
    if args.latest:
        runs += sorted(glob.glob(os.path.join(args.runs_dir, "run_*")))[-args.latest:]
    if not runs:
        ap.error("give run folders or --latest N")
    for run in runs:
        print_report(run)


if __name__ == "__main__":
    main()
