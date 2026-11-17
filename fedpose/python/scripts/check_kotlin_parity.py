#!/usr/bin/env python3
"""Cross-check the Kotlin on-device code against the Python reference.

  1. window features: Kotlin vs Python on real session files, all four feature sets
  2. local training : Kotlin SGD vs Python SGD (same data, seed, epochs)
  3. end to end     : two Kotlin virtual phones <-> the real Flask server over HTTP, compared with
                      the offline simulator (final global model, local baselines, traffic)

    python scripts/check_kotlin_parity.py --jar /tmp/ktbuild/fedpose-core.jar --data ../data_synth
"""
import argparse, json, os, socket, subprocess, sys, tempfile, time, urllib.request
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.constants import FEATURE_SETS, n_features  # noqa: E402
from fedpose.data import load_all, load_split, session_files  # noqa: E402
from fedpose.features import load_session_csv, window_features  # noqa: E402
from fedpose.fl_sim import SimConfig, local_baseline_seed, simulate, train_local_only  # noqa: E402
from fedpose.model import f32_round, init_weights, train_local  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--jar", required=True)
ap.add_argument("--data", default="../data_synth")
ap.add_argument("--rounds", type=int, default=8)
a = ap.parse_args()
JAVA = ["java", "-cp", a.jar, "com.fedpose.cli.VirtualPhoneKt"]
ok = True


def kt(*args, **kw):
    r = subprocess.run(JAVA + [str(x) for x in args], capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stdout, r.stderr); raise SystemExit("kotlin command failed")
    return r.stdout


def report(name, diff, tol):
    global ok
    good = diff <= tol
    ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name}: max|diff| = {diff:.3e} (tol {tol:.0e})")


print("1) feature parity")
files = [f for f in session_files(a.data, "A", "test")][:3] + [f for f in session_files(a.data, "B", "newuser")][:2]
for fs in FEATURE_SETS:
    worst = 0.0
    for path, _, _ in files:
        py = window_features(load_session_csv(path), fs)
        out = kt("features", "--session", path, "--features", fs).strip().splitlines()
        kv = np.array([[float(v) for v in l.split(",")] for l in out]).reshape(-1, n_features(fs))
        if kv.shape != py.shape:
            print(f"  FAIL  {fs} {os.path.basename(path)}: shape {kv.shape} vs {py.shape}"); ok = False; continue
        worst = max(worst, float(np.max(np.abs(kv - py))))
    report(f"{fs} ({len(files)} sessions)", worst, 1e-9)

print("1b) blind-test clip parity (2 s clip -> mean probabilities -> label)")
from fedpose.blind import clip_label, clip_proba, clip_start_of, clip_windows  # noqa: E402
nf0 = 12
Xa, ya = load_split(a.data, "A", "train", "acc_grav")
w_demo = train_local(init_weights(nf0), Xa, ya, nf0, epochs=3, lr=0.3, seed=2)
wtmp = tempfile.mkdtemp()
np.savetxt(f"{wtmp}/w.txt", w_demo, fmt="%.17g")
worst, labels_ok, n_clips = 0.0, True, 0
for path, _, _ in files[:4]:
    sess = load_session_csv(path)
    t0 = clip_start_of(sess, "acc_grav")
    for off in (0.0, 3.25, 9.5):
        for starts in ((0.5, 1.0), (0.0, 0.5, 1.0)):
            wins = clip_windows(sess, "acc_grav", t0 + int(round(off * 1e9)), starts)
            pp = clip_proba(w_demo, wins, nf0)
            out = kt("clip", "--session", path, "--features", "acc_grav", "--weights", f"{wtmp}/w.txt",
                     "--starts", ",".join(str(x) for x in starts), "--offset-s", off).strip().splitlines()
            assert out[0] == f"windows={len(wins)}", (out[0], len(wins))
            pk = np.array([float(v) for v in out[1].split(",")])
            worst = max(worst, float(np.max(np.abs(pk - pp))))
            labels_ok &= int(out[2].split("=")[1]) == clip_label(w_demo, wins, nf0)
            n_clips += 1
report(f"{n_clips} clips, probabilities", worst, 1e-9)
ok &= labels_ok
print(f"  {'PASS' if labels_ok else 'FAIL'}  clip labels identical")

print("2) local SGD parity")
nf = 12
tmp = tempfile.mkdtemp()
kt("dataset", "--root", a.data, "--client", "A", "--split", "train", "--features", "acc_grav", "--out", f"{tmp}/A.csv")
X, y = load_split(a.data, "A", "train", "acc_grav")
for epochs, lr, batch, seed in [(1, 0.3, 32, 1), (6, 0.5, 16, 12345)]:
    kt("train", "--data", f"{tmp}/A.csv", "--features", "acc_grav", "--epochs", epochs, "--lr", lr,
       "--batch", batch, "--seed", seed, "--out", f"{tmp}/w.txt")
    wk = np.loadtxt(f"{tmp}/w.txt")
    wp = train_local(init_weights(nf), X, y, nf, epochs=epochs, lr=lr, batch_size=batch, seed=seed)
    report(f"epochs={epochs} lr={lr} batch={batch}", float(np.max(np.abs(wk - wp))), 1e-9)

print("3) end to end: Kotlin phones <-> Flask server <-> simulator")


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def run_e2e(label, codec, adaptive, thr, seed):
    global ok
    port = free_port()
    ddir = tempfile.mkdtemp(prefix="srv_")
    srv = subprocess.Popen([sys.executable, "-m", "fedpose.server", "--host", "127.0.0.1", "--port", str(port),
                            "--rounds", str(a.rounds), "--timeout", "120", "--max-skips", "2",
                            "--data-dir", ddir, "--autostart"],
                           cwd=os.path.join(os.path.dirname(__file__), ".."), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(url + "/status", timeout=1); break
            except Exception:
                time.sleep(0.1)
        outdir = tempfile.mkdtemp(prefix="phones_")
        procs = []
        for c in "AB":
            args = JAVA + ["phone", "--server", url, "--client", c, "--root", a.data, "--features", "acc_grav",
                           "--epochs", "2", "--lr", "0.3", "--batch", "32", "--seed", str(seed), "--codec", codec,
                           "--out", outdir]
            if adaptive:
                args += ["--adaptive", "--threshold", str(thr)]
            procs.append(subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True))
        logs = [p.communicate(timeout=300)[0] for p in procs]
        if any(p.returncode for p in procs):
            print("\n".join(logs)); raise SystemExit("phone failed")
        st = json.load(urllib.request.urlopen(url + "/status"))
        cd = load_all(a.data)
        cl = {c: d.train for c, d in cd.items()}
        cfg = SimConfig(rounds=a.rounds, local_epochs=2, lr=0.3, seed=seed, codec=codec, adaptive=adaptive,
                        skip_threshold=thr)
        ref = simulate(cl, nf, cfg)
        gk = np.array(json.load(open(f"{outdir}/client_A.json"))["global_weights"])
        report(f"[{label}] final global model vs simulator", float(np.max(np.abs(gk - f32_round(ref["weights"])))), 1e-6)
        for c in "AB":
            lk = np.array(json.load(open(f"{outdir}/client_{c}.json"))["local_weights"])
            lp = train_local_only(*cl[c], nf, cfg, c)
            report(f"[{label}] local-only model {c} vs Python", float(np.max(np.abs(lk - lp))), 1e-8)
        rl = [json.loads(l) for l in open(os.path.join(ddir, "rounds.jsonl"))]
        sk = sum(len(e["skipped"]) for e in rl)
        good = (len(rl) == a.rounds and sk == ref["skipped_updates"] and st["n_reports"] == 12)
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  [{label}] rounds={len(rl)}/{a.rounds}, skipped={sk} (sim {ref['skipped_updates']}), reports={st['n_reports']}/12")
        audit = [json.loads(l) for l in open(os.path.join(ddir, "uploads.jsonl"))]
        acc_up = [e for e in audit if e["kind"] == "update" and e["outcome"] == "accepted"]
        allowed = set(json.load(urllib.request.urlopen(url + "/audit"))["allowed_update_keys"])
        good = (len(acc_up) > 0 and all(set(e["keys"]) <= allowed for e in audit if e["kind"] == "update")
                and all(sum(e["class_counts"]) == e["n_samples"] for e in acc_up)
                and not any("rejected" in e["outcome"] for e in audit))
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  [{label}] audit log: {len(acc_up)} accepted uploads, only allowed fields, class counts match n_samples, none rejected")
        up = sum(e["body_bytes_in"] for e in rl)
        print(f"        server body bytes in over rounds: {up}  | simulator uplink estimate: {ref['bytes_up']}")
        for c in "AB":
            print(f"        phone {c}: " + [l for l in logs[0 if c == 'A' else 1].splitlines() if 'federation done' in l][0].split('] ', 1)[1])
    finally:
        srv.terminate(); srv.wait(timeout=10)


run_e2e("f32", "f32", False, 0.0, 0)
run_e2e("int8+adaptive", "int8", True, 1.0, 3)
print("\nALL PARITY CHECKS PASSED" if ok else "\nPARITY CHECK FAILURES")
sys.exit(0 if ok else 1)
