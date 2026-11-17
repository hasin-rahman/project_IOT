#!/usr/bin/env python3
"""Live dropout test with the real Kotlin client and the real server.

Phone B is made to 'lose connection' (upload delayed past the round timeout) in rounds 3-4.
Expected: the server closes those rounds with A only (reason=timeout), B's late uploads are
rejected as stale (HTTP 409), B re-syncs to the latest model and contributes again, the run
finishes, and the final model still reaches high accuracy on all six poses.

    python scripts/check_dropout_live.py --jar /tmp/ktbuild/fedpose-core.jar --data ../data_synth
"""
import argparse, json, os, socket, subprocess, sys, tempfile, time, urllib.request
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ap = argparse.ArgumentParser()
ap.add_argument("--jar", required=True)
ap.add_argument("--data", default="../data_synth")
ap.add_argument("--rounds", type=int, default=8)
ap.add_argument("--timeout", type=float, default=3.0)
a = ap.parse_args()
JAVA = ["java", "-cp", a.jar, "com.fedpose.cli.VirtualPhoneKt"]

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
ddir = tempfile.mkdtemp(prefix="srv_")
srv = subprocess.Popen([sys.executable, "-m", "fedpose.server", "--host", "127.0.0.1", "--port", str(port),
                        "--rounds", str(a.rounds), "--timeout", str(a.timeout), "--data-dir", ddir, "--autostart"],
                       cwd=os.path.join(os.path.dirname(__file__), ".."), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
url = f"http://127.0.0.1:{port}"
try:
    for _ in range(100):
        try: urllib.request.urlopen(url + "/status", timeout=1); break
        except Exception: time.sleep(0.1)
    out = tempfile.mkdtemp(prefix="phones_")
    base = JAVA + ["phone", "--server", url, "--root", a.data, "--features", "acc_grav", "--seed", "0", "--out", out]
    pa = subprocess.Popen(base + ["--client", "A"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    pb = subprocess.Popen(base + ["--client", "B", "--drop-rounds", "3-4", "--drop-ms", str(int(a.timeout * 1000 + 1500))],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    la, lb = pa.communicate(timeout=300)[0], pb.communicate(timeout=300)[0]
    rl = [json.loads(l) for l in open(os.path.join(ddir, "rounds.jsonl"))]
    print("round log (round, reason, contributors, missing, version_after):")
    for e in rl:
        print(f"  {e['round']:>2}  {e['reason']:<12} contrib={e['contributors']} missing={e['missing']} v={e['version_after']}")
    print("phone B:", [l for l in lb.splitlines() if "rejected" in l or "dropout" in l or "federation done" in l])
    summ = json.load(open(f"{out}/client_A.json"))["eval"]["global"]["own_test"]
    stale = int([l for l in lb.splitlines() if "federation done" in l][0].split("stale=")[1].split()[0])
    checks = {
        "rounds 3-4 closed by timeout": all(e["reason"] == "timeout" and e["missing"] == ["B"] for e in rl if e["round"] in (3, 4)),
        "B's late uploads rejected as stale": stale >= 1,
        "B contributes again after recovery": any("B" in e["contributors"] for e in rl if e["round"] >= 5),
        "all rounds completed": len(rl) == a.rounds,
        "global accuracy still high on six poses": summ["accuracy"] > 0.9,
    }
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"  global model accuracy (phone A own_test, six poses): {summ['accuracy']:.3f}")
    sys.exit(0 if all(checks.values()) else 1)
finally:
    srv.terminate(); srv.wait(timeout=10)
