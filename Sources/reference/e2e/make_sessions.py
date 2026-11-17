"""Write synthetic phone sessions (CSV + metadata) for the Kotlin end-to-end test.

    python3 make_sessions.py /tmp/e2e
    kotlinc ../../android/app/src/main/java/com/posefl/core/*.kt FlEndToEnd.kt -include-runtime -d e2e.jar
    python3 ../../server/fl_server.py --round-timeout 2 &        # or --sweep seeds, then add "follow 60000"
    java -cp e2e.jar FlEndToEndKt http://localhost:8000 /tmp/e2e [follow <ms>] [flaky]
Client A: person "alice" (train: its 3 poses; val + test: all 6). Client B: "bob".
Each phone also gets one test session per pose from a person with no training data (unseen user).
SYNTHETIC DATA: only for testing code paths.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from posefl_ref import features as F  # noqa: E402
from posefl_ref.poses import CLIENT_TRAIN_POSES, KEYS, N_CLASSES  # noqa: E402
from posefl_ref.synth import Person, make_session  # noqa: E402


def main(out):
    rng = np.random.default_rng(7)
    for cid, name in (("A", "alice"), ("B", "bob")):
        d = os.path.join(out, cid)
        os.makedirs(d, exist_ok=True)
        person, unseen = Person(rng), Person(rng, max_tilt_deg=30, tremor=0.25)
        meta = []
        plan = [(p, "train", name, person) for p in CLIENT_TRAIN_POSES[cid] for _ in range(3)]
        plan += [(p, "val", name, person) for p in range(N_CLASSES)]
        plan += [(p, "test", name, person) for p in range(N_CLASSES)]
        plan += [(p, "test", "carol", unseen) for p in range(N_CLASSES)]
        for i, (p, split, who, per) in enumerate(plan):
            rows = make_session(p, per, rng, duration_s=12.0, holding=str(rng.choice(["hand", "surface"])))
            sid = f"{cid}_{who}_{KEYS[p]}_{split}_{i:03d}"
            with open(os.path.join(d, sid + ".csv"), "w") as fh:
                fh.write(F.HEADER + "\n")
                for r in rows:
                    fh.write(f"{r[0]:.1f}," + ",".join(f"{v:.6f}" for v in r[1:]) + "\n")
            meta.append({"id": sid, "pose": KEYS[p], "split": split, "participant": who})
        with open(os.path.join(d, "sessions.json"), "w") as fh:
            json.dump(meta, fh, indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
