import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.blind import clip_label, clip_proba, clip_windows, make_clips, rubric_score
from fedpose.features import load_session_csv
from fedpose.data import session_files

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data_synth")


def test_rubric_example_from_the_handout():
    s = rubric_score(33, 36, 18, 18, 17, 18)
    assert abs(s["part_A"] - 37.2) < 0.05  # "A = 27.5 + 5 + 4.7 = 37.2 points"


def test_clip_windows_and_label():
    if not os.path.isdir(ROOT):
        import pytest; pytest.skip("run scripts/make_synth_data.py first")
    path, pose, _ = session_files(ROOT, "A", "test")[0]
    sess = load_session_csv(path)
    rng = np.random.default_rng(0)
    clips = make_clips(sess, "acc_grav", rng, 5)
    assert len(clips) == 5 and clips[0].shape == (2, 12)
    # hand built classifier on gravity mean: label must equal the pose for a clean clip
    nf = 12
    w = np.zeros(6 * nf + 6)
    up = [(2, 1), (2, -1), (1, 1), (1, -1), (0, -1), (0, 1)]
    for k, (ax, sg) in enumerate(up):
        w[k * nf + 6 + ax] = 10.0 * sg
    assert clip_label(w, clips[0], nf) == pose
    assert abs(clip_proba(w, clips[0], nf).sum() - 1) < 1e-12
