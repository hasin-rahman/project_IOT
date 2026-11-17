import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.aggregate import fedavg
from fedpose.constants import n_params
from fedpose.data import load_all, pooled
from fedpose.fl_sim import SimConfig, simulate
from fedpose.server import FLState, ServerConfig, create_app
from fedpose.virtual import drive_server
from fedpose.wire import b64_to_f32, skip_body, update_body

NF = 12
P = n_params(NF)


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t


def make(timeout=60.0, max_rounds=5, max_skips=2):
    clock = Clock()
    st = FLState(ServerConfig(n_features=NF, round_timeout_s=timeout, max_rounds=max_rounds,
                              max_consecutive_skips=max_skips, data_dir=tempfile.mkdtemp()), clock)
    http = create_app(st).test_client()
    for c in "AB":
        assert http.post("/register", json={"client_id": c}).status_code == 200
    http.post("/admin/start", json={})
    return st, http, clock


def upd(http, c, rnd, ver, delta, n=100, codec="f32"):
    return http.post("/update", json=update_body(c, rnd, ver, n, delta, codec))


def test_round_closes_and_matches_fedavg():
    st, http, _ = make()
    rng = np.random.default_rng(0)
    da, db = rng.normal(0, .1, P), rng.normal(0, .1, P)
    assert upd(http, "A", 1, 0, da, 300).get_json()["round_closed"] is False
    r = upd(http, "B", 1, 0, db, 100).get_json()
    assert r["round_closed"] and r["next_round"] == 2 and r["version"] == 1
    exp = fedavg(np.zeros(P), [(b64_to_f32(update_body("A",1,0,1,da,"f32")["payload"]), 300),
                               (b64_to_f32(update_body("B",1,0,1,db,"f32")["payload"]), 100)])
    assert np.allclose(st.global_w, exp)


def test_stale_update_rejected_and_duplicate_idempotent():
    st, http, _ = make()
    d = np.full(P, 0.01)
    assert upd(http, "A", 7, 0, d).status_code == 409          # wrong round
    assert upd(http, "A", 1, 5, d).status_code == 409          # wrong version
    assert upd(http, "A", 1, 0, d).status_code == 200
    again = upd(http, "A", 1, 0, d)
    assert again.status_code == 200 and again.get_json()["duplicate"] is True
    upd(http, "B", 1, 0, d)
    assert np.allclose(st.global_w, d)                          # A counted once, not twice
    assert upd(http, "A", 1, 0, d).status_code == 409          # old round now stale


def test_timeout_with_one_client_aggregates_only_that_client():
    st, http, clock = make(timeout=30)
    d = np.full(P, 0.02)
    upd(http, "A", 1, 0, d)
    clock.t = 31
    m = http.get("/model?client_id=B").get_json()
    assert m["round"] == 2 and m["version"] == 1
    assert np.allclose(st.global_w, d)
    assert st.round_log[0]["missing"] == ["B"] and st.round_log[0]["reason"] == "timeout"
    # B comes back with an update built on the old round: rejected, must re-sync
    assert upd(http, "B", 1, 0, d).status_code == 409


def test_nobody_contributes_keeps_model():
    st, http, clock = make(timeout=10)
    before = st.global_w.copy()
    clock.t = 11
    m = http.get("/model?client_id=A").get_json()
    assert m["round"] == 2 and m["version"] == 0
    assert np.array_equal(st.global_w, before)
    # all skipped also keeps the model
    http.post("/update", json=skip_body("A", 2, 0, 0.0)); http.post("/update", json=skip_body("B", 2, 0, 0.0))
    assert st.version == 0 and st.round == 3


def test_skip_scheduling_forces_participation():
    st, http, _ = make(max_skips=2)
    d = np.full(P, 0.01)
    for rnd in (1, 2):
        http.post("/update", json=skip_body("A", rnd, st.version, 0.0))
        upd(http, "B", rnd, st.version, d)
    m = http.get("/model?client_id=A").get_json()
    assert m["must_participate"] is True
    assert http.get("/model?client_id=B").get_json()["must_participate"] is False
    upd(http, "A", 3, st.version, d); upd(http, "B", 3, st.version, d)
    assert http.get("/model?client_id=A").get_json()["must_participate"] is False


def test_bad_payloads_rejected():
    st, http, _ = make()
    b = update_body("A", 1, 0, 10, np.zeros(P), "f32")
    assert http.post("/update", json={**b, "payload": "!!!"}).status_code == 400
    bad = update_body("A", 1, 0, 10, np.zeros(P - 1), "f32")
    assert http.post("/update", json=bad).status_code == 400
    nan = np.zeros(P); nan[3] = np.nan
    assert http.post("/update", json=update_body("A", 1, 0, 10, nan, "f32")).status_code == 400
    assert http.post("/update", json={**b, "n_samples": 0}).status_code == 400
    assert http.post("/update", json={**b, "codec": "zip"}).status_code == 400
    assert http.post("/update", json={**b, "client_id": "Z"}).status_code == 400
    assert st.pending == {}


def test_traffic_counters_and_local_model_relay():
    st, http, _ = make()
    http.get("/model?client_id=A")
    t = st.traffic["A"]
    assert t["body_out"] > 400 and t["requests"] >= 2
    from fedpose.wire import f32_to_b64
    w = np.linspace(-1, 1, P)
    assert http.post("/local_model", json={"client_id": "A", "weights": f32_to_b64(w)}).status_code == 200
    got = http.get("/local_model?client_id=A&asker=B").get_json()
    assert np.allclose(b64_to_f32(got["weights"]), w, atol=1e-6)
    assert http.get("/local_model?client_id=B&asker=A").status_code == 404


def test_privacy_guard_rejects_unexpected_fields_and_audits():
    st, http, _ = make()
    b = update_body("A", 1, 0, 10, np.zeros(P), "f32")
    r = http.post("/update", json={**b, "features": [[0.1] * 12]})
    assert r.status_code == 400 and r.get_json()["error"] == "privacy_guard"
    assert r.get_json()["unexpected_keys"] == ["features"]
    assert st.pending == {}
    ok = http.post("/update", json={**b, "class_counts": [0, 0, 0, 10, 20, 30], "train_loss": 0.3})
    assert ok.status_code == 200
    assert http.post("/update", json={**update_body("B", 1, 0, 10, np.zeros(P), "f32"), "class_counts": [1, 2]}).status_code == 400
    a = http.get("/audit").get_json()
    outcomes = [e["outcome"] for e in a["last_uploads"]]
    assert "rejected_privacy_guard" in outcomes and "accepted" in outcomes
    acc = [e for e in a["last_uploads"] if e["outcome"] == "accepted"][0]
    assert acc["class_counts"] == [0, 0, 0, 10, 20, 30] and acc["payload_preview"].endswith("...")
    assert "features" not in acc["keys"]
    assert http.post("/report", json={"client_id": "A", "model": "global", "samples": [1]}).status_code == 400
    assert http.post("/report", json={"client_id": "A", "model": "global", "test_set": "t", "accuracy": 1.0}).status_code == 200
    assert b"What the phones upload" in http.get("/audit.html").data


def test_live_feature_set_switch():
    st, http, _ = make()
    r = http.post("/admin/start", json={"n_features": 18, "max_rounds": 3}).get_json()
    assert r["status"] == "training" and st.P == n_params(18) and st.cfg.max_rounds == 3
    assert http.get("/model?client_id=A").get_json()["n_features"] == 18
    assert http.post("/register", json={"client_id": "A"}).get_json()["n_features"] == 18
    assert http.post("/admin/start", json={"n_features": 7}).status_code == 400


def test_round_log_has_class_counts():
    st, http, _ = make()
    for c, cc in (("A", [5, 5, 5, 0, 0, 0]), ("B", [0, 0, 0, 5, 5, 5])):
        http.post("/update", json={**update_body(c, 1, 0, 15, np.zeros(P), "int8"), "class_counts": cc})
    assert st.round_log[0]["class_counts"] == {"A": [5, 5, 5, 0, 0, 0], "B": [0, 0, 0, 5, 5, 5]}


def _clients():
    root = os.path.join(os.path.dirname(__file__), "..", "..", "data_synth")
    if not os.path.isdir(root):
        import pytest; pytest.skip("run scripts/make_synth_data.py first")
    cd = load_all(root)
    return {c: d.train for c, d in cd.items()}


def test_server_matches_reference_simulator():
    """The task-division item 'check server results against a Python reference model'."""
    cl = _clients()
    for kw in (dict(codec="f32"), dict(codec="int8"),
               dict(codec="int8", adaptive=True, skip_threshold=1.0)):
        cfg = SimConfig(rounds=8, local_epochs=2, lr=0.3, seed=4, **kw)
        ref = simulate(cl, NF, cfg, keep_weights=True)
        srv = drive_server(cl, NF, cfg)
        for r, (h, w) in enumerate(zip(ref["history"], srv["weights"]), start=1):
            assert np.max(np.abs(h["weights"] - w)) < 1e-12, (kw, r)
        if kw.get("adaptive"):
            assert ref["skipped_updates"] > 0  # the skip path really ran
            assert sum(len(e["skipped"]) for e in srv["state"].round_log) == ref["skipped_updates"]
