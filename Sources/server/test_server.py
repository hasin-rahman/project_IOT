"""Server tests (standard library unittest).   python3 -m unittest -v test_server"""
import os
import shutil
import tempfile
import threading
import time
import unittest

from fedavg import fedavg, validate_model, zero_model
from fl_server import make_server
from sim_clients import SimPhone, call


def config(**kw):
    c = {"host": "127.0.0.1", "port": 0, "expected_clients": ["A", "B"], "feature_set": "acc+grav",
         "n_classes": 6, "rounds": 5, "local_epochs": 1, "lr": 0.1, "batch_size": 16, "l2": 1e-4,
         "seed": 0, "baseline_epochs": 10, "round_timeout_s": 0, "min_clients": 1}
    c.update(kw)
    return c


class FedAvgTest(unittest.TestCase):
    def test_weighted_mean(self):
        a = {"W": [1.0, 2.0], "b": [0.0]}
        b = {"W": [3.0, 6.0], "b": [4.0]}
        W, bias = fedavg([(a, 1), (b, 3)])
        self.assertEqual(W, [2.5, 5.0])
        self.assertEqual(bias, [3.0])

    def test_validate(self):
        m = zero_model(12, 6, "acc+grav")
        validate_model(m, 12, 6)
        with self.assertRaises(ValueError):
            validate_model(dict(m, W=m["W"][:-1]), 12, 6)
        with self.assertRaises(ValueError):
            validate_model(dict(m, b=[float("nan")] * 6), 12, 6)


class _ServerBase(unittest.TestCase):
    CONFIG = {}
    SWEEP = None

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.httpd, self.state = make_server(config(**self.CONFIG), self.tmp, self.SWEEP)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        shutil.rmtree(self.tmp)

    def _update(self, cid, version, n=10, run_id=None, W=None):
        model = zero_model(12, 6, "acc+grav")
        if W is not None:
            model["W"] = W
        return call(self.base, "POST", "/update", {"client_id": cid, "run_id": run_id or self.state.run_id,
                                                   "base_version": version, "num_samples": n, "model": model})



class ServerTest(_ServerBase):
    def test_round_aggregates_only_when_both_clients_submit(self):
        st, r = self._update("A", 0, W=[1.0] * 72)
        self.assertEqual((st, r["aggregated"]), (200, False))
        st, r = self._update("B", 0, n=30, W=[5.0] * 72)
        self.assertEqual((st, r["aggregated"], r["version"]), (200, True, 1))
        _, g = call(self.base, "GET", "/global")
        self.assertAlmostEqual(g["model"]["W"][0], 4.0)  # (1*10 + 5*30) / 40

    def test_rejects_stale_wrong_shape_and_unknown_client(self):
        self.assertEqual(self._update("A", 3)[1]["error"], "stale_version")
        self.assertEqual(self._update("A", 0, run_id="old")[1]["error"], "stale_run")
        self.assertEqual(self._update("A", 0, W=[0.0] * 5)[0], 400)
        self.assertEqual(self._update("C", 0)[0], 400)
        self.assertEqual(self._update("A", 0, n=0)[0], 400)

    def test_privacy_allow_list_refuses_sensor_data_and_logs_uploads(self):
        model = zero_model(12, 6, "acc+grav")
        ok = {"client_id": "A", "run_id": self.state.run_id, "base_version": 0, "num_samples": 10,
              "class_counts": [4, 3, 3, 0, 0, 0], "model": model}
        # anything that is not weights / counts / scores is refused, at the top level and inside the model
        self.assertEqual(call(self.base, "POST", "/update", dict(ok, features=[[0.1] * 12]))[0], 400)
        self.assertEqual(call(self.base, "POST", "/update", dict(ok, rows=[[0.0] * 14]))[0], 400)
        self.assertEqual(call(self.base, "POST", "/update", dict(ok, model=dict(model, X=[1.0])))[0], 400)
        self.assertEqual(self.state.pending, {})          # nothing was accepted
        self.assertEqual(call(self.base, "POST", "/update", ok)[0], 200)
        _, up = call(self.base, "GET", "/uploads?client_id=A")
        entry = up["uploads"]["/update"]
        self.assertEqual(entry["body"], ok)                # the exact payload is kept as evidence
        self.assertIn("model{W[72], b[6]}", entry["summary"])
        self.assertIn("uploads.jsonl", os.listdir(self.state.run_dir))
        _, status = call(self.base, "GET", "/status")
        self.assertEqual(status["clients"]["A"]["class_counts"], [4, 3, 3, 0, 0, 0])

    def test_reset_starts_new_run_with_new_feature_set(self):
        st, r = call(self.base, "POST", "/admin/reset", {"feature_set": "acc+grav+gyro", "rounds": 3})
        self.assertEqual(st, 200)
        _, g = call(self.base, "GET", "/global")
        self.assertEqual(len(g["model"]["W"]), 6 * 18)
        self.assertEqual(g["train_config"]["rounds"], 3)

    def test_two_simulated_phones_end_to_end(self):
        phones = [SimPhone(self.base, "A", 1, "acc+grav"), SimPhone(self.base, "B", 2, "acc+grav")]
        for p in phones:
            p.start()
        for p in phones:
            p.join(timeout=120)
        self.assertTrue(self.state.done)
        for p in phones:
            self.assertGreater(p.result["global"]["accuracy"], 0.95)
            self.assertLess(p.result[f"local_{p.cid}"]["accuracy"], 0.6)
        _, res = call(self.base, "GET", "/results")
        self.assertEqual(len(res["results"]), 6)  # 2 phones x 3 models
        self.assertEqual(self.state.finished, {"A", "B"})
        files = os.listdir(self.state.run_dir)
        for f in ["rounds.jsonl", "metrics.jsonl", "global_v5.json", "local_A.json", "traffic.json"]:
            self.assertIn(f, files)
        _, s = call(self.base, "GET", "/status")
        self.assertGreater(s["traffic"]["A"]["wire_out"], s["traffic"]["A"]["payload_out"])


class SecureAggregationTest(_ServerBase):
    CONFIG = {"secure_aggregation": True, "round_timeout_s": 1.0}

    def test_masks_cancel_and_one_update_alone_is_noise(self):
        import random
        import secure_agg as S
        a, b = S.new_private_key(), S.new_private_key()
        key = S.shared_key(a, S.public_key(b))
        self.assertEqual(key, S.shared_key(b, S.public_key(a)))        # both phones get the same key
        wa = [random.uniform(-20, 20) for _ in range(78)]
        wb = [random.uniform(-20, 20) for _ in range(78)]
        label = S.mask_label("run", 3, 0)
        ma, mb = S.mask_update(wa, 200, key, label, True), S.mask_update(wb, 300, key, label, False)
        avg = S.sum_masked([ma, mb], 500)
        for got, x, y in zip(avg, wa, wb):
            self.assertAlmostEqual(got, (200 * x + 300 * y) / 500, places=8)   # exactly FedAvg
        alone = S.decode_alone(ma, 200)
        self.assertGreater(min(abs(v) for v in alone[:20]), 1e3)       # nothing like the real weights
        self.assertNotEqual(ma, S.mask_update(wa, 200, key, S.mask_label("run", 3, 1), True))  # new mask per attempt
        blob = S.encrypt(key, "local|run|A", "secret model")
        self.assertEqual(S.decrypt(key, "local|run|A", blob), "secret model")
        self.assertIsNone(S.decrypt(b"0" * 32, "local|run|A", blob))    # a wrong key is detected

    def test_server_refuses_plain_weights_when_secure(self):
        self.assertEqual(self._update("A", 0)[0], 400)
        st, _ = call(self.base, "POST", "/local_model", {"client_id": "A", "model": zero_model(12, 6, "acc+grav")})
        self.assertEqual(st, 400)

    def test_secure_run_matches_fedavg_accuracy_and_server_sees_no_weights(self):
        phones = [SimPhone(self.base, "A", 1, "acc+grav"), SimPhone(self.base, "B", 2, "acc+grav")]
        for p in phones:
            p.start()
        for p in phones:
            p.join(timeout=120)
        self.assertTrue(self.state.done)
        for p in phones:
            self.assertGreater(p.result["global"]["accuracy"], 0.95)
            self.assertLess(p.result[f"local_{p.cid}"]["accuracy"], 0.6)
            self.assertEqual(len(p.result), 3)                     # the encrypted local model was relayed and opened
        # nothing the phones sent contains plain weights
        import json
        with open(os.path.join(self.state.run_dir, "uploads.jsonl")) as fh:
            uploads = [json.loads(line) for line in fh]
        self.assertTrue(any(u["endpoint"] == "/update" for u in uploads))
        for u in uploads:
            self.assertNotIn("model", u["body"])
        self.assertIn("encrypted_model", self.state.local_models["A"])

    def test_dropout_discards_the_round_then_recovers(self):
        # phone B goes silent for 2.5 s before round 2: A's masked update alone is useless, so the round is repeated
        a = SimPhone(self.base, "A", 1, "acc+grav")
        b = SimPhone(self.base, "B", 2, "acc+grav", pause_rounds={2: 2.5})
        for p in (a, b):
            p.start()
        for p in (a, b):
            p.join(timeout=120)
        self.assertTrue(self.state.done)
        self.assertGreaterEqual(len(self.state.discarded), 1)
        self.assertEqual(self.state.discarded[0]["received"], ["A"])
        self.assertTrue(all(not r["missing"] for r in self.state.history))   # every used round had both phones
        self.assertGreater(a.result["global"]["accuracy"], 0.95)


class PasswordTest(unittest.TestCase):
    def test_requests_need_the_password(self):
        import json
        import urllib.request
        tmp = tempfile.mkdtemp()
        httpd, state = make_server(config(port=0), tmp, None, password="s3cret")
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            def get(path, token=None):
                req = urllib.request.Request(base + path, headers={"X-Auth-Token": token} if token else {})
                try:
                    with urllib.request.urlopen(req, timeout=5) as r:
                        return r.status, r.read()
                except urllib.error.HTTPError as e:
                    return e.code, e.read()
            self.assertEqual(get("/status")[0], 401)
            self.assertEqual(get("/status", "wrong")[0], 401)
            st, body = get("/status", "s3cret")
            self.assertEqual(st, 200)
            self.assertNotIn(b"s3cret", body)                        # never shown in the status
            self.assertEqual(get("/")[0], 200)                         # the dashboard page loads, then asks
            self.assertEqual(get("/global?client_id=A&token=s3cret")[0], 200)
            with open(os.path.join(state.run_dir, "config.json")) as fh:
                self.assertNotIn("s3cret", fh.read())                # never written to the run folder
        finally:
            httpd.shutdown()


class DropoutTest(_ServerBase):
    CONFIG = {"round_timeout_s": 1.0, "rounds": 6}

    def test_round_closes_without_a_silent_client_and_it_resyncs(self):
        st, r = self._update("A", 0, W=[1.0] * 72)
        self.assertFalse(r["aggregated"])
        time.sleep(1.8)
        self.assertEqual(self.state.version, 1)  # aggregated A alone after the timeout
        self.assertEqual(self.state.history[-1]["missing"], ["B"])
        self.assertEqual(self._update("B", 0)[1]["error"], "stale_version")  # late update is rejected

    def test_simulated_phone_drops_out_and_recovers(self):
        phones = [SimPhone(self.base, "A", 1), SimPhone(self.base, "B", 2, pause_rounds={2: 2.5})]
        for p in phones:
            p.start()
        for p in phones:
            p.join(timeout=120)
        self.assertTrue(self.state.done)
        self.assertGreaterEqual(phones[1].stale_count, 1)
        self.assertTrue(any(r["missing"] == ["B"] for r in self.state.history))
        self.assertGreater(phones[1].result["global"]["accuracy"], 0.9)

    def test_no_updates_keeps_the_model(self):
        time.sleep(1.5)
        self.assertEqual(self.state.version, 0)


class SweepTest(_ServerBase):
    CONFIG = {"rounds": 3}
    SWEEP = [{"feature_set": "acc"}, {"feature_set": "acc+grav+gyro"}]

    def test_sweep_runs_back_to_back(self):
        phones = [SimPhone(self.base, c, i, follow_runs=True, max_runs=2) for i, c in enumerate("AB", 1)]
        for p in phones:
            p.start()
        for p in phones:
            p.join(timeout=120)
        self.assertEqual(len(self.state.sweep_runs), 2)
        self.assertEqual(self.state.config["feature_set"], "acc+grav+gyro")
        self.assertTrue(self.state.done)
        for p in phones:
            self.assertEqual(len(p.results_by_run), 2)


if __name__ == "__main__":
    unittest.main()
