import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fedpose.aggregate import fedavg
from fedpose.constants import n_features, n_params
from fedpose.metrics import evaluate
from fedpose.model import init_weights, loss_acc, predict, train_local
from fedpose.rng import SplitMix64
from fedpose.wire import decode_delta, encode_delta, update_body


def test_param_count_matches_proposal():
    assert n_features("acc_grav") == 12 and n_params(12) == 78


def test_splitmix_known_values():
    r = SplitMix64(0)
    # Reference values for SplitMix64 seed 0 (also asserted in the Kotlin test)
    assert r.next_u64() == 0xE220A8397B1DCDAF
    assert r.next_u64() == 0x6E789E6AA1B965F4
    a = list(range(10)); SplitMix64(7).shuffle(a)
    b = list(range(10)); SplitMix64(7).shuffle(b)
    assert a == b and sorted(a) == list(range(10))


def test_int8_roundtrip_error_bounded():
    rng = np.random.default_rng(0)
    d = rng.normal(0, 0.3, 78)
    out = decode_delta(update_body("A", 1, 0, 10, d, "int8"), 78)
    assert np.max(np.abs(out - d)) <= np.max(np.abs(d)) / 127 / 2 + 1e-6
    out32 = decode_delta(update_body("A", 1, 0, 10, d, "f32"), 78)
    assert np.max(np.abs(out32 - d)) < 1e-6


def test_zero_delta_int8():
    out = decode_delta(update_body("A", 1, 0, 10, np.zeros(78), "int8"), 78)
    assert np.all(out == 0)


def test_fedavg_equals_weighted_weights():
    g = np.arange(5.0)
    wa, wb = g + 1, g - 3
    new = fedavg(g, [(wa - g, 30), (wb - g, 10)])
    assert np.allclose(new, (30 * wa + 10 * wb) / 40)
    assert np.array_equal(fedavg(g, []), g)


def test_sgd_learns_separable_data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 4)); y = (X[:, 0] > 0).astype(np.int64) + 2 * (X[:, 1] > 0)
    w = train_local(init_weights(4), X, y, 4, epochs=60, lr=0.5, seed=3)
    assert loss_acc(w, X, y, 4)[1] > 0.9


def test_metrics():
    y = np.array([0, 0, 1, 1, 2, 2]); p = np.array([0, 1, 1, 1, 2, 0])
    m = evaluate(y, p)
    assert abs(m["accuracy"] - 4 / 6) < 1e-12
    assert m["per_class_recall"][:3] == [0.5, 1.0, 0.5]
    assert m["per_class_recall"][3:] == [0.0, 0.0, 0.0]
