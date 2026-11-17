"""Python virtual phones that talk to the Flask app in-process (no sockets).

Used to check the server against the offline reference simulator: both must produce the same
global model after every round when given the same data, seeds and policies.
"""
from __future__ import annotations

import tempfile

import numpy as np

from .constants import n_params
from .fl_sim import SimConfig, client_seed
from .model import train_local
from .server import FLState, ServerConfig, create_app
from .wire import b64_to_f32, skip_body, update_body


def drive_server(clients: dict, nf: int, cfg: SimConfig, data_dir: str | None = None) -> dict:
    """Run cfg.rounds rounds against a fresh server. Returns per-round global weights + status."""
    data_dir = data_dir or tempfile.mkdtemp(prefix="fl_srv_")
    scfg = ServerConfig(n_features=nf, expected_clients=tuple(clients), round_timeout_s=1e9,
                        max_rounds=cfg.rounds, max_consecutive_skips=cfg.max_skips,
                        init_scale=cfg.init_scale, init_seed=cfg.seed, data_dir=data_dir)
    state = FLState(scfg)
    http = create_app(state).test_client()
    for c in clients:
        assert http.post("/register", json={"client_id": c}).status_code == 200
    http.post("/admin/start", json={})
    weights = []
    for _ in range(cfg.rounds):
        for c, (X, y) in clients.items():
            m = http.get(f"/model?client_id={c}").get_json()
            w32 = b64_to_f32(m["weights"])
            wl = train_local(w32, X, y, nf, epochs=cfg.local_epochs, lr=cfg.lr,
                             batch_size=cfg.batch_size, l2=cfg.l2, mu=cfg.mu, w_ref=w32,
                             seed=client_seed(cfg.seed, m["round"], c))
            delta = wl - w32
            norm = float(np.linalg.norm(delta))
            if cfg.adaptive and norm < cfg.skip_threshold and not m["must_participate"]:
                body = skip_body(c, m["round"], m["version"], norm)
            else:
                body = update_body(c, m["round"], m["version"], len(y), delta, cfg.codec)
            r = http.post("/update", json=body)
            assert r.status_code == 200, r.get_json()
        weights.append(state.global_w.copy())
    return {"weights": weights, "state": state}
