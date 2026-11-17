"""Run:  python -m fedpose.server --host 0.0.0.0 --port 8080 --autostart"""
import argparse

from .app import create_app
from .fl_core import FLState, ServerConfig

ap = argparse.ArgumentParser(description="Federated phone-pose server")
ap.add_argument("--host", default="0.0.0.0")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--features", type=int, default=12, help="number of input features (12 = acc+grav)")
ap.add_argument("--rounds", type=int, default=30)
ap.add_argument("--timeout", type=float, default=60.0, help="round timeout in seconds")
ap.add_argument("--max-skips", type=int, default=2, help="force participation after this many skips")
ap.add_argument("--data-dir", default="server_data")
ap.add_argument("--autostart", action="store_true", help="start training once A and B registered")
a = ap.parse_args()

cfg = ServerConfig(n_features=a.features, round_timeout_s=a.timeout, max_rounds=a.rounds,
                   max_consecutive_skips=a.max_skips, data_dir=a.data_dir, autostart=a.autostart)
state = FLState(cfg)
app = create_app(state)
print(f"FL server on http://{a.host}:{a.port}  features={a.features} rounds={a.rounds} "
      f"timeout={a.timeout}s autostart={a.autostart}")
app.run(host=a.host, port=a.port, threaded=True)
