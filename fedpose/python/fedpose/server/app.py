"""Flask HTTP wrapper around FLState. See docs/PROTOCOL.md for the endpoint contract."""
from __future__ import annotations

from flask import Flask, Response, request

from ..wire import dumps
from .fl_core import ALLOWED_REPORT_KEYS, ALLOWED_UPDATE_KEYS, FLState


def _req_header_bytes() -> int:
    return (sum(len(k) + len(v) + 4 for k, v in request.headers.items())
            + len(request.method) + len(request.full_path) + 12)


def create_app(state: FLState) -> Flask:
    app = Flask(__name__)

    def reply(client_id, endpoint, obj, status=200, body_in=None):
        data = dumps(obj)
        resp = Response(data, status=status, mimetype="application/json")
        hdr_out = sum(len(k) + len(v) + 4 for k, v in resp.headers.items()) + 17
        if body_in is None:
            body_in = len(request.get_data())
        state.count(client_id, endpoint, body_in, len(data), _req_header_bytes(), hdr_out)
        return resp

    def json_body():
        b = request.get_json(silent=True)
        return b if isinstance(b, dict) else None

    @app.post("/register")
    def register():
        b = json_body()
        if b is None:
            return reply(None, "register", {"error": "bad_json"}, 400)
        try:
            return reply(b.get("client_id"), "register", state.register(b.get("client_id")))
        except KeyError:
            return reply(None, "register", {"error": "unknown_client"}, 400)

    @app.get("/model")
    def model():
        cid = request.args.get("client_id", "?")
        return reply(cid, "model", state.get_model(cid))

    @app.get("/round_status")
    def round_status():
        return reply(request.args.get("client_id", "?"), "round_status", state.round_status())

    @app.post("/update")
    def update():
        b = json_body()
        if b is None:
            return reply(None, "update", {"error": "bad_json"}, 400)
        code, out = state.submit_update(b)
        return reply(b.get("client_id"), "update", out, code)

    @app.post("/local_model")
    def put_local_model():
        b = json_body()
        if b is None:
            return reply(None, "local_model", {"error": "bad_json"}, 400)
        try:
            state.put_local_model(b.get("client_id"), b.get("weights", ""))
        except KeyError:
            return reply(None, "local_model", {"error": "unknown_client"}, 400)
        except Exception:
            return reply(b.get("client_id"), "local_model", {"error": "bad_weights"}, 400)
        return reply(b.get("client_id"), "local_model", {"ok": True})

    @app.get("/local_model")
    def get_local_model():
        cid = request.args.get("client_id", "?")
        m = state.get_local_model(cid)
        if m is None:
            return reply(request.args.get("asker"), "local_model_get", {"error": "not_found"}, 404)
        return reply(request.args.get("asker", cid), "local_model_get",
                     {"client_id": cid, "weights": m["weights"]})

    @app.post("/report")
    def report():
        b = json_body()
        if b is None:
            return reply(None, "report", {"error": "bad_json"}, 400)
        extra = state.add_report(b)
        if extra:
            return reply(b.get("client_id"), "report", {"error": "privacy_guard", "unexpected_keys": extra}, 400)
        return reply(b.get("client_id"), "report", {"ok": True})

    @app.get("/results")
    def results():
        return reply("admin", "results", {"reports": state.reports})

    @app.get("/status")
    def status():
        return reply("admin", "status", state.status_dict())

    @app.post("/admin/start")
    def admin_start():
        b = json_body() or {}
        try:
            return reply("admin", "start", state.start(b.get("max_rounds"), n_features=b.get("n_features")))
        except ValueError as e:
            return reply("admin", "start", {"error": str(e)}, 400)

    @app.get("/audit")
    def audit():
        """What the phones actually sent: keys, sizes, metadata, payload preview. For the privacy demo."""
        n = int(request.args.get("n", 20))
        return reply("admin", "audit", {"allowed_update_keys": sorted(ALLOWED_UPDATE_KEYS),
                                        "allowed_report_keys": sorted(ALLOWED_REPORT_KEYS),
                                        "last_uploads": state.audit[-n:]})

    @app.get("/audit.html")
    def audit_html():
        """Human-readable page to put on the projector: refreshes every 2 s."""
        rows = []
        for e in reversed(state.audit[-25:]):
            rows.append("<tr>" + "".join(f"<td>{v}</td>" for v in (
                e["client_id"], e["kind"], e["outcome"], e.get("round"), e.get("n_samples"), e.get("codec"),
                e.get("payload_chars"), e.get("payload_preview"), e.get("class_counts"),
                e.get("train_loss") if e.get("train_loss") is None else round(e["train_loss"], 4),
                ", ".join(e["keys"]))) + "</tr>")
        head = ("<tr>" + "".join(f"<th>{h}</th>" for h in (
            "phone", "kind", "outcome", "round", "n_samples", "codec", "payload chars", "payload preview",
            "class counts", "loss", "fields sent")) + "</tr>")
        html = ("<html><head><meta http-equiv='refresh' content='2'><style>body{font-family:sans-serif;margin:16px}"
                "table{border-collapse:collapse;font-size:13px}td,th{border:1px solid #ccc;padding:4px 8px}"
                "th{background:#eee}</style></head><body><h2>What the phones upload</h2>"
                "<p>Allowed fields only: " + ", ".join(sorted(ALLOWED_UPDATE_KEYS)) +
                ". Anything else is rejected.</p><table>" + head + "".join(rows) + "</table></body></html>")
        return Response(html, mimetype="text/html")

    return app
